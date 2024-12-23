from typing import Optional, List, Tuple
from typing import Any, ContextManager, Iterator, cast
from pathlib import Path

import os
import logging
import json
import configparser

import numpy as np

from t_mac.weights import preprocess_weights



logger = logging.getLogger("model_utils")


_PRESET_KERNELS = {
    "llama-2-7b-4bit": [
        # bits, M, K, N, m_groups
        # [4, 12288, 4096, 1, -1],  # unused for llama.cpp
        [4, 4096, 4096, 1, -1],
        [4, 11008, 4096, 1, -1],
        [4, 4096, 11008, 1, -1],
    ],
    "llama-2-7b-2bit": [
        # [2, 12288, 4096, 1, -1],  # unused for llama.cpp
        [2, 4096, 4096, 1, -1],
        [2, 11008, 4096, 1, -1],
        [2, 4096, 11008, 1, -1],
    ],
    "llama-2-13b-2bit": [
        [2, 5120, 5120, 1, -1],
        [2, 13824, 5120, 1, -1],
        [2, 5120, 13824, 1, -1],
    ],
    "llama-3-8b-2bit": [
        [2, 4096, 4096, 1, -1],
        [2, 14336, 4096, 1, -1],
        [2, 4096, 14336, 1, -1],
        [2, 1024, 4096, 1, -1],
    ],
    "llama-3-8b-4bit": [
        [4, 4096, 4096, 1, -1],
        [4, 14336, 4096, 1, -1],
        [4, 4096, 14336, 1, -1],
        [4, 1024, 4096, 1, -1],
    ],
    "hf-bitnet-3b": [
        [2, 3200, 8640, 1, 1],
        [2, 8640, 3200, 1, 1],
        [2, 3200, 3200, 1, 1],
    ],
    "hf-bitnet-large-intn": [    # 700M
        [2, 1536, 4096, 1, 1],
        [2, 4096, 1536, 1, 1],
        [2, 1536, 1536, 1, 1],
    ],
    "hf-bitnet-large-tq": [    # 700M
        [2, 1536, 4096, 1, -1],
        [2, 4096, 1536, 1, -1],
        [2, 1536, 1536, 1, -1],
    ],
    "ms-bitnet-3b": [
        [2, 3200, 800, 1, 1],
        [2, 3200, 3200, 1, 1],
        [2, 3200, 10240, 1, 1],
        [2, 10240, 3200, 1, 1],
        [2, 800, 3200, 1, 1],
    ],
    "phi-3-mini-2bit": [
        [2, 3072, 3072, 1, -1],
        [2, 9216, 3072, 1, -1],
        [2, 3072, 8192, 1, -1],
        [2, 16384, 3072, 1, -1],
    ],
    "trilm-3.9b": [
        [2, 3072, 3072, 1, -1],
        [2, 3072, 9216, 1, -1],
        [2, 9216, 3072, 1, -1],
        [2, 768, 3072, 1, -1],
    ],
    "test": [
        # Add customized kernels here
    ],
    "gptq-auto": [],
}


def get_preset_models() -> List[str]:
    return _PRESET_KERNELS.keys()


def parse_gptqv2(qweight: np.ndarray, scales: np.ndarray, qzeros: np.ndarray) -> Tuple:
    bits = 32 // (scales.shape[1] // qzeros.shape[1])
    K = qweight.shape[0] * (32 // bits)
    M = qweight.shape[1]
    group_size = K // scales.shape[0]

    return K, M, bits, group_size


def unpack_gptqv2(qweight: np.ndarray, scales: np.ndarray, qzeros: np.ndarray, gptq_v2: bool = True):
    """
    Unpack GPTQv2
    Return T-MAC biased uint8 weight [0, 2 ** bits), fp16 scales, biased fp16 zeros, bits, group_size
    """
    assert qweight.dtype == "int32"
    assert qzeros.dtype == "int32"

    K, M, bits, group_size = parse_gptqv2(qweight, scales, qzeros)

    # Unpack qweight
    qweights = [(qweight >> bit_offset) & ((1 << bits) - 1) for bit_offset in range(0, 32, bits)]
    w = np.stack(qweights, axis=1).reshape(K, M).T.astype("uint8")

    scales = scales.T

    # Unpack qzeros
    zeros = [(qzeros >> bit_offset) & ((1 << bits) - 1) for bit_offset in range(0, 32, bits)]
    zeros = np.stack(zeros, axis=-1).reshape(K // group_size, M).T.astype(scales.dtype)
    if not gptq_v2:
        # `zeros = zeros - 1` in AutoGPTQ
        # Not in GPTQModel
        zeros += 1
    zeros = (zeros - (2 ** (bits - 1))) * scales

    return w, scales, zeros, bits, group_size


class _Model:
    """
    Based on https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py
    """
    def __init__(self, dir_model: Path):
        self.dir_model = dir_model
        self.part_names = _Model.get_model_part_names(self.dir_model, "model", ".safetensors")
        self.is_safetensors = len(self.part_names) > 0
        if not self.is_safetensors:
            self.part_names = _Model.get_model_part_names(self.dir_model, "pytorch_model", ".bin")

    @staticmethod
    def get_model_part_names(dir_model: Path, prefix: str, suffix: str) -> List[str]:
        part_names: list[str] = []
        for filename in os.listdir(dir_model):
            if filename.startswith(prefix) and filename.endswith(suffix):
                part_names.append(filename)

        part_names.sort()

        return part_names

    def get_tensors(self) -> Iterator[tuple]:
        import contextlib, torch
        for part_name in self.part_names:
            logger.info(f"gguf: loading model part '{part_name}'")
            ctx: ContextManager[Any]
            if self.is_safetensors:
                from safetensors import safe_open
                ctx = cast(ContextManager[Any], safe_open(self.dir_model / part_name, framework="pt", device="cpu"))
            else:
                ctx = contextlib.nullcontext(torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True))

            with ctx as model_part:
                for name in model_part.keys():
                    data = model_part.get_tensor(name) if self.is_safetensors else model_part[name]
                    yield name, data

    def extract_kernel_shapes(self):
        quant_dict = {}
        # Store scales and qzeros to dict to be later preprocessed
        # Save memory by not storing qweight
        for name, data_torch in self.get_tensors():
            if name.endswith(".scales") or name.endswith(".qzeros"):
                data = data_torch.numpy()
                quant_dict[name] = data

        ks = []
        final_group_size = None
        for name, data_torch in self.get_tensors():
            if name.endswith(".qweight"):
                qweight = data_torch.numpy()
                scales = quant_dict[name.replace(".qweight", ".scales")]
                qzeros = quant_dict[name.replace(".qweight", ".qzeros")]
                K, M, bits, group_size = parse_gptqv2(qweight, scales, qzeros)
                k = [bits, M, K, 1, -1]
                if k not in ks:
                    ks.append(k)
                if final_group_size is None:
                    final_group_size = group_size
                elif final_group_size != group_size:
                    raise RuntimeError("Different group_sizes unsupported")

        if len(ks) == 0:
            raise RuntimeError("Models in {} not in GPTQ format".format(self.dir_model))

        return ks

    @staticmethod
    def load_hparams(dir_model):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)


def extract_kernel_shapes(model_arch: Optional[str] = "gptq-auto", model_dir: Optional[str] = None) -> Tuple[List, int]:
    if model_arch not in get_preset_models():
        raise KeyError("Unsupported model_arch: {}".format(model_arch))

    if model_arch != "gptq-auto":
        # group_size unset
        return _PRESET_KERNELS[model_arch]

    # Detect kernel shape from checkpoint
    # Only support GPTQ based
    return _Model(Path(model_dir)).extract_kernel_shapes()


def get_quantization_config(model_dir: Optional[str] = None) -> dict:
    hparams = _Model.load_hparams(Path(model_dir))
    # GPTQ
    quantization_config = hparams.get("quantization_config", {})
    desc_act = quantization_config.get("desc_act", False)
    assert not desc_act, "desc_act=True currently unsupported by T-MAC"
    quantizer = quantization_config.get("meta", {}).get("quantizer", "")
    group_size = quantization_config.get("group_size", 0)
    bits = quantization_config.get("bits", 0)
    sym = quantization_config.get("sym", False)
    quant_method = quantization_config.get("quant_method", "")
    # BitNet
    weight_bits = hparams.get("weight_bits", 0)

    return {
        "quantizer": quantizer,
        "group_size": group_size,
        "bits": bits,
        "sym": sym,
        "quant_method": quant_method,
        "weight_bits": weight_bits,
    }


def preprocess_for_t_mac(
    kcfg_file: str,
    w: np.ndarray,
    scales: np.ndarray,
    zeros: Optional[np.ndarray] = None,
    bits: int = 2,
    g: int = 4,
) -> np.ndarray:

    M, K = w.shape
    cf = configparser.ConfigParser()
    cf.read(kcfg_file)
    secs = cf.sections()
    found = False
    for sec in secs:
        sec_splits = str(sec).split('_')
        if sec_splits[-4] == "m" + str(M * bits) and sec_splits[-3] == "k" + str(K):
            bm = int(cf.get(sec, 'bm'))
            kfactor = int(cf.get(sec, 'kfactor'))
            simd_n_in = int(cf.get(sec, 'simd_n_in'))
            simd_n_out = int(cf.get(sec, 'simd_n_out'))
            found = True
            break

    if not found:
        raise KeyError("GEMM of shape ({}, {}) is not found in {}. Please compile the kernels using T-MAC first.".format(M, K, kcfg_file))

    w, scales = preprocess_weights(w, scales, zeros, bits=bits, g=g, bm=bm, kfactor=kfactor, simd_n_in=simd_n_in, simd_n_out=simd_n_out)
    return np.concatenate([w.flatten(), scales.astype(np.float32).copy().view(np.uint8).flatten()])
