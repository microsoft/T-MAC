import numpy as np
from typing import Tuple, Optional


def preprocess_weights(
    w: np.ndarray,
    scales: np.ndarray,
    zeros: Optional[np.ndarray] = None,
    bits: int = 4,
    g: int = 4,
    bm: int = 512,
    kfactor: int = 16,
    simd_n_in: int = 16,
    simd_n_out: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Offline preprocess the weights before inference.

    Parameters
    ----------
    w : np.ndarray
        Quantized weights of shape (M, K) and type "uint8".
        Add a bias of 2^(bits-1) to the original int1/2/3/4 values to convert it to uint values.
        E.g., add a bias of 2 to int2: -2, -1, 0, 1 -> 0, 1, 2, 3
    scales: np.ndarray
        Quantization scales of shape (M, K // group_size) or (m_groups,) and type float32/16.
    zeros: np.ndarray
        Same shape and type with scales.
        If None, the actual zero points will be 2^(bits-1) * scales;
        if not None, the actual zero points will be zeros + 2^(bits-1) * scales.
        E.g., before passing the zeros from BitDistiller/GPTQ, you need to modify it as following:
        `zeros = (zeros - (2 ** (bits - 1))) * scales`
    bits: int
        Number of bits for each quantized element
    g: int
        Group size of LUT
    bm: int
        Tuned tiling size of M
    kfactor: int
        Tuned tiling size of K
    simd_width: int
        128 for ARM NEON

    Returns
    -------
    w: np.ndarray
        Permuted weights
    scales: np.ndarray
        Permuted scales
    """
    assert(w.dtype == "uint8")

    M, K = w.shape
    M = M * bits
    ngroups_per_elem = 8 // g

    # (M // bits, K, bits)
    w = np.stack([(w >> ib) & 1 for ib in range(bits)], axis=-1)
    # (M // bits, K, bits) -> (M // bits, bits, K) -> (M // bits, bits, K // g, g)
    w = w.transpose(0, 2, 1).reshape(M // bits, bits, K // g, g)
    w = sum([(w[:, :, :, ig] << ig) for ig in range(g)])
    # 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
    # for bits=3
    # bit0: [0, 8), bit1: [8, 16), bit2: [16, 24), bit0: [24, 32)
    # (M // bits // simd_n_float16, bits, simd_n_float16, K // g)
    w = w.reshape(M // bits // simd_n_out, simd_n_out, bits, K // g).transpose(0, 2, 1, 3)
    mgroup = ngroups_per_elem * simd_n_in
    w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in, K // g).transpose(0, 2, 1, 3)
    #             0        1             2          3                 4                  5
    w = w.reshape(M // bm, bm // mgroup, simd_n_in, ngroups_per_elem, K // g // kfactor, kfactor).transpose(0, 4, 1, 5, 2, 3)
    w = sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])
    w = w.reshape(M // bm, K // g // kfactor, bm // mgroup, kfactor, simd_n_in)
    # input size of current TVM API
    w = w.reshape(M // bm, K // g, bm // ngroups_per_elem)

    if scales.size >= M // bits:
        group_size = K // scales.shape[1]
        scales = scales.reshape(M // bm, bm // bits, K // group_size).transpose(0, 2, 1)
        scales = scales.reshape(M // bm, K // group_size, bm // bits // simd_n_out, simd_n_out)
        if zeros is not None:
            zeros = zeros.reshape(M // bm, bm // bits, K // group_size).transpose(0, 2, 1)
            zeros = zeros.reshape(M // bm, K // group_size, bm // bits // simd_n_out, simd_n_out)
            scales = np.stack([scales, zeros], axis=-2)
        # input size of current TVM API
        scales = scales.reshape(M // bm, K // group_size, -1)
    else:
        if zeros is not None:
            scales = np.concatenate([scales, zeros])
    return w, scales
