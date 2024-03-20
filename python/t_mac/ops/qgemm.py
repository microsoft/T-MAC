from .base import OpCodegen
from tvm import te
import tvm
import tvm.testing
import numpy as np
from ..intrins import tbl, lut_ctor

from typing import List
import os


class QGeMMLUTBitsCodegen(OpCodegen):

    def __init__(
        self,
        *args,
        bits: int = 1,
        g: int = 4,
        group_size: int = 128,
        act_group_size: int = 64,
        out_dtype: str = "float16",
        simd_n_in: int = 16,
        simd_n_out: int = 8,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.out_dtype = out_dtype
        self.has_lut_scale = (self.dtype != self.out_dtype)
        self.weight_dtype = "uint8"
        self._num_per_elem = 8
        self.g = g
        self._ngroups_per_elem = self._num_per_elem // self.g
        self.group_size = group_size
        self.act_group_size = act_group_size
        self.bits = bits
        # NEON-ui8: 16 (= 128 / 8)
        # AVX2-ui8: 16 (= 128 / 8)
        self.simd_n_in = simd_n_in
        # NEON-f16: 8 (= 128 / 16)
        # AVX2-f32: 8 (= 256 / 32)
        # AVX2-i32: 8 (= 256 / 32)
        self.simd_n_out = simd_n_out
        # w = b0 + b1 * 2 + b2 * 4 + b3 * 8 - 8
        #   = 1 / 2 (b0' + gamma * s0) + b1' + b2' * 2 + b3' * 4, where s0 = -1
        self.alphas = [1 / 2, 1, 2, 4]
        self.kfactors = [k for k in [8, 16] if k * 4 >= self.act_group_size]

    def _define_config(self, cfg):
        cfg.define_knob("bm", [256, 128, 512, 1024])
        cfg.define_knob("bn", [32])
        cfg.define_knob("kfactor", self.kfactors)
        super()._define_config(cfg)

    def _compute(self, M: int, N: int, K: int):
        bm = self.bm

        k = te.reduce_axis((0, K // self.g), "k")

        A = te.placeholder((M // bm, K // self.g, bm // self._ngroups_per_elem), dtype=self.weight_dtype, name="A")
        LUT = te.placeholder((N, K // self.g, 2 ** self.g), dtype=self.dtype, name="LUT")
        Scales = te.placeholder((M // bm, K // self.group_size, bm // self.bits), dtype=self.out_dtype, name="Scales")

        if self.has_lut_scale:
            LUT_Scales = te.placeholder((N, K // self.act_group_size), dtype=self.out_dtype, name="LUT_Scales")
            LUT_Biases = te.placeholder((N, K // self.act_group_size), dtype=self.out_dtype, name="LUT_Biases")

        mask = te.const((1 << self.g) - 1, dtype=self.weight_dtype)

        def _get_Abits(m, k):
            return (A[m // bm, k, (m % bm) // self._ngroups_per_elem] >> (self.g * ((m % bm) % self._ngroups_per_elem))) & mask

        # placeholder computation
        # should be tensorized
        CBits = te.compute(
            (N, M),
            lambda n, m: te.sum(
                LUT[n, k, _get_Abits(m, k)].astype(self.out_dtype) * (
                    Scales[m // bm, k * self.g // self.group_size, (m % bm) // self.bits]
                        * LUT_Scales[n, k * self.g // self.act_group_size]
                        + LUT_Biases[n, k * self.g // self.act_group_size]
                    if self.has_lut_scale
                    else Scales[m // bm, k * self.g // self.group_size, (m % bm) // self.bits]
                ),
                axis=k,
            ),
            name="CBits",
        )

        alphas = [te.const(self.alphas[b], dtype=self.out_dtype) for b in range(self.bits)]

        C = te.compute(
            (N, M // self.bits),
            lambda n, m: sum([
                CBits[
                    n,
                    te.indexdiv(m, self.simd_n_out) * self.simd_n_out * self.bits
                        + te.indexmod(m, self.simd_n_out)
                        + b * self.simd_n_out
                ] * alphas[b]
                for b in range(self.bits)
            ]),
            name="C",
        )

        if self.has_lut_scale:
            return [A, LUT, Scales, LUT_Scales, LUT_Biases, C]
        else:
            return [A, LUT, Scales, C]

    def _schedule(self, tensors: List[te.Tensor]):
        C = tensors[-1]
        sch: te.Schedule = te.create_schedule(C.op)

        n, m = sch[C].op.axis

        CC = sch.cache_write(C, "global")
        no, mo, ni, mi = sch[C].tile(n, m, self.bn, self.bm // self.bits)
        sch[CC].compute_at(sch[C], mo)

        CBits = CC.op.input_tensors[0]
        sch[CBits].compute_at(sch[C], mo)

        nC, mC = sch[CBits].op.axis
        (kC,) = sch[CBits].op.reduce_axis
        koC, kiC = sch[CBits].split(kC, factor=self.kfactor)
        sch[CBits].reorder(koC, nC, kiC, mC)

        intrin, ll_code = tbl(
            self.bm,
            self.kfactor,
            self.g,
            self.group_size,
            self.act_group_size,
            self._ngroups_per_elem,
            self.bits,
            self.dtype,
            cc=self.cc,
            cc_opts=self.cc_opts,
            has_scale=True,
            has_lut_scale=self.has_lut_scale,
            out_dtype=self.out_dtype,
        )
        sch[CBits].tensorize(kiC, intrin)
        sch[CBits].pragma(koC, "import_llvm", ll_code)

        sch[C].parallel(mo)

        return sch

    def _verify(self, tvm_arrays: List[tvm.nd.NDArray], arrays: List[np.ndarray]):
        tvm.testing.assert_allclose(tvm_arrays[-1].numpy(), arrays[-1], atol=1e-2, rtol=1e-2)

    def _reference(self, M: int, N: int, K: int):
        a = np.random.randn(M // self.bm, K // self.g // self.kfactor, self.bm // self._ngroups_per_elem // self.simd_n_in, self.kfactor, self.simd_n_in).astype(self.weight_dtype)
        a_t = a.reshape(M // self.bm, K // self.g, self.bm // self._ngroups_per_elem)

        lut = np.random.randn(N, K // self.g, 2 ** self.g).astype(self.dtype)

        scales = np.random.randn(M // self.bm, K // self.group_size, self.bm // self.bits // self.simd_n_out, self.simd_n_out).astype(self.out_dtype)
        scales_t = scales.reshape(M // self.bm, K // self.group_size, self.bm // self.bits)

        if self.has_lut_scale:
            lut_scales = np.random.randn(N, K // self.act_group_size).astype(self.out_dtype)
            lut_biases = np.random.randn(N, K // self.act_group_size).astype(self.out_dtype)

        cbits = np.zeros((N, M), dtype=self.out_dtype)

        a = np.concatenate([(a >> (self.g * ng)) & ((1 << self.g) - 1) for ng in range(self._ngroups_per_elem)], axis=-1)
        for n in range(N):
            for k in range(K // self.g):
                for m in range(M):
                    mo = m // self.bm
                    ko = k // self.kfactor
                    mi = (m % self.bm) // self._ngroups_per_elem // self.simd_n_in
                    ki = k % self.kfactor
                    e = (m % self.bm) % (self._ngroups_per_elem * self.simd_n_in)
                    a_e = a[mo, ko, mi, ki, e]

                    scales_mi = (m % self.bm) // self.bits // self.simd_n_out
                    scales_e = ((m % self.bm) % self.simd_n_out)
                    cbits[n, m] += lut[n, k, a_e] * lut_scales[n, k * self.g // self.act_group_size] * scales[mo, k * self.g // self.group_size, scales_mi, scales_e]
                    if (((k * self.g) % self.act_group_size) == 0) and ((((m % self.bm) // self.simd_n_out) % self.bits) == 0):
                        cbits[n, m] += lut_biases[n, k * self.g // self.act_group_size] * scales[mo, k * self.g // self.group_size, scales_mi, scales_e]

        c = (
            cbits.reshape((N, M // self.simd_n_out // self.bits, self.bits, self.simd_n_out))
                .transpose(0, 1, 3, 2)
                .dot(np.array(self.alphas[:self.bits], dtype=self.out_dtype))
                .reshape((N, M // self.bits))
        )
        
        if self.has_lut_scale:
            return [a_t, lut, scales_t, lut_scales, lut_biases, c]
        else:
            return [a_t, lut, scales_t, c]


class QGeMMLUTBitsPreprocessorCodegen(OpCodegen):
    """Preprocessor of QGeMMLUTBitsCodegen.

    This preprocessor will compute the LUT of the activations,
    and quantize the LUT from `out_dtype` to `dtype`.
    """
    def __init__(self, *args, g: int = 4, act_group_size: int = 64, **kwargs):
        super().__init__(*args, **kwargs)

        self.out_dtype = "float16"
        if self.dtype == "int8":
            self.maxv = (1 << 7) - 1
        self.g = g
        self.act_group_size = act_group_size
        # Weights mapped from s=[0, 1] to s'=[-1, 1]
        # s' = s * 2 - 1
        # s = (s' + 1) / 2
        self._states = [-1, 1]
        # w = b0 + b1 * 2 + b2 * 4 + b3 * 8 - 8
        #   = 1 / 2 (b0' + gamma * s0) + b1' + b2' * 2 + b3' * 4, where s0 = -1
        self._gamma = 1

    def _define_config(self, cfg):
        self.kfactor = self.act_group_size // self.g
        super()._define_config(cfg)

    def _compute(self, N: int, K: int):
        B = te.placeholder((N, K), dtype=self.out_dtype, name="B")

        # Actually outputs
        # Workaround: tensorize doesn't support composite-op
        LUT_Scales = te.placeholder((N, K // self.act_group_size), dtype=self.out_dtype, name="LUT_Scales")
        LUT_Biases = te.placeholder((N, K // self.act_group_size), dtype=self.out_dtype, name="LUT_Biases")

        # placeholder computation
        # should be tensorized
        QLUT = te.compute(
            (N, K // self.g, 1 << self.g),
            lambda n, k, g: (
                B[n, k * self.g + (g % self.g)] / LUT_Scales[n, k * self.g // self.act_group_size] - LUT_Biases[n, k * self.g // self.act_group_size]
            ).astype(self.dtype),
            name="QLUT",
        )

        return [B, LUT_Scales, LUT_Biases, QLUT]

    def _schedule(self, tensors: List[te.Tensor]):
        _, LUT_Scales, LUT_Biases, QLUT = tensors
        sch: te.Schedule = te.create_schedule([QLUT.op])

        n, k, g = sch[QLUT].op.axis

        ko, ki = sch[QLUT].split(k, factor=self.kfactor)

        intrin, ll_code = lut_ctor(
            self.kfactor * 4,
            self.g,
            self.act_group_size,
            self.dtype,
            cc=self.cc,
            cc_opts=self.cc_opts,
        )
        sch[QLUT].tensorize(ki, intrin)
        sch[QLUT].pragma(ko, "import_llvm", ll_code)

        # Currently the parallelism of preprocessor is disabled due to large thread communication overhead in `benchmark.cc`.
        # But according to profiled results of python side, the overhead is not that large and the best NUM_THREADS should be 4.
        # TODO: Find out the reason for the high communication overhead in C++ side.
        # sch[QLUT].parallel(ko)

        return sch

    def _verify(self, tvm_arrays: List[tvm.nd.NDArray], arrays: List[np.ndarray]):
        tvm.testing.assert_allclose(tvm_arrays[-1].numpy(), arrays[-1], atol=1, rtol=0)
        tvm.testing.assert_allclose(tvm_arrays[-2].numpy(), arrays[-2], atol=1e-2, rtol=1e-2)
        tvm.testing.assert_allclose(tvm_arrays[-3].numpy(), arrays[-3], atol=1e-2, rtol=1e-2)

    def _reference(self, N: int, K: int):
        b = np.random.randn(N, K).astype(self.out_dtype)
        b_t = b

        b = b.reshape(N, K // self.g, self.g)

        codes = np.array([[i] for i in range(1 << self.g)], dtype=np.uint8)
        codes = np.unpackbits(codes, axis=1, bitorder="little", count=self.g).T

        def map_states(c):
            return self._states[c]

        m = np.vectorize(map_states)(codes).astype(self.out_dtype)

        # (N, K // self.g, 1 << self.g)
        lut = b.dot(m)
        lut_biases = lut.reshape(N, K // self.act_group_size, self.act_group_size // self.g, 1 << self.g)[:, :, :, 0]
        lut_biases = np.sum(lut_biases, axis=-1) * self._gamma

        # quantization
        qlut = lut.reshape(N, K // self.act_group_size, self.act_group_size // self.g * (1 << self.g))
        absmax = np.max(np.abs(qlut), axis=-1)
        lut_scales = absmax / self.maxv

        def recp(s):
            return 1.0 / s if s != 0 else 0
        
        ils = np.vectorize(recp)(lut_scales).astype(self.out_dtype)
        qlut = np.rint((qlut.transpose(0, 2, 1) * ils).transpose(0, 2, 1).reshape(N, K // self.g, 1 << self.g)).astype(self.dtype)

        return [b_t, lut_scales, lut_biases, qlut]


class QGeMMLUTBitsPreprocessorCodegen_(OpCodegen):
    """Preprocessor of QGeMMLUTBitsCodegen.

    This preprocessor will compute the LUT of the activations,
    and quantize the LUT from `out_dtype` to `dtype`.
    """
    def __init__(self, *args, g: int = 4, act_group_size: int = 64, out_dtype: str = "float16", **kwargs):
        super().__init__(*args, **kwargs)

        self.out_dtype = out_dtype
        if self.dtype == "int8":
            self.maxv = (1 << 7) - 1
        self.g = g
        self.act_group_size = act_group_size
        # Weights mapped from s=[0, 1] to s'=[-1, 1]
        # s' = s * 2 - 1
        # s = (s' + 1) / 2
        self._states = [-1, 1]
        # w = b0 + b1 * 2 + b2 * 4 + b3 * 8 - 8
        #   = 1 / 2 (b0' + gamma * s0) + b1' + b2' * 2 + b3' * 4, where s0 = -1
        self._gamma = 1

    def _define_config(self, cfg):
        self.kfactor = self.act_group_size // self.g
        super()._define_config(cfg)

    def _compute(self, N: int, K: int):
        B = te.placeholder((N, K), dtype=self.out_dtype, name="B")

        k = te.reduce_axis((0, K // self.g), "k")

        packedB = te.compute(
            (N, self.g, K // self.g),
            lambda n, g, k: B[n, k * self.g + g],
            name="packedB",
        )

        LUT_Scales = te.compute(
            (N, K // self.act_group_size),
            lambda n, kk: te.max(
                sum(te.abs(packedB[n, g, kk * self.act_group_size // self.g + k]) for g in range(self.g)),
                axis=k,
            ),
            name="LUT_Scales",
        )
        LUT_Biases = te.placeholder((N, K // self.act_group_size), dtype=self.out_dtype, name="LUT_Biases")

        # placeholder computation
        # should be tensorized
        QLUT = te.compute(
            (N, K // self.g, 1 << self.g),
            lambda n, k, g: (
                B[n, k * self.g + (g % self.g)] / LUT_Scales[n, k * self.g // self.act_group_size] - LUT_Biases[n, k * self.g // self.act_group_size]
            ).astype(self.dtype),
            name="QLUT",
        )

        return [B, LUT_Scales, LUT_Biases, QLUT]

    def _schedule(self, tensors: List[te.Tensor]):
        _, LUT_Scales, LUT_Biases, QLUT = tensors
        sch: te.Schedule = te.create_schedule([QLUT.op])

        n, k, g = sch[QLUT].op.axis
        ko, ki = sch[QLUT].split(k, factor=self.kfactor)
        intrin, ll_code = lut_ctor(
            self.kfactor * 4,
            self.g,
            self.act_group_size,
            self.dtype,
            cc=self.cc,
            cc_opts=self.cc_opts,
            out_dtype=self.out_dtype,
        )
        sch[QLUT].tensorize(ki, intrin)
        sch[QLUT].pragma(ko, "import_llvm", ll_code)

        # Currently the parallelism of preprocessor is disabled due to large thread communication overhead in `benchmark.cc`.
        # But according to profiled results of python side, the overhead is not that large and the best NUM_THREADS should be 4.
        # TODO: Find out the reason for the high communication overhead in C++ side.
        # sch[QLUT].parallel(ko)

        return sch

    def _verify(self, tvm_arrays: List[tvm.nd.NDArray], arrays: List[np.ndarray]):
        tvm.testing.assert_allclose(tvm_arrays[-1].numpy(), arrays[-1], atol=1, rtol=0)
        tvm.testing.assert_allclose(tvm_arrays[-2].numpy(), arrays[-2], atol=1e-2, rtol=1e-2)
        tvm.testing.assert_allclose(tvm_arrays[-3].numpy(), arrays[-3], atol=1e-2, rtol=1e-2)

    def _reference(self, N: int, K: int):
        b = np.random.randn(N, K).astype(self.out_dtype)
        b_t = b

        b = b.reshape(N, K // self.g, self.g)

        codes = np.array([[i] for i in range(1 << self.g)], dtype=np.uint8)
        codes = np.unpackbits(codes, axis=1, bitorder="little", count=self.g).T

        def map_states(c):
            return self._states[c]

        m = np.vectorize(map_states)(codes).astype(self.out_dtype)

        # (N, K // self.g, 1 << self.g)
        lut = b.dot(m)
        lut_biases = lut.reshape(N, K // self.act_group_size, self.act_group_size // self.g, 1 << self.g)[:, :, :, 0]
        lut_biases = np.sum(lut_biases, axis=-1) * self._gamma

        # quantization
        qlut = lut.reshape(N, K // self.act_group_size, self.act_group_size // self.g * (1 << self.g))
        absmax = np.max(np.abs(qlut), axis=-1)
        lut_scales = absmax / self.maxv

        def recp(s):
            return 1.0 / s if s != 0 else 0
        
        ils = np.vectorize(recp)(lut_scales).astype(self.out_dtype)
        qlut = np.rint((qlut.transpose(0, 2, 1) * ils).transpose(0, 2, 1).reshape(N, K // self.g, 1 << self.g)).astype(self.dtype)

        return [b_t, lut_scales, lut_biases, qlut]
