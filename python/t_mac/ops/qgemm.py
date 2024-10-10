from .base import OpCodegen
from tvm import te
import tvm
import tvm.testing
import numpy as np
from ..intrins import tbl, lut_ctor, partial_max
from ..utils import get_bits_alphas, nmse
from tvm.error import TVMError

from typing import List
import logging

logger = logging.getLogger("qgemm")


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
        m_groups: int = -1,
        aggregation_dtype: str = "int32",
        fast_aggregation: bool = False,
        zero_point: bool = False,
        **kwargs
    ):
        """
        Parameters
        ----------
        bits : int
            Number of bits for each group.
        g : int
            Group size of LUT.
        group_size : int
            Group size of weights.
        act_group_size : int
            Group size of activations.
        out_dtype : str
            Output data type.
        simd_n_in : int
            Number of SIMD lanes for input.
            NEON-ui8: 16 (= 128 / 8)
            AVX2-ui8: 16 (= 128 / 8)
        simd_n_out : int
            Number of SIMD lanes for output.
            NEON-f16: 8 (= 128 / 16)
            AVX2-f32: 8 (= 256 / 32)
            AVX2-i32: 8 (= 256 / 32)
        m_groups : int
            Number of groups for M. group_size is invalid if m_groups != -1.
            - -1 for GPTQ-like fine-grained scales
            - 1/2/3 for BitNet-like unified scales
        aggregation_dtype : str
            Data type for aggregation. Only be used if do_scale_final is True.
        fast_aggregation: bool
            Toggle fast aggregation. Additional speedup but with nonnegligible error.
        zero_point: bool
            scales with zero point. Currenlty only available for GTPQ-like scales.
            The zero points and scales are stack together.
        """
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
        self.simd_n_in = simd_n_in
        self.simd_n_out = simd_n_out
        # w = b0 + b1 * 2 + b2 * 4 + b3 * 8 - 8
        #   = 1 / 2 (b0' + gamma * s0) + b1' + b2' * 2 + b3' * 4, where s0 = -1
        self.alphas = get_bits_alphas(bits)
        self.m_groups = m_groups
        self.aggregation_dtype = aggregation_dtype
        self.fast_aggregation = fast_aggregation
        self.zero_point = zero_point
        if self.m_groups != -1:
            if self.zero_point:
                logger.warning("Currently zero point is not supported for BitNet-like scales")
                self.zero_point = False

    def do_scale_final(self, K: int):
        # Current implementation decides do_scale_final only by m_group_size
        # Consider fine-grained lut_scale for m_groups == -1?
        return self.m_groups != -1 and self.act_group_size == K

    def _define_config(self, cfg, M: int, N: int, K: int):
        if self.bits == 3:
            bms = [192, 384, 576, 768]
        else:
            bms = [256, 128, 512, 1024, 320, 640]
        bns = [8, 16, 32, 64]
        kfactors = [8, 16]

        cfg.define_knob("bm", [bm for bm in bms if (M % bm == 0) and (bm % self.bits == 0)])
        cfg.define_knob("bn", [8, 16, 32, 64])
        if N <= 8:
            cfg.define_knob("bn", [8])
        else:
            cfg.define_knob("bn", [bn for bn in bns if (N % bn == 0)])
        if not self.do_scale_final(K):
            w_group_size = self.group_size if self.m_groups == -1 else K
            kfactors = [k for k in kfactors if ((k * self.g) % self.act_group_size == 0) and (w_group_size % (k * self.g) == 0)]
        cfg.define_knob("kfactor", kfactors)
        super()._define_config(cfg)

    def _compute(self, M: int, N: int, K: int):
        bm = self.bm
        if M % bm != 0:
            raise TVMError("M({}) must be divisible by bm({})".format(M, bm))
        if bm % self.bits != 0:
            raise TVMError("bm({}) must be divisible by bits({})".format(bm, self.bits))
        if N >= self.bn and N % self.bn != 0:
            raise TVMError("N({}) must be divisible by bn({})".format(N, self.bn))
        if K % self.act_group_size != 0:
            raise TVMError("K({}) must be devisible by act_group_size({})".format(K, self.act_group_size))

        k = te.reduce_axis((0, K // self.g), "k")

        A = te.placeholder((M // bm, K // self.g, bm // self._ngroups_per_elem), dtype=self.weight_dtype, name="A")
        LUT = te.placeholder((N, K // self.g, 2 ** self.g), dtype=self.dtype, name="LUT")

        if self.m_groups == -1:
            if K % self.group_size != 0:
                raise TVMError("K({}) must be devisible by group_size({})".format(K, self.group_size))
            if self.zero_point:
                scales_shape = (M // bm, K // self.group_size, bm // self.bits * 2)
                def _get_scale(m, k):
                    # Fake _get_scale, should be tensorized
                    return Scales[m // bm, k * self.g // self.group_size, (m % bm) // self.bits * 2] - Scales[m // bm, k * self.g // self.group_size, (m % bm) // self.bits * 2 + 1]
            else:
                scales_shape = (M // bm, K // self.group_size, bm // self.bits)
                def _get_scale(m, k):
                    return Scales[m // bm, k * self.g // self.group_size, (m % bm) // self.bits]
        else:
            m_group_size = M // self.m_groups
            scales_shape = (self.m_groups,)
            def _get_scale(m, k):
                return Scales[m // m_group_size]

        Scales = te.placeholder(scales_shape, dtype=self.out_dtype, name="Scales")

        alphas = [te.const(alpha, dtype=self.out_dtype) for alpha in self.alphas]

        if self.has_lut_scale:
            LUT_Scales = te.placeholder((N, K // self.act_group_size), dtype=self.out_dtype, name="LUT_Scales")
            LUT_Biases = te.placeholder((N, K // self.act_group_size), dtype=self.out_dtype, name="LUT_Biases")
            def _lut_scale(n, k, val):
                return val * LUT_Scales[n, k * self.g // self.act_group_size] + LUT_Biases[n, k * self.g // self.act_group_size] * alphas[0]
        else:
            def _lut_scale(n, k, val):
                return val

        if not self.do_scale_final(K):
            def _scale_first(m, n, k, lut_val):
                return _lut_scale(n, k, lut_val.astype(self.out_dtype)) * _get_scale(m, k)
            def _scale_final(m, n, cbits_sum):
                return cbits_sum
        else:
            def _scale_first(m, n, k, lut_val):
                return lut_val.astype(self.aggregation_dtype)
            def _scale_final(m, n, cbits_sum):
                return _lut_scale(n, 0, cbits_sum) * _get_scale(m, k)

        mask = te.const((1 << self.g) - 1, dtype=self.weight_dtype)

        def _get_Abits(m, k):
            return (A[m // bm, k, (m % bm) // self._ngroups_per_elem] >> (self.g * ((m % bm) % self._ngroups_per_elem))) & mask

        # placeholder computation
        # should be tensorized
        CBits = te.compute(
            (N, M),
            lambda n, m: te.sum(
                _scale_first(m, n, k, LUT[n, k, _get_Abits(m, k)]),
                axis=k,
            ),
            name="CBits",
        )

        C = te.compute(
            (N, M // self.bits),
            lambda n, m: _scale_final(m, n,
                sum([
                    CBits[
                        n,
                        te.indexdiv(m, self.simd_n_out) * self.simd_n_out * self.bits
                            + te.indexmod(m, self.simd_n_out)
                            + b * self.simd_n_out
                    ].astype("float32") * alphas[b]
                    for b in range(self.bits)
                ]),
            ).astype(self.out_dtype),
            name="C",
        )

        if self.has_lut_scale:
            return [A, LUT, Scales, LUT_Scales, LUT_Biases, C]
        else:
            return [A, LUT, Scales, C]

    def _schedule(self, tensors: List[te.Tensor]):
        LUT = tensors[1]
        K = int(LUT.shape[1] * self.g)
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

        intrin, ll_code, header_code, body_code = tbl(
            self.bm,
            self.kfactor,
            self.g,
            self.act_group_size,
            self._ngroups_per_elem,
            self.bits,
            self.dtype,
            cc=self.cc,
            cc_opts=self.cc_opts,
            has_scale=True,
            has_lut_scale=self.has_lut_scale,
            out_dtype=self.out_dtype,
            m_groups=self.m_groups,
            do_scale_final=self.do_scale_final(K),
            aggregation_dtype=self.aggregation_dtype,
            fast_aggregation=self.fast_aggregation,
            zero_point=self.zero_point,
        )
        self.extra_cc_header = header_code
        self.extra_cc_body = body_code
        sch[CBits].tensorize(kiC, intrin)
        sch[CBits].pragma(koC, "import_llvm", ll_code)

        nC, mC = sch[CC].op.axis
        # 32 for vectorization should be enough
        moC, miC = sch[CC].split(mC, factor=32)
        mio, mii = sch[C].split(mi, factor=32)
        if self._vectorization:
            sch[CC].vectorize(miC)
            sch[C].vectorize(mii)
        else:
            sch[CC].unroll(miC)
            sch[C].unroll(mii)

        if self.num_threads > 1:
            N = int(C.shape[0])
            if (N // self.bn >= self.num_threads):
                sch[C].parallel(no)
            else:
                sch[C].parallel(mo)

        return sch

    def _verify(self, tvm_arrays: List[tvm.nd.NDArray], arrays: List[np.ndarray]):
        max_err = 5e-4
        err = nmse(tvm_arrays[-1].numpy(), arrays[-1])
        logger.info("NMSE: {}".format(err))
        if err > max_err:
            logger.warning("tvm_arrays not close to arrays with nmse: {}\ntvm_arrays: {}\narrays: {}".format(err, tvm_arrays[-1].numpy(), arrays[-1]))

    def _reference(self, M: int, N: int, K: int):
        # TODO: rewrite
        a = np.random.randint(0, 256, (M // self.bm, K // self.g // self.kfactor, self.bm // self._ngroups_per_elem // self.simd_n_in, self.kfactor, self.simd_n_in)).astype(self.weight_dtype)
        a_t = a.reshape(M // self.bm, K // self.g, self.bm // self._ngroups_per_elem)

        lut = np.random.randint(-127, 127, (N, K // self.g, 2 ** self.g)).astype(self.dtype)

        if self.m_groups == -1:
            if self.zero_point:
                scales = np.random.randn(M // self.bm, K // self.group_size, self.bm // self.bits // self.simd_n_out, 2, self.simd_n_out).astype(self.out_dtype)
                scales_t = scales.reshape(M // self.bm, K // self.group_size, self.bm // self.bits * 2)
            else:
                scales = np.random.randn(M // self.bm, K // self.group_size, self.bm // self.bits // self.simd_n_out, self.simd_n_out).astype(self.out_dtype)
                scales_t = scales.reshape(M // self.bm, K // self.group_size, self.bm // self.bits)
        else:
            scales = np.random.randn(self.m_groups).astype(self.out_dtype)
            scales_t = scales

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

                    if self.m_groups == -1:
                        if self.zero_point:
                            s = scales[mo, k * self.g // self.group_size, scales_mi, 0, scales_e]
                        else:
                            s = scales[mo, k * self.g // self.group_size, scales_mi, scales_e]
                    else:
                        m_group_size = M // self.m_groups
                        s = scales[m // m_group_size]

                    cbits[n, m] += lut[n, k, a_e] * lut_scales[n, k * self.g // self.act_group_size] * s
                    if (((k * self.g) % self.act_group_size) == 0) and ((((m % self.bm) // self.simd_n_out) % self.bits) == 0):
                        cbits[n, m] += lut_biases[n, k * self.g // self.act_group_size] * s
                        if self.zero_point:
                            cbits[n, m] += lut_biases[n, k * self.g // self.act_group_size] * (1 / self.alphas[0]) * scales[mo, k * self.g // self.group_size, scales_mi, 1, scales_e]

        c = (
            cbits.reshape((N, M // self.simd_n_out // self.bits, self.bits, self.simd_n_out))
                .transpose(0, 1, 3, 2)
                .dot(np.array(self.alphas, dtype=self.out_dtype))
                .reshape((N, M // self.bits))
        )

        if self.has_lut_scale:
            return [a_t, lut, scales_t, lut_scales, lut_biases, c]
        else:
            return [a_t, lut, scales_t, c]

    def get_template_name(self, M: int, N: int, K: int) -> str:
        return super().get_template_name() + f"_m{M}_k{K}_n{N}_b{self.bits}"


class QGeMMLUTBitsPreprocessorCodegen(OpCodegen):
    """Preprocessor of QGeMMLUTBitsCodegen.

    This preprocessor will compute the LUT of the activations,
    and quantize the LUT from `out_dtype` to `dtype`.
    """
    def __init__(
        self,
        *args,
        g: int = 4,
        act_group_size: int = 64,
        out_dtype: str = "float16",
        bits: int = 4,
        fast_aggregation_k: int = 16,
        M: int = 0,
        **kwargs
    ):
        """
        Parameters:
        -----------
        M: int
            Used by get_template_name.
            Could be useful if the tuned parameters are different for different M.
        """
        super().__init__(*args, **kwargs)

        self.out_dtype = out_dtype
        if self.dtype == "int8":
            self.maxv = (1 << 7) - 1
        self.g = g
        self.act_group_size = act_group_size
        self.bits = bits
        # Weights mapped from s=[0, 1] to s'=[-1, 1]
        # s' = s * 2 - 1
        # s = (s' + 1) / 2
        self._states = [-1, 1]
        # w = b0 + b1 * 2 + b2 * 4 + b3 * 8 - 8
        #   = 1 / 2 (b0' + gamma * s0) + b1' + b2' * 2 + b3' * 4, where s0 = -1
        self._gamma = 1
        self.fast_aggregation_k = fast_aggregation_k
        self.M = M

    def _define_config(self, cfg, *args):
        self.kfactor = self.act_group_size // self.g
        super()._define_config(cfg)

    def _compute(self, N: int, K: int):
        if K % self.act_group_size != 0:
            raise TVMError("K({}) must be devisible by act_group_size({})".format(K, self.act_group_size))
        # TODO: modify partial_max to support self.act_group_size % 32 != 0
        if self.act_group_size % 32 != 0:
            raise TVMError("act_group_size({}) must be devisible by 32".format(self.act_group_size))

        B = te.placeholder((N, K), dtype=self.out_dtype, name="B")

        sk = te.reduce_axis((0, self.act_group_size // self.g), "k")
        # TODO: fuse with QLUT compute
        LUT_Scales = te.compute(
            (N, K // self.act_group_size),
            lambda n, kk: te.max(
                te.abs(sum(B[n, kk * self.act_group_size + sk * self.g + g] for g in range(self.g))) / self.maxv,
                axis=sk,
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

        sk = sch[LUT_Scales].op.reduce_axis[0]
        sn, skk = sch[LUT_Scales].op.axis
        skko, skki = sch[LUT_Scales].split(skk, factor=1)
        sko, ski = sch[LUT_Scales].split(sk, factor=8)
        sch[LUT_Scales].reorder(sn, skko, sko, skki, ski)
        intrin, ll_code, _, _ = partial_max(
            self.g,
            self.dtype,
            cc=self.cc,
            cc_opts=self.cc_opts,
            out_dtype=self.out_dtype,
        )
        sch[LUT_Scales].tensorize(skki, intrin)
        sch[LUT_Scales].pragma(sko, "import_llvm", ll_code)

        n, k, g = sch[QLUT].op.axis
        ko, ki = sch[QLUT].split(k, factor=self.kfactor)
        intrin, ll_code, header_code, body_code = lut_ctor(
            self.kfactor * 4,
            self.g,
            self.act_group_size,
            self.bits,
            self.dtype,
            cc=self.cc,
            cc_opts=self.cc_opts,
            out_dtype=self.out_dtype,
            fast_aggregation_k=self.fast_aggregation_k,
        )
        self.extra_cc_header = header_code
        self.extra_cc_body = body_code
        sch[QLUT].tensorize(ki, intrin)
        sch[QLUT].pragma(ko, "import_llvm", ll_code)

        # Currently the parallelism of preprocessor is disabled due to large thread communication overhead in `benchmark.cc`.
        # But according to profiled results of python side, the overhead is not that large and the best NUM_THREADS should be 4.
        # TODO: Find out the reason for the high communication overhead in C++ side.
        # if self.num_threads > 1:
        #     sch[QLUT].parallel(ko)

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
        qlut = np.rint(
            (qlut.transpose(2, 0, 1).reshape(-1, qlut.shape[0] * qlut.shape[1]) * ils.reshape(1, qlut.shape[0] * qlut.shape[1]))
            .reshape(qlut.shape[2], qlut.shape[0], qlut.shape[1]).transpose(1, 2, 0).reshape(N, K // self.g, 1 << self.g)
        ).astype(self.dtype)

        return [b_t, lut_scales, lut_biases, qlut]

    def get_template_name(self, N: int, K: int) -> str:
        return super().get_template_name() + f"_m{self.M}_k{K}_n{N}_b{self.bits}"
