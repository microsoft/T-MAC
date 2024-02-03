from .base import OpCodegen
from tvm import te, autotvm
import tvm
import numpy as np
from ..intrins.tbl import tbl

from typing import List


class QGeMVCodegen(OpCodegen):

    def __init__(self, *args, bits: int = 1, **kwargs):
        super().__init__(*args, **kwargs)

        self.weight_dtype = "uint8"
        self._num_per_elem = 8 // bits

    def _compute(self, M: int, N: int, K: int):
        k = te.reduce_axis((0, K), "k")

        A = te.placeholder((M, K // self._num_per_elem), dtype=self.weight_dtype, name="A")
        B = te.placeholder((N, K), dtype=self.dtype, name="B")

        C = te.compute(
            (N, M),
            lambda n, m: te.sum(
                te.if_then_else(
                    (A[m, k // self._num_per_elem] & (1 << (k % self._num_per_elem))) != 0,
                    B[n, k],
                    -B[n, k],
                ),
                axis=k,
            ),
            name="C",
        )

        return [A, B, C]

    def _schedule(self, tensors):
        out = tensors[-1]
        sch = te.create_schedule(out.op)

        cfg = autotvm.get_config()
        cfg.define_knob("bn", [64])
        no, mo, ni, mi = sch[out].tile(out.op.axis[0], out.op.axis[1], 1, cfg["bn"].val)
        sch[out].parallel(mo)
        return sch

    def _reference(self, M: int, N: int, K: int):
        a = np.random.randn(M, K // self._num_per_elem).astype(self.weight_dtype)
        b = np.random.randn(N, K).astype(self.dtype)
        a_decode = np.unpackbits(a, axis=1, bitorder="little").astype(self.dtype)
        a_decode = np.where(a_decode == 1, 1, -1).astype(self.dtype)
        c = np.dot(b, a_decode.T)
        return [a, b, c]


class QGeMVLUTCodegen_(QGeMVCodegen):

    def __init__(self, *args, g: int = 4, **kwargs):
        super().__init__(*args, **kwargs)

        self.g = g
        self._ngroups_per_elem = self._num_per_elem // self.g

    def _compute(self, M: int, N: int, K: int):
        k = te.reduce_axis((0, K // self.g), "k")

        A = te.placeholder((K // self.g, M // self._ngroups_per_elem), dtype=self.weight_dtype, name="A")
        B = te.placeholder((N, K), dtype=self.dtype, name="B")

        LUT = te.compute(
            (N, K // self.g, 2 ** self.g),
            lambda n, k, i: i.astype(self.dtype),
            name="LUT",
        )

        mask = te.const((1 << self.g) - 1, dtype=self.dtype)

        def _get_Abits(m, k):
            return (A[k, m // self._ngroups_per_elem] >> (self.g * (m % self._ngroups_per_elem))) & mask

        C = te.compute(
            (N, M),
            lambda n, m: te.sum(LUT[n, k, _get_Abits(m, k)], axis=k),
            name="C",
        )

        return [A, B, C]

    def _schedule(self, tensors: List[te.Tensor]):
        out = tensors[-1]
        sch: te.Schedule = te.create_schedule(out.op)
        cfg = autotvm.get_config()

        (k,) = sch[out].op.reduce_axis
        cfg.define_knob("num_threads", [tvm.runtime.num_threads()])
        ko, ki = sch[out].split(k, nparts=cfg["num_threads"].val)
        outF = sch.rfactor(out, ko)

        n, m = sch[out].op.axis

        (k,) = sch[out].op.reduce_axis
        sch[out].reorder(k, n, m)
        sch[outF].compute_at(sch[out], k)

        koF, nF, mF = sch[outF].op.axis
        (kiF,) = sch[outF].op.reduce_axis
        sch[outF].reorder(koF, kiF, nF, mF)

        cfg.define_knob("gemv_factor", [1024])
        gemv_factor = cfg["gemv_factor"].val
        moF, miF = sch[outF].split(mF, factor=gemv_factor)

        intrin, ll_code = tbl((1 << self.g) // 16, gemv_factor, self.g, self._ngroups_per_elem, self.dtype)
        sch[outF].tensorize(miF, intrin)
        sch[outF].pragma(koF, "import_llvm", ll_code)

        # TODO: parallel reduction not supported on CPU
        sch[out].parallel(k)

        return sch

    def _reference(self, M: int, N: int, K: int):
        a, b, c = super()._reference(M, N, K)

        # DEBUGGING: intermediates
        a_t = np.random.randn(K // self.g, M // self._num_per_elem * self.g).astype(self.weight_dtype)

        return [a_t, b, c]


class QGeMVLUTCodegen(QGeMVCodegen):

    def __init__(self, *args, g: int = 4, **kwargs):
        super().__init__(*args, **kwargs)

        self.g = g
        self._ngroups_per_elem = self._num_per_elem // self.g

    def _compute(self, M: int, N: int, K: int):
        cfg = autotvm.get_config()
        cfg.define_knob("num_threads", [tvm.runtime.num_threads()])
        T = cfg["num_threads"].val

        k = te.reduce_axis((0, K // self.g // T), "k")
        t = te.reduce_axis((0, T), "t")

        A = te.placeholder((K // self.g, M // self._ngroups_per_elem), dtype=self.weight_dtype, name="A")
        B = te.placeholder((N, K), dtype=self.dtype, name="B")

        LUT = te.compute(
            (N, K // self.g, 2 ** self.g),
            lambda n, k, i: i.astype(self.dtype),
            name="LUT",
        )

        mask = te.const((1 << self.g) - 1, dtype=self.dtype)

        def _get_Abits(m, k, t):
            return (A[K // self.g // T * t + k, m // self._ngroups_per_elem] >> (self.g * (m % self._ngroups_per_elem))) & mask

        CF = te.compute(
            (T, N, M),
            lambda t, n, m: te.sum(LUT[n, K // self.g // T * t + k, _get_Abits(m, k, t)], axis=k),
            name="CF",
        )

        C = te.compute(
            (N, M),
            lambda n, m: te.sum(CF[t, n, m], axis=t),
            name="C",
        )

        return [A, B, C]

    def _schedule(self, tensors: List[te.Tensor]):
        out = tensors[-1]
        sch: te.Schedule = te.create_schedule(out.op)
        cfg = autotvm.get_config()

        outF = out.op.input_tensors[0]
        tF, nF, mF = sch[outF].op.axis
        outFC = sch.cache_write(outF, "global")
        sch[outFC].compute_at(sch[outF], tF)

        tFC, nFC, mFC = sch[outFC].op.axis
        (kFC,) = sch[outFC].op.reduce_axis
        sch[outFC].reorder(tFC, nFC, kFC, mFC)

        cfg.define_knob("gemv_factor", [1024])
        gemv_factor = cfg["gemv_factor"].val
        moFC, miFC = sch[outFC].split(mFC, factor=gemv_factor)

        intrin, ll_code = tbl((1 << self.g) // 16, gemv_factor, self.g, self._ngroups_per_elem, self.dtype)
        sch[outFC].tensorize(miFC, intrin)
        sch[outFC].pragma(tFC, "import_llvm", ll_code)

        sch[outF].parallel(tF)

        n, m = sch[out].op.axis
        (t,) = sch[out].op.reduce_axis
        sch[out].reorder(t, n, m)
        outC = sch.cache_write(out, "global")
        mo, mi = sch[out].split(m, nparts=cfg["num_threads"].val)
        sch[outC].compute_at(sch[out], mo)
        sch[out].parallel(mo)

        return sch

    def _reference(self, M: int, N: int, K: int):
        a, b, c = super()._reference(M, N, K)

        # DEBUGGING: intermediates
        a_t = np.random.randn(K // self.g, M // self._num_per_elem * self.g).astype(self.weight_dtype)

        return [a_t, b, c]
