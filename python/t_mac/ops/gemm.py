from .base import OpCodegen
from tvm import te, autotvm
import tvm
from typing import List

import numpy as np


class GeMMCodegen(OpCodegen):

    def _define_config(self, cfg):
        cfg.define_knob("bm", [256, 128, 64, 32])
        cfg.define_knob("bn", [32, 16, 64])
        cfg.define_knob("kfactor", [4, 8])
        super()._define_config(cfg)

    def _compute(self, M: int, N: int, K: int):
        bm = self.bm

        k = te.reduce_axis((0, K), "k")

        A = te.placeholder((M // bm, K, bm), dtype=self.dtype, name="A")
        B = te.placeholder((N, K), dtype=self.dtype, name="B")

        C = te.compute((N, M), lambda n, m: te.sum(A[m // bm, k, tvm.tir.indexmod(m, bm)] * B[n, k], axis=k), name="C")

        return [A, B, C]

    def _schedule(self, tensors: List[te.Tensor]):
        C = tensors[-1]
        sch = te.create_schedule(C.op)

        CC = sch.cache_write(C, "global")
        no, mo, ni, mi = sch[C].tile(C.op.axis[0], C.op.axis[1], self.bn, self.bm)
        sch[CC].compute_at(sch[C], mo)

        nC, mC = sch[CC].op.axis
        (kC,) = sch[CC].op.reduce_axis
        koC, kiC = sch[CC].split(kC, factor=self.kfactor)
        sch[CC].reorder(koC, nC, kiC, mC)

        sch[CC].vectorize(mC)
        sch[CC].unroll(kiC)

        sch[C].parallel(mo)

        return sch

    def _reference(self, M: int, N: int, K: int):
        a = np.random.randn(M, K).astype(self.dtype)
        b = np.random.randn(N, K).astype(self.dtype)
        answer = np.dot(b, a.T)

        # DEBUGGING: intermediates
        a_t = a.reshape(M // self.bm, self.bm, K).transpose(0, 2, 1)

        return [a_t, b, answer]


class GeMMCLCodegen(OpCodegen):

    def _define_config(self, cfg):
        cfg.define_knob("bk", [8])
        cfg.define_knob("num_thread", [8])
        cfg.define_knob("num_block", [2])
        super()._define_config(cfg)

    def _compute(self, M: int, N: int, K: int):
        bk = self.bk

        k = te.reduce_axis((0, K), "k")

        A = te.placeholder((M, K // bk, bk), dtype=self.dtype, name="A")
        B = te.placeholder((N, K), dtype=self.dtype, name="B")

        C = te.compute((N, M), lambda n, m: te.sum(A[m, k // bk, m % bk] * B[n, k], axis=k), name="C")

        return [A, B, C]

    def _schedule(self, tensors: List[te.Tensor]):
        A, B, C = tensors
        sch = te.create_schedule(C.op)

        (k,) = sch[C].op.reduce_axis

        CC = sch.cache_write(C, "local")

        block_x = te.thread_axis("blockIdx.x")
        block_y = te.thread_axis("blockIdx.y")
        thread_x = te.thread_axis("threadIdx.x")
        thread_y = te.thread_axis("threadIdx.y")

        thread_xz = te.thread_axis((0, 2), "vthread", name="vx")
        thread_yz = te.thread_axis((0, 2), "vthread", name="vy")

        # pay, pai = sch[A].split(A.op.axis[0], nparts=self.num_thread)
        # pax, paj = sch[A].split(A.op.axis[1], nparts=self.num_thread)
        # sch[A].bind(pay, thread_y)
        # sch[A].bind(pax, thread_x)
        # paz, pak = sch[A].split(A.op.axis[2], factor=8)
        # sch[A].vectorize(pak)

        by, yi = sch[C].split(C.op.axis[0], nparts=self.num_block)
        bx, xi = sch[C].split(C.op.axis[1], nparts=self.num_thread)

        sch[C].bind(by, block_y)
        sch[C].bind(bx, thread_y)
        sch[C].reorder(by, bx, yi, xi)

        tyz, yi = sch[C].split(yi, nparts=2)
        ty, yi = sch[C].split(yi, nparts=self.num_block)
        txz, xi = sch[C].split(xi, nparts=2)
        tx, xi = sch[C].split(xi, nparts=self.num_thread)

        sch[C].reorder(tyz, txz, ty, tx, yi, xi)
        sch[C].bind(tyz, thread_yz)
        sch[C].bind(txz, thread_xz)

        sch[C].bind(ty, block_x)
        sch[C].bind(tx, thread_x)

        xyi, xxi = sch[C].split(xi, factor=8)
        sch[C].reorder(tyz, txz, ty, tx, yi, xyi, xxi)
        sch[C].vectorize(xxi)

        sch[CC].compute_at(sch[C], yi)
        yo, xo = CC.op.axis
        sch[CC].reorder(k, yo, xo)
        xo, xi = sch[CC].split(xo, factor=8)
        sch[CC].vectorize(xi)

        ko, ki = sch[CC].split(k, factor=2)
        sch[CC].unroll(ki)
        return sch

    def _reference(self, M: int, N: int, K: int):
        a = np.random.randn(M, K).astype(self.dtype)
        b = np.random.randn(N, K).astype(self.dtype)
        answer = np.dot(b, a.T)

        # DEBUGGING: intermediates
        a_t = np.random.randn(M, K // self.bk, self.bk).astype(self.dtype)

        return [a_t, b, answer]
