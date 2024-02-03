import numpy as np


g = 4
K = 4096
M = 4096
dtype = "float16"

act = np.random.randn(1, K).astype(dtype)
w0 = np.random.randn(M, K // g, 1).astype(dtype)
weights = np.concatenate([w0] * g, axis=-1).reshape(M, K).T
w0 = w0.reshape(M, K // g).T
c_ref = act.dot(weights)
print(c_ref)

act_group_size = 32
maxv = (1 << 7) - 1


def mse(a, b):
    return (np.square(a - b)).mean()


def quant_before(act: np.ndarray):
    act = act.reshape(1, K // act_group_size, act_group_size)
    absmax = np.max(np.abs(act), axis=-1)
    d = absmax / maxv
    act = (act.transpose(0, 2, 1) / d).transpose(0, 2, 1)
    return act.astype("int8"), d


act_quant_before, scale = quant_before(act)
dq_act = act_quant_before.astype(dtype)
dq_act = (dq_act.transpose(0, 2, 1) * scale).transpose(0, 2, 1).reshape(1, K)

c_before = dq_act.dot(weights)
print(c_before)
print(mse(c_ref, c_before))


def quant_after(act: np.ndarray):
    act = act.reshape(1, K // g, g)
    abs_act_sum = np.sum(np.abs(act), axis=-1).reshape(1, K // act_group_size, act_group_size // g)
    absmax = np.max(abs_act_sum, axis=-1)
    act = np.sum(act, axis=-1).reshape(1, K // act_group_size, act_group_size // g)
    d = absmax / maxv
    act = (act.transpose(0, 2, 1) / d).transpose(0, 2, 1)
    return act.astype("int8"), d

act_quant_after, scale = quant_after(act)
dq_act = (act_quant_after.transpose(0, 2, 1) * scale).transpose(0, 2, 1).reshape(1, K // g)

c_after = dq_act.dot(w0)
print(c_after)
print(mse(c_ref, c_after))


# class Test:
#     def __init__(self):
#         self.bm = 512
#         self.simd_width = 128
#         self.g = 4
#         self._ngroups_per_elem = 2
#         self.kfactor = 8
#         self.act_group_size = 32
#         self.bits = 1
#         self.group_size = 128
#         self.out_dtype = "float16"
#         self._states = [0, 1]
#         self.maxv = 127
#         self.weight_dtype = "uint8"
#         self.dtype = "int8"

#     def _reference(self, M, K, N):
#         b = np.random.randn(N, K).astype(self.out_dtype)

#         b = b.reshape(N, K // self.g, self.g)

#         codes = np.array([[i] for i in range(1 << self.g)], dtype=np.uint8)
#         codes = np.unpackbits(codes, axis=1, bitorder="little", count=self.g).T

#         def f(c):
#             return self._states[c]

#         m = np.vectorize(f)(codes).astype(self.out_dtype)

#         # (N, K // self.g, 1 << self.g)
#         lut = b.dot(m)

#         # quantization
#         qlut = lut.reshape(N, K // self.act_group_size, self.act_group_size // self.g * (1 << self.g))
#         absmax = np.max(np.abs(qlut), axis=-1)
#         lut_scale = absmax / self.maxv
#         qlut = (qlut.transpose(0, 2, 1) / lut_scale).transpose(0, 2, 1).reshape(N, K // self.g, 1 << self.g).astype(self.dtype)

#         simd_n_uint8 = self.simd_width // 8
#         a = np.random.randn(M // self.bm, K // self.g // self.kfactor, self.bm // self._ngroups_per_elem // simd_n_uint8, self.kfactor, simd_n_uint8).astype(self.weight_dtype)
#         a_t = a.reshape(M // self.bm, K // self.g, self.bm // self._ngroups_per_elem)

#         lut = qlut
#         lut_scales = lut_scale

#         simd_n_float16 = self.simd_width // 16
#         scales = np.random.randn(M // self.bm, K // self.group_size, self.bm // self.bits // simd_n_float16, simd_n_float16).astype(self.out_dtype)
#         scales_t = scales.reshape(M // self.bm, K // self.group_size, self.bm // self.bits)

#         c = np.zeros((N, M), dtype=self.out_dtype)

#         a = np.concatenate([(a >> (self.g * ng)) & ((1 << self.g) - 1) for ng in range(self._ngroups_per_elem)], axis=-1)
#         for n in range(N):
#             for k in range(K // self.g):
#                 for m in range(M):
#                     mo = m // self.bm
#                     ko = k // self.kfactor
#                     mi = (m % self.bm) // self._ngroups_per_elem // simd_n_uint8
#                     ki = k % self.kfactor
#                     e = (m % self.bm) % (self._ngroups_per_elem * simd_n_uint8)
#                     a_e = a[mo, ko, mi, ki, e]

#                     scales_mi = (m % self.bm) // self.bits // simd_n_float16
#                     scales_e = ((m % self.bm) % simd_n_float16)
#                     c[n, m] += lut[n, k, a_e] * lut_scales[n, k * self.g // self.act_group_size] * scales[mo, k * self.g // self.group_size, scales_mi, scales_e]
        

