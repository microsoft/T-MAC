import tvm
import tvm.testing
from tvm import te
import numpy

# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = 8192
K = 16384
N = 1

# The default tensor type in tvm
dtype = "int8"

# using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
# To get the best performance, please change the following line
# to llvm -mcpu=core-avx2, or specific type of CPU you use
target = "llvm -mtriple=arm64-apple-darwin23.1.0 -mcpu=apple-m2"
dev = tvm.runtime.device("cpu")

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

answer = numpy.dot(a.numpy(), b.numpy())

# Algorithm
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), dtype=dtype, name="A")
B = te.placeholder((K, N), dtype=dtype, name="B")
C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

bn = 64
kfactor = 4

s = te.create_schedule(C.op)
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, 1)
# (kaxis,) = s[C].op.reduce_axis
# ko, ki = s[C].split(kaxis, factor=kfactor)

# # re-ordering
# s[C].reorder(mo, no, ko, mi, ki, ni)
# s[C].vectorize(ni)

# # prefetch
# s[C].prefetch(A, mi, 1)

# parallel
s[C].parallel(mo)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt0: %f" % evaluator(a, b, c).mean)

################################################################################################
# Here is the generated IR after loop permutation.

print(tvm.lower(s, [A, B, C], simple_mode=True))
