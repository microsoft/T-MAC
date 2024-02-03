import numpy as np
import os
import tvm
from tvm.autotvm.measure.measure_methods import request_remote
from t_mac.ops import QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen
from t_mac.weights import preprocess_weights


dtype = "int8"
target = "llvm -mtriple=arm64-apple-darwin23.1.0 -mcpu=apple-m2"
bits = 4
g = 4
group_size = 128
act_group_size = 64
num_threads = 16
out_dtype = "float16"

remote_kwargs = None
codegen_kwargs = {
    "dtype": dtype,
    "target": target,
    "verify": False,
    "remote_kwargs": remote_kwargs,
    "g": g,
    "act_group_size": act_group_size,
    "save_dir": "/Users/user/jianyu/qgemv_lut/out",
}

preprocessor = QGeMMLUTBitsPreprocessorCodegen(name="preprocessor", **codegen_kwargs)
qgemm = QGeMMLUTBitsCodegen(name="qgemm_lut", bits=bits, group_size=group_size, **codegen_kwargs)

M = 4096 * bits
N = 1
K = 4096

pf, arrays = preprocessor.compile(N, K, num_threads=num_threads)
qf, _ = qgemm.compile(M, N, K, num_threads=num_threads)

bm = qgemm.bm
kfactor = qgemm.kfactor
weight_dtype = qgemm.weight_dtype

# Inputs
Aref = np.random.randint(0, 16, size=(M // bits, K)).astype(weight_dtype)
Sref = np.abs(np.random.randn(M // bits, K // group_size).astype(out_dtype))
Bref = np.random.randn(N, K).astype(out_dtype)

# Outputs
Adq = Aref.T.reshape(K // group_size, group_size, M // bits).astype("float16") - 8
Adq = (Adq.transpose(1, 0, 2) * Sref.T).transpose(1, 0, 2).reshape(K, M // bits)
Cref = Bref.dot(Adq)
print(Cref)

dev = tvm.device("llvm")
# TVM Inputs
A_t, Scales_t = preprocess_weights(Aref, Sref, bm=bm, kfactor=kfactor)
A_t = tvm.nd.array(A_t, dev)
B_t = tvm.nd.array(Bref, dev)
Scales_t = tvm.nd.array(Scales_t, dev)

# TVM Outputs
C_t = tvm.nd.array(Cref, dev)

# TVM Intermediates
LUT_Scales = tvm.nd.array(np.zeros((N, K // act_group_size), dtype=out_dtype), dev)
LUT_Biases = tvm.nd.array(np.zeros((N, K // act_group_size), dtype=out_dtype), dev)
QLUT = tvm.nd.array(np.zeros((N, K // g, 1 << g), dtype=dtype), dev)

pf(B_t, LUT_Scales, LUT_Biases, QLUT)
qf(A_t, QLUT, Scales_t, LUT_Scales, LUT_Biases, C_t)

print(C_t)
