import numpy as np
import os
import tvm
from tvm.autotvm.measure.measure_methods import request_remote
from t_mac.ops import QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen
from t_mac.weights import preprocess_weights
from t_mac.utils import get_default_device_kwargs, nmse
import logging


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

bits = 2
M = 4096 * bits
N = 1
K = 4096
zero_point = True
dtype = "int8"
g = 4
group_size = 128
act_group_size = 64
m_groups = -1  # should be -1 or 1 in test_e2e.py

if act_group_size == -1:
    act_group_size = K

device_kwargs = get_default_device_kwargs()

out_dtype = device_kwargs["out_dtype"]

remote_kwargs = None
codegen_kwargs = {
    "save_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out"),
    "dtype": dtype,
    "target": device_kwargs["target"],
    "verify": True,
    "tune": False,
    "remote_kwargs": device_kwargs["remote_kwargs"],
    "bits": bits,
    "out_dtype": out_dtype,
    "act_group_size": act_group_size,
    "cc_opts": device_kwargs["cc_opts"],
}

preprocessor = QGeMMLUTBitsPreprocessorCodegen(name="preprocessor", fast_aggregation_k=0, **codegen_kwargs)
qgemm = QGeMMLUTBitsCodegen(name="qgemm_lut", group_size=group_size, m_groups=m_groups, aggregation_dtype=device_kwargs["aggregation_dtype"], zero_point=zero_point, **codegen_kwargs)

pf, _ = preprocessor.compile(N, K)
qf, _ = qgemm.compile(M, N, K)

bm = qgemm.bm
kfactor = qgemm.kfactor
weight_dtype = qgemm.weight_dtype

# Inputs
Aref = np.random.randint(0, 2 ** bits, size=(M // bits, K)).astype(weight_dtype)
Zref = None
if m_groups == -1:
    Sref = np.abs(np.random.randn(M // bits, K // group_size).astype(out_dtype))
    if zero_point:
        Zref = np.random.randn(M // bits, K // group_size).astype(out_dtype)
else:
    Sref = np.abs(np.random.randn(m_groups,).astype(out_dtype))
Bref = np.random.randn(N, K).astype(out_dtype)

# Outputs
if m_groups == -1:
    Adq = Aref.T.reshape(K // group_size, group_size, M // bits).astype(out_dtype) - (2 ** (bits - 1))
    Adq = Adq.transpose(1, 0, 2) * Sref.T
    if zero_point:
        Adq = Adq - Zref.T
    Adq = Adq.transpose(1, 0, 2).reshape(K, M // bits)
else:
    Adq = (Aref.T.astype(out_dtype) - (2 ** (bits - 1))) * Sref[0]

Cref = Bref.dot(Adq)
print(Cref)

dev = tvm.device("llvm")
# TVM Inputs
A_t, Scales_t = preprocess_weights(Aref, Sref, Zref, bits=bits, g=g, bm=bm, kfactor=kfactor)
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

print(C_t.numpy())

print(nmse(Cref, C_t.numpy()))
