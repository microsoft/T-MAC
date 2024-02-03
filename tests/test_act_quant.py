import numpy as np


def mse(a, b):
    return (np.square(a - b)).mean()


g = 4
K = 4096
dtype = "float16"
act_group_size = 32
maxv = (1 << 7) - 1
states = [0, 1]

sum = 0
for r in range(0, 100):
    codes = np.array([[i] for i in range(1 << g)], dtype=np.uint8)
    codes = np.unpackbits(codes, axis=1, bitorder="little", count=g).T

    def f(c):
        return states[c]

    m = np.vectorize(f)(codes).astype(dtype)

    act = np.random.randn(1, K).astype(dtype)
    c_ref = act.reshape(1, K // g, g).dot(m)

    def quant_before(act: np.ndarray):
        act = act.reshape(1, K // act_group_size, act_group_size)
        absmax = np.max(np.abs(act), axis=-1)
        d = absmax / maxv
        act = (act.transpose(0, 2, 1) / d).transpose(0, 2, 1)
        return act.astype("int8"), d

    act_quant_before, scale = quant_before(act)
    dq_act = act_quant_before.astype(dtype)
    dq_act = (dq_act.transpose(0, 2, 1) * scale).transpose(0, 2, 1).reshape(1, K)

    c_baseline = dq_act.reshape(1, K //g, g).dot(m)
    mse_baseline = mse(c_ref, c_baseline)
    print(mse_baseline)

    def quant_after(act: np.ndarray):
        act = act.reshape(1, K // act_group_size, act_group_size // g * (1 << g))
        absmax = np.max(np.abs(act), axis=-1)
        d = absmax / maxv
        act = (act.transpose(0, 2, 1) / d).transpose(0, 2, 1)
        return act.astype("int8"), d

    act_quant_after, scale = quant_after(c_ref)
    dq_act = (act_quant_after.transpose(0, 2, 1) * scale).transpose(0, 2, 1).reshape(1, K // g, 1 << g)

    c_qgemm = dq_act
    mse_qgemm = mse(c_ref, c_qgemm)
    print(mse_qgemm)

    mse_diff = mse_qgemm / mse_baseline - 1
    sum += mse_diff

print(sum / 100)
