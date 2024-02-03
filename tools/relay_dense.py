import tvm
from tvm import relay, autotvm
from tvm.relay.op.contrib.bnns import partition_for_bnns
from tvm.contrib import graph_executor
import numpy as np
from tvm.autotvm.measure.measure_methods import request_remote
import os
from tvm.contrib import utils, ndk
from tvm.relay.testing.init import create_workload


M = 1
K = 16384
N = 8192
dtype = "float16"

log_path = os.path.join("out", "tune.log")
tune = False
target_host = "llvm -mtriple=arm64-linux-android"
target_host = "llvm -mtriple=arm64-apple-darwin23.1.0 -mcpu=apple-m2"
target = "metal"
build_func = "ndk"
build_func = "default"
fcompile = ndk.create_shared
fcompile = None

target = tvm.target.Target(target, target_host)
x_var = relay.var("x", shape=(M, K), dtype=dtype)
m = relay.var("fc_weight")
y = relay.nn.dense(x_var, m, units=N)

func = relay.Function(relay.analysis.free_vars(y), y)
mod = tvm.IRModule.from_expr(func)
params = {"fc_weight": tvm.nd.array(np.random.rand(N, K).astype(dtype))}

remote_kwargs = None
rpc_remote_kwargs = None
remote_kwargs = {
    "key": "local",
    "host": os.environ["TVM_TRACKER_HOST"],
    "port": int(os.environ["TVM_TRACKER_PORT"]),
    "timeout": 600,
}
rpc_remote_kwargs = {
    "device_key" if k == "key" else k: v
    for k, v in remote_kwargs.items()
}
remote_kwargs = None
rpc_remote_kwargs = None
if tune:
    tasks = autotvm.task.extract_from_program(
        mod["main"],
        target=target,
        params=params,
        ops=(relay.op.get("nn.dense"),),
    )
    n_trial = 1000
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type="reg")
        if remote_kwargs is not None:
            measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func=build_func),
                runner=autotvm.RPCRunner(**remote_kwargs),
            )
        else:
            measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(),
            )
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=n_trial,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_path),
            ],
        )
    ctx = autotvm.apply_history_best(log_path)
else:
    ctx = autotvm.FallbackContext()

with ctx:
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

if remote_kwargs is not None:
    remote = request_remote(**rpc_remote_kwargs)
    relpath = f"lib_{target.kind.name}.so"
    temp = utils.tempdir()
    path_dso = temp.relpath(relpath)

    dev = remote.device(str(target), 0)

    lib.export_library(path_dso, fcompile=fcompile)
    remote.upload(path_dso)
    rlib = remote.load_module(relpath)
    # create module
    func = graph_executor.GraphModule(rlib["default"](dev))
else:
    dev = tvm.device(str(target))
    func = graph_executor.GraphModule(lib["default"](dev))
# set input and parameters
x = tvm.nd.array(np.random.rand(M, K).astype(dtype), dev)
func.set_input("x", x)
# run
print(func.benchmark(dev, number=100).mean)
