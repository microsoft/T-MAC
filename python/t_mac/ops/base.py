import tvm
import tvm.testing
from tvm import te, autotvm, rpc, relay
from tvm.contrib import utils, ndk
from tvm.autotvm.measure.measure_methods import request_remote, default_module_loader
from tvm._ffi import get_global_func
import numpy as np

from typing import List, Optional, Union
import logging
import os
import pathlib
import gc

logger = logging.getLogger("ops")


class OpCodegen:

    def __init__(
            self,
            dtype: str,
            target: str,
            name: str,
            tune: bool = False,
            reuse_tuned: bool = False,
            verify: bool = True,
            save_dir: str = "",
            target_host: Optional[str] = None,
            remote_kwargs: Optional[dict] = None,
            cc_opts: Optional[list] = None) -> None:
        self.dtype = dtype
        self.name = name
        self.tune = tune
        self.reuse_tuned = reuse_tuned
        self.verify = verify
        self.save_path = os.path.join(save_dir, name)
        pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.target = tvm.target.Target(target, host=target_host)
        self.remote_kwargs = remote_kwargs.copy() if remote_kwargs is not None else None
        self.build_func = self.remote_kwargs.pop("build_func") if remote_kwargs is not None else None
        self.cc = os.environ.get("TVM_NDK_CC", None) if self.build_func == "ndk" else None
        self.cc_opts = ["-O3", "-march=armv8.2a+fp16"] if self.build_func == "ndk" else None
        self.cc_opts = cc_opts

    def _schedule(self, tensors: List[te.Tensor]):
        raise NotImplementedError

    def _compute(self, *args) -> List[te.Tensor]:
        raise NotImplementedError

    def _reference(self, *args) -> List[np.ndarray]:
        raise NotImplementedError

    def _define_config(self, cfg: Union[autotvm.ConfigSpace, autotvm.ConfigEntity]):
        for key in cfg:
            setattr(self, key, cfg[key].val)

    def template(self, template_name: Optional[str] = None):
        if template_name is None:
            template_name = self.name

        @autotvm.template(template_name)
        def _func(*args):
            cfg = autotvm.get_config()
            self._define_config(cfg)
            tensors = self._compute(*args)
            sch = self._schedule(tensors)
            return sch, tensors
        
        return _func

    def compile(
        self, 
        *args,
        template_name: Optional[str] = None,
        n_trial: int = 1000,
        num_threads: int = 4,
        thread_affinity: int = 1,
        return_lower: bool = False,
        **eval_kwargs,
    ):
        if template_name is None:
            template_name = self.name
        template = self.template(template_name)

        if self.tune:
            log_path = os.path.join(self.save_path, "tune.log")
            if not (self.reuse_tuned and os.path.exists(log_path)):
                task = autotvm.task.create(template_name, args=args, target=self.target)
                tuner = autotvm.tuner.GridSearchTuner(task)

                def _preload_function(remote: rpc.RPCSession, build_result: tvm.runtime.Module):
                    remote.get_function("runtime.config_threadpool")(thread_affinity, num_threads)

                if self.remote_kwargs is not None:
                    measure_option = autotvm.measure_option(
                        builder=autotvm.LocalBuilder(build_func=self.build_func),
                        runner=autotvm.RPCRunner(
                            module_loader=default_module_loader(_preload_function),
                            **self.remote_kwargs,
                            **eval_kwargs,
                        ),
                    )
                else:
                    measure_option = autotvm.measure_option(
                        builder=autotvm.LocalBuilder(),
                        runner=autotvm.LocalRunner(
                            module_loader=default_module_loader(),
                            **eval_kwargs,
                        ),
                    )
                n_trial = min(1000, len(task.config_space))
                prefix = f"[Task {template_name}] "
                tuner.tune(
                    n_trial=n_trial,
                    measure_option=measure_option,
                    callbacks=[
                        autotvm.callback.progress_bar(n_trial, prefix=prefix),
                        autotvm.callback.log_to_file(log_path),
                    ],
                )
            ctx = autotvm.apply_history_best(log_path)
        else:
            ctx = autotvm.FallbackContext()

        with ctx:
            with self.target:
                s, tensors = template(*args)
                logger.info(tvm.lower(s, tensors, simple_mode=True))
                if return_lower:
                    return tvm.lower(s, tensors, name=template_name)

                func = tvm.build(s, tensors, name=self.name)
                if self.target.kind.name == "llvm":
                    func.save(os.path.join(self.save_path, "src.S"), "s")
                elif self.target.kind.name == "c":
                    func.save(os.path.join(self.save_path, "src.c"), "c")
                func_syslib = tvm.build(
                    s,
                    tensors,
                    name=template_name,
                    runtime=relay.backend.Runtime("cpp", {"system-lib": True}),
                )
                func_syslib.save(os.path.join(self.save_path, f"kernels.o"))
                return func, self._reference(*args)

    def _verify(self, tvm_arrays: List[tvm.nd.NDArray], arrays: List[np.ndarray]):
        tvm.testing.assert_allclose(tvm_arrays[-1].numpy(), arrays[-1], rtol=1e-5)

    def evaluate(
        self,
        *args,
        num_threads: int = 4,
        thread_affinity: int = 1,
        template_name: Optional[str] = None,
        **eval_kwargs,
    ):
        func, arrays = self.compile(
            *args,
            template_name=template_name,
            num_threads=num_threads,
            thread_affinity=thread_affinity,
            **eval_kwargs,
        )
        assert func

        if self.remote_kwargs is not None:
            remote_kwargs = {
                "device_key" if k == "key" else k: v
                for k, v in self.remote_kwargs.items()
            }
            remote = request_remote(**remote_kwargs)
            relpath = f"lib_{self.target.kind.name}.so"
            temp_dir = utils.tempdir()
            path_dso = temp_dir.relpath(relpath)
            fcompile = ndk.create_shared if self.build_func == "ndk" else None
            func.export_library(path_dso, fcompile=fcompile)
            remote.upload(path_dso)
            func = remote.load_module(relpath)
            config_threadpool = remote.get_function("runtime.config_threadpool")
            get_num_threads = remote.get_function("runtime.NumThreads")

            if self.target.kind.name == "opencl":
                dev = remote.cl()
            elif self.target.kind.name == "vulkan":
                dev = remote.vulkan()
            else:
                dev = remote.cpu()
        else:
            config_threadpool = get_global_func("runtime.config_threadpool")
            get_num_threads = tvm.runtime.num_threads
            dev = tvm.device(self.target.kind.name)

        config_threadpool(thread_affinity, num_threads)
        logger.info(f"Threads: {get_num_threads()}")

        tvm_arrays = [tvm.nd.array(a, dev) for a in arrays]
        func(*tvm_arrays)
        if self.verify:
            self._verify(tvm_arrays, arrays)

        evaluator = func.time_evaluator(
            func.entry_name,
            dev,
            **eval_kwargs,
        )
        lat = evaluator(*tvm_arrays).min

        if self.remote_kwargs is not None:
            del remote
            gc.collect()

        return lat
