import tvm
import tvm.testing
from tvm import te, autotvm, rpc, relay
from tvm.contrib import utils, ndk
from tvm.autotvm.measure.measure_methods import request_remote, default_module_loader
from tvm._ffi import get_global_func
import numpy as np

from typing import List, Optional, Union, Literal
import logging
import os
import pathlib
import gc
import re

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
            cc: Optional[str] = None,
            cc_opts: Optional[list] = None,
            num_threads: int = 4,
    ) -> None:
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
        self.cc = cc
        self.cc_opts = cc_opts
        self.num_threads = num_threads
        self.extra_cc_header = ""
        self.extra_cc_body = ""
        self._vectorization = True

    def _schedule(self, tensors: List[te.Tensor]):
        raise NotImplementedError

    def _compute(self, *args) -> List[te.Tensor]:
        raise NotImplementedError

    def _reference(self, *args) -> List[np.ndarray]:
        raise NotImplementedError

    def _define_config(self, cfg: Union[autotvm.ConfigSpace, autotvm.ConfigEntity], *args):
        for key in cfg:
            setattr(self, key, cfg[key].val)

    def template(self, template_name: str):
        @autotvm.template(template_name)
        def _func(*args):
            cfg = autotvm.get_config()
            self._define_config(cfg, *args)
            tensors = self._compute(*args)
            sch = self._schedule(tensors)
            return sch, tensors

        return _func

    def get_template_name(self, *args) -> str:
        return self.name + f"_t{self.num_threads}_{self.dtype}"

    @property
    def log_path(self):
        return os.path.join(self.save_path, "tune.log")

    def tuning(
        self,
        *args,
        n_trial: int = 1000,
        thread_affinity: int = 1,
        **eval_kwargs,
    ):
        template_name = self.get_template_name(*args)
        log_path = self.log_path

        if not (self.reuse_tuned and os.path.exists(log_path)):
            task = autotvm.task.create(template_name, args=args, target=self.target)
            tuner = autotvm.tuner.GridSearchTuner(task)

            def _preload_function(remote: rpc.RPCSession, build_result: tvm.runtime.Module):
                remote.get_function("runtime.config_threadpool")(thread_affinity, self.num_threads)

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

    def _postprocess_tvm_c_code(self, c_code: str, template_name: str):
        to_removes = [
            r'#define TVM_EXPORTS',
            r'#include "tvm/runtime/c_runtime_api.h"',
            r'#include "tvm/runtime/c_backend_api.h"',
            r'TVM_DLL',
        ]
        for s in to_removes:
            c_code = c_code.replace(s, "")

        # Remove unrelated function defines
        # E.g., intrinsics
        func_dcl_ptn = """#ifdef __cplusplus
extern "C"
#endif
 int32_t (\w+)\(.+\);"""
        for m in re.finditer(func_dcl_ptn, c_code, re.MULTILINE):
            if m[1] != template_name:
                c_code = c_code.replace(m[0], "")

        # Remove __tvm_main__
        tvm_main_def_ptn = """#ifdef __cplusplus
extern "C"
#endif
 int32_t __tvm_main__\(void\* args, int\* arg_type_ids, int num_args, void\* out_ret_value, int\* out_ret_tcode, void\* resource_handle\) {
.+
}
"""
        c_code = re.sub(tvm_main_def_ptn, "", c_code, flags=re.MULTILINE)

        # Modify kernel args
        kernel_dcl_ptn = """#ifdef __cplusplus
extern "C"
#endif
 (int32_t """ + template_name + """\(void\* args, int32_t\* arg_type_ids, int32_t num_args, void\* out_ret_value, int32_t\* out_ret_tcode, void\* resource_handle\)) {
([\s\S]+)
}"""
        kernel_m = re.search(kernel_dcl_ptn, c_code, re.MULTILINE)
        if not kernel_m:
            raise RuntimeError("can't find kernel declaration")

        kernel_body = kernel_m[2]
        # Modify args retrieve
        args_def_ptn = """(void\* \w+) = \(\(\(TVMValue\*\)args\)\[(\d+)\]\.v_handle\);"""
        args = []
        for m in re.finditer(args_def_ptn, kernel_body):
            if int(m[2]) != len(args):
                raise RuntimeError("Error parsing line: {}".format(m[0]))
            args.append(m[1])
            kernel_body = kernel_body.replace(m[0], "")
        args_code_ptn = """int32_t \w+_code = arg_type_ids\[\d+\];"""
        kernel_body = re.sub(args_code_ptn, "", kernel_body)

        # Replace .shape .strides with NULL
        shape_or_strides_ptr_ptn = """\(\(DLTensor\*\)\w+\)\[0\]\.(?:shape|strides)"""
        kernel_body = re.sub(shape_or_strides_ptr_ptn, "NULL", kernel_body)

        # Replace .data with args
        data_ptr_ptn = """\(\(DLTensor\*\)(\w+)\)\[0\]\.data"""
        kernel_body = re.sub(data_ptr_ptn, r"\1", kernel_body)

        # Replace .device_id with 0
        dev_id_ptn = """\(\(DLTensor\*\)\w+\)\[0\]\.device\.device_id"""
        kernel_body = re.sub(dev_id_ptn, "0", kernel_body)

        # Remove TVMBackendFreeWorkspace
        free_ptn = """TVMBackendFreeWorkspace\(1, dev_id, \w+\)"""
        kernel_body = re.sub(free_ptn, "0", kernel_body)

        # Replace TVMBackendAllocWorkspace with stack array
        alloc_ptn = """void\* (\w+) = TVMBackendAllocWorkspace\(1, dev_id, \(uint64_t\)(\d+), \d+, \d+\)"""
        for m in re.finditer(alloc_ptn, kernel_body):
            stm, var_name, alloc_size = m[0], m[1], m[2]
            new_stm = f"uint64_t temp_{var_name}[{(int(alloc_size) + 7) // 8}]; void* {var_name} = (void*)temp_{var_name}"
            kernel_body = kernel_body.replace(stm, new_stm)

        # Add alignas specifier to all stack array
        stack_array_ptn = r"(\w+ \w+\[\d+\];)"
        # 32 = 256 / 8
        kernel_body = re.sub(stack_array_ptn, r"alignas(32) \1", kernel_body)

        kernel_args = "int32_t {}({})".format(template_name, ", ".join(args))
        c_code = c_code.replace(kernel_m[1], kernel_args)
        c_code = c_code.replace(kernel_m[2], kernel_body)

        # Move kernel def to header
        kernel_def = """#ifdef __cplusplus
extern "C"
#endif
 """ + kernel_args + ";"
        c_code = c_code.replace(kernel_def, "")
        c_header = kernel_def

        # Add half vectorization related
        half_typedef = """
#ifndef TMAC_HALF_TYPEDEF_H
#define TMAC_HALF_TYPEDEF_H

#ifndef __AVX2__
typedef _Float16 half;
#endif
#endif
"""
        c_code = half_typedef + c_code

        return c_header, c_code

    def compile(
        self,
        *args,
        n_trial: int = 1000,
        thread_affinity: int = 1,
        return_type: Literal["mod", "lower", "c"] = "mod",
        preserve_cfg: bool = False,
        **eval_kwargs,
    ):
        template_name = self.get_template_name(*args)

        log_path = self.log_path

        self._vectorization = (return_type != "c")

        with self.target:
            if not preserve_cfg:
                template = self.template(template_name)
                if self.tune:
                    self.tuning(*args, n_trial=n_trial, thread_affinity=thread_affinity, **eval_kwargs)
                    ctx = autotvm.apply_history_best(log_path)
                else:
                    ctx = autotvm.FallbackContext()

                with ctx:
                    template(*args)

            tensors = self._compute(*args)
            s = self._schedule(tensors)

            logger.info(tvm.lower(s, tensors, simple_mode=True))

            func = tvm.build(s, tensors, name=template_name)
            if self.target.kind.name == "llvm":
                func.save(os.path.join(self.save_path, "src.S"), "s")
                func.save(os.path.join(self.save_path, "src.ll"), "ll")
            elif self.target.kind.name == "c":
                func.save(os.path.join(self.save_path, "src.c"), "c")

            if return_type == "c":
                func_c = tvm.build(s, tensors, target="c", name=template_name)
                return self._postprocess_tvm_c_code(func_c.get_source(), template_name)

            if return_type == "lower":
                return tvm.lower(s, tensors, name=template_name)

            func_syslib = tvm.build(
                s,
                tensors,
                name=template_name,
                runtime=relay.backend.Runtime("cpp", {"system-lib": True}),
            )
            func_syslib.save(os.path.join(self.save_path, f"kernels.o"))

            if self.verify:
                arrays = self._reference(*args)
            else:
                arrays = [np.zeros(shape=[int(s) for s in list(t.shape)], dtype=t.dtype) for t in tensors]
            return func, arrays

    def _verify(self, tvm_arrays: List[tvm.nd.NDArray], arrays: List[np.ndarray]):
        tvm.testing.assert_allclose(tvm_arrays[-1].numpy(), arrays[-1], rtol=1e-5)

    def evaluate(
        self,
        *args,
        thread_affinity: int = 1,
        **eval_kwargs,
    ):
        func, arrays = self.compile(
            *args,
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

        config_threadpool(thread_affinity, self.num_threads)
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
