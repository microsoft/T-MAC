import tvm
from tvm import te
import os
from tvm.contrib import utils, clang
from typing import Tuple, Optional


def lut_ctor(
    k: int,
    g: int,
    act_group_size: int,
    dtype: str,
    cc: Optional[str] = None,
    cc_opts: Optional[list] = None,
    out_dtype = "float16",
) -> Tuple[tvm.tir.TensorIntrin, str]:
    
    B = te.placeholder((k,), out_dtype, name="B")
    LUT_Scales = te.placeholder((k // act_group_size,), out_dtype, name="LUT_Scales")
    LUT_Biases = te.placeholder((k // act_group_size,), out_dtype, name="LUT_Biases")

    QLUT = te.compute(
        (k // g, 1 << g),
        lambda i, j: (
            B[i * g + (j % g)] / LUT_Scales[i * g // act_group_size] - LUT_Biases[i * g // act_group_size]
        ).astype(dtype),
        name="QLUT",
    )

    b_buffer = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="b_buffer", offset_factor=1, strides=[1]
    )
    lut_scales_buffer = tvm.tir.decl_buffer(
        LUT_Scales.shape, LUT_Scales.dtype, name="lut_scales", offset_factor=1, strides=[1]
    )
    lut_biases_buffer = tvm.tir.decl_buffer(
        LUT_Biases.shape, LUT_Biases.dtype, name="lut_biases", offset_factor=1, strides=[1]
    )
    qlut_buffer = tvm.tir.decl_buffer(
        QLUT.shape, QLUT.dtype, name="qlut_buffer", offset_factor=1, strides=[te.var("sc"), 1]
    )

    def _intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                f"lut_ctor_g4_{dtype}_{k // g}",
                qlut_buffer.access_ptr("w"),
                b_buffer.access_ptr("r"),
                lut_scales_buffer.access_ptr("w"),
                lut_biases_buffer.access_ptr("w"),
            )
        )
        return ib.get()

    with open(os.path.join(os.path.dirname(__file__), "lut_ctor.cc"), "r") as fp:
        cc_code = fp.read()

    temp = utils.tempdir()
    ll_path = temp.relpath("lut_ctor.ll")
    cc_opts = (cc_opts or []) + ["-I" + os.path.dirname(__file__)]
    ll_code = clang.create_llvm(
        cc_code,
        output=ll_path,
        options=cc_opts,
        cc=cc,
    )

    buffer_params = {"offset_factor": 1}
    binds = {B: b_buffer, LUT_Scales: lut_scales_buffer, LUT_Biases: lut_biases_buffer, QLUT: qlut_buffer}
    return (
        te.decl_tensor_intrin(
            QLUT.op,
            _intrin_func,
            binds=binds,
            default_buffer_params=buffer_params,
        ),
        ll_code
    )
