import tvm
from tvm import te
import os
from tvm.contrib import utils, clang
from typing import Tuple, Optional
from .utils import _create_llvm


def lut_ctor(
    k: int,
    g: int,
    act_group_size: int,
    bits: int,
    dtype: str,
    cc: Optional[str] = None,
    cc_opts: Optional[list] = None,
    out_dtype = "float16",
    fast_aggregation_k: int = 16,
) -> Tuple[tvm.tir.TensorIntrin, str, str, str]:

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
                f"lut_ctor_g4_{dtype}_k{fast_aggregation_k}_b{bits}",
                k,
                qlut_buffer.access_ptr("w"),
                b_buffer.access_ptr("r"),
                lut_scales_buffer.access_ptr("w"),
                lut_biases_buffer.access_ptr("w"),
            )
        )
        return ib.get()

    body_code = f"lut_ctor({fast_aggregation_k}, {bits})"
    ll_code, header_code, body_code = _create_llvm("lut_ctor.cc", body_code, cc, cc_opts)

    buffer_params = {"offset_factor": 1}
    binds = {B: b_buffer, LUT_Scales: lut_scales_buffer, LUT_Biases: lut_biases_buffer, QLUT: qlut_buffer}
    return (
        te.decl_tensor_intrin(
            QLUT.op,
            _intrin_func,
            binds=binds,
            default_buffer_params=buffer_params,
        ),
        ll_code,
        header_code,
        body_code,
    )


def partial_max(
    g: int,
    dtype: str,
    k: int = 32,
    cc: Optional[str] = None,
    cc_opts: Optional[list] = None,
    out_dtype = "float16",
) -> Tuple[tvm.tir.TensorIntrin, str, str, str]:

    if dtype == "int8":
        maxv = 127

    B = te.placeholder((k,), out_dtype, name="B")
    sk = te.reduce_axis((0, k // g), "k")

    LUT_Scales = te.compute(
        (1,),
        lambda _: te.max(
            te.abs(sum(B[sk * g + ig] for ig in range(g))) / maxv,
            axis=sk,
        ),
        name="LUT_Scales",
    )

    b_buffer = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="b_buffer", offset_factor=1, strides=[1]
    )
    lut_scales_buffer = tvm.tir.decl_buffer(
        LUT_Scales.shape, LUT_Scales.dtype, name="lut_scales", offset_factor=1, strides=[1]
    )

    def _intrin_func(ins, outs):
        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    f"partial_max_g{g}_{dtype}_k{k // g}",
                    lut_scales_buffer.access_ptr("w"),
                    b_buffer.access_ptr("r"),
                )
            )
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    f"partial_max_reset",
                    lut_scales_buffer.access_ptr("w"),
                )
            )
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    ll_code, header_code, _ = _create_llvm("lut_ctor.cc", "", cc, cc_opts)

    buffer_params = {"offset_factor": 1}
    binds = {LUT_Scales: lut_scales_buffer, B: b_buffer}
    return (
        te.decl_tensor_intrin(
            LUT_Scales.op,
            _intrin_func,
            binds=binds,
            default_buffer_params=buffer_params,
        ),
        ll_code,
        header_code,
        "",
    )
