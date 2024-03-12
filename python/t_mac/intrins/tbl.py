import tvm
from tvm import te
import os
from tvm.contrib import utils, clang
from typing import Tuple, Optional


def tbl(
    m: int,
    kfactor: int,
    g: int,
    group_size: int,
    act_group_size: int,
    ngroups_per_elem: int,
    bits: int,
    dtype: str,
    cc: Optional[str] = None,
    cc_opts: Optional[list] = None,
    has_scale: bool = True,
    has_lut_scale: bool = False,
    out_dtype: str = "float16",
) -> Tuple[tvm.tir.TensorIntrin, str]:

    LUT = te.placeholder((kfactor, 2 ** g), dtype, name="LUT")
    A = te.placeholder((kfactor, m // ngroups_per_elem), "uint8", name="A")
    Scales = te.placeholder((max(1, kfactor * g // group_size), m // bits), out_dtype, name="Scales")

    if has_lut_scale:
        LUT_Scales = te.placeholder((max(1, kfactor * g // act_group_size),), dtype=out_dtype, name="LUT_Scales")
        LUT_Biases = te.placeholder((max(1, kfactor * g // act_group_size),), dtype=out_dtype, name="LUT_Biases")

    k = te.reduce_axis((0, kfactor), name="k")

    mask = te.const((1 << g) - 1, dtype="uint8")

    def _get_Abits(m, k):
        return (A[k, m // ngroups_per_elem] >> (g * (m % ngroups_per_elem))) & mask

    C = te.compute(
        (m,),
        lambda i: te.sum(
            LUT[k, _get_Abits(i, k)].astype(out_dtype) * (
                Scales[k * g // group_size, i // bits]
                    * LUT_Scales[k * g // act_group_size]
                    + LUT_Biases[k * g // act_group_size]
                if has_lut_scale
                else Scales[k * g // group_size, i // bits]
            ),
            axis=k,
        ),
        name="C",
    )

    lut_buffer = tvm.tir.decl_buffer(
        LUT.shape, LUT.dtype, name="lut_buffer", offset_factor=1, strides=[te.var("sl"), 1]
    )
    a_buffer = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="a_buffer", offset_factor=1, strides=[te.var("sa"), 1]
    )
    scales_buffer = tvm.tir.decl_buffer(
        Scales.shape, Scales.dtype, name="scales_buffer", offset_factor=1, strides=[te.var("ss"), 1]
    )
    if has_lut_scale:
        lut_scales_buffer = tvm.tir.decl_buffer(
            LUT_Scales.shape, LUT_Scales.dtype, name="lut_scales_buffer", offset_factor=1, strides=[1]
        )
        lut_biases_buffer = tvm.tir.decl_buffer(
            LUT_Biases.shape, LUT_Biases.dtype, name="lut_biases_buffer", offset_factor=1, strides=[1]
        )
    c_buffer = tvm.tir.decl_buffer(
        C.shape, C.dtype, name="c_buffer", offset_factor=1, strides=[1],
    )

    def to_intrinstr(t):
        if t == "float16" or t == "float32":
            return "float"
        else:
            return t

    def _intrin_func(ins, outs):
        def _body():
            ib = tvm.tir.ir_builder.create()
            args = [
                m,
                c_buffer.access_ptr("w"),
                lut_buffer.access_ptr("r"),
                a_buffer.access_ptr("r"),
                scales_buffer.access_ptr("r"),
            ]
            if has_lut_scale:
                args.append(lut_scales_buffer.access_ptr("r"))
                args.append(lut_biases_buffer.access_ptr("r"))
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    f"tbl_g{g}_{to_intrinstr(dtype)}_{to_intrinstr(out_dtype)}_update_{str(has_scale).lower()}_{kfactor}_{bits}",
                    *args,
                )
            )
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    f"tbl_{to_intrinstr(dtype)}_reset",
                    m,
                    c_buffer.access_ptr("w"),
                )
            )
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    with open(os.path.join(os.path.dirname(__file__), "tbl.cc"), "r") as fp:
        cc_code = fp.read()

    temp = utils.tempdir()
    ll_path = temp.relpath("tbl.ll")
    cc_opts = (cc_opts or []) + ["-I" + os.path.dirname(__file__)]
    ll_code = clang.create_llvm(
        cc_code,
        output=ll_path,
        options=cc_opts,
        cc=cc,
    )

    buffer_params = {"offset_factor": 1}
    binds = {LUT: lut_buffer, A: a_buffer, Scales: scales_buffer, C: c_buffer}
    if has_lut_scale:
        binds[LUT_Scales] = lut_scales_buffer
        binds[LUT_Biases] = lut_biases_buffer
    return (
        te.decl_tensor_intrin(
            C.op,
            _intrin_func,
            binds=binds,
            default_buffer_params=buffer_params,
        ),
        ll_code
    )
