import tvm
from tvm import te
import os
from tvm.contrib import utils, clang
from typing import Tuple, Optional
from ..utils import get_bits_alphas
from .utils import _create_llvm


def tbl(
    m: int,
    kfactor: int,
    g: int,
    act_group_size: int,
    ngroups_per_elem: int,
    bits: int,
    dtype: str,
    cc: Optional[str] = None,
    cc_opts: Optional[list] = None,
    has_scale: bool = True,
    has_lut_scale: bool = False,
    out_dtype: str = "float16",
    m_groups: int = -1,
    do_scale_final: bool = False,
    aggregation_dtype: str = "int32",
    fast_aggregation: bool = False,
    zero_point: bool = False,
) -> Tuple[tvm.tir.TensorIntrin, str, str, str]:
    """Create a table lookup intrinsics for a given table size and bitwidth,
    weights should be within the same group.
    """

    LUT = te.placeholder((kfactor, 2 ** g), dtype, name="LUT")
    lut_buffer = tvm.tir.decl_buffer(
        LUT.shape, LUT.dtype, name="lut_buffer", offset_factor=1, strides=[te.var("sl"), 1]
    )
    A = te.placeholder((kfactor, m // ngroups_per_elem), "uint8", name="A")
    a_buffer = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="a_buffer", offset_factor=1, strides=[te.var("sa"), 1]
    )

    if m_groups == -1:
        if zero_point:
            scales_shape = (kfactor * g // act_group_size, m // bits * 2)
            def _get_scale(m, k):
                return Scales[k * g // act_group_size, m // bits * 2] - Scales[k * g // act_group_size, m // bits * 2 + 1]
        else:
            scales_shape = (kfactor * g // act_group_size, m // bits)
            def _get_scale(m, k):
                return Scales[k * g // act_group_size, m // bits]
        scale_buffer_strides = [te.var("ss"), 1]
    else:
        scales_shape = (kfactor * g // act_group_size,)
        def _get_scale(m, k):
            return Scales[k * g // act_group_size]
        scale_buffer_strides = [1]

    alpha = te.const(get_bits_alphas(bits)[0], dtype=out_dtype)

    if not do_scale_final:
        Scales = te.placeholder(scales_shape, out_dtype, name="Scales")
        scales_buffer = tvm.tir.decl_buffer(
            Scales.shape, Scales.dtype, name="scales_buffer", offset_factor=1, strides=scale_buffer_strides
        )
        if has_lut_scale:
            LUT_Scales = te.placeholder((max(1, kfactor * g // act_group_size),), dtype=out_dtype, name="LUT_Scales")
            lut_scales_buffer = tvm.tir.decl_buffer(
                LUT_Scales.shape, LUT_Scales.dtype, name="lut_scales_buffer", offset_factor=1, strides=[1]
            )
            LUT_Biases = te.placeholder((max(1, kfactor * g // act_group_size),), dtype=out_dtype, name="LUT_Biases")
            lut_biases_buffer = tvm.tir.decl_buffer(
                LUT_Biases.shape, LUT_Biases.dtype, name="lut_biases_buffer", offset_factor=1, strides=[1]
            )
            def _lut_scale(k, val):
                return val * LUT_Scales[k * g // act_group_size] + LUT_Biases[k * g // act_group_size] * alpha
        else:
            def _lut_scale(k, val):
                return val

        def _scale_first(m, k, lut_val):
            return _lut_scale(k, lut_val.astype(out_dtype)) * _get_scale(m, k)
    else:
        def _scale_first(m, k, lut_val):
            return lut_val.astype(aggregation_dtype)

    k = te.reduce_axis((0, kfactor), name="k")

    mask = te.const((1 << g) - 1, dtype="uint8")

    def _get_Abits(m, k):
        return (A[k, m // ngroups_per_elem] >> (g * (m % ngroups_per_elem))) & mask

    C = te.compute(
        (m,),
        lambda i: te.sum(
            _scale_first(i, k, LUT[k, _get_Abits(i, k)]),
            axis=k,
        ),
        name="C",
    )

    c_buffer = tvm.tir.decl_buffer(
        C.shape, C.dtype, name="c_buffer", offset_factor=1, strides=[1],
    )

    def to_intrinstr(t):
        if t == "float16" or t == "float32":
            return "float"
        else:
            return t

    api_args = (
        g,
        to_intrinstr(dtype),
        to_intrinstr(out_dtype if not do_scale_final else aggregation_dtype),
        str(has_scale).lower(),
        kfactor,
        bits,
        min(act_group_size // 4, kfactor),
        str(fast_aggregation).lower(),
        str(zero_point).lower(),
        str(m_groups != -1).lower(),
    )

    def _intrin_func(ins, outs):
        def _body():
            ib = tvm.tir.ir_builder.create()
            args = [
                m,
                c_buffer.access_ptr("w"),
                lut_buffer.access_ptr("r"),
                a_buffer.access_ptr("r"),
            ]
            if not do_scale_final:
                args.append(scales_buffer.access_ptr("r"))
                if has_lut_scale:
                    args.append(lut_scales_buffer.access_ptr("r"))
                    args.append(lut_biases_buffer.access_ptr("r"))
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    "tbl_g{}_{}_{}_update_s{}_k{}_b{}_ak{}_fa{}_z{}_os{}".format(*api_args),
                    *args,
                )
            )
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    f"tbl_{to_intrinstr(out_dtype if not do_scale_final else aggregation_dtype)}_reset",
                    m,
                    c_buffer.access_ptr("w"),
                )
            )
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    body_code = "tbl_g{}_{}_{}_update({}, {}, {}, {}, {}, {}, {})".format(*api_args)
    ll_code, header_code, body_code = _create_llvm("tbl.cc", body_code, cc, cc_opts)

    buffer_params = {"offset_factor": 1}
    binds = {LUT: lut_buffer, A: a_buffer, C: c_buffer}
    if not do_scale_final:
        binds[Scales] = scales_buffer
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
        ll_code,
        header_code,
        body_code,
    )
