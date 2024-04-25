import os
from tvm.contrib import utils, clang

def _extern_cpp(cc_code):
    return """
#ifdef __cplusplus
extern "C" {
#endif
""" + cc_code + """
#ifdef __cplusplus
}
#endif
"""

def _create_llvm(header_file, body_code, cc, cc_opts):
    with open(os.path.join(os.path.dirname(__file__), header_file), "r") as fp:
        header_code = fp.read()
        body_code_cpp = _extern_cpp(body_code)
        cc_code = header_code + body_code_cpp

    temp = utils.tempdir()
    ll_path = temp.relpath("src.ll")
    ll_code = clang.create_llvm(
        cc_code,
        output=ll_path,
        options=cc_opts,
        cc=cc,
    )

    return ll_code, header_code, "\n{}\n".format(body_code)
