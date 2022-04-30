from akg.ops.math.add import Add
from akg.ops.nn.gpu.conv import Conv
from akg.utils import kernel_exec as utils

from tests.common.test_run import conv_run


def test_ms_conv(shape1, shape2, dtype):
    mod = utils.op_build_test(Conv, (shape1, shape2), (dtype, dtype), kernel_name="Conv", attrs={"target": "cuda"})
    return mod

if __name__ == '__main__':
    args_default = [ # (batch, in_c, in_h, in_w), (out_c, in_c, k_h, k_w), stride, pad, dilation
        ("000_case", conv_run, ((32, 64, 56, 56), (64, 64, 3, 3), (1, 1), (1, 1, 1, 1), (1,1), "float32", "float32", "NCHW"), ["level0"]),
        ("001_case", conv_run, ((16, 4, 4, 16), (16, 3, 3, 16), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float16"), ["level0"]),
        ("002_case", conv_run, ((64, 6, 6, 64), (64, 3, 3, 64), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float32"), ["level0"]),
        ("003_case", conv_run, ((64, 6, 6, 64), (64, 3, 3, 64), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float16"), ["level0"]),
    ]
    
    attrs = {}
    attrs["target"] = utils.CUDA
    attrs["profiling"] = False
    attrs["repeat_times"] = 1000
    
    conv_run((32, 64, 56, 56), (64, 64, 3, 3), (1, 1), (1, 1, 1, 1), (1, 1), "float32", "float32", "NCHW", attrs=attrs)
    # conv_run((16, 4, 4, 16), (16, 3, 3, 16), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float16", "NHWC", attrs=attrs)
    # conv_run((16, 4, 300, 300), (16, 4, 3, 3), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float16", "NHWC", attrs=attrs)
    