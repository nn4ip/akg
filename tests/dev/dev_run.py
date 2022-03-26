from akg.ops.math.add import Add
from akg.utils import kernel_exec as utils

def test_ms_add(shape1, shape2, dtype):
    mod = utils.op_build_test(Add, (shape1, shape2), (dtype, dtype), kernel_name="add", attrs={"target": "cuda"})    

if __name__ == '__main__':
    test_ms_add((1, 1024, 512), (1, 1024, 512), 'float32')