from ..dtypes import *
from ..rtllib.lmul import lmul_fast, lmul_simple
from ..rtllib.multipliers import float_multiplier


NN_TEST_BATCH_SIZE = 64

NN_TEST_SYSTOLIC_ARRAY_SIZE = 8

NN_TEST_ACCUM_ADDR_WIDTH = 12

NN_TEST_MUL_FNS = [
    float_multiplier,
    lmul_simple,
    # lmul_fast,
]

NN_TEST_WA_DTYPES = [
    # (Float8, Float8),
    (Float8, BF16),
    (Float8, Float32),
    (BF16, BF16),
    # (BF16, Float32),
    # (Float32, Float32),
]
