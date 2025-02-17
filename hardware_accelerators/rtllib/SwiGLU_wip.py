import pyrtl
from pyrtl import WireVector, Register, Const
from functools import reduce
from lmul import lmul_fast


class Swiglu:
    def __init__(
        self,
        x: pyrtl.WireVector,
        W: pyrtl.WireVector,
        V: pyrtl.WireVector,
        b: pyrtl.WireVector,
        c: pyrtl.WireVector,
        beta: int,
        e_bits: int,
        m_bits: int,
        layer: int = 0,
    ):

        self.x = x
        self.W = W
        self.V = V
        self.b = b
        self.c = c

        # Output wire for SwiGLU results (one per layer)
        self._result_out = pyrtl.WireVector(
            e_bits + m_bits + 1, f"swiGLU_result_l{layer}"
        )

        self.beta = beta
        self.e_bits = e_bits
        self.m_bits = m_bits

    def compute_swiglu(self):
        """
        $SwiGLU(x,W,V,b,c,β) = Swish_β(xW+b) ⊗ (xV+c)$
        """
        lin1 = (
            lmul_fast(self.x, self.W, self.e_bits, self.m_bits) + self.b
        )  # xW+b for the first half
        lin2 = (
            lmul_fast(self.x, self.V, self.e_bits, self.m_bits) + self.c
        )  # xV+c for the second half

        swish_output = self.swish(lin1)  # Swish_β(xW+b)

        result = lmul_fast(
            swish_output, lin2, self.e_bits, self.m_bits
        )  # Swish_β(xW+b) ⊗ (xV+c)
        self._result_out <<= result

    def swish(self, x):
        """
        $swish(x) = x * sigmoid(β * x)$
        """
        beta_x = lmul_fast(
            x,
            pyrtl.Const(self.beta, bitwidth=self.e_bits + self.m_bits),
            self.e_bits,
            self.m_bits,
        )
        sigmoid_val = self.sigmoid_vector(beta_x)

        return lmul_fast(x, sigmoid_val, self.e_bits, self.m_bits)

    def sigmoid(self, x):
        """
        See https://github.com/UCSBarchlab/OpenTPU/blob/master/activate.py (from Kai!)
        """
        rb = pyrtl.RomBlock(
            bitwidth=8,
            addrwidth=3,
            asynchronous=True,
            romdata={0: 128, 1: 187, 2: 225, 3: 243, 4: 251, 5: 254, 6: 255, 7: 255},
        )

        addr = pyrtl.select(
            reduce(lambda a, b: a | b, x[3:]),  # OR of bits 3 and above
            falsecase=x[:3],  # Use lower 3 bits for ROM address
            truecase=pyrtl.Const(7, bitwidth=3),  # Saturate at maximum address
        )
        return rb[addr]

    ### Original Functions
    # def sigmoid(self, x):
    #     rb = pyrtl.RomBlock(
    #         bitwidth=8, addrwidth=3, asynchronous=True,
    #         romdata={0: 128, 1: 187, 2: 225, 3: 243, 4: 251, 5: 254, 6: 255, 7: 255, 8: 255}
    #     )

    #     x_gt_7 = reduce(lambda x, y: x | y, x[3:])  # OR of bits 3 and up
    #     return pyrtl.select(x_gt_7, falsecase=rb[x[:3]], truecase=pyrtl.Const(255, bitwidth=8))

    # def sigmoid_vector(self, vec):
    #     return pyrtl.concat_list([self.sigmoid(x) for x in vec])

    @property
    def result(self):
        """Output wire for the SwiGLU computation."""
        return self._result_out
