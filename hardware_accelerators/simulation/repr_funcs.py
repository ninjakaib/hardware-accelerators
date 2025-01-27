from ..dtypes import BF16

# TODO: make parametrizeable by dtype, add float8 reprs

M_BITS = 7


def repr_bf16(x):
    return f"{BF16(binint=x).decimal_approx:.4f}"


def repr_bf16_binary(x):
    return f"{BF16(binint=x)._format_binary_string()}"


def repr_sign(x):
    return "0 (+)" if x == 0 else "1 (-)"


def repr_exp(x):
    return f"{format(x, '08b')} (unbiased={x - 127})"


def repr_mantissa(x):
    binary = format(x, "08b")
    decimal = int(binary[0], 2) + int(binary[1:], 2) / (2**7)
    return f"{binary[:1]}.{binary[1:]} ({decimal:.3f})"


def repr_mantissa_hidden(x):
    binary = format(x, "07b")
    decimal = 1 + int(binary, 2) / (2**7)
    return f"1.{binary} ({decimal:.3f})"


def repr_ext_mantissa(x):
    binary = format(x, "016b")
    decimal = int(binary[0], 2) + int(binary[1:], 2) / (2**15)
    return f"{binary[:1]}.{binary[1:]} ({decimal:.3f})"


def repr_mantissa_sum(x):
    binary = format(x, "09b")
    decimal = int(binary[:2], 2) + int(binary[2:], 2) / (2**7)
    return f"{binary[:2]}.{binary[2:]} ({decimal:.3f})"


def repr_mantissa_product(x):
    # represents a 16 bit result of multiplying two fixed point numbers 2 bits left of binary point 7 bits to the right
    binary = format(x, f"0{(M_BITS+1)*2}b")
    whole = int(binary[:2], 2)
    frac = int(binary[2:], 2) / 2 ** (M_BITS * 2)
    return f"{whole+frac}"


def repr_num(x):
    return format(x, "0b") + f" ({x})"
