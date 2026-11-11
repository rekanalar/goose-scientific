#!/usr/bin/env python3
"""Print the SciPy values that the cosine-sine tests pin.

The script needs SciPy 1.18.1. Run it from the directory of the
package. It prints the matrix, and for each split the angles and the
cosine-sine matrix. It also prints the first blocks of the factors for
the split 2 and 2, and the cosine-sine matrix with the sign swapped.
"""

import numpy as np
from scipy.linalg import cossin, qr


def fmt(values):
    return "[" + ", ".join(repr(float(v)) for v in np.asarray(values).ravel()) + "]"


def main():
    import scipy
    print("scipy", scipy.__version__)
    x, _ = qr(np.random.default_rng(0).standard_normal((4, 4)))
    print("x =", fmt(x))
    for p, q in [(2, 2), (1, 3), (3, 1), (2, 1), (1, 1)]:
        (u1, u2), theta, (v1h, v2h) = cossin(x, p=p, q=q, separate=True)
        _, cs, _ = cossin(x, p=p, q=q)
        print(f"split {p} {q}")
        print("  theta =", fmt(theta))
        print("  cs =", fmt(cs))
        if (p, q) == (2, 2):
            print("  u1 =", fmt(u1))
            print("  v1h =", fmt(v1h))
    _, cs_swap, _ = cossin(x, p=2, q=2, swap_sign=True)
    print("swap cs 2 2 =", fmt(cs_swap))
    # A stack of two matrices.
    y, _ = qr(np.random.default_rng(1).standard_normal((4, 4)))
    stack = np.stack([x, y])
    _, cs_stack, _ = cossin(stack, p=2, q=2)
    print("y =", fmt(y))
    print("stack cs 2 2 =", fmt(cs_stack))


main()
