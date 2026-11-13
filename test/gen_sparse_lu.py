# Prints the SciPy values that test/suite/sparse_lu_tests.goose pins for the sparse LU factor.
# The output is Goose source: the function test_sparse_lu_pins with one call
# for each pinned system. The helpers lu_pin and lu_pin_inverse live in
# test/suite/sparse_lu_tests.goose. The factors themselves differ from
# SuperLU, so the pins hold the solutions, and the test checks the identity
# L U = A[perm_r][:, perm_c] itself.
#   python3 test/gen_sparse_lu.py > pins.goose
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl


def num(x):
    return np.format_float_positional(float(x), unique=True, trim='0')


def lst(xs):
    return "[" + ", ".join(num(x) for x in np.asarray(xs, float).reshape(-1)) + "]"


calls = []


def triplets(a, extra=()):
    ri, ci = np.nonzero(a)
    vv = a[ri, ci]
    ri, ci, vv = list(ri), list(ci), list(vv)
    for (r, c) in extra:
        ri.append(r)
        ci.append(c)
        vv.append(0.0)
    return ri, ci, vv


def add(a, b, extra=()):
    a = np.array(a, float)
    n = a.shape[0]
    x = sl.splu(sp.csc_array(a)).solve(np.array(b, float))
    ri, ci, vv = triplets(a, extra)
    calls.append("    lu_pin(%d, %s, %s, %s, %s, %s);" % (
        n, lst(ri), lst(ci), lst(vv), lst(b), lst(x)))


def add_inverse(a):
    a = np.array(a, float)
    n = a.shape[0]
    inv = sl.splu(sp.csc_array(a)).solve(np.eye(n))
    ri, ci, vv = triplets(a)
    calls.append("    lu_pin_inverse(%d, %s, %s, %s, %s);" % (
        n, lst(ri), lst(ci), lst(vv), lst(inv)))


add([[4, 3], [6, 3]], [10, 12])
add([[0, 1, 0], [0, 0, 1], [1, 0, 0]], [1, 2, 3])
n = 5
t = np.diag([2.0] * n) + np.diag([-1.0] * (n - 1), 1) + np.diag([-1.0] * (n - 1), -1)
add(t, [1, 0, 0, 0, 1])
add([[0, 2, 0], [3, 0, 1], [0, 1, 4]], [2, 5, 6])
add_inverse([[4, 3], [6, 3]])
add([[5]], [10])
add(np.diag([2.0, -4.0, 0.5, 8.0]), [1, 2, 3, 4])
add([[2, 0, 0], [1, 3, 0], [0, 1, 4]], [2, 7, 9], extra=[(0, 2), (2, 0)])
add_inverse(np.diag([2.0, -4.0, 0.5]))

rng = np.random.default_rng(329)
for size in range(1, 13):
    for kind in range(2):
        while True:
            a = rng.normal(size=(size, size))
            a[rng.random((size, size)) < (0.55 if kind == 0 else 0.3)] = 0.0
            if abs(np.linalg.det(a)) > 1e-3 and np.linalg.cond(a) < 1e6:
                break
        add(np.round(a, 3), np.round(rng.normal(size=size), 3))
for size in (6, 9, 12):
    a = np.diag(rng.normal(size=size) + 3.0)
    for off in (1, -1, 2, -2):
        a += np.diag(rng.normal(size=size - abs(off)), off)
    add(np.round(a, 3), np.round(rng.normal(size=size), 3))
for size in (4, 7):
    a = np.zeros((size, size))
    for i, j in enumerate(rng.permutation(size)):
        a[i, j] = np.round(rng.normal() + 2.0, 3)
    add(a, np.round(rng.normal(size=size), 3))

print("fn test_sparse_lu_pins() {")
print("\n".join(calls))
print("}")
