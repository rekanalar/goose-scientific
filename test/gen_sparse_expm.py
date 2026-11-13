# Prints the SciPy values that test/suite/sparse_expm_tests.goose pins for the sparse one-norm
# estimate and the matrix exponential. The output is Goose source: the
# function test_sparse_expm_pins with one call for each pinned system. The
# helpers expm_est_pin, expm_est_vec_pin, expm_pin and expm_range_pin live
# in test/suite/sparse_expm_tests.goose.
#
# SciPy draws the columns of the block at random, so its own estimate can
# differ between two runs. The estimate of the package uses a fixed
# generator. The pins therefore hold the exact one-norm of the matrix and
# not the value of SciPy.
#
#   tmp/venv/bin/python -I test/gen_sparse_expm.py > pins.goose
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl


def num(x):
    return np.format_float_positional(float(x), unique=True, trim='0')


def lst(xs):
    return "[" + ", ".join(num(x) for x in np.asarray(xs, float).reshape(-1)) + "]"


calls = []


def triplets(a):
    ri, ci = np.nonzero(a)
    return list(ri), list(ci), list(a[ri, ci])


def est_pin(a, exact):
    a = np.array(a, float)
    ri, ci, vv = triplets(a)
    calls.append("    expm_est_pin(%d, %s, %s, %s, %s);" % (
        a.shape[0], lst(ri), lst(ci), lst(vv), num(exact)))


def est_vec_pin(a, vec, prod):
    a = np.array(a, float)
    ri, ci, vv = triplets(a)
    calls.append("    expm_est_vec_pin(%d, %s, %s, %s, %s, %s);" % (
        a.shape[0], lst(ri), lst(ci), lst(vv), lst(vec), lst(prod)))


def expm_pin(a, b, x):
    a = np.array(a, float)
    ri, ci, vv = triplets(a)
    calls.append("    expm_pin(%d, %s, %s, %s, %s, %s);" % (
        a.shape[0], lst(ri), lst(ci), lst(vv), lst(b), lst(x)))


def num_(x):
    return num(x)


def range_pin(a, b, start, stop, num, endpoint):
    a = np.array(a, float)
    ri, ci, vv = triplets(a)
    times = np.linspace(start, stop, num, endpoint=endpoint)
    rows = np.array([sl.expm_multiply(a * t, np.array(b, float)) for t in times])
    calls.append("    expm_range_pin(%d, %s, %s, %s, %s, %s, %s, %s, %d, %s);" % (
        a.shape[0], lst(ri), lst(ci), lst(vv), lst(b), lst(rows),
        num_(start), num_(stop), num, "true" if endpoint else "false"))


m1 = np.array([[4.0, 3.0, 0.0], [6.0, 3.0, 1.0], [0.0, -2.0, 5.0]])
tri5 = np.diag([2.0] * 5) + np.diag([-1.0] * 4, 1) + np.diag([-1.0] * 4, -1)
perm4 = np.eye(4)[[1, 2, 3, 0]]
zero3 = np.zeros((3, 3))

# The estimates. The pin holds the exact one-norm of the matrix.
est_pin(m1, 10.0)
est_pin(np.diag([3.0, -8.0, 2.0]), 8.0)
est_pin(tri5, 4.0)
est_pin(perm4, 1.0)
est_pin(zero3, 0.0)

# The vectors of the estimate of the first matrix.
est_vec_pin(m1, [1.0, 0.0, 0.0], [4.0, 6.0, 0.0])

# The product of the exponential.
expm_pin(0.1 * m1, [1.0, 2.0, 3.0],
         sl.expm_multiply(0.1 * m1, np.array([1.0, 2.0, 3.0])))
expm_pin(0.1 * m1, [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
         sl.expm_multiply(0.1 * m1, np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])))
expm_pin(zero3, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

# The rows of the range form.
range_pin(m1, [1.0, 2.0, 3.0], 0.0, 1.0, 3, True)
range_pin(m1, [1.0, 2.0, 3.0], 0.0, 1.0, 3, False)

# The banded matrix of side 30.
n = 30
band = np.zeros((n, n))
for i in range(n):
    band[i, i] = 4.0
    if i + 1 < n:
        band[i, i + 1] = -1.0
    if i + 2 < n:
        band[i, i + 2] = 0.5
ones = np.ones(n)
expm_pin(band, ones, sl.expm_multiply(sp.csc_array(band), ones))

# A dense matrix of side 12 with a seeded generator.
rng = np.random.default_rng(330)
dense = np.round(rng.normal(size=(12, 12)), 3)
b12 = np.round(rng.normal(size=12), 3)
expm_pin(dense, b12, sl.expm_multiply(dense, b12))

# The estimates for the sides 1 to 12. The forms cover the diagonal, the
# banded, the permutation and the random sparse matrices.
rng = np.random.default_rng(3301)
for size in range(1, 13):
    diag = np.diag(rng.normal(size=size) + 3.0)
    est_pin(diag, float(np.abs(diag).sum(axis=0).max()))
    band = np.diag(np.full(size, 2.0))
    for off in (1, -1):
        if size > abs(off):
            band += np.diag(np.full(size - abs(off), -1.0), off)
    est_pin(band, float(np.abs(band).sum(axis=0).max()))
    perm = np.zeros((size, size))
    for i, j in enumerate(rng.permutation(size)):
        perm[i, j] = 1.0
    est_pin(perm, 1.0)
    sparse = np.round(rng.normal(size=(size, size)), 3)
    sparse[rng.random((size, size)) < 0.5] = 0.0
    est_pin(sparse, float(np.abs(sparse).sum(axis=0).max()))

print("fn test_sparse_expm_pins() {")
print("\n".join(calls))
print("}")
