# Prints the SciPy values that test/suite/sparse_lgmres_tests.goose pins for the sparse solvers
# lgmres and gcrotmk. The output is Goose source: the function
# test_sparse_lgmres_pins with one call for each pinned system. The helpers
# lg_pin and lg_pin_opts live in test/suite/sparse_lgmres_tests.goose.
#
# SciPy and the package may take different paths to the solution. The pins
# therefore hold the solution of SciPy at a tight tolerance, and the tests
# compare to 1e-8.
#
#   tmp/venv/bin/python -I test/gen_sparse_lgmres.py > pins.goose
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


def solve(a, b):
    """The solution of both SciPy calls at a tight tolerance."""
    s = sp.csr_array(a)
    x1, i1 = sl.lgmres(s, b, rtol=1e-12, atol=0.0, maxiter=1000)
    x2, i2 = sl.gcrotmk(s, b, rtol=1e-12, atol=0.0, maxiter=1000)
    assert i1 == 0 and i2 == 0
    exact = np.linalg.solve(a, b)
    assert np.allclose(x1, exact, rtol=1e-9, atol=1e-12)
    assert np.allclose(x2, exact, rtol=1e-9, atol=1e-12)
    return x1


def pin(kind, a, b):
    a = np.array(a, float)
    b = np.array(b, float)
    x = solve(a, b)
    ri, ci, vv = triplets(a)
    calls.append("    %s(%d, %s, %s, %s, %s, %s);" % (
        kind, a.shape[0], lst(ri), lst(ci), lst(vv), lst(b), lst(x)))


def tri(n):
    return np.diag([4.0] * n) + np.diag([-1.0] * (n - 1), 1) + np.diag([-1.0] * (n - 1), -1)


nons = np.array([[4, 1, 0, 2, 0], [1, 5, 2, 0, 0], [0, 3, 6, 1, 1],
                 [2, 0, 1, 7, 3], [0, 0, 1, 2, 8.0]])
perm4 = np.eye(4)[[1, 2, 3, 0]]

pin("lg_pin", tri(6), np.ones(6))
pin("lg_pin", tri(30), np.ones(30))
pin("lg_pin", nons, np.arange(1.0, 6.0))
pin("lg_pin", perm4, np.array([1.0, 2, 3, 4]))
pin("lg_pin", np.diag([1.0, 2, 3, 4]), np.ones(4))
pin("lg_pin", np.array([[4.0]]), np.array([2.0]))

# A nonsymmetric sparse matrix of side 20. A small inner size forces many
# restarts, so the kept vectors take part.
rng = np.random.default_rng(333)
n = 20
m = np.round(rng.normal(size=(n, n)), 3)
m[rng.random((n, n)) < 0.7] = 0.0
m += np.diag(np.full(n, 6.0))
b = np.round(rng.normal(size=n), 3)
pin("lg_pin_opts", m, b)

# A second system whose eigenvalues spread, so the restarts stall without the
# kept vectors.
t = tri(40) + np.diag(np.linspace(0.0, 3.0, 40))
t[0, 5] = 0.7
t[7, 2] = -0.4
pin("lg_pin_opts", t, np.ones(40))

print("fn test_sparse_lgmres_pins() {")
print("\n".join(calls))
print("}")
