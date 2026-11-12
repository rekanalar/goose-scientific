# Prints the SciPy values that test/run.goose pins for the biconjugate
# gradient solvers. The output is Goose source.
#   python3 test/gen_sparse_bicg.py > pins.goose
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

def num(x):
    s = np.format_float_positional(float(x), unique=True, trim='0')
    if s.endswith('.'):
        s += '0'
    if '.' not in s:
        s += '.0'
    return s

def ilst(a):
    return "[" + ", ".join(str(int(x)) for x in a) + "]"

def lst(a):
    return "[" + ", ".join(num(x) for x in np.asarray(a, float).reshape(-1)) + "]"

rng = np.random.default_rng(314)
funcs = []
body = []
count = [0]

def flush():
    global body
    if body:
        funcs.append(f"fn test_sparse_bicg_pins{len(funcs)}() {{\n" + "\n".join(body) + "\n}\n")
        body = []

def case(a, b, rtol):
    """Solve the system with the four methods and pin the solutions."""
    m = sp.csc_array(a)
    k = count[0]; count[0] += 1
    body.append(f"    let qa{k}, qf{k} = scientific::csr_from_dense(ni_nd({ilst(a.shape)}, {lst(a)}));")
    body.append(f"    assert(qf{k}.kind == scientific::F_NONE);")
    for name, fn in [("bicg", sla.bicg), ("cgs", sla.cgs), ("qmr", sla.qmr), ("tfqmr", sla.tfqmr)]:
        x, info = fn(m, b, rtol=rtol)
        if info != 0:
            # The method did not reach the answer. Pin nothing for it.
            continue
        body.append(f"    let qx{k}{name}, qc{k}{name}, qg{k}{name} = scientific::{name}(qa{k}, {lst(b)}, {rtol!r});")
        body.append(f"    assert(qg{k}{name}.kind == scientific::F_NONE && qc{k}{name} == 0 && all_near(qx{k}{name}, {lst(x)}));")

for _ in range(14):
    n = int(rng.integers(1, 11))
    a = np.round(rng.random((n, n)) * 6 - 3, 3)
    a[rng.random((n, n)) < 0.35] = 0.0
    # A shift of the diagonal keeps the matrix away from singular for this seed.
    a = a + np.eye(n) * (1.0 + abs(np.trace(a)) / n)
    b = np.round(rng.random(n) * 4 - 2, 3)
    case(a, b, 1.0e-12)
    if len(body) >= 30:
        flush()

# The matrix of one row and the matrix of two rows.
case(np.array([[2.0]]), np.array([3.0]), 1.0e-12)
case(np.array([[4.0, 1.0], [1.0, 3.0]]), np.array([1.0, 2.0]), 1.0e-12)

flush()
for f in funcs:
    print(f)
print("fn test_sparse_bicg_pins() {")
for i in range(len(funcs)):
    print(f"    test_sparse_bicg_pins{i}();")
print("}")
