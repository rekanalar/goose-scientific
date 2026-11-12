# Prints the SciPy values that test/run.goose pins for csr_kronsum. The output
# is Goose source.
#   python3 test/gen_sparse_kronsum.py > pins.goose
import numpy as np
import scipy.sparse as sp

def num(x):
    s = np.format_float_positional(float(x), unique=True, trim='0')
    if s.endswith('.'):
        s += '0'
    if '.' not in s:
        s += '.0'
    return s

def lst(a):
    return "[" + ", ".join(num(x) for x in np.asarray(a, float).reshape(-1)) + "]"

rng = np.random.default_rng(321)
funcs = []

def pin(a, b):
    n, m = a.shape[0], b.shape[0]
    k = sp.kronsum(sp.csr_array(a), sp.csr_array(b))
    side = n * m
    i = len(funcs)
    funcs.append(f"""fn test_sparse_kronsum_pins{i}() {{
    let a, fa = scientific::csr_from_dense(mx({n}, {n}, {lst(a)}));
    let b, fb = scientific::csr_from_dense(mx({m}, {m}, {lst(b)}));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == {side} && k.cols == {side} && k.nnz == {k.nnz});
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && all_near(kd, {lst(k.toarray())}));
    return;
}}
""")

pin(np.array([[1., 2.], [0., 3.]]), np.array([[0., 5.], [6., 0.]]))
pin(np.array([[1., 2.], [0., 3.]]), np.array([[0., 5., 1.], [6., 0., 0.], [0., 0., 2.]]))
pin(np.array([[2.]]), np.array([[3.]]))
pin(np.zeros((2, 2)), np.zeros((2, 2)))
pin(np.array([[1., 0.], [0., 2.]]), np.eye(2))
for n, m in [(1, 2), (3, 3), (4, 2), (5, 5), (2, 4), (3, 1), (1, 1), (4, 5)]:
    a = np.round(rng.random((n, n)), 3); a[a < 0.6] = 0
    b = np.round(rng.random((m, m)), 3); b[b < 0.6] = 0
    pin(a, b)

# A stored zero in the first operand: the sum keeps four values.
za = sp.csr_array((np.array([0.0, 2.0]), (np.array([0, 1]), np.array([0, 1]))), shape=(2, 2))
zk = sp.kronsum(za, sp.eye_array(2, format='csr'))
funcs.append(f"""fn test_sparse_kronsum_pins_zero() {{
    let a, fa = scientific::csr_from_triplets(2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 2.0]);
    let b, fb = scientific::csr_eye(2);
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE && a.nnz == 2);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.nnz == {zk.nnz});
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && all_near(kd, {lst(zk.toarray())}));
    return;
}}
""")

for f in funcs:
    print(f)
print("fn test_sparse_kronsum_pins() {")
for i in range(len(funcs) - 1):
    print(f"    test_sparse_kronsum_pins{i}();")
print("    test_sparse_kronsum_pins_zero();")
print("    return;\n}")
