# Prints the SciPy values that test/suite/sparse_csc_2_tests.goose pins for the sparse matrices of
# compressed columns. The output is Goose source.
#   python3 test/gen_sparse_csc.py > pins.goose
import numpy as np
import scipy.sparse as sp

def num(x):
    x = float(x)
    s = np.format_float_positional(x, unique=True, trim='0')
    if s.endswith('.'):
        s += '0'
    if '.' not in s:
        s += '.0'
    return s

def lst(a):
    return "[" + ", ".join(num(x) for x in np.asarray(a, float).reshape(-1)) + "]"

def ilst(a):
    return "[" + ", ".join(str(int(x)) for x in a) + "]"

rng = np.random.default_rng(310)
count = [0]
funcs = []
body = []

def flush():
    global body
    if body:
        funcs.append(f"fn test_sparse_csc_pins{len(funcs)}() {{\n" + "\n".join(body) + "\n}\n")
        body = []

def dense(x):
    return "[" + ", ".join(num(v) for v in np.asarray(x, float).reshape(-1)) + "]"

def check(k, name, want):
    want = np.asarray(want, float)
    body.append(f"    assert({name}{k}.shape == {ilst(want.shape)} && ms_same({name}{k}, {lst(want)}));")

def emit_dense(a, want):
    k = count[0]; count[0] += 1
    body.append(f"    let c{k}, g{k} = scientific::csc_from_dense(ni_nd({ilst(a.shape)}, {dense(a)}));")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE);")
    body.append(f"    let d{k}, h{k} = scientific::csc_to_dense(c{k});")
    body.append(f"    assert(h{k}.kind == scientific::F_NONE);")
    check(k, "d", want)
    if len(body) >= 40: flush()

def pick():
    rows = int(rng.integers(1, 7))
    cols = int(rng.integers(1, 7))
    a = np.round(rng.random((rows, cols)) * 8 - 4, 3)
    a[rng.random((rows, cols)) < 0.6] = 0.0
    return a

for t in range(24):
    a = pick()
    emit_dense(a, a)
    # The transpose of a dense matrix.
    k = count[0]; count[0] += 1
    body.append(f"    let t{k}, g{k} = scientific::csc_from_dense(ni_nd({ilst(a.shape)}, {dense(a)}));")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE);")
    body.append(f"    let u{k}, h{k} = scientific::csc_transpose(t{k});")
    body.append(f"    assert(h{k}.kind == scientific::F_NONE);")
    body.append(f"    let v{k}, i{k} = scientific::csc_to_dense(u{k});")
    body.append(f"    assert(i{k}.kind == scientific::F_NONE);")
    check(k, "v", a.T)
    # The product of the matrix and a vector.
    x = np.round(rng.random(a.shape[1]) * 4 - 2, 3)
    body.append(f"    let w{k}, j{k} = scientific::csc_matvec(t{k}, {lst(x)});")
    body.append(f"    assert(j{k}.kind == scientific::F_NONE);")
    check(k, "w", a @ x)
    # The product of two matrices.
    b = np.round(rng.random((a.shape[1], 3)) * 8 - 4, 3)
    b[rng.random(b.shape) < 0.6] = 0.0
    c = a @ b
    body.append(f"    let m{k}, n{k} = scientific::csc_matmul(t{k}, scientific::csc_from_dense(ni_nd({ilst(b.shape)}, {dense(b)})));")
    body.append(f"    assert(n{k}.kind == scientific::F_NONE);")
    body.append(f"    let o{k}, p{k} = scientific::csc_to_dense(m{k});")
    body.append(f"    assert(p{k}.kind == scientific::F_NONE);")
    check(k, "o", c)
    # The sum and the difference of two matrices of one shape.
    b2 = np.round(rng.random(a.shape) * 8 - 4, 3)
    b2[rng.random(b2.shape) < 0.6] = 0.0
    body.append(f"    let q{k}, r{k} = scientific::csc_add(t{k}, scientific::csc_from_dense(ni_nd({ilst(b2.shape)}, {dense(b2)})));")
    body.append(f"    assert(r{k}.kind == scientific::F_NONE);")
    body.append(f"    let s{k}, t2{k} = scientific::csc_to_dense(q{k});")
    body.append(f"    assert(t2{k}.kind == scientific::F_NONE);")
    check(k, "s", a + b2)
    body.append(f"    let u2{k}, v2{k} = scientific::csc_sub(t{k}, scientific::csc_from_dense(ni_nd({ilst(b2.shape)}, {dense(b2)})));")
    body.append(f"    assert(v2{k}.kind == scientific::F_NONE);")
    body.append(f"    let w2{k}, x2{k} = scientific::csc_to_dense(u2{k});")
    body.append(f"    assert(x2{k}.kind == scientific::F_NONE);")
    check(k, "w2", a - b2)
    # The diagonal of the matrix.
    body.append(f"    let y{k}, z{k} = scientific::csc_diagonal(t{k});")
    body.append(f"    assert(z{k}.kind == scientific::F_NONE);")
    check(k, "y", np.diagonal(a))
    # The sum of every column and of every row.
    body.append(f"    let aa{k}, ab{k} = scientific::csc_sum(t{k}, 0);")
    body.append(f"    assert(ab{k}.kind == scientific::F_NONE);")
    check(k, "aa", a.sum(axis=0))
    body.append(f"    let ac{k}, ad{k} = scientific::csc_sum(t{k}, 1);")
    body.append(f"    assert(ad{k}.kind == scientific::F_NONE);")
    check(k, "ac", a.sum(axis=1))
    if len(body) >= 40: flush()

# The identity and the diagonals.
for n in [1, 4, 6]:
    k = count[0]; count[0] += 1
    body.append(f"    let e{k}, g{k} = scientific::csc_eye({n});")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE);")
    body.append(f"    let d{k}, h{k} = scientific::csc_to_dense(e{k});")
    body.append(f"    assert(h{k}.kind == scientific::F_NONE);")
    check(k, "d", np.eye(n))
for off in [-2, -1, 0, 1, 2]:
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rows, cols = 5, 5
    want = sp.diags(vals, off, shape=(rows, cols), format="csc").toarray()
    k = count[0]; count[0] += 1
    body.append(f"    let f{k}, g{k} = scientific::csc_diags({lst(vals)}, {off}, {rows}, {cols});")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE);")
    body.append(f"    let d{k}, h{k} = scientific::csc_to_dense(f{k});")
    body.append(f"    assert(h{k}.kind == scientific::F_NONE);")
    check(k, "d", want)
    if len(body) >= 40: flush()

# The entries of one place, and the duplicate entries of one place.
k = count[0]; count[0] += 1
body.append(f"    let g{k}, h{k} = scientific::csc_get(scientific::csc_eye(3), 1, 1);")
body.append(f"    assert(h{k}.kind == scientific::F_NONE && g{k} == 1.0);")
k = count[0]; count[0] += 1
body.append(f"    let i{k}, j{k} = scientific::csc_get(scientific::csc_eye(3), 0, 2);")
body.append(f"    assert(j{k}.kind == scientific::F_NONE && i{k} == 0.0);")
k = count[0]; count[0] += 1
body.append(f"    let l{k}, m{k} = scientific::csc_from_triplets(2, 2, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]);")
body.append(f"    assert(m{k}.kind == scientific::F_NONE);")
body.append(f"    let n{k}, o{k} = scientific::csc_to_dense(l{k});")
body.append(f"    assert(o{k}.kind == scientific::F_NONE);")
check(k, "n", np.array([[3.0, 0.0], [0.0, 3.0]]))

flush()
for f in funcs:
    print(f)
print("fn test_sparse_csc_pins() {")
for i in range(len(funcs)):
    print(f"    test_sparse_csc_pins{i}();")
print("}")
