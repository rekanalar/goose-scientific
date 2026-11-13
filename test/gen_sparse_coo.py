# Prints the SciPy values that test/suite/sparse_coo_2_tests.goose pins for the sparse matrices of
# coordinates. The output is Goose source.
#   python3 test/gen_sparse_coo.py > pins.goose
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

def ilst(a):
    return "[" + ", ".join(str(int(x)) for x in a) + "]"

rng = np.random.default_rng(311)
funcs = []
body = []
count = [0]

def flush():
    global body
    if body:
        funcs.append(f"fn test_sparse_coo_pins{len(funcs)}() {{\n" + "\n".join(body) + "\n}\n")
        body = []

def check(k, name, want):
    want = np.asarray(want, float)
    body.append(f"    assert({name}{k}.shape == {ilst(want.shape)} && ms_same({name}{k}, {lst(want)}));")

for t in range(30):
    rows = int(rng.integers(1, 7)); cols = int(rng.integers(1, 7))
    n = int(rng.integers(0, 12))
    ri = rng.integers(0, rows, n); ci = rng.integers(0, cols, n)
    v = np.round(rng.random(n) * 8 - 4, 3)
    m = sp.coo_array((v, (ri, ci)), shape=(rows, cols))
    k = count[0]; count[0] += 1
    body.append(f"    let qc{k}, qg{k} = scientific::coo_from_triplets({rows}, {cols}, {lst(ri)}, {lst(ci)}, {lst(v)});")
    body.append(f"    assert(qg{k}.kind == scientific::F_NONE && qc{k}.nnz == {n});")
    body.append(f"    let qd{k}, qh{k} = scientific::coo_to_dense(qc{k});")
    body.append(f"    assert(qh{k}.kind == scientific::F_NONE);")
    check(k, "qd", m.toarray())
    body.append(f"    let qt{k}, qi{k} = scientific::coo_transpose(qc{k});")
    body.append(f"    assert(qi{k}.kind == scientific::F_NONE && qt{k}.nnz == {n});")
    body.append(f"    let qe{k}, qj{k} = scientific::coo_to_dense(qt{k});")
    body.append(f"    assert(qj{k}.kind == scientific::F_NONE);")
    check(k, "qe", m.toarray().T)
    x = np.round(rng.random(cols) * 4 - 2, 3)
    body.append(f"    let qw{k}, ql{k} = scientific::coo_matvec(qc{k}, {lst(x)});")
    body.append(f"    assert(ql{k}.kind == scientific::F_NONE);")
    check(k, "qw", m.toarray() @ x)
    body.append(f"    let qs{k} = scientific::coo_scale(qc{k}, 2.5);")
    body.append(f"    let qu{k}, qo{k} = scientific::coo_to_dense(qs{k});")
    body.append(f"    assert(qo{k}.kind == scientific::F_NONE);")
    check(k, "qu", m.toarray() * 2.5)
    # The compressed rows add the duplicates up and the round trip keeps the values.
    body.append(f"    let qr{k}, qp{k} = scientific::coo_to_csr(qc{k});")
    body.append(f"    assert(qp{k}.kind == scientific::F_NONE && qr{k}.nnz == {m.tocsr().nnz});")
    body.append(f"    let qb{k}, qq{k} = scientific::coo_from_csr(qr{k});")
    body.append(f"    assert(qq{k}.kind == scientific::F_NONE && qb{k}.nnz == {m.tocsr().nnz});")
    body.append(f"    let qf{k}, qy{k} = scientific::coo_to_dense(qb{k});")
    body.append(f"    assert(qy{k}.kind == scientific::F_NONE);")
    check(k, "qf", m.toarray())
    if n:
        r0, c0 = int(ri[0]), int(ci[0])
        want = m.toarray()[r0, c0]
        body.append(f"    let qz{k}, qv{k} = scientific::coo_get(qc{k}, {r0}, {c0});")
        body.append(f"    assert(qv{k}.kind == scientific::F_NONE && qz{k} > {num(want)} - 1.0e-12 && qz{k} < {num(want)} + 1.0e-12);")
    if len(body) >= 40: flush()

# The dense matrix.
for t in range(8):
    rows = int(rng.integers(1, 6)); cols = int(rng.integers(1, 6))
    a = np.round(rng.random((rows, cols)) * 6 - 3, 3)
    a[rng.random(a.shape) < 0.5] = 0.0
    k = count[0]; count[0] += 1
    body.append(f"    let qc{k}, qg{k} = scientific::coo_from_dense(ni_nd({ilst(a.shape)}, {lst(a)}));")
    body.append(f"    assert(qg{k}.kind == scientific::F_NONE && qc{k}.nnz == {sp.coo_array(a).nnz});")
    body.append(f"    let qd{k}, qh{k} = scientific::coo_to_dense(qc{k});")
    body.append(f"    assert(qh{k}.kind == scientific::F_NONE);")
    check(k, "qd", a)
flush()
for f in funcs:
    print(f)
print("fn test_sparse_coo_pins() {")
for i in range(len(funcs)):
    print(f"    test_sparse_coo_pins{i}();")
print("}")
