# Prints the SciPy values that test/run.goose pins for the sparse matrices of
# diagonals. The output is Goose source.
#   python3 test/gen_sparse_dia.py > pins.goose
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

def dia_values(d, rows, cols):
    """The values of the diagonals of a SciPy DIA, in the order of the row."""
    out = []
    for i, k in enumerate(d.offsets):
        k = int(k)
        for j in range(max(0, k), min(cols - 1, rows - 1 + k) + 1):
            out.append(d.data[i][j] if j < d.data.shape[1] else 0.0)
    return out

rng = np.random.default_rng(316)
funcs = []

def pin(a):
    """The pins of one matrix, taken from SciPy. Each matrix gets its own
    function, so the names inside it need no number."""
    rows, cols = a.shape
    d = sp.dia_array(a)
    offs = [int(x) for x in d.offsets]
    vals = dia_values(d, rows, cols)
    x = np.round(rng.random(cols) * 4 - 2, 3)
    b = []
    b.append(f"    let qc, qg = scientific::dia_from_dense(ni_nd({ilst(a.shape)}, {lst(a)}));")
    b.append(f"    assert(qg.kind == scientific::F_NONE && qc.nnz == {len(vals)});")
    b.append(f"    assert(qc.ndiag == {len(offs)} && scientific::dia_nnz(qc) == {len(vals)});")
    b.append(f"    let qo, qv, qu = scientific::dia_diagonals(qc);")
    b.append(f"    assert(qu.kind == scientific::F_NONE);")
    b.append(f"    assert(qo.shape == {ilst([len(offs)])} && ms_same(qo, {lst(offs)}));")
    b.append(f"    assert(qv.shape == {ilst([len(vals)])} && ms_same(qv, {lst(vals)}));")
    b.append(f"    let qd, qh = scientific::dia_to_dense(qc);")
    b.append(f"    assert(qh.kind == scientific::F_NONE && ms_same(qd, {lst(a)}));")
    b.append(f"    assert(dia_same(qc, {rows}, {cols}, {lst(a)}));")
    b.append(f"    let qm, qn = scientific::dia_matvec(qc, {lst(x)});")
    b.append(f"    assert(qn.kind == scientific::F_NONE && ms_same(qm, {lst(a @ x)}));")
    b.append(f"    let qt, qtg = scientific::dia_transpose(qc);")
    b.append(f"    assert(qtg.kind == scientific::F_NONE && qt.nnz == {len(vals)});")
    b.append(f"    let qy, qz = scientific::dia_to_dense(qt);")
    b.append(f"    assert(qz.kind == scientific::F_NONE && ms_same(qy, {lst(a.T)}));")
    b.append(f"    let qsc, qsg = scientific::dia_scale(qc, 2.5);")
    b.append(f"    assert(qsg.kind == scientific::F_NONE && qsc.nnz == {len(vals)});")
    b.append(f"    let qsd, qse = scientific::dia_to_dense(qsc);")
    b.append(f"    assert(qse.kind == scientific::F_NONE && ms_same(qsd, {lst(a * 2.5)}));")
    b.append(f"    assert(near(scientific::dia_sum(qc), {num(d.sum())}));")
    b.append(f"    let qk, ql = scientific::dia_to_csr(qc);")
    b.append(f"    assert(ql.kind == scientific::F_NONE && qk.nnz == {d.tocsr().nnz});")
    b.append(f"    let qj, qi = scientific::dia_to_coo(qc);")
    b.append(f"    assert(qi.kind == scientific::F_NONE && qj.nnz == {d.tocoo().nnz});")
    b.append(f"    let qf, qe = scientific::dia_from_csr(qk);")
    b.append(f"    assert(qe.kind == scientific::F_NONE && qf.nnz == {len(vals)});")
    b.append(f"    let qb, qa = scientific::dia_to_dense(qf);")
    b.append(f"    assert(qa.kind == scientific::F_NONE && ms_same(qb, {lst(a)}));")
    b.append(f"    let qp, qq = scientific::dia_from_coo(qj);")
    b.append(f"    assert(qq.kind == scientific::F_NONE && qp.nnz == {len(vals)});")
    b.append(f"    let qr, qo2 = scientific::dia_to_dense(qp);")
    b.append(f"    assert(qo2.kind == scientific::F_NONE && ms_same(qr, {lst(a)}));")
    b.append(f"    let qu2, qw2 = scientific::dia_from_diagonals({rows}, {cols}, {lst(offs)}, {lst(vals)});")
    b.append(f"    assert(qw2.kind == scientific::F_NONE && qu2.nnz == {len(vals)});")
    b.append(f"    let qx2, qy2 = scientific::dia_to_dense(qu2);")
    b.append(f"    assert(qy2.kind == scientific::F_NONE && ms_same(qx2, {lst(a)}));")
    funcs.append(f"fn test_sparse_dia_pins{len(funcs)}() {{\n" + "\n".join(b) + "\n}\n")

# The matrices of the specification.
pinned = [
    np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]]),
    np.array([[1.0, 2.0], [0.0, 3.0], [4.0, 0.0]]),
    np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]),
    np.zeros((2, 2)),
    np.array([[1.0, 0.0], [0.0, 2.0]]),
    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
]
for a in pinned:
    pin(a)

# The matrices of one to eight rows and columns.
for t in range(30):
    rows = int(rng.integers(1, 9)); cols = int(rng.integers(1, 9))
    a = np.round(rng.random((rows, cols)) * 8 - 4, 3)
    a[rng.random(a.shape) < 0.6] = 0.0
    pin(a)

for f in funcs:
    print(f)
print("fn test_sparse_dia_pins() {")
for i in range(len(funcs)):
    print(f"    test_sparse_dia_pins{i}();")
print("}")
