# Prints the SciPy values that test/suite/sparse_bsr_2_tests.goose pins for the sparse matrices of
# blocks. The output is Goose source.
#   python3 test/gen_sparse_bsr.py > pins.goose
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

rng = np.random.default_rng(322)
funcs = []

def block_lines(b, tag):
    """The asserts that compare the blocks of the Goose matrix `tag`."""
    return f"""    let {tag}p, {tag}i, {tag}d, {tag}f = scientific::bsr_blocks({tag});
    assert({tag}f.kind == scientific::F_NONE && all_near({tag}p, {lst(b.indptr)}) && all_near({tag}i, {lst(b.indices)}) && all_near({tag}d, {lst(b.data)}));
    assert(scientific::bsr_nnz({tag}) == {b.nnz} && {tag}.br == {b.blocksize[0]} && {tag}.bc == {b.blocksize[1]});
"""

def pin(a, br, bc):
    rows, cols = a.shape
    b = sp.bsr_array(a, blocksize=(br, bc))
    t = b.T
    x = np.round(rng.random(cols) * 4 - 2, 3)
    gets = ""
    places = [(i, j) for i in range(rows) for j in range(cols) if a[i, j] != 0][:3]
    places += [(i, j) for i in range(rows) for j in range(cols) if a[i, j] == 0][:1]
    for k, (i, j) in enumerate(places):
        gets += f"    let g{k}, gf{k} = scientific::bsr_get(s, {i}, {j});\n    assert(gf{k}.kind == scientific::F_NONE && near(g{k}, {num(a[i, j])}));\n"
    diag = f"    let dg, dgf = scientific::bsr_diagonal(s);\n    assert(dgf.kind == scientific::F_NONE && all_near(dg, {lst(b.diagonal())}));\n"
    tc = b.tocoo()
    coo = f"    let oc, ocf = scientific::bsr_to_coo(s);\n    assert(ocf.kind == scientific::F_NONE && oc.nnz == {tc.nnz});\n    let od, odf = scientific::coo_to_dense(oc);\n    assert(odf.kind == scientific::F_NONE && all_near(od, {lst(a)}));\n"
    # The coordinates of SciPy build the blocks in the order of SciPy.
    bb = sp.coo_array((tc.data, (tc.row, tc.col)), shape=tc.shape).tobsr(blocksize=(br, bc))
    back = f"    let ic, icf = scientific::coo_from_triplets({rows}, {cols}, {lst(tc.row)}, {lst(tc.col)}, {lst(tc.data)});\n    assert(icf.kind == scientific::F_NONE);\n    let s3, f9 = scientific::bsr_from_coo(ic, {br}, {bc});\n    assert(f9.kind == scientific::F_NONE);\n" + block_lines(bb, "s3")
    i = len(funcs)
    funcs.append(f"""fn test_sparse_bsr_pins{i}() {{
    let a = mx({rows}, {cols}, {lst(a)});
    let s, f0 = scientific::bsr_from_dense(a, {br}, {bc});
    assert(f0.kind == scientific::F_NONE);
{block_lines(b, "s")}    let d, f1 = scientific::bsr_to_dense(s);
    assert(f1.kind == scientific::F_NONE && all_near(d, {lst(a)}));
    assert(bsr_same(s, {rows}, {cols}, {lst(a)}));
    let c, f2 = scientific::csr_from_dense(a);
    let s2, f3 = scientific::bsr_from_csr(c, {br}, {bc});
    assert(f2.kind == scientific::F_NONE && f3.kind == scientific::F_NONE);
{block_lines(b, "s2")}    let k, f4 = scientific::bsr_to_csr(s);
    assert(f4.kind == scientific::F_NONE && k.nnz == {b.tocsr().nnz});
    let y, f5 = scientific::bsr_matvec(s, {lst(x)});
    assert(f5.kind == scientific::F_NONE && all_near(y, {lst(b @ x)}));
    let st, f6 = scientific::bsr_transpose(s);
    assert(f6.kind == scientific::F_NONE);
{block_lines(t, "st")}    let sc, f7 = scientific::bsr_scale(s, 2.5);
    assert(f7.kind == scientific::F_NONE);
    let scp, sci, scd, f8 = scientific::bsr_blocks(sc);
    assert(f8.kind == scientific::F_NONE && all_near(scd, {lst(b.data * 2.5)}));
    assert(near(scientific::bsr_sum(s), {num(b.sum())}));
{gets}{diag}{coo}{back}    return;
}}
""")

pin(np.array([[1,2,0,0],[3,4,0,0],[0,0,5,0],[0,0,0,6]], float), 2, 2)
pin(np.array([[1,0,2,0,0,0],[0,3,0,0,0,0],[0,0,0,4,5,6],[0,0,0,0,0,7]], float), 2, 3)
pin(np.array([[0,0,0,0],[0,0,0,0],[0,7,0,0],[0,0,0,8]], float), 2, 2)
pin(np.zeros((4, 4)), 2, 2)
for rows, cols, br, bc in [(4,4,2,2),(6,6,3,3),(6,4,2,2),(4,6,1,3),(5,5,1,1),(3,3,3,3),(8,6,4,2),(2,8,2,4),(6,6,2,3),(4,4,4,1),(6,9,3,3),(4,8,2,4)]:
    a = np.round(rng.random((rows, cols)), 3)
    a[a < 0.65] = 0
    pin(a, br, bc)

# A stored zero keeps its block, and the conversion to compressed rows keeps the zero.
zc = sp.csr_array((np.array([0.0, 2.0]), (np.array([0, 1]), np.array([0, 3]))), shape=(2, 4))
zb = sp.bsr_array(zc, blocksize=(2, 2))
funcs.append(f"""fn test_sparse_bsr_pins_zero() {{
    let c, f0 = scientific::csr_from_triplets(2, 4, [0.0, 1.0], [0.0, 3.0], [0.0, 2.0]);
    assert(f0.kind == scientific::F_NONE);
    let s, f1 = scientific::bsr_from_csr(c, 2, 2);
    assert(f1.kind == scientific::F_NONE);
{block_lines(zb, "s")}    let k, f2 = scientific::bsr_to_csr(s);
    assert(f2.kind == scientific::F_NONE && k.nnz == {zb.tocsr().nnz});
    return;
}}
""")

for f in funcs:
    print(f)
print("fn test_sparse_bsr_pins() {")
for i in range(len(funcs) - 1):
    print(f"    test_sparse_bsr_pins{i}();")
print("    test_sparse_bsr_pins_zero();")
print("    return;\n}")
