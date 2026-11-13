# Prints the SciPy values that test/suite/sparse_bmat_tests.goose pins for the block matrix and
# the columnwise diagonals. The output is Goose source.
#   python3 test/gen_sparse_bmat.py > pins.goose
import random
import numpy as np
import scipy.sparse as sp

def num(x):
    s = np.format_float_positional(float(x), unique=True, trim='0')
    if s.endswith('.'):
        s += '0'
    if '.' not in s:
        s += '.0'
    return s

def lst(xs):
    return "[" + ", ".join(num(x) for x in np.asarray(xs, float).reshape(-1)) + "]"

cases = []

def emit_bmat(name, rows, cols, blocks):
    # blocks holds (grid row, grid column, dense array) for every block.
    # A block may carry triplets, so that it holds a stored zero.
    body = ["    var g, f0 = scientific::bmat_new(%d, %d);" % (rows, cols),
            "    assert(f0.kind == scientific::F_NONE);"]
    sgrid = [[None] * cols for _ in range(rows)]
    for blk in blocks:
        i, j, arr = blk[0], blk[1], blk[2]
        if len(blk) > 3:
            ri, ci, vv = blk[3]
            body.append("    bmat_place_triplets(g, %d.0, %d.0, %d, %d, %s, %s, %s);" % (
                i, j, arr.shape[0], arr.shape[1], lst(ri), lst(ci), lst(vv)))
            sgrid[i][j] = sp.csr_matrix((np.array(vv), (np.array(ri), np.array(ci))), shape=arr.shape)
        else:
            body.append("    bmat_place(g, %d.0, %d.0, %s, %d, %d);" % (
                i, j, lst(arr.flatten()), arr.shape[0], arr.shape[1]))
            sgrid[i][j] = sp.csr_matrix(arr)
    res = sp.bmat(sgrid)
    body.append("    let m, fm = scientific::csr_bmat(g);")
    body.append("    assert(fm.kind == scientific::F_NONE);")
    body.append("    let dn, fz = scientific::csr_to_dense(m);")
    body.append("    assert(fz.kind == scientific::F_NONE && all_near(dn, %s));" % lst(res.toarray().flatten()))
    body.append("    assert(m.nnz == %d);" % res.nnz)
    cases.append((name, "\n".join(body)))

def dia_len(rows, cols, off):
    if off >= 0:
        return min(rows, cols - off)
    return min(rows + off, cols)

def emit_spdiags(name, data, offsets, rows, cols):
    body = ["    let a2 = mx(%d, %d, %s);" % (data.shape[0], data.shape[1], lst(data.flatten())),
            "    let d, fd = scientific::spdiags(a2, %s, %d, %d);" % (lst(offsets), rows, cols),
            "    assert(fd.kind == scientific::F_NONE);"]
    res = sp.spdiags(data, offsets, rows, cols)
    body.append("    let dn, fz = scientific::dia_to_dense(d);")
    body.append("    assert(fz.kind == scientific::F_NONE && all_near(dn, %s));" % lst(res.toarray().flatten()))
    total = sum(dia_len(rows, cols, int(o)) for o in offsets)
    body.append("    assert(d.nnz == %d);" % total)
    cases.append((name, "\n".join(body)))

# the blocks of the contract
a = np.array([[1.0, 2.0], [0.0, 3.0]])
b = np.array([[4.0], [5.0]])
c = np.array([[6.0, 0.0]])

emit_bmat("p1", 2, 2, [(0, 0, a), (0, 1, b), (1, 0, c)])
emit_bmat("p2", 2, 2, [(0, 0, a), (1, 1, c)])
emit_bmat("p3", 1, 2, [(0, 0, a), (0, 1, a)])
emit_bmat("p4", 2, 2, [(0, 1, a), (1, 0, a)])
emit_bmat("p5", 2, 1, [(0, 0, a)])
emit_bmat("p6", 1, 3, [(0, 0, a), (0, 2, a)])
# a block that holds a stored zero
emit_bmat("p2z", 2, 2, [(0, 0, a), (1, 1, c, ([0, 0], [0, 1], [6.0, 0.0]))])

# the diagonal data of the contract
d1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
emit_spdiags("p7", d1, [0.0, 1.0], 3, 4)
emit_spdiags("p8", d1, [0.0, -1.0], 4, 3)
emit_spdiags("p9", np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]), [0.0], 3, 4)
emit_spdiags("p10", np.array([[1.0, 2.0, 3.0, 4.0]]), [-2.0], 3, 4)
emit_spdiags("p11", d1, [1.0, 0.0], 3, 4)
emit_spdiags("p12", np.array([[1.0, 2.0], [5.0, 6.0]]), [0.0, 1.0], 3, 4)

# seeded random grids of blocks
rng = random.Random(327)
for n in range(40):
    rows = rng.randint(1, 3)
    cols = rng.randint(1, 3)
    hrow = [rng.randint(1, 3) for _ in range(rows)]
    wcol = [rng.randint(1, 3) for _ in range(cols)]
    blocks = []
    for i in range(rows):
        for j in range(cols):
            if rng.random() < 0.25:
                continue
            arr = np.zeros((hrow[i], wcol[j]))
            for _ in range(rng.randint(0, 4)):
                arr[rng.randrange(hrow[i]), rng.randrange(wcol[j])] = float(rng.choice([0, 1, 2, -3, 5]))
            blocks.append((i, j, arr))
    if not blocks:
        blocks.append((0, 0, np.array([[1.0]])))
    emit_bmat("r%d" % n, rows, cols, blocks)

# seeded random diagonal data
rng = random.Random(328)
for n in range(40):
    rows = rng.randint(1, 5)
    cols = rng.randint(1, 5)
    pool = list(range(-(rows - 1), cols))
    k = rng.randint(1, min(3, len(pool)))
    offs = sorted(rng.sample(pool, k))
    width = rng.randint(1, cols + 2)
    data = np.array([[float(rng.choice([0, 1, 2, -3, 5])) for _ in range(width)] for _ in range(k)])
    emit_spdiags("s%d" % n, data, [float(o) for o in offs], rows, cols)

print("// The pins of the block matrix and the columnwise diagonals, measured from SciPy.")
print("fn bmat_place(g: scientific::Bmat&, i: f64, j: f64, v: f64[:], rows: i64, cols: i64) {")
print("    let a2 = mx(rows, cols, v);")
print("    let c, fc = scientific::csr_from_dense(a2);")
print("    assert(fc.kind == scientific::F_NONE);")
print("    assert(scientific::bmat_put(g, i, j, c).kind == scientific::F_NONE);")
print("}")
print()
print("fn bmat_place_triplets(g: scientific::Bmat&, i: f64, j: f64, rows: i64, cols: i64, ri: f64[:], ci: f64[:], vv: f64[:]) {")
print("    let c, fc = scientific::csr_from_triplets(rows, cols, ri, ci, vv);")
print("    assert(fc.kind == scientific::F_NONE);")
print("    assert(scientific::bmat_put(g, i, j, c).kind == scientific::F_NONE);")
print("}")
print()
for name, body in cases:
    print("fn bmat_pin_%s() {\n%s\n}\n" % (name, body))
print("fn test_sparse_bmat_pins() {")
for name, _ in cases:
    print("    bmat_pin_%s();" % name)
print("}")
