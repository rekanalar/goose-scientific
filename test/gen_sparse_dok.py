# Prints the SciPy values that test/run.goose pins for the sparse matrices of
# keys. The output is Goose source.
#   python3 test/gen_sparse_dok.py > pins.goose
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

def keys(d):
    ks = list(d.keys())
    return [k[0] for k in ks], [k[1] for k in ks], [float(d[k]) for k in ks]

def pin(d):
    return ", ".join(lst(q) for q in keys(d))

cases = []

def emit(name, rows, cols, ops):
    d = sp.dok_array((rows, cols))
    body = ["    var d, f0 = scientific::dok_new(%d, %d);" % (rows, cols),
            "    assert(f0.kind == scientific::F_NONE);"]
    for op in ops:
        if op[0] == "set":
            _, i, j, v = op
            d[i, j] = v
            body.append("    assert(scientific::dok_set(d, %s, %s, %s).kind == scientific::F_NONE);" % (num(i), num(j), num(v)))
        elif op[0] == "pop":
            _, i, j = op
            got = d.pop((i, j), 0.0)
            tag = len(body)
            body.append("    let p%d, g%d = scientific::dok_pop(d, %s, %s);" % (tag, tag, num(i), num(j)))
            body.append("    assert(g%d.kind == scientific::F_NONE && p%d == %s);" % (tag, tag, num(got)))
        elif op[0] == "diag":
            _, k, vals = op
            d.setdiag(np.array(vals, dtype=float), k)
            body.append("    assert(scientific::dok_setdiag(d, %d, %s).kind == scientific::F_NONE);" % (k, lst(vals)))
        elif op[0] == "resize":
            _, r, c = op
            d.resize(r, c)
            body.append("    assert(scientific::dok_resize(d, %d, %d).kind == scientific::F_NONE);" % (r, c))
            rows, cols = r, c
    r, c, v = keys(d)
    body.append("    assert(dok_is(d, %s));" % pin(d))
    body.append("    assert(scientific::dok_nnz(d) == %d);" % d.nnz)
    body.append("    let dn, fd = scientific::dok_to_dense(d);")
    body.append("    assert(fd.kind == scientific::F_NONE && all_near(dn, %s));" % lst(d.toarray().flatten()))
    t = d.transpose()
    tr, tc, tv = keys(t)
    body.append("    let t, ft = scientific::dok_transpose(d);")
    body.append("    assert(ft.kind == scientific::F_NONE && dok_is(t, %s));" % pin(t))
    body.append("    let co, fc = scientific::dok_to_coo(d);")
    body.append("    assert(fc.kind == scientific::F_NONE && co.nnz == %d);" % d.nnz)
    body.append("    let cs, fs = scientific::dok_to_csr(d);")
    body.append("    assert(fs.kind == scientific::F_NONE && cs.nnz == %d);" % d.nnz)
    x = [float(((q * 3) % 7) - 2) for q in range(cols)]
    body.append("    let mv, fm = scientific::dok_matvec(d, %s);" % lst(x))
    body.append("    assert(fm.kind == scientific::F_NONE && all_near(mv, %s));" % lst(d @ np.array(x)))
    body.append("    let sm, fz = scientific::dok_sum(d);")
    body.append("    assert(fz.kind == scientific::F_NONE && near(sm, %s));" % num(d.sum()))
    body.append("    let dg, fg = scientific::dok_diagonal(d);")
    body.append("    assert(fg.kind == scientific::F_NONE && all_near(dg, %s));" % lst(d.diagonal()))
    sc = d * -1.5
    body.append("    let sd, fk = scientific::dok_scale(d, -1.5);")
    body.append("    assert(fk.kind == scientific::F_NONE && dok_is(sd, %s));" % pin(sc))
    cases.append((name, "\n".join(body)))

# the pins of the contract
emit("p1", 3, 4, [("set", 2, 1, 5.0), ("set", 0, 0, 1.0), ("set", 1, 3, 3.0), ("set", 0, 2, 2.0)])
emit("p2", 3, 4, [("set", 2, 1, 5.0), ("set", 0, 0, 1.0), ("set", 1, 3, 3.0), ("set", 0, 2, 2.0), ("set", 2, 1, 6.0)])
emit("p3", 3, 4, [("set", 2, 1, 6.0), ("set", 0, 0, 1.0), ("set", 1, 3, 3.0), ("set", 0, 2, 2.0), ("set", 0, 0, 0.0)])
emit("p4", 3, 4, [("set", 2, 1, 6.0), ("set", 0, 0, 1.0), ("set", 1, 3, 3.0), ("set", 0, 2, 2.0), ("set", 0, 0, 0.0), ("set", 0, 0, 9.0)])
emit("p5", 3, 4, [("set", 2, 1, 6.0), ("set", 0, 0, 9.0), ("set", 1, 3, 3.0), ("set", 0, 2, 2.0), ("pop", 1, 3), ("pop", 2, 2)])
for n, (k, vals) in enumerate([(0, [2, 2, 2]), (0, [1, 0, 3]), (1, [4, 5, 6]), (-1, [7, 8]), (1, [4, 5]), (-1, [7, 8, 9]), (0, [3])]):
    emit("p10_%d" % n, 3, 4, [("set", 1, 1, 5.0), ("diag", k, vals)])
emit("p10_edge_a", 3, 4, [("set", 1, 1, 5.0), ("diag", 3, [1.0])])
emit("p10_edge_b", 3, 4, [("set", 1, 1, 5.0), ("diag", -2, [6.0])])
emit("p10_long", 3, 4, [("set", 1, 1, 5.0), ("diag", 0, [1.0, 2.0, 3.0, 4.0, 5.0])])
emit("p11a", 3, 4, [("set", 2, 3, 1.0), ("set", 0, 0, 2.0), ("set", 1, 1, 3.0), ("set", 0, 3, 4.0), ("resize", 2, 3)])
emit("p11b", 3, 4, [("set", 2, 3, 1.0), ("set", 0, 0, 2.0), ("set", 1, 1, 3.0), ("set", 0, 3, 4.0), ("resize", 2, 3), ("resize", 4, 5)])

# seeded random series
rng = random.Random(323)
for n in range(60):
    rows, cols = rng.randint(1, 6), rng.randint(1, 6)
    r0, c0 = rows, cols
    ops = []
    for _ in range(rng.randint(5, 40)):
        t = rng.random()
        if t < 0.55:
            ops.append(("set", rng.randrange(rows), rng.randrange(cols), float(rng.choice([0, 0, 1, 2, -3, 5, 7]))))
        elif t < 0.7:
            ops.append(("pop", rng.randrange(rows), rng.randrange(cols)))
        elif t < 0.85:
            k = rng.randint(-(rows - 1), cols - 1)
            ln = min(rows, cols - k) if k >= 0 else min(rows + k, cols)
            ops.append(("diag", k, [float(rng.choice([0, 1, 2, 4])) for _ in range(rng.randint(1, ln + 1))]))
        else:
            r, c = rng.randint(1, 6), rng.randint(1, 6)
            ops.append(("resize", r, c))
            rows, cols = r, c
    emit("r%d" % n, r0, c0, ops)

# the conversions
dense = np.array([[0, 2, 0], [3, 0, 4]], dtype=float)
dd = sp.dok_array(dense)
cs = sp.csr_array(dense)
co = sp.coo_array((np.array([1.0, 2.0, 3.0]), (np.array([1, 0, 1]), np.array([1, 2, 1]))), shape=(2, 3))
cod = sp.dok_array(co)
conv = """    let a, fa = scientific::from_values(%s);
    assert(fa.kind == scientific::F_NONE);
    let a2, fr = scientific::reshape(a, [2, 3]);
    assert(fr.kind == scientific::F_NONE);
    let d, fd = scientific::dok_from_dense(a2);
    assert(fd.kind == scientific::F_NONE && dok_is(d, %s));
    let cs, fc = scientific::csr_from_dense(a2);
    let d2, fe = scientific::dok_from_csr(cs);
    assert(fe.kind == scientific::F_NONE && dok_is(d2, %s));
    let co, fo = scientific::coo_from_triplets(2, 3, %s, %s, %s);
    assert(fo.kind == scientific::F_NONE);
    let d3, ff = scientific::dok_from_coo(co);
    assert(ff.kind == scientific::F_NONE && dok_is(d3, %s));""" % (
    lst(dense.flatten()),
    pin(dd),
    pin(sp.dok_array(cs)),
    lst(co.row), lst(co.col), lst(co.data),
    pin(cod),
)
cases.append(("conv", conv))

print("// The pins of the sparse matrices of keys, measured from SciPy.")
print("fn dok_is(d: scientific::Dok&, r: f64[:], c: f64[:], v: f64[:]) -> bool {")
print("    let rr, cc, vv, f = scientific::dok_keys(d);")
print("    if f.kind != scientific::F_NONE || rr.data.len != r.len { return false; }")
print("    var i: i64 = 0;")
print("    while i < r.len {")
print("        if rr.data[i] != r[i] || cc.data[i] != c[i] || !near(vv.data[i], v[i]) { return false; }")
print("        i += 1;")
print("    }")
print("    return true;")
print("}")
print()
for name, body in cases:
    print("fn dok_pin_%s() {\n%s\n}\n" % (name, body))
print("fn test_sparse_dok_pins() {")
for name, _ in cases:
    print("    dok_pin_%s();" % name)
print("}")
