# Prints the SciPy values that test/suite/sparse_lil_tests.goose pins for the sparse matrices of
# lists. The output is Goose source.
#   python3 test/gen_sparse_lil.py > pins.goose
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
    r, c, v = [], [], []
    for i, (cs, vs) in enumerate(zip(d.rows, d.data)):
        for j, x in zip(cs, vs):
            r.append(i); c.append(j); v.append(float(x))
    return r, c, v

def pin(d):
    return ", ".join(lst(q) for q in keys(d))

cases = []

def emit(name, rows, cols, ops):
    d = sp.lil_array((rows, cols))
    body = ["    var d, f0 = scientific::lil_new(%d, %d);" % (rows, cols),
            "    assert(f0.kind == scientific::F_NONE);"]
    for op in ops:
        if op[0] == "set":
            _, i, j, v = op
            d[i, j] = v
            body.append("    assert(scientific::lil_set(d, %s, %s, %s).kind == scientific::F_NONE);" % (num(i), num(j), num(v)))
        elif op[0] == "diag":
            _, k, vals = op
            d.setdiag(np.array(vals, dtype=float), k)
            body.append("    assert(scientific::lil_setdiag(d, %d, %s).kind == scientific::F_NONE);" % (k, lst(vals)))
        elif op[0] == "resize":
            _, r, c = op
            d.resize(r, c)
            body.append("    assert(scientific::lil_resize(d, %d, %d).kind == scientific::F_NONE);" % (r, c))
            rows, cols = r, c
    body.append("    assert(lil_is(d, %s));" % pin(d))
    body.append("    assert(scientific::lil_nnz(d) == %d);" % d.nnz)
    body.append("    let dn, fd = scientific::lil_to_dense(d);")
    body.append("    assert(fd.kind == scientific::F_NONE && all_near(dn, %s));" % lst(d.toarray().flatten()))
    t = d.T.tolil()
    body.append("    let t, ft = scientific::lil_transpose(d);")
    body.append("    assert(ft.kind == scientific::F_NONE && lil_is(t, %s));" % pin(t))
    body.append("    let co, fc = scientific::lil_to_coo(d);")
    body.append("    assert(fc.kind == scientific::F_NONE && co.nnz == %d);" % d.nnz)
    body.append("    let cs, fs = scientific::lil_to_csr(d);")
    body.append("    assert(fs.kind == scientific::F_NONE && cs.nnz == %d);" % d.nnz)
    x = [float(((q * 3) % 7) - 2) for q in range(cols)]
    body.append("    let mv, fm = scientific::lil_matvec(d, %s);" % lst(x))
    body.append("    assert(fm.kind == scientific::F_NONE && all_near(mv, %s));" % lst(d @ np.array(x)))
    body.append("    let sm, fz = scientific::lil_sum(d);")
    body.append("    assert(fz.kind == scientific::F_NONE && near(sm, %s));" % num(d.sum()))
    body.append("    let dg, fg = scientific::lil_diagonal(d);")
    body.append("    assert(fg.kind == scientific::F_NONE && all_near(dg, %s));" % lst(d.diagonal()))
    sc = (d * -1.5).tolil()
    body.append("    let sd, fk = scientific::lil_scale(d, -1.5);")
    body.append("    assert(fk.kind == scientific::F_NONE && lil_is(sd, %s));" % pin(sc))
    cases.append((name, "\n".join(body)))

# the pins of the contract
emit("p1", 3, 4, [("set", 2, 3, 1.5), ("set", 0, 2, 2.0), ("set", 0, 0, 3.0), ("set", 1, 1, 4.0)])
emit("p2", 3, 4, [("set", 2, 3, 1.5), ("set", 0, 2, 2.0), ("set", 0, 0, 3.0), ("set", 1, 1, 4.0), ("set", 0, 2, 7.0)])
emit("p3", 3, 4, [("set", 2, 3, 1.5), ("set", 0, 2, 2.0), ("set", 0, 0, 3.0), ("set", 1, 1, 4.0), ("set", 0, 2, 7.0), ("set", 0, 0, 0.0)])
emit("p3z", 2, 2, [("set", 0, 0, 0.0)])
emit("p4", 4, 4, [("diag", 0, [1, 2, 3, 4]), ("diag", 1, [5, 6, 7]), ("diag", -2, [8, 9])])
for n, (k, vals) in enumerate([(0, [2, 2, 2]), (0, [1, 0, 3]), (1, [4, 5, 6]), (-1, [7, 8]), (1, [4, 5]), (-1, [7, 8, 9]), (0, [3])]):
    emit("p5_%d" % n, 3, 4, [("set", 1, 1, 5.0), ("diag", k, vals)])
emit("p5_edge_a", 3, 4, [("set", 1, 1, 5.0), ("diag", 3, [1.0])])
emit("p5_edge_b", 3, 4, [("set", 1, 1, 5.0), ("diag", -2, [6.0])])
emit("p5_long", 3, 4, [("set", 1, 1, 5.0), ("diag", 0, [1.0, 2.0, 3.0, 4.0, 5.0])])
base = [("diag", 0, [1, 2, 3, 4]), ("diag", 1, [5, 6, 7]), ("diag", -2, [8, 9])]
emit("p6a", 4, 4, base + [("resize", 2, 3)])
emit("p6b", 4, 4, base + [("resize", 5, 6)])

# seeded random series, columns in any order
rng = random.Random(324)
for n in range(60):
    rows, cols = rng.randint(1, 6), rng.randint(1, 6)
    r0, c0 = rows, cols
    ops = []
    for _ in range(rng.randint(5, 40)):
        t = rng.random()
        if t < 0.7:
            ops.append(("set", rng.randrange(rows), rng.randrange(cols), float(rng.choice([0, 0, 1, 2, -3, 5, 7]))))
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
dd = sp.lil_array(dense)
co = sp.coo_array((np.array([1.0, 2.0, 3.0]), (np.array([1, 0, 1]), np.array([1, 2, 1]))), shape=(2, 3))
cod = sp.lil_array(co)
conv = """    let a, fa = scientific::from_values(%s);
    assert(fa.kind == scientific::F_NONE);
    let a2, fr = scientific::reshape(a, [2, 3]);
    assert(fr.kind == scientific::F_NONE);
    let d, fd = scientific::lil_from_dense(a2);
    assert(fd.kind == scientific::F_NONE && lil_is(d, %s));
    let cs, fc = scientific::csr_from_dense(a2);
    let d2, fe = scientific::lil_from_csr(cs);
    assert(fe.kind == scientific::F_NONE && lil_is(d2, %s));
    let co, fo = scientific::coo_from_triplets(2, 3, %s, %s, %s);
    assert(fo.kind == scientific::F_NONE);
    let d3, ff = scientific::lil_from_coo(co);
    assert(ff.kind == scientific::F_NONE && lil_is(d3, %s));
    // The rows of the sets in any order stand sorted.
    var s, fs = scientific::lil_new(2, 5);
    assert(fs.kind == scientific::F_NONE);
    assert(scientific::lil_set(s, 1.0, 4.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(s, 1.0, 0.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(s, 1.0, 2.0, 3.0).kind == scientific::F_NONE);
    let rc, rv, fw = scientific::lil_row(s, 1.0);
    assert(fw.kind == scientific::F_NONE && all_near(rc, [0.0, 2.0, 4.0]) && all_near(rv, [2.0, 3.0, 1.0]));
    let ec, ev, fx = scientific::lil_row(s, 0.0);
    assert(fx.kind == scientific::F_NONE && ec.data.len == 0 && ev.data.len == 0);""" % (
    lst(dense.flatten()),
    pin(dd),
    pin(sp.lil_array(sp.csr_array(dense))),
    lst(co.row), lst(co.col), lst(co.data),
    pin(cod),
)
cases.append(("conv", conv))

print("// The pins of the sparse matrices of lists, measured from SciPy.")
print("fn lil_is(l: scientific::Lil&, r: f64[:], c: f64[:], v: f64[:]) -> bool {")
print("    var p: i64 = 0;")
print("    var i: i64 = 0;")
print("    while i < l.rows {")
print("        let cc, vv, f = scientific::lil_row(l, i as f64);")
print("        if f.kind != scientific::F_NONE { return false; }")
print("        var q: i64 = 0;")
print("        while q < cc.data.len {")
print("            if p >= r.len || r[p] != i as f64 || c[p] != cc.data[q] || !near(vv.data[q], v[p]) { return false; }")
print("            p += 1;")
print("            q += 1;")
print("        }")
print("        i += 1;")
print("    }")
print("    return p == r.len;")
print("}")
print()
for name, body in cases:
    print("fn lil_pin_%s() {\n%s\n}\n" % (name, body))
print("fn test_sparse_lil_pins() {")
for name, _ in cases:
    print("    lil_pin_%s();" % name)
print("}")
