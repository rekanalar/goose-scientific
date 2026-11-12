# Prints the SciPy values that test/run.goose pins for lsmr. The output is
# Goose source.
#   python3 test/gen_sparse_lsmr.py > pins.goose
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

def lst(a):
    return "[" + ", ".join(num(x) for x in np.asarray(a, float).reshape(-1)) + "]"

def ilst(a):
    return "[" + ", ".join(str(int(x)) for x in a) + "]"

rng = np.random.default_rng(315)
funcs = []
body = []
count = [0]

def flush():
    global body
    if body:
        funcs.append(f"fn test_sparse_lsmr_pins{len(funcs)}() {{\n" + "\n".join(body) + "\n}\n")
        body = []

def case(a, b, damp, maxiter):
    k = count[0]; count[0] += 1
    kw = dict(damp=damp)
    if maxiter:
        kw["maxiter"] = maxiter
    x, istop, itn, normr, normar, norma, conda, normx = sla.lsmr(sp.csr_array(a), b, atol=1e-6, btol=1e-6, **kw)
    body.append(f"    let qa{k}, qf{k} = scientific::csr_from_dense(ni_nd({ilst(a.shape)}, {lst(a)}));")
    body.append(f"    assert(qf{k}.kind == scientific::F_NONE);")
    body.append(f"    let qr{k}, qg{k} = scientific::lsmr(qa{k}, {lst(b)}, {num(damp)}, 1.0e-6, 1.0e-6, 1.0e8, {maxiter});")
    body.append(f"    assert(qg{k}.kind == scientific::F_NONE && qr{k}.istop == {istop} && qr{k}.itn == {itn});")
    body.append(f"    assert(all_near(qr{k}.x, {lst(x)}));")
    est = [normr, normar, norma, conda, normx]
    body.append(f"    let qe{k} = ni_nd([5], [qr{k}.normr, qr{k}.normar, qr{k}.norma, qr{k}.conda, qr{k}.normx]);")
    body.append(f"    assert(all_near(qe{k}, {lst(est)}));")
    if len(body) >= 30:
        flush()

for t in range(18):
    m = int(rng.integers(1, 9)); n = int(rng.integers(1, 9))
    a = np.round(rng.random((m, n)) * 6 - 3, 3)
    a[rng.random((m, n)) < 0.3] = 0.0
    if not a.any():
        a[0, 0] = 1.0
    b = np.round(rng.random(m) * 4 - 2, 3)
    if not b.any():
        b[0] = 1.0
    damp = [0.0, 0.0, 0.5, 2.0][t % 4]
    maxiter = 0 if t % 5 else 2
    case(a, b, damp, maxiter)
flush()
for f in funcs:
    print(f)
print("fn test_sparse_lsmr_pins() {")
for i in range(len(funcs)):
    print(f"    test_sparse_lsmr_pins{i}();")
print("}")
