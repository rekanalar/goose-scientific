# Prints the SciPy values that test/suite/dist_pins_tests.goose pins for distance_transform_cdt,
# distance_transform_bf and value_indices. The output is Goose source.
#   python3 test/gen_dist.py > pins.goose
import numpy as np
from scipy import ndimage as ndi

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

rng = np.random.default_rng(307)
count = [0]
funcs = []
body = []

def flush():
    global body
    if body:
        funcs.append(f"fn test_dist_pins{len(funcs)}() {{\n" + "\n".join(body) + "\n}\n")
        body = []

def check(k, name, want):
    want = np.asarray(want, float)
    body.append(f"    assert({name}{k}.shape == {ilst(want.shape)} && ms_same({name}{k}, {lst(want)}));")

def emit1(call, want):
    k = count[0]; count[0] += 1
    body.append(f"    let r{k}, g{k} = {call};")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE);")
    check(k, "r", want)
    if len(body) >= 60: flush()

def emit2(call, want, idx):
    k = count[0]; count[0] += 1
    body.append(f"    let r{k}, i{k}, g{k} = {call};")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE);")
    check(k, "r", want)
    check(k, "i", idx)
    if len(body) >= 60: flush()

def nd(x):
    return f"ni_nd({ilst(x.shape)}, {lst(x)})"

def pick():
    nd_ = int(rng.integers(1, 4))
    shape = tuple(int(x) for x in rng.integers(1, 6, size=nd_))
    a = rng.random(shape) < 0.7
    if a.all():
        a.reshape(-1)[0] = False
    return nd_, shape, a

for t in range(60):
    nd_, shape, a = pick()
    af = a.astype(float)
    m = str(rng.choice(['chessboard', 'taxicab', 'cityblock', 'manhattan']))
    d, i = ndi.distance_transform_cdt(a, metric=m, return_indices=True)
    emit1(f'scientific::distance_transform_cdt({nd(af)}, "{m}")', d)
    emit2(f'scientific::distance_transform_cdt_indices({nd(af)}, "{m}")', d, i)
    st = rng.random((3,) * nd_) < 0.6
    if t % 2 == 0:
        st = st | st[(slice(None, None, -1),) * nd_]
    d, i = ndi.distance_transform_cdt(a, metric=st, return_indices=True)
    emit2(f'scientific::distance_transform_cdt_st_indices({nd(af)}, {nd(st.astype(float))})', d, i)
    emit1(f'scientific::distance_transform_cdt_st({nd(af)}, {nd(st.astype(float))})', d)

for t in range(60):
    nd_, shape, a = pick()
    af = a.astype(float)
    m = str(rng.choice(['euclidean', 'taxicab', 'chessboard']))
    smp = [float(x) for x in np.round(rng.random(nd_) * 3 + 0.5, 1)] if t % 2 == 0 else None
    d, i = ndi.distance_transform_bf(a, m, smp, True, True)
    s = ilst([]) if smp is None else lst(smp)
    emit1(f'scientific::distance_transform_bf({nd(af)}, "{m}", {s})', d)
    emit2(f'scientific::distance_transform_bf_indices({nd(af)}, "{m}", {s})', d, i)

for t in range(40):
    nd_ = int(rng.integers(1, 4))
    shape = tuple(int(x) for x in rng.integers(1, 6, size=nd_))
    a = rng.integers(-2, 4, size=shape)
    vals = {}
    r = ndi.value_indices(a)
    keys = sorted(r.keys())
    def rows(k):
        return np.stack(r[k], axis=1) if True else None
    uv = np.array(keys, float)
    uc = np.array([len(r[k][0]) for k in keys], float)
    rw = np.concatenate([np.stack(r[k], axis=1) for k in keys], axis=0)
    k = count[0]; count[0] += 1
    body.append(f"    let v{k}, c{k}, w{k}, g{k} = scientific::value_indices({nd(a.astype(float))});")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE);")
    check(k, "v", uv); check(k, "c", uc); check(k, "w", rw)
    ig = int(rng.integers(-2, 4))
    r2 = ndi.value_indices(a, ignore_value=ig)
    keys2 = sorted(r2.keys())
    k = count[0]; count[0] += 1
    body.append(f"    let v{k}, c{k}, w{k}, g{k} = scientific::value_indices({nd(a.astype(float))}, {num(ig)}, true);")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE);")
    if keys2:
        check(k, "v", np.array(keys2, float))
        check(k, "c", np.array([len(r2[q][0]) for q in keys2], float))
        check(k, "w", np.concatenate([np.stack(r2[q], axis=1) for q in keys2], axis=0))
    else:
        body.append(f"    assert(v{k}.data.len == 0 && w{k}.data.len == 0);")
    if len(body) >= 60: flush()
flush()
for f in funcs:
    print(f)
print("fn test_dist_pins() {")
for i in range(len(funcs)):
    print(f"    test_dist_pins{i}();")
print("}")
