# Prints the SciPy values that test/run.goose pins for the spline filters
# and the interpolating transforms. The output is Goose source.
#   python3 test/gen_transform.py > pins.goose
import numpy as np
from scipy import ndimage as ndi

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

rng = np.random.default_rng(304)
A = {1: np.round(rng.normal(size=7), 3),
     2: np.round(rng.normal(size=(4, 5)), 3),
     3: np.round(rng.normal(size=(3, 2, 4)), 3)}
MODES = ["mirror", "reflect", "wrap", "grid-wrap", "nearest", "constant", "grid-constant"]
count = [0]
funcs = []
body = []

def flush():
    global body
    if body:
        funcs.append(f"fn test_transform_pins{len(funcs)}() {{\n" + "\n".join(body) + "\n}\n")
        body = []

def emit(call, want):
    k = count[0]
    count[0] += 1
    want = np.asarray(want, float)
    body.append(f"    let r{k}, g{k} = {call};")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE && r{k}.shape == {ilst(want.shape)} && tr_same(r{k}, {lst(want)}));")
    if len(body) >= 60:
        flush()

def arr(k):
    return f"tr_arr({k})"

def nd(x):
    return f"ni_nd({ilst(x.shape)}, {lst(x)})"

print("fn tr_arr(k: i64) -> scientific::Arr {")
print(f"    if k == 1 {{ return {nd(A[1])}; }}")
print(f"    if k == 2 {{ return {nd(A[2])}; }}")
print(f"    return {nd(A[3])};")
print("}\n")

for k in (1, 2, 3):
    for mode in MODES:
        for order in (2, 3, 4, 5):
            if k == 3 and order in (2, 4): continue
            for ax in range(k):
                emit(f'scientific::spline_filter1d({arr(k)}, {order}, {ax}, "{mode}")',
                     ndi.spline_filter1d(A[k], order, ax, mode=mode))
            emit(f'scientific::spline_filter({arr(k)}, {order}, "{mode}")', ndi.spline_filter(A[k], order, mode=mode))
for order in (0, 1):
    emit(f'scientific::spline_filter1d({arr(2)}, {order}, 0, "mirror")', ndi.spline_filter1d(A[2], order, 0))

for k in (1, 2, 3):
    shape = A[k].shape
    for mode in MODES:
        for order in range(6):
            pts = np.array([rng.uniform(-3.5, s + 2.5, size=6) for s in shape])
            pts[:, 0] = [s - 1 for s in shape]
            pts[:, 1] = 0.0
            emit(f'scientific::map_coordinates({arr(k)}, {nd(pts)}, {order}, "{mode}", 0.7)',
                 ndi.map_coordinates(A[k], pts, order=order, mode=mode, cval=0.7))
    emit(f'scientific::map_coordinates({arr(k)}, {nd(pts)}, 3, "mirror", 0.7, false)',
         ndi.map_coordinates(A[k], pts, order=3, mode="mirror", cval=0.7, prefilter=False))
    p2 = np.array([rng.uniform(0, s - 1, size=(2, 3)) for s in shape])
    emit(f'scientific::map_coordinates({arr(k)}, {nd(p2)}, 3, "constant", 0.7)',
         ndi.map_coordinates(A[k], p2, order=3, mode="constant", cval=0.7))

for k in (1, 2, 3):
    for mode in MODES:
        for order in (0, 1, 3, 4):
            sh = np.round(rng.uniform(-2, 2, size=k), 2)
            emit(f'scientific::shift({arr(k)}, {lst(sh)}, {order}, "{mode}", 0.4)',
                 ndi.shift(A[k], sh, order=order, mode=mode, cval=0.4))
        for order in (1, 3, 5):
            zm = np.round(rng.uniform(0.4, 2.2, size=k), 2)
            for gm in (False, True):
                emit(f'scientific::zoom({arr(k)}, {lst(zm)}, {order}, "{mode}", 0.4, true, {"true" if gm else "false"})',
                     ndi.zoom(A[k], zm, order=order, mode=mode, cval=0.4, grid_mode=gm))
        for order in (1, 3):
            m1 = np.round(rng.uniform(0.5, 1.8, size=k), 2)
            off = np.round(rng.uniform(-2, 2, size=k), 2)
            emit(f'scientific::affine_transform({arr(k)}, {nd(m1)}, {lst(off)}, [], {order}, "{mode}", 0.4)',
                 ndi.affine_transform(A[k], m1, off, order=order, mode=mode, cval=0.4))
            m2 = np.round(rng.uniform(-1.2, 1.2, size=(k, k)), 2)
            osh = [int(x) for x in rng.integers(2, 6, size=k)]
            emit(f'scientific::affine_transform({arr(k)}, {nd(m2)}, {lst(off)}, {ilst(osh)}, {order}, "{mode}", 0.4)',
                 ndi.affine_transform(A[k], m2, off, output_shape=osh, order=order, mode=mode, cval=0.4))
            hm = np.vstack([np.hstack([m2, off[:, None]]), np.eye(k + 1)[-1:]])
            emit(f'scientific::affine_transform({arr(k)}, {nd(hm)}, [], [], {order}, "{mode}", 0.4)',
                 ndi.affine_transform(A[k], hm, order=order, mode=mode, cval=0.4))

for k in (2, 3):
    for mode in MODES:
        for order in (1, 3):
            for ang in (17.5, 90.0, -30.0):
                for rs in (True, False):
                    emit(f'scientific::rotate({arr(k)}, {num(ang)}, [1, 0], {"true" if rs else "false"}, {order}, "{mode}", 0.4)',
                         ndi.rotate(A[k], ang, reshape=rs, order=order, mode=mode, cval=0.4))
    emit(f'scientific::rotate({arr(k)}, 33.0, [0, {k - 1}], true, 3, "constant", 0.4)',
         ndi.rotate(A[k], 33.0, axes=(0, k - 1), reshape=True, order=3, mode="constant", cval=0.4))
flush()

for f in funcs:
    print(f)
print("fn test_transform_pins() {")
for i in range(len(funcs)):
    print(f"    test_transform_pins{i}();")
print("}")
