# Prints the SciPy values that test/run.goose pins for the linear filters of
# N dimensions. The output is Goose source: the bodies of the pinned tests.
# Run it with SciPy and paste the output between the markers of run.goose:
#   python3 test/gen_ndimage.py > pins.goose
import numpy as np
from scipy.ndimage import (correlate1d, convolve1d, uniform_filter1d,
                           gaussian_filter1d, correlate, convolve,
                           uniform_filter, gaussian_filter)

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

rng = np.random.default_rng(300)
A = {1: np.round(rng.normal(size=9), 3),
     2: np.round(rng.normal(size=(5, 6)), 3),
     3: np.round(rng.normal(size=(3, 4, 5)), 3)}

MODES = ["reflect", "constant", "nearest", "mirror", "wrap"]
ALIAS = ["grid-mirror", "grid-constant", "grid-wrap"]
count = [0]
funcs = []
body = []

def emit(call, want):
    k = count[0]
    count[0] += 1
    body.append(f"    let r{k}, g{k} = {call};")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE && nd_same(r{k}, {lst(want)}));")
    if len(body) >= 70:
        flush()

def flush():
    global body
    if not body:
        return
    n = len(funcs)
    funcs.append(f"fn test_ndimage_pins{n}() {{\n" + "\n".join(body) + "\n}\n")
    body = []

def arr(k):
    return f"ni_arr({k})"

# one axis: correlate1d and convolve1d
for mode in MODES + ALIAS:
    for L in [1, 2, 3, 4, 5]:
        lo, hi = -(L // 2), (L - 1) // 2
        origins = sorted({lo, 0, hi}) if mode in MODES else [0]
        if mode in ALIAS and L not in (3, 4):
            continue
        w = np.round(rng.normal(size=L), 3)
        for o in origins:
            for k, axis in [(1, -1), (2, 0), (2, 1), (3, 1), (3, -1)]:
                if rng.random() > 0.35 and not (k == 1):
                    continue
                cv = 1.5
                emit(f'scientific::correlate1d({arr(k)}, {lst(w)}, {axis}, "{mode}", {num(cv)}, {o})',
                     correlate1d(A[k], w, axis=axis, mode=mode, cval=cv, origin=o))
                emit(f'scientific::convolve1d({arr(k)}, {lst(w)}, {axis}, "{mode}", {num(cv)}, {o})',
                     convolve1d(A[k], w, axis=axis, mode=mode, cval=cv, origin=o))

# uniform_filter1d
for mode in MODES + ALIAS:
    for size in [1, 2, 3, 4, 5, 8, 12]:
        lo, hi = -(size // 2), size - 1 - size // 2
        for o in sorted({lo, 0, hi}):
            if mode in ALIAS and (size not in (3, 4) or o != 0):
                continue
            if mode in MODES and mode not in ("reflect", "constant", "wrap") and o != 0:
                continue
            k, axis = [(1, -1), (2, 1), (2, 0), (3, 2)][int(rng.integers(0, 4))]
            emit(f'scientific::uniform_filter1d({arr(k)}, {size}, {axis}, "{mode}", -0.75, {o})',
                 uniform_filter1d(A[k], size, axis=axis, mode=mode, cval=-0.75, origin=o))

# gaussian_filter1d
for mode in MODES + ALIAS:
    for sigma in [0.5, 1.0, 2.3]:
        for order in [0, 1, 2, 3]:
            if mode in ALIAS and (order > 1 or sigma != 1.0):
                continue
            tr = 4.0 if sigma != 2.3 else 2.0
            k, axis = [(1, -1), (2, 1), (2, 0), (3, 0)][int(rng.integers(0, 4))]
            emit(f'scientific::gaussian_filter1d({arr(k)}, {num(sigma)}, {axis}, {order}, "{mode}", 0.5, {num(tr)})',
                 gaussian_filter1d(A[k], sigma, axis=axis, order=order, mode=mode, cval=0.5, truncate=tr))

# N dimensions: correlate_nd and convolve_nd
for shape in [(3, 3), (2, 2), (4, 3), (2, 3), (2, 2, 2), (3,), (3, 2, 3)]:
    for mode in MODES + ["grid-wrap"]:
        W = np.round(rng.normal(size=shape), 3)
        k = len(shape)
        ranges = [range(-(s // 2), (s - 1) // 2 + 1) for s in shape]
        origins = [[0] * k]
        if mode in ("reflect", "constant"):
            origins.append([int(r[-1]) for r in ranges])
            origins.append([int(r[0]) for r in ranges])
        for o in origins:
            data = A[k]
            wexpr = f"ni_nd({ilst(shape)}, {lst(W)})"
            emit(f'scientific::correlate_nd({arr(k)}, {wexpr}, "{mode}", 2.0, {ilst(o)})',
                 correlate(data, W, mode=mode, cval=2.0, origin=o))
            emit(f'scientific::convolve_nd({arr(k)}, {wexpr}, "{mode}", 2.0, {ilst(o)})',
                 convolve(data, W, mode=mode, cval=2.0, origin=o))

# uniform_filter and gaussian_filter on all axes
for k, sizes in [(1, [3]), (1, [4]), (2, [3]), (2, [2, 5]), (3, [3, 2, 4]), (3, [1, 3, 5])]:
    for mode in ["reflect", "constant", "wrap", "mirror", "nearest"]:
        s = sizes if len(sizes) > 1 else sizes[0]
        emit(f'scientific::uniform_filter({arr(k)}, {ilst(sizes)}, "{mode}", 0.25)',
             uniform_filter(A[k], size=s, mode=mode, cval=0.25))
for k, sig, order in [(1, [1.0], [0]), (2, [1.0], [0]), (2, [0.8, 1.5], [1, 0]), (3, [0.7, 1.0, 1.4], [0, 1, 2]),
                      (3, [1.0], [2]), (2, [2.0, 0.5], [0, 3])]:
    for mode in ["reflect", "constant", "wrap", "mirror", "nearest"]:
        sg = sig if len(sig) > 1 else sig[0]
        od = order if len(order) > 1 else order[0]
        emit(f'scientific::gaussian_filter({arr(k)}, {lst(sig)}, {ilst(order)}, "{mode}", -0.5, 3.0)',
             gaussian_filter(A[k], sigma=sg, order=od, mode=mode, cval=-0.5, truncate=3.0))
flush()

print("// The inputs of the pinned filter tests. Printed by test/gen_ndimage.py.")
print("fn ni_arr(k: i64) -> scientific::Arr {")
print(f"    if k == 1 {{ return ni_nd([9], {lst(A[1])}); }}")
print(f"    if k == 2 {{ return ni_nd([5, 6], {lst(A[2])}); }}")
print(f"    return ni_nd([3, 4, 5], {lst(A[3])});")
print("}\n")
for f in funcs:
    print(f)
print("fn test_ndimage_pins() {")
for i in range(len(funcs)):
    print(f"    test_ndimage_pins{i}();")
print("}")
