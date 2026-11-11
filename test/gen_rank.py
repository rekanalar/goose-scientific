# Prints the SciPy values that test/run.goose pins for the rank and
# derivative filters of N dimensions. The output is Goose source.
#   python3 test/gen_rank.py > pins.goose
import numpy as np
from scipy.ndimage import (rank_filter, median_filter, percentile_filter,
                           minimum_filter, maximum_filter, minimum_filter1d,
                           maximum_filter1d, sobel, prewitt, laplace,
                           gaussian_laplace, gaussian_gradient_magnitude)

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
rng = np.random.default_rng(301)
MODES = ["reflect", "constant", "nearest", "mirror", "wrap"]
count = [0]
funcs = []
body = []

def flush():
    global body
    if body:
        funcs.append(f"fn test_rank_pins{len(funcs)}() {{\n" + "\n".join(body) + "\n}\n")
        body = []

def emit(call, want):
    k = count[0]
    count[0] += 1
    body.append(f"    let r{k}, g{k} = {call};")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE && nd_same(r{k}, {lst(want)}));")
    if len(body) >= 70:
        flush()

def arr(k):
    return f"ni_arr({k})"

def fpexpr(fp):
    return f"ni_nd({ilst(fp.shape)}, {lst(fp.astype(float))})"

def origins_of(shape, full):
    k = len(shape)
    out = [[0] * k]
    if full:
        out.append([-(s // 2) for s in shape])
        out.append([s - 1 - s // 2 for s in shape])
    return out

SIZES = {1: [(3,), (4,), (5,)], 2: [(3, 3), (2, 3), (4, 2), (1, 5)], 3: [(3, 2, 3), (2, 2, 2)]}
for k in (1, 2, 3):
    for size in SIZES[k]:
        t = int(np.prod(size))
        for mode in MODES + ["grid-wrap"]:
            for o in origins_of(size, mode in ("reflect", "constant")):
                for rank in sorted({0, t // 2, t - 1, -1, 1 % t}):
                    if mode == "grid-wrap" and rank not in (0, t // 2):
                        continue
                    emit(f'scientific::rank_filter({arr(k)}, {rank}, {ilst(size)}, "{mode}", 0.7, {ilst(o)})',
                         rank_filter(A[k], rank, size=size, mode=mode, cval=0.7, origin=o))
                emit(f'scientific::median_filter({arr(k)}, {ilst(size)}, "{mode}", 0.7, {ilst(o)})',
                     median_filter(A[k], size=size, mode=mode, cval=0.7, origin=o))
                emit(f'scientific::minimum_filter({arr(k)}, {ilst(size)}, "{mode}", 0.7, {ilst(o)})',
                     minimum_filter(A[k], size=size, mode=mode, cval=0.7, origin=o))
                emit(f'scientific::maximum_filter({arr(k)}, {ilst(size)}, "{mode}", 0.7, {ilst(o)})',
                     maximum_filter(A[k], size=size, mode=mode, cval=0.7, origin=o))
            for pc in (0.0, 30.0, 50.0, 73.5, 100.0, -20.0):
                if mode in ("reflect", "constant", "wrap"):
                    emit(f'scientific::percentile_filter({arr(k)}, {num(pc)}, {ilst(size)}, "{mode}", 0.7)',
                         percentile_filter(A[k], pc, size=size, mode=mode, cval=0.7))

# footprints, with holes for two and three dimensions
for shape in [(3, 3), (2, 3), (3, 2, 3), (4,), (3,)]:
    k = len(shape)
    for mode in MODES:
        fp = rng.random(shape) > (0.35 if k > 1 else -1.0)
        fp.flat[0] = True
        if k > 1:
            fp.flat[fp.size // 2] = True
        t = int(fp.sum())
        for o in origins_of(shape, mode in ("reflect", "wrap")):
            for rank in sorted({0, t // 2, t - 1}):
                emit(f'scientific::rank_filter_fp({arr(k)}, {rank}, {fpexpr(fp)}, "{mode}", -0.4, {ilst(o)})',
                     rank_filter(A[k], rank, footprint=fp, mode=mode, cval=-0.4, origin=o))
            emit(f'scientific::median_filter_fp({arr(k)}, {fpexpr(fp)}, "{mode}", -0.4, {ilst(o)})',
                 median_filter(A[k], footprint=fp, mode=mode, cval=-0.4, origin=o))
            emit(f'scientific::minimum_filter_fp({arr(k)}, {fpexpr(fp)}, "{mode}", -0.4, {ilst(o)})',
                 minimum_filter(A[k], footprint=fp, mode=mode, cval=-0.4, origin=o))
            emit(f'scientific::maximum_filter_fp({arr(k)}, {fpexpr(fp)}, "{mode}", -0.4, {ilst(o)})',
                 maximum_filter(A[k], footprint=fp, mode=mode, cval=-0.4, origin=o))
        emit(f'scientific::percentile_filter_fp({arr(k)}, 60.0, {fpexpr(fp)}, "{mode}", -0.4)',
             percentile_filter(A[k], 60.0, footprint=fp, mode=mode, cval=-0.4))

# one axis
for mode in MODES + ["grid-mirror"]:
    for size in (1, 2, 3, 4, 7, 12):
        for o in sorted({-(size // 2), 0, size - 1 - size // 2}):
            if mode not in ("reflect", "constant") and o != 0:
                continue
            k, axis = [(1, -1), (2, 1), (2, 0), (3, 2), (3, 1)][int(rng.integers(0, 5))]
            emit(f'scientific::minimum_filter1d({arr(k)}, {size}, {axis}, "{mode}", 0.3, {o})',
                 minimum_filter1d(A[k], size, axis=axis, mode=mode, cval=0.3, origin=o))
            emit(f'scientific::maximum_filter1d({arr(k)}, {size}, {axis}, "{mode}", 0.3, {o})',
                 maximum_filter1d(A[k], size, axis=axis, mode=mode, cval=0.3, origin=o))

# derivatives
for mode in MODES:
    for k, axes in [(1, [0, -1]), (2, [0, 1, -1]), (3, [0, 1, 2, -1])]:
        for ax in axes:
            emit(f'scientific::sobel({arr(k)}, {ax}, "{mode}", 0.5)', sobel(A[k], axis=ax, mode=mode, cval=0.5))
            emit(f'scientific::prewitt({arr(k)}, {ax}, "{mode}", 0.5)', prewitt(A[k], axis=ax, mode=mode, cval=0.5))
    for k in (1, 2, 3):
        emit(f'scientific::laplace({arr(k)}, "{mode}", 0.5)', laplace(A[k], mode=mode, cval=0.5))
    for k, sg in [(1, [1.0]), (2, [0.8, 1.3]), (3, [0.7, 1.0, 1.4]), (3, [1.2])]:
        s = sg if len(sg) > 1 else sg[0]
        emit(f'scientific::gaussian_laplace({arr(k)}, {lst(sg)}, "{mode}", 0.5, 3.0)',
             gaussian_laplace(A[k], s, mode=mode, cval=0.5, truncate=3.0))
        emit(f'scientific::gaussian_gradient_magnitude({arr(k)}, {lst(sg)}, "{mode}", 0.5, 3.0)',
             gaussian_gradient_magnitude(A[k], s, mode=mode, cval=0.5, truncate=3.0))
flush()

for f in funcs:
    print(f)
print("fn test_rank_pins() {")
for i in range(len(funcs)):
    print(f"    test_rank_pins{i}();")
print("}")
