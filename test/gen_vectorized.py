# Prints the SciPy values that test/run.goose pins for vectorized_filter.
# The output is Goose source.
#   python3 test/gen_vectorized.py > pins.goose
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

rng = np.random.default_rng(309)
count = [0]
funcs = []
body = []

def flush():
    global body
    if body:
        funcs.append(f"fn test_vectorized_pins{len(funcs)}() {{\n" + "\n".join(body) + "\n}\n")
        body = []

def nd(x):
    return f"ni_nd({ilst(np.asarray(x).shape)}, {lst(x)})"

def emit(a, size, mode, cval, origin):
    # The block sums the window axes. It holds the shape of the windowed
    # array, so it knows the count of the axes and the count of the window.
    block = ("{ x, o => { "
             "var h = x.ndim / 2; var d = 0; var n = 1; while d < h { n *= x.shape[d]; d += 1; } "
             "var w = 1; d = h; while d < x.ndim { w *= x.shape[d]; d += 1; } "
             "var p = 0; while p < n { var s = 0.0; var j = 0; while j < w { s += x.data[p * w + j]; j += 1; } o[p] = s; p += 1; } "
             "} }")
    k = count[0]; count[0] += 1
    body.append(f"    let r{k}, g{k} = scientific::vectorized_filter({nd(a)}, {ilst(size)}, \"{mode}\", {num(cval)}, {ilst(origin)}) {block};")
    def fnc(x, axis=None):
        d = x.ndim // 2
        return np.sum(x, axis=tuple(range(d, x.ndim)))
    if mode == 'constant':
        want = ndi.vectorized_filter(np.asarray(a, float), fnc, size=list(size), mode=mode, cval=cval, origin=list(origin))
    else:
        want = ndi.vectorized_filter(np.asarray(a, float), fnc, size=list(size), mode=mode, origin=list(origin))
    body.append(f"    assert(g{k}.kind == scientific::F_NONE && r{k}.shape == {ilst(want.shape)} && tr_same(r{k}, {lst(want)}));")
    if len(body) >= 40: flush()

for t in range(40):
    nd_ = int(rng.integers(1, 4))
    shape = tuple(int(x) for x in rng.integers(2, 7, size=nd_))
    a = np.round(rng.random(shape) * 10 - 5, 3)
    size = tuple(int(rng.choice([1, 3, 5, 2])) for _ in range(nd_))
    mode = str(rng.choice(['constant', 'nearest', 'mirror', 'reflect']))
    cval = 0.0
    if mode == 'constant':
        cval = float(np.round(rng.random() * 4 - 2, 2))
    origin = tuple(int(rng.choice([0, 0, 1])) if (size[d] % 2 == 1 and size[d] > 1) else 0 for d in range(nd_))
    emit(a, size, mode, cval, origin)

flush()
for f in funcs:
    print(f)
print("fn test_vectorized_pins() {")
for i in range(len(funcs)):
    print(f"    test_vectorized_pins{i}();")
print("}")
