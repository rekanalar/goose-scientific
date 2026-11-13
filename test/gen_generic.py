# Prints the SciPy values that test/suite/generic_pins_tests.goose pins for the filters that take
# a block. The output is Goose source.
#   python3 test/gen_generic.py > pins.goose
import numpy as np
from scipy.ndimage import generic_filter, generic_laplace, generic_gradient_magnitude, correlate1d

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

rng = np.random.default_rng(303)
A = {1: np.round(rng.normal(size=9), 3),
     2: np.round(rng.normal(size=(5, 6)), 3),
     3: np.round(rng.normal(size=(3, 4, 5)), 3)}
# The arrays of ni_arr in test/suite/generic_pins_tests.goose come from the generator of the rank
# filters, so the pins read them from a seed of that generator.
rng0 = np.random.default_rng(300)
A = {1: np.round(rng0.normal(size=9), 3),
     2: np.round(rng0.normal(size=(5, 6)), 3),
     3: np.round(rng0.normal(size=(3, 4, 5)), 3)}
MODES = ["reflect", "constant", "nearest", "mirror", "wrap"]
FUNCS = {"gsum": lambda w: float(np.sum(w)),
         "grange": lambda w: float(np.max(w) - np.min(w)),
         "gwt": lambda w: float(np.sum(w * np.arange(1, len(w) + 1)))}
count = [0]
funcs = []
body = []

def flush():
    global body
    if body:
        funcs.append(f"fn test_generic_pins{len(funcs)}() {{\n" + "\n".join(body) + "\n}\n")
        body = []

def emit(call, want):
    k = count[0]
    count[0] += 1
    body.append(f"    let r{k}, g{k} = {call};")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE && nd_same(r{k}, {lst(want)}));")
    if len(body) >= 60:
        flush()

def arr(k):
    return f"ni_arr({k})"

def fpexpr(fp):
    return f"ni_nd({ilst(fp.shape)}, {lst(fp.astype(float))})"

SIZES = {1: [(3,), (4,)], 2: [(3, 3), (2, 3), (1, 4)], 3: [(2, 2, 3)]}
for k in (1, 2, 3):
    for size in SIZES[k]:
        for mode in MODES:
            for fname, fn in FUNCS.items():
                orgs = [[0] * k]
                if mode in ("reflect", "constant"):
                    orgs += [[-(s // 2) for s in size], [s - 1 - s // 2 for s in size]]
                for o in orgs:
                    emit(f'scientific::generic_filter({arr(k)}, {ilst(size)}, "{mode}", 0.7, {ilst(o)}) {{ w => {fname}(w) }}',
                         generic_filter(A[k], fn, size=size, mode=mode, cval=0.7, origin=o))
for shape in [(3, 3), (2, 3), (3, 2, 3), (4,)]:
    k = len(shape)
    for mode in MODES:
        fp = rng.random(shape) > (0.35 if k > 1 else -1.0)
        fp.flat[0] = True
        if k > 1:
            fp.flat[fp.size // 2] = True
        for fname, fn in FUNCS.items():
            emit(f'scientific::generic_filter_fp({arr(k)}, {fpexpr(fp)}, "{mode}", -0.4) {{ w => {fname}(w) }}',
                 generic_filter(A[k], fn, footprint=fp, mode=mode, cval=-0.4))

for k in (1, 2, 3):
    for mode in MODES:
        emit(f'scientific::generic_laplace({arr(k)}, "{mode}", 0.5) {{ x, ax, md, cv => gsecond(copy(x), ax, md, cv) }}',
             generic_laplace(A[k], lambda x, ax, o, m, c: correlate1d(x, [1, -2, 1], ax, o, m, c, 0), mode=mode, cval=0.5))
        emit(f'scientific::generic_gradient_magnitude({arr(k)}, "{mode}", 0.5) {{ x, ax, md, cv => gfirst(copy(x), ax, md, cv) }}',
             generic_gradient_magnitude(A[k], lambda x, ax, o, m, c: correlate1d(x, [-1, 0, 1], ax, o, m, c, 0), mode=mode, cval=0.5))
flush()

print('''fn gsum(w: f64[:]) -> f64 {
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < w.len { s += w[i]; i += 1; }
    return s;
}

fn grange(w: f64[:]) -> f64 {
    var lo = w[0];
    var hi = w[0];
    var i: i64 = 1;
    while i < w.len {
        if w[i] < lo { lo = w[i]; }
        if w[i] > hi { hi = w[i]; }
        i += 1;
    }
    return hi - lo;
}

fn gwt(w: f64[:]) -> f64 {
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < w.len { s += w[i] * ((i + 1) as f64); i += 1; }
    return s;
}

fn gsecond(a: scientific::Arr, axis: i64, mode: const u8[:], cval: f64) -> scientific::Arr {
    let d, g = scientific::correlate1d(a, [1.0, -2.0, 1.0], axis, mode, cval, 0);
    assert(g.kind == scientific::F_NONE);
    return d;
}

fn gfirst(a: scientific::Arr, axis: i64, mode: const u8[:], cval: f64) -> scientific::Arr {
    let d, g = scientific::correlate1d(a, [-1.0, 0.0, 1.0], axis, mode, cval, 0);
    assert(g.kind == scientific::F_NONE);
    return d;
}
''')
for f in funcs:
    print(f)
print("fn test_generic_pins() {")
for i in range(len(funcs)):
    print(f"    test_generic_pins{i}();")
print("}")
