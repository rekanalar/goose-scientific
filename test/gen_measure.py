# Prints the SciPy values that test/suite/measure_pins_tests.goose pins for the measurements of
# labeled regions. The output is Goose source.
#   python3 test/gen_measure.py > pins.goose
import numpy as np
from scipy import ndimage as ndi

def num(x):
    x = float(x)
    if x != x:
        return "(0.0 / 0.0)"
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

rng = np.random.default_rng(305)
count = [0]
funcs = []
body = []

def flush():
    global body
    if body:
        funcs.append(f"fn test_measure_pins{len(funcs)}() {{\n" + "\n".join(body) + "\n}\n")
        body = []

def emit(call, want):
    k = count[0]
    count[0] += 1
    want = np.asarray(want, float)
    body.append(f"    let r{k}, g{k} = {call};")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE && r{k}.shape == {ilst(want.shape)} && ms_same(r{k}, {lst(want)}));")
    if len(body) >= 60:
        flush()

def nd(x):
    return f"ni_nd({ilst(x.shape)}, {lst(x)})"

CASES = []
for shape in ((9,), (4, 5), (3, 2, 4)):
    a = np.round(rng.normal(size=shape) * 3, 0)           # whole numbers make ties
    b = np.round(rng.normal(size=shape), 3)
    l = rng.integers(-1, 5, size=shape)
    CASES.append((a, l))
    CASES.append((b, l))
IDX = [[1, 2, 3], [3, 1], [2, 2, 4], [1, 7, -1, 0, 2], [4], [3, 3, 3]]

def call(name, a, l, idx):
    return f'scientific::{name}({nd(a)}, {nd(l.astype(float))}, {ilst(idx)})'

for a, l in CASES:
    for idx in [None] + IDX:
        ix = [] if idx is None else idx
        ref_idx = None if idx is None else idx
        def one(f, **kw):
            r = f(a, l, ref_idx, **kw) if f is not ndi.sum_labels else f(a, l, ref_idx)
            return np.atleast_1d(np.asarray(r, float))
        emit(call("sum_labels", a, l, ix), one(ndi.sum_labels))
        emit(call("mean_labels", a, l, ix), one(ndi.mean))
        emit(call("variance_labels", a, l, ix), one(ndi.variance))
        emit(call("standard_deviation_labels", a, l, ix), one(ndi.standard_deviation))
        emit(call("minimum_labels", a, l, ix), one(ndi.minimum))
        emit(call("maximum_labels", a, l, ix), one(ndi.maximum))
        # SciPy gives an arbitrary value for the median of a label that is
        # absent from the labels, so the pins keep to present labels.
        if all(int(v) in set(l.reshape(-1).tolist()) for v in ix):
            emit(call("median_labels", a, l, ix), one(ndi.median))
        mp = ndi.minimum_position(a, l, ref_idx)
        xp = ndi.maximum_position(a, l, ref_idx)
        cm = ndi.center_of_mass(a, l, ref_idx)
        def rows(v):
            if ref_idx is None:
                return np.array([v], float)
            return np.array(v, float).reshape(len(ix), a.ndim)
        # With an index list SciPy sorts the values with an unstable sort, so
        # a tie for a place depends on that sort. The pins keep to cases
        # where the extreme of each label is unique.
        def unique_extremes():
            if ref_idx is None:
                return True
            for v in ix:
                vals = a[l == v]
                if len(vals) and ((vals == vals.min()).sum() > 1 or (vals == vals.max()).sum() > 1):
                    return False
            return True
        if unique_extremes():
            emit(call("minimum_position", a, l, ix), rows(mp))
            emit(call("maximum_position", a, l, ix), rows(xp))
        emit(call("center_of_mass", a, l, ix), rows(cm))
    h = ndi.histogram(a, -2.0, 3.0, 5, l, [1, 3])
    emit(f'scientific::histogram_labels({nd(a)}, -2.0, 3.0, 5, {nd(l.astype(float))}, [1, 3])', np.array([x for x in h], float))
    emit(f'scientific::histogram_labels({nd(a)}, -2.0, 3.0, 5, {nd(l.astype(float))})', np.asarray(ndi.histogram(a, -2.0, 3.0, 5, l), float))
    emit(f'scientific::histogram_labels({nd(a)}, -2.0, 3.0, 5, scientific::empty_arr())', np.asarray(ndi.histogram(a, -2.0, 3.0, 5), float))
    emit(f'scientific::histogram_labels({nd(a)}, 0.5, 0.5, 3, {nd(l.astype(float))}, [1])', np.array([ndi.histogram(a, 0.5, 0.5, 3, l, [1])[0]], float))
    emit(f'scientific::sum_labels({nd(a)}, scientific::empty_arr())', np.array([ndi.sum_labels(a)], float))
    emit(f'scientific::mean_labels({nd(a)}, scientific::empty_arr())', np.array([ndi.mean(a)], float))
    emit(f'scientific::variance_labels({nd(a)}, scientific::empty_arr())', np.array([ndi.variance(a)], float))
    emit(f'scientific::median_labels({nd(a)}, scientific::empty_arr())', np.array([ndi.median(a)], float))
    emit(f'scientific::center_of_mass({nd(a)}, scientific::empty_arr())', np.array([ndi.center_of_mass(a)], float))
    emit(f'scientific::minimum_position({nd(a)}, scientific::empty_arr())', np.array([ndi.minimum_position(a)], float))
    fo = ndi.find_objects(l)
    mx = max(int(l.max()), 0)
    box = np.full((mx, 2 * a.ndim), -1.0)
    for i, sl in enumerate(fo):
        if sl is not None:
            box[i] = [s.start for s in sl] + [s.stop for s in sl]
    emit(f'scientific::find_objects({nd(l.astype(float))})', box)
    fo = ndi.find_objects(l, 6)
    box = np.full((6, 2 * a.ndim), -1.0)
    for i, sl in enumerate(fo):
        if sl is not None:
            box[i] = [s.start for s in sl] + [s.stop for s in sl]
    emit(f'scientific::find_objects({nd(l.astype(float))}, 6)', box)
    emit(f'scientific::labeled_comprehension({nd(a)}, {nd(l.astype(float))}, [1, 2, 9], -7.5) {{ v => msrange(v) }}',
         ndi.labeled_comprehension(a, l, [1, 2, 9], lambda v: v.max() - v.min(), float, -7.5))
    emit(f'scientific::labeled_comprehension({nd(a)}, {nd(l.astype(float))}) {{ v => msrange(v) }}',
         np.array([ndi.labeled_comprehension(a, l, None, lambda v: v.max() - v.min(), float, 0.0)]))
flush()

print('''// ms_same compares the data of an array with the values of SciPy. A NaN
// in the pin needs a NaN in the result.
fn ms_same(a: scientific::Arr&, want: f64[:]) -> bool {
    if a.data.len != want.len { return false; }
    var i: i64 = 0;
    while i < want.len {
        let x = a.data[i];
        let y = want[i];
        if y != y {
            if x == x { return false; }
        } else if !gclose(x, y, 1.0e-12) {
            return false;
        }
        i += 1;
    }
    return true;
}

fn msrange(w: f64[:]) -> f64 {
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
''')
for f in funcs:
    print(f)
print("fn test_measure_pins() {")
for i in range(len(funcs)):
    print(f"    test_measure_pins{i}();")
print("}")
