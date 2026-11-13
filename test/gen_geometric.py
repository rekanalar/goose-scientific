# Prints the SciPy values that test/suite/geometric_pins_tests.goose pins for geometric_transform
# and generic_filter1d. The output is Goose source.
#   python3 test/gen_geometric.py > pins.goose
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

rng = np.random.default_rng(308)
count = [0]
funcs = []
body = []

def flush():
    global body
    if body:
        funcs.append(f"fn test_geometric_pins{len(funcs)}() {{\n" + "\n".join(body) + "\n}\n")
        body = []

def nd(x):
    return f"ni_nd({ilst(np.asarray(x).shape)}, {lst(x)})"

def emit_geo(a, oshape, order, mode, cval, m, off):
    # affine mapping: o[d] = m[d]*c[d] + off[d], one cross term on some cases
    lines = []
    for d in range(len(m)):
        lines.append(f"o[{d}] = {num(m[d])} * c[{d}] + {num(off[d])};")
    block = "{ c, o => { " + " ".join(lines) + " } }"
    sh = ilst(oshape) if oshape != a.shape else "[]"
    k = count[0]; count[0] += 1
    body.append(f"    let r{k}, g{k} = scientific::geometric_transform({nd(a)}, {sh}, {order}, \"{mode}\", {num(cval)}, true) {block};")
    want = ndi.geometric_transform(np.asarray(a, float), lambda c: tuple(m[d] * c[d] + off[d] for d in range(len(c))),
                                   output_shape=oshape if oshape != a.shape else None,
                                   order=order, mode=mode, cval=cval)
    body.append(f"    assert(g{k}.kind == scientific::F_NONE && r{k}.shape == {ilst(want.shape)} && tr_same(r{k}, {lst(want)}));")
    if len(body) >= 40: flush()

for t in range(48):
    nd_ = int(rng.integers(1, 4))
    shape = tuple(int(x) for x in rng.integers(2, 7, size=nd_))
    a = np.round(rng.random(shape) * 10 - 5, 3)
    order = int(rng.integers(0, 4))
    mode = str(rng.choice(['constant', 'nearest', 'mirror', 'reflect']))
    cval = float(np.round(rng.random() * 4 - 2, 2)) if mode == 'constant' else 0.0
    m = [round(float(x), 2) for x in rng.random(nd_) * 1.6 - 0.3]
    if m == [1.0] * nd_: m = [0.9] * nd_
    off = [round(float(x), 2) for x in rng.random(nd_) * 2 - 1]
    emit_geo(a, shape, order, mode, cval, m, off)
    if t % 3 == 0:
        oshape = tuple(int(x) for x in rng.integers(2, 6, size=nd_))
        emit_geo(a, oshape, order, mode, cval, m, off)

def emit_line(a, size, axis, mode, cval, origin):
    w = " + ".join(f"line[j + {i}] * {float(i + 1)}" for i in range(size))
    block = f"{{ line, out => {{ var j = 0; while j < out.len {{ out[j] = {w}; j += 1; }} }} }}"
    k = count[0]; count[0] += 1
    body.append(f"    let r{k}, g{k} = scientific::generic_filter1d({nd(a)}, {size}, {axis}, \"{mode}\", {num(cval)}, {origin}) {block};")
    def fnc(il, ol):
        n = len(ol)
        for j in range(n):
            s = 0.0
            for i in range(size):
                s += il[j + i] * (i + 1)
            ol[j] = s
    want = ndi.generic_filter1d(np.asarray(a, float), fnc, size, axis=axis, mode=mode, cval=cval, origin=origin)
    body.append(f"    assert(g{k}.kind == scientific::F_NONE && r{k}.shape == {ilst(want.shape)} && tr_same(r{k}, {lst(want)}));")
    if len(body) >= 40: flush()

for t in range(48):
    nd_ = int(rng.integers(1, 4))
    shape = tuple(int(x) for x in rng.integers(2, 7, size=nd_))
    a = np.round(rng.random(shape) * 10 - 5, 3)
    size = int(rng.choice([3, 5, 3, 5, 2]))
    axis = int(rng.integers(-nd_, nd_))
    mode = str(rng.choice(['constant', 'nearest', 'mirror', 'reflect']))
    cval = float(np.round(rng.random() * 4 - 2, 2)) if mode == 'constant' else 0.0
    origin = int(rng.choice([0, 0, 1])) if size % 2 == 1 else 0
    emit_line(a, size, axis, mode, cval, origin)

flush()
for f in funcs:
    print(f)
print("fn test_geometric_pins() {")
for i in range(len(funcs)):
    print(f"    test_geometric_pins{i}();")
print("}")
