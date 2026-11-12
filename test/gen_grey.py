# Prints the SciPy values that test/run.goose pins for the grey morphology,
# binary_propagation and binary_hit_or_miss. The output is Goose source.
#   python3 test/gen_grey.py > pins.goose
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

rng = np.random.default_rng(306)
count = [0]
funcs = []
body = []

def flush():
    global body
    if body:
        funcs.append(f"fn test_grey_pins{len(funcs)}() {{\n" + "\n".join(body) + "\n}\n")
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

OPS = ['grey_erosion', 'grey_dilation', 'grey_opening', 'grey_closing',
       'morphological_gradient', 'morphological_laplace', 'white_tophat', 'black_tophat']
MODES = ['reflect', 'mirror', 'nearest', 'wrap', 'constant']

def rorigin(fshape):
    return [int(rng.integers(-(f // 2), f - 1 - f // 2 + 1)) for f in fshape]

for trial in range(26):
    nd_ = int(rng.integers(1, 4))
    shape = tuple(int(x) for x in rng.integers(1, 6, size=nd_))
    a = np.round(rng.normal(size=shape) * 3, 1)
    mode = str(rng.choice(MODES))
    cval = float(np.round(rng.normal(), 1))
    fshape = tuple(int(x) for x in rng.integers(1, 4, size=nd_))
    org = rorigin(fshape)
    for op in OPS:
        fn = getattr(ndi, op)
        # size form
        emit(f'scientific::{op}({nd(a)}, {ilst(fshape)}, "{mode}", {num(cval)}, {ilst(org)})',
             fn(a, size=fshape, mode=mode, cval=cval, origin=org))
        # footprint and structure form
        fp = rng.random(fshape) < 0.7
        if not fp.any():
            fp[(0,) * nd_] = True
        st = np.round(rng.normal(size=fshape), 1)
        kind = trial % 3
        if kind == 0:
            emit(f'scientific::{op}_st({nd(a)}, {nd(fp.astype(float))}, scientific::empty_arr(), "{mode}", {num(cval)}, {ilst(org)})',
                 fn(a, footprint=fp, mode=mode, cval=cval, origin=org))
        elif kind == 1:
            emit(f'scientific::{op}_st({nd(a)}, scientific::empty_arr(), {nd(st)}, "{mode}", {num(cval)}, {ilst(org)})',
                 fn(a, structure=st, mode=mode, cval=cval, origin=org))
        else:
            emit(f'scientific::{op}_st({nd(a)}, {nd(fp.astype(float))}, {nd(st)}, "{mode}", {num(cval)}, {ilst(org)})',
                 fn(a, footprint=fp, structure=st, mode=mode, cval=cval, origin=org))

for trial in range(40):
    nd_ = int(rng.integers(1, 4))
    shape = tuple(int(x) for x in rng.integers(1, 7, size=nd_))
    a = rng.random(shape) < 0.3
    m = rng.random(shape) < 0.7
    bv = int(rng.integers(0, 2))
    conn = int(rng.integers(1, nd_ + 1))
    st = ndi.generate_binary_structure(nd_, conn)
    emit(f'scientific::binary_propagation({nd(a.astype(float))}, {conn}, {bv})',
         ndi.binary_propagation(a, st, border_value=bv))
    emit(f'scientific::binary_propagation_st({nd(a.astype(float))}, {nd(st.astype(float))}, {nd(m.astype(float))}, {bv})',
         ndi.binary_propagation(a, st, mask=m, border_value=bv))
    emit(f'scientific::binary_propagation_st({nd(a.astype(float))}, {nd(st.astype(float))}, scientific::empty_arr(), {bv})',
         ndi.binary_propagation(a, st, border_value=bv))

for trial in range(40):
    nd_ = int(rng.integers(1, 4))
    shape = tuple(int(x) for x in rng.integers(1, 6, size=nd_))
    a = rng.random(shape) < 0.5
    fs = tuple(int(x) for x in rng.integers(1, 4, size=nd_))
    s1 = rng.random(fs) < 0.6
    s2 = rng.random(fs) < 0.4
    o1 = rorigin(fs)
    o2 = rorigin(fs)
    emit(f'scientific::binary_hit_or_miss_st({nd(a.astype(float))}, {nd(s1.astype(float))}, {nd(s2.astype(float))}, {ilst(o1)}, {ilst(o2)})',
         ndi.binary_hit_or_miss(a, s1, s2, origin1=o1, origin2=o2))
    emit(f'scientific::binary_hit_or_miss_st({nd(a.astype(float))}, {nd(s1.astype(float))}, scientific::empty_arr(), {ilst(o1)})',
         ndi.binary_hit_or_miss(a, s1, origin1=o1))
    conn = int(rng.integers(1, nd_ + 1))
    emit(f'scientific::binary_hit_or_miss({nd(a.astype(float))}, {conn})',
         ndi.binary_hit_or_miss(a, ndi.generate_binary_structure(nd_, conn)))
flush()
for f in funcs:
    print(f)
print("fn test_grey_pins() {")
for i in range(len(funcs)):
    print(f"    test_grey_pins{i}();")
print("}")
