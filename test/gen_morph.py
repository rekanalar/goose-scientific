# Prints the SciPy values that test/run.goose pins for the labels, the binary
# morphology and the distance transform of arrays of N dimensions.
# The output is Goose source.
#   python3 test/gen_morph.py > pins.goose
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

rng = np.random.default_rng(302)
A = {1: (rng.random(12) > 0.4),
     2: (rng.random((6, 7)) > 0.45),
     3: (rng.random((3, 4, 5)) > 0.4)}
A[2][2:4, 2:5] = True
A[2][3, 3] = False
A[2][0, :] = False
A[2][:, 0] = False
count = [0]
funcs = []
body = []

def flush():
    global body
    if body:
        funcs.append(f"fn test_morph_pins{len(funcs)}() {{\n" + "\n".join(body) + "\n}\n")
        body = []

def emit(call, want):
    k = count[0]
    count[0] += 1
    body.append(f"    let r{k}, g{k} = {call};")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE && nd_same(r{k}, {lst(want)}));")
    if len(body) >= 70:
        flush()

def emit_label(call, want, n):
    k = count[0]
    count[0] += 1
    body.append(f"    let r{k}, n{k}, g{k} = {call};")
    body.append(f"    assert(g{k}.kind == scientific::F_NONE && n{k} == {n} && nd_same(r{k}, {lst(want)}));")

def arr(k):
    return f"mo_arr({k})"

def nd(x):
    return f"ni_nd({ilst(x.shape)}, {lst(x.astype(float))})"

print("fn mo_arr(k: i64) -> scientific::Arr {")
print(f"    if k == 1 {{ return {nd(A[1])}; }}")
print(f"    if k == 2 {{ return {nd(A[2])}; }}")
print(f"    return {nd(A[3])};")
print("}\n")

for k in (1, 2, 3):
    for c in (1, 2, 3):
        st = ndi.generate_binary_structure(k, c)
        emit(f'scientific::generate_binary_structure({k}, {c})', st)
    emit_label(f'scientific::label({arr(k)}, 1)', *ndi.label(A[k]))
    for c in (2, 3):
        if c <= k:
            emit_label(f'scientific::label({arr(k)}, {c})', *ndi.label(A[k], ndi.generate_binary_structure(k, c)))
    for it in (1, 2, 3):
        emit(f'scientific::iterate_structure({nd(ndi.generate_binary_structure(k, 1))}, {it})',
             ndi.iterate_structure(ndi.generate_binary_structure(k, 1), it))
    for c in (1, 2, 3):
        st = ndi.generate_binary_structure(k, min(c, k))
        for it in (1, 2, 0):
            for bv in (0, 1):
                for name, fn in (("erosion", ndi.binary_erosion), ("dilation", ndi.binary_dilation),
                                 ("opening", ndi.binary_opening), ("closing", ndi.binary_closing)):
                    if name in ("opening", "closing") and bv == 1 and it == 0:
                        continue
                    emit(f'scientific::binary_{name}({arr(k)}, {c}, {it}, {bv})',
                         fn(A[k], st, iterations=it, border_value=bv))
    emit(f'scientific::binary_fill_holes({arr(k)}, 1)', ndi.binary_fill_holes(A[k]))
    emit(f'scientific::binary_fill_holes({arr(k)}, {min(2, k)})',
         ndi.binary_fill_holes(A[k], ndi.generate_binary_structure(k, min(2, k))))
    # structures with an even or odd size, origins and a mask
    for shape in ([(2,), (3,), (4,)] if k == 1 else [(2, 3), (3, 3), (2, 2)] if k == 2 else [(2, 2, 3), (3, 3, 3)]):
        st = rng.random(shape) > 0.3
        st.flat[0] = True
        for og in ([[0] * k, [-(s // 2) for s in shape], [s - 1 - s // 2 for s in shape]]):
            mask = rng.random(A[k].shape) > 0.3
            for name, fn in (("erosion", ndi.binary_erosion), ("dilation", ndi.binary_dilation),
                             ("opening", ndi.binary_opening), ("closing", ndi.binary_closing)):
                for bv in (0, 1):
                    emit(f'scientific::binary_{name}_st({arr(k)}, {nd(st)}, scientific::empty_arr(), 1, {bv}, {ilst(og)})',
                         fn(A[k], st, border_value=bv, origin=og))
                emit(f'scientific::binary_{name}_st({arr(k)}, {nd(st)}, {nd(mask)}, 2, 1, {ilst(og)})',
                     fn(A[k], st, iterations=2, mask=mask, border_value=1, origin=og)) if name in ("erosion", "dilation") else None
    sym = rng.random((3,) * k) > 0.4
    sym = sym | sym[(slice(None, None, -1),) * k]
    sym.flat[sym.size // 2] = True
    emit_label(f'scientific::label_st({arr(k)}, {nd(sym)})', *ndi.label(A[k], sym))
    emit(f'scientific::binary_fill_holes_st({arr(k)}, {nd(sym)})', ndi.binary_fill_holes(A[k], sym))
    # distances
    emit(f'scientific::distance_transform_edt({arr(k)})', ndi.distance_transform_edt(A[k]))
    sp = [0.5 + 0.7 * i for i in range(k)]
    emit(f'scientific::distance_transform_edt({arr(k)}, {lst(sp)})', ndi.distance_transform_edt(A[k], sampling=sp))
    emit(f'scientific::distance_transform_edt({arr(k)}, [1.5])', ndi.distance_transform_edt(A[k], sampling=1.5))
    d, ix = ndi.distance_transform_edt(A[k], return_indices=True)
    kk = count[0]
    count[0] += 1
    body.append(f"    let d{kk}, x{kk}, g{kk} = scientific::distance_transform_edt_indices({arr(k)});")
    body.append(f"    assert(g{kk}.kind == scientific::F_NONE && nd_same(d{kk}, {lst(d)}) && nd_same(x{kk}, {lst(ix)}));")
flush()

for f in funcs:
    print(f)
print("fn test_morph_pins() {")
for i in range(len(funcs)):
    print(f"    test_morph_pins{i}();")
print("}")
