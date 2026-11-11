// Labels, binary morphology and the Euclidean distance transform of arrays
// of N dimensions, as in scipy.ndimage: generate_binary_structure,
// iterate_structure, label, binary_erosion, binary_dilation, binary_opening,
// binary_closing, binary_fill_holes and distance_transform_edt.
//
// A nonzero value is foreground. A mask that a call gives holds 0.0 and
// 1.0. A call with a connectivity uses the structure of that connectivity.
// A call with the suffix _st takes a structure array and a mask. An empty
// array stands for no mask.
//
// Erosion is the base. A value is true when every place of the structure finds
// a true value. Dilation is the erosion of the complement with the structure
// turned on every axis, as in SciPy.
namespace scientific;

import mathshim;

// generate_binary_structure gives the structure of a rank and a
// connectivity. A neighbor joins when it differs from the centre on at most
// connectivity axes. The connectivity is clamped to 1 up to the rank.
fn generate_binary_structure(rank: i64, connectivity: i64) -> Arr, Fault {
    if rank < 1 || rank > MAX_DIMS {
        return empty_arr(), fault(F_VALUE, "generate_binary_structure", "the rank holds a value outside 1 to 8");
    }
    var c = connectivity;
    if c < 1 { c = 1; }
    if c > rank { c = rank; }
    var total: i64 = 1;
    var d: i64 = 0;
    while d < rank { total *= 3; d += 1; }
    var data: f64[>..] = [];
    var q: i64 = 0;
    while q < total {
        var rem = q;
        var apart: i64 = 0;
        d = 0;
        while d < rank {
            if rem % 3 != 1 { apart += 1; }
            rem /= 3;
            d += 1;
        }
        if apart <= c { data.push(1.0); } else { data.push(0.0); }
        q += 1;
    }
    var shape: i64[..8] = [];
    d = 0;
    while d < rank { shape.push(3); d += 1; }
    return arr_of(copy(data), shape), no_fault();
}

// mo_places lists the nonzero places of a structure, one row of N
// coordinates for each place. The flag turns every axis around. It gives
// the count of places.
fn mo_places(st, turn: bool, out: i64[>..]&) -> i64 {
    let nd = st.ndim;
    var ox: i64[..8] = [];
    var d: i64 = 0;
    while d < nd { ox.push(0); d += 1; }
    var t: i64 = 0;
    var q: i64 = 0;
    while q < st.data.len {
        if st.data[q] != 0.0 {
            d = 0;
            while d < nd {
                if turn { out.push(st.shape[d] - 1 - ox[d]); } else { out.push(ox[d]); }
                d += 1;
            }
            t += 1;
        }
        d = nd - 1;
        while d >= 0 {
            ox[d] += 1;
            if ox[d] < st.shape[d] { d = -1; } else { ox[d] = 0; d -= 1; }
        }
        q += 1;
    }
    return t;
}

// mo_erode gives 1.0 where every place of the window finds a value that is
// not zero. A place outside the array reads the border value bv, which is 0.0
// or 1.0. The data hold only 0.0 and 1.0. The buffer out holds zeros.
fn mo_erode(src: f64[:], shape: i64[:], offs: i64[:], t: i64, shift: i64[:], bv: f64, out: f64[>..]&) {
    let nd = shape.len;
    var ix: i64[..8] = [];
    var d: i64 = 0;
    while d < nd { ix.push(0); d += 1; }
    var p: i64 = 0;
    while p < src.len {
        var hit = true;
        var k: i64 = 0;
        while k < t && hit {
            var flat: i64 = 0;
            var outside = false;
            d = 0;
            while d < nd && !outside {
                let s = ix[d] + offs[k * nd + d] - shift[d];
                if s < 0 || s >= shape[d] { outside = true; } else { flat = flat * shape[d] + s; }
                d += 1;
            }
            var v = bv;
            if !outside { v = src[flat]; }
            if v == 0.0 { hit = false; }
            k += 1;
        }
        if hit { out[p] = 1.0; }
        d = nd - 1;
        while d >= 0 {
            ix[d] += 1;
            if ix[d] < shape[d] { d = -1; } else { ix[d] = 0; d -= 1; }
        }
        p += 1;
    }
}

// mo_rounds applies erosion or dilation with a structure. A count of
// iterations of one or more gives that many rounds. A smaller count repeats
// until no value changes. The mask keeps the old value where it is zero. The
// start holds only 0.0 and 1.0. The buffer out receives the result.
fn mo_rounds(start: f64[:], shape: i64[:], st, mask: f64[:], iterations: i64, bv: f64, og: i64[:], dilate: bool, tag: const u8[:], out: f64[>..]&) -> Fault {
    var offs: i64[>..] = [];
    let t = mo_places(st, dilate, offs);
    if t < 1 { return fault(F_VALUE, tag, "the structure holds no nonzero place"); }
    var shift: i64[..8] = [];
    var d: i64 = 0;
    while d < shape.len {
        let fs = st.shape[d];
        if og[d] < -(fs / 2) || og[d] > fs - 1 - fs / 2 {
            return fault(F_RANGE, tag, "the origin stands outside the structure");
        }
        var o = og[d];
        if dilate {
            o = -o;
            if fs % 2 == 0 { o -= 1; }
        }
        shift.push(fs / 2 + o);
        d += 1;
    }
    let n = start.len;
    if n > ND_COST / t { return fault(F_LIMIT, tag, "the cost of the filter is above the limit"); }
    let allowed = ND_COST / (n * t);
    if iterations > allowed { return fault(F_LIMIT, tag, "the cost of the filter is above the limit"); }
    var cur: f64[>..] = [];
    var i: i64 = 0;
    while i < n { cur.push(start[i]); i += 1; }
    var rounds: i64 = 0;
    var done = false;
    while !done {
        if rounds >= allowed { return fault(F_LIMIT, tag, "the cost of the filter is above the limit"); }
        var nxt: f64[>..] = nd_zeros(n);
        if dilate {
            var comp: f64[>..] = nd_zeros(n);
            i = 0;
            while i < n { comp[i] = 1.0 - cur[i]; i += 1; }
            mo_erode(comp, shape, offs, t, shift, 1.0 - bv, nxt);
            i = 0;
            while i < n { nxt[i] = 1.0 - nxt[i]; i += 1; }
        } else {
            mo_erode(cur, shape, offs, t, shift, bv, nxt);
        }
        var changed = false;
        i = 0;
        while i < n {
            if mask.len > 0 && mask[i] == 0.0 { nxt[i] = cur[i]; }
            if nxt[i] != cur[i] { changed = true; }
            i += 1;
        }
        cur = copy(nxt);
        rounds += 1;
        if iterations >= 1 {
            if rounds >= iterations { done = true; }
        } else if !changed {
            done = true;
        }
    }
    i = 0;
    while i < n { out[i] = cur[i]; i += 1; }
    return no_fault();
}

// mo_binary reads the operands of a morphology call and runs the kinds: 0
// for erosion, 1 for dilation, 2 for opening and 3 for closing.
fn mo_binary(a, st, mask, iterations: i64, border_value: i64, origin: i64[:], kind: i64, tag: const u8[:]) -> Arr, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fa.kind != F_NONE { return empty_arr(), fa; }
    let fs = check_stack(st.ndim, st.shape, st.data.len, 1, tag);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    if st.ndim != a.ndim {
        return empty_arr(), fault(F_SHAPE, tag, "the structure holds another count of dimensions than the array");
    }
    var mk: f64[>..] = [];
    if mask.ndim != 0 {
        let fm = check_stack(mask.ndim, mask.shape, mask.data.len, 1, tag);
        if fm.kind != F_NONE { return empty_arr(), fm; }
        if mask.data.len != a.data.len || mask.ndim != a.ndim {
            return empty_arr(), fault(F_SHAPE, tag, "the mask holds another shape than the array");
        }
        var i: i64 = 0;
        while i < mask.data.len { mk.push(mask.data[i]); i += 1; }
    }
    var og: i64[..8] = [];
    if !nd_list(origin, a.ndim, 0, og) {
        return empty_arr(), fault(F_SHAPE, tag, "the origin holds a count that is not one for each axis");
    }
    var bv: f64 = 0.0;
    if border_value != 0 { bv = 1.0; }
    var cur: f64[>..] = [];
    var j: i64 = 0;
    while j < a.data.len {
        if a.data[j] != 0.0 { cur.push(1.0); } else { cur.push(0.0); }
        j += 1;
    }
    var first = kind == 1 || kind == 3;
    var steps: i64 = 1;
    if kind >= 2 { steps = 2; }
    var s: i64 = 0;
    while s < steps {
        var out: f64[>..] = nd_zeros(a.data.len);
        let g = mo_rounds(cur, a.shape, st, mk, iterations, bv, og, first, tag, out);
        if g.kind != F_NONE { return empty_arr(), g; }
        cur = copy(out);
        first = !first;
        s += 1;
    }
    return arr_of(copy(cur), a.shape), no_fault();
}

// mo_default gives the structure of a connectivity for an array.
fn mo_default(a, connectivity: i64) -> Arr, Fault {
    return generate_binary_structure(a.ndim, connectivity);
}

// binary_erosion keeps a value when every neighbor of the structure of the
// connectivity is foreground. The border value is 0 or 1.
fn binary_erosion(a, connectivity: i64 = 1, iterations: i64 = 1, border_value: i64 = 0, origin: i64[:] = []) -> Arr, Fault {
    let st, f = mo_default(a, connectivity);
    if f.kind != F_NONE { return empty_arr(), f; }
    return mo_binary(a, st, empty_arr(), iterations, border_value, origin, 0, "binary_erosion");
}

// binary_dilation sets a value when a neighbor of the structure is foreground.
fn binary_dilation(a, connectivity: i64 = 1, iterations: i64 = 1, border_value: i64 = 0, origin: i64[:] = []) -> Arr, Fault {
    let st, f = mo_default(a, connectivity);
    if f.kind != F_NONE { return empty_arr(), f; }
    return mo_binary(a, st, empty_arr(), iterations, border_value, origin, 1, "binary_dilation");
}

// binary_opening is an erosion and then a dilation.
fn binary_opening(a, connectivity: i64 = 1, iterations: i64 = 1, border_value: i64 = 0, origin: i64[:] = []) -> Arr, Fault {
    let st, f = mo_default(a, connectivity);
    if f.kind != F_NONE { return empty_arr(), f; }
    return mo_binary(a, st, empty_arr(), iterations, border_value, origin, 2, "binary_opening");
}

// binary_closing is a dilation and then an erosion.
fn binary_closing(a, connectivity: i64 = 1, iterations: i64 = 1, border_value: i64 = 0, origin: i64[:] = []) -> Arr, Fault {
    let st, f = mo_default(a, connectivity);
    if f.kind != F_NONE { return empty_arr(), f; }
    return mo_binary(a, st, empty_arr(), iterations, border_value, origin, 3, "binary_closing");
}

// The forms with the suffix _st take a structure array and a mask. An empty
// array as the mask means no mask.
fn binary_erosion_st(a, structure, mask, iterations: i64 = 1, border_value: i64 = 0, origin: i64[:] = []) -> Arr, Fault {
    return mo_binary(a, structure, mask, iterations, border_value, origin, 0, "binary_erosion_st");
}

fn binary_dilation_st(a, structure, mask, iterations: i64 = 1, border_value: i64 = 0, origin: i64[:] = []) -> Arr, Fault {
    return mo_binary(a, structure, mask, iterations, border_value, origin, 1, "binary_dilation_st");
}

fn binary_opening_st(a, structure, mask, iterations: i64 = 1, border_value: i64 = 0, origin: i64[:] = []) -> Arr, Fault {
    return mo_binary(a, structure, mask, iterations, border_value, origin, 2, "binary_opening_st");
}

fn binary_closing_st(a, structure, mask, iterations: i64 = 1, border_value: i64 = 0, origin: i64[:] = []) -> Arr, Fault {
    return mo_binary(a, structure, mask, iterations, border_value, origin, 3, "binary_closing_st");
}

// mo_fill fills the holes that do not touch the border. It grows an empty
// array inside the complement of the input, from a border of ones.
fn mo_fill(a, st, tag: const u8[:]) -> Arr, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fa.kind != F_NONE { return empty_arr(), fa; }
    var comp: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        if a.data[i] != 0.0 { comp.push(0.0); } else { comp.push(1.0); }
        i += 1;
    }
    let seed = nd_zeros(a.data.len);
    var og: i64[..8] = [];
    var d: i64 = 0;
    while d < a.ndim { og.push(0); d += 1; }
    if st.ndim != a.ndim {
        return empty_arr(), fault(F_SHAPE, tag, "the structure holds another count of dimensions than the array");
    }
    let fs = check_stack(st.ndim, st.shape, st.data.len, 1, tag);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    var out: f64[>..] = nd_zeros(a.data.len);
    let g = mo_rounds(seed, a.shape, st, comp, 0, 1.0, og, true, tag, out);
    if g.kind != F_NONE { return empty_arr(), g; }
    i = 0;
    while i < out.len { out[i] = 1.0 - out[i]; i += 1; }
    return arr_of(copy(out), a.shape), no_fault();
}

// binary_fill_holes fills the holes of a mask. A hole is a part of the
// background that the connectivity does not join to the border.
fn binary_fill_holes(a, connectivity: i64 = 1) -> Arr, Fault {
    let st, f = mo_default(a, connectivity);
    if f.kind != F_NONE { return empty_arr(), f; }
    return mo_fill(a, st, "binary_fill_holes");
}

fn binary_fill_holes_st(a, structure) -> Arr, Fault {
    return mo_fill(a, structure, "binary_fill_holes_st");
}

// iterate_structure grows a structure by a dilation with itself, as many
// times as the count of iterations minus one.
fn iterate_structure(structure, iterations: i64) -> Arr, Fault {
    let f = check_stack(structure.ndim, structure.shape, structure.data.len, 1, "iterate_structure");
    if f.kind != F_NONE { return empty_arr(), f; }
    if iterations < 2 { return clone(structure); }
    let ni = iterations - 1;
    var shape: i64[..8] = [];
    var total: i64 = 1;
    var d: i64 = 0;
    while d < structure.ndim {
        let fs = structure.shape[d];
        if ni > (MAX_ELEMENTS - fs) / (fs - 1 + 1) {
            return empty_arr(), fault(F_LIMIT, "iterate_structure", "the grown structure holds more elements than the limit");
        }
        let sz = fs + ni * (fs - 1);
        if total > MAX_ELEMENTS / sz {
            return empty_arr(), fault(F_LIMIT, "iterate_structure", "the grown structure holds more elements than the limit");
        }
        total *= sz;
        shape.push(sz);
        d += 1;
    }
    var big: f64[>..] = nd_zeros(total);
    var ox: i64[..8] = [];
    d = 0;
    while d < structure.ndim { ox.push(0); d += 1; }
    var q: i64 = 0;
    while q < structure.data.len {
        var flat: i64 = 0;
        d = 0;
        while d < structure.ndim {
            flat = flat * shape[d] + ni * (structure.shape[d] / 2) + ox[d];
            d += 1;
        }
        big[flat] = structure.data[q];
        d = structure.ndim - 1;
        while d >= 0 {
            ox[d] += 1;
            if ox[d] < structure.shape[d] { d = -1; } else { ox[d] = 0; d -= 1; }
        }
        q += 1;
    }
    let grown = arr_of(copy(big), shape);
    var mk: f64[>..] = [];
    var og: i64[..8] = [];
    d = 0;
    while d < structure.ndim { og.push(0); d += 1; }
    var out: f64[>..] = nd_zeros(total);
    let g = mo_rounds(grown.data, grown.shape, structure, mk, ni, 0.0, og, true, "iterate_structure", out);
    if g.kind != F_NONE { return empty_arr(), g; }
    return arr_of(copy(out), shape), no_fault();
}

// mo_label finds the connected parts of the nonzero values. A part gets a
// label from 1 in the order of its first value in row-major order.
fn mo_label(a, st, tag: const u8[:]) -> Arr, i64, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fa.kind != F_NONE { return empty_arr(), 0, fa; }
    let fs = check_stack(st.ndim, st.shape, st.data.len, 1, tag);
    if fs.kind != F_NONE { return empty_arr(), 0, fs; }
    if st.ndim != a.ndim {
        return empty_arr(), 0, fault(F_SHAPE, tag, "the structure holds another count of dimensions than the array");
    }
    var d: i64 = 0;
    while d < a.ndim {
        if st.shape[d] != 3 {
            return empty_arr(), 0, fault(F_VALUE, tag, "the structure holds a size other than 3 on an axis");
        }
        d += 1;
    }
    var q: i64 = 0;
    while q < st.data.len {
        let mirror = st.data[st.data.len - 1 - q];
        if (st.data[q] != 0.0) != (mirror != 0.0) {
            return empty_arr(), 0, fault(F_VALUE, tag, "the structure is not symmetric");
        }
        q += 1;
    }
    var offs: i64[>..] = [];
    let t = mo_places(st, false, offs);
    let n = a.data.len;
    var lab: f64[>..] = nd_zeros(n);
    var stack: i64[>..] = [];
    var cx: i64[..8] = [];
    d = 0;
    while d < a.ndim { cx.push(0); d += 1; }
    var count: i64 = 0;
    var p: i64 = 0;
    while p < n {
        if a.data[p] != 0.0 && lab[p] == 0.0 {
            count += 1;
            lab[p] = count as f64;
            stack.push(p);
            while stack.len > 0 {
                let cur = stack[stack.len - 1];
                stack.pop();
                var rem = cur;
                d = a.ndim - 1;
                while d >= 0 { cx[d] = rem % a.shape[d]; rem /= a.shape[d]; d -= 1; }
                var k: i64 = 0;
                while k < t {
                    var flat: i64 = 0;
                    var inside = true;
                    d = 0;
                    while d < a.ndim && inside {
                        let y = cx[d] + offs[k * a.ndim + d] - 1;
                        if y < 0 || y >= a.shape[d] { inside = false; } else { flat = flat * a.shape[d] + y; }
                        d += 1;
                    }
                    if inside && a.data[flat] != 0.0 && lab[flat] == 0.0 {
                        lab[flat] = count as f64;
                        stack.push(flat);
                    }
                    k += 1;
                }
            }
        }
        p += 1;
    }
    return arr_of(copy(lab), a.shape), count, no_fault();
}

// label numbers the connected parts of the foreground. The connectivity
// picks the neighbors, as generate_binary_structure does.
fn label(a, connectivity: i64 = 1) -> Arr, i64, Fault {
    let st, f = mo_default(a, connectivity);
    if f.kind != F_NONE { return empty_arr(), 0, f; }
    return mo_label(a, st, "label");
}

// label_st numbers the parts with a structure of size 3 on every axis. The
// structure must be symmetric.
fn label_st(a, structure) -> Arr, i64, Fault {
    return mo_label(a, structure, "label_st");
}

// mo_edt gives the Euclidean distance of each foreground value to the
// nearest background value. A pass along each axis takes the lower
// envelope of the parabolas of the earlier passes.
fn mo_edt(a, sampling: f64[:], want: bool, tag: const u8[:]) -> Arr, Arr, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), fa; }
    let nd = a.ndim;
    if sampling.len != 0 && sampling.len != 1 && sampling.len != nd {
        return empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the sampling holds a count that is not one for each axis");
    }
    var sp: f64[>..] = [];
    var d: i64 = 0;
    while d < nd {
        var s: f64 = 1.0;
        if sampling.len == 1 { s = sampling[0]; }
        if sampling.len > 1 { s = sampling[d]; }
        if !(s - s == 0.0) || !(s > 0.0) || s * (a.shape[d] as f64) > 1.0e100 {
            return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the sampling holds a value that is not above zero or is too large");
        }
        sp.push(s);
        d += 1;
    }
    let n = a.data.len;
    let inf = 1.0 / 0.0;
    var f: f64[>..] = nd_zeros(n);
    var background: i64 = 0;
    var p: i64 = 0;
    while p < n {
        if a.data[p] == 0.0 { background += 1; } else { f[p] = inf; }
        p += 1;
    }
    if background == 0 {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the array holds no background value");
    }
    var feat: i64[>..] = [];
    if want {
        var cx: i64[..8] = [];
        d = 0;
        while d < nd { cx.push(0); d += 1; }
        p = 0;
        while p < n {
            d = 0;
            while d < nd { feat.push(cx[d]); d += 1; }
            d = nd - 1;
            while d >= 0 {
                cx[d] += 1;
                if cx[d] < a.shape[d] { d = -1; } else { cx[d] = 0; d -= 1; }
            }
            p += 1;
        }
    }
    d = 0;
    while d < nd {
        mo_edt_axis(f, feat, a.shape, d, sp[d], want, nd);
        d += 1;
    }
    var dist: f64[>..] = nd_zeros(n);
    p = 0;
    while p < n {
        if f[p] != 0.0 { dist[p] = mathshim::root(f[p]); }
        p += 1;
    }
    var idx: f64[>..] = [];
    var ishape: i64[..8] = [];
    if want {
        ishape.push(nd);
        d = 0;
        while d < nd { ishape.push(a.shape[d]); d += 1; }
        d = 0;
        while d < nd {
            p = 0;
            while p < n { idx.push(feat[p * nd + d] as f64); p += 1; }
            d += 1;
        }
    }
    var ires = empty_arr();
    if want { ires = arr_of(copy(idx), ishape); }
    return arr_of(copy(dist), a.shape), ires, no_fault();
}

// mo_edt_axis takes the lower envelope of parabolas along one axis. The
// values of f hold squared distances, and inf stands for none yet.
fn mo_edt_axis(f: f64[>..]&, feat: i64[>..]&, shape: i64[:], axis: i64, s: f64, want: bool, nd: i64) {
    let n = shape[axis];
    var inner: i64 = 1;
    var k: i64 = axis + 1;
    while k < shape.len { inner *= shape[k]; k += 1; }
    let outer = f.len / (n * inner);
    let inf = 1.0 / 0.0;
    var line: f64[>..] = nd_zeros(n);
    var res: f64[>..] = nd_zeros(n);
    var v: i64[>..] = [];
    var z: f64[>..] = [];
    var from: i64[>..] = [];
    var q: i64 = 0;
    while q < n { v.push(0); from.push(0); q += 1; }
    q = 0;
    while q <= n { z.push(0.0); q += 1; }
    var tmp: i64[>..] = [];
    if want {
        q = 0;
        while q < n * nd { tmp.push(0); q += 1; }
    }
    var o: i64 = 0;
    while o < outer {
        var r: i64 = 0;
        while r < inner {
            let base = o * n * inner + r;
            q = 0;
            while q < n { line[q] = f[base + q * inner]; q += 1; }
            var top: i64 = -1;
            q = 0;
            while q < n {
                if line[q] != inf {
                    let xq = s * (q as f64);
                    if top < 0 {
                        top = 0;
                        v[0] = q;
                        z[0] = -inf;
                        z[1] = inf;
                    } else {
                        var sx: f64 = 0.0;
                        var again = true;
                        while again {
                            let xv = s * (v[top] as f64);
                            sx = ((line[q] + xq * xq) - (line[v[top]] + xv * xv)) / (2.0 * (xq - xv));
                            if sx <= z[top] && top > 0 { top -= 1; } else { again = false; }
                        }
                        if sx <= z[top] {
                            v[top] = q;
                        } else {
                            top += 1;
                            v[top] = q;
                            z[top] = sx;
                            z[top + 1] = inf;
                        }
                    }
                }
                q += 1;
            }
            if top >= 0 {
                var j: i64 = 0;
                q = 0;
                while q < n {
                    let xq = s * (q as f64);
                    while z[j + 1] < xq { j += 1; }
                    let xv = s * (v[j] as f64);
                    res[q] = (xq - xv) * (xq - xv) + line[v[j]];
                    from[q] = v[j];
                    q += 1;
                }
                q = 0;
                while q < n { f[base + q * inner] = res[q]; q += 1; }
                if want {
                    q = 0;
                    while q < n {
                        var c: i64 = 0;
                        while c < nd { tmp[q * nd + c] = feat[(base + from[q] * inner) * nd + c]; c += 1; }
                        q += 1;
                    }
                    q = 0;
                    while q < n {
                        var c: i64 = 0;
                        while c < nd { feat[(base + q * inner) * nd + c] = tmp[q * nd + c]; c += 1; }
                        q += 1;
                    }
                }
            }
            r += 1;
        }
        o += 1;
    }
}

// distance_transform_edt gives the Euclidean distance of each foreground
// value to the nearest background value. The sampling holds none, one, or
// one spacing for each axis.
fn distance_transform_edt(a, sampling: f64[:] = []) -> Arr, Fault {
    let d, i, f = mo_edt(a, sampling, false, "distance_transform_edt");
    return d, f;
}

// distance_transform_edt_indices gives the distances and the index of the
// nearest background value. The array of indices holds one row of positions for
// each axis, so its shape is the count of axes followed by the shape.
fn distance_transform_edt_indices(a, sampling: f64[:] = []) -> Arr, Arr, Fault {
    return mo_edt(a, sampling, true, "distance_transform_edt_indices");
}
