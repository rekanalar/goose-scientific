// The grey morphology of N dimensions, as in scipy.ndimage: grey_erosion,
// grey_dilation, grey_opening, grey_closing, morphological_gradient,
// morphological_laplace, white_tophat and black_tophat. Also the binary
// calls binary_propagation and binary_hit_or_miss.
//
// Each grey call has two forms. The plain form takes a size and works with
// a flat box. The form with the suffix _st takes a footprint and a structure.
// An empty array stands for None. A footprint and a structure of the same
// shape make a non-flat element: an erosion takes the minimum of
// value - structure, and a dilation takes the maximum of value + structure
// over the reflected element. With a structure and no footprint, all places
// of the structure count.
//
// NaN is the largest value, as in the rank filters, so the result does not
// depend on the order of the window.
namespace scientific;

import mathshim;

// gr_run runs one erosion or one dilation. A dilation reflects the footprint
// and the structure and flips the origin, as SciPy does.
fn gr_run(a, fp, st, size: i64[:], mode: const u8[:], cval: f64, origin: i64[:], dilate: bool, tag: const u8[:]) -> Arr, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fa.kind != F_NONE { return empty_arr(), fa; }
    var fd: f64[>..] = [];
    var fsh: i64[..8] = [];
    var i: i64 = 0;
    var d: i64 = 0;
    if fp.ndim != 0 {
        let ff = check_stack(fp.ndim, fp.shape, fp.data.len, 1, tag);
        if ff.kind != F_NONE { return empty_arr(), ff; }
        while d < fp.ndim { fsh.push(fp.shape[d]); d += 1; }
        while i < fp.data.len { fd.push(fp.data[i]); i += 1; }
    } else if st.ndim != 0 {
        let fs = check_stack(st.ndim, st.shape, st.data.len, 1, tag);
        if fs.kind != F_NONE { return empty_arr(), fs; }
        while d < st.ndim { fsh.push(st.shape[d]); d += 1; }
        while i < st.data.len { fd.push(1.0); i += 1; }
    } else {
        let bx, g = nd_box_fp(a, size, tag);
        if g.kind != F_NONE { return empty_arr(), g; }
        while d < bx.ndim { fsh.push(bx.shape[d]); d += 1; }
        while i < bx.data.len { fd.push(bx.data[i]); i += 1; }
    }
    var sd: f64[>..] = [];
    let flat = st.ndim == 0;
    if !flat {
        let fs = check_stack(st.ndim, st.shape, st.data.len, 1, tag);
        if fs.kind != F_NONE { return empty_arr(), fs; }
        if st.ndim != fsh.len || st.data.len != fd.len {
            return empty_arr(), fault(F_SHAPE, tag, "the structure holds another shape than the footprint");
        }
        d = 0;
        while d < st.ndim {
            if st.shape[d] != fsh[d] { return empty_arr(), fault(F_SHAPE, tag, "the structure holds another shape than the footprint"); }
            d += 1;
        }
        i = 0;
        while i < st.data.len { sd.push(st.data[i]); i += 1; }
    }
    if fsh.len != a.ndim {
        return empty_arr(), fault(F_SHAPE, tag, "the footprint holds another count of dimensions than the array");
    }
    var og: i64[..8] = [];
    if !nd_list(origin, a.ndim, 0, og) {
        return empty_arr(), fault(F_SHAPE, tag, "the origin holds a count that is not one for each axis");
    }
    var o2: i64[..8] = [];
    d = 0;
    while d < a.ndim {
        if og[d] < -1000000000 || og[d] > 1000000000 {
            return empty_arr(), fault(F_RANGE, tag, "the origin stands outside the footprint");
        }
        if dilate {
            var o = -og[d];
            if fsh[d] % 2 == 0 { o -= 1; }
            o2.push(o);
        } else {
            o2.push(og[d]);
        }
        d += 1;
    }
    if dilate {
        var lo: i64 = 0;
        var hi: i64 = fd.len - 1;
        while lo < hi {
            let t1 = fd[lo]; fd[lo] = fd[hi]; fd[hi] = t1;
            if !flat { let t2 = sd[lo]; sd[lo] = sd[hi]; sd[hi] = t2; }
            lo += 1;
            hi -= 1;
        }
    }
    let ft = arr_of(copy(fd), fsh);
    var shift: i64[..8] = [];
    let m, f = gf_check(a, ft, mode, o2, shift, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    var offs: i64[>..] = [];
    let t = gf_offsets(ft.data, ft.shape, offs);
    var sv: f64[>..] = [];
    i = 0;
    while i < fd.len {
        if fd[i] != 0.0 { if flat { sv.push(0.0); } else { sv.push(sd[i]); } }
        i += 1;
    }
    var buf: f64[>..] = nd_zeros(t);
    var out: f64[>..] = nd_zeros(a.data.len);
    var ix: i64[..8] = [];
    d = 0;
    while d < a.ndim { ix.push(0); d += 1; }
    var p: i64 = 0;
    while p < a.data.len {
        gf_gather(a.data, a.shape, ix, offs, t, shift, m, cval, buf);
        var best: f64 = 0.0;
        var k: i64 = 0;
        while k < t {
            var v = buf[k];
            if !flat { if dilate { v = v + sv[k]; } else { v = v - sv[k]; } }
            if k == 0 { best = v; }
            else if dilate { if nd_less(best, v) { best = v; } }
            else { if nd_less(v, best) { best = v; } }
            k += 1;
        }
        out[p] = best;
        gf_next(ix, a.shape);
        p += 1;
    }
    return arr_of(copy(out), a.shape), no_fault();
}

// gr_sub gives x - y element by element. gr_laplace gives d + e - 2 * a.
fn gr_sub(x, y) -> Arr {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < x.data.len { out.push(x.data[i] - y.data[i]); i += 1; }
    return arr_of(copy(out), x.shape);
}

fn gr_laplace(dl, er, a) -> Arr {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { out.push(dl.data[i] + er.data[i] - a.data[i] - a.data[i]); i += 1; }
    return arr_of(copy(out), a.shape);
}

// gr_op runs one of the eight calls: 0 erosion, 1 dilation, 2 opening,
// 3 closing, 4 gradient, 5 laplace, 6 white tophat and 7 black tophat.
fn gr_op(a, fp, st, size: i64[:], mode: const u8[:], cval: f64, origin: i64[:], op: i64, tag: const u8[:]) -> Arr, Fault {
    if op == 0 || op == 1 {
        return gr_run(a, fp, st, size, mode, cval, origin, op == 1, tag);
    }
    if op == 2 || op == 6 {
        let e, f1 = gr_run(a, fp, st, size, mode, cval, origin, false, tag);
        if f1.kind != F_NONE { return empty_arr(), f1; }
        let o, f2 = gr_run(e, fp, st, size, mode, cval, origin, true, tag);
        if f2.kind != F_NONE { return empty_arr(), f2; }
        if op == 2 { return o, no_fault(); }
        return gr_sub(a, o), no_fault();
    }
    if op == 3 || op == 7 {
        let e, f1 = gr_run(a, fp, st, size, mode, cval, origin, true, tag);
        if f1.kind != F_NONE { return empty_arr(), f1; }
        let o, f2 = gr_run(e, fp, st, size, mode, cval, origin, false, tag);
        if f2.kind != F_NONE { return empty_arr(), f2; }
        if op == 3 { return o, no_fault(); }
        return gr_sub(o, a), no_fault();
    }
    let dl, f3 = gr_run(a, fp, st, size, mode, cval, origin, true, tag);
    if f3.kind != F_NONE { return empty_arr(), f3; }
    let er, f4 = gr_run(a, fp, st, size, mode, cval, origin, false, tag);
    if f4.kind != F_NONE { return empty_arr(), f4; }
    if op == 4 { return gr_sub(dl, er), no_fault(); }
    return gr_laplace(dl, er, a), no_fault();
}

// grey_erosion gives the minimum of a box around each value.
fn grey_erosion(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, empty_arr(), empty_arr(), size, mode, cval, origin, 0, "grey_erosion");
}

// grey_dilation gives the maximum of a box around each value.
fn grey_dilation(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, empty_arr(), empty_arr(), size, mode, cval, origin, 1, "grey_dilation");
}

// grey_opening is an erosion and then a dilation.
fn grey_opening(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, empty_arr(), empty_arr(), size, mode, cval, origin, 2, "grey_opening");
}

// grey_closing is a dilation and then an erosion.
fn grey_closing(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, empty_arr(), empty_arr(), size, mode, cval, origin, 3, "grey_closing");
}

// morphological_gradient is the dilation minus the erosion.
fn morphological_gradient(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, empty_arr(), empty_arr(), size, mode, cval, origin, 4, "morphological_gradient");
}

// morphological_laplace is the dilation plus the erosion minus twice the array.
fn morphological_laplace(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, empty_arr(), empty_arr(), size, mode, cval, origin, 5, "morphological_laplace");
}

// white_tophat is the array minus its opening.
fn white_tophat(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, empty_arr(), empty_arr(), size, mode, cval, origin, 6, "white_tophat");
}

// black_tophat is the closing minus the array.
fn black_tophat(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, empty_arr(), empty_arr(), size, mode, cval, origin, 7, "black_tophat");
}

// The forms with the suffix _st take a footprint and a structure. An empty
// array stands for None. One of the two must hold a value.
fn grey_erosion_st(a, footprint, structure, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, footprint, structure, [], mode, cval, origin, 0, "grey_erosion_st");
}

fn grey_dilation_st(a, footprint, structure, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, footprint, structure, [], mode, cval, origin, 1, "grey_dilation_st");
}

fn grey_opening_st(a, footprint, structure, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, footprint, structure, [], mode, cval, origin, 2, "grey_opening_st");
}

fn grey_closing_st(a, footprint, structure, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, footprint, structure, [], mode, cval, origin, 3, "grey_closing_st");
}

fn morphological_gradient_st(a, footprint, structure, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, footprint, structure, [], mode, cval, origin, 4, "morphological_gradient_st");
}

fn morphological_laplace_st(a, footprint, structure, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, footprint, structure, [], mode, cval, origin, 5, "morphological_laplace_st");
}

fn white_tophat_st(a, footprint, structure, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, footprint, structure, [], mode, cval, origin, 6, "white_tophat_st");
}

fn black_tophat_st(a, footprint, structure, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return gr_op(a, footprint, structure, [], mode, cval, origin, 7, "black_tophat_st");
}

// binary_propagation grows the foreground inside the mask until no value
// changes. It is a dilation that repeats.
fn binary_propagation(a, connectivity: i64 = 1, border_value: i64 = 0, origin: i64[:] = []) -> Arr, Fault {
    let st, f = mo_default(a, connectivity);
    if f.kind != F_NONE { return empty_arr(), f; }
    return mo_binary(a, st, empty_arr(), 0, border_value, origin, 1, "binary_propagation");
}

// binary_propagation_st takes a structure and a mask. An empty array as the
// mask means no mask.
fn binary_propagation_st(a, structure, mask, border_value: i64 = 0, origin: i64[:] = []) -> Arr, Fault {
    return mo_binary(a, structure, mask, 0, border_value, origin, 1, "binary_propagation_st");
}

// gr_hit runs the hit or miss. The first erosion finds the places where
// structure1 fits the foreground. The second finds the places where
// structure2 fits the background. A border counts as background for the
// first and as foreground of the complement for the second.
fn gr_hit(a, s1, s2, o1: i64[:], o2: i64[:], tag: const u8[:]) -> Arr, Fault {
    let e1, f1 = mo_binary(a, s1, empty_arr(), 1, 0, o1, 0, tag);
    if f1.kind != F_NONE { return empty_arr(), f1; }
    var comp: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        if a.data[i] != 0.0 { comp.push(0.0); } else { comp.push(1.0); }
        i += 1;
    }
    let e2, f2 = mo_binary(arr_of(copy(comp), a.shape), s2, empty_arr(), 1, 1, o2, 0, tag);
    if f2.kind != F_NONE { return empty_arr(), f2; }
    var out: f64[>..] = [];
    i = 0;
    while i < a.data.len {
        if e1.data[i] != 0.0 && e2.data[i] != 0.0 { out.push(1.0); } else { out.push(0.0); }
        i += 1;
    }
    return arr_of(copy(out), a.shape), no_fault();
}

// gr_not gives the complement of a structure.
fn gr_not(st) -> Arr {
    var inv: f64[>..] = [];
    var i: i64 = 0;
    while i < st.data.len { if st.data[i] != 0.0 { inv.push(0.0); } else { inv.push(1.0); } i += 1; }
    return arr_of(copy(inv), st.shape);
}

// binary_hit_or_miss finds the places where the structure of the connectivity
// fits the foreground and its complement fits the background. An empty origin2
// stands for origin1.
fn binary_hit_or_miss(a, connectivity: i64 = 1, origin1: i64[:] = [], origin2: i64[:] = []) -> Arr, Fault {
    let s1, f = mo_default(a, connectivity);
    if f.kind != F_NONE { return empty_arr(), f; }
    if origin2.len == 0 { return gr_hit(a, s1, gr_not(s1), origin1, origin1, "binary_hit_or_miss"); }
    return gr_hit(a, s1, gr_not(s1), origin1, origin2, "binary_hit_or_miss");
}

// binary_hit_or_miss_st takes both structures. An empty structure2 stands for
// the complement of structure1. An empty origin2 stands for origin1.
fn binary_hit_or_miss_st(a, structure1, structure2, origin1: i64[:] = [], origin2: i64[:] = []) -> Arr, Fault {
    if structure2.ndim == 0 {
        if origin2.len == 0 { return gr_hit(a, structure1, gr_not(structure1), origin1, origin1, "binary_hit_or_miss_st"); }
        return gr_hit(a, structure1, gr_not(structure1), origin1, origin2, "binary_hit_or_miss_st");
    }
    if origin2.len == 0 { return gr_hit(a, structure1, structure2, origin1, origin1, "binary_hit_or_miss_st"); }
    return gr_hit(a, structure1, structure2, origin1, origin2, "binary_hit_or_miss_st");
}
