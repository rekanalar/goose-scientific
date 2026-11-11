// The filters of N dimensions that take a block, as in scipy.ndimage:
// generic_filter, generic_laplace and generic_gradient_magnitude.
//
// A block is a trailing block over a generic type. The call
// `generic_filter(a, [3, 3]) { w => w[0] + w[1] }` runs the block for each
// value. The block takes the values of the window in row-major order of the
// footprint, and it gives one number. A block of generic_laplace and of
// generic_gradient_magnitude takes the array, the axis, the mode and the
// fill value, and it gives the derivative along that axis.
namespace scientific;

import mathshim;

// gf_offsets lists the nonzero places of a footprint, one row of N
// coordinates for each place. It gives the count of places.
fn gf_offsets(fp: f64[:], fshape: i64[:], offs: i64[>..]&) -> i64 {
    let nd = fshape.len;
    var ox: i64[..8] = [];
    var d: i64 = 0;
    while d < nd { ox.push(0); d += 1; }
    var t: i64 = 0;
    var q: i64 = 0;
    while q < fp.len {
        if fp[q] != 0.0 {
            d = 0;
            while d < nd { offs.push(ox[d]); d += 1; }
            t += 1;
        }
        d = nd - 1;
        while d >= 0 {
            ox[d] += 1;
            if ox[d] < fshape[d] { d = -1; } else { ox[d] = 0; d -= 1; }
        }
        q += 1;
    }
    return t;
}

// gf_gather fills buf with the window of the value p. The index ix holds
// the position of p. A place outside the array reads cval, or the value
// that the mode names.
fn gf_gather(src: f64[:], shape: i64[:], ix: i64[:], offs: i64[:], t: i64, shift: i64[:], mode: i64, cval: f64, buf: f64[>..]&) {
    let nd = shape.len;
    var k: i64 = 0;
    while k < t {
        var flat: i64 = 0;
        var outside = false;
        var d: i64 = 0;
        while d < nd && !outside {
            let m = nd_src(ix[d] + offs[k * nd + d] - shift[d], shape[d], mode);
            if m < 0 { outside = true; } else { flat = flat * shape[d] + m; }
            d += 1;
        }
        if outside { buf[k] = cval; } else { buf[k] = src[flat]; }
        k += 1;
    }
}

// gf_next moves the index ix to the next position in row-major order.
fn gf_next(ix: i64[..8]&, shape: i64[:]) {
    var d: i64 = shape.len - 1;
    while d >= 0 {
        ix[d] += 1;
        if ix[d] < shape[d] { d = -1; } else { ix[d] = 0; d -= 1; }
    }
}

// gf_check checks the operands and gives the mode and the shift of every
// axis. The footprint must hold a nonzero place.
fn gf_check(a, fp, mode: const u8[:], origin: i64[:], shift: i64[..8]&, tag: const u8[:]) -> i64, Fault {
    let m, f = nd_check(a, mode, tag);
    if f.kind != F_NONE { return 0, f; }
    let ff = check_stack(fp.ndim, fp.shape, fp.data.len, 1, tag);
    if ff.kind != F_NONE { return 0, ff; }
    if fp.ndim != a.ndim {
        return 0, fault(F_SHAPE, tag, "the footprint holds another count of dimensions than the array");
    }
    var og: i64[..8] = [];
    if !nd_list(origin, a.ndim, 0, og) {
        return 0, fault(F_SHAPE, tag, "the origin holds a count that is not one for each axis");
    }
    var d: i64 = 0;
    while d < a.ndim {
        let fs = fp.shape[d];
        if og[d] < -(fs / 2) || og[d] > fs - 1 - fs / 2 {
            return 0, fault(F_RANGE, tag, "the origin stands outside the footprint");
        }
        shift.push(fs / 2 + og[d]);
        d += 1;
    }
    var t: i64 = 0;
    var i: i64 = 0;
    while i < fp.data.len { if fp.data[i] != 0.0 { t += 1; } i += 1; }
    if t < 1 { return 0, fault(F_VALUE, tag, "the footprint holds no nonzero place"); }
    if a.data.len > ND_RANK_COST / t {
        return 0, fault(F_LIMIT, tag, "the cost of the filter is above the limit");
    }
    return m, no_fault();
}

// generic_filter_fp runs the block over the footprint around each value.
fn generic_filter_fp<F>(a, footprint, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    var shift: i64[..8] = [];
    let m, f = gf_check(a, footprint, mode, origin, shift, "generic_filter_fp");
    if f.kind != F_NONE { return empty_arr(), f; }
    var offs: i64[>..] = [];
    let t = gf_offsets(footprint.data, footprint.shape, offs);
    var buf: f64[>..] = nd_zeros(t);
    var out: f64[>..] = nd_zeros(a.data.len);
    var ix: i64[..8] = [];
    var d: i64 = 0;
    while d < a.ndim { ix.push(0); d += 1; }
    var p: i64 = 0;
    while p < a.data.len {
        gf_gather(a.data, a.shape, ix, offs, t, shift, m, cval, buf);
        out[p] = F(copy(buf));
        gf_next(ix, a.shape);
        p += 1;
    }
    return arr_of(copy(out), a.shape), no_fault();
}

// generic_filter runs the block over a box around each value.
fn generic_filter<F>(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    let fp, g = nd_box_fp(a, size, "generic_filter");
    if g.kind != F_NONE { return empty_arr(), g; }
    var shift: i64[..8] = [];
    let m, f = gf_check(a, fp, mode, origin, shift, "generic_filter");
    if f.kind != F_NONE { return empty_arr(), f; }
    var offs: i64[>..] = [];
    let t = gf_offsets(fp.data, fp.shape, offs);
    var buf: f64[>..] = nd_zeros(t);
    var out: f64[>..] = nd_zeros(a.data.len);
    var ix: i64[..8] = [];
    var d: i64 = 0;
    while d < a.ndim { ix.push(0); d += 1; }
    var p: i64 = 0;
    while p < a.data.len {
        gf_gather(a.data, a.shape, ix, offs, t, shift, m, cval, buf);
        out[p] = F(copy(buf));
        gf_next(ix, a.shape);
        p += 1;
    }
    return arr_of(copy(out), a.shape), no_fault();
}

// generic_laplace sums the second derivatives that the block gives for each
// axis. The block takes the array, the axis, the mode and the fill value. A
// result of another size than the array is a fault.
fn generic_laplace<F>(a, mode: const u8[:] = "reflect", cval: f64 = 0.0) -> Arr, Fault {
    let m, f = nd_check(a, mode, "generic_laplace");
    if f.kind != F_NONE { return empty_arr(), f; }
    var acc: f64[>..] = nd_zeros(a.data.len);
    var ax: i64 = 0;
    while ax < a.ndim {
        let d = F(clone_of(a), ax, mode, cval);
        if d.data.len != a.data.len {
            return empty_arr(), fault(F_SHAPE, "generic_laplace", "the block gave an array of another size");
        }
        var i: i64 = 0;
        while i < acc.len { acc[i] += d.data[i]; i += 1; }
        ax += 1;
    }
    return arr_of(copy(acc), a.shape), no_fault();
}

// generic_gradient_magnitude gives the root of the sum of the squares of the
// derivatives that the block gives for each axis.
fn generic_gradient_magnitude<F>(a, mode: const u8[:] = "reflect", cval: f64 = 0.0) -> Arr, Fault {
    let m, f = nd_check(a, mode, "generic_gradient_magnitude");
    if f.kind != F_NONE { return empty_arr(), f; }
    var acc: f64[>..] = nd_zeros(a.data.len);
    var ax: i64 = 0;
    while ax < a.ndim {
        let d = F(clone_of(a), ax, mode, cval);
        if d.data.len != a.data.len {
            return empty_arr(), fault(F_SHAPE, "generic_gradient_magnitude", "the block gave an array of another size");
        }
        var i: i64 = 0;
        while i < acc.len { acc[i] += d.data[i] * d.data[i]; i += 1; }
        ax += 1;
    }
    var i: i64 = 0;
    while i < acc.len { acc[i] = mathshim::root(acc[i]); i += 1; }
    return arr_of(copy(acc), a.shape), no_fault();
}
