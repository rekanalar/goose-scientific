// The calls of N dimensions that take a block and fill a buffer, as in
// scipy.ndimage: geometric_transform and generic_filter1d.
//
// geometric_transform takes a trailing block `{ c, o => ... }`. The block gets
// the coordinate of an output value in c and writes the coordinate in the
// input into o. The buffer o holds one entry for each axis of the input and
// starts at zero. SciPy takes a function that gives a tuple.
//
// generic_filter1d takes a trailing block `{ line, out => ... }`. The block
// gets one line of the array along the axis, extended by the border rule to
// the length n + size - 1, and fills the buffer out of the length n. SciPy
// takes a function with the same two arguments.
//
// The time of a block is the time of the caller. The package cannot interrupt
// a block that does not stop.
namespace scientific;

import mathshim;

// geometric_transform evaluates the block at every output value, and then
// reads the spline of the input at the coordinates that the block gives. The
// order, the mode, the fill value and the prefilter follow map_coordinates.
fn geometric_transform<F>(a, output_shape: i64[:] = [], order: i64 = 3, mode: const u8[:] = "constant", cval: f64 = 0.0, prefilter: bool = true) -> Arr, Fault {
    // The order and the mode are checked before the first block runs.
    let m, f = tr_checked(a, order, mode, "geometric_transform");
    if f.kind != F_NONE { return empty_arr(), f; }
    var oshape: i64[..8] = [];
    var d: i64 = 0;
    if output_shape.len == 0 {
        while d < a.ndim { oshape.push(a.shape[d]); d += 1; }
    } else {
        if output_shape.len > 8 { return empty_arr(), fault(F_SHAPE, "geometric_transform", "the output shape holds more than 8 axes"); }
        while d < output_shape.len { oshape.push(output_shape[d]); d += 1; }
    }
    let fo = check_shape(oshape);
    if fo.kind != F_NONE { return empty_arr(), fo; }
    let nout = numel(oshape);
    let nd = a.ndim;
    if nout > MAX_ELEMENTS / nd {
        return empty_arr(), fault(F_LIMIT, "geometric_transform", "the coordinates hold more elements than the limit");
    }
    // The cost of the spline is checked before the first block runs, as in
    // map_coordinates.
    var taps: i64 = 1;
    var nt: i64 = order + 1;
    if order == 0 { nt = 1; }
    d = 0;
    while d < nd {
        if taps > TR_COST / nt { return empty_arr(), fault(F_LIMIT, "geometric_transform", "the cost of the transform is above the limit"); }
        taps *= nt;
        d += 1;
    }
    if nout > TR_COST / taps {
        return empty_arr(), fault(F_LIMIT, "geometric_transform", "the cost of the transform is above the limit");
    }
    var coords: f64[>..] = nd_zeros(nout * nd);
    var ix: i64[..8] = [];
    d = 0;
    while d < oshape.len { ix.push(0); d += 1; }
    var cbuf: f64[>..] = nd_zeros(oshape.len);
    var obuf: f64[>..] = nd_zeros(nd);
    var p: i64 = 0;
    while p < nout {
        d = 0;
        while d < oshape.len { cbuf[d] = ix[d] as f64; d += 1; }
        d = 0;
        while d < nd { obuf[d] = 0.0; d += 1; }
        var o = obuf;
        F(copy(cbuf), o);
        d = 0;
        while d < nd { coords[d * nout + p] = obuf[d]; d += 1; }
        gf_next(ix, oshape);
        p += 1;
    }
    var sh: i64[..8] = [];
    sh.push(nd);
    d = 0;
    while d < oshape.len { sh.push(oshape[d]); d += 1; }
    return map_coordinates(a, arr_of(copy(coords), sh), order, mode, cval, prefilter);
}

// generic_filter1d runs the block over every line of the array along the axis.
fn generic_filter1d<F>(a, size: i64, axis: i64 = -1, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64 = 0) -> Arr, Fault {
    let m, f = nd_check(a, mode, "generic_filter1d");
    if f.kind != F_NONE { return empty_arr(), f; }
    let ax = nd_axis(axis, a.ndim);
    if ax < 0 { return empty_arr(), fault(F_RANGE, "generic_filter1d", "the axis stands outside the array"); }
    if size < 1 { return empty_arr(), fault(F_VALUE, "generic_filter1d", "the size holds a value below one"); }
    if size > ND_COST {
        return empty_arr(), fault(F_LIMIT, "generic_filter1d", "the size is above the limit");
    }
    if !nd_origin_ok(origin, size) {
        return empty_arr(), fault(F_RANGE, "generic_filter1d", "the origin stands outside the window");
    }
    let n = a.shape[ax];
    let before = size / 2 + origin;
    let ne = n + size - 1;
    // The line and its copy stay below the limit of the elements.
    if ne > MAX_ELEMENTS / 2 {
        return empty_arr(), fault(F_LIMIT, "generic_filter1d", "the line holds more values than the limit");
    }
    var inner: i64 = 1;
    var d: i64 = ax + 1;
    while d < a.ndim { inner *= a.shape[d]; d += 1; }
    let lines = a.data.len / n;
    if lines > ND_COST / (n + size) {
        return empty_arr(), fault(F_LIMIT, "generic_filter1d", "the cost of the filter is above the limit");
    }
    var out: f64[>..] = nd_zeros(a.data.len);
    var ext: f64[>..] = nd_zeros(ne);
    var obuf: f64[>..] = nd_zeros(n);
    var ln: i64 = 0;
    while ln < lines {
        let base = (ln / inner) * n * inner + ln % inner;
        var j: i64 = 0;
        while j < ne {
            let s = nd_src(j - before, n, m);
            if s < 0 { ext[j] = cval; } else { ext[j] = a.data[base + s * inner]; }
            j += 1;
        }
        j = 0;
        while j < n { obuf[j] = 0.0; j += 1; }
        var o = obuf;
        F(copy(ext), o);
        j = 0;
        while j < n { out[base + j * inner] = obuf[j]; j += 1; }
        ln += 1;
    }
    return arr_of(copy(out), a.shape), no_fault();
}
