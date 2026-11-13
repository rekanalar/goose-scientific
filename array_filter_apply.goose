// The application of a digital filter: lfilter, lfilter_zi, filtfilt, sosfilt and hilbert.
//
namespace scientific;

import mathshim;

// lfilter filters a signal with the filter of the coefficients b and
// a. The state starts at zero.
fn lfilter(b: f64[:], a: f64[:], x: f64[:]) -> Arr, Fault {
    var pb: f64[>..] = [];
    var pa: f64[>..] = [];
    let f = flt_prepare(b, a, pb, pa, "lfilter");
    if f.kind != F_NONE { return empty_arr(), f; }
    var z: f64[>..] = [];
    var i: i64 = 0;
    while i < pb.len { z.push(0.0); i += 1; }
    var y: f64[>..] = [];
    flt_run(pb, pa, x, z, y);
    return flt_arr(copy(y)), no_fault();
}

// lfilter_lanes filters every line of an array along an axis. The
// buffer zi holds the initial states when have_zi holds, with the lines
// in the order of the array and the states along the axis. The
// results go in y, with the shape of the signal, and in zf.
fn lfilter_lanes(pb: f64[:], pa: f64[:], x, ax: i64, zi: f64[:], have_zi: bool, y: f64[>..]&, zf: f64[>..]&) {
    let n = x.shape[ax];
    let inner = product_after_of(x.shape, ax);
    let outer = product_before_of(x.shape, ax);
    let m = pb.len - 1;
    var i: i64 = 0;
    while i < x.data.len { y.push(0.0); i += 1; }
    i = 0;
    while i < outer * m * inner { zf.push(0.0); i += 1; }
    var o: i64 = 0;
    while o < outer {
        var j: i64 = 0;
        while j < inner {
            var lane: f64[>..] = [];
            var k: i64 = 0;
            while k < n { lane.push(x.data[o * n * inner + k * inner + j]); k += 1; }
            var z: f64[>..] = [];
            k = 0;
            while k < pb.len {
                if have_zi && k < m { z.push(zi[(o * m + k) * inner + j]); } else { z.push(0.0); }
                k += 1;
            }
            var out: f64[>..] = [];
            flt_run(pb, pa, lane, z, out);
            k = 0;
            while k < n { y[o * n * inner + k * inner + j] = out[k]; k += 1; }
            k = 0;
            while k < m { zf[(o * m + k) * inner + j] = z[k]; k += 1; }
            j += 1;
        }
        o += 1;
    }
}

// lfilter_shape gives the shape of the signal with the axis length
// replaced.
fn lfilter_shape(x, ax: i64, len: i64) -> i64[..8] {
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < x.ndim {
        if i == ax { s.push(len); } else { s.push(x.shape[i]); }
        i += 1;
    }
    return s;
}

// lfilter_axis filters an array along an axis, as SciPy lfilter does
// for an array of N dimensions. The axis is the last one by default.
fn lfilter_axis(b: f64[:], a: f64[:], x, axis: i64 = -1) -> Arr, Fault {
    let ax = fix_axis(axis, x.ndim);
    if ax < 0 {
        return empty_arr(), fault(F_RANGE, "lfilter_axis", "the axis stands outside the shape");
    }
    var pb: f64[>..] = [];
    var pa: f64[>..] = [];
    let f = flt_prepare(b, a, pb, pa, "lfilter_axis");
    if f.kind != F_NONE { return empty_arr(), f; }
    var y: f64[>..] = [];
    var zf: f64[>..] = [];
    let none: f64[0] = [];
    lfilter_lanes(pb, pa, x, ax, none, false, y, zf);
    return Arr { ndim: x.ndim, shape: lfilter_shape(x, ax, x.shape[ax]), data: copy(y) }, no_fault();
}

// lfilter_state filters an array along an axis from the initial state
// zi, and gives the signal and the final state. The state holds one
// value less than the longer of b and a along the axis, and the other
// axes have the shape of the signal, as SciPy needs.
fn lfilter_state(b: f64[:], a: f64[:], x, zi, axis: i64 = -1) -> Arr, Arr, Fault {
    let ax = fix_axis(axis, x.ndim);
    if ax < 0 {
        return empty_arr(), empty_arr(), fault(F_RANGE, "lfilter_state", "the axis stands outside the shape");
    }
    var pb: f64[>..] = [];
    var pa: f64[>..] = [];
    let f = flt_prepare(b, a, pb, pa, "lfilter_state");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let m = pb.len - 1;
    if zi.ndim != x.ndim {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "lfilter_state", "the state does not hold the shape of the signal");
    }
    var i: i64 = 0;
    while i < x.ndim {
        var want = x.shape[i];
        if i == ax { want = m; }
        if zi.shape[i] != want {
            return empty_arr(), empty_arr(), fault(F_SHAPE, "lfilter_state", "the state does not hold the shape of the signal");
        }
        i += 1;
    }
    i = 0;
    while i < zi.data.len {
        if !(zi.data[i] - zi.data[i] == 0.0) {
            return empty_arr(), empty_arr(), fault(F_VALUE, "lfilter_state", "the state holds a value that is not finite");
        }
        i += 1;
    }
    var y: f64[>..] = [];
    var zf: f64[>..] = [];
    lfilter_lanes(pb, pa, x, ax, zi.data, true, y, zf);
    let ys = Arr { ndim: x.ndim, shape: lfilter_shape(x, ax, x.shape[ax]), data: copy(y) };
    let zs = Arr { ndim: x.ndim, shape: lfilter_shape(x, ax, m), data: copy(zf) };
    return ys, zs, no_fault();
}

// flt_zi gives the state of the filter at rest for a constant input
// of one.
fn flt_zi(pb: f64[:], pa: f64[:], zi: f64[>..]&) -> Fault {
    let n = pb.len;
    var sb: f64 = 0.0;
    var sa: f64 = 0.0;
    var i: i64 = 0;
    while i < n { sb = sb + pb[i]; sa = sa + pa[i]; i += 1; }
    if sa == 0.0 {
        return fault(F_VALUE, "lfilter_zi", "the denominator sums to zero, so the filter holds no steady state");
    }
    let gain = sb / sa;
    i = 0;
    while i < n { zi.push(0.0); i += 1; }
    var acc: f64 = 0.0;
    i = n - 1;
    while i >= 1 {
        acc = acc + pb[i] - gain * pa[i];
        zi[i - 1] = acc;
        i -= 1;
    }
    return no_fault();
}

// lfilter_zi gives the initial state for a step input of one.
fn lfilter_zi(b: f64[:], a: f64[:]) -> Arr, Fault {
    var pb: f64[>..] = [];
    var pa: f64[>..] = [];
    let f = flt_prepare(b, a, pb, pa, "lfilter_zi");
    if f.kind != F_NONE { return empty_arr(), f; }
    var zi: f64[>..] = [];
    let g = flt_zi(pb, pa, zi);
    if g.kind != F_NONE { return empty_arr(), g; }
    if zi.len < 2 { return empty_arr(), fault(F_VALUE, "lfilter_zi", "the filter holds no state"); }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i + 1 < zi.len { out.push(zi[i]); i += 1; }
    return flt_arr(copy(out)), no_fault();
}

// flt_odd builds the odd extension of a signal: each end reflects
// about its end point. The caller checks that the signal is longer
// than the edge.
fn flt_odd(x: f64[:], edge: i64, ext: f64[>..]&) {
    var i: i64 = 0;
    while i < edge { ext.push(2.0 * x[0] - x[edge - i]); i += 1; }
    i = 0;
    while i < x.len { ext.push(x[i]); i += 1; }
    i = 0;
    while i < edge { ext.push(2.0 * x[x.len - 1] - x[x.len - 2 - i]); i += 1; }
}

// filtfilt filters a signal forward and backward, so the phase
// shift is zero. The signal is extended at both ends with an odd
// reflection of three times the length of the coefficients.
fn filtfilt(b: f64[:], a: f64[:], x: f64[:]) -> Arr, Fault {
    var pb: f64[>..] = [];
    var pa: f64[>..] = [];
    let f = flt_prepare(b, a, pb, pa, "filtfilt");
    if f.kind != F_NONE { return empty_arr(), f; }
    let edge = 3 * pb.len;
    if x.len <= edge {
        return empty_arr(), fault(F_SHAPE, "filtfilt", "the signal is not longer than three times the coefficients");
    }
    var zi: f64[>..] = [];
    let g = flt_zi(pb, pa, zi);
    if g.kind != F_NONE { return empty_arr(), g; }
    var ext: f64[>..] = [];
    flt_odd(x, edge, ext);
    var i: i64 = 0;
    var z: f64[>..] = [];
    i = 0;
    while i < zi.len { z.push(zi[i] * ext[0]); i += 1; }
    var y: f64[>..] = [];
    flt_run(pb, pa, ext, z, y);
    var r: f64[>..] = [];
    i = 0;
    while i < y.len { r.push(y[y.len - 1 - i]); i += 1; }
    z = [];
    i = 0;
    while i < zi.len { z.push(zi[i] * r[0]); i += 1; }
    var y2: f64[>..] = [];
    flt_run(pb, pa, r, z, y2);
    var out: f64[>..] = [];
    i = 0;
    while i < x.len { out.push(y2[y2.len - 1 - edge - i]); i += 1; }
    return flt_arr(copy(out)), no_fault();
}

// sos_check checks a matrix of sections: six columns and a first
// denominator coefficient of one in each row.
fn sos_check(sos, tag: const u8[:]) -> Fault {
    if sos.ndim != 2 || sos.shape[1] != 6 {
        return fault(F_SHAPE, tag, "the sections do not hold a matrix of six columns");
    }
    var s: i64 = 0;
    while s < sos.shape[0] {
        if sos.data[s * 6 + 3] != 1.0 {
            return fault(F_VALUE, tag, "a section holds a first denominator coefficient that is not one");
        }
        s += 1;
    }
    return no_fault();
}

// sos_cascade runs a signal through the sections one after the other.
// The list zi holds two state values per section, or no value for a
// state of zero. The gain scales the state values. The caller checks
// the sections.
fn sos_cascade(sos, x: f64[:], zi: f64[:], gain: f64, out: f64[>..]&) {
    var cur: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len { cur.push(x[i]); i += 1; }
    var s: i64 = 0;
    while s < sos.shape[0] {
        let r = s * 6;
        var pb: f64[>..] = [sos.data[r], sos.data[r + 1], sos.data[r + 2]];
        var pa: f64[>..] = [1.0, sos.data[r + 4], sos.data[r + 5]];
        var z: f64[>..] = [0.0, 0.0, 0.0];
        if zi.len > 0 {
            z[0] = zi[s * 2] * gain;
            z[1] = zi[s * 2 + 1] * gain;
        }
        var y: f64[>..] = [];
        flt_run(pb, pa, cur, z, y);
        cur = copy(y);
        s += 1;
    }
    i = 0;
    while i < cur.len { out.push(cur[i]); i += 1; }
}

// sosfilt filters a signal with a cascade of second order sections.
// Each row of the matrix holds b0, b1, b2, a0, a1 and a2. The first
// denominator coefficient of each row must hold one.
fn sosfilt(sos, x: f64[:]) -> Arr, Fault {
    let chk = sos_check(sos, "sosfilt");
    if chk.kind != F_NONE { return empty_arr(), chk; }
    var y: f64[>..] = [];
    sos_cascade(sos, x, [], 0.0, y);
    return flt_arr(copy(y)), no_fault();
}

// hilbert gives the analytic signal of a real signal. The result holds
// the real parts, which equal the signal, and the imaginary parts.
fn hilbert(x: f64[:]) -> Arr, Arr, Fault {
    let n = x.len;
    let fr, fi, f = fft(x);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    var re: f64[>..] = [];
    var im: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var h: f64 = 0.0;
        if i == 0 { h = 1.0; }
        else if n % 2 == 0 && i == n / 2 { h = 1.0; }
        else if i < (n + 1) / 2 { h = 2.0; }
        re.push(fr.data[i] * h);
        im.push(fi.data[i] * h);
        i += 1;
    }
    return ifft(re, im);
}
