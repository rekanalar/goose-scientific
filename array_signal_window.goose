// The conditions of a window and the ways of filtering a signal.
//
// check_cola and check_nola tell whether a window allows the overlap
// add. closest_stft_dual_window gives the dual window that lies
// closest to a wanted one. choose_conv_method tells which way of
// convolving is faster.
//
// The formulas follow SciPy 1.18.1 (scipy.signal).
namespace scientific;

import mathshim;

// sig_window_values gives the values of a window. The name is a name
// of get_window with its parameter param. The values in win hold the
// window when win is not empty.
fn sig_window_values(window: const u8[:], param: f64, win: f64[:], nx: i64,
                     tag: const u8[:]) -> f64[>..], Fault {
    var w: f64[>..] = [];
    if win.len > 0 {
        if win.len != nx {
            return w, fault(F_SHAPE, tag, "the window values must hold the count of the places");
        }
        var i: i64 = 0;
        while i < win.len {
            if !(win[i] - win[i] == 0.0) {
                return w, fault(F_VALUE, tag, "the window holds a value that is not finite");
            }
            w.push(win[i]);
            i += 1;
        }
        return w, no_fault();
    }
    let gw, f = get_window(window, nx, true, param);
    if f.kind != F_NONE { return w, f; }
    var i: i64 = 0;
    while i < nx { w.push(gw.data[i]); i += 1; }
    return w, no_fault();
}

// check_cola tells whether a window gives the same sum at every place
// when the parts of a signal overlap, as SciPy check_COLA does. The
// count nperseg holds the length of a part and noverlap the count of
// the places that two parts share. The tolerance tol holds the
// largest difference that passes.
fn check_cola(window: const u8[:] = "hann", param: f64 = 0.0 / 0.0,
              win: f64[:] = [], nperseg: i64 = 256, noverlap: i64 = -1,
              tol: f64 = 1.0e-10) -> bool, Fault {
    let tag = "check_cola";
    if nperseg < 1 {
        return false, fault(F_VALUE, tag, "the count of a part must hold one value or more");
    }
    var ov = noverlap;
    if ov < 0 { ov = nperseg / 2; }
    if ov >= nperseg {
        return false, fault(F_VALUE, tag, "the overlap must stand below the count of a part");
    }
    let w, f = sig_window_values(window, param, win, nperseg, tag);
    if f.kind != F_NONE { return false, f; }
    let step = nperseg - ov;
    let count = nperseg / step;
    var bins: f64[>..] = [];
    var i: i64 = 0;
    while i < step { bins.push(0.0); i += 1; }
    var s: i64 = 0;
    while s < count {
        var k: i64 = 0;
        while k < step {
            bins[k] += w[s * step + k];
            k += 1;
        }
        s += 1;
    }
    let rest = nperseg % step;
    if rest != 0 {
        var k: i64 = 0;
        while k < rest {
            bins[k] += w[nperseg - rest + k];
            k += 1;
        }
    }
    var med = sig_median(copy(bins));
    var k: i64 = 0;
    while k < step {
        var d = bins[k] - med;
        if d < 0.0 { d = -d; }
        if !(d < tol) { return false, no_fault(); }
        k += 1;
    }
    return true, no_fault();
}

// check_nola tells whether a window gives a sum above the tolerance
// at every place when the parts of a signal overlap, as SciPy
// check_NOLA does. The options follow check_cola.
fn check_nola(window: const u8[:] = "hann", param: f64 = 0.0 / 0.0,
              win: f64[:] = [], nperseg: i64 = 256, noverlap: i64 = -1,
              tol: f64 = 1.0e-10) -> bool, Fault {
    let tag = "check_nola";
    if nperseg < 1 {
        return false, fault(F_VALUE, tag, "the count of a part must hold one value or more");
    }
    var ov = noverlap;
    if ov < 0 { ov = nperseg / 2; }
    if ov >= nperseg {
        return false, fault(F_VALUE, tag, "the overlap must stand below the count of a part");
    }
    let w, f = sig_window_values(window, param, win, nperseg, tag);
    if f.kind != F_NONE { return false, f; }
    let step = nperseg - ov;
    let count = nperseg / step;
    var bins: f64[>..] = [];
    var i: i64 = 0;
    while i < step { bins.push(0.0); i += 1; }
    var s: i64 = 0;
    while s < count {
        var k: i64 = 0;
        while k < step {
            let v = w[s * step + k];
            bins[k] += v * v;
            k += 1;
        }
        s += 1;
    }
    let rest = nperseg % step;
    if rest != 0 {
        var k: i64 = 0;
        while k < rest {
            let v = w[nperseg - rest + k];
            bins[k] += v * v;
            k += 1;
        }
    }
    var k: i64 = 0;
    while k < step {
        if !(bins[k] > tol) { return false, no_fault(); }
        k += 1;
    }
    return true, no_fault();
}

// sig_median gives the median of a list. The list is sorted in place.
fn sig_median(v: f64[>..]) -> f64 {
    var i: i64 = 1;
    while i < v.len {
        let key = v[i];
        var j = i - 1;
        while j >= 0 && v[j] > key { v[j + 1] = v[j]; j -= 1; }
        v[j + 1] = key;
        i += 1;
    }
    let n = v.len;
    if n % 2 == 1 { return v[n / 2]; }
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// closest_stft_dual_window gives the dual window of a window that
// lies closest to a wanted one, as SciPy closest_STFT_dual_window
// does. The window and the wanted window must hold the same count of
// values. The step hop tells how far the window moves at each step.
// The empty wanted window asks for the flat window of ones. The call
// gives the dual window and the factor that scales the window.
fn closest_stft_dual_window(win: f64[:], hop: i64, desired: f64[:] = [],
                            scaled: bool = true) -> Arr, f64, Fault {
    let tag = "closest_stft_dual_window";
    if win.len < 1 || !all_finite(win) {
        return empty_arr(), 0.0, fault(F_VALUE, tag, "the window must be a finite list");
    }
    if hop < 1 || hop > win.len {
        return empty_arr(), 0.0, fault(F_VALUE, tag, "the step must hold from one to the count of the window");
    }
    var want: f64[>..] = [];
    var i: i64 = 0;
    if desired.len > 0 {
        if desired.len != win.len || !all_finite(desired) {
            return empty_arr(), 0.0, fault(F_SHAPE, tag, "the wanted window must hold the count of the window");
        }
        while i < desired.len { want.push(desired[i]); i += 1; }
    } else {
        while i < win.len { want.push(1.0); i += 1; }
    }
    // The canonical dual window holds the window over the sum of the
    // squares of the windows that share a place. The places that share
    // a place lie the same step apart.
    var sq: f64[>..] = [];
    i = 0;
    while i < win.len { sq.push(win[i] * win[i]); i += 1; }
    let w2 = sig_step_sum(sq, hop);
    var wd: f64[>..] = [];
    i = 0;
    while i < win.len {
        if w2[i] <= 0.0 {
            return empty_arr(), 0.0, fault(F_VALUE, tag, "the window does not allow an inverse transform");
        }
        wd.push(win[i] / w2[i]);
        i += 1;
    }
    // The sum of the products of the window and the wanted window,
    // gathered at each place.
    var prod: f64[>..] = [];
    i = 0;
    while i < win.len { prod.push(win[i] * want[i]); i += 1; }
    let q0 = sig_step_sum(prod, hop);
    var qd: f64[>..] = [];
    var k: i64 = 0;
    i = 0;
    while i < win.len { qd.push(wd[i] * q0[i]); i += 1; }
    if !scaled {
        var out: f64[>..] = [];
        k = 0;
        while k < win.len { out.push(wd[k] + want[k] - qd[k]); k += 1; }
        return line_arr(copy(out)), 1.0, no_fault();
    }
    var num: f64 = 0.0;
    var den: f64 = 0.0;
    k = 0;
    while k < win.len {
        num += qd[k] * wd[k];
        den += qd[k] * qd[k];
        k += 1;
    }
    if num == 0.0 || !(den > 1.0e-15) {
        return empty_arr(), 0.0, fault(F_VALUE, tag, "the factor of the window stands outside the range of the numbers");
    }
    let alpha = num / den;
    var out: f64[>..] = [];
    k = 0;
    while k < win.len { out.push(wd[k] + alpha * (want[k] - qd[k])); k += 1; }
    return line_arr(copy(out)), alpha, no_fault();
}

// sig_step_sum gathers the values of a list at the places that lie
// the same step apart. The place k holds the sum of the values at the
// places that lie the same step apart as k, on either side.
fn sig_step_sum(v: f64[:], step: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < v.len { out.push(v[i]); i += 1; }
    var j = step;
    while j < v.len {
        var k: i64 = 0;
        while k < v.len - j { out[k + j] += v[k]; k += 1; }
        k = 0;
        while k < v.len - j { out[k] += v[k + j]; k += 1; }
        j += step;
    }
    return out;
}

// choose_conv_method tells which way of convolving is faster, as SciPy
// choose_conv_method does. The shapes hold the counts of the sides of
// the two lists. The mode is "full", "same" or "valid". The call gives
// the name "fft" or "direct".
fn choose_conv_method(shape_a: i64[:], shape_b: i64[:],
                      mode: const u8[:] = "full") -> const u8[:], Fault {
    let tag = "choose_conv_method";
    if shape_a.len < 1 || shape_b.len < 1 || shape_a.len != shape_b.len {
        return "", fault(F_SHAPE, tag, "the shapes must hold the same count of sides");
    }
    if mode != "full" && mode != "same" && mode != "valid" {
        return "", fault(F_VALUE, tag, "the mode is unknown");
    }
    var d: i64 = 0;
    while d < shape_a.len {
        if shape_a[d] < 1 || shape_b[d] < 1 {
            return "", fault(F_VALUE, tag, "a side must hold one value or more");
        }
        d += 1;
    }
    // The count of the places of the result.
    var out_len: f64 = 1.0;
    d = 0;
    while d < shape_a.len {
        var side = shape_a[d] as f64;
        if mode == "full" { side = shape_a[d] as f64 + shape_b[d] - 1.0; }
        else if mode == "valid" {
            let da = shape_a[d] as f64;
            let db = shape_b[d] as f64;
            side = mathshim::abs_of(da - db) + 1.0;
        }
        out_len *= side;
        d += 1;
    }
    // The count of the sums of the direct way.
    var prod_a: f64 = 1.0;
    var prod_b: f64 = 1.0;
    d = 0;
    while d < shape_a.len { prod_a *= shape_a[d]; prod_b *= shape_b[d]; d += 1; }
    var direct: f64 = 0.0;
    if shape_a.len == 1 {
        let a = shape_a[0] as f64;
        let b = shape_b[0] as f64;
        if mode == "full" {
            direct = a * b;
        } else if mode == "valid" {
            if b >= a { direct = (b - a + 1.0) * a; } else { direct = (a - b + 1.0) * b; }
        } else {
            if a < b { direct = a * b; } else { direct = a * b - mathshim::trunc_of(b / 2.0) * mathshim::trunc_of((b + 1.0) / 2.0); }
        }
    } else {
        if mode == "same" {
            direct = prod_a * prod_b;
        } else {
            direct = (if prod_a < prod_b { prod_a } else { prod_b }) * out_len;
        }
    }
    // The count of the sums of the transform way.
    var full_len: f64 = 1.0;
    d = 0;
    while d < shape_a.len {
        full_len *= shape_a[d] as f64 + shape_b[d] - 1.0;
        d += 1;
    }
    let fft_ops = 3.0 * full_len * mathshim::ln(full_len);
    // The constants of SciPy hold the times of the two ways.
    var c_fft: f64 = 0.0;
    var c_direct: f64 = 0.0;
    var c_off: f64 = -1.0e-3;
    if shape_a.len == 1 {
        if mode == "valid" {
            c_fft = 1.89095737e-9;
            c_direct = 2.1364985e-10;
        } else if mode == "full" {
            c_fft = 1.7649070e-9;
            c_direct = 2.1414831e-10;
        } else if shape_b[0] > shape_a[0] {
            c_fft = 3.21635404e-9;
            c_direct = 1.1773253e-8;
            c_off = -1.0e-5;
        } else {
            c_fft = 3.2646654e-9;
            c_direct = 2.8478277e-10;
        }
    } else {
        c_off = -1.0e-4;
        if mode == "valid" {
            c_fft = 1.85927e-9;
            c_direct = 2.11242e-8;
        } else if mode == "full" {
            c_fft = 1.99817e-9;
            c_direct = 1.66174e-8;
        } else {
            c_fft = 2.04735e-9;
            c_direct = 1.55367e-8;
        }
    }
    if c_fft * fft_ops < c_direct * direct + c_off {
        return "fft", no_fault();
    }
    return "direct", no_fault();
}
