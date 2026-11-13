// The filter of Savitzky and Golay, the search of peaks and the periodogram.
//
namespace scientific;

import mathshim;

// savgol_dot gives the coefficients of the filter of Savitzky and
// Golay in the order of the window, for a dot product with the data.
fn savgol_dot(window: i64, polyorder: i64, deriv: i64, delta: f64, out: f64[>..]&) -> Fault {
    let h = window / 2;
    let p1 = polyorder + 1;
    var g: f64[>..] = [];
    var r: i64 = 0;
    while r < p1 {
        var c: i64 = 0;
        while c < p1 {
            var s: f64 = 0.0;
            var j: i64 = 0;
            while j < window {
                s += mathshim::power(1.0 * (j - h), 1.0 * (r + c));
                j += 1;
            }
            g.push(s);
            c += 1;
        }
        r += 1;
    }
    // The point zero holds the power zero, which is one.
    var rhs: f64[>..] = [];
    r = 0;
    while r < p1 { rhs.push(0.0); r += 1; }
    var fact: f64 = 1.0;
    var q: i64 = 2;
    while q <= deriv { fact = fact * q; q += 1; }
    rhs[deriv] = fact / mathshim::power(delta, 1.0 * deriv);
    let z, fz = solve(mat2(copy(g), p1, p1), line_arr(copy(rhs)));
    if fz.kind != F_NONE { return fz; }
    var j: i64 = 0;
    while j < window {
        var s: f64 = 0.0;
        var k: i64 = 0;
        while k < p1 {
            var pw: f64 = 1.0;
            if k > 0 { pw = mathshim::power(1.0 * (j - h), 1.0 * k); }
            if k == 0 { pw = 1.0; }
            s += pw * z.data[k];
            k += 1;
        }
        out.push(s);
        j += 1;
    }
    return no_fault();
}

// savgol_check checks the window, the order and the derivative.
fn savgol_check(window: i64, polyorder: i64, deriv: i64, tag: const u8[:]) -> Fault {
    if window < 1 || window % 2 == 0 || window > 65535 {
        return fault(F_VALUE, tag, "the window holds a value that is not odd and positive");
    }
    if polyorder < 0 || polyorder >= window {
        return fault(F_VALUE, tag, "the order of the polynomial must stay below the window");
    }
    if deriv < 0 || deriv > polyorder {
        return fault(F_VALUE, tag, "the derivative order stands outside zero to the order of the polynomial");
    }
    return no_fault();
}

// savgol_coeffs gives the coefficients of the filter of Savitzky and
// Golay, in the order that a convolution takes.
fn savgol_coeffs(window: i64, polyorder: i64, deriv: i64 = 0, delta: f64 = 1.0) -> Arr, Fault {
    let c = savgol_check(window, polyorder, deriv, "savgol_coeffs");
    if c.kind != F_NONE { return empty_arr(), c; }
    var d: f64[>..] = [];
    let f = savgol_dot(window, polyorder, deriv, delta, d);
    if f.kind != F_NONE { return empty_arr(), f; }
    var out: f64[>..] = [];
    var i = window - 1;
    while i >= 0 { out.push(d[i]); i -= 1; }
    return line_arr(copy(out)), no_fault();
}

// savgol_extend gives the signal value at a place that may lie outside,
// under the edge mode "nearest", "mirror", "constant" or "wrap".
fn savgol_extend(x: f64[:], at: i64, mode: i64) -> f64 {
    let n = x.len;
    if at >= 0 && at < n { return x[at]; }
    if mode == 0 { if at < 0 { return x[0]; } return x[n - 1]; }
    if mode == 2 { return 0.0; }
    if mode == 3 {
        var w = at % n;
        if w < 0 { w += n; }
        return x[w];
    }
    // The mirror holds the period 2 (n - 1) and keeps the edge once.
    if n == 1 { return x[0]; }
    let per = 2 * (n - 1);
    var w = at % per;
    if w < 0 { w += per; }
    if w >= n { w = per - w; }
    return x[w];
}

// savgol_filter smooths a signal with a polynomial fit over a sliding
// window. The mode is "interp", "nearest", "mirror", "constant" or
// "wrap". The mode "interp" fits a polynomial to the edge windows.
fn savgol_filter(x: f64[:], window: i64, polyorder: i64, deriv: i64 = 0, delta: f64 = 1.0, mode: const u8[:] = "interp") -> Arr, Fault {
    let c = savgol_check(window, polyorder, deriv, "savgol_filter");
    if c.kind != F_NONE { return empty_arr(), c; }
    if mode != "interp" && mode != "nearest" && mode != "mirror" && mode != "constant" && mode != "wrap" {
        return empty_arr(), fault(F_VALUE, "savgol_filter", "the mode holds an unknown name");
    }
    if x.len < 1 || !(delta > 0.0) {
        return empty_arr(), fault(F_VALUE, "savgol_filter", "the signal holds no value or delta is not above zero");
    }
    let n = x.len;
    let h = window / 2;
    if mode == "interp" && window > n {
        return empty_arr(), fault(F_SHAPE, "savgol_filter", "the window is longer than the signal");
    }
    var d: f64[>..] = [];
    let f = savgol_dot(window, polyorder, deriv, delta, d);
    if f.kind != F_NONE { return empty_arr(), f; }
    var em: i64 = 0;
    if mode == "mirror" { em = 1; }
    if mode == "constant" { em = 2; }
    if mode == "wrap" { em = 3; }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var s: f64 = 0.0;
        var j: i64 = 0;
        while j < window {
            s += d[j] * savgol_extend(x, i - h + j, em);
            j += 1;
        }
        out.push(s);
        i += 1;
    }
    if mode == "interp" {
        // The edges come from the polynomial fitted to the end windows.
        var side: i64 = 0;
        while side < 2 {
            var t: f64[>..] = [];
            var v: f64[>..] = [];
            var j: i64 = 0;
            while j < window {
                t.push(j);
                if side == 0 { v.push(x[j]); } else { v.push(x[n - window + j]); }
                j += 1;
            }
            let pc, fp = polyfit(t, v, polyorder);
            if fp.kind != F_NONE { return empty_arr(), fp; }
            var co: f64[>..] = [];
            j = 0;
            while j < pc.data.len { co.push(pc.data[j]); j += 1; }
            var turn: i64 = 0;
            while turn < deriv {
                var dd: f64[>..] = [];
                var q: i64 = 0;
                while q + 1 < co.len { dd.push(co[q] * (co.len - 1 - q)); q += 1; }
                if dd.len == 0 { dd.push(0.0); }
                co = copy(dd);
                turn += 1;
            }
            j = 0;
            while j < h {
                var tp: f64 = j;
                if side == 1 { tp = window - h + j; }
                var val: f64 = 0.0;
                var q: i64 = 0;
                while q < co.len { val = val * tp + co[q]; q += 1; }
                val = val / mathshim::power(delta, 1.0 * deriv);
                if side == 0 { out[j] = val; } else { out[n - h + j] = val; }
                j += 1;
            }
            side += 1;
        }
    }
    return line_arr(copy(out)), no_fault();
}

// peak_prominences gives the prominence of each peak: the height of the
// peak above the higher of the two lowest points between the peak and
// a higher point on each side.
fn peak_prominences(x: f64[:], peaks: f64[:]) -> Arr, Fault {
    if peaks.len < 1 {
        return empty_arr(), fault(F_SHAPE, "peak_prominences", "the list of peaks holds no value");
    }
    var out: f64[>..] = [];
    var k: i64 = 0;
    while k < peaks.len {
        let pk = peaks[k];
        if !(pk >= 0.0) || pk >= x.len || mathshim::trunc_of(pk) != pk {
            return empty_arr(), fault(F_RANGE, "peak_prominences", "a peak lies outside the signal");
        }
        let p = trunc_to(pk);
        var lmin = x[p];
        var i = p;
        while i >= 0 && x[i] <= x[p] {
            if x[i] < lmin { lmin = x[i]; }
            i -= 1;
        }
        var rmin = x[p];
        i = p;
        while i < x.len && x[i] <= x[p] {
            if x[i] < rmin { rmin = x[i]; }
            i += 1;
        }
        var base = lmin;
        if rmin > base { base = rmin; }
        out.push(x[p] - base);
        k += 1;
    }
    return line_arr(copy(out)), no_fault();
}

fn trunc_to(v: f64) -> i64 {
    return mathshim::trunc_of(v) as i64;
}

// find_peaks gives the places of the local maxima of a signal. A flat
// peak holds the place of its middle. The missing value for height or
// prominence means no limit. The height and the prominence are lower
// limits. The distance is the least gap between two peaks, in places,
// and zero means no limit. The order of the tests is the order of
// SciPy: the height, the distance, then the prominence.
fn find_peaks(x: f64[:], height: f64 = 0.0 / 0.0, distance: f64 = 0.0, prominence: f64 = 0.0 / 0.0) -> Arr, Fault {
    if x.len < 1 {
        return empty_arr(), fault(F_SHAPE, "find_peaks", "the signal holds no value");
    }
    if distance != 0.0 && !(distance >= 1.0) {
        return empty_arr(), fault(F_VALUE, "find_peaks", "the distance holds a value below one");
    }
    var pk: f64[>..] = [];
    var i: i64 = 1;
    while i < x.len - 1 {
        if x[i - 1] < x[i] {
            var ahead = i + 1;
            while ahead < x.len - 1 && x[ahead] == x[i] { ahead += 1; }
            if x[ahead] < x[i] {
                pk.push((i + ahead - 1) / 2);
                i = ahead;
            }
        }
        i += 1;
    }
    if height == height {
        var keep: f64[>..] = [];
        i = 0;
        while i < pk.len { if x[trunc_to(pk[i])] >= height { keep.push(pk[i]); } i += 1; }
        pk = copy(keep);
    }
    if distance >= 1.0 && pk.len > 1 {
        var dist = distance;
        let cd = mathshim::trunc_of(dist);
        if cd < dist { dist = cd + 1.0; }
        var heights: f64[>..] = [];
        i = 0;
        while i < pk.len { heights.push(x[trunc_to(pk[i])]); i += 1; }
        let ord = order_places(heights, 1, heights.len);
        var alive: bool[>..] = [];
        i = 0;
        while i < pk.len { alive.push(true); i += 1; }
        i = pk.len - 1;
        while i >= 0 {
            let j = ord[i];
            if alive[j] {
                var k = j - 1;
                while k >= 0 && pk[j] - pk[k] < dist { alive[k] = false; k -= 1; }
                k = j + 1;
                while k < pk.len && pk[k] - pk[j] < dist { alive[k] = false; k += 1; }
            }
            i -= 1;
        }
        var keep: f64[>..] = [];
        i = 0;
        while i < pk.len { if alive[i] { keep.push(pk[i]); } i += 1; }
        pk = copy(keep);
    }
    if prominence == prominence && pk.len > 0 {
        let pr, fp = peak_prominences(x, pk);
        if fp.kind != F_NONE { return empty_arr(), fp; }
        var keep: f64[>..] = [];
        i = 0;
        while i < pk.len { if pr.data[i] >= prominence { keep.push(pk[i]); } i += 1; }
        pk = copy(keep);
    }
    if pk.len == 0 {
        return empty_arr(), fault(F_VALUE, "find_peaks", "the signal holds no peak that passes the limits");
    }
    return line_arr(copy(pk)), no_fault();
}

// periodogram gives the power spectrum of a real signal. The window
// is a name of get_window. With detrend true the mean leaves the
// signal. The scaling is "density" or "spectrum". It gives the
// frequencies and the one sided power.
fn periodogram(x: f64[:], fs: f64 = 1.0, window: const u8[:] = "boxcar", detrend: bool = true, scaling: const u8[:] = "density") -> Arr, Arr, Fault {
    if x.len < 1 || !(fs > 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "periodogram", "the signal holds no value or the rate is not above zero");
    }
    if scaling != "density" && scaling != "spectrum" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "periodogram", "the scaling holds an unknown name");
    }
    let n = x.len;
    let win, fw = get_window(window, n, true);
    if fw.kind != F_NONE { return empty_arr(), empty_arr(), fw; }
    var mean: f64 = 0.0;
    if detrend {
        var i: i64 = 0;
        while i < n { mean += x[i]; i += 1; }
        mean = mean / n;
    }
    var xs: f64[>..] = [];
    var s1: f64 = 0.0;
    var s2: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        xs.push((x[i] - mean) * win.data[i]);
        s1 += win.data[i];
        s2 += win.data[i] * win.data[i];
        i += 1;
    }
    let re, im, ft = rfft(xs);
    if ft.kind != F_NONE { return empty_arr(), empty_arr(), ft; }
    var scale = 1.0 / (fs * s2);
    if scaling == "spectrum" { scale = 1.0 / (s1 * s1); }
    var p: f64[>..] = [];
    var f: f64[>..] = [];
    i = 0;
    while i < re.data.len {
        var v = (re.data[i] * re.data[i] + im.data[i] * im.data[i]) * scale;
        if i > 0 && !(n % 2 == 0 && i == n / 2) { v = v * 2.0; }
        p.push(v);
        f.push(fs * i / n);
        i += 1;
    }
    return line_arr(copy(f)), line_arr(copy(p)), no_fault();
}

