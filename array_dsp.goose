// The signal processing of one dimension: the design of a filter by
// the window method, the frequency response, the fast convolution, the
// median filter, the filter of Savitzky and Golay, the search of peaks,
// the periodogram and the resampling by the Fourier method.
namespace scientific;

import mathshim;

// The largest length that these routines take.
const MAX_DSP_LENGTH: i64 = 16777216;

// firwin designs a linear phase filter by the window method. The
// cutoffs are frequencies in the units of fs. With pass_zero true the
// first band is a pass band that reaches down to zero. The window is a
// name of get_window. The param holds the parameter of the window. With
// scale true the gain at the centre of the first pass band is one.
fn firwin(numtaps: i64, cutoff: f64[:], pass_zero: bool = true, window: const u8[:] = "hamming", scale: bool = true, fs: f64 = 2.0, param: f64 = 0.0 / 0.0) -> Arr, Fault {
    if numtaps < 1 || numtaps > MAX_DSP_LENGTH {
        return empty_arr(), fault(F_RANGE, "firwin", "the count of taps stands outside one to the limit");
    }
    if cutoff.len < 1 || !(fs > 0.0) {
        return empty_arr(), fault(F_VALUE, "firwin", "the cutoff holds no value or the rate is not above zero");
    }
    let nyq = 0.5 * fs;
    var edges: f64[>..] = [];
    var i: i64 = 0;
    while i < cutoff.len {
        let v = cutoff[i] / nyq;
        if !(v > 0.0 && v < 1.0) {
            return empty_arr(), fault(F_RANGE, "firwin", "a cutoff stands outside zero to the Nyquist rate");
        }
        if i > 0 && !(v > edges[i - 1]) {
            return empty_arr(), fault(F_VALUE, "firwin", "the cutoffs do not rise strictly");
        }
        edges.push(v);
        i += 1;
    }
    var odd = false;
    if cutoff.len % 2 == 1 { odd = true; }
    let pass_nyquist = odd != pass_zero;
    if pass_nyquist && numtaps % 2 == 0 {
        return empty_arr(), fault(F_VALUE, "firwin", "a filter that passes the Nyquist rate needs an odd count of taps");
    }
    var all: f64[>..] = [];
    if pass_zero { all.push(0.0); }
    i = 0;
    while i < edges.len { all.push(edges[i]); i += 1; }
    if pass_nyquist { all.push(1.0); }
    let alpha = 0.5 * (numtaps - 1);
    var h: f64[>..] = [];
    var k: i64 = 0;
    while k < numtaps {
        let m = k - alpha;
        var s: f64 = 0.0;
        var b: i64 = 0;
        while b + 1 < all.len {
            s += all[b + 1] * mathshim::sinc_of(all[b + 1] * m);
            s -= all[b] * mathshim::sinc_of(all[b] * m);
            b += 2;
        }
        h.push(s);
        k += 1;
    }
    let win, fw = get_window(window, numtaps, false, param);
    if fw.kind != F_NONE { return empty_arr(), fw; }
    k = 0;
    while k < numtaps { h[k] = h[k] * win.data[k]; k += 1; }
    if scale {
        let left = all[0];
        let right = all[1];
        var sf: f64 = 0.0;
        if left == 0.0 { sf = 0.0; }
        else if right == 1.0 { sf = 1.0; }
        else { sf = 0.5 * (left + right); }
        var total: f64 = 0.0;
        k = 0;
        while k < numtaps {
            total += h[k] * mathshim::cosine(3.141592653589793 * (k - alpha) * sf);
            k += 1;
        }
        k = 0;
        while k < numtaps { h[k] = h[k] / total; k += 1; }
    }
    return line_arr(copy(h)), no_fault();
}

// freqz gives the frequency response of a digital filter at n points
// from zero up to, but not including, pi. With whole true the points
// reach up to 2 pi. It gives the frequencies, the real parts and the
// imaginary parts of the response.
fn freqz(b: f64[:], a: f64[:], n: i64 = 512, whole: bool = false) -> Arr, Arr, Arr, Fault {
    if b.len < 1 || a.len < 1 || n < 1 || n > MAX_DSP_LENGTH {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "freqz", "a coefficient list holds no value or the count is out of range");
    }
    var top = 3.141592653589793;
    if whole { top = 6.283185307179586; }
    var w: f64[>..] = [];
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        let om = top * i / n;
        w.push(om);
        var br: f64 = 0.0;
        var bi: f64 = 0.0;
        var k: i64 = 0;
        while k < b.len {
            br += b[k] * mathshim::cosine(om * k);
            bi -= b[k] * mathshim::sine(om * k);
            k += 1;
        }
        var ar: f64 = 0.0;
        var ai: f64 = 0.0;
        k = 0;
        while k < a.len {
            ar += a[k] * mathshim::cosine(om * k);
            ai -= a[k] * mathshim::sine(om * k);
            k += 1;
        }
        let d = ar * ar + ai * ai;
        hr.push((br * ar + bi * ai) / d);
        hi.push((bi * ar - br * ai) / d);
        i += 1;
    }
    return line_arr(copy(w)), line_arr(copy(hr)), line_arr(copy(hi)), no_fault();
}

// lfiltic gives the initial state of the filter for the past outputs y
// and the past inputs x. The first element of each list is the most
// recent one.
fn lfiltic(b: f64[:], a: f64[:], y: f64[:], x: f64[:]) -> Arr, Fault {
    if b.len < 1 || a.len < 1 || a[0] == 0.0 {
        return empty_arr(), fault(F_VALUE, "lfiltic", "a coefficient list holds no value or a0 is zero");
    }
    let n = a.len - 1;
    let m = b.len - 1;
    var k = m;
    if n > k { k = n; }
    if k < 1 { return empty_arr(), fault(F_VALUE, "lfiltic", "the filter holds no state"); }
    var zi: f64[>..] = [];
    var i: i64 = 0;
    while i < k { zi.push(0.0); i += 1; }
    i = 0;
    while i < m {
        var s: f64 = 0.0;
        var j: i64 = 0;
        while j < m - i {
            if j < x.len { s += b[i + 1 + j] / a[0] * x[j]; }
            j += 1;
        }
        zi[i] = s;
        i += 1;
    }
    i = 0;
    while i < n {
        var s: f64 = 0.0;
        var j: i64 = 0;
        while j < n - i {
            if j < y.len { s += a[i + 1 + j] / a[0] * y[j]; }
            j += 1;
        }
        zi[i] = zi[i] - s;
        i += 1;
    }
    return line_arr(copy(zi)), no_fault();
}

// fftconvolve convolves two signals with the fast transform. The mode
// is "full", "same" for the size of a, or "valid" for the points that
// need no padding.
fn fftconvolve(a: f64[:], b: f64[:], mode: const u8[:] = "full") -> Arr, Fault {
    if a.len < 1 || b.len < 1 {
        return empty_arr(), fault(F_SHAPE, "fftconvolve", "a signal holds no value");
    }
    if mode != "full" && mode != "same" && mode != "valid" {
        return empty_arr(), fault(F_VALUE, "fftconvolve", "the mode holds an unknown name");
    }
    let n = a.len + b.len - 1;
    if n > MAX_DSP_LENGTH {
        return empty_arr(), fault(F_LIMIT, "fftconvolve", "the result is longer than the limit");
    }
    var pa: f64[>..] = [];
    var pb: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        if i < a.len { pa.push(a[i]); } else { pa.push(0.0); }
        if i < b.len { pb.push(b[i]); } else { pb.push(0.0); }
        i += 1;
    }
    let ar, ai, f1 = rfft(pa);
    if f1.kind != F_NONE { return empty_arr(), f1; }
    let br, bi, f2 = rfft(pb);
    if f2.kind != F_NONE { return empty_arr(), f2; }
    var re: f64[>..] = [];
    var im: f64[>..] = [];
    i = 0;
    while i < ar.data.len {
        re.push(ar.data[i] * br.data[i] - ai.data[i] * bi.data[i]);
        im.push(ar.data[i] * bi.data[i] + ai.data[i] * br.data[i]);
        i += 1;
    }
    let full, f3 = irfft(re, im, n);
    if f3.kind != F_NONE { return empty_arr(), f3; }
    if mode == "full" { return full, no_fault(); }
    var size = a.len;
    if mode == "valid" {
        size = a.len - b.len + 1;
        if b.len > a.len { size = b.len - a.len + 1; }
    }
    let start = (n - size) / 2;
    var out: f64[>..] = [];
    i = 0;
    while i < size { out.push(full.data[start + i]); i += 1; }
    return line_arr(copy(out)), no_fault();
}

// medfilt gives the median filter of a signal with an odd kernel size.
// The edges hold zeros outside the signal.
fn medfilt(x: f64[:], kernel_size: i64 = 3) -> Arr, Fault {
    if x.len < 1 || kernel_size < 1 || kernel_size % 2 == 0 || kernel_size > 65535 {
        return empty_arr(), fault(F_VALUE, "medfilt", "the signal holds no value or the kernel size is not odd");
    }
    let h = kernel_size / 2;
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len {
        var win: f64[>..] = [];
        var j: i64 = -h;
        while j <= h {
            let at = i + j;
            if at >= 0 && at < x.len { win.push(x[at]); } else { win.push(0.0); }
            j += 1;
        }
        let ord = order_places(win, 1, win.len);
        out.push(win[ord[h]]);
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

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

// resample changes the count of samples of a signal to num by the
// method of Fourier. The method treats the signal as periodic.
fn resample(x: f64[:], num: i64) -> Arr, Fault {
    if x.len < 1 || num < 1 || num > MAX_DSP_LENGTH {
        return empty_arr(), fault(F_RANGE, "resample", "the signal holds no value or the new count is out of range");
    }
    let nx = x.len;
    let re, im, f = rfft(x);
    if f.kind != F_NONE { return empty_arr(), f; }
    let ny = num / 2 + 1;
    var yr: f64[>..] = [];
    var yi: f64[>..] = [];
    var i: i64 = 0;
    while i < ny { yr.push(0.0); yi.push(0.0); i += 1; }
    var m = nx;
    if num < m { m = num; }
    let nyq = m / 2 + 1;
    i = 0;
    while i < nyq && i < re.data.len && i < ny { yr[i] = re.data[i]; yi[i] = im.data[i]; i += 1; }
    if m % 2 == 0 {
        let at = m / 2;
        if num < nx {
            yr[at] = yr[at] * 2.0;
            yi[at] = yi[at] * 2.0;
        } else if nx < num {
            yr[at] = yr[at] * 0.5;
            yi[at] = yi[at] * 0.5;
        }
    }
    let r, g = irfft(yr, yi, num);
    if g.kind != F_NONE { return empty_arr(), g; }
    var out: f64[>..] = [];
    i = 0;
    while i < num { out.push(r.data[i] * num / nx); i += 1; }
    return line_arr(copy(out)), no_fault();
}

// sos_response gives the response of the cascade of sections at the
// frequency om, as the product of the response of each section.
fn sos_response(sos, om: f64, re: f64&, im: f64&) {
    var pr: f64 = 1.0;
    var pi: f64 = 0.0;
    var s: i64 = 0;
    while s < sos.shape[0] {
        var br: f64 = 0.0;
        var bi: f64 = 0.0;
        var ar: f64 = 0.0;
        var ai: f64 = 0.0;
        var k: i64 = 0;
        while k < 3 {
            let c = mathshim::cosine(om * k);
            let n = mathshim::sine(om * k);
            br += sos.data[s * 6 + k] * c;
            bi -= sos.data[s * 6 + k] * n;
            ar += sos.data[s * 6 + 3 + k] * c;
            ai -= sos.data[s * 6 + 3 + k] * n;
            k += 1;
        }
        let d = ar * ar + ai * ai;
        let hr = (br * ar + bi * ai) / d;
        let hi = (bi * ar - br * ai) / d;
        let t = pr * hr - pi * hi;
        pi = pr * hi + pi * hr;
        pr = t;
        s += 1;
    }
    re = pr;
    im = pi;
    return;
}

// sosfreqz gives the frequency response of a filter in sections at the
// frequencies w, in radians per sample. It gives the real and the
// imaginary parts of the response.
fn sosfreqz_at(sos, w: f64[:]) -> Arr, Arr, Fault {
    let f = sos_check(sos, "sosfreqz");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    if w.len < 1 || w.len > MAX_DSP_LENGTH {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "sosfreqz", "the count of frequencies is out of range");
    }
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    var i: i64 = 0;
    while i < w.len {
        var r: f64 = 0.0;
        var m: f64 = 0.0;
        sos_response(sos, w[i], r, m);
        hr.push(r);
        hi.push(m);
        i += 1;
    }
    return line_arr(copy(hr)), line_arr(copy(hi)), no_fault();
}

// sosfreqz gives the response at n points from zero up to, but not
// including, pi. With whole true the points reach up to 2 pi. It gives
// the frequencies, the real parts and the imaginary parts.
fn sosfreqz(sos, n: i64 = 512, whole: bool = false) -> Arr, Arr, Arr, Fault {
    if n < 1 || n > MAX_DSP_LENGTH {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "sosfreqz", "the count is out of range");
    }
    var top = 3.141592653589793;
    if whole { top = 6.283185307179586; }
    var w: f64[>..] = [];
    var i: i64 = 0;
    while i < n { w.push(top * i / n); i += 1; }
    let hr, hi, f = sosfreqz_at(sos, w);
    return line_arr(copy(w)), hr, hi, f;
}

// resample_poly changes the rate of a signal by the ratio up over down.
// The signal goes through a low pass filter of the window method with a
// Kaiser window of beta 5, as SciPy does, and the edges hold zeros.
fn resample_poly(x: f64[:], up0: i64, down0: i64) -> Arr, Fault {
    if up0 < 1 || down0 < 1 {
        return empty_arr(), fault(F_RANGE, "resample_poly", "up and down must be one or more");
    }
    if x.len < 1 || x.len > MAX_DSP_LENGTH {
        return empty_arr(), fault(F_SHAPE, "resample_poly", "the signal holds no value or too many values");
    }
    var ga = up0;
    var gb = down0;
    while gb != 0 { let t = ga % gb; ga = gb; gb = t; }
    let up = up0 / ga;
    let down = down0 / ga;
    if up == 1 && down == 1 { return line_arr(copy_slice(x)), no_fault(); }
    var big = up;
    if down > up { big = down; }
    if big > 100000 || x.len * up > MAX_DSP_LENGTH {
        return empty_arr(), fault(F_LIMIT, "resample_poly", "the ratio is too large");
    }
    let h, hf = firwin(2 * 10 * big + 1, [1.0 / big], true, "kaiser", true, 2.0, 5.0);
    if hf.kind != F_NONE { return empty_arr(), hf; }
    return resample_filtered(x, up, down, h.data);
}

// resample_filtered does the work of resample_poly with the filter h.
// The half length is the count of taps less one, halved. The up and
// down hold no common factor.
fn resample_filtered(x: f64[:], up: i64, down: i64, h: f64[:]) -> Arr, Fault {
    let n_in = x.len;
    let n_out = (n_in * up + down - 1) / down;
    let half = (h.len - 1) / 2;
    let pre = down - half % down;
    let remove = (half + pre) / down;
    var y: f64[>..] = [];
    var j: i64 = 0;
    while j < n_out {
        let n = j + remove;
        var s: f64 = 0.0;
        // The tap k holds h[k - pre] and meets the sample at n down - k.
        var k: i64 = pre;
        while k < pre + h.len {
            let idx = n * down - k;
            if idx >= 0 && idx % up == 0 && idx / up < n_in {
                s += up * h[k - pre] * x[idx / up];
            }
            k += 1;
        }
        y.push(s);
        j += 1;
    }
    return line_arr(copy(y)), no_fault();
}

// decimate lowers the rate of a signal by the whole factor q after a
// low pass filter with no phase shift. The kind "fir" takes a Hamming
// filter of 20 q + 1 taps. The kind "iir" takes a Chebyshev filter of
// type one, of order 8 and with a ripple of 0.05 dB, run forward and
// backward. The order n of zero takes the default of SciPy.
fn decimate(x: f64[:], q: i64, ftype: const u8[:] = "iir", n: i64 = 0) -> Arr, Fault {
    if q < 1 || q > 100000 {
        return empty_arr(), fault(F_RANGE, "decimate", "the factor stands outside one to the limit");
    }
    if x.len < 1 || x.len > MAX_DSP_LENGTH {
        return empty_arr(), fault(F_SHAPE, "decimate", "the signal holds no value or too many values");
    }
    if n < 0 || n > 1000 {
        return empty_arr(), fault(F_RANGE, "decimate", "the order stands outside the range");
    }
    if ftype == "fir" {
        var taps = 20 * q + 1;
        if n > 0 { taps = n + 1; }
        let h, hf = firwin(taps, [1.0 / q], true, "hamming");
        if hf.kind != F_NONE { return empty_arr(), hf; }
        return resample_filtered(x, 1, q, h.data);
    }
    if ftype != "iir" {
        return empty_arr(), fault(F_VALUE, "decimate", "the kind holds an unknown name");
    }
    var order = n;
    if order == 0 { order = 8; }
    let sos, sf = cheby1_sos(order, 0.05, [0.8 / q], "lowpass");
    if sf.kind != F_NONE { return empty_arr(), sf; }
    let y, yf = sosfiltfilt(sos, x);
    if yf.kind != F_NONE { return empty_arr(), yf; }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < y.data.len { out.push(y.data[i]); i += q; }
    return line_arr(copy(out)), no_fault();
}
