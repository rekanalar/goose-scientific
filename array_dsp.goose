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

// firwin2 designs a linear phase filter by the frequency sampling
// method. The list freq holds the frequencies from zero up to fs / 2 and
// the list gain holds the response at each one. A frequency may occur
// twice, for a jump. The window is a name of get_window, or "none" for
// no window. With antisymmetric true the filter has odd symmetry.
fn firwin2(numtaps: i64, freq: f64[:], gain: f64[:], nfreqs: i64 = 0, window: const u8[:] = "hamming", antisymmetric: bool = false, fs: f64 = 2.0) -> Arr, Fault {
    if numtaps < 1 || numtaps > MAX_DSP_LENGTH / 4 {
        return empty_arr(), fault(F_RANGE, "firwin2", "the count of taps stands outside one to the limit");
    }
    if freq.len != gain.len || freq.len < 2 || !(fs > 0.0) {
        return empty_arr(), fault(F_SHAPE, "firwin2", "freq and gain must have one length of two or more, and the rate must be above zero");
    }
    let nyq = 0.5 * fs;
    var nf = nfreqs;
    if nf == 0 {
        var p: i64 = 1;
        while p < numtaps { p *= 2; }
        nf = 1 + p;
    }
    if numtaps >= nf || nf > MAX_DSP_LENGTH / 4 {
        return empty_arr(), fault(F_RANGE, "firwin2", "numtaps must be below nfreqs, and nfreqs must stay within the limit");
    }
    if freq[0] != 0.0 || freq[freq.len - 1] != nyq {
        return empty_arr(), fault(F_VALUE, "firwin2", "freq must start with 0 and end with fs / 2");
    }
    var k: i64 = 0;
    while k < freq.len {
        if !(freq[k] - freq[k] == 0.0) || !(gain[k] - gain[k] == 0.0) {
            return empty_arr(), fault(F_VALUE, "firwin2", "a value is not finite");
        }
        if k > 0 && freq[k] < freq[k - 1] {
            return empty_arr(), fault(F_VALUE, "firwin2", "the values in freq must not fall");
        }
        if k > 1 && freq[k] == freq[k - 1] && freq[k - 1] == freq[k - 2] {
            return empty_arr(), fault(F_VALUE, "firwin2", "a value in freq must not occur more than twice");
        }
        k += 1;
    }
    if freq[1] == 0.0 || freq[freq.len - 2] == nyq {
        return empty_arr(), fault(F_VALUE, "firwin2", "the value 0 and the value fs / 2 must not repeat");
    }
    var ftype: i64 = 1;
    if antisymmetric { ftype = 3; }
    if numtaps % 2 == 0 { ftype += 1; }
    let g0 = gain[0];
    let g1 = gain[gain.len - 1];
    if (ftype == 2 && g1 != 0.0) || (ftype == 3 && (g0 != 0.0 || g1 != 0.0)) || (ftype == 4 && g0 != 0.0) {
        return empty_arr(), fault(F_VALUE, "firwin2", "the gain at zero or at fs / 2 does not suit the type of the filter");
    }
    // A repeated frequency moves apart by a small step, so that the
    // interpolation has a rising row.
    var fq: f64[>..] = [];
    k = 0;
    while k < freq.len { fq.push(freq[k]); k += 1; }
    let eps = 2.220446049250313e-16 * nyq;
    k = 0;
    while k + 1 < fq.len {
        if freq[k] == freq[k + 1] {
            fq[k] = fq[k] - eps;
            fq[k + 1] = fq[k + 1] + eps;
        }
        k += 1;
    }
    k = 0;
    while k + 1 < fq.len {
        if !(fq[k + 1] > fq[k]) {
            return empty_arr(), fault(F_VALUE, "firwin2", "freq holds numbers too close to a repeated value");
        }
        k += 1;
    }
    var x: f64[>..] = [];
    k = 0;
    while k < nf { x.push(nyq * k / (nf - 1)); k += 1; }
    let fx, ff = interp(gain, fq, x);
    if ff.kind != F_NONE { return empty_arr(), ff; }
    var re: f64[>..] = [];
    var im: f64[>..] = [];
    k = 0;
    while k < nf {
        let ph = -(numtaps - 1) / 2.0 * 3.141592653589793 * x[k] / nyq;
        var cr = fx.data[k] * mathshim::cosine(ph);
        var ci = fx.data[k] * mathshim::sine(ph);
        if ftype > 2 { let t = cr; cr = -ci; ci = t; }
        re.push(cr);
        im.push(ci);
        k += 1;
    }
    let full, fr = irfft(re, im, 2 * (nf - 1));
    if fr.kind != F_NONE { return empty_arr(), fr; }
    var out: f64[>..] = [];
    var wd: f64[>..] = [];
    if window == "none" {
        k = 0;
        while k < numtaps { wd.push(1.0); k += 1; }
    } else {
        let w, wf = get_window(window, numtaps, false);
        if wf.kind != F_NONE { return empty_arr(), wf; }
        k = 0;
        while k < numtaps { wd.push(w.data[k]); k += 1; }
    }
    k = 0;
    while k < numtaps { out.push(full.data[k] * wd[k]); k += 1; }
    if ftype == 3 { out[numtaps / 2] = 0.0; }
    return line_arr(copy(out)), no_fault();
}

// The limits of remez. The grid holds at most MAX_REMEZ_GRID points, so
// the cost of one pass stays bounded.
const MAX_REMEZ_GRID: i64 = 262144;
const MAX_REMEZ_TAPS: i64 = 4096;
const MAX_REMEZ_PASSES: i64 = 1000;
const MAX_REMEZ_DENSITY: i64 = 1024;
const REMEZ_PI: f64 = 3.141592653589793;

// rz_lagrange gives the coefficient of the barycentric form for the
// node k of n nodes. The routine splits the product in m strides to
// keep the product in range. The arrays of remez hold the index one
// first, so slot zero stays unused.
fn rz_lagrange(k: i64, n: i64, m: i64, x: f64[:]) -> f64 {
    var r: f64 = 1.0;
    let q = x[k];
    var l: i64 = 1;
    while l <= m {
        var j = l;
        while j <= n {
            if j != k { r = r * 2.0 * (q - x[j]); }
            j += m;
        }
        l += 1;
    }
    return 1.0 / r;
}

// rz_eval gives the response at the grid point k by the barycentric
// form over the n extremal nodes.
fn rz_eval(k: i64, n: i64, grid: f64[:], x: f64[:], y: f64[:], ad: f64[:]) -> f64 {
    var d: f64 = 0.0;
    var p: f64 = 0.0;
    let xf = mathshim::cosine(2.0 * REMEZ_PI * grid[k]);
    var j: i64 = 1;
    while j <= n {
        let c = ad[j] / (xf - x[j]);
        d += c;
        p += c * y[j];
        j += 1;
    }
    return p / d;
}

// rz_err gives the weighted error at the grid point l.
fn rz_err(l: i64, nz: i64, grid: f64[:], x: f64[:], y: f64[:], ad: f64[:], des: f64[:], wt: f64[:]) -> f64 {
    return (rz_eval(l, nz, grid, x, y, ad) - des[l]) * wt[l];
}

// rz_exchange runs the exchange of the extremal set. The set iext holds
// nfcns + 1 grid indexes and one slot more. The arrays x, y and ad keep
// the nodes, the values and the weights of the last pass. The result is
// 0 when the exchange ends, -1 when the deviation stops rising, and -2
// when the count of steps passes the bound. The block of each label of
// the original routine is one branch of the loop. The variable pc names
// the next label, and 999 leaves the loop.
fn rz_exchange(des: f64[:], grid: f64[:], wt: f64[:], ngrid: i64, iext: i64[>..]&, nfcns: i64, itrmax: i64, x: f64[>..]&, y: f64[>..]&, ad: f64[>..]&) -> i64 {
    let nz = nfcns + 1;
    let nzz = nfcns + 2;
    var devl: f64 = -1.0;
    var dev: f64 = 0.0;
    var niter: i64 = 0;
    var j: i64 = 0;
    var k1: i64 = 0;
    var knz: i64 = 0;
    var klow: i64 = 0;
    var kup: i64 = 0;
    var l: i64 = 0;
    var luck: i64 = 0;
    var jchnge: i64 = 0;
    var nu: f64 = 0.0;
    var nut: f64 = 0.0;
    var nut1: f64 = 0.0;
    var ynz: f64 = 0.0;
    var comp: f64 = 0.0;
    var y1: f64 = 0.0;
    var err: f64 = 0.0;
    var pc: i64 = 100;
    var steps: i64 = 0;
    let bound = (itrmax + 1) * 16 * (ngrid + nz + 16);
    while pc != 999 {
        steps += 1;
        if steps > bound { return -2; }
        if pc == 100 {
            iext[nzz] = ngrid + 1;
            niter += 1;
            if niter > itrmax { pc = 999; }
            else {
                j = 1;
                while j <= nz { x[j] = mathshim::cosine(2.0 * REMEZ_PI * grid[iext[j]]); j += 1; }
                let jet = (nfcns - 1) / 15 + 1;
                j = 1;
                while j <= nz { ad[j] = rz_lagrange(j, nz, jet, x); j += 1; }
                var dnum: f64 = 0.0;
                var dden: f64 = 0.0;
                var kf: f64 = 1.0;
                j = 1;
                while j <= nz {
                    l = iext[j];
                    dnum += ad[j] * des[l];
                    dden += kf * ad[j] / wt[l];
                    kf = -kf;
                    j += 1;
                }
                dev = dnum / dden;
                nu = 1.0;
                if dev > 0.0 { nu = -1.0; }
                dev = -nu * dev;
                kf = nu;
                j = 1;
                while j <= nz {
                    l = iext[j];
                    y[j] = des[l] + kf * dev / wt[l];
                    kf = -kf;
                    j += 1;
                }
                if dev <= devl { return -1; }
                devl = dev;
                jchnge = 0;
                k1 = iext[1];
                knz = iext[nz];
                klow = 0;
                nut = -nu;
                j = 1;
                pc = 200;
            }
        } else if pc == 200 {
            if j == nzz { ynz = comp; }
            if j >= nzz { pc = 300; }
            else {
                kup = iext[j + 1];
                l = iext[j] + 1;
                nut = -nut;
                if j == 2 { y1 = comp; }
                comp = dev;
                if l >= kup { pc = 220; }
                else {
                    err = rz_err(l, nz, grid, x, y, ad, des, wt);
                    if nut * err - comp <= 0.0 { pc = 220; }
                    else { comp = nut * err; pc = 210; }
                }
            }
        } else if pc == 210 {
            l += 1;
            if l >= kup { pc = 215; }
            else {
                err = rz_err(l, nz, grid, x, y, ad, des, wt);
                if nut * err - comp <= 0.0 { pc = 215; }
                else { comp = nut * err; }
            }
        } else if pc == 215 {
            iext[j] = l - 1;
            j += 1;
            klow = l - 1;
            jchnge += 1;
            pc = 200;
        } else if pc == 220 {
            l -= 1;
            pc = 225;
        } else if pc == 225 {
            l -= 1;
            if l <= klow { pc = 250; }
            else {
                err = rz_err(l, nz, grid, x, y, ad, des, wt);
                if nut * err - comp > 0.0 { pc = 230; }
                else if jchnge > 0 { pc = 260; }
            }
        } else if pc == 230 {
            comp = nut * err;
            pc = 235;
        } else if pc == 235 {
            l -= 1;
            if l <= klow { pc = 240; }
            else {
                err = rz_err(l, nz, grid, x, y, ad, des, wt);
                if nut * err - comp <= 0.0 { pc = 240; }
                else { comp = nut * err; }
            }
        } else if pc == 240 {
            klow = iext[j];
            iext[j] = l + 1;
            j += 1;
            jchnge += 1;
            pc = 200;
        } else if pc == 250 {
            l = iext[j] + 1;
            if jchnge > 0 { pc = 215; } else { pc = 255; }
        } else if pc == 255 {
            l += 1;
            if l >= kup { pc = 260; }
            else {
                err = rz_err(l, nz, grid, x, y, ad, des, wt);
                if nut * err - comp > 0.0 { comp = nut * err; pc = 210; }
            }
        } else if pc == 260 {
            klow = iext[j];
            j += 1;
            pc = 200;
        } else if pc == 300 {
            if j > nzz { pc = 320; }
            else {
                if k1 > iext[1] { k1 = iext[1]; }
                if knz < iext[nz] { knz = iext[nz]; }
                nut1 = nut;
                nut = -nu;
                l = 0;
                kup = k1;
                comp = ynz * 1.00001;
                luck = 1;
                pc = 310;
            }
        } else if pc == 310 {
            l += 1;
            if l >= kup { pc = 315; }
            else {
                err = rz_err(l, nz, grid, x, y, ad, des, wt);
                if nut * err - comp > 0.0 { comp = nut * err; j = nzz; pc = 210; }
            }
        } else if pc == 315 {
            luck = 6;
            pc = 325;
        } else if pc == 320 {
            if luck > 9 { pc = 350; }
            else {
                if comp > y1 { y1 = comp; }
                k1 = iext[nzz];
                pc = 325;
            }
        } else if pc == 325 {
            l = ngrid + 1;
            klow = knz;
            nut = -nut1;
            comp = y1 * 1.00001;
            pc = 330;
        } else if pc == 330 {
            l -= 1;
            if l <= klow { pc = 340; }
            else {
                err = rz_err(l, nz, grid, x, y, ad, des, wt);
                if nut * err - comp > 0.0 { j = nzz; comp = nut * err; luck += 10; pc = 235; }
            }
        } else if pc == 340 {
            if luck == 6 { pc = 370; }
            else {
                j = 1;
                while j <= nfcns { iext[nzz - j] = iext[nz - j]; j += 1; }
                iext[1] = k1;
                pc = 100;
            }
        } else if pc == 350 {
            let kn = iext[nzz];
            j = 1;
            while j <= nfcns { iext[j] = iext[j + 1]; j += 1; }
            iext[nz] = kn;
            pc = 100;
        } else {
            // the label 370: the loop ends when no value changed
            if jchnge > 0 { pc = 100; } else { pc = 999; }
        }
    }
    return 0;
}

// rz_coeffs reads the cosine coefficients of the best fit by the inverse
// transform at the nfcns points. It writes them into alpha, with the
// index one first. The grid slot one changes during the call and comes
// back.
fn rz_coeffs(grid: f64[>..]&, ngrid: i64, edge: f64[:], nbands: i64, nfcns: i64, x: f64[>..]&, y: f64[:], ad: f64[:], alpha: f64[>..]&) {
    let nz = nfcns + 1;
    let nzz = nfcns + 2;
    let nm1 = nfcns - 1;
    let tp = 2.0 * REMEZ_PI;
    let gtemp = grid[1];
    x[nzz] = -2.0;
    let cn = 2 * nfcns - 1;
    let delf = 1.0 / cn;
    var l: i64 = 1;
    var kkk: i64 = 0;
    var aa: f64 = 0.0;
    var bb: f64 = 0.0;
    var a: f64[>..] = [];
    var p: f64[>..] = [];
    var q: f64[>..] = [];
    var i: i64 = 0;
    while i < nfcns + 4 { a.push(0.0); p.push(0.0); q.push(0.0); i += 1; }
    if edge[1] == 0.0 && edge[2 * nbands] == 0.5 { kkk = 1; }
    if nfcns <= 3 { kkk = 1; }
    if kkk != 1 {
        let dtemp = mathshim::cosine(tp * grid[1]);
        let dnum = mathshim::cosine(tp * grid[ngrid]);
        aa = 2.0 / (dtemp - dnum);
        bb = -(dtemp + dnum) / (dtemp - dnum);
    }
    var j: i64 = 1;
    while j <= nfcns {
        var ft = (j - 1) * delf;
        var xt = mathshim::cosine(tp * ft);
        if kkk != 1 {
            xt = (xt - bb) / aa;
            ft = mathshim::arccosine_of(xt) / tp;
        }
        var st: i64 = 410;
        while st != 425 {
            if st == 410 {
                let xe = x[l];
                if xt > xe { st = 420; }
                else if xe - xt < 1.0e-6 { st = 415; }
                else { l += 1; }
            } else if st == 415 {
                a[j] = y[l];
                st = 425;
            } else {
                let xe = x[l];
                if xt - xe < 1.0e-6 { st = 415; }
                else {
                    grid[1] = ft;
                    a[j] = rz_eval(1, nz, grid, x, y, ad);
                    st = 425;
                }
            }
        }
        if l > 1 { l -= 1; }
        j += 1;
    }
    grid[1] = gtemp;
    let dden = tp / cn;
    j = 1;
    while j <= nfcns {
        var dtemp: f64 = 0.0;
        let dnum = (j - 1) * dden;
        var k: i64 = 1;
        while k <= nm1 { dtemp += a[k + 1] * mathshim::cosine(dnum * k); k += 1; }
        alpha[j] = 2.0 * dtemp + a[1];
        j += 1;
    }
    j = 2;
    while j <= nfcns { alpha[j] = alpha[j] * 2.0 / cn; j += 1; }
    alpha[1] = alpha[1] / cn;
    if kkk != 1 {
        p[1] = 2.0 * alpha[nfcns] * bb + alpha[nm1];
        p[2] = 2.0 * aa * alpha[nfcns];
        q[1] = alpha[nfcns - 2] - alpha[nfcns];
        j = 2;
        while j <= nm1 {
            if j >= nm1 { aa = aa * 0.5; bb = bb * 0.5; }
            p[j + 1] = 0.0;
            var k: i64 = 1;
            while k <= j { a[k] = p[k]; p[k] = 2.0 * bb * a[k]; k += 1; }
            p[2] += a[1] * 2.0 * aa;
            k = 1;
            while k <= j - 1 { p[k] += q[k] + aa * a[k + 1]; k += 1; }
            k = 3;
            while k <= j + 1 { p[k] += aa * a[k - 1]; k += 1; }
            if j != nm1 {
                k = 1;
                while k <= j { q[k] = -a[k]; k += 1; }
                q[1] += alpha[nfcns - 1 - j];
            }
            j += 1;
        }
        j = 1;
        while j <= nfcns { alpha[j] = p[j]; j += 1; }
    }
    if nfcns <= 3 {
        alpha[nfcns + 1] = 0.0;
        alpha[nfcns + 2] = 0.0;
    }
}

// remez designs the taps of the optimal filter of Parks and McClellan.
// The bands hold two edges for each value of desired, in the units of
// fs, and the edges rise. The weight holds one value for each band, or
// it stays empty for a weight of one. The kind is bandpass,
// differentiator or hilbert. The exchange stops after maxiter passes.
// It gives a fault when the deviation stops rising.
fn remez(numtaps: i64, bands: f64[:], desired: f64[:], weight: f64[:] = [], kind: const u8[:] = "bandpass", maxiter: i64 = 25, grid_density: i64 = 16, fs: f64 = 1.0) -> Arr, Fault {
    var jtype: i64 = 0;
    if kind == "bandpass" { jtype = 1; }
    else if kind == "differentiator" { jtype = 2; }
    else if kind == "hilbert" { jtype = 3; }
    else { return empty_arr(), fault(F_VALUE, "remez", "the kind is not bandpass, differentiator or hilbert"); }
    if numtaps < 2 {
        return empty_arr(), fault(F_RANGE, "remez", "the count of taps is below two");
    }
    if numtaps > MAX_REMEZ_TAPS {
        return empty_arr(), fault(F_LIMIT, "remez", "the count of taps is above 4096");
    }
    if maxiter < 1 || maxiter > MAX_REMEZ_PASSES {
        return empty_arr(), fault(F_RANGE, "remez", "maxiter stands outside one to 1000");
    }
    if grid_density < 1 || grid_density > MAX_REMEZ_DENSITY {
        return empty_arr(), fault(F_RANGE, "remez", "grid_density stands outside one to 1024");
    }
    if !(fs > 0.0) || fs - fs != 0.0 {
        return empty_arr(), fault(F_VALUE, "remez", "fs is not a finite value above zero");
    }
    let nbands = desired.len;
    if nbands == 0 || bands.len != 2 * nbands || (weight.len != 0 && weight.len != nbands) {
        return empty_arr(), fault(F_SHAPE, "remez", "bands holds two edges for each value of desired, and weight one value for each band");
    }
    // The arrays keep the index one first, as the routine does.
    var edge: f64[>..] = [0.0];
    var fx: f64[>..] = [0.0];
    var wtx: f64[>..] = [0.0];
    var old: f64 = 0.0;
    var i: i64 = 0;
    while i < bands.len {
        let b = bands[i];
        if !(b >= old) || b - b != 0.0 {
            return empty_arr(), fault(F_VALUE, "remez", "the bands do not rise from zero");
        }
        if b * 2.0 > fs {
            return empty_arr(), fault(F_RANGE, "remez", "a band edge lies above half of fs");
        }
        old = b;
        edge.push(b / fs);
        i += 1;
    }
    i = 0;
    while i < nbands {
        let d = desired[i];
        if d - d != 0.0 {
            return empty_arr(), fault(F_VALUE, "remez", "desired holds a value that is not finite");
        }
        fx.push(d);
        var w: f64 = 1.0;
        if weight.len != 0 { w = weight[i]; }
        if !(w > 0.0) || w - w != 0.0 {
            return empty_arr(), fault(F_VALUE, "remez", "a weight is not a finite value above zero");
        }
        wtx.push(w);
        i += 1;
    }
    let neg = jtype != 1;
    let nodd = numtaps % 2 == 1;
    var nfcns = numtaps / 2;
    if nodd && !neg { nfcns += 1; }
    let dimsize = (numtaps + 1) / 2 + 2;
    let wrksize = grid_density * dimsize;
    if wrksize > MAX_REMEZ_GRID {
        return empty_arr(), fault(F_LIMIT, "remez", "the grid would hold more points than the limit, so reduce grid_density or the count of taps");
    }
    // The dense grid and the desired response and the weight on it.
    var des: f64[>..] = [0.0];
    var wt: f64[>..] = [0.0];
    let delf = 0.5 / (grid_density * nfcns * 1.0);
    var first = edge[1];
    if neg && first < delf { first = delf; }
    // The grid g keeps the index one first, so slot zero stays unused.
    var g: f64[>..] = [0.0, first];
    var j: i64 = 1;
    var l: i64 = 1;
    var lband: i64 = 1;
    des.push(0.0);
    wt.push(0.0);
    while true {
        let fup = edge[l + 1];
        var more = true;
        while more {
            let temp = g[j];
            var dv = fx[lband];
            var wv = wtx[lband];
            if jtype == 2 {
                dv = fx[lband] * temp;
                if fx[lband] >= 0.0001 { wv = wtx[lband] / temp; }
            }
            des[j] = dv;
            wt[j] = wv;
            j += 1;
            if j > wrksize {
                return empty_arr(), fault(F_LIMIT, "remez", "the grid holds too many points, so reduce grid_density or the count of taps");
            }
            g.push(temp + delf);
            des.push(0.0);
            wt.push(0.0);
            more = g[j] <= fup;
        }
        g[j - 1] = fup;
        var d2 = fx[lband];
        var w2 = wtx[lband];
        if jtype == 2 {
            d2 = fx[lband] * fup;
            if fx[lband] >= 0.0001 { w2 = wtx[lband] / fup; }
        }
        des[j - 1] = d2;
        wt[j - 1] = w2;
        lband += 1;
        l += 2;
        if lband > nbands { break; }
        g[j] = edge[l];
    }
    var ngrid = j - 1;
    if neg == nodd {
        if g[ngrid] > 0.5 - delf { ngrid -= 1; }
    }
    if ngrid < nfcns + 1 {
        return empty_arr(), fault(F_RANGE, "remez", "the bands are too close for the grid to hold the extremal set");
    }
    // The equivalent problem of the cosine series.
    j = 1;
    while j <= ngrid {
        var change: f64 = 1.0;
        if !neg { if !nodd { change = mathshim::cosine(REMEZ_PI * g[j]); } }
        else if !nodd { change = mathshim::sine(REMEZ_PI * g[j]); }
        else { change = mathshim::sine(2.0 * REMEZ_PI * g[j]); }
        des[j] = des[j] / change;
        wt[j] = wt[j] * change;
        j += 1;
    }
    var iext: i64[>..] = [];
    i = 0;
    while i < nfcns + 4 { iext.push(0); i += 1; }
    let temp = (ngrid - 1) * 1.0 / nfcns;
    j = 1;
    while j <= nfcns { iext[j] = ((j - 1) * temp) as i64 + 1; j += 1; }
    iext[nfcns + 1] = ngrid;
    var x: f64[>..] = [];
    var y: f64[>..] = [];
    var ad: f64[>..] = [];
    var alpha: f64[>..] = [];
    i = 0;
    while i < nfcns + 4 { x.push(0.0); y.push(0.0); ad.push(0.0); alpha.push(0.0); i += 1; }
    let st = rz_exchange(des, g, wt, ngrid, iext, nfcns, maxiter, x, y, ad);
    if st == -1 {
        return empty_arr(), fault(F_VALUE, "remez", "the exchange failed to converge, so reduce the width of the transition bands");
    }
    if st == -2 {
        return empty_arr(), fault(F_LIMIT, "remez", "the exchange passed the bound of its steps");
    }
    rz_coeffs(g, ngrid, edge, nbands, nfcns, x, y, ad, alpha);
    // The taps.
    let nz = nfcns + 1;
    let nm1 = nfcns - 1;
    var h: f64[>..] = [];
    i = 0;
    while i < numtaps + 2 { h.push(0.0); i += 1; }
    if !neg {
        if nodd {
            j = 1;
            while j <= nm1 { h[j] = 0.5 * alpha[nz - j]; j += 1; }
            h[nfcns] = alpha[1];
        } else {
            h[1] = 0.25 * alpha[nfcns];
            j = 2;
            while j <= nm1 { h[j] = 0.25 * (alpha[nz - j] + alpha[nfcns + 2 - j]); j += 1; }
            h[nfcns] = 0.5 * alpha[1] + 0.25 * alpha[2];
        }
    } else {
        if nodd {
            h[1] = 0.25 * alpha[nfcns];
            h[2] = 0.25 * alpha[nm1];
            j = 3;
            while j <= nm1 { h[j] = 0.25 * (alpha[nz - j] - alpha[nfcns + 3 - j]); j += 1; }
            h[nfcns] = 0.5 * alpha[1] - 0.25 * alpha[3];
            h[nz] = 0.0;
        } else {
            h[1] = 0.25 * alpha[nfcns];
            j = 2;
            while j <= nm1 { h[j] = 0.25 * (alpha[nz - j] - alpha[nfcns + 2 - j]); j += 1; }
            h[nfcns] = 0.5 * alpha[1] - 0.25 * alpha[2];
        }
    }
    j = 1;
    while j <= nfcns {
        let k = numtaps + 1 - j;
        if !neg { h[k] = h[j]; } else { h[k] = -h[j]; }
        j += 1;
    }
    if neg && nodd { h[nz] = 0.0; }
    var out: f64[>..] = [];
    j = 1;
    while j <= numtaps { out.push(h[j]); j += 1; }
    return line_arr(copy(out)), no_fault();
}
