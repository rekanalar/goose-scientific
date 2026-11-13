// The design of a filter by the window method, the response, the convolution and the median filter.
//
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

