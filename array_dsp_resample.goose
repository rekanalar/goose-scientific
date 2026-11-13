// The resampling, the response of a cascade and the design by frequency sampling.
//
namespace scientific;

import mathshim;

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

