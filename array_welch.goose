// Welch power spectrum with the full options of scipy.signal.welch:
// a window by name, with its parameter, or by values, a transform
// length above the size of a part, a constant or linear trend, the
// spectrum of one side or of two sides, and the mean or the median of
// the parts.
namespace scientific;

import mathshim;

// welch_median_bias gives the bias of the median of n spectra against
// their mean.
fn welch_median_bias(n: i64) -> f64 {
    var t: f64 = 1.0;
    var i: i64 = 1;
    while i <= (n - 1) / 2 {
        let ii = 2.0 * i;
        t += 1.0 / (ii + 1.0) - 1.0 / ii;
        i += 1;
    }
    return t;
}

// welch_median gives the median of a list. The list is sorted in place.
fn welch_median(v: f64[>..]&) -> f64 {
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

// welch_ex gives the frequencies and the power of a signal, as SciPy
// welch does. The window is a name of get_window with its parameter
// param, or the values in win when win is not empty. The count nfft
// of zero takes the size of a part. The trend is "constant", "linear"
// or "none". The scaling is "density" or "spectrum". The average is
// "mean" or "median". A signal shorter than a part is a fault.
fn welch_ex(x: f64[:], fs: f64 = 1.0, window: const u8[:] = "hann", param: f64 = 0.0 / 0.0, win: f64[:] = [], nperseg: i64 = 256, noverlap: i64 = -1, nfft: i64 = 0, trend: const u8[:] = "constant", onesided: bool = true, scaling: const u8[:] = "density", average: const u8[:] = "mean") -> Arr, Arr, Fault {
    let tag = "welch_ex";
    if !(fs > 0.0) || !(fs - fs == 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the rate must be above zero");
    }
    if scaling != "density" && scaling != "spectrum" {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the scaling holds an unknown name");
    }
    if average != "mean" && average != "median" {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the average holds an unknown name");
    }
    if trend != "constant" && trend != "linear" && trend != "none" {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the trend holds an unknown name");
    }
    if nperseg < 2 || nperseg > x.len || nperseg > 4194304 {
        return empty_arr(), empty_arr(), fault(F_RANGE, tag, "the size of a part must hold two or more values and no more than the signal");
    }
    var ov = noverlap;
    if ov < 0 { ov = nperseg / 2; }
    if ov >= nperseg {
        return empty_arr(), empty_arr(), fault(F_RANGE, tag, "the overlap must stand from zero to below the size of a part");
    }
    var nf = nfft;
    if nf == 0 { nf = nperseg; }
    if nf < nperseg || nf > 4194304 {
        return empty_arr(), empty_arr(), fault(F_RANGE, tag, "the transform length must hold at least the size of a part, and at most the limit");
    }
    var k: i64 = 0;
    while k < x.len {
        if !(x[k] - x[k] == 0.0) {
            return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the signal holds a value that is not finite");
        }
        k += 1;
    }
    var w: f64[>..] = [];
    if win.len > 0 {
        if win.len != nperseg {
            return empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the window values must hold the size of a part");
        }
        k = 0;
        while k < win.len {
            if !(win[k] - win[k] == 0.0) {
                return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the window holds a value that is not finite");
            }
            w.push(win[k]);
            k += 1;
        }
    } else {
        let gw, wf = get_window(window, nperseg, true, param);
        if wf.kind != F_NONE { return empty_arr(), empty_arr(), wf; }
        k = 0;
        while k < nperseg { w.push(gw.data[k]); k += 1; }
    }
    var sw: f64 = 0.0;
    var sq: f64 = 0.0;
    k = 0;
    while k < nperseg { sw += w[k]; sq += w[k] * w[k]; k += 1; }
    var scale = 1.0 / (sw * sw);
    if scaling == "density" { scale = 1.0 / (fs * sq); }
    if !(scale - scale == 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the window gives a scale that is not finite");
    }
    let step = nperseg - ov;
    let nseg = (x.len - ov) / step;
    var bins = nf;
    if onesided { bins = nf / 2 + 1; }
    var m: f64[>..] = [];
    k = 0;
    while k < bins * nseg { m.push(0.0); k += 1; }
    var s: i64 = 0;
    while s < nseg {
        let st = s * step;
        var mean: f64 = 0.0;
        var slope: f64 = 0.0;
        let mt = (nperseg - 1) / 2.0;
        if trend != "none" {
            k = 0;
            while k < nperseg { mean += x[st + k]; k += 1; }
            mean = mean / nperseg;
        }
        if trend == "linear" {
            var num: f64 = 0.0;
            var den: f64 = 0.0;
            k = 0;
            while k < nperseg {
                num += (k - mt) * (x[st + k] - mean);
                den += (k - mt) * (k - mt);
                k += 1;
            }
            slope = num / den;
        }
        var part: f64[>..] = [];
        k = 0;
        while k < nperseg {
            part.push((x[st + k] - mean - slope * (k - mt)) * w[k]);
            k += 1;
        }
        while part.len < nf { part.push(0.0); }
        let re, im, f = fft(part);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
        k = 0;
        while k < bins {
            var twice: f64 = 1.0;
            if onesided && k > 0 && !(nf % 2 == 0 && k == bins - 1) { twice = 2.0; }
            m[k * nseg + s] = twice * scale * (re.data[k] * re.data[k] + im.data[k] * im.data[k]);
            k += 1;
        }
        s += 1;
    }
    var p: f64[>..] = [];
    k = 0;
    while k < bins {
        if average == "mean" {
            var t: f64 = 0.0;
            s = 0;
            while s < nseg { t += m[k * nseg + s]; s += 1; }
            p.push(t / nseg);
        } else {
            var row: f64[>..] = [];
            s = 0;
            while s < nseg { row.push(m[k * nseg + s]); s += 1; }
            p.push(welch_median(row) / welch_median_bias(nseg));
        }
        k += 1;
    }
    var fr: f64[>..] = [];
    k = 0;
    while k < bins {
        var idx = k;
        if !onesided && k >= (nf + 1) / 2 { idx = k - nf; }
        fr.push(idx * fs / nf);
        k += 1;
    }
    return line_arr(copy(fr)), line_arr(copy(p)), no_fault();
}
