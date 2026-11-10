// The functions of a signal.
//
// The functions take the values of a signal. The transform of the
// package takes the parts of a signal by their frequency.
namespace scientific;

import mathshim;

// convolve gives the overlap of two signals. The method is the full
// overlap, as NumPy holds under its default. The kernel turns
// around, as NumPy does. The result holds a count that is the sum of
// the two counts minus one.
fn convolve(a: f64[:], kernel: f64[:]) -> Arr, Fault {
    let n = a.len;
    let m = kernel.len;
    if n < 1 || m < 1 {
        return empty_arr(), fault(F_VALUE, "convolve", "a signal holds no value");
    }
    if n + m - 1 > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "convolve", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n + m - 1 {
        var t: f64 = 0.0;
        // The product of the two signals at every shift. The value of
        // the kernel at the place of the shift comes in the order of
        // the kernel, as NumPy does.
        let lo = i - m + 1;
        var hi = i;
        if hi > n - 1 { hi = n - 1; }
        var j = lo;
        if j < 0 { j = 0; }
        while j <= hi {
            t += a[j] * kernel[i - j];
            j += 1;
        }
        d.push(t);
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// correlate gives the match of a pattern in a signal. The match is
// the overlap with the pattern turned around.
fn correlate(a: f64[:], pattern: f64[:]) -> Arr, Fault {
    let n = a.len;
    let m = pattern.len;
    if n < 1 || m < 1 {
        return empty_arr(), fault(F_VALUE, "correlate", "a signal holds no value");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n + m - 1 {
        var t: f64 = 0.0;
        var k: i64 = 0;
        while k < m {
            let at = i - (m - 1 - k);
            if at >= 0 && at < n {
                t += a[at] * pattern[k];
            }
            k += 1;
        }
        d.push(t);
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// detrend takes the straight line that the least squares of the
// package fits under a signal, and gives the signal with the line
// taken out. The result holds the values of the signal about zero.
fn detrend(y: f64[:]) -> Arr, Fault {
    let n = y.len;
    if n < 2 {
        return empty_arr(), fault(F_VALUE, "detrend", "the signal holds fewer than two values");
    }
    // The design matrix holds a column of ones and a column of the
    // places of the values.
    var design: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        design.push(1.0);
        design.push(i);
        i += 1;
    }
    var ds: i64[..8] = [];
    ds.push(n);
    ds.push(2);
    let design_arr = Arr { ndim: 2, shape: ds, data: copy(design) };
    var ys: i64[..8] = [];
    ys.push(n);
    var ybuf: f64[>..] = [];
    i = 0;
    while i < n { ybuf.push(y[i]); i += 1; }
    let y_arr = Arr { ndim: 1, shape: ys, data: copy(ybuf) };
    // The least squares of the package gives the line.
    let coef, fc = lstsq(design_arr, y_arr);
    if fc.kind != F_NONE { return empty_arr(), fc; }
    // The signal holds the values with the line taken out.
    var d: f64[>..] = [];
    i = 0;
    while i < n {
        d.push(y[i] - (coef.data[0] + coef.data[1] * i));
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// welch gives the size of the parts of a signal by their frequency.
// The signal is cut into parts that overlap by half. Every part
// takes the window of Hann. The transform of the package takes every
// part, and the sizes of the parts come as the mean of the squares
// of the values of the transform. The result holds the sizes of the
// parts of the frequency from zero to half of the count of a part.
fn welch(y: f64[:], size: i64) -> Arr, Fault {
    if size < 2 {
        return empty_arr(), fault(F_VALUE, "welch", "the size of a part holds fewer than two");
    }
    let n = y.len;
    if n < size {
        return empty_arr(), fault(F_VALUE, "welch", "the signal holds fewer values than a part");
    }
    // The window of Hann over a part.
    var window: f64[>..] = [];
    var i: i64 = 0;
    while i < size {
        window.push(0.5 - 0.5 * mathshim::cosine(6.283185307179586 * i / size));
        i += 1;
    }
    // The parts start at every half of the size, so they overlap by
    // half.
    let hop = size / 2;
    var count: i64 = 0;
    var start: i64 = 0;
    while start + size <= n {
        count += 1;
        start += hop;
    }
    if count < 1 {
        return empty_arr(), fault(F_VALUE, "welch", "the signal holds no whole part");
    }
    // The sizes of the parts of the frequency. The transform of a
    // part holds a count of the size, and the sizes of the positive
    // frequencies hold half of it plus one.
    let bins = size / 2 + 1;
    var power: f64[>..] = [];
    i = 0;
    while i < bins { power.push(0.0); i += 1; }
    start = 0;
    while start + size <= n {
        // The part takes the window.
        var part: f64[>..] = [];
        i = 0;
        while i < size {
            part.push(y[start + i] * window[i]);
            i += 1;
        }
        let re, im, f = fft(part);
        if f.kind != F_NONE { return empty_arr(), f; }
        i = 0;
        while i < bins {
            power[i] += re.data[i] * re.data[i] + im.data[i] * im.data[i];
            i += 1;
        }
        start += hop;
    }
    // The mean of the squares of the values of the transform. The
    // scale of the window of Hann divides the sizes. The parts of
    // the frequency above zero hold twice the size, as SciPy holds
    // for a spectrum of one side.
    var scale: f64 = 0.0;
    i = 0;
    while i < size { scale += window[i] * window[i]; i += 1; }
    var d: f64[>..] = [];
    i = 0;
    while i < bins {
        var twice: f64 = 1.0;
        if i > 0 && i < bins - 1 { twice = 2.0; }
        if size % 2 == 1 && i == bins - 1 { twice = 2.0; }
        d.push(twice * power[i] / (count * scale));
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// spectral_core cuts the signal into parts, takes the window and the
// transform of each part and gives the power. The result is a matrix
// with one row per frequency and one column per part, as SciPy gives it.
fn spectral_core(x: f64[:], fs: f64, window: const u8[:], param: f64, nperseg: i64, noverlap: i64, detrend: bool, density: bool, tag: const u8[:]) -> Arr, Arr, Arr, Fault {
    if !(fs > 0.0) || !(fs - fs == 0.0) {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, tag, "the rate must be above zero");
    }
    if nperseg < 2 || nperseg > x.len || nperseg > 4194304 {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_RANGE, tag, "the size of a part must hold two or more values and no more than the signal");
    }
    if noverlap < 0 || noverlap >= nperseg {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_RANGE, tag, "the overlap must stand from zero to below the size of a part");
    }
    var k: i64 = 0;
    while k < x.len {
        if !(x[k] - x[k] == 0.0) {
            return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, tag, "the signal holds a value that is not finite");
        }
        k += 1;
    }
    let w, wf = get_window(window, nperseg, true, param);
    if wf.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), wf; }
    var sw: f64 = 0.0;
    var sq: f64 = 0.0;
    k = 0;
    while k < nperseg { sw += w.data[k]; sq += w.data[k] * w.data[k]; k += 1; }
    var scale = 1.0 / (sw * sw);
    if density { scale = 1.0 / (fs * sq); }
    let step = nperseg - noverlap;
    let nseg = (x.len - noverlap) / step;
    let bins = nperseg / 2 + 1;
    var m: f64[>..] = [];
    k = 0;
    while k < bins * nseg { m.push(0.0); k += 1; }
    var tm: f64[>..] = [];
    var s: i64 = 0;
    while s < nseg {
        let st = s * step;
        var mean: f64 = 0.0;
        if detrend {
            k = 0;
            while k < nperseg { mean += x[st + k]; k += 1; }
            mean = mean / nperseg;
        }
        var part: f64[>..] = [];
        k = 0;
        while k < nperseg { part.push((x[st + k] - mean) * w.data[k]); k += 1; }
        let re, im, f = fft(part);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
        k = 0;
        while k < bins {
            var twice: f64 = 2.0;
            if k == 0 || (nperseg % 2 == 0 && k == bins - 1) { twice = 1.0; }
            m[k * nseg + s] = twice * scale * (re.data[k] * re.data[k] + im.data[k] * im.data[k]);
            k += 1;
        }
        tm.push((nperseg / 2.0 + st) / fs);
        s += 1;
    }
    var fr: f64[>..] = [];
    k = 0;
    while k < bins { fr.push(k * fs / nperseg); k += 1; }
    return line_arr(copy(fr)), line_arr(copy(tm)), mat2(copy(m), bins, nseg), no_fault();
}

// spectrogram gives the power of the parts of a signal. The overlap of
// minus one takes the eighth part of the size, as SciPy does. The
// window is a name of get_window; SciPy takes a Tukey window of 0.25.
// It gives the frequencies, the times of the centres of the parts and
// the matrix of the power. The scaling is "density" or "spectrum".
fn spectrogram(x: f64[:], fs: f64 = 1.0, window: const u8[:] = "tukey", param: f64 = 0.25, nperseg: i64 = 256, noverlap: i64 = -1, detrend: bool = true, scaling: const u8[:] = "density") -> Arr, Arr, Arr, Fault {
    if scaling != "density" && scaling != "spectrum" {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "spectrogram", "the scaling holds an unknown name");
    }
    var ov = noverlap;
    if ov < 0 { ov = nperseg / 8; }
    return spectral_core(x, fs, window, param, nperseg, ov, detrend, scaling == "density", "spectrogram");
}

// welch_psd gives the mean power of the parts of a signal by their
// frequency, as SciPy welch does. The window is a name of get_window.
// The overlap of minus one takes half of the size.
fn welch_psd(x: f64[:], fs: f64 = 1.0, window: const u8[:] = "hann", nperseg: i64 = 256, noverlap: i64 = -1, detrend: bool = true, scaling: const u8[:] = "density") -> Arr, Arr, Fault {
    if scaling != "density" && scaling != "spectrum" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "welch_psd", "the scaling holds an unknown name");
    }
    var ov = noverlap;
    if ov < 0 { ov = nperseg / 2; }
    let fr, tm, m, f = spectral_core(x, fs, window, 0.0 / 0.0, nperseg, ov, detrend, scaling == "density", "welch_psd");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let bins = m.shape[0];
    let nseg = m.shape[1];
    var p: f64[>..] = [];
    var k: i64 = 0;
    while k < bins {
        var t: f64 = 0.0;
        var s: i64 = 0;
        while s < nseg { t += m.data[k * nseg + s]; s += 1; }
        p.push(t / nseg);
        k += 1;
    }
    return fr, line_arr(copy(p)), no_fault();
}

// stft gives the short time transform of a signal. It gives the
// frequencies, the times, and the real and the imaginary parts of the
// matrix of the transform, with one row per frequency. With boundary
// true the signal gets nperseg / 2 zeros at both ends. With padded true
// zeros at the end fill the last part. The window is a name of
// get_window. The overlap of minus one takes half of the size. No
// trend leaves the parts, and the values scale by one over the sum of
// the window, as SciPy does.
fn stft(x: f64[:], fs: f64 = 1.0, window: const u8[:] = "hann", nperseg: i64 = 256, noverlap: i64 = -1, boundary: bool = true, padded: bool = true) -> Arr, Arr, Arr, Arr, Fault {
    var ov = noverlap;
    if ov < 0 { ov = nperseg / 2; }
    if !(fs > 0.0) || !(fs - fs == 0.0) {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "stft", "the rate must be above zero");
    }
    if nperseg < 2 || nperseg > 4194304 || ov < 0 || ov >= nperseg {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_RANGE, "stft", "the size of a part or the overlap stands outside the range");
    }
    if x.len < 1 || x.len > MAX_DSP_LENGTH {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "stft", "the signal holds no value or too many values");
    }
    var sig: f64[>..] = [];
    var k: i64 = 0;
    var edge: i64 = 0;
    if boundary { edge = nperseg / 2; }
    while k < edge { sig.push(0.0); k += 1; }
    k = 0;
    while k < x.len {
        if !(x[k] - x[k] == 0.0) {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "stft", "the signal holds a value that is not finite");
        }
        sig.push(x[k]);
        k += 1;
    }
    k = 0;
    while k < edge { sig.push(0.0); k += 1; }
    let step = nperseg - ov;
    if padded {
        let rem = (sig.len - nperseg) % step;
        if sig.len >= nperseg && rem != 0 {
            var add = step - rem;
            while add > 0 { sig.push(0.0); add -= 1; }
        }
        if sig.len < nperseg {
            while sig.len < nperseg { sig.push(0.0); }
        }
    }
    if sig.len < nperseg {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_RANGE, "stft", "the signal is shorter than a part");
    }
    let w, wf = get_window(window, nperseg, true);
    if wf.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), wf; }
    var sw: f64 = 0.0;
    k = 0;
    while k < nperseg { sw += w.data[k]; k += 1; }
    let nseg = (sig.len - nperseg) / step + 1;
    let bins = nperseg / 2 + 1;
    var mr: f64[>..] = [];
    var mi: f64[>..] = [];
    k = 0;
    while k < bins * nseg { mr.push(0.0); mi.push(0.0); k += 1; }
    var tm: f64[>..] = [];
    var s: i64 = 0;
    while s < nseg {
        var part: f64[>..] = [];
        k = 0;
        while k < nperseg { part.push(sig[s * step + k] * w.data[k]); k += 1; }
        let re, im, f = fft(part);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), f; }
        k = 0;
        while k < bins {
            mr[k * nseg + s] = re.data[k] / sw;
            mi[k * nseg + s] = im.data[k] / sw;
            k += 1;
        }
        tm.push((s * step + nperseg / 2 - edge) / fs);
        s += 1;
    }
    var fr: f64[>..] = [];
    k = 0;
    while k < bins { fr.push(k * fs / nperseg); k += 1; }
    return line_arr(copy(fr)), line_arr(copy(tm)), mat2(copy(mr), bins, nseg), mat2(copy(mi), bins, nseg), no_fault();
}

// istft gives the signal of a short time transform, with the parts of
// the real matrix zr and the imaginary matrix zi, one row per frequency.
// It adds the parts by overlap and divides by the sum of the squares of
// the window. The window, the size and the overlap must equal those of
// stft. With boundary true it removes nperseg / 2 values at both ends.
// It gives the times and the signal.
fn istft(zr, zi, fs: f64 = 1.0, window: const u8[:] = "hann", nperseg: i64 = 256, noverlap: i64 = -1, boundary: bool = true) -> Arr, Arr, Fault {
    var ov = noverlap;
    if ov < 0 { ov = nperseg / 2; }
    if !(fs > 0.0) || !(fs - fs == 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "istft", "the rate must be above zero");
    }
    if nperseg < 2 || nperseg > 4194304 || ov < 0 || ov >= nperseg {
        return empty_arr(), empty_arr(), fault(F_RANGE, "istft", "the size of a part or the overlap stands outside the range");
    }
    if zr.ndim != 2 || zi.ndim != 2 || zr.shape[0] != zi.shape[0] || zr.shape[1] != zi.shape[1] || zr.shape[0] != nperseg / 2 + 1 || zr.shape[1] < 1 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "istft", "the matrices must agree and hold nperseg / 2 + 1 rows");
    }
    let nseg = zr.shape[1];
    let bins = zr.shape[0];
    let step = nperseg - ov;
    let outlen = nperseg + (nseg - 1) * step;
    if outlen > MAX_DSP_LENGTH {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "istft", "the signal would be too long");
    }
    let w, wf = get_window(window, nperseg, true);
    if wf.kind != F_NONE { return empty_arr(), empty_arr(), wf; }
    var sw: f64 = 0.0;
    var k: i64 = 0;
    while k < nperseg { sw += w.data[k]; k += 1; }
    var x: f64[>..] = [];
    var norm: f64[>..] = [];
    k = 0;
    while k < outlen { x.push(0.0); norm.push(0.0); k += 1; }
    var s: i64 = 0;
    while s < nseg {
        var cr: f64[>..] = [];
        var ci: f64[>..] = [];
        k = 0;
        while k < bins { cr.push(zr.data[k * nseg + s]); ci.push(zi.data[k * nseg + s]); k += 1; }
        let part, pf = irfft(cr, ci, nperseg);
        if pf.kind != F_NONE { return empty_arr(), empty_arr(), pf; }
        k = 0;
        while k < nperseg {
            x[s * step + k] += part.data[k] * sw * w.data[k];
            norm[s * step + k] += w.data[k] * w.data[k];
            k += 1;
        }
        s += 1;
    }
    var lo: i64 = 0;
    var hi = outlen;
    if boundary { lo = nperseg / 2; hi = outlen - nperseg / 2; }
    if hi <= lo {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "istft", "the transform is too short for the boundary");
    }
    var out: f64[>..] = [];
    var tm: f64[>..] = [];
    k = lo;
    while k < hi {
        if !(norm[k] > 1.0e-10) {
            return empty_arr(), empty_arr(), fault(F_VALUE, "istft", "the window does not meet the overlap condition");
        }
        out.push(x[k] / norm[k]);
        tm.push((k - lo) / fs);
        k += 1;
    }
    return line_arr(copy(tm)), line_arr(copy(out)), no_fault();
}
