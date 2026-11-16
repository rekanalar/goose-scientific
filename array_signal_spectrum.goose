// The chirp z-transform, the cross spectra and the envelope.
//
// czt and czt_points evaluate a signal on a spiral of points of the z
// plane. zoom_fft gives a part of the spectrum at a fine step. csd_ex
// and coherence follow welch_ex and give the cross spectrum and the
// squared coherence of two signals. correlation_lags names the lag of
// each place of a full, same or valid correlation. envelope gives the
// envelope of a signal through its analytic part, and the residual
// that the bandpass leaves out.
//
// The formulas follow SciPy 1.18.1 (scipy.signal).
namespace scientific;

import mathshim;

// czt_points gives the places of the z plane of a chirp z-transform
// of m points. The points are a times w to the power of minus k for
// k from 0 to m minus 1. The missing values of w and a give the
// defaults of the transform: w turns by 2 pi over m, and a holds one.
fn czt_points(m: i64, w_re: f64 = 0.0 / 0.0, w_im: f64 = 0.0 / 0.0,
              a_re: f64 = 0.0 / 0.0, a_im: f64 = 0.0 / 0.0) -> Arr, Arr, Fault {
    if m < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "czt_points", "the point count holds a value below one");
    }
    // The default step turns by 2 pi over the count of the points.
    let ang = -6.283185307179586 / (m as f64);
    var w = Cplx { re: mathshim::cosine(ang), im: mathshim::sine(ang) };
    var a = Cplx { re: 1.0, im: 0.0 };
    if w_re == w_re { w = Cplx { re: w_re, im: w_im }; }
    if a_re == a_re { a = Cplx { re: a_re, im: a_im }; }
    if c_abs(w) == 0.0 || c_abs(a) == 0.0 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "czt_points", "w and a must hold values of no zero");
    }
    var pr: f64[>..] = [];
    var pi: f64[>..] = [];
    // The point of the place k is a times w to the power of minus k.
    // The powers follow by multiplication, so that the step holds every
    // turn of its circle.
    var p = a;
    var k: i64 = 0;
    while k < m {
        pr.push(p.re);
        pi.push(p.im);
        p = c_mul(p, c_div(Cplx { re: 1.0, im: 0.0 }, w));
        k += 1;
    }
    return line_arr(copy(pr)), line_arr(copy(pi)), no_fault();
}

// czt gives the chirp z-transform of a signal at m points. The value
// of the place k holds the sum of x[n] times a to the minus n times w
// to the power of n times k. The missing values of w and a give the
// defaults of the transform: w turns by 2 pi over m, and a holds one.
// A transform with these defaults is the Fourier transform.
fn czt(x: f64[:], m: i64 = 0, w_re: f64 = 0.0 / 0.0, w_im: f64 = 0.0 / 0.0,
       a_re: f64 = 0.0 / 0.0, a_im: f64 = 0.0 / 0.0) -> Arr, Arr, Fault {
    if x.len < 1 || !all_finite(x) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "czt", "the signal must be a finite list");
    }
    var mm = m;
    if mm == 0 { mm = x.len; }
    if mm < 1 || mm > MAX_DSP_LENGTH {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "czt", "the point count is out of range");
    }
    // The default step turns by 2 pi over the count of the points.
    let ang = -6.283185307179586 / (mm as f64);
    var w = Cplx { re: mathshim::cosine(ang), im: mathshim::sine(ang) };
    var a = Cplx { re: 1.0, im: 0.0 };
    if w_re == w_re { w = Cplx { re: w_re, im: w_im }; }
    if a_re == a_re { a = Cplx { re: a_re, im: a_im }; }
    if c_abs(w) == 0.0 || c_abs(a) == 0.0 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "czt", "w and a must hold values of no zero");
    }
    let ore, oim, f = czt_core(x, mm, w, a);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    return ore, oim, no_fault();
}

// czt_core gives the chirp z-transform of a signal at m points. The
// value of the place k holds the sum of x[n] times a to the minus n
// times w to the power of n times k. The powers follow by
// multiplication, so that the step holds every turn of its circle.
fn czt_core(x: f64[:], m: i64, w: Cplx, a: Cplx) -> Arr, Arr, Fault {
    var ore: f64[>..] = [];
    var oim: f64[>..] = [];
    let one = Cplx { re: 1.0, im: 0.0 };
    let inva = c_div(one, a);
    var k: i64 = 0;
    while k < m {
        // The power of w to the power k.
        var wpow = one;
        var t: i64 = 0;
        while t < k { wpow = c_mul(wpow, w); t += 1; }
        var acc = Cplx { re: 0.0, im: 0.0 };
        // The factor of the place n is a to the minus n times w to
        // the power of n times k.
        var wa = one;
        var wk = one;
        var n: i64 = 0;
        while n < x.len {
            let pre = c_mul(wa, wk);
            acc = c_add(acc, c_scale(pre, x[n]));
            wa = c_mul(wa, inva);
            wk = c_mul(wk, wpow);
            n += 1;
        }
        ore.push(acc.re);
        oim.push(acc.im);
        k += 1;
    }
    return line_arr(copy(ore)), line_arr(copy(oim)), no_fault();
}

// zoom_fft gives m values of the spectrum of a signal between the two
// frequencies fn[0] and fn[1]. One value in fn gives the band from
// zero to that value. The step of the frequency is (fn[1] - fn[0])
// over fs times m minus one, or over fs times m with endpoint. The
// call gives the real and the imaginary part of the spectrum.
fn zoom_fft(x: f64[:], band: f64[:], m: i64 = 0, fs: f64 = 2.0,
            endpoint: bool = false) -> Arr, Arr, Fault {
    if x.len < 1 || !all_finite(x) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "zoom_fft", "the signal must be a finite list");
    }
    if !(fs > 0.0) || band.len < 1 || !all_finite(band) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "zoom_fft", "the rate and the band must be finite");
    }
    var f1: f64 = 0.0;
    var f2: f64 = 0.0;
    if band.len == 2 {
        f1 = band[0];
        f2 = band[1];
    } else if band.len == 1 {
        f2 = band[0];
    } else {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "zoom_fft", "the band holds one value or two");
    }
    var mm = m;
    if mm == 0 { mm = x.len; }
    if mm < 1 || mm > MAX_DSP_LENGTH {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "zoom_fft", "the point count is out of range");
    }
    var scale = (f2 - f1) / fs;
    if endpoint { scale = (f2 - f1) * mm / (fs * (mm - 1)); }
    // The step turns by the scale over the count of the points.
    let ang = -6.283185307179586 * scale / mm;
    let w = Cplx { re: mathshim::cosine(ang), im: mathshim::sine(ang) };
    let aang = 6.283185307179586 * f1 / fs;
    let a = Cplx { re: mathshim::cosine(aang), im: mathshim::sine(aang) };
    let ore, oim, f = czt_core(x, mm, w, a);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    return ore, oim, no_fault();
}

// csd_avg folds the parts of the cross spectrum into one spectrum.
// It gives the real and the imaginary part of conj(X) times Y, twice
// on the inner places of one side, times the scale. The average is
// the mean or the median of the parts. The median holds the bias of
// the median of the count of parts.
fn csd_avg(x: f64[:], y: f64[:], fs: f64, w: f64[:], nperseg: i64, noverlap: i64,
           nf: i64, detrend: bool, onesided: bool, scaling: const u8[:],
           average: const u8[:], tag: const u8[:]) -> Arr, Arr, Fault {
    var sw: f64 = 0.0;
    var sq: f64 = 0.0;
    var k: i64 = 0;
    while k < nperseg { sw += w[k]; sq += w[k] * w[k]; k += 1; }
    var scale = 1.0 / (sw * sw);
    if scaling == "density" { scale = 1.0 / (fs * sq); }
    let step = nperseg - noverlap;
    let nseg = (x.len - noverlap) / step;
    var acc_r: f64[>..] = [];
    var acc_i: f64[>..] = [];
    k = 0;
    while k < nf * nseg { acc_r.push(0.0); acc_i.push(0.0); k += 1; }
    var s: i64 = 0;
    while s < nseg {
        let st = s * step;
        var mx: f64 = 0.0;
        var my: f64 = 0.0;
        if detrend {
            k = 0;
            while k < nperseg { mx += x[st + k]; my += y[st + k]; k += 1; }
            mx = mx / nperseg;
            my = my / nperseg;
        }
        var px: f64[>..] = [];
        var py: f64[>..] = [];
        k = 0;
        while k < nperseg {
            px.push((x[st + k] - mx) * w[k]);
            py.push((y[st + k] - my) * w[k]);
            k += 1;
        }
        while px.len < nf { px.push(0.0); }
        while py.len < nf { py.push(0.0); }
        let xr, xi, f1 = fft(px);
        if f1.kind != F_NONE { return empty_arr(), empty_arr(), f1; }
        let yr, yi, f2 = fft(py);
        if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
        // The cross spectrum holds conj(X) times Y, as SciPy csd does.
        k = 0;
        while k < nf {
            acc_r[k * nseg + s] += xr.data[k] * yr.data[k] + xi.data[k] * yi.data[k];
            acc_i[k * nseg + s] += xr.data[k] * yi.data[k] - xi.data[k] * yr.data[k];
            k += 1;
        }
        s += 1;
    }
    var bins = nf;
    if onesided { bins = nf / 2 + 1; }
    var outr: f64[>..] = [];
    var outi: f64[>..] = [];
    // The bias of the median depends on the count of the parts alone,
    // so it is computed once. The mean does not need it.
    var bias: f64 = 1.0;
    if average != "mean" { bias = welch_median_bias(nseg); }
    k = 0;
    while k < bins {
        var rr: f64 = 0.0;
        var ii: f64 = 0.0;
        if average == "mean" {
            s = 0;
            while s < nseg { rr += acc_r[k * nseg + s]; ii += acc_i[k * nseg + s]; s += 1; }
            rr /= nseg;
            ii /= nseg;
        } else {
            var cr: f64[>..] = [];
            var ci: f64[>..] = [];
            s = 0;
            while s < nseg { cr.push(acc_r[k * nseg + s]); ci.push(acc_i[k * nseg + s]); s += 1; }
            rr = welch_median(cr) / bias;
            ii = welch_median(ci) / bias;
        }
        if onesided {
            var twice: f64 = 2.0;
            if k == 0 || (nf % 2 == 0 && k == bins - 1) { twice = 1.0; }
            rr *= twice;
            ii *= twice;
        }
        outr.push(rr * scale);
        outi.push(ii * scale);
        k += 1;
    }
    return line_arr(copy(outr)), line_arr(copy(outi)), no_fault();
}

// csd_pub checks the options of the cross spectrum, builds the window
// and gives the parts. The two lists must hold the same count of
// values. A count nfft of zero takes the size of a part. It gives the
// real and the imaginary part of the cross spectrum.
fn csd_pub(x: f64[:], y: f64[:], fs: f64, window: const u8[:], param: f64,
           win: f64[:], nperseg: i64, noverlap: i64, nfft: i64, detrend: bool,
           onesided: bool, scaling: const u8[:], average: const u8[:],
           tag: const u8[:]) -> Arr, Arr, Fault {
    if !(fs > 0.0) || !(fs - fs == 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the rate must be above zero");
    }
    if scaling != "density" && scaling != "spectrum" {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the scaling holds an unknown name");
    }
    if average != "mean" && average != "median" {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the average holds an unknown name");
    }
    if x.len < 1 || y.len < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the signals must hold one value or more");
    }
    if x.len != y.len {
        return empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the signals must hold the same count of values");
    }
    var k: i64 = 0;
    while k < x.len {
        if !(x[k] - x[k] == 0.0) || !(y[k] - y[k] == 0.0) {
            return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the signals hold a value that is not finite");
        }
        k += 1;
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
    return csd_avg(x, y, fs, w, nperseg, ov, nf, detrend, onesided, scaling, average, tag);
}

// csd_ex gives the cross spectral density of two signals, as SciPy
// csd does. The window is a name of get_window with its parameter
// param, or the values in win when win is not empty. The count nfft
// of zero takes the size of a part. Detrend true leaves the mean of
// each part out. The scaling is "density" or "spectrum". The average
// is "mean" or "median". The call gives the frequencies and the real
// and the imaginary part of the cross spectrum.
fn csd_ex(x: f64[:], y: f64[:], fs: f64 = 1.0, window: const u8[:] = "hann",
          param: f64 = 0.0 / 0.0, win: f64[:] = [], nperseg: i64 = 256,
          noverlap: i64 = -1, nfft: i64 = 0, detrend: bool = true,
          onesided: bool = true, scaling: const u8[:] = "density",
          average: const u8[:] = "mean") -> Arr, Arr, Arr, Fault {
    let pr, pi, f = csd_pub(x, y, fs, window, param, win, nperseg, noverlap, nfft,
                            detrend, onesided, scaling, average, "csd_ex");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
    var nf = nfft;
    if nf == 0 { nf = nperseg; }
    let bins = pr.data.len;
    var fr: f64[>..] = [];
    var k: i64 = 0;
    while k < bins {
        var idx = k;
        if !onesided && k >= (nf + 1) / 2 { idx = k - nf; }
        fr.push(idx * fs / nf);
        k += 1;
    }
    return line_arr(copy(fr)), pr, pi, no_fault();
}

// coherence gives the squared coherence of two signals, as SciPy
// coherence does. The value of each frequency is the square of the
// size of the cross spectrum over the product of the two power
// spectra. The options follow csd_ex. A place where a power spectrum
// holds a value of the size of zero gives a value of no meaning.
fn coherence(x: f64[:], y: f64[:], fs: f64 = 1.0, window: const u8[:] = "hann",
             param: f64 = 0.0 / 0.0, win: f64[:] = [], nperseg: i64 = 256,
             noverlap: i64 = -1, nfft: i64 = 0, detrend: bool = true) -> Arr, Arr, Fault {
    let xr, xi, f1 = csd_pub(x, y, fs, window, param, win, nperseg, noverlap, nfft,
                             detrend, true, "density", "mean", "coherence");
    if f1.kind != F_NONE { return empty_arr(), empty_arr(), f1; }
    let xxr, xxi, f2 = csd_pub(x, x, fs, window, param, win, nperseg, noverlap, nfft,
                               detrend, true, "density", "mean", "coherence");
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
    let yyr, yyi, f3 = csd_pub(y, y, fs, window, param, win, nperseg, noverlap, nfft,
                               detrend, true, "density", "mean", "coherence");
    if f3.kind != F_NONE { return empty_arr(), empty_arr(), f3; }
    let n = xr.data.len;
    var out: f64[>..] = [];
    var k: i64 = 0;
    while k < n {
        let px = xxr.data[k] * yyr.data[k];
        out.push((xr.data[k] * xr.data[k] + xi.data[k] * xi.data[k]) / px);
        k += 1;
    }
    var nf = nfft;
    if nf == 0 { nf = nperseg; }
    let bins = nf / 2 + 1;
    var fr: f64[>..] = [];
    var j: i64 = 0;
    while j < bins {
        fr.push(j * fs / nf);
        j += 1;
    }
    return line_arr(copy(fr)), line_arr(copy(out)), no_fault();
}

// correlation_lags gives the lag of each place of the correlation of
// two signals of the lengths in1_len and in2_len. The mode is "full",
// "same" or "valid", as in correlate.
fn correlation_lags(in1_len: i64, in2_len: i64, mode: const u8[:] = "full") -> Arr, Fault {
    if in1_len < 1 || in2_len < 1 {
        return empty_arr(), fault(F_VALUE, "correlation_lags", "the lengths must hold one or more");
    }
    var lo: i64 = 0;
    var hi: i64 = 0;
    if mode == "full" {
        lo = -(in2_len - 1);
        hi = in1_len - 1;
    } else if mode == "same" {
        let full = in1_len + in2_len - 1;
        let mid = full / 2;
        let bound = in1_len / 2;
        lo = -(in2_len - 1) + mid - bound;
        if in1_len % 2 == 0 {
            hi = -(in2_len - 1) + mid + bound - 1;
        } else {
            hi = -(in2_len - 1) + mid + bound;
        }
    } else if mode == "valid" {
        let bound = in1_len - in2_len;
        if bound >= 0 {
            lo = 0;
            hi = bound;
        } else {
            lo = bound;
            hi = 0;
        }
    } else {
        return empty_arr(), fault(F_VALUE, "correlation_lags", "the mode is unknown");
    }
    var d: f64[>..] = [];
    var k = lo;
    while k <= hi { d.push(k); k += 1; }
    return line_arr(copy(d)), no_fault();
}

// envelope gives the envelope of a signal and the residual that the
// band leaves out, as SciPy envelope does. The band runs over the
// places of the transform from bp_lo to bp_hi. A place above zero is
// a positive frequency. The default band holds the places from one to
// the Nyquist place, and leaves out the place of no frequency. The
// count n_out of zero keeps the count of the signal. The residual is
// "lowpass" for the part below the band, "all" for every part outside
// the band, or "none" for no residual.
fn envelope(z: f64[:], bp_lo: i64 = 1, bp_hi: i64 = 2147483647,
            n_out: i64 = 0, squared: bool = false,
            residual: const u8[:] = "lowpass") -> Arr, Arr, Fault {
    let tag = "envelope";
    if z.len < 1 || !all_finite(z) {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the signal must be a finite list");
    }
    if residual != "lowpass" && residual != "all" && residual != "none" {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the residual holds an unknown name");
    }
    let n = z.len;
    var n_out_v = n_out;
    if n_out_v == 0 { n_out_v = n; }
    if n_out_v < 1 || n_out_v > MAX_DSP_LENGTH {
        return empty_arr(), empty_arr(), fault(F_LIMIT, tag, "the output count is out of range");
    }
    let fak = (n_out_v as f64) / n;
    var start = bp_lo;
    if start == 2147483647 { start = -(n / 2); }
    var stop = bp_hi;
    if stop == 2147483647 { stop = (n + 1) / 2; }
    if !(-(n / 2) <= start && start < stop && stop <= (n + 1) / 2) {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the band does not hold the order of the places");
    }
    let zr, zi, f0 = rfft(z);
    if f0.kind != F_NONE { return empty_arr(), empty_arr(), f0; }
    // A full spectrum of n places that holds the positive half.
    var vr: f64[>..] = [];
    var vi: f64[>..] = [];
    var k: i64 = 0;
    while k < n { vr.push(0.0); vi.push(0.0); k += 1; }
    k = 0;
    while k < zr.data.len { vr[k] = zr.data[k]; vi[k] = zi.data[k]; k += 1; }
    if start > 0 {
        k = start;
        while k < stop { vr[k] *= 2.0; vi[k] *= 2.0; k += 1; }
    } else if stop > 0 {
        k = 1;
        while k < stop { vr[k] *= 2.0; vi[k] *= 2.0; k += 1; }
    }
    // The baseband signal of the doubled band, in a spectrum of the
    // count n_out_v.
    var br: f64[>..] = [];
    var bi: f64[>..] = [];
    k = 0;
    while k < n_out_v { br.push(0.0); bi.push(0.0); k += 1; }
    if !(start <= 0 && 0 < stop) {
        k = start;
        var j: i64 = 0;
        while k < stop { br[j] = vr[k]; bi[j] = vi[k]; k += 1; j += 1; }
    } else {
        // The shifted band holds the negative places of the band.
        k = start + n / 2;
        var j: i64 = 0;
        while k < stop + n / 2 { br[j] = vr[k]; bi[j] = vi[k]; k += 1; j += 1; }
    }
    let bbr, bbi, f1 = ifft(br, bi);
    if f1.kind != F_NONE { return empty_arr(), empty_arr(), f1; }
    var env: f64[>..] = [];
    k = 0;
    while k < bbr.data.len {
        if squared {
            env.push(bbr.data[k] * bbr.data[k] + bbi.data[k] * bbi.data[k]);
        } else {
            env.push(mathshim::root(bbr.data[k] * bbr.data[k] + bbi.data[k] * bbi.data[k]));
        }
        k += 1;
    }
    if residual == "none" {
        var s: i64[..8] = [];
        s.push(0);
        return line_arr(copy(env)), Arr { ndim: 1, shape: s, data: [] }, no_fault();
    }
    // The residual spectrum holds the places outside the band.
    if !(start <= 0 && 0 < stop) {
        k = start;
        while k < stop { vr[k] = 0.0; vi[k] = 0.0; k += 1; }
    } else {
        k = 0;
        while k < stop { vr[k] = 0.0; vi[k] = 0.0; k += 1; }
        k = start;
        while k < n { vr[k] = 0.0; vi[k] = 0.0; k += 1; }
    }
    if residual == "lowpass" {
        if stop > 0 {
            k = stop;
            while k < (n + 1) / 2 { vr[k] = 0.0; vi[k] = 0.0; k += 1; }
        } else {
            k = 0;
            while k < (n + 1) / 2 { vr[k] = 0.0; vi[k] = 0.0; k += 1; }
            k = start;
            while k < n { vr[k] = 0.0; vi[k] = 0.0; k += 1; }
        }
    }
    // The place of the Nyquist frequency holds no pair in a signal of
    // an even count, so the resample scales it.
    let m = n;
    if n_out_v != n && m % 2 == 0 {
        if n_out_v < n {
            vr[m / 2] *= 2.0;
            vi[m / 2] *= 2.0;
        } else {
            vr[m / 2] *= 0.5;
            vi[m / 2] *= 0.5;
        }
    }
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    k = 0;
    while k < n_out_v / 2 + 1 && k < n {
        hr.push(vr[k]);
        hi.push(vi[k]);
        k += 1;
    }
    while hr.len < n_out_v / 2 + 1 { hr.push(0.0); hi.push(0.0); }
    let res, f2 = irfft(hr, hi, n_out_v);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
    k = 0;
    while k < res.data.len { res.data[k] *= fak; k += 1; }
    return line_arr(copy(env)), res, no_fault();
}
