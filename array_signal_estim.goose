// The estimates of a signal: the division of polynomials, the
// periodogram of uneven times, the filter with the lowest phase and
// the smooth signal of Whittaker and Henderson.
//
// deconvolve divides a polynomial by another one. lombscargle gives
// the power of a signal at uneven times. minimum_phase gives the
// filter with the lowest phase. whittaker_henderson smooths a signal
// with a penalty on its second difference.
//
// The formulas follow SciPy 1.18.1 (scipy.signal).
namespace scientific;

import mathshim;

// deconvolve divides a signal by another one, as SciPy deconvolve
// does. The first list holds the dividend and the second the divisor.
// The call gives the quotient and the remainder.
fn deconvolve(signal: f64[:], divisor: f64[:]) -> Arr, Arr, Fault {
    let tag = "deconvolve";
    if signal.len < 1 || divisor.len < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the lists must hold one value or more");
    }
    var i: i64 = 0;
    while i < signal.len {
        if !(signal[i] - signal[i] == 0.0) {
            return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the signal holds a value that is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < divisor.len {
        if !(divisor[i] - divisor[i] == 0.0) {
            return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the divisor holds a value that is not finite");
        }
        i += 1;
    }
    let n = signal.len;
    let d = divisor.len;
    if d > n {
        var s: i64[..8] = [];
        s.push(n);
        let r = Arr { ndim: 1, shape: s, data: copy_slice(signal) };
        var e: i64[..8] = [];
        e.push(0);
        return Arr { ndim: 1, shape: e, data: [] }, r, no_fault();
    }
    // The quotient is the answer of the filter of the dividend by the
    // divisor for one impulse.
    var imp: f64[>..] = [];
    i = 0;
    while i < n - d + 1 { imp.push(0.0); i += 1; }
    imp[0] = 1.0;
    let quot, f = lfilter(signal[..], divisor[..], imp[..]);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let prod, f2 = convolve(divisor[..], quot.data[..]);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
    var rem: f64[>..] = [];
    i = 0;
    while i < n { rem.push(signal[i] - prod.data[i]); i += 1; }
    return quot, line_arr(copy(rem)), no_fault();
}

// lombscargle gives the power of a signal at the given angular
// frequencies, as SciPy lombscargle does. The times need not be even.
// The normalize is "power" or "normalize". The floating mean leaves
// the mean of the signal out of the fit. The call gives the power of
// each frequency.
fn lombscargle(times: f64[:], y: f64[:], freqs: f64[:],
                normalize: const u8[:] = "power",
                floating_mean: bool = false) -> Arr, Fault {
    let tag = "lombscargle";
    if times.len < 1 || times.len != y.len {
        return empty_arr(), fault(F_SHAPE, tag, "the times and the signal must hold the same count of values");
    }
    if freqs.len < 1 {
        return empty_arr(), fault(F_VALUE, tag, "the frequencies must hold one value or more");
    }
    if normalize != "power" && normalize != "normalize" {
        return empty_arr(), fault(F_VALUE, tag, "the way of the norm holds an unknown name");
    }
    var i: i64 = 0;
    while i < times.len {
        if !(times[i] - times[i] == 0.0) || !(y[i] - y[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, tag, "the times and the signal must be finite");
        }
        i += 1;
    }
    let n = times.len;
    let w = 1.0 / (n as f64);
    // The mean of the signal and the mean of its square do not depend on
    // the frequency, so they are computed once.
    var yy: f64 = 0.0;
    var y2: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        yy += w * y[i];
        y2 += w * y[i] * y[i];
        i += 1;
    }
    // The cosine and the sine of every time of the frequency. The three
    // sums of the frequency read the same values.
    var cbuf: f64[>..] = [];
    var sbuf: f64[>..] = [];
    i = 0;
    while i < n {
        cbuf.push(0.0);
        sbuf.push(0.0);
        i += 1;
    }
    var out: f64[>..] = [];
    var f: i64 = 0;
    while f < freqs.len {
        let om = freqs[f];
        i = 0;
        while i < n {
            cbuf[i] = mathshim::cosine(om * times[i]);
            sbuf[i] = mathshim::sine(om * times[i]);
            i += 1;
        }
        var cc: f64 = 0.0;
        i = 0;
        while i < n { cc += w * cbuf[i] * cbuf[i]; i += 1; }
        var ss = 1.0 - cc;
        var cs: f64 = 0.0;
        i = 0;
        while i < n {
            cs += w * cbuf[i] * sbuf[i];
            i += 1;
        }
        if floating_mean {
            var cm: f64 = 0.0;
            var sm: f64 = 0.0;
            i = 0;
            while i < n {
                cm += w * cbuf[i];
                sm += w * sbuf[i];
                i += 1;
            }
            cc -= cm * cm;
            ss -= sm * sm;
            cs -= cm * sm;
        }
        let tau = 0.5 * mathshim::arctangent2_of(2.0 * cs, cc - ss);
        var yc: f64 = 0.0;
        var ys: f64 = 0.0;
        cc = 0.0;
        i = 0;
        while i < n {
            let ct = mathshim::cosine(om * times[i] - tau);
            let st = mathshim::sine(om * times[i] - tau);
            yc += w * y[i] * ct;
            ys += w * y[i] * st;
            cc += w * ct * ct;
            i += 1;
        }
        ss = 1.0 - cc;
        if floating_mean {
            var cm: f64 = 0.0;
            var sm: f64 = 0.0;
            i = 0;
            while i < n {
                cm += w * mathshim::cosine(om * times[i] - tau);
                sm += w * mathshim::sine(om * times[i] - tau);
                i += 1;
            }
            yc -= yy * cm;
            ys -= yy * sm;
            cc -= cm * cm;
            ss -= sm * sm;
        }
        // A value of no meaning would divide by a value of no meaning.
        if cc < 1.1102230246251565e-16 { cc = 1.1102230246251565e-16; }
        if ss < 1.1102230246251565e-16 { ss = 1.1102230246251565e-16; }
        let a = yc / cc;
        let b = ys / ss;
        var p = 2.0 * (a * yc + b * ys);
        if normalize == "power" {
            p *= n as f64 / 4.0;
        } else {
            var yy2 = y2;
            if floating_mean { yy2 -= yy * yy; }
            p *= 0.5 / yy2;
        }
        out.push(p);
        f += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// minimum_phase gives the filter with the lowest phase that holds the
// same magnitude, as SciPy minimum_phase does with the method
// "homomorphic". The filter must hold at least three values. The
// count n_fft of zero takes the next power of two above two hundred
// times the count of the filter. The call gives the filter.
fn minimum_phase(h: f64[:], n_fft: i64 = 0) -> Arr, Fault {
    let tag = "minimum_phase";
    if h.len < 3 {
        return empty_arr(), fault(F_VALUE, tag, "the filter must hold three values or more");
    }
    var i: i64 = 0;
    while i < h.len {
        if !(h[i] - h[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, tag, "the filter holds a value that is not finite");
        }
        i += 1;
    }
    var nf = n_fft;
    if nf == 0 {
        // The next power of two above two hundred times the count.
        var need = 200 * (h.len - 1);
        nf = 1;
        while nf < need { nf = nf * 2; }
    }
    if nf < h.len {
        return empty_arr(), fault(F_VALUE, tag, "the count of the transform must hold at least the count of the filter");
    }
    if nf > MAX_DSP_LENGTH {
        return empty_arr(), fault(F_LIMIT, tag, "the count of the transform is out of range");
    }
    // The logarithm of the magnitude of the filter, over a transform
    // of the count nf.
    let mag = mag_spectrum(h, nf);
    var small: f64 = 0.0 / 0.0;
    i = 0;
    while i < mag.len {
        if mag[i] > 0.0 && (small != small || mag[i] < small) { small = mag[i]; }
        i += 1;
    }
    if small != small {
        return empty_arr(), fault(F_VALUE, tag, "the filter holds no value of no zero");
    }
    var lr: f64[>..] = [];
    i = 0;
    while i < nf {
        lr.push(mathshim::ln(mag[i] + 1.0e-7 * small) * 0.5);
        i += 1;
    }
    // The inverse transform of the logarithm gives the spectrum of the
    // half of the magnitude.
    var zr: f64[>..] = [];
    var zi: f64[>..] = [];
    i = 0;
    while i < nf { zr.push(0.0); zi.push(0.0); i += 1; }
    i = 0;
    while i < nf { zr[i] = lr[i]; i += 1; }
    let ir, ii, f = ifft(zr, zi);
    if f.kind != F_NONE { return empty_arr(), f; }
    // The filter of the homomorphic method doubles the places from one
    // to the middle and holds the place of no frequency.
    var wr: f64[>..] = [];
    var wi: f64[>..] = [];
    i = 0;
    while i < nf { wr.push(0.0); wi.push(0.0); i += 1; }
    wr[0] = 1.0;
    let stop = nf / 2;
    i = 1;
    while i < stop { wr[i] = 2.0; i += 1; }
    wr[stop] = 1.0 + (nf % 2) as f64;
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    i = 0;
    while i < nf {
        hr.push(ir.data[i] * wr[i] - ii.data[i] * wi[i]);
        hi.push(ir.data[i] * wi[i] + ii.data[i] * wr[i]);
        i += 1;
    }
    // The transform of the exponent of the spectrum gives the filter.
    let er, ei, f2 = fft(hr);
    if f2.kind != F_NONE { return empty_arr(), f2; }
    var xr: f64[>..] = [];
    var xi: f64[>..] = [];
    i = 0;
    while i < nf {
        let e = Cplx { re: er.data[i], im: ei.data[i] };
        let v = c_exp(e);
        xr.push(v.re);
        xi.push(v.im);
        i += 1;
    }
    let yr, yi, f3 = ifft(xr, xi);
    if f3.kind != F_NONE { return empty_arr(), f3; }
    let n_out = h.len / 2 + h.len % 2;
    var out: f64[>..] = [];
    i = 0;
    while i < n_out { out.push(yr.data[i]); i += 1; }
    return line_arr(copy(out)), no_fault();
}

// mag_spectrum gives the magnitude of the transform of a filter over a
// transform of the count nf.
fn mag_spectrum(h: f64[:], nf: i64) -> f64[>..] {
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < h.len { buf.push(h[i]); i += 1; }
    while buf.len < nf { buf.push(0.0); }
    let re, im, f = fft(buf);
    var out: f64[>..] = [];
    i = 0;
    while i < nf {
        out.push(mathshim::root(re.data[i] * re.data[i] + im.data[i] * im.data[i]));
        i += 1;
    }
    return out;
}

// whittaker_henderson smooths a signal with a penalty on its second
// difference, as SciPy whittaker_henderson does with a given number
// lamb. The order sets the count of the differences of the penalty.
// The weights hold a weight for each value of the signal. A weight of
// zero leaves the value out. The call gives the smooth signal and the
// number lamb that was used.
fn whittaker_henderson(signal: f64[:], lamb: f64, order: i64 = 2,
                       weights: f64[:] = []) -> Arr, f64, Fault {
    let tag = "whittaker_henderson";
    if signal.len < order + 1 {
        return empty_arr(), 0.0, fault(F_SHAPE, tag, "the signal must hold more values than the order");
    }
    if order < 1 {
        return empty_arr(), 0.0, fault(F_VALUE, tag, "the order must hold one value or more");
    }
    if lamb != lamb || lamb < 0.0 {
        return empty_arr(), 0.0, fault(F_VALUE, tag, "the number lamb must hold a value of no minus");
    }
    var i: i64 = 0;
    while i < signal.len {
        if !(signal[i] - signal[i] == 0.0) {
            return empty_arr(), 0.0, fault(F_VALUE, tag, "the signal holds a value that is not finite");
        }
        i += 1;
    }
    var w: f64[>..] = [];
    if weights.len > 0 {
        if weights.len != signal.len {
            return empty_arr(), 0.0, fault(F_SHAPE, tag, "the weights must hold the count of the signal");
        }
        i = 0;
        while i < weights.len {
            if !(weights[i] - weights[i] == 0.0) || weights[i] < 0.0 {
                return empty_arr(), 0.0, fault(F_VALUE, tag, "the weights must hold finite values of no minus");
            }
            w.push(weights[i]);
            i += 1;
        }
    }
    let n = signal.len;
    if lamb == 0.0 {
        return line_arr(copy_slice(signal)), 0.0, no_fault();
    }
    // The band of the penalty holds the differences of the order. The
    // place of the band at the line d and the column j holds the
    // number of the ways to walk from the place j to the place j
    // below by the order, times the sign of the order.
    let p = order;
    var band: f64[>..] = [];
    var i: i64 = 0;
    while i < (p + 1) * n { band.push(0.0); i += 1; }
    var d: i64 = 0;
    while d <= p {
        var sign: f64 = 1.0;
        if d % 2 == 1 { sign = -1.0; }
        var j: i64 = 0;
        while j + d < n {
            var lo: i64 = j - (n - p - 1);
            if lo < 0 { lo = 0; }
            var hi: i64 = p - d;
            if hi > j { hi = j; }
            var sum: f64 = 0.0;
            var u: i64 = lo;
            while u <= hi {
                sum += sig_binom(p, u) * sig_binom(p, u + d);
                u += 1;
            }
            band[d * n + j] = sign * sum * lamb;
            j += 1;
        }
        d += 1;
    }
    // The first line of the matrix holds the weights and the ones.
    i = 0;
    while i < n {
        if w.len > 0 {
            band[i] += w[i];
        } else {
            band[i] += 1.0;
        }
        i += 1;
    }
    var rhs: f64[>..] = [];
    i = 0;
    while i < n {
        if w.len > 0 {
            rhs.push(w[i] * signal[i]);
        } else {
            rhs.push(signal[i]);
        }
        i += 1;
    }
    var shp: i64[>..] = [];
    shp.push(p + 1);
    shp.push(n);
    var ab_arr = arr_of(copy(band), shp);
    let rhs_arr = line_arr(copy(rhs));
    let sol, f = solveh_banded(ab_arr, rhs_arr, true);
    if f.kind != F_NONE {
        // A number that stands too high breaks the solver. The smooth
        // signal then is the fit of a polynomial of the order.
        let fit, f2 = polyfit_places(signal, order - 1);
        if f2.kind != F_NONE { return empty_arr(), 0.0, f2; }
        return fit, lamb, no_fault();
    }
    return sol, lamb, no_fault();
}

// sig_binom gives the count of the ways to choose k places of n.
fn sig_binom(n: i64, k: i64) -> f64 {
    if k < 0 || k > n { return 0.0; }
    var r: f64 = 1.0;
    var i: i64 = 0;
    while i < k {
        r = r * ((n - i) as f64) / ((i + 1) as f64);
        i += 1;
    }
    return r;
}

// polyfit_places fits a polynomial of the given degree to the values
// of a signal at the places 0 to n minus 1.
fn polyfit_places(signal: f64[:], deg: i64) -> Arr, Fault {
    var x: f64[>..] = [];
    var i: i64 = 0;
    while i < signal.len { x.push(i as f64); i += 1; }
    let coeffs, f = polyfit(copy(x), copy_slice(signal), deg);
    if f.kind != F_NONE { return empty_arr(), f; }
    var out: f64[>..] = [];
    i = 0;
    while i < signal.len {
        out.push(polyval(coeffs.data, i as f64));
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}
