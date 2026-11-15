// The design of a filter from its cuts: the digital filters and the
// notch, peak and comb filters.
//
// The design of a filter takes the order, the cuts, the band and the
// kind of the filter. It gives the zeros, the poles and the gain, or
// the numerator and the denominator. The file also holds the notch,
// the peak and the comb filters, and the helpers of the Kaiser
// window.
namespace scientific;

import mathshim;

// sig_band_code gives the kind of band of a name. It holds minus one
// for an unknown name.
fn sig_band_code(btype: const u8[:]) -> i64 {
    if btype == "lowpass" || btype == "low" || btype == "l" { return SIG_LOW; }
    if btype == "highpass" || btype == "high" || btype == "hp" { return SIG_HIGH; }
    if btype == "bandpass" || btype == "band" || btype == "pass" || btype == "bp" { return SIG_PASS; }
    if btype == "bandstop" || btype == "stop" || btype == "bs" { return SIG_STOP; }
    return -1;
}

// sig_ftype_code gives the kind of filter of a name. It holds minus
// one for an unknown name.
fn sig_ftype_code(ftype: const u8[:]) -> i64 {
    if ftype == "butter" || ftype == "butterworth" { return SIG_BUTTER; }
    if ftype == "cheby1" || ftype == "chebyshev1" { return SIG_CHEBY1; }
    if ftype == "cheby2" || ftype == "chebyshev2" { return SIG_CHEBY2; }
    if ftype == "ellip" || ftype == "elliptic" { return SIG_ELLIP; }
    if ftype == "bessel" { return SIG_BESSEL; }
    return -1;
}

// sig_proto gives the analog prototype of a filter. It holds the
// zeros, the poles and the gain.
fn sig_proto(kind: i64, order: i64, rp: f64, rs: f64) -> Cplx[>..], Cplx[>..], f64, Fault {
    var zr: f64[>..] = [];
    var zi: f64[>..] = [];
    var pr: f64[>..] = [];
    var pi_: f64[>..] = [];
    var k: f64 = 0.0;
    var f: Fault = no_fault();
    if kind == SIG_BUTTER {
        let a, b, c, d, e, g = buttap(order);
        f = g;
        zr = copy(a.data); zi = copy(b.data); pr = copy(c.data); pi_ = copy(d.data); k = e;
    } else if kind == SIG_CHEBY1 {
        let a, b, c, d, e, g = cheb1ap(order, rp);
        f = g;
        zr = copy(a.data); zi = copy(b.data); pr = copy(c.data); pi_ = copy(d.data); k = e;
    } else if kind == SIG_CHEBY2 {
        let a, b, c, d, e, g = cheb2ap(order, rs);
        f = g;
        zr = copy(a.data); zi = copy(b.data); pr = copy(c.data); pi_ = copy(d.data); k = e;
    } else if kind == SIG_ELLIP {
        let a, b, c, d, e, g = ellipap(order, rp, rs);
        f = g;
        zr = copy(a.data); zi = copy(b.data); pr = copy(c.data); pi_ = copy(d.data); k = e;
    } else {
        let a, b, c, d, e, g = besselap(order, "phase");
        f = g;
        zr = copy(a.data); zi = copy(b.data); pr = copy(c.data); pi_ = copy(d.data); k = e;
    }
    if f.kind != F_NONE { return [], [], 0.0, f; }
    return rs_zip(zr, zi), rs_zip(pr, pi_), k, no_fault();
}

// sig_iir_zpk designs a filter and gives its zeros, its poles and its
// gain. The band holds the kind of band and the kind holds the kind of
// filter.
fn sig_iir_zpk(order: i64, wn: f64[:], rp: f64, rs: f64, band: i64, kind: i64, analog: bool) -> Cplx[>..], Cplx[>..], f64, Fault {
    if order < 1 || order > FORM_ROOTS_MAX {
        return [], [], 0.0, fault(F_VALUE, "iirfilter", "the order holds a value outside one to 64");
    }
    var need: i64 = 1;
    if band == SIG_PASS || band == SIG_STOP { need = 2; }
    if wn.len != need {
        return [], [], 0.0, fault(F_SHAPE, "iirfilter", "the count of cuts does not fit the type");
    }
    var i: i64 = 0;
    while i < wn.len {
        if !(wn[i] > 0.0) {
            return [], [], 0.0, fault(F_RANGE, "iirfilter", "a cut holds a value of zero or below");
        }
        if !analog && !(wn[i] < 1.0) {
            return [], [], 0.0, fault(F_RANGE, "iirfilter", "a cut holds a value of one or above");
        }
        i += 1;
    }
    if need == 2 && !(wn[0] < wn[1]) {
        return [], [], 0.0, fault(F_RANGE, "iirfilter", "the low cut holds a value that is not below the high cut");
    }
    // The analog prototype.
    let z0, p0, k0, pf = sig_proto(kind, order, rp, rs);
    if pf.kind != F_NONE { return [], [], 0.0, pf; }
    // The bent cuts.
    var warped: f64[>..] = [];
    i = 0;
    while i < wn.len {
        if analog { warped.push(wn[i]); }
        else { warped.push(4.0 * mathshim::tan_of(SIG_PI * wn[i] / 2.0)); }
        i += 1;
    }
    // The map to the band.
    var z: Cplx[>..] = [];
    var p: Cplx[>..] = [];
    var k: f64 = k0;
    if band == SIG_LOW {
        let a, b, c, f = sig_lp2lp_zpk(z0, p0, k0, warped[0]);
        if f.kind != F_NONE { return [], [], 0.0, f; }
        z = copy(a); p = copy(b); k = c;
    } else if band == SIG_HIGH {
        let a, b, c, f = sig_lp2hp_zpk(z0, p0, k0, warped[0]);
        if f.kind != F_NONE { return [], [], 0.0, f; }
        z = copy(a); p = copy(b); k = c;
    } else {
        let bw = warped[1] - warped[0];
        let wo = mathshim::root(warped[0] * warped[1]);
        if band == SIG_PASS {
            let a, b, c, f = sig_lp2bp_zpk(z0, p0, k0, wo, bw);
            if f.kind != F_NONE { return [], [], 0.0, f; }
            z = copy(a); p = copy(b); k = c;
        } else {
            let a, b, c, f = sig_lp2bs_zpk(z0, p0, k0, wo, bw);
            if f.kind != F_NONE { return [], [], 0.0, f; }
            z = copy(a); p = copy(b); k = c;
        }
    }
    // The map to the digital plane.
    if !analog {
        let a, b, c, f = sig_bilinear_zpk(z, p, k, 2.0);
        if f.kind != F_NONE { return [], [], 0.0, f; }
        z = copy(a); p = copy(b); k = c;
    }
    return z, p, k, no_fault();
}

// iirfilter designs a filter and gives the numerator and the
// denominator. The cuts sit in the range of zero to one for a digital
// filter. An analog filter takes them in radians per second. The type
// of the band is "lowpass", "highpass", "bandpass" or "bandstop". The
// kind of the filter is "butter", "cheby1", "cheby2", "ellip" or
// "bessel". The ripple and the attenuation hold decibels. They serve
// the kinds of Chebyshev and elliptic filters.
fn iirfilter(order: i64, wn: f64[:], rp: f64, rs: f64, btype: const u8[:], ftype: const u8[:], analog: bool = false) -> Arr, Arr, Fault {
    let zr, zi, pr, pi_, k, f = iirfilter_zpk(order, wn, rp, rs, btype, ftype, analog);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    return zpk2tf(zr.data, zi.data, pr.data, pi_.data, k);
}

// iirfilter_zpk designs a filter and gives its zeros, its poles and
// its gain. The arguments are those of iirfilter.
fn iirfilter_zpk(order: i64, wn: f64[:], rp: f64, rs: f64, btype: const u8[:], ftype: const u8[:], analog: bool = false) -> Arr, Arr, Arr, Arr, f64, Fault {
    let band = sig_band_code(btype);
    if band < 0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_VALUE, "iirfilter", "the type of the band holds an unknown name");
    }
    let kind = sig_ftype_code(ftype);
    if kind < 0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_VALUE, "iirfilter", "the kind of the filter holds an unknown name");
    }
    let z, p, k, f = sig_iir_zpk(order, wn, rp, rs, band, kind, analog);
    if f.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f;
    }
    var zr: f64[>..] = [];
    var zi: f64[>..] = [];
    var pr: f64[>..] = [];
    var pi_: f64[>..] = [];
    sig_split(z, zr, zi);
    sig_split(p, pr, pi_);
    return flt_arr(copy(zr)), flt_arr(copy(zi)), flt_arr(copy(pr)), flt_arr(copy(pi_)), k, no_fault();
}

// iirdesign designs a filter from its edges and gives the numerator
// and the denominator. The kind of the filter is "butter", "cheby1",
// "cheby2" or "ellip". The ripple and the attenuation hold decibels.
fn iirdesign(wp: f64[:], ws: f64[:], gpass: f64, gstop: f64, analog: bool = false, ftype: const u8[:] = "ellip") -> Arr, Arr, Fault {
    let kind = sig_ftype_code(ftype);
    if kind < 0 || kind == SIG_BESSEL {
        return empty_arr(), empty_arr(), fault(F_VALUE, "iirdesign", "the kind of the filter holds an unknown name");
    }
    if wp.len != ws.len || wp.len < 1 || wp.len > 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "iirdesign", "the edges hold a count of one or two, or the two lists differ");
    }
    var i: i64 = 0;
    while i < wp.len {
        if !(wp[i] > 0.0) || !(ws[i] > 0.0) {
            return empty_arr(), empty_arr(), fault(F_RANGE, "iirdesign", "an edge holds a value of zero or below");
        }
        if !analog && (wp[i] >= 1.0 || ws[i] >= 1.0) {
            return empty_arr(), empty_arr(), fault(F_RANGE, "iirdesign", "an edge holds a value of one or above");
        }
        i += 1;
    }
    if wp.len == 2 {
        let inside = (ws[0] < wp[0] && wp[1] < ws[1]) || (wp[0] < ws[0] && ws[1] < wp[1]);
        if !inside {
            return empty_arr(), empty_arr(), fault(F_RANGE, "iirdesign", "the pass band does not lie inside the stop band");
        }
    }
    // The kind of the band: one for low pass, two for high pass,
    // three for band stop and four for band pass.
    var ftype_id: i64 = 1;
    if wp.len == 2 { ftype_id = 3; }
    if wp[0] >= ws[0] { ftype_id += 1; }
    var btype: const u8[:] = "lowpass";
    if ftype_id == 2 { btype = "highpass"; }
    else if ftype_id == 3 { btype = "bandstop"; }
    else if ftype_id == 4 { btype = "bandpass"; }
    // The order of the filter.
    let ord, wn, of = sig_ord(wp, ws, gpass, gstop, analog, kind);
    if of.kind != F_NONE { return empty_arr(), empty_arr(), of; }
    return iirfilter(ord, wn.data, gpass, gstop, btype, ftype, analog);
}

// sig_notch_peak designs a notch filter or a peak filter. The kind is
// "notch" or "peak".
fn sig_notch_peak(w0: f64, q: f64, kind: const u8[:], fs: f64, tag: const u8[:]) -> Arr, Arr, Fault {
    if !(fs > 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the rate holds a value of zero or below");
    }
    if !(q > 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the quality holds a value of zero or below");
    }
    let wn = 2.0 * w0 / fs;
    if !(wn >= 0.0 && wn <= 1.0) {
        return empty_arr(), empty_arr(), fault(F_RANGE, tag, "the frequency holds a value outside the range of zero to the rate");
    }
    if kind != "notch" && kind != "peak" {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the kind of the filter holds an unknown name");
    }
    let bw = wn / q * SIG_PI;
    let wp = wn * SIG_PI;
    let beta = mathshim::tan_of(bw / 2.0);
    let gain = 1.0 / (1.0 + beta);
    var b: f64[>..] = [];
    var a: f64[>..] = [];
    if kind == "notch" {
        b.push(gain);
        b.push(-2.0 * gain * mathshim::cosine(wp));
        b.push(gain);
    } else {
        b.push(1.0 - gain);
        b.push(0.0);
        b.push(-(1.0 - gain));
    }
    a.push(1.0);
    a.push(-2.0 * gain * mathshim::cosine(wp));
    a.push(2.0 * gain - 1.0);
    return flt_arr(copy(b)), flt_arr(copy(a)), no_fault();
}

// iirnotch designs a notch filter at the frequency w0. The quality q
// holds the width of the notch. The rate fs holds the rate of the
// samples.
fn iirnotch(w0: f64, q: f64, fs: f64 = 2.0) -> Arr, Arr, Fault {
    return sig_notch_peak(w0, q, "notch", fs, "iirnotch");
}

// iirpeak designs a peak filter at the frequency w0. The quality q
// holds the width of the peak. The rate fs holds the rate of the
// samples.
fn iirpeak(w0: f64, q: f64, fs: f64 = 2.0) -> Arr, Arr, Fault {
    return sig_notch_peak(w0, q, "peak", fs, "iirpeak");
}

// iircomb designs a comb filter at the frequency w0. The kind is
// "notch" or "peak". The rate fs holds the rate of the samples. With
// pass_zero true the comb filter of a peak passes zero.
fn iircomb(w0: f64, q: f64, kind: const u8[:] = "notch", fs: f64 = 2.0, pass_zero: bool = false) -> Arr, Arr, Fault {
    if !(fs > 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "iircomb", "the rate holds a value of zero or below");
    }
    if !(w0 > 0.0 && w0 < fs / 2.0) {
        return empty_arr(), empty_arr(), fault(F_RANGE, "iircomb", "the frequency holds a value outside the range of zero to half the rate");
    }
    if !(q > 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "iircomb", "the quality holds a value of zero or below");
    }
    if kind != "notch" && kind != "peak" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "iircomb", "the kind of the filter holds an unknown name");
    }
    let n: i64 = mathshim::round_of(fs / w0) as i64;
    if n < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "iircomb", "the count of taps holds a value of zero or below");
    }
    // The rate must divide the frequency, so the taps sit on the comb.
    let dev = mathshim::abs_of(w0 - fs / n) / fs;
    if dev > 1.0e-14 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "iircomb", "the rate does not divide the frequency");
    }
    let wp = 2.0 * SIG_PI * w0 / fs;
    let wd = wp / q;
    let beta = mathshim::tan_of(n * wd / 4.0);
    var g0: f64 = 1.0;
    var g: f64 = 0.0;
    if kind == "peak" { g0 = 0.0; g = 1.0; }
    let ax = (1.0 - beta) / (1.0 + beta);
    let bx = (g0 + g * beta) / (1.0 + beta);
    let cx = (g0 - g * beta) / (1.0 + beta);
    var negative = false;
    if kind == "peak" { negative = pass_zero; } else { negative = !pass_zero; }
    var sgn: f64 = 1.0;
    if negative { sgn = -1.0; }
    var b: f64[>..] = [];
    var a: f64[>..] = [];
    var i: i64 = 0;
    while i <= n { b.push(0.0); a.push(0.0); i += 1; }
    b[0] = bx;
    b[n] = sgn * cx;
    a[0] = 1.0;
    a[n] = sgn * ax;
    return flt_arr(copy(b)), flt_arr(copy(a)), no_fault();
}

// kaiser_beta gives the shape of the Kaiser window for an attenuation
// of a decibels. It holds zero for an attenuation below 21 decibels.
fn kaiser_beta(a: f64) -> f64 {
    if a > 50.0 { return 0.1102 * (a - 8.7); }
    if a > 21.0 { return 0.5842 * mathshim::power(a - 21.0, 0.4) + 0.07886 * (a - 21.0); }
    return 0.0;
}

// kaiser_atten gives the attenuation of a Kaiser window of the given
// number of taps and the given width.
fn kaiser_atten(numtaps: i64, width: f64) -> f64 {
    return 2.285 * (numtaps - 1) * SIG_PI * width + 7.95;
}

// kaiserord gives the number of taps of a Kaiser window and the shape
// of the window. The ripple holds the attenuation in decibels. The
// width holds the width of the transition band.
fn kaiserord(ripple: f64, width: f64) -> i64, f64, Fault {
    let a = mathshim::abs_of(ripple);
    if a < 8.0 {
        return 0, 0.0, fault(F_VALUE, "kaiserord", "the attenuation holds a value below eight decibels");
    }
    if !(width > 0.0) {
        return 0, 0.0, fault(F_VALUE, "kaiserord", "the width holds a value of zero or below");
    }
    let beta = kaiser_beta(a);
    let numtaps = (a - 7.95) / 2.285 / (SIG_PI * width) + 1.0;
    return mathshim::ceil_of(numtaps) as i64, beta, no_fault();
}
