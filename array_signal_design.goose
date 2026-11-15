// The design of a filter: the analog prototypes and the choice of
// the order.
//
// The package holds the analog prototypes of the Butterworth, the
// Chebyshev of type I, the Chebyshev of type II, the elliptic and the
// Bessel filters. It chooses the order of a filter from its edges.
//
// A prototype gives the zeros and the poles as a list of real parts
// and a list of imaginary parts, and the gain. The list of the zeros
// is empty for the prototypes without zeros.
namespace scientific;

import mathshim;

const SIG_PI: f64 = 3.141592653589793;

// The kinds of filter that the design knows.
const SIG_BUTTER: i64 = 0;
const SIG_CHEBY1: i64 = 1;
const SIG_CHEBY2: i64 = 2;
const SIG_ELLIP: i64 = 3;
const SIG_BESSEL: i64 = 4;

// The kinds of band: low pass, high pass, band pass and band stop.
const SIG_LOW: i64 = 0;
const SIG_HIGH: i64 = 1;
const SIG_PASS: i64 = 2;
const SIG_STOP: i64 = 3;

// sig_split writes the real parts and the imaginary parts of a list of
// complex numbers into two lists.
fn sig_split(r: Cplx[:], re: f64[>..]&, im: f64[>..]&) {
    var i: i64 = 0;
    while i < r.len { re.push(r[i].re); im.push(r[i].im); i += 1; }
}

// sig_out gives the zeros, the poles and the gain of a prototype as
// four arrays and a number.
fn sig_out(z: Cplx[:], p: Cplx[:], k: f64) -> Arr, Arr, Arr, Arr, f64, Fault {
    var zr: f64[>..] = [];
    var zi: f64[>..] = [];
    var pr: f64[>..] = [];
    var pi_: f64[>..] = [];
    sig_split(z, zr, zi);
    sig_split(p, pr, pi_);
    if !all_finite(zr) || !all_finite(zi) || !all_finite(pr) || !all_finite(pi_) {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_VALUE, "the prototype", "a root holds a value that is not finite");
    }
    return flt_arr(copy(zr)), flt_arr(copy(zi)), flt_arr(copy(pr)), flt_arr(copy(pi_)), k, no_fault();
}

// sig_order_fault checks the order of a prototype.
fn sig_order_fault(order: i64, tag: const u8[:]) -> Fault {
    if order < 0 || order > FORM_ROOTS_MAX {
        return fault(F_VALUE, tag, "the order holds a value outside zero to 64");
    }
    return no_fault();
}

// buttap gives the analog prototype of a Butterworth filter of the
// order n. The cut of the prototype sits at one radian per second.
// The prototype holds no zeros.
fn buttap(order: i64) -> Arr, Arr, Arr, Arr, f64, Fault {
    let of = sig_order_fault(order, "buttap");
    if of.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, of; }
    var p: Cplx[>..] = [];
    var m: i64 = 1 - order;
    while m < order {
        let ang = SIG_PI * m / (2.0 * order);
        p.push(Cplx { re: -mathshim::cosine(ang), im: -mathshim::sine(ang) });
        m += 2;
    }
    var z: Cplx[>..] = [];
    return sig_out(z, p, 1.0);
}

// cheb1ap gives the analog prototype of a Chebyshev filter of type I
// with a pass band ripple of rp decibels. The prototype holds no
// zeros.
fn cheb1ap(order: i64, rp: f64) -> Arr, Arr, Arr, Arr, f64, Fault {
    let rf = ripple_fault(rp, "cheb1ap", "the ripple holds a value outside zero to one million");
    if rf.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, rf; }
    let of = sig_order_fault(order, "cheb1ap");
    if of.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, of; }
    if order == 0 {
        var z: Cplx[>..] = [];
        var p: Cplx[>..] = [];
        return sig_out(z, p, mathshim::power(10.0, -rp / 20.0));
    }
    let eps = mathshim::root(mathshim::power(10.0, 0.1 * rp) - 1.0);
    let mu = mathshim::arcsinh_of(1.0 / eps) / order;
    var p: Cplx[>..] = [];
    var m: i64 = 1 - order;
    while m < order {
        let th = SIG_PI * m / (2.0 * order);
        p.push(Cplx { re: -mathshim::sinh_of(mu) * mathshim::cosine(th), im: -mathshim::cosh_of(mu) * mathshim::sine(th) });
        m += 2;
    }
    var k = cx_prod_neg(p).re;
    if order % 2 == 0 { k = k / mathshim::root(1.0 + eps * eps); }
    var z: Cplx[>..] = [];
    return sig_out(z, p, k);
}

// cheb2ap gives the analog prototype of a Chebyshev filter of type II
// with a stop band attenuation of rs decibels. The zeros of the
// prototype sit on the imaginary axis.
fn cheb2ap(order: i64, rs: f64) -> Arr, Arr, Arr, Arr, f64, Fault {
    let rf = ripple_fault(rs, "cheb2ap", "the attenuation holds a value outside zero to one million");
    if rf.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, rf; }
    let of = sig_order_fault(order, "cheb2ap");
    if of.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, of; }
    if order == 0 {
        var z: Cplx[>..] = [];
        var p: Cplx[>..] = [];
        return sig_out(z, p, 1.0);
    }
    let de = 1.0 / mathshim::root(mathshim::power(10.0, 0.1 * rs) - 1.0);
    let mu = mathshim::arcsinh_of(1.0 / de) / order;
    // The zeros. An odd order holds no zero on the imaginary axis.
    var z: Cplx[>..] = [];
    var m: i64 = 1 - order;
    while m < order {
        if m != 0 {
            z.push(Cplx { re: 0.0, im: 1.0 / mathshim::sine(SIG_PI * m / (2.0 * order)) });
        }
        m += 2;
    }
    // The poles.
    var p: Cplx[>..] = [];
    m = 1 - order;
    while m < order {
        let th = SIG_PI * m / (2.0 * order);
        let pr = -mathshim::sinh_of(mu) * mathshim::cosine(th);
        let pim = -mathshim::cosh_of(mu) * mathshim::sine(th);
        p.push(c_div(Cplx { re: 1.0, im: 0.0 }, Cplx { re: pr, im: pim }));
        m += 2;
    }
    let k = c_div(cx_prod_neg(p), cx_prod_neg(z)).re;
    return sig_out(z, p, k);
}

// ellipap gives the analog prototype of an elliptic filter with a pass
// band ripple of rp decibels and a stop band attenuation of rs
// decibels. The attenuation must not be below the ripple.
fn ellipap(order: i64, rp: f64, rs: f64) -> Arr, Arr, Arr, Arr, f64, Fault {
    let rf = ripple_fault(rp, "ellipap", "the ripple holds a value outside zero to one million");
    if rf.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, rf; }
    let sf = ripple_fault(rs, "ellipap", "the attenuation holds a value outside zero to one million");
    if sf.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, sf; }
    if rs < rp {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_VALUE, "ellipap", "the attenuation holds a value below the ripple");
    }
    let of = sig_order_fault(order, "ellipap");
    if of.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, of; }
    var z: Cplx[>..] = [];
    var p: Cplx[>..] = [];
    let k = ellip_proto(order, rp, rs, z, p);
    return sig_out(z, p, k);
}

// besselap gives the analog prototype of a Bessel filter. The norm is
// "phase", "delay" or "mag". It says what the cut means, as in SciPy.
// The prototype holds no zeros.
fn besselap(order: i64, norm: const u8[:] = "phase") -> Arr, Arr, Arr, Arr, f64, Fault {
    let nc = bessel_check(norm, "besselap");
    if nc.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, nc; }
    let of = sig_order_fault(order, "besselap");
    if of.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, of; }
    if order == 0 {
        var z: Cplx[>..] = [];
        var p: Cplx[>..] = [];
        return sig_out(z, p, 1.0);
    }
    let poly = bessel_poly(order);
    let rts, rf = roots_of(poly);
    if rf.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, rf; }
    var p: Cplx[>..] = [];
    var i: i64 = 0;
    while i < rts.len { p.push(bessel_polish(poly, rts[i])); i += 1; }
    let a_last = poly[poly.len - 1];
    var k = a_last;
    if norm == "phase" {
        let f = mathshim::power(10.0, -mathshim::log10_of(a_last) / order);
        i = 0;
        while i < p.len { p[i] = c_scale(p[i], f); i += 1; }
        k = 1.0;
    } else if norm == "mag" {
        // The gain falls with the frequency, so bisect for the half size.
        let target = 0.7071067811865476;
        var hi: f64 = 1.0;
        while bessel_gain(p, a_last, hi) > target { hi = hi * 2.0; }
        var lo: f64 = 0.0;
        var turn: i64 = 0;
        while turn < 200 {
            let mid = 0.5 * (lo + hi);
            if bessel_gain(p, a_last, mid) > target { lo = mid; } else { hi = mid; }
            turn += 1;
        }
        let nf = 0.5 * (lo + hi);
        i = 0;
        while i < p.len { p[i] = c_scale(p[i], 1.0 / nf); i += 1; }
        k = a_last / mathshim::power(nf, order * 1.0);
    }
    var z: Cplx[>..] = [];
    return sig_out(z, p, k);
}

// sig_gpass_gstop checks a ripple and an attenuation in decibels.
fn sig_gpass_gstop(gpass: f64, gstop: f64, tag: const u8[:]) -> Fault {
    if !(gpass > 0.0) {
        return fault(F_VALUE, tag, "the ripple holds a value of zero or below");
    }
    if !(gstop > 0.0) {
        return fault(F_VALUE, tag, "the attenuation holds a value of zero or below");
    }
    if gpass > gstop {
        return fault(F_VALUE, tag, "the ripple holds a value above the attenuation");
    }
    return no_fault();
}

// sig_bso gives the order of a band stop filter for one edge of the
// pass band. The kind holds SIG_BUTTER for Butterworth, SIG_CHEBY1
// and SIG_CHEBY2 for Chebyshev and SIG_ELLIP for elliptic filters.
fn sig_bso(wp: f64, ind: i64, passb: f64[:], stopb: f64[:], gpass: f64, gstop: f64, kind: i64) -> f64, Fault {
    let gf = sig_gpass_gstop(gpass, gstop, "band_stop_obj");
    if gf.kind != F_NONE { return 0.0, gf; }
    if passb.len != 2 || stopb.len != 2 {
        return 0.0, fault(F_SHAPE, "band_stop_obj", "the edges hold a count of two or not");
    }
    if ind < 0 || ind > 1 {
        return 0.0, fault(F_RANGE, "band_stop_obj", "the index of the edge holds a bad value");
    }
    var pb0 = passb[0];
    var pb1 = passb[1];
    if ind == 0 { pb0 = wp; } else { pb1 = wp; }
    var nat: f64 = 0.0;
    var i: i64 = 0;
    while i < 2 {
        let v = stopb[i] * (pb0 - pb1) / (stopb[i] * stopb[i] - pb0 * pb1);
        let av = mathshim::abs_of(v);
        if i == 0 || av < nat { nat = av; }
        i += 1;
    }
    if !(nat > 1.0) {
        return 0.0, fault(F_VALUE, "band_stop_obj", "the natural frequency holds a value of one or below");
    }
    let gs = mathshim::power(10.0, 0.1 * mathshim::abs_of(gstop));
    let gp = mathshim::power(10.0, 0.1 * mathshim::abs_of(gpass));
    if kind == SIG_BUTTER {
        return mathshim::log10_of((gs - 1.0) / (gp - 1.0)) / (2.0 * mathshim::log10_of(nat)), no_fault();
    }
    if kind == SIG_CHEBY1 || kind == SIG_CHEBY2 {
        let v = mathshim::arccosh_of(mathshim::root((gs - 1.0) / (gp - 1.0)));
        return v / mathshim::arccosh_of(nat), no_fault();
    }
    let arg1 = mathshim::root((gp - 1.0) / (gs - 1.0));
    let arg0 = 1.0 / nat;
    let d00 = ell_k(arg0 * arg0);
    let d01 = ell_km1(arg0 * arg0);
    let d10 = ell_k(arg1 * arg1);
    let d11 = ell_km1(arg1 * arg1);
    return d00 * d11 / (d01 * d10), no_fault();
}

// band_stop_obj gives the order of a band stop filter. The wp holds
// the edge of the pass band that moves. The ind holds zero for the low
// edge and one for the high edge. The type of the filter is "butter",
// "cheby" or "ellip".
fn band_stop_obj(wp: f64, ind: i64, passb: f64[:], stopb: f64[:], gpass: f64, gstop: f64, kind: const u8[:]) -> f64, Fault {
    var k: i64 = -1;
    if kind == "butter" { k = 0; }
    else if kind == "cheby" { k = 1; }
    else if kind == "ellip" { k = 2; }
    if k < 0 {
        return 0.0, fault(F_VALUE, "band_stop_obj", "the type of the filter holds an unknown name");
    }
    return sig_bso(wp, ind, passb, stopb, gpass, gstop, k);
}

// sig_bso_min finds the edge of the pass band that gives the smallest
// order of a band stop filter. The method is the bounded search of
// Brent, the method that SciPy runs in fminbound with a tolerance of
// 1e-5. A point where the order is not defined counts as a huge value.
fn sig_bso_min(ind: i64, a: f64, b: f64, passb: f64[:], stopb: f64[:], gpass: f64, gstop: f64, kind: i64) -> f64 {
    let sqrt_eps = 1.4832396974191326e-8;
    let gm = 0.3819660112501051;
    let xatol = 1.0e-5;
    var xa = a;
    var xb = b;
    if xa > xb { let t = xa; xa = xb; xb = t; }
    let fulc0 = xa + gm * (xb - xa);
    var fulc = fulc0;
    var nfc = fulc0;
    var xf = fulc0;
    var rat: f64 = 0.0;
    var e: f64 = 0.0;
    var fx = sig_bso_big(xf, ind, passb, stopb, gpass, gstop, kind);
    var num: i64 = 1;
    var fu: f64 = 1.0e300;
    var ffulc = fx;
    var fnfc = fx;
    var xm = 0.5 * (xa + xb);
    var tol1 = sqrt_eps * mathshim::abs_of(xf) + xatol / 3.0;
    var tol2 = 2.0 * tol1;
    while mathshim::abs_of(xf - xm) > (tol2 - 0.5 * (xb - xa)) && num < 500 {
        var golden = true;
        if mathshim::abs_of(e) > tol1 {
            golden = false;
            let r0 = (xf - nfc) * (fx - ffulc);
            var q = (xf - fulc) * (fx - fnfc);
            var p = (xf - fulc) * q - (xf - nfc) * r0;
            q = 2.0 * (q - r0);
            if q > 0.0 { p = -p; }
            q = mathshim::abs_of(q);
            let r1 = e;
            e = rat;
            if mathshim::abs_of(p) < mathshim::abs_of(0.5 * q * r1) && p > q * (xa - xf) && p < q * (xb - xf) {
                rat = p / q;
                let xp = xf + rat;
                if (xp - xa) < tol2 || (xb - xp) < tol2 {
                    let d = xm - xf;
                    var si = 0.0;
                    if d > 0.0 { si = 1.0; } else if d < 0.0 { si = -1.0; }
                    if d == 0.0 { si = si + 1.0; }
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if golden {
            if xf >= xm { e = xa - xf; } else { e = xb - xf; }
            rat = gm * e;
        }
        var si2 = 0.0;
        if rat > 0.0 { si2 = 1.0; } else if rat < 0.0 { si2 = -1.0; }
        if rat == 0.0 { si2 = si2 + 1.0; }
        var step = mathshim::abs_of(rat);
        if tol1 > step { step = tol1; }
        let x = xf + si2 * step;
        fu = sig_bso_big(x, ind, passb, stopb, gpass, gstop, kind);
        num += 1;
        if fu <= fx {
            if x >= xf { xa = xf; } else { xb = xf; }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf { xa = x; } else { xb = x; }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (xa + xb);
        tol1 = sqrt_eps * mathshim::abs_of(xf) + xatol / 3.0;
        tol2 = 2.0 * tol1;
    }
    return xf;
}

// sig_bso_big wraps sig_bso for the search. A point where the order
// is not defined counts as a huge value, so the search moves away
// from it.
fn sig_bso_big(wp: f64, ind: i64, passb: f64[:], stopb: f64[:], gpass: f64, gstop: f64, kind: i64) -> f64 {
    let v, f = sig_bso(wp, ind, passb, stopb, gpass, gstop, kind);
    if f.kind != F_NONE { return 1.0e300; }
    return v;
}

// sig_ord chooses the order of a filter and gives the edges of the
// pass band. The kind holds the kind of the filter.
fn sig_ord(wp: f64[:], ws: f64[:], gpass: f64, gstop: f64, analog: bool, kind: i64) -> i64, Arr, Fault {
    let gf = sig_gpass_gstop(gpass, gstop, "the order of the filter");
    if gf.kind != F_NONE { return 0, empty_arr(), gf; }
    if wp.len != ws.len || wp.len < 1 || wp.len > 2 {
        return 0, empty_arr(), fault(F_SHAPE, "the order of the filter", "the edges hold a count of one or two, or the two lists differ");
    }
    var i: i64 = 0;
    while i < wp.len {
        if !(wp[i] > 0.0) || !(ws[i] > 0.0) {
            return 0, empty_arr(), fault(F_RANGE, "the order of the filter", "an edge holds a value of zero or below");
        }
        i += 1;
    }
    // The type of the filter: one for low pass, two for high pass,
    // three for band stop and four for band pass.
    var ftype: i64 = 1;
    if wp.len == 2 { ftype = 3; }
    if wp[0] >= ws[0] { ftype += 1; }
    // The edges of the pass band and of the stop band, bent.
    var passb: f64[>..] = [];
    var stopb: f64[>..] = [];
    i = 0;
    while i < wp.len {
        if analog {
            passb.push(wp[i]);
            stopb.push(ws[i]);
        } else {
            passb.push(mathshim::tan_of(SIG_PI * wp[i] / 2.0));
            stopb.push(mathshim::tan_of(SIG_PI * ws[i] / 2.0));
        }
        i += 1;
    }
    // The natural frequency of the prototype.
    var nat: f64 = 0.0;
    if ftype == 1 {
        nat = stopb[0] / passb[0];
    } else if ftype == 2 {
        nat = passb[0] / stopb[0];
    } else if ftype == 3 {
        // The edges of the pass band move to give the smallest order.
        let w0 = sig_bso_min(0, passb[0], stopb[0] - 1.0e-12, passb, stopb, gpass, gstop, kind);
        let w1 = sig_bso_min(1, stopb[1] + 1.0e-12, passb[1], passb, stopb, gpass, gstop, kind);
        passb[0] = w0;
        passb[1] = w1;
        nat = 1.0e300;
        i = 0;
        while i < 2 {
            let v = mathshim::abs_of(stopb[i] * (passb[0] - passb[1]) / (stopb[i] * stopb[i] - passb[0] * passb[1]));
            if v < nat { nat = v; }
            i += 1;
        }
    } else {
        nat = 1.0e300;
        i = 0;
        while i < 2 {
            let v = mathshim::abs_of((stopb[i] * stopb[i] - passb[0] * passb[1]) / (stopb[i] * (passb[0] - passb[1])));
            if v < nat { nat = v; }
            i += 1;
        }
    }
    if !(nat > 1.0) {
        return 0, empty_arr(), fault(F_VALUE, "the order of the filter", "the natural frequency holds a value of one or below");
    }
    let gs = mathshim::power(10.0, 0.1 * mathshim::abs_of(gstop));
    let gp = mathshim::power(10.0, 0.1 * mathshim::abs_of(gpass));
    var ord: i64 = 0;
    var wn: f64[>..] = [];
    if kind == SIG_BUTTER {
        let v = mathshim::log10_of((gs - 1.0) / (gp - 1.0)) / (2.0 * mathshim::log10_of(nat));
        ord = mathshim::ceil_of(v) as i64;
        if ord < 1 {
            return 0, empty_arr(), fault(F_VALUE, "the order of the filter", "the order holds a value of zero or below");
        }
        let w0 = mathshim::power(gp - 1.0, -1.0 / (2.0 * ord));
        if ftype == 1 {
            wn.push(w0 * passb[0]);
        } else if ftype == 2 {
            wn.push(passb[0] / w0);
        } else if ftype == 3 {
            let d = passb[1] - passb[0];
            let discr = mathshim::root(d * d + 4.0 * w0 * w0 * passb[0] * passb[1]);
            let a = mathshim::abs_of((d + discr) / (2.0 * w0));
            let b = mathshim::abs_of((d - discr) / (2.0 * w0));
            if a <= b { wn.push(a); wn.push(b); } else { wn.push(b); wn.push(a); }
        } else {
            let d = passb[1] - passb[0];
            let s = mathshim::root(w0 * w0 / 4.0 * d * d + passb[0] * passb[1]);
            let a = mathshim::abs_of(-w0 * d / 2.0 + s);
            let b = mathshim::abs_of(w0 * d / 2.0 + s);
            if a <= b { wn.push(a); wn.push(b); } else { wn.push(b); wn.push(a); }
        }
    } else if kind == SIG_CHEBY1 || kind == SIG_CHEBY2 {
        let v = mathshim::arccosh_of(mathshim::root((gs - 1.0) / (gp - 1.0)));
        ord = mathshim::ceil_of(v / mathshim::arccosh_of(nat)) as i64;
        if ord < 1 {
            return 0, empty_arr(), fault(F_VALUE, "the order of the filter", "the order holds a value of zero or below");
        }
        if kind == SIG_CHEBY1 {
            i = 0;
            while i < passb.len { wn.push(passb[i]); i += 1; }
        } else {
            let nf = 1.0 / mathshim::cosh_of(v / ord);
            if ftype == 1 {
                wn.push(passb[0] / nf);
            } else if ftype == 2 {
                wn.push(passb[0] * nf);
            } else if ftype == 3 {
                let d = passb[1] - passb[0];
                let a = nf / 2.0 * (passb[0] - passb[1]) + mathshim::root(nf * nf * d * d / 4.0 + passb[1] * passb[0]);
                let b = passb[1] * passb[0] / a;
                wn.push(a);
                wn.push(b);
            } else {
                let d = passb[1] - passb[0];
                let a = 1.0 / (2.0 * nf) * (passb[0] - passb[1]) + mathshim::root(d * d / (4.0 * nf * nf) + passb[1] * passb[0]);
                let b = passb[0] * passb[1] / a;
                wn.push(a);
                wn.push(b);
            }
        }
    } else {
        let arg1_sq = mathshim::expm1_of(0.1 * gpass * 2.302585092994046) / mathshim::expm1_of(0.1 * gstop * 2.302585092994046);
        let arg0 = 1.0 / nat;
        let d00 = ell_k(arg0 * arg0);
        let d01 = ell_km1(arg0 * arg0);
        let d10 = ell_k(arg1_sq);
        let d11 = ell_km1(arg1_sq);
        ord = mathshim::ceil_of(d00 * d11 / (d01 * d10)) as i64;
        if ord < 1 {
            return 0, empty_arr(), fault(F_VALUE, "the order of the filter", "the order holds a value of zero or below");
        }
        i = 0;
        while i < passb.len { wn.push(passb[i]); i += 1; }
    }
    if ord < 1 {
        return 0, empty_arr(), fault(F_VALUE, "the order of the filter", "the order holds a value of zero or below");
    }
    // The edges back from the bent plane.
    if !analog {
        i = 0;
        while i < wn.len { wn[i] = mathshim::arctangent_of(wn[i]) * 2.0 / SIG_PI; i += 1; }
    }
    return ord, flt_arr(copy(wn)), no_fault();
}

// buttord chooses the order of a Butterworth filter. The edges of the
// pass band and of the stop band sit in the range of zero to one for a
// digital filter. An analog filter takes them in radians per second.
fn buttord(wp: f64[:], ws: f64[:], gpass: f64, gstop: f64, analog: bool = false) -> i64, Arr, Fault {
    return sig_ord(wp, ws, gpass, gstop, analog, SIG_BUTTER);
}

// cheb1ord chooses the order of a Chebyshev filter of type I.
fn cheb1ord(wp: f64[:], ws: f64[:], gpass: f64, gstop: f64, analog: bool = false) -> i64, Arr, Fault {
    return sig_ord(wp, ws, gpass, gstop, analog, SIG_CHEBY1);
}

// cheb2ord chooses the order of a Chebyshev filter of type II.
fn cheb2ord(wp: f64[:], ws: f64[:], gpass: f64, gstop: f64, analog: bool = false) -> i64, Arr, Fault {
    return sig_ord(wp, ws, gpass, gstop, analog, SIG_CHEBY2);
}

// ellipord chooses the order of an elliptic filter.
fn ellipord(wp: f64[:], ws: f64[:], gpass: f64, gstop: f64, analog: bool = false) -> i64, Arr, Fault {
    return sig_ord(wp, ws, gpass, gstop, analog, SIG_ELLIP);
}
