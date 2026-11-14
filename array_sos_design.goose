// The design of Butterworth, Chebyshev, elliptic and Bessel filters as sections.
//
namespace scientific;

import mathshim;

// butter_zpk gives the zeros, the poles and the gain of a digital
// Butterworth filter. The steps follow SciPy: the poles of the analog
// prototype, the map to the band, the bilinear rule.
fn butter_zpk(order: i64, wn: f64[:], btype: const u8[:], z: Cplx[>..]&, p: Cplx[>..]&) -> f64 {
    var p0: Cplx[>..] = [];
    var m = 1 - order;
    while m < order {
        let ang = PI_VALUE * m / (2.0 * order);
        p0.push(Cplx { re: -mathshim::cosine(ang), im: -mathshim::sine(ang) });
        m += 2;
    }
    return proto_zpk(order, wn, btype, p0, 1.0, z, p);
}

// proto_zpk maps the poles p0 and the gain kp0 of an analog lowpass
// prototype without zeros to a digital filter.
fn proto_zpk(order: i64, wn: f64[:], btype: const u8[:], p0: Cplx[:], kp0: f64, z: Cplx[>..]&, p: Cplx[>..]&) -> f64 {
    var none: Cplx[>..] = [];
    return proto_zpk2(order, wn, btype, none, p0, kp0, z, p);
}

// cx_prod_neg gives the product of the negated values of a list.
fn cx_prod_neg(a: Cplx[:]) -> Cplx {
    var t = Cplx { re: 1.0, im: 0.0 };
    var i: i64 = 0;
    while i < a.len { t = c_mul(t, c_scale(a[i], -1.0)); i += 1; }
    return t;
}

// proto_zpk2 maps the zeros z0, the poles p0 and the gain kp0 of an
// analog lowpass prototype to a digital filter. The map to the band and
// the bilinear rule follow SciPy: lp2lp, lp2hp, lp2bp or lp2bs, then
// bilinear with a sampling rate of two.
fn proto_zpk2(order: i64, wn: f64[:], btype: const u8[:], z0: Cplx[:], p0: Cplx[:], kp0: f64, z: Cplx[>..]&, p: Cplx[>..]&) -> f64 {
    let zero = Cplx { re: 0.0, im: 0.0 };
    let w0 = 4.0 * mathshim::tan_of(PI_VALUE * wn[0] / 2.0);
    let degree = p0.len - z0.len;
    var k: f64 = kp0;
    // The real ratio of the product of the negated zeros to that of the
    // poles, for the types that invert the prototype.
    let ratio = c_div(cx_prod_neg(z0), cx_prod_neg(p0)).re;
    var sz: Cplx[>..] = [];
    var sp: Cplx[>..] = [];
    var i: i64 = 0;
    if btype == "lowpass" {
        while i < p0.len { sp.push(c_scale(p0[i], w0)); i += 1; }
        i = 0;
        while i < z0.len { sz.push(c_scale(z0[i], w0)); i += 1; }
        i = 0;
        while i < degree { k = k * w0; i += 1; }
    } else if btype == "highpass" {
        while i < p0.len { sp.push(c_div(Cplx { re: w0, im: 0.0 }, p0[i])); i += 1; }
        i = 0;
        while i < z0.len { sz.push(c_div(Cplx { re: w0, im: 0.0 }, z0[i])); i += 1; }
        i = 0;
        while i < degree { sz.push(zero); i += 1; }
        k = k * ratio;
    } else {
        let w1 = 4.0 * mathshim::tan_of(PI_VALUE * wn[1] / 2.0);
        let bw = w1 - w0;
        let wo = mathshim::root(w0 * w1);
        let wo2 = Cplx { re: wo * wo, im: 0.0 };
        var plus: Cplx[>..] = [];
        var minus: Cplx[>..] = [];
        var zplus: Cplx[>..] = [];
        var zminus: Cplx[>..] = [];
        i = 0;
        while i < p0.len {
            var pl: Cplx;
            if btype == "bandpass" { pl = c_scale(p0[i], bw / 2.0); }
            else { pl = c_div(Cplx { re: bw / 2.0, im: 0.0 }, p0[i]); }
            let r = c_sqrt(c_sub(c_mul(pl, pl), wo2));
            plus.push(c_add(pl, r));
            minus.push(c_sub(pl, r));
            i += 1;
        }
        i = 0;
        while i < z0.len {
            var zl: Cplx;
            if btype == "bandpass" { zl = c_scale(z0[i], bw / 2.0); }
            else { zl = c_div(Cplx { re: bw / 2.0, im: 0.0 }, z0[i]); }
            let r = c_sqrt(c_sub(c_mul(zl, zl), wo2));
            zplus.push(c_add(zl, r));
            zminus.push(c_sub(zl, r));
            i += 1;
        }
        cx_join(plus, minus, sp);
        cx_join(zplus, zminus, sz);
        if btype == "bandpass" {
            i = 0;
            while i < degree { sz.push(zero); k = k * bw; i += 1; }
        } else {
            i = 0;
            while i < degree { sz.push(Cplx { re: 0.0, im: wo }); i += 1; }
            i = 0;
            while i < degree { sz.push(Cplx { re: 0.0, im: -wo }); i += 1; }
            k = k * ratio;
        }
    }
    // The bilinear rule with a sampling rate of two.
    var kz = Cplx { re: 1.0, im: 0.0 };
    var kq = Cplx { re: 1.0, im: 0.0 };
    let f4 = Cplx { re: 4.0, im: 0.0 };
    i = 0;
    while i < sz.len {
        kz = c_mul(kz, c_sub(f4, sz[i]));
        z.push(c_div(c_add(f4, sz[i]), c_sub(f4, sz[i])));
        i += 1;
    }
    i = 0;
    while i < sp.len {
        kq = c_mul(kq, c_sub(f4, sp[i]));
        p.push(c_div(c_add(f4, sp[i]), c_sub(f4, sp[i])));
        i += 1;
    }
    i = sz.len;
    while i < sp.len { z.push(Cplx { re: -1.0, im: 0.0 }); i += 1; }
    return k * c_div(kz, kq).re;
}

// butter_sos designs a digital Butterworth filter and gives it as
// sections of the order two. The arguments are those of butter. The
// result holds one row per section: b0, b1, b2, a0, a1 and a2.
fn butter_sos(order: i64, wn: f64[:], btype: const u8[:]) -> Arr, Fault {
    let chk = flt_check_design(order, wn, btype, "butter_sos");
    if chk.kind != F_NONE { return empty_arr(), chk; }
    var z: Cplx[>..] = [];
    var p: Cplx[>..] = [];
    let k = butter_zpk(order, wn, btype, z, p);
    return sos_from_zpk(z, p, k);
}

// sosfilt_zi gives the initial state of each section for a step input
// of one. The result holds one row of two values per section. Scale
// the rows by the first sample of the signal for the start of a
// filter that has no jump at the start.
fn sosfilt_zi(sos) -> Arr, Fault {
    let chk = sos_check(sos, "sosfilt_zi");
    if chk.kind != F_NONE { return empty_arr(), chk; }
    var out: f64[>..] = [];
    var scale: f64 = 1.0;
    var s: i64 = 0;
    while s < sos.shape[0] {
        let r = s * 6;
        let pb: f64[>..] = [sos.data[r], sos.data[r + 1], sos.data[r + 2]];
        let pa: f64[>..] = [1.0, sos.data[r + 4], sos.data[r + 5]];
        var zi: f64[>..] = [];
        let f = flt_zi(pb, pa, zi);
        if f.kind != F_NONE { return empty_arr(), fault(F_VALUE, "sosfilt_zi", "a section holds a denominator that sums to zero"); }
        out.push(scale * zi[0]);
        out.push(scale * zi[1]);
        scale = scale * (pb[0] + pb[1] + pb[2]) / (1.0 + pa[1] + pa[2]);
        s += 1;
    }
    var sh: i64[..8] = [];
    sh.push(sos.shape[0]);
    sh.push(2);
    return Arr { ndim: 2, shape: sh, data: copy(out) }, no_fault();
}

// sosfiltfilt filters a signal forward and backward with sections, so
// the phase shift is zero. The extension is the odd one of SciPy, with
// a pad of three times the count of taps.
fn sosfiltfilt(sos, x: f64[:]) -> Arr, Fault {
    let chk = sos_check(sos, "sosfiltfilt");
    if chk.kind != F_NONE { return empty_arr(), chk; }
    let n = sos.shape[0];
    var zb: i64 = 0;
    var za: i64 = 0;
    var s: i64 = 0;
    while s < n {
        if sos.data[s * 6 + 2] == 0.0 { zb += 1; }
        if sos.data[s * 6 + 5] == 0.0 { za += 1; }
        s += 1;
    }
    var taps = 2 * n + 1;
    if zb < za { taps -= zb; } else { taps -= za; }
    let edge = 3 * taps;
    if x.len <= edge {
        return empty_arr(), fault(F_SHAPE, "sosfiltfilt", "the signal is not longer than three times the taps");
    }
    let zi, f = sosfilt_zi(sos);
    if f.kind != F_NONE { return empty_arr(), f; }
    var ext: f64[>..] = [];
    flt_odd(x, edge, ext);
    var y: f64[>..] = [];
    sos_cascade(sos, ext, zi.data, ext[0], y);
    var r: f64[>..] = [];
    var i: i64 = 0;
    while i < y.len { r.push(y[y.len - 1 - i]); i += 1; }
    var y2: f64[>..] = [];
    sos_cascade(sos, r, zi.data, r[0], y2);
    var out: f64[>..] = [];
    i = 0;
    while i < x.len { out.push(y2[y2.len - 1 - edge - i]); i += 1; }
    return flt_arr(copy(out)), no_fault();
}

// cheby1_digital gives the digital zeros, poles and gain of the
// Chebyshev filter of type I. The arguments are already checked.
fn cheby1_digital(order: i64, rp: f64, wn: f64[:], btype: const u8[:], z: Cplx[>..]&, p: Cplx[>..]&) -> f64 {
    let eps = mathshim::root(mathshim::power(10.0, 0.1 * rp) - 1.0);
    let mu = mathshim::arcsinh_of(1.0 / eps) / order;
    var p0: Cplx[>..] = [];
    var m = 1 - order;
    while m < order {
        let th = PI_VALUE * m / (2.0 * order);
        p0.push(Cplx { re: -mathshim::sinh_of(mu) * mathshim::cosine(th), im: -mathshim::cosh_of(mu) * mathshim::sine(th) });
        m += 2;
    }
    var prod = Cplx { re: 1.0, im: 0.0 };
    var i: i64 = 0;
    while i < p0.len { prod = c_mul(prod, c_scale(p0[i], -1.0)); i += 1; }
    var kp = prod.re;
    if order % 2 == 0 { kp = kp / mathshim::root(1.0 + eps * eps); }
    return proto_zpk(order, wn, btype, p0, kp, z, p);
}

// cheby2_digital gives the digital zeros, poles and gain of the
// Chebyshev filter of type II. The arguments are already checked.
fn cheby2_digital(order: i64, rs: f64, wn: f64[:], btype: const u8[:], z: Cplx[>..]&, p: Cplx[>..]&) -> f64 {
    let de = 1.0 / mathshim::root(mathshim::power(10.0, 0.1 * rs) - 1.0);
    let mu = mathshim::arcsinh_of(1.0 / de) / order;
    var z0: Cplx[>..] = [];
    var p0: Cplx[>..] = [];
    var m = 1 - order;
    while m < order {
        let th = PI_VALUE * m / (2.0 * order);
        if m != 0 {
            // The zero is the conjugate of j over the sine of the angle.
            z0.push(Cplx { re: 0.0, im: -1.0 / mathshim::sine(th) });
        }
        // The pole is the inverse of sinh(mu) re + j cosh(mu) im of -e^(j th).
        let pr = -mathshim::sinh_of(mu) * mathshim::cosine(th);
        let pi_ = -mathshim::cosh_of(mu) * mathshim::sine(th);
        p0.push(c_div(Cplx { re: 1.0, im: 0.0 }, Cplx { re: pr, im: pi_ }));
        m += 2;
    }
    let kp = c_div(cx_prod_neg(p0), cx_prod_neg(z0)).re;
    return proto_zpk2(order, wn, btype, z0, p0, kp, z, p);
}

// ripple_fault checks a ripple or an attenuation in decibels.
fn ripple_fault(v: f64, tag: const u8[:], what: const u8[:]) -> Fault {
    if !(v > 0.0) || !(v < 1.0e6) {
        return fault(F_VALUE, tag, what);
    }
    return no_fault();
}

// zpk_to_ba gives the coefficients of the numerator and the denominator
// from the zeros, the poles and the gain. The roots come in pairs of
// conjugates, so the real parts hold the polynomials.
fn zpk_to_ba(z: Cplx[:], p: Cplx[:], k: f64) -> Arr, Arr, Fault {
    var nb: f64[>..] = [];
    var na: f64[>..] = [];
    let fz = roots_poly(z, nb);
    if fz.kind != F_NONE { return empty_arr(), empty_arr(), fz; }
    let fp = roots_poly(p, na);
    if fp.kind != F_NONE { return empty_arr(), empty_arr(), fp; }
    var i: i64 = 0;
    while i < nb.len { nb[i] = nb[i] * k; i += 1; }
    return flt_arr(copy(nb)), flt_arr(copy(na)), no_fault();
}

// roots_poly writes the coefficients of the monic polynomial of the
// roots, from the highest power to the lowest. A real root gives the
// factor (x - r). A pair of conjugate roots gives the factor
// (x^2 - 2 re x + re^2 + im^2). The product of a pair keeps the
// polynomial real and exact. A root without a partner gives F_SHAPE,
// because the polynomial would not be real.
fn roots_poly(r: Cplx[:], out: f64[>..]&) -> Fault {
    var cr: f64[>..] = [1.0];
    var used: f64[>..] = [];
    var i: i64 = 0;
    while i < r.len { used.push(0.0); i += 1; }
    var j: i64 = 0;
    while j < r.len {
        if used[j] != 0.0 { j += 1; continue; }
        let im = r[j].im;
        if mathshim::abs_of(im) <= 1.0e-13 * (1.0 + mathshim::abs_of(r[j].re)) {
            used[j] = 1.0;
            // The factor (x - re). The new lowest slot is the zero of
            // the product, and the loop runs from the top down.
            cr.push(0.0);
            var k = cr.len - 1;
            while k >= 1 {
                cr[k] = cr[k] - r[j].re * cr[k - 1];
                k -= 1;
            }
            j += 1;
        } else {
            // The partner of the root, with the opposite imaginary part.
            var k: i64 = 0;
            var found = false;
            while k < r.len {
                if k != j && used[k] == 0.0 && mathshim::abs_of(r[k].re - r[j].re) <= 1.0e-10 * (1.0 + mathshim::abs_of(r[j].re)) && mathshim::abs_of(r[k].im + im) <= 1.0e-10 * (1.0 + mathshim::abs_of(im)) {
                    found = true;
                    break;
                }
                k += 1;
            }
            if !found {
                return fault(F_SHAPE, "roots_poly", "a root has no conjugate partner");
            }
            used[j] = 1.0;
            used[k] = 1.0;
            let re = r[j].re;
            let c1 = -2.0 * re;
            let c0 = re * re + im * im;
            let n = cr.len - 1;
            var nr: f64[>..] = [];
            var q: i64 = 0;
            while q <= n + 2 { nr.push(0.0); q += 1; }
            q = 0;
            while q <= n {
                nr[q] = nr[q] + cr[q];
                nr[q + 1] = nr[q + 1] + c1 * cr[q];
                nr[q + 2] = nr[q + 2] + c0 * cr[q];
                q += 1;
            }
            cr = copy(nr);
            j += 1;
        }
    }
    i = 0;
    while i < cr.len { out.push(cr[i]); i += 1; }
    return no_fault();
}

// ellip designs a digital elliptic filter with a pass band ripple of
// rp decibels and a stop band attenuation of rs decibels. The
// attenuation must be above the ripple. The cut is a fraction of the
// Nyquist rate. The result holds the numerator and the denominator.
fn ellip(order: i64, rp: f64, rs: f64, wn: f64[:], btype: const u8[:]) -> Arr, Arr, Fault {
    let chk = flt_check_design(order, wn, btype, "ellip");
    if chk.kind != F_NONE { return empty_arr(), empty_arr(), chk; }
    let f1 = ripple_fault(rp, "ellip", "the ripple holds a value outside zero to a million");
    if f1.kind != F_NONE { return empty_arr(), empty_arr(), f1; }
    let f2 = ripple_fault(rs, "ellip", "the attenuation holds a value outside zero to a million");
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
    if !(rs > rp) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "ellip", "the attenuation is not above the ripple");
    }
    var z0: Cplx[>..] = [];
    var p0: Cplx[>..] = [];
    let kp = ellip_proto(order, rp, rs, z0, p0);
    var z: Cplx[>..] = [];
    var p: Cplx[>..] = [];
    let k = proto_zpk2(order, wn, btype, z0, p0, kp, z, p);
    let b, a, fb = zpk_to_ba(z, p, k);
    return b, a, fb;
}

// cheby1 designs a digital Chebyshev filter of type I with a ripple of
// rp decibels in the pass band. The cut is a fraction of the Nyquist
// rate, as in butter. The result holds the numerator and the
// denominator.
fn cheby1(order: i64, rp: f64, wn: f64[:], btype: const u8[:]) -> Arr, Arr, Fault {
    let chk = flt_check_design(order, wn, btype, "cheby1");
    if chk.kind != F_NONE { return empty_arr(), empty_arr(), chk; }
    let rf = ripple_fault(rp, "cheby1", "the ripple holds a value outside zero to a million");
    if rf.kind != F_NONE { return empty_arr(), empty_arr(), rf; }
    var z: Cplx[>..] = [];
    var p: Cplx[>..] = [];
    let k = cheby1_digital(order, rp, wn, btype, z, p);
    let b, a, fb = zpk_to_ba(z, p, k);
    return b, a, fb;
}

// cheby2 designs a digital Chebyshev filter of type II with a stop band
// attenuation of rs decibels. The cuts are the edges of the stop band.
// The result holds the numerator and the denominator.
fn cheby2(order: i64, rs: f64, wn: f64[:], btype: const u8[:]) -> Arr, Arr, Fault {
    let chk = flt_check_design(order, wn, btype, "cheby2");
    if chk.kind != F_NONE { return empty_arr(), empty_arr(), chk; }
    let rf = ripple_fault(rs, "cheby2", "the attenuation holds a value outside zero to a million");
    if rf.kind != F_NONE { return empty_arr(), empty_arr(), rf; }
    var z: Cplx[>..] = [];
    var p: Cplx[>..] = [];
    let k = cheby2_digital(order, rs, wn, btype, z, p);
    let b, a, fb = zpk_to_ba(z, p, k);
    return b, a, fb;
}

// cheby1_sos designs a digital Chebyshev filter of type I with a ripple
// of rp decibels in the pass band, and gives it as sections of the
// order two. The arguments follow cheby1.
fn cheby1_sos(order: i64, rp: f64, wn: f64[:], btype: const u8[:]) -> Arr, Fault {
    let chk = flt_check_design(order, wn, btype, "cheby1_sos");
    if chk.kind != F_NONE { return empty_arr(), chk; }
    let rf = ripple_fault(rp, "cheby1_sos", "the ripple holds a value outside zero to a million");
    if rf.kind != F_NONE { return empty_arr(), rf; }
    var z: Cplx[>..] = [];
    var p: Cplx[>..] = [];
    let k = cheby1_digital(order, rp, wn, btype, z, p);
    return sos_from_zpk(z, p, k);
}

// cheby2_sos designs a digital Chebyshev filter of type II with a stop
// band attenuation of rs decibels, and gives it as sections of the
// order two. The cuts are the edges of the stop band, as in cheby2.
fn cheby2_sos(order: i64, rs: f64, wn: f64[:], btype: const u8[:]) -> Arr, Fault {
    let chk = flt_check_design(order, wn, btype, "cheby2_sos");
    if chk.kind != F_NONE { return empty_arr(), chk; }
    let rf = ripple_fault(rs, "cheby2_sos", "the attenuation holds a value outside zero to a million");
    if rf.kind != F_NONE { return empty_arr(), rf; }
    var z: Cplx[>..] = [];
    var p: Cplx[>..] = [];
    let k = cheby2_digital(order, rs, wn, btype, z, p);
    return sos_from_zpk(z, p, k);
}

// bessel_poly gives the coefficients of the reverse Bessel polynomial
// of the order n, from the highest power to the lowest.
fn bessel_poly(n: i64) -> f64[>..] {
    var prev: f64[>..] = [1.0];
    var cur: f64[>..] = [1.0, 1.0];
    var m: i64 = 2;
    while m <= n {
        // theta(m) = (2m - 1) theta(m - 1) + s^2 theta(m - 2), lowest power first.
        var next: f64[>..] = [];
        var i: i64 = 0;
        while i <= m {
            var v: f64 = 0.0;
            if i < cur.len { v = (2.0 * m - 1.0) * cur[i]; }
            if i >= 2 && i - 2 < prev.len { v += prev[i - 2]; }
            next.push(v);
            i += 1;
        }
        prev = copy(cur);
        cur = copy(next);
        m += 1;
    }
    var out: f64[>..] = [];
    var i = cur.len - 1;
    while i >= 0 { out.push(cur[i]); i -= 1; }
    return out;
}

// bessel_polish improves a root of a polynomial by the rule of Newton.
fn bessel_polish(c: f64[:], z0: Cplx) -> Cplx {
    var dc: f64[>..] = [];
    let n = c.len - 1;
    var i: i64 = 0;
    while i < n { dc.push(c[i] * (n - i)); i += 1; }
    var z = z0;
    var turn: i64 = 0;
    while turn < 8 {
        let d = c_value(dc, z);
        if c_abs(d) == 0.0 { break; }
        z = c_sub(z, c_div(c_value(c, z), d));
        turn += 1;
    }
    return z;
}

// bessel_gain gives the size of the gain of the all pole filter with
// the poles p and the gain k at the angular frequency w.
fn bessel_gain(p: Cplx[:], k: f64, w: f64) -> f64 {
    var t = Cplx { re: 1.0, im: 0.0 };
    var i: i64 = 0;
    while i < p.len { t = c_mul(t, c_sub(Cplx { re: 0.0, im: w }, p[i])); i += 1; }
    return k / c_abs(t);
}

// bessel_digital gives the digital zeros, poles and gain of the Bessel
// filter. The norm is "phase", "delay" or "mag". The arguments are
// already checked.
fn bessel_digital(order: i64, wn: f64[:], btype: const u8[:], norm: const u8[:], z: Cplx[>..]&, p: Cplx[>..]&) -> f64, Fault {
    let poly = bessel_poly(order);
    let rts, rf = roots_of(poly);
    if rf.kind != F_NONE { return 0.0, rf; }
    var p0: Cplx[>..] = [];
    var i: i64 = 0;
    while i < rts.len { p0.push(bessel_polish(poly, rts[i])); i += 1; }
    let a_last = poly[poly.len - 1];
    var kp = a_last;
    if norm == "phase" {
        let f = mathshim::power(10.0, -mathshim::log10_of(a_last) / order);
        i = 0;
        while i < p0.len { p0[i] = c_scale(p0[i], f); i += 1; }
        kp = 1.0;
    } else if norm == "mag" {
        // The gain falls with the frequency, so bisect for the 3 dB point.
        let target = 0.7071067811865476;
        var hi: f64 = 1.0;
        while bessel_gain(p0, a_last, hi) > target { hi = hi * 2.0; }
        var lo: f64 = 0.0;
        var turn: i64 = 0;
        while turn < 200 {
            let mid = 0.5 * (lo + hi);
            if bessel_gain(p0, a_last, mid) > target { lo = mid; } else { hi = mid; }
            turn += 1;
        }
        let nf = 0.5 * (lo + hi);
        i = 0;
        while i < p0.len { p0[i] = c_scale(p0[i], 1.0 / nf); i += 1; }
        kp = a_last / mathshim::power(nf, order * 1.0);
    }
    return proto_zpk(order, wn, btype, p0, kp, z, p), no_fault();
}

// bessel_check checks the name of the norm.
fn bessel_check(norm: const u8[:], tag: const u8[:]) -> Fault {
    if norm != "phase" && norm != "delay" && norm != "mag" {
        return fault(F_VALUE, tag, "the norm holds an unknown name");
    }
    return no_fault();
}

// bessel designs a digital Bessel filter. The cut is a fraction of the
// Nyquist rate. The norm is "phase", "delay" or "mag", and says what
// the cut means, as in SciPy. The result holds the numerator and the
// denominator.
fn bessel(order: i64, wn: f64[:], btype: const u8[:], norm: const u8[:] = "phase") -> Arr, Arr, Fault {
    let chk = flt_check_design(order, wn, btype, "bessel");
    if chk.kind != F_NONE { return empty_arr(), empty_arr(), chk; }
    let nf = bessel_check(norm, "bessel");
    if nf.kind != F_NONE { return empty_arr(), empty_arr(), nf; }
    var z: Cplx[>..] = [];
    var p: Cplx[>..] = [];
    let k, df = bessel_digital(order, wn, btype, norm, z, p);
    if df.kind != F_NONE { return empty_arr(), empty_arr(), df; }
    let b, a, fb = zpk_to_ba(z, p, k);
    return b, a, fb;
}

// bessel_sos designs the same filter as bessel and gives it as
// sections of the order two.
fn bessel_sos(order: i64, wn: f64[:], btype: const u8[:], norm: const u8[:] = "phase") -> Arr, Fault {
    let chk = flt_check_design(order, wn, btype, "bessel_sos");
    if chk.kind != F_NONE { return empty_arr(), chk; }
    let nf = bessel_check(norm, "bessel_sos");
    if nf.kind != F_NONE { return empty_arr(), nf; }
    var z: Cplx[>..] = [];
    var p: Cplx[>..] = [];
    let k, df = bessel_digital(order, wn, btype, norm, z, p);
    if df.kind != F_NONE { return empty_arr(), df; }
    return sos_from_zpk(z, p, k);
}
