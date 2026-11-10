// The elliptic functions that the design of an elliptic filter needs.
// The functions serve ellip only. They are no part of the API of the
// package, and the package may change them. The complete integral
// comes from the arithmetic and geometric mean. The Jacobi functions
// come from the descending Landen transformation, with the series of
// Cephes near the ends of the parameter.
namespace scientific;

import mathshim;

const ELL_PI: f64 = 3.141592653589793;
const ELL_EPS: f64 = 2.0e-16;
const ELL_MACHEP: f64 = 1.1102230246251565e-16;

// ell_agm gives the arithmetic and geometric mean of a and b.
fn ell_agm(a0: f64, b0: f64) -> f64 {
    var a = a0;
    var b = b0;
    var i: i64 = 0;
    while i < 64 {
        let t = 0.5 * (a + b);
        b = mathshim::root(a * b);
        a = t;
        if mathshim::abs_of(a - b) <= 1.0e-16 * a { break; }
        i += 1;
    }
    return a;
}

// ell_k gives the complete integral of the first kind of the
// parameter m, for m from zero up to but not including one.
fn ell_k(m: f64) -> f64 {
    return ELL_PI / (2.0 * ell_agm(1.0, mathshim::root(1.0 - m)));
}

// ell_km1 gives the complete integral of the first kind of the
// parameter 1 - p, which stays exact for a small p.
fn ell_km1(p: f64) -> f64 {
    return ELL_PI / (2.0 * ell_agm(1.0, mathshim::root(p)));
}

// ell_j gives sn, cn and dn of the argument u and the parameter m.
fn ell_j(u: f64, m: f64) -> f64, f64, f64 {
    if m < 1.0e-9 {
        let t = mathshim::sine(u);
        let b = mathshim::cosine(u);
        let ai = 0.25 * m * (u - t * b);
        return t - ai * b, b + ai * t, 1.0 - 0.5 * m * t * t;
    }
    if m >= 0.9999999999 {
        let ai = 0.25 * (1.0 - m);
        let b = mathshim::cosh_of(u);
        let t = mathshim::tanh_of(u);
        let phi = 1.0 / b;
        let twon = b * mathshim::sinh_of(u);
        let sn = t + ai * (twon - u) / (b * b);
        let ai2 = ai * t * phi;
        return sn, phi - ai2 * (twon - u), phi + ai2 * (twon + u);
    }
    var a: f64[>..] = [1.0];
    var c: f64[>..] = [mathshim::root(m)];
    var b = mathshim::root(1.0 - m);
    var n: i64 = 0;
    while n < 8 && mathshim::abs_of(c[n] / a[n]) > ELL_MACHEP {
        let an = a[n];
        a.push(0.5 * (an + b));
        c.push(0.5 * (an - b));
        b = mathshim::root(an * b);
        n += 1;
    }
    var twon: f64 = 1.0;
    var i: i64 = 0;
    while i < n { twon = twon * 2.0; i += 1; }
    var phi = twon * a[n] * u;
    var phi_prev: f64 = phi;
    var k = n;
    while k > 0 {
        let t = c[k] * mathshim::sine(phi) / a[k];
        phi_prev = phi;
        phi = 0.5 * (phi + mathshim::arcsine_of(t));
        k -= 1;
    }
    let sn = mathshim::sine(phi);
    let cn = mathshim::cosine(phi);
    var dn = cn;
    if n > 0 {
        dn = cn / mathshim::cosine(phi_prev - phi);
    } else {
        dn = 1.0;
    }
    return sn, cn, dn;
}

// ell_degree gives the parameter of the degree equation: the m that
// makes the nome of the order n equal to the nome of m1.
fn ell_degree(n: i64, m1: f64) -> f64 {
    let k1 = ell_k(m1);
    let k1p = ell_km1(m1);
    let q1 = mathshim::power_e(-ELL_PI * k1p / k1);
    let q = mathshim::power(q1, 1.0 / n);
    // The sums group the terms in pairs, as NumPy does. The order of
    // the sum decides the last bit of the result.
    var t: f64[>..] = [];
    var i: i64 = 0;
    while i <= 7 { t.push(mathshim::power(q, (i * (i + 1)) * 1.0)); i += 1; }
    let num = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
    i = 0;
    while i <= 7 { t[i] = mathshim::power(q, ((i + 1) * (i + 1)) * 1.0); i += 1; }
    let dsum = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
    let den = 1.0 + 2.0 * dsum;
    let r = num / den;
    return 16.0 * q * mathshim::power(r, 4.0);
}

// ell_arc_sc1 solves w = sc(z, 1 - m) for the real z. The Landen
// transformation runs on an imaginary argument, which stays imaginary,
// so the work stays real.
fn ell_arc_sc1(w: f64, m: f64) -> f64 {
    let k0 = mathshim::root(m);
    if k0 == 1.0 { return mathshim::arctangent_of(w); }
    var ks: f64[>..] = [k0];
    var guardc: i64 = 0;
    var more = true;
    while more {
        let kk = ks[ks.len - 1];
        let kp = mathshim::root((1.0 - kk) * (1.0 + kk));
        let kn = (1.0 - kp) / (1.0 + kp);
        ks.push(kn);
        guardc += 1;
        more = (kn != 0.0) && (guardc < 10);
    }
    var kprod: f64 = 1.0;
    var i: i64 = 1;
    while i < ks.len { kprod = kprod * (1.0 + ks[i]); i += 1; }
    var y = w;
    i = 0;
    while i + 1 < ks.len {
        let t = ks[i] * y;
        y = 2.0 * y / ((1.0 + ks[i + 1]) * (1.0 + mathshim::root(1.0 + t * t)));
        i += 1;
    }
    return kprod * mathshim::arcsinh_of(y);
}

// ellip_proto gives the zeros, the poles and the gain of the analog
// elliptic lowpass prototype of the order n, with a pass band ripple
// of rp decibels and a stop band attenuation of rs decibels.
fn ellip_proto(n: i64, rp: f64, rs: f64, z: Cplx[>..]&, p: Cplx[>..]&) -> f64 {
    let eps_sq = mathshim::expm1_of(0.1 * rp * 2.302585092994046);
    if n == 1 {
        let pole = -mathshim::root(1.0 / eps_sq);
        p.push(Cplx { re: pole, im: 0.0 });
        return -pole;
    }
    let eps = mathshim::root(eps_sq);
    let ck1_sq = eps_sq / mathshim::expm1_of(0.1 * rs * 2.302585092994046);
    let k1 = ell_k(ck1_sq);
    let m = ell_degree(n, ck1_sq);
    let capk = ell_k(m);
    var js: f64[>..] = [];
    var j = 1 - n % 2;
    while j < n { js.push(j * 1.0); j += 2; }
    var s: f64[>..] = [];
    var c: f64[>..] = [];
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < js.len {
        let sn, cn, dn = ell_j(js[i] * capk / n, m);
        s.push(sn);
        c.push(cn);
        d.push(dn);
        i += 1;
    }
    var zs: Cplx[>..] = [];
    i = 0;
    while i < s.len {
        if mathshim::abs_of(s[i]) > ELL_EPS {
            zs.push(Cplx { re: 0.0, im: 1.0 / (mathshim::root(m) * s[i]) });
        }
        i += 1;
    }
    i = 0;
    while i < zs.len { z.push(zs[i]); i += 1; }
    i = 0;
    while i < zs.len { z.push(c_conj(zs[i])); i += 1; }
    let r = ell_arc_sc1(1.0 / eps, ck1_sq);
    let v0 = capk * r / (n * k1);
    let sv, cv, dv = ell_j(v0, 1.0 - m);
    var ps: Cplx[>..] = [];
    i = 0;
    while i < s.len {
        let den = 1.0 - (d[i] * sv) * (d[i] * sv);
        ps.push(Cplx { re: -(c[i] * d[i] * sv * cv) / den, im: -(s[i] * dv) / den });
        i += 1;
    }
    var norm: f64 = 0.0;
    i = 0;
    while i < ps.len { norm += ps[i].re * ps[i].re + ps[i].im * ps[i].im; i += 1; }
    norm = mathshim::root(norm);
    i = 0;
    while i < ps.len { p.push(ps[i]); i += 1; }
    i = 0;
    while i < ps.len {
        if n % 2 == 0 || mathshim::abs_of(ps[i].im) > ELL_EPS * norm { p.push(c_conj(ps[i])); }
        i += 1;
    }
    var kn = Cplx { re: 1.0, im: 0.0 };
    i = 0;
    while i < p.len { kn = c_mul(kn, c_scale(p[i], -1.0)); i += 1; }
    var kd = Cplx { re: 1.0, im: 0.0 };
    i = 0;
    while i < z.len { kd = c_mul(kd, c_scale(z[i], -1.0)); i += 1; }
    var k = c_div(kn, kd).re;
    if n % 2 == 0 { k = k / mathshim::root(1.0 + eps_sq); }
    return k;
}
