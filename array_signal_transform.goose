// The transforms of a filter: the changes between the low pass and the
// other bands, and the bilinear rule.
//
// A transform takes the coefficients of the numerator and the
// denominator, or the zeros, the poles and the gain, and maps them to
// another band. The names follow the calls of scipy.signal. A set of
// roots comes as two lists: the real parts and the imaginary parts.
// The maps follow the exact formulas of SciPy 1.18.1.
namespace scientific;

import mathshim;

// sig_rel_degree gives the count of poles beyond the zeros.
fn sig_rel_degree(z: Cplx[:], p: Cplx[:], tag: const u8[:]) -> i64, Fault {
    let degree = p.len - z.len;
    if degree < 0 {
        return 0, fault(F_VALUE, tag, "the filter holds more zeros than poles");
    }
    return degree, no_fault();
}

// sig_shift writes the same complex number a count of times.
fn sig_shift(c: Cplx, count: i64, out: Cplx[>..]&) {
    var i: i64 = 0;
    while i < count { out.push(c); i += 1; }
}

// sig_lp2lp_zpk scales the zeros and the poles of an analog lowpass
// filter to a new cut.
fn sig_lp2lp_zpk(z: Cplx[:], p: Cplx[:], k: f64, wo: f64) -> Cplx[>..], Cplx[>..], f64, Fault {
    let degree, f = sig_rel_degree(z, p, "lp2lp_zpk");
    if f.kind != F_NONE { return [], [], 0.0, f; }
    var zz: Cplx[>..] = [];
    var pp: Cplx[>..] = [];
    var i: i64 = 0;
    while i < z.len { zz.push(c_scale(z[i], wo)); i += 1; }
    i = 0;
    while i < p.len { pp.push(c_scale(p[i], wo)); i += 1; }
    return zz, pp, k * mathshim::power(wo, degree * 1.0), no_fault();
}

// sig_lp2hp_zpk inverts the zeros and the poles about the origin.
fn sig_lp2hp_zpk(z: Cplx[:], p: Cplx[:], k: f64, wo: f64) -> Cplx[>..], Cplx[>..], f64, Fault {
    let degree, f = sig_rel_degree(z, p, "lp2hp_zpk");
    if f.kind != F_NONE { return [], [], 0.0, f; }
    var zz: Cplx[>..] = [];
    var pp: Cplx[>..] = [];
    let one = Cplx { re: wo, im: 0.0 };
    var i: i64 = 0;
    while i < z.len {
        if z[i].re == 0.0 && z[i].im == 0.0 {
            return [], [], 0.0, fault(F_VALUE, "lp2hp_zpk", "a zero holds the value zero");
        }
        zz.push(c_div(one, z[i]));
        i += 1;
    }
    i = 0;
    while i < p.len {
        if p[i].re == 0.0 && p[i].im == 0.0 {
            return [], [], 0.0, fault(F_VALUE, "lp2hp_zpk", "a pole holds the value zero");
        }
        pp.push(c_div(one, p[i]));
        i += 1;
    }
    sig_shift(Cplx { re: 0.0, im: 0.0 }, degree, zz);
    return zz, pp, k * c_div(cx_prod_neg(z), cx_prod_neg(p)).re, no_fault();
}

// sig_lp2bp_zpk maps a lowpass filter to a band pass filter.
fn sig_lp2bp_zpk(z: Cplx[:], p: Cplx[:], k: f64, wo: f64, bw: f64) -> Cplx[>..], Cplx[>..], f64, Fault {
    let degree, f = sig_rel_degree(z, p, "lp2bp_zpk");
    if f.kind != F_NONE { return [], [], 0.0, f; }
    let wo2 = Cplx { re: wo * wo, im: 0.0 };
    let half = bw / 2.0;
    var zp: Cplx[>..] = [];
    var zm: Cplx[>..] = [];
    var pp: Cplx[>..] = [];
    var pm: Cplx[>..] = [];
    var i: i64 = 0;
    while i < z.len {
        let base = c_scale(z[i], half);
        let disc = c_sqrt(c_sub(c_mul(base, base), wo2));
        zp.push(c_add(base, disc));
        zm.push(c_sub(base, disc));
        i += 1;
    }
    i = 0;
    while i < p.len {
        let base = c_scale(p[i], half);
        let disc = c_sqrt(c_sub(c_mul(base, base), wo2));
        pp.push(c_add(base, disc));
        pm.push(c_sub(base, disc));
        i += 1;
    }
    var zz: Cplx[>..] = [];
    var pz: Cplx[>..] = [];
    cx_join(zp, zm, zz);
    cx_join(pp, pm, pz);
    sig_shift(Cplx { re: 0.0, im: 0.0 }, degree, zz);
    return zz, pz, k * mathshim::power(bw, degree * 1.0), no_fault();
}

// sig_lp2bs_zpk maps a lowpass filter to a band stop filter.
fn sig_lp2bs_zpk(z: Cplx[:], p: Cplx[:], k: f64, wo: f64, bw: f64) -> Cplx[>..], Cplx[>..], f64, Fault {
    let degree, f = sig_rel_degree(z, p, "lp2bs_zpk");
    if f.kind != F_NONE { return [], [], 0.0, f; }
    let wo2 = Cplx { re: wo * wo, im: 0.0 };
    let half = bw / 2.0;
    var zp: Cplx[>..] = [];
    var zm: Cplx[>..] = [];
    var pp: Cplx[>..] = [];
    var pm: Cplx[>..] = [];
    var i: i64 = 0;
    while i < z.len {
        if z[i].re == 0.0 && z[i].im == 0.0 {
            return [], [], 0.0, fault(F_VALUE, "lp2bs_zpk", "a zero holds the value zero");
        }
        let base = c_div(Cplx { re: half, im: 0.0 }, z[i]);
        let disc = c_sqrt(c_sub(c_mul(base, base), wo2));
        zp.push(c_add(base, disc));
        zm.push(c_sub(base, disc));
        i += 1;
    }
    i = 0;
    while i < p.len {
        if p[i].re == 0.0 && p[i].im == 0.0 {
            return [], [], 0.0, fault(F_VALUE, "lp2bs_zpk", "a pole holds the value zero");
        }
        let base = c_div(Cplx { re: half, im: 0.0 }, p[i]);
        let disc = c_sqrt(c_sub(c_mul(base, base), wo2));
        pp.push(c_add(base, disc));
        pm.push(c_sub(base, disc));
        i += 1;
    }
    var zz: Cplx[>..] = [];
    var pz: Cplx[>..] = [];
    cx_join(zp, zm, zz);
    cx_join(pp, pm, pz);
    sig_shift(Cplx { re: 0.0, im: wo }, degree, zz);
    sig_shift(Cplx { re: 0.0, im: -wo }, degree, zz);
    return zz, pz, k * c_div(cx_prod_neg(z), cx_prod_neg(p)).re, no_fault();
}

// sig_bilinear_zpk maps an analog filter to the digital plane with the
// bilinear rule at the rate fs.
fn sig_bilinear_zpk(z: Cplx[:], p: Cplx[:], k: f64, fs: f64) -> Cplx[>..], Cplx[>..], f64, Fault {
    let degree, f = sig_rel_degree(z, p, "bilinear_zpk");
    if f.kind != F_NONE { return [], [], 0.0, f; }
    let fs2 = Cplx { re: 2.0 * fs, im: 0.0 };
    var kz = Cplx { re: 1.0, im: 0.0 };
    var kq = Cplx { re: 1.0, im: 0.0 };
    var zz: Cplx[>..] = [];
    var pp: Cplx[>..] = [];
    var i: i64 = 0;
    while i < z.len {
        let dn = c_sub(fs2, z[i]);
        if dn.re == 0.0 && dn.im == 0.0 {
            return [], [], 0.0, fault(F_VALUE, "bilinear_zpk", "a zero sits at the point of the map");
        }
        kz = c_mul(kz, dn);
        zz.push(c_div(c_add(fs2, z[i]), dn));
        i += 1;
    }
    i = 0;
    while i < p.len {
        let dn = c_sub(fs2, p[i]);
        if dn.re == 0.0 && dn.im == 0.0 {
            return [], [], 0.0, fault(F_VALUE, "bilinear_zpk", "a pole sits at the point of the map");
        }
        kq = c_mul(kq, dn);
        pp.push(c_div(c_add(fs2, p[i]), dn));
        i += 1;
    }
    sig_shift(Cplx { re: -1.0, im: 0.0 }, degree, zz);
    return zz, pp, k * c_div(kz, kq).re, no_fault();
}

// sig_zpk_repack gives four lists of real numbers and the gain from a
// set of complex roots.
fn sig_zpk_repack(z: Cplx[:], p: Cplx[:], k: f64, tag: const u8[:]) -> Arr, Arr, Arr, Arr, f64, Fault {
    var zr: f64[>..] = [];
    var zi: f64[>..] = [];
    var pr: f64[>..] = [];
    var pi_: f64[>..] = [];
    sig_split(z, zr, zi);
    sig_split(p, pr, pi_);
    if !all_finite(zr) || !all_finite(zi) || !all_finite(pr) || !all_finite(pi_) {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_VALUE, tag, "a root holds a value that is not finite");
    }
    return flt_arr(copy(zr)), flt_arr(copy(zi)), flt_arr(copy(pr)), flt_arr(copy(pi_)), k, no_fault();
}

// sig_zpk_in reads four lists of real numbers as a set of complex
// roots.
fn sig_zpk_in(zr: f64[:], zi: f64[:], pr: f64[:], pi_: f64[:], k: f64, tag: const u8[:]) -> Cplx[>..], Cplx[>..], f64, Fault {
    let fz = form_roots_check(zr, zi, tag);
    if fz.kind != F_NONE { return [], [], 0.0, fz; }
    let fp = form_roots_check(pr, pi_, tag);
    if fp.kind != F_NONE { return [], [], 0.0, fp; }
    return rs_zip(zr, zi), rs_zip(pr, pi_), k, no_fault();
}

// lp2lp_zpk scales the zeros and the poles of an analog lowpass
// filter to the cut wo.
fn lp2lp_zpk(zr: f64[:], zi: f64[:], pr: f64[:], pi_: f64[:], k: f64, wo: f64) -> Arr, Arr, Arr, Arr, f64, Fault {
    let z, p, g, f = sig_zpk_in(zr, zi, pr, pi_, k, "lp2lp_zpk");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f; }
    let zz, pp, kk, f2 = sig_lp2lp_zpk(z, p, g, wo);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f2; }
    return sig_zpk_repack(zz, pp, kk, "lp2lp_zpk");
}

// lp2hp_zpk maps an analog lowpass filter to a highpass filter.
fn lp2hp_zpk(zr: f64[:], zi: f64[:], pr: f64[:], pi_: f64[:], k: f64, wo: f64) -> Arr, Arr, Arr, Arr, f64, Fault {
    let z, p, g, f = sig_zpk_in(zr, zi, pr, pi_, k, "lp2hp_zpk");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f; }
    let zz, pp, kk, f2 = sig_lp2hp_zpk(z, p, g, wo);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f2; }
    return sig_zpk_repack(zz, pp, kk, "lp2hp_zpk");
}

// lp2bp_zpk maps an analog lowpass filter to a band pass filter of
// the center wo and the width bw.
fn lp2bp_zpk(zr: f64[:], zi: f64[:], pr: f64[:], pi_: f64[:], k: f64, wo: f64, bw: f64) -> Arr, Arr, Arr, Arr, f64, Fault {
    let z, p, g, f = sig_zpk_in(zr, zi, pr, pi_, k, "lp2bp_zpk");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f; }
    let zz, pp, kk, f2 = sig_lp2bp_zpk(z, p, g, wo, bw);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f2; }
    return sig_zpk_repack(zz, pp, kk, "lp2bp_zpk");
}

// lp2bs_zpk maps an analog lowpass filter to a band stop filter of
// the center wo and the width bw.
fn lp2bs_zpk(zr: f64[:], zi: f64[:], pr: f64[:], pi_: f64[:], k: f64, wo: f64, bw: f64) -> Arr, Arr, Arr, Arr, f64, Fault {
    let z, p, g, f = sig_zpk_in(zr, zi, pr, pi_, k, "lp2bs_zpk");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f; }
    let zz, pp, kk, f2 = sig_lp2bs_zpk(z, p, g, wo, bw);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f2; }
    return sig_zpk_repack(zz, pp, kk, "lp2bs_zpk");
}

// bilinear_zpk maps an analog filter to the digital plane at the rate
// fs.
fn bilinear_zpk(zr: f64[:], zi: f64[:], pr: f64[:], pi_: f64[:], k: f64, fs: f64) -> Arr, Arr, Arr, Arr, f64, Fault {
    let z, p, g, f = sig_zpk_in(zr, zi, pr, pi_, k, "bilinear_zpk");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f; }
    let zz, pp, kk, f2 = sig_bilinear_zpk(z, p, g, fs);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f2; }
    return sig_zpk_repack(zz, pp, kk, "bilinear_zpk");
}

// normalize scales the coefficients so that the first coefficient of
// the denominator holds one. It drops the leading zeros of the
// denominator and the small leading coefficients of the numerator.
fn normalize(b: f64[:], a: f64[:]) -> Arr, Arr, Fault {
    var nb: f64[>..] = [];
    var na: f64[>..] = [];
    let f = form_normalize(b, a, "normalize", nb, na);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    return flt_arr(copy(nb)), flt_arr(copy(na)), no_fault();
}

// sig_poly_to_band gives the scaled coefficients of one polynomial of
// the lowpass to lowpass map.
fn sig_poly_to_band(c: f64[:], pwo: f64[:], start1: i64, start2: i64, out: f64[>..]&) {
    var i: i64 = 0;
    while i < c.len { out.push(c[i] * pwo[start1] / pwo[start2 + i]); i += 1; }
}

// lp2lp scales the cut of an analog lowpass filter to wo.
fn lp2lp(b: f64[:], a: f64[:], wo: f64) -> Arr, Arr, Fault {
    let d = a.len;
    let n = b.len;
    if d < 1 || n < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "lp2lp", "a coefficient list holds no element");
    }
    var m = d;
    if n > m { m = n; }
    var pwo: f64[>..] = [];
    var p: f64 = 1.0;
    var j: i64 = 0;
    while j < m { pwo.push(0.0); j += 1; }
    j = m - 1;
    while j >= 0 { pwo[j] = p; p = p * wo; j -= 1; }
    var s1: i64 = 0;
    if n > d { s1 = n - d; }
    var s2: i64 = 0;
    if d > n { s2 = d - n; }
    var nb: f64[>..] = [];
    var na: f64[>..] = [];
    sig_poly_to_band(b, pwo, s1, s2, nb);
    sig_poly_to_band(a, pwo, s1, s1, na);
    return normalize(nb, na);
}

// lp2hp maps an analog lowpass filter to a highpass filter.
fn lp2hp(b: f64[:], a: f64[:], wo: f64) -> Arr, Arr, Fault {
    let d = a.len;
    let n = b.len;
    if d < 1 || n < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "lp2hp", "a coefficient list holds no element");
    }
    var m = d;
    if n > m { m = n; }
    var pwo: f64[>..] = [];
    var p: f64 = 1.0;
    var j: i64 = 0;
    while j < m { pwo.push(p); p = p * wo; j += 1; }
    var ob: f64[>..] = [];
    var oa: f64[>..] = [];
    if d >= n {
        j = 0;
        while j < d { oa.push(a[d - 1 - j] * pwo[j]); j += 1; }
        j = 0;
        while j < n { ob.push(b[n - 1 - j] * pwo[j]); j += 1; }
        j = n;
        while j < d { ob.push(0.0); j += 1; }
    } else {
        j = 0;
        while j < n { ob.push(b[n - 1 - j] * pwo[j]); j += 1; }
        j = 0;
        while j < d { oa.push(a[d - 1 - j] * pwo[j]); j += 1; }
        j = d;
        while j < n { oa.push(0.0); j += 1; }
    }
    return normalize(ob, oa);
}

// sig_band_poly gives the coefficients of one polynomial of the map
// to a band. The band holds true for the band stop and false for the
// band pass.
fn sig_band_poly(c: f64[:], deg: i64, ma: i64, wo2: f64, bw: f64, band: bool, out: f64[>..]&) {
    var np = deg + ma;
    if band { np = 2 * ma; }
    var i: i64 = 0;
    while i <= np { out.push(0.0); i += 1; }
    var j: i64 = 0;
    while j <= np {
        var val: f64 = 0.0;
        var i: i64 = 0;
        while i <= deg {
            var top = i;
            if band { top = ma - i; }
            var k: i64 = 0;
            while k <= top {
                var hit = false;
                var w: f64 = 0.0;
                if band {
                    // The band stop: the powers sit with ma - i.
                    if i + 2 * k == j {
                        hit = true;
                        w = flt_binom(ma - i, k) * c[deg - i] * mathshim::power(wo2, (ma - i - k) * 1.0) * mathshim::power(bw, i * 1.0);
                    }
                } else {
                    // The band pass: the powers sit with i.
                    if ma - i + 2 * k == j {
                        hit = true;
                        w = flt_binom(i, k) * c[deg - i] * mathshim::power(wo2, (i - k) * 1.0) / mathshim::power(bw, i * 1.0);
                    }
                }
                if hit { val += w; }
                k += 1;
            }
            i += 1;
        }
        out[np - j] = val;
        j += 1;
    }
}

// lp2bp maps an analog lowpass filter to a band pass filter of the
// center wo and the width bw.
fn lp2bp(b: f64[:], a: f64[:], wo: f64, bw: f64) -> Arr, Arr, Fault {
    if a.len < 1 || b.len < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "lp2bp", "a coefficient list holds no element");
    }
    if !(bw > 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "lp2bp", "the width holds a value of zero or below");
    }
    let d = a.len - 1;
    let n = b.len - 1;
    var ma = n;
    if d > ma { ma = d; }
    var nb: f64[>..] = [];
    var na: f64[>..] = [];
    sig_band_poly(b, n, ma, wo * wo, bw, false, nb);
    sig_band_poly(a, d, ma, wo * wo, bw, false, na);
    return normalize(nb, na);
}

// lp2bs maps an analog lowpass filter to a band stop filter of the
// center wo and the width bw.
fn lp2bs(b: f64[:], a: f64[:], wo: f64, bw: f64) -> Arr, Arr, Fault {
    if a.len < 1 || b.len < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "lp2bs", "a coefficient list holds no element");
    }
    if !(bw > 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "lp2bs", "the width holds a value of zero or below");
    }
    let d = a.len - 1;
    let n = b.len - 1;
    var ma = n;
    if d > ma { ma = d; }
    var nb: f64[>..] = [];
    var na: f64[>..] = [];
    sig_band_poly(b, n, ma, wo * wo, bw, true, nb);
    sig_band_poly(a, d, ma, wo * wo, bw, true, na);
    return normalize(nb, na);
}

// sig_bilinear_poly maps one analog polynomial to the digital plane.
// The list holds the coefficients from the highest power. The list
// sits in a common degree of n.
fn sig_bilinear_poly(c: f64[:], n: i64, fs: f64, out: f64[>..]&) {
    var i: i64 = 0;
    let size = n + 1;
    while i < size { out.push(0.0); i += 1; }
    i = 0;
    while i < c.len {
        // The coefficient of s to the power i.
        let q = c.len - 1 - i;
        if q < 0 { break; }
        let scale = c[q] * mathshim::power(2.0 * fs, i * 1.0);
        if scale == 0.0 { i += 1; continue; }
        var u: f64[>..] = [];
        var v: f64[>..] = [];
        var t: f64[>..] = [];
        flt_shift_pow(i, -1.0, u);
        flt_shift_pow(n - i, 1.0, v);
        flt_pmul(u, v, t);
        var j: i64 = 0;
        while j < t.len { out[j] = out[j] + scale * t[j]; j += 1; }
        i += 1;
    }
}

// bilinear maps an analog filter to the digital plane with the rule
// s = 2 fs (z - 1) / (z + 1). The rate fs holds hertz.
fn bilinear(b: f64[:], a: f64[:], fs: f64 = 1.0) -> Arr, Arr, Fault {
    if a.len < 1 || b.len < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "bilinear", "a coefficient list holds no element");
    }
    if !(fs > 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "bilinear", "the rate holds a value of zero or below");
    }
    // The leading zeros go, as in SciPy.
    var i: i64 = 0;
    while i < b.len - 1 && b[i] == 0.0 { i += 1; }
    var cb: f64[>..] = [];
    var j = i;
    while j < b.len { cb.push(b[j]); j += 1; }
    i = 0;
    while i < a.len - 1 && a[i] == 0.0 { i += 1; }
    var ca: f64[>..] = [];
    j = i;
    while j < a.len { ca.push(a[j]); j += 1; }
    var n = cb.len - 1;
    if ca.len - 1 > n { n = ca.len - 1; }
    var nb: f64[>..] = [];
    var na: f64[>..] = [];
    sig_bilinear_poly(cb, n, fs, nb);
    sig_bilinear_poly(ca, n, fs, na);
    return normalize(nb, na);
}
