// The forms of a cascade of second order sections.
//
// A cascade of sections holds one row of six coefficients for each
// section: three of the numerator and three of the denominator, from
// the highest power of z to the lowest. The calls of this file turn a
// cascade into one transfer function or into the zeros, the poles and
// the gain of the whole filter, as the calls of scipy.signal do:
// sos2tf multiplies the polynomials of the sections and sos2zpk joins
// the roots of the sections and multiplies the gains.
namespace scientific;

import mathshim;

// The largest count of sections of a cascade. The product of the
// polynomials of the sections stays inside the limit of the
// coefficients of a form.
const SOS_SECTIONS_MAX: i64 = 64;

// sos_sections_check holds the rules of a cascade: two dimensions, six
// coefficients per row, at least one row, and a buffer that agrees
// with the shape.
fn sos_sections_check(sos, tag: const u8[:]) -> i64, Fault {
    if sos.ndim != 2 || sos.shape.len != 2 || sos.shape[1] != 6 || sos.shape[0] < 1 {
        return 0, fault(F_SHAPE, tag, "the sections must hold one row of six coefficients each");
    }
    let n = sos.shape[0];
    if n > SOS_SECTIONS_MAX {
        return 0, fault(F_LIMIT, tag, "the cascade holds more sections than the limit");
    }
    let fs = check_stack(sos.ndim, sos.shape, sos.data.len, 1, tag);
    if fs.kind != F_NONE { return 0, fs; }
    if !all_finite(sos.data) {
        return 0, fault(F_VALUE, tag, "a coefficient is not finite");
    }
    return n, no_fault();
}

// sostf_mul multiplies two polynomials. The coefficients come from the
// highest power to the lowest.
fn sostf_mul(x: f64[:], y: f64[:], out: f64[>..]&) -> Fault {
    flt_pmul(x, y, out);
    if !all_finite(out) {
        return fault(F_VALUE, "sos2tf", "a coefficient of the product overflows");
    }
    return no_fault();
}

// sos2tf gives the numerator and the denominator of a cascade of
// second order sections. The call multiplies the numerators of the
// sections and the denominators of the sections. The coefficients come
// from the highest power to the lowest.
fn sos2tf(sos) -> Arr, Arr, Fault {
    let n, f = sos_sections_check(sos, "sos2tf");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    var b: f64[>..] = [1.0];
    var aa: f64[>..] = [1.0];
    var s: i64 = 0;
    while s < n {
        var rowb: f64[>..] = [];
        var rowa: f64[>..] = [];
        var j: i64 = 0;
        while j < 3 {
            rowb.push(sos.data[s * 6 + j]);
            rowa.push(sos.data[s * 6 + 3 + j]);
            j += 1;
        }
        var nb: f64[>..] = [];
        var na: f64[>..] = [];
        let fb = sostf_mul(b, rowb, nb);
        if fb.kind != F_NONE { return empty_arr(), empty_arr(), fb; }
        let fa = sostf_mul(aa, rowa, na);
        if fa.kind != F_NONE { return empty_arr(), empty_arr(), fa; }
        b = copy(nb);
        aa = copy(na);
        s += 1;
    }
    return flt_arr(copy(b)), flt_arr(copy(aa)), no_fault();
}

// sos2zpk gives the zeros, the poles and the gain of a cascade of
// second order sections. Each section holds up to two zeros and two
// poles, so the results hold two roots for each section, even where a
// root sits at zero. The gain is the product of the gains of the
// sections. The order of the roots is the order of the package, not
// the order of SciPy.
fn sos2zpk(sos) -> Arr, Arr, Arr, Arr, f64, Fault {
    let n, f = sos_sections_check(sos, "sos2zpk");
    if f.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f;
    }
    var zr: f64[>..] = [];
    var zi: f64[>..] = [];
    var pr: f64[>..] = [];
    var pi_: f64[>..] = [];
    var i: i64 = 0;
    while i < 2 * n {
        zr.push(0.0);
        zi.push(0.0);
        pr.push(0.0);
        pi_.push(0.0);
        i += 1;
    }
    var k: f64 = 1.0;
    var s: i64 = 0;
    while s < n {
        var rowb: f64[>..] = [];
        var rowa: f64[>..] = [];
        var j: i64 = 0;
        while j < 3 {
            rowb.push(sos.data[s * 6 + j]);
            rowa.push(sos.data[s * 6 + 3 + j]);
            j += 1;
        }
        let zsr, zsi, psr, psi, ks, fs = tf2zpk(rowb, rowa);
        if fs.kind != F_NONE {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, fs;
        }
        i = 0;
        while i < zsr.data.len && i < 2 {
            zr[2 * s + i] = zsr.data[i];
            zi[2 * s + i] = zsi.data[i];
            i += 1;
        }
        i = 0;
        while i < psr.data.len && i < 2 {
            pr[2 * s + i] = psr.data[i];
            pi_[2 * s + i] = psi.data[i];
            i += 1;
        }
        k *= ks;
        if !all_finite([k]) {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0,
                fault(F_VALUE, "sos2zpk", "the product of the gains overflows");
        }
        s += 1;
    }
    return flt_arr(copy(zr)), flt_arr(copy(zi)), flt_arr(copy(pr)),
        flt_arr(copy(pi_)), k, no_fault();
}
