// The search of the line by the rule of the strong Wolfe conditions, as
// the call line_search of SciPy computes it. SciPy uses the algorithm
// of Wright and Nocedal, so the step of the package matches the step of
// SciPy.
//
// The rule takes a point, a direction, the value of the function at the
// point and the slope of the function in the direction. It gives a step
// that holds the fall of Armijo and the fall of the slope. The first
// stage doubles the step until the value rises or the slope turns. The
// second stage, the zoom, narrows an interval that holds the step. The
// zoom uses a cubic and a quadratic fit of the function, and it falls
// back to the half of the interval.
//
// The tolerance of the fall of Armijo is one in ten thousand, and the
// tolerance of the fall of the slope is the value c2 of the caller.
namespace scientific;

import mathshim;

// LsZoom holds the result of the zoom of the search.
struct LsZoom {
    a: f64, ok: bool,
}

// ls_quadmin gives the place of the smallest value of the quadratic
// that holds the values fa and fb at the places a and b, and the slope
// fpa at the place a. The flag tells whether the place holds a number.
fn ls_quadmin(a: f64, fa: f64, fpa: f64, b: f64, fb: f64) -> f64, bool {
    let d = fa;
    let c = fpa;
    let db = b - a;
    if db == 0.0 { return 0.0, false; }
    let q = (fb - d - c * db) / (db * db);
    if q == 0.0 { return 0.0, false; }
    let x = a - c / (2.0 * q);
    if !(x - x == 0.0) { return 0.0, false; }
    return x, true;
}

// ls_cubicmin gives the place of the smallest value of the cubic that
// holds the values fa, fb and fc at the places a, b and c, and the
// slope fpa at the place a. The flag tells whether the place holds a
// number.
fn ls_cubicmin(a: f64, fa: f64, fpa: f64, b: f64, fb: f64, c: f64, fc: f64) -> f64, bool {
    let cc = fpa;
    let db = b - a;
    let dc = c - a;
    if db == 0.0 || dc == 0.0 || db == dc { return 0.0, false; }
    let db2 = db * db;
    let dc2 = dc * dc;
    let den = db2 * dc2 * (db - dc);
    if den == 0.0 { return 0.0, false; }
    let v1 = fb - fa - cc * db;
    let v2 = fc - fa - cc * dc;
    let aa = (dc2 * v1 - db2 * v2) / den;
    let bb = (-dc2 * dc * v1 + db2 * db * v2) / den;
    let radical = bb * bb - 3.0 * aa * cc;
    if radical < 0.0 { return 0.0, false; }
    if aa == 0.0 { return 0.0, false; }
    let x = a + (-bb + mathshim::root(radical)) / (3.0 * aa);
    if !(x - x == 0.0) { return 0.0, false; }
    return x, true;
}

// ls_zoom narrows the interval that holds the step of the search. The
// ends of the interval hold the values phi_lo and phi_hi, and the slope
// at the lower end holds derphi_lo. The call gives the step that holds
// the two conditions of Wolfe.
fn ls_zoom<F: fn(f64[:]) -> f64[:]>(x: f64[:], p: f64[:], a_lo_in: f64, a_hi_in: f64, phi_lo_in: f64, phi_hi_in: f64,
                                    derphi_lo_in: f64, phi0: f64, derphi0: f64,
                                    c1: f64, c2: f64, n: i64) -> LsZoom {
    var a_lo = a_lo_in;
    var a_hi = a_hi_in;
    var phi_lo = phi_lo_in;
    var phi_hi = phi_hi_in;
    var derphi_lo = derphi_lo_in;
    var phi_rec = phi0;
    var a_rec: f64 = 0.0;
    var i: i64 = 0;
    while true {
        let dalpha = a_hi - a_lo;
        var a: f64 = a_lo;
        var b: f64 = a_hi;
        if dalpha < 0.0 {
            a = a_hi;
            b = a_lo;
        }
        // The cubic fit of the function, and the quadratic fit when the
        // cubic lies too near an end of the interval.
        var have_j = false;
        var a_j: f64 = 0.0;
        if i > 0 {
            let cchk = 0.2 * dalpha;
            let j, ok = ls_cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec);
            if ok {
                have_j = true;
                a_j = j;
                if a_j > b - cchk || a_j < a + cchk { have_j = false; }
            }
        }
        if !have_j {
            let qchk = 0.1 * dalpha;
            let j, ok = ls_quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi);
            if ok && !(j > b - qchk || j < a + qchk) {
                have_j = true;
                a_j = j;
            }
        }
        if !have_j { a_j = a_lo + 0.5 * dalpha; }
        let phi_aj = ls_value(x, p, a_j, n) { t => F(copy(t)) };
        if phi_aj > phi0 + c1 * a_j * derphi0 || phi_aj >= phi_lo {
            phi_rec = phi_hi;
            a_rec = a_hi;
            a_hi = a_j;
            phi_hi = phi_aj;
        } else {
            let derphi_aj = ls_slope(x, p, a_j, n) { t => F(copy(t)) };
            if mathshim::abs_of(derphi_aj) <= -c2 * derphi0 {
                return LsZoom { a: a_j, ok: true };
            }
            if derphi_aj * (a_hi - a_lo) >= 0.0 {
                phi_rec = phi_hi;
                a_rec = a_hi;
                a_hi = a_lo;
                phi_hi = phi_lo;
            } else {
                phi_rec = phi_lo;
                a_rec = a_lo;
            }
            a_lo = a_j;
            phi_lo = phi_aj;
            derphi_lo = derphi_aj;
        }
        i += 1;
        if i > 10 {
            return LsZoom { a: 0.0, ok: false };
        }
    }
    return LsZoom { a: 0.0, ok: false };
}

// ls_value gives the value of the function of the caller at the step a
// along the direction of the search. The direction and the point of the
// search live in the state of the search.
fn ls_value<F: fn(f64[:]) -> f64[:]>(x: f64[:], p: f64[:], a: f64, n: i64) -> f64 {
    var trial: f64[>..] = [];
    var i: i64 = 0;
    while i < n { trial.push(x[i] + a * p[i]); i += 1; }
    let r = F(copy(trial));
    return r[0];
}

// ls_slope gives the slope of the function of the caller in the
// direction of the search at the step a.
fn ls_slope<F: fn(f64[:]) -> f64[:]>(x: f64[:], p: f64[:], a: f64, n: i64) -> f64 {
    var trial: f64[>..] = [];
    var i: i64 = 0;
    while i < n { trial.push(x[i] + a * p[i]); i += 1; }
    let r = F(copy(trial));
    var d: f64 = 0.0;
    i = 1;
    while i <= n {
        d += r[i] * p[i - 1];
        i += 1;
    }
    return d;
}

// ls_wolfe2 gives the step of the search of the line. The function of
// the caller gives the value and the gradient of the function at a
// point. The result holds the step and the flag that says whether the
// search found a step that holds the two conditions of Wolfe.
fn ls_wolfe2<F: fn(f64[:]) -> f64[:]>(x: f64[:], p: f64[:], f0: f64, g0: f64,
                                      c1: f64, c2: f64, n: i64) -> f64, bool {
    let phi0 = f0;
    let derphi0 = g0;
    var alpha0: f64 = 0.0;
    var alpha1: f64 = 1.0;
    var phi_a1 = ls_value(x, p, alpha1, n) { t => F(copy(t)) };
    var phi_a0 = phi0;
    var derphi_a0 = derphi0;
    var i: i64 = 0;
    while i < 10 {
        let first = i == 0;
        if phi_a1 > phi0 + c1 * alpha1 * derphi0 || (!first && phi_a1 >= phi_a0) {
            let z = ls_zoom(x, p, alpha0, alpha1, phi_a0, phi_a1, derphi_a0,
                               phi0, derphi0, c1, c2, n) { t => F(copy(t)) };
            return z.a, z.ok;
        }
        let derphi_a1 = ls_slope(x, p, alpha1, n) { t => F(copy(t)) };
        if mathshim::abs_of(derphi_a1) <= -c2 * derphi0 {
            return alpha1, true;
        }
        if derphi_a1 >= 0.0 {
            let z = ls_zoom(x, p, alpha1, alpha0, phi_a1, phi_a0, derphi_a1,
                               phi0, derphi0, c1, c2, n) { t => F(copy(t)) };
            return z.a, z.ok;
        }
        // The step grows by the factor of two.
        let alpha2 = 2.0 * alpha1;
        alpha0 = alpha1;
        alpha1 = alpha2;
        phi_a0 = phi_a1;
        phi_a1 = ls_value(x, p, alpha1, n) { t => F(copy(t)) };
        derphi_a0 = derphi_a1;
        i += 1;
    }
    return alpha1, false;
}
