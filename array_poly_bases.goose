// The orthogonal polynomial bases: the checks, the recurrence rule, the
// value of a series and the two conversions to and from the monomials.
//
// The five families are Legendre, Chebyshev, Hermite, HermiteE and
// Laguerre. A series holds its coefficients from the lowest degree to
// the highest, as numpy.polynomial holds them. This differs from the
// monomial calls of this package, which run from the highest power.
namespace scientific;

import mathshim;

const PB_LEG: i64 = 0;
const PB_CHEB: i64 = 1;
const PB_HERM: i64 = 2;
const PB_HERME: i64 = 3;
const PB_LAG: i64 = 4;

// A series holds at most PB_MAX_LEN coefficients. The order of a
// derivative or an integral stays at or below PB_MAX_ORDER.
const PB_MAX_LEN: i64 = 1001;
const PB_MAX_ORDER: i64 = 1000;

// pb_finite tells whether a value is neither NaN nor infinite.
fn pb_finite(x: f64) -> bool { return x - x == 0.0; }

// pb_rule gives (a, b, g) so that B(k+1) = (a x + b) B(k) + g B(k-1).
fn pb_rule(fam: i64, k: i64) -> f64, f64, f64 {
    let kf = k as f64;
    if fam == PB_LEG { return (2.0 * kf + 1.0) / (kf + 1.0), 0.0, -kf / (kf + 1.0); }
    if fam == PB_CHEB { return 2.0, 0.0, -1.0; }
    if fam == PB_HERM { return 2.0, 0.0, -2.0 * kf; }
    if fam == PB_HERME { return 1.0, 0.0, -kf; }
    return -1.0 / (kf + 1.0), (2.0 * kf + 1.0) / (kf + 1.0), -kf / (kf + 1.0);
}

// pb_first gives B(1)(x) of a family.
fn pb_first(fam: i64, x: f64) -> f64 {
    if fam == PB_HERM { return 2.0 * x; }
    if fam == PB_LAG { return 1.0 - x; }
    return x;
}

// pb_check_series checks the length and the values of a series.
fn pb_check_series(c: f64[:], tag: const u8[:]) -> Fault {
    if c.len < 1 { return fault(F_SHAPE, tag, "the series holds no coefficient"); }
    if c.len > PB_MAX_LEN { return fault(F_LIMIT, tag, "the series holds more than 1001 coefficients"); }
    var i: i64 = 0;
    while i < c.len {
        if !pb_finite(c[i]) { return fault(F_VALUE, tag, "a coefficient is not finite"); }
        i += 1;
    }
    return no_fault();
}

// pb_check_order checks the order m of a derivative or an integral.
fn pb_check_order(m: i64, tag: const u8[:]) -> Fault {
    if m < 0 { return fault(F_RANGE, tag, "the order is negative"); }
    if m > PB_MAX_ORDER { return fault(F_LIMIT, tag, "the order is above 1000"); }
    return no_fault();
}

// pb_check_scale checks the scale of a derivative or an integral.
fn pb_check_scale(scl: f64, tag: const u8[:]) -> Fault {
    if !pb_finite(scl) { return fault(F_VALUE, tag, "the scale is not finite"); }
    if scl == 0.0 { return fault(F_VALUE, tag, "the scale is zero"); }
    return no_fault();
}

// pb_clenshaw gives the value of a series at x. The series is not
// checked.
fn pb_clenshaw(fam: i64, x: f64, c: f64[:]) -> f64 {
    let n = c.len - 1;
    if n == 0 { return c[0]; }
    var c0 = c[n - 1];
    var c1 = c[n];
    var j = n - 1;
    while j >= 1 {
        let a, b, g = pb_rule(fam, j);
        let t = c0;
        c0 = c[j - 1] + g * c1;
        c1 = t + (a * x + b) * c1;
        j -= 1;
    }
    return c0 + c1 * pb_first(fam, x);
}

// pb_val checks its input and gives the value of a series at x.
fn pb_val(fam: i64, x: f64, c: f64[:], tag: const u8[:]) -> f64, Fault {
    let f = pb_check_series(c, tag);
    if f.kind != F_NONE { return 0.0, f; }
    if !pb_finite(x) { return 0.0, fault(F_VALUE, tag, "the point is not finite"); }
    let v = pb_clenshaw(fam, x, c);
    if !pb_finite(v) { return 0.0, fault(F_VALUE, tag, "the value is not finite"); }
    return v, no_fault();
}

// pb_zeros gives a buffer of n zeros.
fn pb_zeros(n: i64) -> f64[>..] {
    var z: f64[>..] = [];
    var i: i64 = 0;
    while i < n { z.push(0.0); i += 1; }
    return z;
}

// pb_all_finite tells whether every value of a buffer is finite.
fn pb_all_finite(v: f64[:]) -> bool {
    var i: i64 = 0;
    while i < v.len {
        if !pb_finite(v[i]) { return false; }
        i += 1;
    }
    return true;
}

// pb_to_mono gives the monomial coefficients of a series, lowest degree
// first. The series holds no trailing zero. The method adds the basis
// polynomials, which a rolling pair of rows builds from the rule.
fn pb_to_mono(fam: i64, c: f64[:]) -> f64[>..] {
    let n = c.len;
    var res = pb_zeros(n);
    res[0] = c[0];
    if n == 1 { return res; }
    var prev = pb_zeros(n);
    var cur = pb_zeros(n);
    prev[0] = 1.0;
    if fam == PB_HERM { cur[1] = 2.0; }
    else if fam == PB_LAG { cur[0] = 1.0; cur[1] = -1.0; }
    else { cur[1] = 1.0; }
    var i: i64 = 0;
    while i < 2 { res[i] += c[1] * cur[i]; i += 1; }
    var k: i64 = 1;
    while k < n - 1 {
        let a, b, g = pb_rule(fam, k);
        // Update both rows in place from the top place down: cur[i - 1] still
        // holds its old value when place i reads it.
        i = k + 1;
        while i >= 0 {
            let old = cur[i];
            var v = b * old + g * prev[i];
            if i > 0 { v += a * cur[i - 1]; }
            cur[i] = v;
            prev[i] = old;
            i -= 1;
        }
        i = 0;
        while i <= k + 1 { res[i] += c[k + 1] * cur[i]; i += 1; }
        k += 1;
    }
    return res;
}

// pb_mulx gives the series of x times a series, of one more place.
fn pb_mulx(fam: i64, a: f64[:]) -> f64[>..] {
    var out = pb_zeros(a.len + 1);
    var k: i64 = 0;
    while k < a.len {
        if k == 0 {
            if fam == PB_HERM { out[1] += 0.5 * a[0]; }
            else if fam == PB_LAG { out[0] += a[0]; out[1] -= a[0]; }
            else { out[1] += a[0]; }
        } else {
            let r, b, g = pb_rule(fam, k);
            out[k + 1] += a[k] / r;
            out[k] -= b * a[k] / r;
            out[k - 1] -= g * a[k] / r;
        }
        k += 1;
    }
    return out;
}

// pb_from_mono gives the series of a monomial polynomial, lowest degree
// first. The polynomial holds no trailing zero. The method is the
// scheme of Horner with the product by x of the family.
fn pb_from_mono(fam: i64, p: f64[:]) -> f64[>..] {
    var res = pb_zeros(1);
    var j = p.len - 1;
    res[0] = p[j];
    j -= 1;
    while j >= 0 {
        var nx = pb_mulx(fam, res);
        nx[0] += p[j];
        res = copy(nx);
        j -= 1;
    }
    return res;
}

// pb_convert checks a series and converts it. The direction is to the
// monomials when to_mono is true.
fn pb_convert(fam: i64, c: f64[:], to_mono: bool, tag: const u8[:]) -> Arr, Fault {
    let f = pb_check_series(c, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let n = poly_trim(c);
    var cut: f64[>..] = [];
    var i: i64 = 0;
    while i < n { cut.push(c[i]); i += 1; }
    var out: f64[>..] = [];
    if to_mono { out = pb_to_mono(fam, cut); } else { out = pb_from_mono(fam, cut); }
    if !pb_all_finite(out) { return empty_arr(), fault(F_VALUE, tag, "a coefficient of the result is not finite"); }
    return line_arr(copy(out)), no_fault();
}
