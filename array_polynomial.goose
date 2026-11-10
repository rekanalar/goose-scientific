// The polynomials as a type.
//
// The type Polynomial holds the coefficients of a polynomial from the
// lowest power to the highest, as the class of NumPy holds them. The
// functions of this file carry the name of the type, because the
// namespace of the package is global. The functions of
// array_poly.goose hold the names of NumPy and take the coefficients
// from the highest power.
namespace scientific;

import mathshim;

// Polynomial holds the coefficients from the lowest power to the
// highest. The coefficient of the lowest power sits at the place 0.
struct Polynomial { coe: f64[>..] }

// polynomial builds a polynomial from the coefficients. The
// coefficients run from the lowest power to the highest. Trailing
// coefficients of zero drop, as the call of NumPy drops them.
fn polynomial(c: f64[:]) -> Polynomial {
    var out: f64[>..] = [];
    let n = poly_trim(c);
    var i: i64 = 0;
    while i < n { out.push(c[i]); i += 1; }
    return Polynomial { coe: copy(out) };
}

// poly_trim gives the length after the trailing zeros drop. One
// coefficient stays, so the zero polynomial holds one coefficient.
fn poly_trim(c: f64[:]) -> i64 {
    var n = c.len;
    while n > 1 && c[n - 1] == 0.0 { n -= 1; }
    return n;
}

// poly_low turns the coefficients of the type into a buffer from the
// highest power to the lowest, for the functions of array_poly.
fn poly_low(c: f64[:], out: f64[>..]&) {
    var i = c.len - 1;
    while i >= 0 { out.push(c[i]); i -= 1; }
}

// poly_high turns a buffer from the highest power into the buffer of
// the type.
fn poly_high(c: f64[:], out: f64[>..]&) {
    var i = c.len - 1;
    while i >= 0 { out.push(c[i]); i -= 1; }
}

// poly_polish gives a polynomial from a buffer with the trailing
// zeros dropped.
fn poly_polish(c: f64[>..]) -> Polynomial {
    let n = poly_trim(c);
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n { out.push(c[i]); i += 1; }
    return Polynomial { coe: copy(out) };
}

// polynomial_call gives the value of the polynomial at a point. The
// method is the rule of Horner on the coefficients from the lowest
// power.
fn polynomial_call(p, x: f64) -> f64 {
    var t: f64 = 0.0;
    var i = p.coe.len - 1;
    while i >= 0 {
        t = t * x + p.coe[i];
        i -= 1;
    }
    return t;
}

// polynomial_degree gives the degree. The zero polynomial holds the
// degree of zero, as NumPy holds it.
fn polynomial_degree(p) -> i64 {
    return poly_trim(p.coe) - 1;
}

// poly_join adds or subtracts two buffers of coefficients and gives
// the longest length.
fn poly_join(a: f64[:], b: f64[:], sub: bool, out: f64[>..]&) {
    var n = a.len;
    if b.len > n { n = b.len; }
    var i: i64 = 0;
    while i < n {
        var v: f64 = 0.0;
        if i < a.len { v = a[i]; }
        if i < b.len { if sub { v = v - b[i]; } else { v = v + b[i]; } }
        out.push(v);
        i += 1;
    }
}

// polynomial_add adds two polynomials.
fn polynomial_add(p, q) -> Polynomial {
    var out: f64[>..] = [];
    poly_join(p.coe, q.coe, false, out);
    return poly_polish(copy(out));
}

// polynomial_sub subtracts the second polynomial from the first.
fn polynomial_sub(p, q) -> Polynomial {
    var out: f64[>..] = [];
    poly_join(p.coe, q.coe, true, out);
    return poly_polish(copy(out));
}

// polynomial_mul multiplies two polynomials.
fn polynomial_mul(p, q) -> Polynomial {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < p.coe.len + q.coe.len - 1 { out.push(0.0); i += 1; }
    i = 0;
    while i < p.coe.len {
        var j: i64 = 0;
        while j < q.coe.len {
            out[i + j] = out[i + j] + p.coe[i] * q.coe[j];
            j += 1;
        }
        i += 1;
    }
    return poly_polish(copy(out));
}

// polynomial_pow raises a polynomial to a whole power. The power
// holds a value below zero, or above 16, as a fault.
fn polynomial_pow(p, n: i64) -> Polynomial, Fault {
    if n < 0 || n > 16 {
        return Polynomial { coe: [] }, fault(F_RANGE, "polynomial_pow", "the power holds a value outside zero to 16");
    }
    var acc: f64[>..] = [1.0];
    var i: i64 = 0;
    while i < n {
        var t: f64[>..] = [];
        var j: i64 = 0;
        while j < acc.len + p.coe.len - 1 { t.push(0.0); j += 1; }
        j = 0;
        while j < acc.len {
            var k: i64 = 0;
            while k < p.coe.len {
                t[j + k] = t[j + k] + acc[j] * p.coe[k];
                k += 1;
            }
            j += 1;
        }
        acc = copy(t);
        i += 1;
    }
    return poly_polish(copy(acc)), no_fault();
}

// polynomial_divmod divides two polynomials and gives the quotient
// and the remainder. A polynomial of the degree below zero as the
// divisor is a fault.
fn polynomial_divmod(p, q) -> Polynomial, Polynomial, Fault {
    let dq = poly_trim(q.coe) - 1;
    if dq < 0 || (dq == 0 && q.coe[0] == 0.0) {
        return Polynomial { coe: [] }, Polynomial { coe: [] }, fault(F_VALUE, "polynomial_divmod", "the divisor holds no term");
    }
    // The division runs on the coefficients from the highest power.
    var a: f64[>..] = [];
    poly_low(p.coe, a);
    var b: f64[>..] = [];
    poly_low(q.coe, b);
    let da = a.len - 1;
    var quo: f64[>..] = [];
    var i: i64 = 0;
    while i <= da - dq { quo.push(0.0); i += 1; }
    i = 0;
    while i <= da - dq {
        let f = a[i] / b[0];
        quo[i] = f;
        var j: i64 = 0;
        while j <= dq {
            a[i + j] = a[i + j] - f * b[j];
            j += 1;
        }
        i += 1;
    }
    var rem: f64[>..] = [];
    i = 0;
    while i < dq + 1 { rem.push(a[da - dq + i]); i += 1; }
    var qo: f64[>..] = [];
    poly_high(quo, qo);
    var ro: f64[>..] = [];
    poly_high(rem, ro);
    return poly_polish(copy(qo)), poly_polish(copy(ro)), no_fault();
}

// polynomial_der gives the derivative. The count m holds the count of
// the turns. A count below one is a fault.
fn polynomial_der(p, m: i64) -> Polynomial, Fault {
    if m < 1 {
        return Polynomial { coe: [] }, fault(F_VALUE, "polynomial_der", "the count holds a value below one");
    }
    var c: f64[>..] = [];
    var i: i64 = 0;
    while i < p.coe.len { c.push(p.coe[i]); i += 1; }
    var turn: i64 = 0;
    while turn < m {
        var d: f64[>..] = [];
        i = 1;
        while i < c.len { d.push(c[i] * i); i += 1; }
        if d.len < 1 { d.push(0.0); }
        c = copy(d);
        turn += 1;
    }
    return poly_polish(copy(c)), no_fault();
}

// polynomial_integ gives the integral. The constant of the
// integration holds zero.
fn polynomial_integ(p) -> Polynomial {
    var out: f64[>..] = [0.0];
    var i: i64 = 0;
    while i < p.coe.len {
        out.push(p.coe[i] / (i + 1));
        i += 1;
    }
    return poly_polish(copy(out));
}

// polynomial_roots gives the roots of the polynomial. The method is
// the companion matrix of array_poly, so the roots of a polynomial
// with a double root hold few digits.
fn polynomial_roots(p) -> Arr, Arr, Fault {
    var c: f64[>..] = [];
    poly_low(p.coe, c);
    return roots(c);
}

// polynomial_fromroots gives the polynomial of the roots. The result
// holds the leading coefficient of one, as NumPy holds it.
fn polynomial_fromroots(r: f64[:]) -> Polynomial {
    var acc: f64[>..] = [1.0];
    var i: i64 = 0;
    while i < r.len {
        var t: f64[>..] = [];
        var j: i64 = 0;
        while j < acc.len + 2 - 1 { t.push(0.0); j += 1; }
        j = 0;
        while j < acc.len {
            t[j] = t[j] - r[i] * acc[j];
            t[j + 1] = t[j + 1] + acc[j];
            j += 1;
        }
        acc = copy(t);
        i += 1;
    }
    return poly_polish(copy(acc));
}

// polynomial_fit fits the points in the sense of the least squares.
// The polynomial holds the degree deg.
fn polynomial_fit(x: f64[:], y: f64[:], deg: i64) -> Polynomial, Fault {
    let c, f = polyfit(x, y, deg);
    if f.kind != F_NONE { return Polynomial { coe: [] }, f; }
    var out: f64[>..] = [];
    poly_high(c.data, out);
    return poly_polish(copy(out)), no_fault();
}

// polynomial_cutdeg drops the coefficients above the degree.
fn polynomial_cutdeg(p, deg: i64) -> Polynomial, Fault {
    if deg < 0 {
        return Polynomial { coe: [] }, fault(F_RANGE, "polynomial_cutdeg", "the degree holds a value below zero");
    }
    var out: f64[>..] = [];
    let n = poly_trim(p.coe);
    var i: i64 = 0;
    while i < n && i <= deg { out.push(p.coe[i]); i += 1; }
    if out.len < 1 { out.push(0.0); }
    return Polynomial { coe: copy(out) }, no_fault();
}

// polynomial_linspace gives count points of the range from lo to hi
// and the values of the polynomial at them. A count below two is a
// fault.
fn polynomial_linspace(p, lo: f64, hi: f64, count: i64) -> Arr, Arr, Fault {
    if count < 2 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "polynomial_linspace", "the count holds a value below two");
    }
    let step = (hi - lo) / (count - 1);
    var xs: f64[>..] = [];
    var ys: f64[>..] = [];
    var i: i64 = 0;
    while i < count {
        let v = lo + step * i;
        xs.push(v);
        ys.push(polynomial_call(p, v));
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(count);
    return Arr { ndim: 1, shape: s, data: xs }, Arr { ndim: 1, shape: s, data: ys }, no_fault();
}
