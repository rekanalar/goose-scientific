// The complex numbers.
//
// The package holds real numbers only. A complex number therefore
// lives in two numbers: the real part and the imaginary part. The
// type Cplx holds the two. The functions of this file serve the
// roots of a polynomial, which may hold a complex number.
namespace scientific;

import mathshim;

// The largest degree that roots takes. The cost of the method grows
// with the square of the degree, so a larger degree could stall a
// caller.
const MAX_ROOTS_DEGREE: i64 = 256;

// Cplx holds a complex number. The real part sits in re and the
// imaginary part in im.
struct Cplx { re: f64, im: f64 }

// c_add adds two complex numbers.
fn c_add(a: Cplx, b: Cplx) -> Cplx {
    return Cplx { re: a.re + b.re, im: a.im + b.im };
}

// c_sub subtracts the second complex number from the first.
fn c_sub(a: Cplx, b: Cplx) -> Cplx {
    return Cplx { re: a.re - b.re, im: a.im - b.im };
}

// c_mul multiplies two complex numbers.
fn c_mul(a: Cplx, b: Cplx) -> Cplx {
    return Cplx { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re };
}

// c_div divides the first complex number by the second. A divisor of
// zero gives the missing value.
fn c_div(a: Cplx, b: Cplx) -> Cplx {
    let d = b.re * b.re + b.im * b.im;
    return Cplx { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
}

// c_abs gives the size of a complex number.
fn c_abs(a: Cplx) -> f64 {
    return mathshim::root(a.re * a.re + a.im * a.im);
}

// c_conj gives the conjugate of a complex number.
fn c_conj(a: Cplx) -> Cplx {
    return Cplx { re: a.re, im: -a.im };
}

// c_scale multiplies a complex number by a real number.
fn c_scale(a: Cplx, k: f64) -> Cplx {
    return Cplx { re: a.re * k, im: a.im * k };
}

// c_sqrt gives the principal square root of a complex number. The
// root of a negative real number holds a positive imaginary part.
fn c_sqrt(a: Cplx) -> Cplx {
    let m = c_abs(a);
    if a.re >= 0.0 {
        let t = mathshim::root((m + a.re) / 2.0);
        if t == 0.0 { return Cplx { re: 0.0, im: 0.0 }; }
        return Cplx { re: t, im: a.im / (2.0 * t) };
    }
    var t = mathshim::root((m - a.re) / 2.0);
    if a.im < 0.0 { t = -t; }
    return Cplx { re: a.im / (2.0 * t), im: t };
}

// c_value gives the value of a polynomial with real coefficients at a
// complex point. The method is the rule of Horner.
fn c_value(c: f64[:], z: Cplx) -> Cplx {
    var t = Cplx { re: c[0], im: 0.0 };
    var i: i64 = 1;
    while i < c.len {
        t = c_mul(t, z);
        t = c_add(t, Cplx { re: c[i], im: 0.0 });
        i += 1;
    }
    return t;
}

// roots gives the roots of a polynomial. The coefficients come from
// the highest power to the lowest. The result holds the real parts
// and the imaginary parts in two arrays, as NumPy holds the roots in
// one array of complex numbers. The roots come in no fixed order.
//
// The method is the rule of Durand and Kerner. The rule starts with
// points on a circle and moves every point to the root that sits
// near it. The method finds every root, real or complex.
fn roots(c: f64[:]) -> Arr, Arr, Fault {
    // The leading coefficient must not be zero.
    var i: i64 = 0;
    while i < c.len {
        if c[i] != 0.0 { break; }
        i += 1;
    }
    if i >= c.len {
        return empty_arr(), empty_arr(), fault(F_VALUE, "roots", "the polynomial holds no term");
    }
    // The polynomial without its leading zeros.
    var p: f64[>..] = [];
    var j = i;
    while j < c.len { p.push(c[j]); j += 1; }
    let n = p.len - 1;
    if n > MAX_ROOTS_DEGREE {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "roots", "the degree holds a value above the limit of the roots");
    }
    var s: i64[..8] = [];
    s.push(n);
    if n == 0 {
        // The polynomial holds no root.
        return Arr { ndim: 1, shape: s, data: [] }, Arr { ndim: 1, shape: s, data: [] }, no_fault();
    }
    let lead = p[0];
    // The circle of the start holds the largest ratio of the
    // coefficients.
    var radius: f64 = 1.0;
    j = 1;
    while j < p.len {
        let v = mathshim::abs_of(p[j] / lead);
        if v > radius { radius = v; }
        j += 1;
    }
    radius = radius + 1.0;
    var z: Cplx[>..] = [];
    j = 0;
    while j < n {
        let angle = 6.283185307179586 * j / n + 0.4;
        z.push(Cplx { re: radius * mathshim::cosine(angle), im: radius * mathshim::sine(angle) });
        j += 1;
    }
    // The turns move every point toward its root.
    var turn: i64 = 0;
    while turn < 500 {
        var moved: f64 = 0.0;
        j = 0;
        while j < n {
            let value = c_value(p, z[j]);
            var den = Cplx { re: lead, im: 0.0 };
            var k: i64 = 0;
            while k < n {
                if k != j { den = c_mul(den, c_sub(z[j], z[k])); }
                k += 1;
            }
            let step = c_div(value, den);
            z[j] = c_sub(z[j], step);
            let size = c_abs(step);
            if size > moved { moved = size; }
            j += 1;
        }
        if moved < 0.0000000000000001 { break; }
        turn += 1;
    }
    var re: f64[>..] = [];
    var im: f64[>..] = [];
    j = 0;
    while j < n {
        re.push(z[j].re);
        im.push(z[j].im);
        j += 1;
    }
    return Arr { ndim: 1, shape: s, data: re }, Arr { ndim: 1, shape: s, data: im }, no_fault();
}

// roots_of gives the roots of a polynomial as a list of complex
// numbers. The coefficients come from the highest power to the
// lowest.
fn roots_of(c: f64[:]) -> Cplx[>..], Fault {
    let re, im, f = roots(c);
    if f.kind != F_NONE { return [], f; }
    var out: Cplx[>..] = [];
    var i: i64 = 0;
    while i < re.data.len {
        out.push(Cplx { re: re.data[i], im: im.data[i] });
        i += 1;
    }
    return out, no_fault();
}
