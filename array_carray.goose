// The complex array: its type, its elements and the arithmetic of complex numbers.
//
// The complex arrays.
//
// A complex array of the type CArr holds the real part and the
// imaginary part of each element in one buffer, side by side: the
// element i holds its real part at 2 i and its imaginary part at
// 2 i + 1. The shape and the order of the elements follow Arr. The
// complex arrays have their own functions, which carry the prefix c
// when a function of the same name exists for real arrays. The
// package builds no complex array from a real function: it converts
// with complex_array or complex_real.
namespace scientific;

import mathshim;

// CArr holds a complex array of the given shape.
struct CArr { ndim: i64, shape: i64[..8], data: f64[>..] }

fn carr_empty() -> CArr {
    var s: i64[..8] = [];
    s.push(0);
    return CArr { ndim: 1, shape: s, data: [] };
}

fn carr_make(shape: i64[:], data: f64[>..]) -> CArr {
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < shape.len { s.push(shape[i]); i += 1; }
    return CArr { ndim: shape.len, shape: s, data: data };
}

// complex_array builds a complex array from the real parts and the
// imaginary parts, which hold one shape.
fn complex_array(re, im) -> CArr, Fault {
    if re.ndim != im.ndim || re.data.len != im.data.len {
        return carr_empty(), fault(F_SHAPE, "complex_array", "the parts differ in shape");
    }
    var i: i64 = 0;
    while i < re.ndim {
        if re.shape[i] != im.shape[i] {
            return carr_empty(), fault(F_SHAPE, "complex_array", "the parts differ in shape");
        }
        i += 1;
    }
    var d: f64[>..] = [];
    i = 0;
    while i < re.data.len { d.push(re.data[i]); d.push(im.data[i]); i += 1; }
    return carr_make(re.shape, copy(d)), no_fault();
}

// complex_real builds a complex array whose imaginary parts are zero.
fn complex_real(re) -> CArr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < re.data.len { d.push(re.data[i]); d.push(0.0); i += 1; }
    return carr_make(re.shape, copy(d));
}

// complex_from_values builds a one dimensional complex array from two
// lists.
fn complex_from_values(re: f64[:], im: f64[:]) -> CArr, Fault {
    if re.len != im.len || re.len < 1 {
        return carr_empty(), fault(F_SHAPE, "complex_from_values", "the lists differ in length or hold no value");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < re.len { d.push(re[i]); d.push(im[i]); i += 1; }
    var s: i64[..8] = [];
    s.push(re.len);
    return carr_make(s, copy(d)), no_fault();
}

fn carr_size(c) -> i64 { return c.data.len / 2; }

fn carr_part(c, off: i64) -> Arr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < carr_size(c) { d.push(c.data[2 * i + off]); i += 1; }
    var s: i64[..8] = [];
    i = 0;
    while i < c.ndim { s.push(c.shape[i]); i += 1; }
    return Arr { ndim: c.ndim, shape: s, data: d };
}

// real gives the real parts of a complex array.
fn real(c) -> Arr { return carr_part(c, 0); }

// imag gives the imaginary parts of a complex array.
fn imag(c) -> Arr { return carr_part(c, 1); }

// cabs gives the size of each element.
fn cabs(c) -> Arr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < carr_size(c) { d.push(mathshim::hypot_of(c.data[2 * i], c.data[2 * i + 1])); i += 1; }
    var s: i64[..8] = [];
    i = 0;
    while i < c.ndim { s.push(c.shape[i]); i += 1; }
    return Arr { ndim: c.ndim, shape: s, data: d };
}

// angle gives the angle of each element, in radians. With deg true it
// gives degrees.
fn angle(c, deg: bool = false) -> Arr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < carr_size(c) {
        var v = mathshim::arctangent2_of(c.data[2 * i + 1], c.data[2 * i]);
        if deg { v = mathshim::rad2deg_of(v); }
        d.push(v);
        i += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < c.ndim { s.push(c.shape[i]); i += 1; }
    return Arr { ndim: c.ndim, shape: s, data: d };
}

// conj gives the conjugate of each element.
fn conj(c) -> CArr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < carr_size(c) { d.push(c.data[2 * i]); d.push(-c.data[2 * i + 1]); i += 1; }
    return carr_make(c.shape, copy(d));
}

// iscomplex gives one for each element whose imaginary part is not
// zero, and zero for the others.
fn iscomplex(c) -> Arr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < carr_size(c) {
        if c.data[2 * i + 1] != 0.0 { d.push(1.0); } else { d.push(0.0); }
        i += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < c.ndim { s.push(c.shape[i]); i += 1; }
    return Arr { ndim: c.ndim, shape: s, data: d };
}

// isreal gives one for each element whose imaginary part is zero.
fn isreal(c) -> Arr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < carr_size(c) {
        if c.data[2 * i + 1] == 0.0 { d.push(1.0); } else { d.push(0.0); }
        i += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < c.ndim { s.push(c.shape[i]); i += 1; }
    return Arr { ndim: c.ndim, shape: s, data: d };
}

// cbin applies a binary operation under the broadcast rule. The code
// is 0 for the sum, 1 for the difference, 2 for the product, 3 for the
// quotient and 4 for the power.
fn cbin(a, b, op: i64) -> CArr, Fault {
    let s, fs = bshape(a, b);
    if fs.kind != F_NONE { return carr_empty(), fs; }
    let total = numel(s);
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    bcast_indices(s, a.shape, b.shape, ia, ib);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < total {
        let x = Cplx { re: a.data[2 * ia[i]], im: a.data[2 * ia[i] + 1] };
        let y = Cplx { re: b.data[2 * ib[i]], im: b.data[2 * ib[i] + 1] };
        var r = Cplx { re: 0.0, im: 0.0 };
        if op == 0 { r = c_add(x, y); }
        else if op == 1 { r = c_sub(x, y); }
        else if op == 2 { r = c_mul(x, y); }
        else if op == 3 { r = c_div(x, y); }
        else { r = c_pow(x, y); }
        d.push(r.re);
        d.push(r.im);
        i += 1;
    }
    return carr_make(s, copy(d)), no_fault();
}

// c_exp gives e to the power of a complex number.
fn c_exp(a: Cplx) -> Cplx {
    let m = mathshim::power_e(a.re);
    return Cplx { re: m * mathshim::cosine(a.im), im: m * mathshim::sine(a.im) };
}

// c_log gives the principal logarithm of a complex number.
fn c_log(a: Cplx) -> Cplx {
    return Cplx { re: mathshim::ln(mathshim::hypot_of(a.re, a.im)), im: mathshim::arctangent2_of(a.im, a.re) };
}

// c_pow gives x to the power of y. Zero to the power of zero gives one,
// and zero to another power gives zero.
fn c_pow(x: Cplx, y: Cplx) -> Cplx {
    if x.re == 0.0 && x.im == 0.0 {
        if y.re == 0.0 && y.im == 0.0 { return Cplx { re: 1.0, im: 0.0 }; }
        return Cplx { re: 0.0, im: 0.0 };
    }
    return c_exp(c_mul(y, c_log(x)));
}

// cadd gives the sum of two complex arrays.
fn cadd(a, b) -> CArr, Fault { return cbin(a, b, 0); }

// csub gives the difference of two complex arrays.
fn csub(a, b) -> CArr, Fault { return cbin(a, b, 1); }

// cmul gives the product of two complex arrays, element by element.
fn cmul(a, b) -> CArr, Fault { return cbin(a, b, 2); }

// cdiv gives the quotient of two complex arrays, element by element.
fn cdiv(a, b) -> CArr, Fault { return cbin(a, b, 3); }

// cpower gives the power of two complex arrays, element by element.
fn cpower(a, b) -> CArr, Fault { return cbin(a, b, 4); }

// cun applies a unary function. The code is 0 for exp, 1 for log, 2
// for sqrt, 3 for sin, 4 for cos, 5 for sinh, 6 for cosh and 7 for
// the negative.
fn cun(a, op: i64) -> CArr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < carr_size(a) {
        let x = Cplx { re: a.data[2 * i], im: a.data[2 * i + 1] };
        var r = Cplx { re: 0.0, im: 0.0 };
        if op == 0 { r = c_exp(x); }
        else if op == 1 { r = c_log(x); }
        else if op == 2 { r = c_sqrt(x); }
        else if op == 3 { r = Cplx { re: mathshim::sine(x.re) * mathshim::cosh_of(x.im), im: mathshim::cosine(x.re) * mathshim::sinh_of(x.im) }; }
        else if op == 4 { r = Cplx { re: mathshim::cosine(x.re) * mathshim::cosh_of(x.im), im: -mathshim::sine(x.re) * mathshim::sinh_of(x.im) }; }
        else if op == 5 { r = Cplx { re: mathshim::sinh_of(x.re) * mathshim::cosine(x.im), im: mathshim::cosh_of(x.re) * mathshim::sine(x.im) }; }
        else if op == 6 { r = Cplx { re: mathshim::cosh_of(x.re) * mathshim::cosine(x.im), im: mathshim::sinh_of(x.re) * mathshim::sine(x.im) }; }
        else { r = Cplx { re: -x.re, im: -x.im }; }
        d.push(r.re);
        d.push(r.im);
        i += 1;
    }
    return carr_make(a.shape, copy(d));
}

fn cexp(a) -> CArr { return cun(a, 0); }
fn clog(a) -> CArr { return cun(a, 1); }
fn csqrt(a) -> CArr { return cun(a, 2); }
fn csin(a) -> CArr { return cun(a, 3); }
fn ccos(a) -> CArr { return cun(a, 4); }
fn csinh(a) -> CArr { return cun(a, 5); }
fn ccosh(a) -> CArr { return cun(a, 6); }
fn cneg(a) -> CArr { return cun(a, 7); }

// ctranspose gives the transpose of a complex matrix.
fn ctranspose(a) -> CArr, Fault {
    if a.ndim != 2 {
        return carr_empty(), fault(F_SHAPE, "ctranspose", "the operand is not a matrix");
    }
    let m = a.shape[0];
    let n = a.shape[1];
    var d: f64[>..] = [];
    var j: i64 = 0;
    while j < n {
        var i: i64 = 0;
        while i < m {
            d.push(a.data[2 * (i * n + j)]);
            d.push(a.data[2 * (i * n + j) + 1]);
            i += 1;
        }
        j += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    s.push(m);
    return carr_make(s, copy(d)), no_fault();
}

// adjoint gives the conjugate transpose of a complex matrix.
fn adjoint(a) -> CArr, Fault {
    let t, f = ctranspose(a);
    if f.kind != F_NONE { return carr_empty(), f; }
    return conj(t), no_fault();
}

