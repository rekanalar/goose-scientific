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

// cmatmul gives the product of two complex matrices.
fn cmatmul(a, b) -> CArr, Fault {
    if a.ndim != 2 || b.ndim != 2 || a.shape[1] != b.shape[0] {
        return carr_empty(), fault(F_SHAPE, "cmatmul", "the operands are not matrices with agreeing inner sides");
    }
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    if m * n > MAX_ELEMENTS {
        return carr_empty(), fault(F_LIMIT, "cmatmul", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < m {
        var j: i64 = 0;
        while j < n {
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            var p: i64 = 0;
            while p < k {
                let ar = a.data[2 * (i * k + p)];
                let ai = a.data[2 * (i * k + p) + 1];
                let br = b.data[2 * (p * n + j)];
                let bi = b.data[2 * (p * n + j) + 1];
                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
                p += 1;
            }
            d.push(sr);
            d.push(si);
            j += 1;
        }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(m);
    s.push(n);
    return carr_make(s, copy(d)), no_fault();
}

// csolve solves a x = b for a complex square matrix a. The right side
// holds a vector or a matrix. The method solves the real system of
// twice the side that holds the real and the imaginary parts.
fn csolve(a, b) -> CArr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return carr_empty(), fault(F_SHAPE, "csolve", "the matrix is not square");
    }
    let n = a.shape[0];
    var cols: i64 = 1;
    var vec = false;
    if b.ndim == 1 { vec = true; }
    else if b.ndim == 2 { cols = b.shape[1]; }
    else { return carr_empty(), fault(F_SHAPE, "csolve", "the right side is not a vector or a matrix"); }
    if b.shape[0] != n {
        return carr_empty(), fault(F_SHAPE, "csolve", "the right side does not agree with the matrix");
    }
    if n > 1024 {
        return carr_empty(), fault(F_LIMIT, "csolve", "the side is above 1024");
    }
    let w = 2 * n;
    // The matrix [[ar, -ai], [ai, ar]] and the right side [br; bi].
    var big: f64[>..] = [];
    var i: i64 = 0;
    while i < w * w { big.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            let ar = a.data[2 * (i * n + j)];
            let ai = a.data[2 * (i * n + j) + 1];
            big[i * w + j] = ar;
            big[i * w + n + j] = -ai;
            big[(n + i) * w + j] = ai;
            big[(n + i) * w + n + j] = ar;
            j += 1;
        }
        i += 1;
    }
    var rhs: f64[>..] = [];
    i = 0;
    while i < w * cols { rhs.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var c: i64 = 0;
        while c < cols {
            rhs[i * cols + c] = b.data[2 * (i * cols + c)];
            rhs[(n + i) * cols + c] = b.data[2 * (i * cols + c) + 1];
            c += 1;
        }
        i += 1;
    }
    var rb = line_arr(copy(rhs));
    if !vec { rb = mat2(copy_slice(rhs), w, cols); }
    let x, f = solve(mat2(copy(big), w, w), rb);
    if f.kind != F_NONE { return carr_empty(), f; }
    var d: f64[>..] = [];
    i = 0;
    while i < n {
        var c: i64 = 0;
        while c < cols {
            d.push(x.data[i * cols + c]);
            d.push(x.data[(n + i) * cols + c]);
            c += 1;
        }
        i += 1;
    }
    return carr_make(b.shape, copy(d)), no_fault();
}

// cinv gives the inverse of a complex square matrix.
fn cinv(a) -> CArr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return carr_empty(), fault(F_SHAPE, "cinv", "the matrix is not square");
    }
    let n = a.shape[0];
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n {
        if i / n == i % n { d.push(1.0); } else { d.push(0.0); }
        d.push(0.0);
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    s.push(n);
    return csolve(a, carr_make(s, copy(d)));
}

// cdet gives the determinant of a complex square matrix, as the real
// part and the imaginary part. The method is the elimination with the
// largest pivot.
fn cdet(a) -> f64, f64, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return 0.0, 0.0, fault(F_SHAPE, "cdet", "the matrix is not square");
    }
    let n = a.shape[0];
    if n > 2048 {
        return 0.0, 0.0, fault(F_LIMIT, "cdet", "the side is above 2048");
    }
    var m: f64[>..] = [];
    var i: i64 = 0;
    while i < 2 * n * n { m.push(a.data[i]); i += 1; }
    var dr: f64 = 1.0;
    var di: f64 = 0.0;
    var col: i64 = 0;
    while col < n {
        var best = col;
        var bv: f64 = -1.0;
        var r: i64 = col;
        while r < n {
            let v = mathshim::hypot_of(m[2 * (r * n + col)], m[2 * (r * n + col) + 1]);
            if v > bv { bv = v; best = r; }
            r += 1;
        }
        if bv == 0.0 { return 0.0, 0.0, no_fault(); }
        if best != col {
            var c: i64 = 0;
            while c < 2 * n {
                let t = m[2 * col * n + c];
                m[2 * col * n + c] = m[2 * best * n + c];
                m[2 * best * n + c] = t;
                c += 1;
            }
            dr = -dr;
            di = -di;
        }
        let p = Cplx { re: m[2 * (col * n + col)], im: m[2 * (col * n + col) + 1] };
        let nd = c_mul(Cplx { re: dr, im: di }, p);
        dr = nd.re;
        di = nd.im;
        r = col + 1;
        while r < n {
            let q = c_div(Cplx { re: m[2 * (r * n + col)], im: m[2 * (r * n + col) + 1] }, p);
            var c: i64 = col;
            while c < n {
                let u = c_mul(q, Cplx { re: m[2 * (col * n + c)], im: m[2 * (col * n + c) + 1] });
                m[2 * (r * n + c)] = m[2 * (r * n + c)] - u.re;
                m[2 * (r * n + c) + 1] = m[2 * (r * n + c) + 1] - u.im;
                c += 1;
            }
            r += 1;
        }
        col += 1;
    }
    return dr, di, no_fault();
}

// cfft gives the transform of a complex signal of one dimension.
fn cfft(a) -> CArr, Fault {
    if a.ndim != 1 {
        return carr_empty(), fault(F_SHAPE, "cfft", "the operand is not a vector");
    }
    let re = carr_part(a, 0);
    let im = carr_part(a, 1);
    let r, i, f = fft_parts(re.data, im.data);
    if f.kind != F_NONE { return carr_empty(), f; }
    return complex_array(r, i);
}

// cifft gives the inverse transform of a complex signal.
fn cifft(a) -> CArr, Fault {
    if a.ndim != 1 {
        return carr_empty(), fault(F_SHAPE, "cifft", "the operand is not a vector");
    }
    let re = carr_part(a, 0);
    let im = carr_part(a, 1);
    let r, i, f = ifft(re.data, im.data);
    if f.kind != F_NONE { return carr_empty(), f; }
    return complex_array(r, i);
}

// hqr_value reads the element (i, j) of the work matrix. The indexes
// start at one, as in the method.
fn hqr_at(i: i64, j: i64, n: i64) -> i64 { return (i - 1) * n + (j - 1); }

fn hqr_sign(a: f64, b: f64) -> f64 {
    var x = mathshim::abs_of(a);
    if b < 0.0 { x = -x; }
    return x;
}

// eigvals_complex gives the eigenvalues of a real square matrix, which
// may hold complex pairs. The method is the Hessenberg form and the QR
// iteration with two shifts. The eigenvalues come sorted by the real
// part, then by the imaginary part. NumPy gives them in another order.
fn eigvals_complex(a) -> CArr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] || a.shape[0] < 1 {
        return carr_empty(), fault(F_SHAPE, "eigvals_complex", "the operand is not a square matrix");
    }
    let n = a.shape[0];
    if n > 512 {
        return carr_empty(), fault(F_LIMIT, "eigvals_complex", "the side is above 512");
    }
    var i: i64 = 0;
    while i < n * n {
        if !(a.data[i] - a.data[i] == 0.0) {
            return carr_empty(), fault(F_VALUE, "eigvals_complex", "the matrix holds a value that is not finite");
        }
        i += 1;
    }
    var h: f64[>..] = [];
    hessenberg(a.data, n, h);
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < i - 1 { h[i * n + j] = 0.0; j += 1; }
        i += 1;
    }
    var wr: f64[>..] = [];
    var wi: f64[>..] = [];
    i = 0;
    while i < n { wr.push(0.0); wi.push(0.0); i += 1; }
    var anorm: f64 = 0.0;
    i = 1;
    while i <= n {
        var j = i - 1;
        if j < 1 { j = 1; }
        while j <= n { anorm += mathshim::abs_of(h[hqr_at(i, j, n)]); j += 1; }
        i += 1;
    }
    var nn = n;
    var t: f64 = 0.0;
    var p: f64 = 0.0;
    var q: f64 = 0.0;
    var r: f64 = 0.0;
    var x: f64 = 0.0;
    var y: f64 = 0.0;
    var z: f64 = 0.0;
    var w: f64 = 0.0;
    var s: f64 = 0.0;
    while nn >= 1 {
        var its: i64 = 0;
        var again = true;
        while again {
            var l = nn;
            while l >= 2 {
                s = mathshim::abs_of(h[hqr_at(l - 1, l - 1, n)]) + mathshim::abs_of(h[hqr_at(l, l, n)]);
                if s == 0.0 { s = anorm; }
                if mathshim::abs_of(h[hqr_at(l, l - 1, n)]) + s == s {
                    h[hqr_at(l, l - 1, n)] = 0.0;
                    break;
                }
                l -= 1;
            }
            x = h[hqr_at(nn, nn, n)];
            if l == nn {
                wr[nn - 1] = x + t;
                wi[nn - 1] = 0.0;
                nn -= 1;
                again = false;
            } else {
                y = h[hqr_at(nn - 1, nn - 1, n)];
                w = h[hqr_at(nn, nn - 1, n)] * h[hqr_at(nn - 1, nn, n)];
                if l == nn - 1 {
                    p = 0.5 * (y - x);
                    q = p * p + w;
                    z = mathshim::root(mathshim::abs_of(q));
                    x = x + t;
                    if q >= 0.0 {
                        z = p + hqr_sign(z, p);
                        wr[nn - 2] = x + z;
                        wr[nn - 1] = wr[nn - 2];
                        if z != 0.0 { wr[nn - 1] = x - w / z; }
                        wi[nn - 2] = 0.0;
                        wi[nn - 1] = 0.0;
                    } else {
                        wr[nn - 2] = x + p;
                        wr[nn - 1] = x + p;
                        wi[nn - 2] = z;
                        wi[nn - 1] = -z;
                    }
                    nn -= 2;
                    again = false;
                } else {
                    if its >= 60 {
                        return carr_empty(), fault(F_LIMIT, "eigvals_complex", "the iteration did not converge");
                    }
                    if its == 10 || its == 20 {
                        t += x;
                        i = 1;
                        while i <= nn { h[hqr_at(i, i, n)] = h[hqr_at(i, i, n)] - x; i += 1; }
                        s = mathshim::abs_of(h[hqr_at(nn, nn - 1, n)]) + mathshim::abs_of(h[hqr_at(nn - 1, nn - 2, n)]);
                        x = 0.75 * s;
                        y = x;
                        w = -0.4375 * s * s;
                    }
                    its += 1;
                    var m = nn - 2;
                    var found = false;
                    while m >= l && !found {
                        z = h[hqr_at(m, m, n)];
                        r = x - z;
                        s = y - z;
                        p = (r * s - w) / h[hqr_at(m + 1, m, n)] + h[hqr_at(m, m + 1, n)];
                        q = h[hqr_at(m + 1, m + 1, n)] - z - r - s;
                        r = h[hqr_at(m + 2, m + 1, n)];
                        s = mathshim::abs_of(p) + mathshim::abs_of(q) + mathshim::abs_of(r);
                        p = p / s;
                        q = q / s;
                        r = r / s;
                        if m == l { found = true; }
                        else {
                            let u = mathshim::abs_of(h[hqr_at(m, m - 1, n)]) * (mathshim::abs_of(q) + mathshim::abs_of(r));
                            let v = mathshim::abs_of(p) * (mathshim::abs_of(h[hqr_at(m - 1, m - 1, n)]) + mathshim::abs_of(z) + mathshim::abs_of(h[hqr_at(m + 1, m + 1, n)]));
                            if u + v == v { found = true; } else { m -= 1; }
                        }
                    }
                    i = m + 2;
                    while i <= nn {
                        h[hqr_at(i, i - 2, n)] = 0.0;
                        if i != m + 2 { h[hqr_at(i, i - 3, n)] = 0.0; }
                        i += 1;
                    }
                    var k = m;
                    while k <= nn - 1 {
                        if k != m {
                            p = h[hqr_at(k, k - 1, n)];
                            q = h[hqr_at(k + 1, k - 1, n)];
                            r = 0.0;
                            if k != nn - 1 { r = h[hqr_at(k + 2, k - 1, n)]; }
                            x = mathshim::abs_of(p) + mathshim::abs_of(q) + mathshim::abs_of(r);
                            if x != 0.0 { p = p / x; q = q / x; r = r / x; }
                        }
                        s = hqr_sign(mathshim::root(p * p + q * q + r * r), p);
                        if s != 0.0 {
                            if k == m {
                                if l != m { h[hqr_at(k, k - 1, n)] = -h[hqr_at(k, k - 1, n)]; }
                            } else {
                                h[hqr_at(k, k - 1, n)] = -s * x;
                            }
                            p = p + s;
                            x = p / s;
                            y = q / s;
                            z = r / s;
                            q = q / p;
                            r = r / p;
                            var j = k;
                            while j <= nn {
                                p = h[hqr_at(k, j, n)] + q * h[hqr_at(k + 1, j, n)];
                                if k != nn - 1 {
                                    p = p + r * h[hqr_at(k + 2, j, n)];
                                    h[hqr_at(k + 2, j, n)] = h[hqr_at(k + 2, j, n)] - p * z;
                                }
                                h[hqr_at(k + 1, j, n)] = h[hqr_at(k + 1, j, n)] - p * y;
                                h[hqr_at(k, j, n)] = h[hqr_at(k, j, n)] - p * x;
                                j += 1;
                            }
                            var mmin = k + 3;
                            if nn < mmin { mmin = nn; }
                            i = l;
                            while i <= mmin {
                                p = x * h[hqr_at(i, k, n)] + y * h[hqr_at(i, k + 1, n)];
                                if k != nn - 1 {
                                    p = p + z * h[hqr_at(i, k + 2, n)];
                                    h[hqr_at(i, k + 2, n)] = h[hqr_at(i, k + 2, n)] - p * r;
                                }
                                h[hqr_at(i, k + 1, n)] = h[hqr_at(i, k + 1, n)] - p * q;
                                h[hqr_at(i, k, n)] = h[hqr_at(i, k, n)] - p;
                                i += 1;
                            }
                        }
                        k += 1;
                    }
                }
            }
        }
    }
    // Sort by the real part, then by the imaginary part.
    var keys: f64[>..] = [];
    i = 0;
    while i < n { keys.push(wi[i]); i += 1; }
    i = 0;
    while i < n { keys.push(wr[i]); i += 1; }
    let ord = order_places(keys, 2, n);
    var d: f64[>..] = [];
    i = 0;
    while i < n { d.push(wr[ord[i]]); d.push(wi[ord[i]]); i += 1; }
    var s1: i64[..8] = [];
    s1.push(n);
    return carr_make(s1, copy(d)), no_fault();
}
