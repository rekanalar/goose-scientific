// The product, the solvers and the factors of complex matrices.
//
namespace scientific;

import mathshim;

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

// carr_finite gives false when an element of a complex array holds a
// value that is not finite. It checks the real part and the imaginary
// part of every element.
fn carr_finite(c) -> bool {
    var i: i64 = 0;
    while i < c.data.len {
        if !(c.data[i] - c.data[i] == 0.0) { return false; }
        i += 1;
    }
    return true;
}

// carr_check checks the buffer of a complex array against its shape. The
// buffer holds two values for each element of the shape. A fault of
// kind F_SHAPE says that the counts differ.
fn carr_check(a, tag: const u8[:]) -> Fault {
    return check_stack(a.ndim, a.shape, a.data.len, 2, tag);
}

// c_dot_stride gives the sum of the products of two complex buffers.
// The offsets give the first element of each side. The strides give
// the step between two elements. The flag conjugates the first side.
fn c_dot_stride(a: f64[>..]&, aoff: i64, astride: i64, b: f64[>..]&, boff: i64, bstride: i64, n: i64, conjugate: bool) -> f64, f64 {
    var sr: f64 = 0.0;
    var si: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        var ai = a[aoff + astride * i + 1];
        if conjugate { ai = -ai; }
        let p = c_mul(Cplx { re: a[aoff + astride * i], im: ai },
                      Cplx { re: b[boff + bstride * i], im: b[boff + bstride * i + 1] });
        sr += p.re;
        si += p.im;
        i += 1;
    }
    return sr, si;
}

// cdot gives the product of two complex arrays. Two vectors give a
// scalar. A matrix and a vector give a vector. A vector and a matrix
// give a vector. Two matrices give the matrix product. The products do
// not conjugate, as NumPy does not. The count of dimensions of an
// operand must be one or two.
fn cdot(a, b) -> CArr, Fault {
    let fa = carr_check(a, "cdot");
    if fa.kind != F_NONE { return carr_empty(), fa; }
    let fb = carr_check(b, "cdot");
    if fb.kind != F_NONE { return carr_empty(), fb; }
    if a.ndim > 2 || b.ndim > 2 {
        return carr_empty(), fault(F_SHAPE, "cdot", "an operand holds more than two dimensions");
    }
    if a.ndim < 1 || b.ndim < 1 {
        return carr_empty(), fault(F_SHAPE, "cdot", "an operand holds no dimension");
    }
    if a.shape[a.ndim - 1] != b.shape[0] {
        return carr_empty(), fault(F_SHAPE, "cdot", "the inner dimensions do not agree");
    }
    // Two vectors give a scalar.
    if a.ndim == 1 && b.ndim == 1 {
        let sr, si = c_dot_stride(a.data, 0, 2, b.data, 0, 2, a.shape[0], false);
        var d: f64[>..] = [];
        d.push(sr);
        d.push(si);
        return CArr { ndim: 0, shape: [], data: d }, no_fault();
    }
    // Two matrices give the matrix product.
    if a.ndim == 2 && b.ndim == 2 {
        let m = a.shape[0];
        let p = b.shape[1];
        // The count of the elements of the result must not overflow.
        if m > MAX_ELEMENTS / p {
            return carr_empty(), fault(F_LIMIT, "cdot", "the result holds more elements than the limit");
        }
        let r, f = cmatmul(a, b);
        if f.kind != F_NONE { return carr_empty(), fault(f.kind, "cdot", f.detail); }
        return r, no_fault();
    }
    // A matrix and a vector give a vector.
    if a.ndim == 2 {
        let m = a.shape[0];
        let n = a.shape[1];
        var d: f64[>..] = [];
        var i: i64 = 0;
        while i < m {
            let sr, si = c_dot_stride(a.data, 2 * i * n, 2, b.data, 0, 2, n, false);
            d.push(sr);
            d.push(si);
            i += 1;
        }
        var s: i64[..8] = [];
        s.push(m);
        return CArr { ndim: 1, shape: s, data: d }, no_fault();
    }
    // A vector and a matrix give a vector.
    let m = b.shape[0];
    let n = b.shape[1];
    var d: f64[>..] = [];
    var j: i64 = 0;
    while j < n {
        let sr, si = c_dot_stride(a.data, 0, 2, b.data, 2 * j, 2 * n, m, false);
        d.push(sr);
        d.push(si);
        j += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    return CArr { ndim: 1, shape: s, data: d }, no_fault();
}

// cvdot gives the scalar product of two complex arrays. It flattens
// both operands and conjugates the first one, as NumPy does.
fn cvdot(a, b) -> f64, f64, Fault {
    let fa = carr_check(a, "cvdot");
    if fa.kind != F_NONE { return 0.0, 0.0, fa; }
    let fb = carr_check(b, "cvdot");
    if fb.kind != F_NONE { return 0.0, 0.0, fb; }
    if a.data.len != b.data.len {
        return 0.0, 0.0, fault(F_SHAPE, "cvdot", "the counts of elements do not agree");
    }
    let sr, si = c_dot_stride(a.data, 0, 2, b.data, 0, 2, a.data.len / 2, true);
    return sr, si, no_fault();
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

// ccholesky gives the lower triangle of the Cholesky factor of a
// complex Hermitian matrix. It reads the lower triangle of the operand
// and takes the real part of the diagonal, as NumPy does. The upper
// triangle of the result holds zero. A matrix that is not positive
// definite gives a fault of kind F_VALUE.
fn ccholesky(a) -> CArr, Fault {
    if a.ndim != 2 || a.shape.len != 2 || a.shape[0] != a.shape[1] {
        return carr_empty(), fault(F_SHAPE, "ccholesky", "the matrix is not square");
    }
    let n = a.shape[0];
    if n > 1024 {
        return carr_empty(), fault(F_LIMIT, "ccholesky", "the side is above 1024");
    }
    let f = carr_check(a, "ccholesky");
    if f.kind != F_NONE { return carr_empty(), f; }
    if !carr_finite(a) {
        return carr_empty(), fault(F_VALUE, "ccholesky", "the matrix holds a value that is not finite");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { d.push(0.0); d.push(0.0); i += 1; }
    var j: i64 = 0;
    while j < n {
        // The pivot of the column is the real part of the diagonal
        // minus the sum of the squares of the row above it.
        var s = a.data[2 * (j * n + j)];
        var k: i64 = 0;
        while k < j {
            let lr = d[2 * (j * n + k)];
            let li = d[2 * (j * n + k) + 1];
            s -= lr * lr + li * li;
            k += 1;
        }
        if !(s > 0.0) {
            return carr_empty(), fault(F_VALUE, "ccholesky", "the matrix is not positive definite");
        }
        let piv = mathshim::root(s);
        d[2 * (j * n + j)] = piv;
        d[2 * (j * n + j) + 1] = 0.0;
        var r = j + 1;
        while r < n {
            var tr = a.data[2 * (r * n + j)];
            var ti = a.data[2 * (r * n + j) + 1];
            k = 0;
            while k < j {
                // The sum takes the conjugate of the row above.
                let ar = d[2 * (r * n + k)];
                let ai = d[2 * (r * n + k) + 1];
                let br = d[2 * (j * n + k)];
                let bi = -d[2 * (j * n + k) + 1];
                tr -= ar * br - ai * bi;
                ti -= ar * bi + ai * br;
                k += 1;
            }
            d[2 * (r * n + j)] = tr / piv;
            d[2 * (r * n + j) + 1] = ti / piv;
            r += 1;
        }
        j += 1;
    }
    var sh: i64[..8] = [];
    sh.push(n);
    sh.push(n);
    return CArr { ndim: 2, shape: sh, data: d }, no_fault();
}

