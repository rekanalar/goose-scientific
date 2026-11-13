// The trigonometric and hyperbolic matrix functions, signm and funm.
//
namespace scientific;

import mathshim;

// mf_trig gives cos A and sin A from the exponential of the real block
// [[0, -A], [A, 0]], which holds exp(iA) in real form. The side of the
// block is twice the side of A, so the limit of expm halves.
fn mf_trig(a, tag: const u8[:]) -> Arr, Arr, Fault {
    let c = mf_square(a, tag);
    if c.kind != F_NONE { return empty_arr(), empty_arr(), c; }
    let n = a.shape[0];
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 4 * n * n { d.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            d[i * 2 * n + n + j] = -a.data[i * n + j];
            d[(n + i) * 2 * n + j] = a.data[i * n + j];
            j += 1;
        }
        i += 1;
    }
    let e, f = expm(mat2(copy(d), 2 * n, 2 * n));
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    var cs: f64[>..] = [];
    var sn: f64[>..] = [];
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            cs.push(e.data[i * 2 * n + j]);
            sn.push(e.data[(n + i) * 2 * n + j]);
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(cs), n, n), mat2(copy(sn), n, n), no_fault();
}

// cosm gives the cosine of a square matrix.
fn cosm(a) -> Arr, Fault {
    let c, s, f = mf_trig(a, "cosm");
    return c, f;
}

// sinm gives the sine of a square matrix.
fn sinm(a) -> Arr, Fault {
    let c, s, f = mf_trig(a, "sinm");
    return s, f;
}

// tanm gives the tangent of a square matrix, the solution of cos(A) x = sin(A).
fn tanm(a) -> Arr, Fault {
    let c, s, f = mf_trig(a, "tanm");
    if f.kind != F_NONE { return empty_arr(), f; }
    return solve(c, s);
}

// mf_hyper gives cosh A and sinh A from exp(A) and exp(-A).
fn mf_hyper(a, tag: const u8[:]) -> Arr, Arr, Fault {
    let c = mf_square(a, tag);
    if c.kind != F_NONE { return empty_arr(), empty_arr(), c; }
    var neg: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { neg.push(-a.data[i]); i += 1; }
    let n = a.shape[0];
    let ep, f1 = expm(a);
    if f1.kind != F_NONE { return empty_arr(), empty_arr(), f1; }
    let em, f2 = expm(mat2(copy(neg), n, n));
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
    var ch: f64[>..] = [];
    var sh: f64[>..] = [];
    i = 0;
    while i < n * n {
        ch.push(0.5 * (ep.data[i] + em.data[i]));
        sh.push(0.5 * (ep.data[i] - em.data[i]));
        i += 1;
    }
    return mat2(copy(ch), n, n), mat2(copy(sh), n, n), no_fault();
}

// coshm gives the hyperbolic cosine of a square matrix.
fn coshm(a) -> Arr, Fault {
    let c, s, f = mf_hyper(a, "coshm");
    return c, f;
}

// sinhm gives the hyperbolic sine of a square matrix.
fn sinhm(a) -> Arr, Fault {
    let c, s, f = mf_hyper(a, "sinhm");
    return s, f;
}

// tanhm gives the hyperbolic tangent of a square matrix.
fn tanhm(a) -> Arr, Fault {
    let c, s, f = mf_hyper(a, "tanhm");
    if f.kind != F_NONE { return empty_arr(), f; }
    return solve(c, s);
}

// signm gives the sign function of a matrix by the Newton iteration
// S = (S + S^-1) / 2. The matrix must hold no eigenvalue on the
// imaginary axis.
fn signm(a) -> Arr, Fault {
    let c = mf_square(a, "signm");
    if c.kind != F_NONE { return empty_arr(), c; }
    let n = a.shape[0];
    var s = mat2(copy_slice(a.data), n, n);
    var turn: i64 = 0;
    while turn < 100 {
        let si, f = inv(s);
        if f.kind != F_NONE { return empty_arr(), fault(F_VALUE, "signm", "the matrix holds an eigenvalue on the imaginary axis"); }
        var nx: f64[>..] = [];
        var delta: f64 = 0.0;
        var size: f64 = 0.0;
        var i: i64 = 0;
        while i < n * n {
            let v = 0.5 * (s.data[i] + si.data[i]);
            let d = mathshim::abs_of(v - s.data[i]);
            if d > delta { delta = d; }
            if mathshim::abs_of(v) > size { size = mathshim::abs_of(v); }
            nx.push(v);
            i += 1;
        }
        if !(size - size == 0.0) {
            return empty_arr(), fault(F_VALUE, "signm", "the matrix holds an eigenvalue on the imaginary axis");
        }
        s = mat2(copy(nx), n, n);
        if delta <= 1.0e-14 * size { return s, no_fault(); }
        turn += 1;
    }
    return empty_arr(), fault(F_VALUE, "signm", "the iteration did not converge: the matrix holds an eigenvalue on the imaginary axis");
}

// funm gives the function of a matrix through the Schur form. The
// caller hands over a function of one number. The method applies the
// function to the diagonal of the triangular form and fills the rest
// with the Parlett recurrence. A pair of complex values on the
// diagonal is a fault, and so is a difference of two diagonal values
// that is too small for the recurrence.
fn funm<F: fn(f64) -> f64>(a) -> Arr, Fault {
    let fs = mf_square(a, "funm");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    let t, z, fq = schur(a);
    if fq.kind != F_NONE { return empty_arr(), fq; }
    let n = t.shape[0];
    var i: i64 = 0;
    while i < n {
        if i > 0 && t.data[i * n + (i - 1)] != 0.0 {
            return empty_arr(), fault(F_VALUE, "funm", "the matrix holds a complex pair of eigenvalues");
        }
        i += 1;
    }
    var scale: f64 = 0.0;
    i = 0;
    while i < n {
        var v = mathshim::root(t.data[i * n + i] * t.data[i * n + i]);
        if v > scale { scale = v; }
        i += 1;
    }
    if scale == 0.0 { scale = 1.0; }
    var fm: f64[>..] = [];
    i = 0;
    while i < n * n { fm.push(0.0); i += 1; }
    i = 0;
    while i < n {
        fm[i * n + i] = F(t.data[i * n + i]);
        i += 1;
    }
    // The recurrence walks over the difference of the indexes, so
    // every value it needs is in place.
    var d: i64 = 1;
    while d < n {
        i = 0;
        while i + d < n {
            let j = i + d;
            let den = t.data[j * n + j] - t.data[i * n + i];
            if mathshim::root(den * den) <= 1.0e-12 * scale {
                return empty_arr(), fault(F_VALUE, "funm", "the matrix holds two diagonal values that are too close");
            }
            var s: f64 = t.data[i * n + j] * (fm[j * n + j] - fm[i * n + i]);
            var k: i64 = i + 1;
            while k < j {
                s += t.data[i * n + k] * fm[k * n + j] - fm[i * n + k] * t.data[k * n + j];
                k += 1;
            }
            fm[i * n + j] = s / den;
            i += 1;
        }
        d += 1;
    }
    var farr: i64[..8] = [];
    farr.push(n);
    farr.push(n);
    let fz = Arr { ndim: 2, shape: farr, data: copy(fm) };
    let left, fl = matmul(z, fz);
    if fl.kind != F_NONE { return empty_arr(), fl; }
    let zt, ft = transpose(z);
    if ft.kind != F_NONE { return empty_arr(), ft; }
    return matmul(left, zt);
}
