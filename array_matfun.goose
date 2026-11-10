// The functions of matrices and the equations of matrices: the square
// root, the logarithm, the power by a fraction, the polar form, the
// solve by the factor of Cholesky, the equations of Sylvester and
// Lyapunov and the eigenvalues of a banded symmetric matrix.
//
// The package holds real numbers only. A function of a matrix whose
// result needs a complex number, such as the root of a matrix with a
// negative eigenvalue, is a fault.
namespace scientific;

import mathshim;

// The largest side of a matrix for the equations by the Kronecker form.
const MAX_MATFUN_SIDE: i64 = 4096;

fn mf_eye(n: i64) -> Arr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { d.push(0.0); i += 1; }
    i = 0;
    while i < n { d[i * n + i] = 1.0; i += 1; }
    return mat2(copy(d), n, n);
}

fn mf_square(a, tag: const u8[:]) -> Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] || a.shape[0] < 1 {
        return fault(F_SHAPE, tag, "the operand is not a square matrix");
    }
    var i: i64 = 0;
    while i < a.data.len {
        if !(a.data[i] - a.data[i] == 0.0) {
            return fault(F_VALUE, tag, "the matrix holds a value that is not finite");
        }
        i += 1;
    }
    return no_fault();
}

// sqrtm gives the principal square root of a matrix by the iteration of
// Denman and Beavers. The matrix must have no eigenvalue on the
// negative real axis and no zero eigenvalue.
fn sqrtm(a) -> Arr, Fault {
    let c = mf_square(a, "sqrtm");
    if c.kind != F_NONE { return empty_arr(), c; }
    if a.shape[0] > 512 {
        return empty_arr(), fault(F_LIMIT, "sqrtm", "the side is above 512");
    }
    let n = a.shape[0];
    var y = mat2(copy_slice(a.data), n, n);
    var z = mf_eye(n);
    var turn: i64 = 0;
    while turn < 100 {
        let yi, f1 = inv(y);
        if f1.kind != F_NONE { return empty_arr(), fault(F_VALUE, "sqrtm", "the matrix holds no principal root"); }
        let zi, f2 = inv(z);
        if f2.kind != F_NONE { return empty_arr(), fault(F_VALUE, "sqrtm", "the matrix holds no principal root"); }
        var ny: f64[>..] = [];
        var nz: f64[>..] = [];
        var delta: f64 = 0.0;
        var size: f64 = 0.0;
        var i: i64 = 0;
        while i < n * n {
            let v = 0.5 * (y.data[i] + zi.data[i]);
            let w = 0.5 * (z.data[i] + yi.data[i]);
            let d = mathshim::abs_of(v - y.data[i]);
            if d > delta { delta = d; }
            if mathshim::abs_of(v) > size { size = mathshim::abs_of(v); }
            ny.push(v);
            nz.push(w);
            i += 1;
        }
        if !(size - size == 0.0) {
            return empty_arr(), fault(F_VALUE, "sqrtm", "the matrix holds no principal root");
        }
        y = mat2(copy(ny), n, n);
        z = mat2(copy(nz), n, n);
        if delta <= 1.0e-15 * size { return y, no_fault(); }
        turn += 1;
    }
    return empty_arr(), fault(F_VALUE, "sqrtm", "the iteration did not converge: the matrix has an eigenvalue on the negative axis");
}

// mf_dist1 gives the largest column sum of |a - I|.
fn mf_dist1(a, n: i64) -> f64 {
    var best: f64 = 0.0;
    var j: i64 = 0;
    while j < n {
        var s: f64 = 0.0;
        var i: i64 = 0;
        while i < n {
            var v = a.data[i * n + j];
            if i == j { v = v - 1.0; }
            s += mathshim::abs_of(v);
            i += 1;
        }
        if s > best { best = s; }
        j += 1;
    }
    return best;
}

// logm gives the principal logarithm of a matrix. The method takes
// square roots until the matrix lies near the identity, sums the
// series of the logarithm, and scales the result back. The matrix must
// have no eigenvalue on the negative real axis and no zero eigenvalue.
fn logm(a) -> Arr, Fault {
    let c = mf_square(a, "logm");
    if c.kind != F_NONE { return empty_arr(), c; }
    let n = a.shape[0];
    var cur = mat2(copy_slice(a.data), n, n);
    var k: i64 = 0;
    while mf_dist1(cur, n) > 0.25 {
        if k >= 60 { return empty_arr(), fault(F_VALUE, "logm", "the matrix holds no principal logarithm"); }
        let r, f = sqrtm(cur);
        if f.kind != F_NONE { return empty_arr(), fault(F_VALUE, "logm", "the matrix holds no principal logarithm"); }
        cur = mat2(copy_slice(r.data), n, n);
        k += 1;
    }
    // x = cur - I.
    var xd: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { xd.push(cur.data[i]); i += 1; }
    i = 0;
    while i < n { xd[i * n + i] = xd[i * n + i] - 1.0; i += 1; }
    let x = mat2(copy(xd), n, n);
    var pw = mat2(copy_slice(x.data), n, n);
    var sum: f64[>..] = [];
    i = 0;
    while i < n * n { sum.push(x.data[i]); i += 1; }
    var term: i64 = 2;
    while term <= 80 {
        let nx, f = matmul(pw, x);
        if f.kind != F_NONE { return empty_arr(), f; }
        pw = mat2(copy_slice(nx.data), n, n);
        var sg: f64 = 1.0;
        if term % 2 == 0 { sg = -1.0; }
        var big: f64 = 0.0;
        i = 0;
        while i < n * n {
            let t = sg * pw.data[i] / term;
            sum[i] = sum[i] + t;
            if mathshim::abs_of(t) > big { big = mathshim::abs_of(t); }
            i += 1;
        }
        if big < 1.0e-18 { break; }
        term += 1;
    }
    var scale: f64 = 1.0;
    i = 0;
    while i < k { scale = scale * 2.0; i += 1; }
    i = 0;
    while i < n * n { sum[i] = sum[i] * scale; i += 1; }
    return mat2(copy(sum), n, n), no_fault();
}

// fractional_matrix_power raises a matrix to a real power t. The
// method is expm of t times logm, so the matrix needs a principal
// logarithm.
fn fractional_matrix_power(a, t: f64) -> Arr, Fault {
    let l, f = logm(a);
    if f.kind != F_NONE { return empty_arr(), f; }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < l.data.len { d.push(l.data[i] * t); i += 1; }
    return expm(mat2(copy(d), l.shape[0], l.shape[1]));
}

// polar gives the polar form a = u p of a square matrix. The factor u
// is orthogonal and p is symmetric and positive semidefinite.
fn polar(a) -> Arr, Arr, Fault {
    let c = mf_square(a, "polar");
    if c.kind != F_NONE { return empty_arr(), empty_arr(), c; }
    let n = a.shape[0];
    let u, s, vt, f = svd(a);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let w, f1 = matmul(u, vt);
    if f1.kind != F_NONE { return empty_arr(), empty_arr(), f1; }
    // p = V diag(s) V' with V' = vt.
    var sv: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { sv.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n { sv[i * n + j] = s.data[i] * vt.data[i * n + j]; j += 1; }
        i += 1;
    }
    var vtt: f64[>..] = [];
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n { vtt.push(vt.data[j * n + i]); j += 1; }
        i += 1;
    }
    let p, f2 = matmul(mat2(copy(vtt), n, n), mat2(copy(sv), n, n));
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
    return w, p, no_fault();
}

// cho_factor gives the lower factor of Cholesky of a symmetric
// positive definite matrix.
fn cho_factor(a) -> Arr, Fault { return cholesky(a); }

// cho_solve solves a x = b with the lower factor of Cholesky of a. The
// right side holds a vector or a matrix.
fn cho_solve(l, b) -> Arr, Fault {
    let y, f = solve_triangular(l, b, true);
    if f.kind != F_NONE { return empty_arr(), f; }
    return solve_triangular(l, y, true, 1);
}

// mf_vec gives the matrix as one row, in the row order.
fn mf_vec(q) -> Arr {
    return line_arr(copy_slice(q.data));
}

// solve_sylvester solves a x + x b = q for x. The matrix a holds n
// rows, b holds m rows, and q holds n rows and m columns. The method
// builds the Kronecker form, so n m stays at 4096 or below.
fn solve_sylvester(a, b, q) -> Arr, Fault {
    let c1 = mf_square(a, "solve_sylvester");
    if c1.kind != F_NONE { return empty_arr(), c1; }
    let c2 = mf_square(b, "solve_sylvester");
    if c2.kind != F_NONE { return empty_arr(), c2; }
    let n = a.shape[0];
    let m = b.shape[0];
    if q.ndim != 2 || q.shape[0] != n || q.shape[1] != m {
        return empty_arr(), fault(F_SHAPE, "solve_sylvester", "q does not hold n rows and m columns");
    }
    if n * m > MAX_MATFUN_SIDE {
        return empty_arr(), fault(F_LIMIT, "solve_sylvester", "n times m is above 4096");
    }
    // The unknown follows the row order: index i m + j. The matrix is
    // a (x) I_m + I_n (x) b'.
    let side = n * m;
    var k: f64[>..] = [];
    var r: i64 = 0;
    while r < side * side { k.push(0.0); r += 1; }
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < m {
            let row = i * m + j;
            var p: i64 = 0;
            while p < n { k[row * side + p * m + j] = k[row * side + p * m + j] + a.data[i * n + p]; p += 1; }
            var s: i64 = 0;
            while s < m { k[row * side + i * m + s] = k[row * side + i * m + s] + b.data[s * m + j]; s += 1; }
            j += 1;
        }
        i += 1;
    }
    let x, f = solve(mat2(copy(k), side, side), mf_vec(q));
    if f.kind != F_NONE { return empty_arr(), fault(F_VALUE, "solve_sylvester", "the equation holds no unique solution"); }
    return mat2(copy_slice(x.data), n, m), no_fault();
}

// solve_continuous_lyapunov solves a x + x a' = q for x.
fn solve_continuous_lyapunov(a, q) -> Arr, Fault {
    let c = mf_square(a, "solve_continuous_lyapunov");
    if c.kind != F_NONE { return empty_arr(), c; }
    let n = a.shape[0];
    var t: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n { t.push(a.data[j * n + i]); j += 1; }
        i += 1;
    }
    return solve_sylvester(a, mat2(copy(t), n, n), q);
}

// solve_discrete_lyapunov solves x - a x a' = q for x. The method
// builds the Kronecker form, so n n stays at 4096 or below.
fn solve_discrete_lyapunov(a, q) -> Arr, Fault {
    let c = mf_square(a, "solve_discrete_lyapunov");
    if c.kind != F_NONE { return empty_arr(), c; }
    let n = a.shape[0];
    if q.ndim != 2 || q.shape[0] != n || q.shape[1] != n {
        return empty_arr(), fault(F_SHAPE, "solve_discrete_lyapunov", "q is not a square matrix of the side of a");
    }
    if n * n > MAX_MATFUN_SIDE {
        return empty_arr(), fault(F_LIMIT, "solve_discrete_lyapunov", "n squared is above 4096");
    }
    let side = n * n;
    var k: f64[>..] = [];
    var r: i64 = 0;
    while r < side * side { k.push(0.0); r += 1; }
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            let row = i * n + j;
            k[row * side + row] = 1.0;
            var p: i64 = 0;
            while p < n {
                var s: i64 = 0;
                while s < n {
                    k[row * side + p * n + s] = k[row * side + p * n + s] - a.data[i * n + p] * a.data[j * n + s];
                    s += 1;
                }
                p += 1;
            }
            j += 1;
        }
        i += 1;
    }
    let x, f = solve(mat2(copy(k), side, side), mf_vec(q));
    if f.kind != F_NONE { return empty_arr(), fault(F_VALUE, "solve_discrete_lyapunov", "the equation holds no unique solution"); }
    return mat2(copy_slice(x.data), n, n), no_fault();
}

// banded_dense expands a symmetric matrix in the banded form of
// SciPy. The matrix holds u + 1 rows for the upper form, or the lower
// form with lower true.
fn banded_dense(ab, lower: bool, out: f64[>..]&) -> Fault {
    if ab.ndim != 2 || ab.shape[0] < 1 || ab.shape[1] < 1 {
        return fault(F_SHAPE, "eig_banded", "the operand is not a matrix of the banded form");
    }
    let rows = ab.shape[0];
    let n = ab.shape[1];
    if n > 2048 {
        return fault(F_LIMIT, "eig_banded", "the side is above 2048");
    }
    var i: i64 = 0;
    while i < n * n { out.push(0.0); i += 1; }
    var r: i64 = 0;
    while r < rows {
        var j: i64 = 0;
        while j < n {
            // Upper form: ab[u + i - j, j] = a[i, j]. Lower form:
            // ab[i - j, j] = a[i, j].
            if lower {
                let ii = j + r;
                if ii < n {
                    out[ii * n + j] = ab.data[r * n + j];
                    out[j * n + ii] = ab.data[r * n + j];
                }
            } else {
                let ii = j - (rows - 1 - r);
                if ii >= 0 {
                    out[ii * n + j] = ab.data[r * n + j];
                    out[j * n + ii] = ab.data[r * n + j];
                }
            }
            j += 1;
        }
        r += 1;
    }
    return no_fault();
}

// eig_banded gives the eigenvalues and the eigenvectors of a real
// symmetric banded matrix. The eigenvalues rise. The matrix comes in
// the banded form of SciPy, in the upper form or with lower true in the
// lower form. The method expands the matrix, so the side stays at 2048
// or below.
fn eig_banded(ab, lower: bool = false, select: const u8[:] = "a", vl: f64 = 0.0, vu: f64 = 0.0, il: i64 = 0, iu: i64 = 0) -> Arr, Arr, Fault {
    var d: f64[>..] = [];
    let f = banded_dense(ab, lower, d);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let n = ab.shape[1];
    let w, v, g = eigh(mat2(copy(d), n, n));
    if g.kind != F_NONE { return empty_arr(), empty_arr(), g; }
    var keep: i64[>..] = [];
    let fs = band_select(select, vl, vu, il, iu, w, n, keep);
    if fs.kind != F_NONE { return empty_arr(), empty_arr(), fs; }
    let m = keep.len;
    var out: f64[>..] = [];
    var vec: f64[>..] = [];
    var i: i64 = 0;
    while i < m { out.push(w.data[keep[i]]); i += 1; }
    var r: i64 = 0;
    while r < n {
        i = 0;
        while i < m { vec.push(v.data[r * n + keep[i]]); i += 1; }
        r += 1;
    }
    var sw: i64[..8] = [];
    sw.push(m);
    var sv: i64[..8] = [];
    sv.push(n);
    sv.push(m);
    return Arr { ndim: 1, shape: sw, data: out }, Arr { ndim: 2, shape: sv, data: vec }, no_fault();
}

// band_select fills keep with the indexes of the values that the
// select holds. The range v holds the values in (vl, vu]. The range i
// holds the indexes il to iu, both ends in, counted from zero.
fn band_select(select: const u8[:], vl: f64, vu: f64, il: i64, iu: i64, w, n: i64, keep: i64[>..]&) -> Fault {
    var i: i64 = 0;
    if select == "a" {
        while i < n { keep.push(i); i += 1; }
        return no_fault();
    }
    if select == "v" {
        if vu < vl {
            return fault(F_RANGE, "eigvals_banded", "the range of the values runs backward");
        }
        while i < n {
            if w.data[i] > vl && w.data[i] <= vu { keep.push(i); }
            i += 1;
        }
        if keep.len == 0 {
            return fault(F_RANGE, "eigvals_banded", "the range of the values holds no value");
        }
        return no_fault();
    }
    if select == "i" {
        if il > iu || iu >= n || il < 0 {
            return fault(F_RANGE, "eigvals_banded", "the range of the indexes stands outside the matrix");
        }
        while il + i <= iu { keep.push(il + i); i += 1; }
        return no_fault();
    }
    return fault(F_VALUE, "eigvals_banded", "the select holds a name that is not a, v or i");
}

// eigvals_banded gives the eigenvalues of a real symmetric banded
// matrix, in rising order. The select takes a for every value, v for
// the values in (vl, vu], and i for the indexes il to iu, both ends
// in, counted from zero.
fn eigvals_banded(ab, lower: bool = false, select: const u8[:] = "a", vl: f64 = 0.0, vu: f64 = 0.0, il: i64 = 0, iu: i64 = 0) -> Arr, Fault {
    var d: f64[>..] = [];
    let f = banded_dense(ab, lower, d);
    if f.kind != F_NONE { return empty_arr(), f; }
    let n = ab.shape[1];
    let w, g = eigvalsh(mat2(copy(d), n, n));
    if g.kind != F_NONE { return empty_arr(), g; }
    var keep: i64[>..] = [];
    let fs = band_select(select, vl, vu, il, iu, w, n, keep);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < keep.len {
        out.push(w.data[keep[i]]);
        i += 1;
    }
    var sw: i64[..8] = [];
    sw.push(keep.len);
    return Arr { ndim: 1, shape: sw, data: out }, no_fault();
}

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
