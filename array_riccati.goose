// The algebraic Riccati equations, as solve_continuous_are and
// solve_discrete_are of scipy.linalg. The call gives the stabilizing
// solution. The method reduces the matrices e and s with exact
// algebraic shifts, then uses the matrix sign function for the
// continuous equation and the structure preserving doubling for the
// discrete equation. SciPy uses the generalized Schur form of the
// pencil, which the package does not hold. Both methods give the same
// matrix, and this package checks the residual of the answer in the
// original equation before it returns it.
namespace scientific;

import mathshim;

// The largest side of a matrix that the equations take. The method of
// the continuous equation works on a matrix of twice the side.
const ARE_MAX_SIDE: i64 = 128;

// ar_mul gives the product of a matrix of ar rows and ac columns with
// a matrix of ac rows and bc columns.
fn ar_mul(a: f64[:], ar: i64, ac: i64, b: f64[:], bc: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < ar * bc { out.push(0.0); i += 1; }
    i = 0;
    while i < ar {
        var p: i64 = 0;
        while p < ac {
            let v = a[i * ac + p];
            if v != 0.0 {
                var j: i64 = 0;
                while j < bc {
                    out[i * bc + j] += v * b[p * bc + j];
                    j += 1;
                }
            }
            p += 1;
        }
        i += 1;
    }
    return out;
}

// ar_tr gives the transpose of a matrix of r rows and c columns.
fn ar_tr(a: f64[:], r: i64, c: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < r * c { out.push(0.0); i += 1; }
    i = 0;
    while i < r {
        var j: i64 = 0;
        while j < c {
            out[j * r + i] = a[i * c + j];
            j += 1;
        }
        i += 1;
    }
    return out;
}

// ar_inv gives the inverse of a square matrix by the elimination of
// Gauss and Jordan with a partial pivot. It gives false when the
// matrix is singular. The value logdet holds the logarithm of the
// absolute value of the determinant.
fn ar_inv(a: f64[:], n: i64, out: f64[>..]&, logdet: f64&) -> bool {
    var w: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { w.push(a[i]); i += 1; }
    out.clear();
    i = 0;
    while i < n * n { out.push(0.0); i += 1; }
    i = 0;
    while i < n { out[i * n + i] = 1.0; i += 1; }
    var ld: f64 = 0.0;
    var col: i64 = 0;
    while col < n {
        var best = col;
        var bv: f64 = mathshim::root(w[col * n + col] * w[col * n + col]);
        var r: i64 = col + 1;
        while r < n {
            let v = mathshim::root(w[r * n + col] * w[r * n + col]);
            if v > bv { bv = v; best = r; }
            r += 1;
        }
        if !(bv > 0.0) || !(bv - bv == 0.0) { return false; }
        if best != col {
            var c: i64 = 0;
            while c < n {
                let t1 = w[col * n + c];
                w[col * n + c] = w[best * n + c];
                w[best * n + c] = t1;
                let t2 = out[col * n + c];
                out[col * n + c] = out[best * n + c];
                out[best * n + c] = t2;
                c += 1;
            }
        }
        let piv = w[col * n + col];
        ld += mathshim::ln(mathshim::root(piv * piv));
        var c: i64 = 0;
        while c < n {
            w[col * n + c] = w[col * n + c] / piv;
            out[col * n + c] = out[col * n + c] / piv;
            c += 1;
        }
        r = 0;
        while r < n {
            if r != col {
                let f = w[r * n + col];
                if f != 0.0 {
                    c = 0;
                    while c < n {
                        w[r * n + c] -= f * w[col * n + c];
                        out[r * n + c] -= f * out[col * n + c];
                        c += 1;
                    }
                }
            }
            r += 1;
        }
        col += 1;
    }
    logdet = ld;
    return true;
}

// ar_norm1 gives the largest column sum of a matrix.
fn ar_norm1(a: f64[:], r: i64, c: i64) -> f64 {
    var best: f64 = 0.0;
    var j: i64 = 0;
    while j < c {
        var s: f64 = 0.0;
        var i: i64 = 0;
        while i < r {
            let v = a[i * c + j];
            if v < 0.0 { s -= v; } else { s += v; }
            i += 1;
        }
        if s > best { best = s; }
        j += 1;
    }
    return best;
}

// ar_anti gives the largest column sum of the antisymmetric part of a
// square matrix. A symmetric matrix gives zero.
fn ar_anti(a: f64[:], n: i64) -> f64 {
    var best: f64 = 0.0;
    var j: i64 = 0;
    while j < n {
        var s: f64 = 0.0;
        var i: i64 = 0;
        while i < n {
            let d = a[i * n + j] - a[j * n + i];
            if d < 0.0 { s -= d; } else { s += d; }
            i += 1;
        }
        if s > best { best = s; }
        j += 1;
    }
    return best;
}

// ar_finite gives true when every element of the buffer is finite.
fn ar_finite(a: f64[:]) -> bool {
    var i: i64 = 0;
    while i < a.len {
        if !(a[i] - a[i] == 0.0) { return false; }
        i += 1;
    }
    return true;
}

// ar_check checks one operand: the count of dimensions, the sides, the
// buffer and the values. The operand holds r rows and c columns when r
// is not negative.
fn ar_check(x, r: i64, c: i64, tag: const u8[:], name: const u8[:]) -> Fault {
    if x.ndim != x.shape.len {
        return fault(F_SHAPE, tag, "an operand holds a count of dimensions that does not match its shape");
    }
    if x.ndim != 2 {
        return fault(F_SHAPE, tag, "an operand is not a matrix");
    }
    var i: i64 = 0;
    while i < 2 {
        if x.shape[i] < 1 || x.shape[i] > ARE_MAX_SIDE {
            return fault(F_LIMIT, tag, "an operand holds a side that is out of the limits");
        }
        i += 1;
    }
    if x.data.len != x.shape[0] * x.shape[1] {
        return fault(F_SHAPE, tag, "an operand holds a buffer that does not match its shape");
    }
    if r >= 0 && (x.shape[0] != r || x.shape[1] != c) {
        return fault(F_SHAPE, tag, name);
    }
    if !ar_finite(x.data) {
        return fault(F_VALUE, tag, "an operand holds a value that is not finite");
    }
    return no_fault();
}

// ar_args checks the operands that both equations take. The matrices q
// and r must be symmetric, like SciPy requires.
fn ar_args(a, b, q, r, e, s, tag: const u8[:]) -> Fault {
    var f = ar_check(a, -1, 0, tag, "");
    if f.kind != F_NONE { return f; }
    if a.shape[0] != a.shape[1] {
        return fault(F_SHAPE, tag, "the matrix a is not square");
    }
    let n = a.shape[0];
    f = ar_check(b, -1, 0, tag, "");
    if f.kind != F_NONE { return f; }
    if b.shape[0] != n {
        return fault(F_SHAPE, tag, "the matrix b holds a count of rows that differs from a");
    }
    let m = b.shape[1];
    f = ar_check(q, n, n, tag, "the matrix q does not fit the matrix a");
    if f.kind != F_NONE { return f; }
    if ar_anti(q.data, n) > ar_norm1(q.data, n, n) * 1.0e-13 {
        return fault(F_VALUE, tag, "the matrix q is not symmetric");
    }
    f = ar_check(r, m, m, tag, "the matrix r does not fit the matrix b");
    if f.kind != F_NONE { return f; }
    if ar_anti(r.data, m) > ar_norm1(r.data, m, m) * 1.0e-13 {
        return fault(F_VALUE, tag, "the matrix r is not symmetric");
    }
    if e.data.len > 0 {
        f = ar_check(e, n, n, tag, "the matrix e does not fit the matrix a");
        if f.kind != F_NONE { return f; }
    }
    if s.data.len > 0 {
        f = ar_check(s, n, m, tag, "the matrix s does not fit the matrices a and b");
        if f.kind != F_NONE { return f; }
    }
    return no_fault();
}

// ar_reduce turns the data of an equation into the standard form. It
// removes the matrix e with the substitution x = e^-T y e^-1 and the
// matrix s with the shift of a and q. The call gives a2, the product g
// of b1 with the inverse of r and the transposed b1, q2, the inverse
// of e and the inverse of r. It gives false when e or r is singular.
fn ar_reduce(a: f64[:], b: f64[:], q: f64[:], r: f64[:], e: f64[:], s: f64[:], n: i64, m: i64, has_e: bool, has_s: bool, ao: f64[>..]&, go: f64[>..]&, qo: f64[>..]&, eio: f64[>..]&, rio: f64[>..]&) -> bool {
    var ld: f64 = 0.0;
    if !ar_inv(r, m, rio, ld) { return false; }
    var a1: f64[>..] = [];
    var b1: f64[>..] = [];
    var i: i64 = 0;
    if has_e {
        if !ar_inv(e, n, eio, ld) { return false; }
        a1 = ar_mul(eio, n, n, a, n);
        b1 = ar_mul(eio, n, n, b, m);
    } else {
        eio.clear();
        i = 0;
        while i < n * n { eio.push(0.0); i += 1; }
        i = 0;
        while i < n { eio[i * n + i] = 1.0; i += 1; }
        i = 0;
        while i < n * n { a1.push(a[i]); i += 1; }
        i = 0;
        while i < n * m { b1.push(b[i]); i += 1; }
    }
    ao.clear();
    qo.clear();
    go.clear();
    i = 0;
    while i < n * n { ao.push(a1[i]); qo.push(q[i]); i += 1; }
    if has_s {
        // a1 minus b1 r^-1 s^T, and q minus s r^-1 s^T.
        let st = ar_tr(s, n, m);
        let brs = ar_mul(ar_mul(b1, n, m, rio, m), n, m, st, n);
        let srs = ar_mul(ar_mul(s, n, m, rio, m), n, m, st, n);
        i = 0;
        while i < n * n {
            ao[i] -= brs[i];
            qo[i] -= srs[i];
            i += 1;
        }
    }
    let bt = ar_tr(b1, n, m);
    let gg = ar_mul(ar_mul(b1, n, m, rio, m), n, m, bt, n);
    i = 0;
    while i < n * n { go.push(gg[i]); i += 1; }
    return true;
}

// ar_finish turns the solution y of the standard form back into the
// solution x of the equation through x = e^-T y e^-1. The result is
// symmetric.
fn ar_finish(y: f64[:], ei: f64[:], n: i64) -> f64[>..] {
    let eit = ar_tr(ei, n, n);
    let t1 = ar_mul(eit, n, n, y, n);
    let x = ar_mul(t1, n, n, ei, n);
    var sym: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            sym.push(0.5 * (x[i * n + j] + x[j * n + i]));
            j += 1;
        }
        i += 1;
    }
    return sym;
}

// ar_resid3 gives the largest absolute value of p minus u plus v, with
// three buffers of one length.
fn ar_resid3(p: f64[:], u: f64[:], v: f64[:]) -> f64 {
    var best: f64 = 0.0;
    var i: i64 = 0;
    while i < p.len {
        let d = p[i] - u[i] + v[i];
        if d < 0.0 { if -d > best { best = -d; } } else if d > best { best = d; }
        i += 1;
    }
    return best;
}

// ar_res_care gives the largest absolute value of the residual of the
// continuous equation at x. The matrices e and s hold no elements when
// the call gives none.
fn ar_res_care(x: f64[:], a: f64[:], b: f64[:], q: f64[:], ri: f64[:], e: f64[:], s: f64[:], n: i64, m: i64, has_e: bool, has_s: bool) -> f64 {
    var xe: f64[>..] = [];
    var et: f64[>..] = [];
    var i: i64 = 0;
    if has_e {
        xe = ar_mul(x, n, n, e, n);
        et = ar_tr(e, n, n);
    } else {
        while i < n * n { xe.push(x[i]); i += 1; }
    }
    // The terms e^T x a and a^T x e, and the product t = e^T x b + s.
    var exa: f64[>..] = [];
    var xb: f64[>..] = [];
    if has_e {
        exa = ar_mul(ar_mul(et, n, n, x, n), n, n, a, n);
        xb = ar_mul(ar_mul(et, n, n, x, n), n, n, b, m);
    } else {
        exa = ar_mul(x, n, n, a, n);
        xb = ar_mul(x, n, n, b, m);
    }
    let axe = ar_mul(ar_tr(a, n, n), n, n, xe, n);
    if has_s {
        i = 0;
        while i < n * m { xb[i] += s[i]; i += 1; }
    }
    let tt = ar_mul(ar_mul(xb, n, m, ri, m), n, m, ar_tr(xb, n, m), n);
    var lin: f64[>..] = [];
    i = 0;
    while i < n * n { lin.push(exa[i] + axe[i]); i += 1; }
    return ar_resid3(lin, tt, q);
}

// ar_res_dare gives the largest absolute value of the residual of the
// discrete equation at x. The call gives false when the matrix r plus
// the product of the transposed b, x and b is singular.
fn ar_res_dare(x: f64[:], a: f64[:], b: f64[:], q: f64[:], r: f64[:], e: f64[:], s: f64[:], n: i64, m: i64, has_e: bool, has_s: bool, out: f64&) -> bool {
    let xat = ar_mul(ar_tr(a, n, n), n, n, x, n);
    var t = ar_mul(xat, n, n, b, m);
    let axa = ar_mul(xat, n, n, a, n);
    var exe: f64[>..] = [];
    var i: i64 = 0;
    if has_e {
        exe = ar_mul(ar_mul(ar_tr(e, n, n), n, n, x, n), n, n, e, n);
    } else {
        while i < n * n { exe.push(x[i]); i += 1; }
    }
    if has_s {
        i = 0;
        while i < n * m { t[i] += s[i]; i += 1; }
    }
    var m2 = ar_mul(ar_mul(ar_tr(b, n, m), m, n, x, n), m, n, b, m);
    i = 0;
    while i < m * m { m2[i] += r[i]; i += 1; }
    var mi: f64[>..] = [];
    var ld: f64 = 0.0;
    if !ar_inv(m2, m, mi, ld) { return false; }
    let tt = ar_mul(ar_mul(t, n, m, mi, m), n, m, ar_tr(t, n, m), n);
    var lin: f64[>..] = [];
    i = 0;
    while i < n * n { lin.push(axa[i] - exe[i]); i += 1; }
    out = ar_resid3(lin, tt, q);
    return true;
}

// solve_continuous_are solves the continuous algebraic Riccati
// equation e^T x a + a^T x e - (e^T x b + s) r^-1 (b^T x e + s^T) + q
// = 0 for the stabilizing solution x. The matrices e and s hold no
// elements when the call gives none, and then e is the identity and s
// is zero. The matrices q and r must be symmetric. The matrix r and
// the matrix e must be nonsingular. The method gives a fault when the
// residual of the answer stays above its scale by more than a small
// part, which happens when the equation has no solution of the
// stabilizing kind.
fn solve_continuous_are(a, b, q, r, e, s) -> Arr, Fault {
    let tag = "solve_continuous_are";
    let f = ar_args(a, b, q, r, e, s, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let n = a.shape[0];
    let m = b.shape[1];
    var a2: f64[>..] = [];
    var g: f64[>..] = [];
    var q2: f64[>..] = [];
    var ei: f64[>..] = [];
    var ri: f64[>..] = [];
    if !ar_reduce(a.data, b.data, q.data, r.data, e.data, s.data, n, m, e.data.len > 0, s.data.len > 0, a2, g, q2, ei, ri) {
        return empty_arr(), fault(F_VALUE, tag, "the matrix r or the matrix e is singular");
    }
    // The Hamiltonian matrix of the standard form.
    let nn = 2 * n;
    var z: f64[>..] = [];
    var i: i64 = 0;
    while i < nn * nn { z.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            z[i * nn + j] = a2[i * n + j];
            z[i * nn + n + j] = -g[i * n + j];
            z[(n + i) * nn + j] = -q2[i * n + j];
            z[(n + i) * nn + n + j] = -a2[j * n + i];
            j += 1;
        }
        i += 1;
    }
    // The sign function by the scaled iteration of Newton.
    var zi: f64[>..] = [];
    var zn: f64[>..] = [];
    var converged = false;
    var it: i64 = 0;
    while it < 100 && !converged {
        var ld: f64 = 0.0;
        if !ar_inv(z, nn, zi, ld) {
            return empty_arr(), fault(F_VALUE, tag, "the iteration of the sign function meets a singular matrix");
        }
        let d = mathshim::power_e(-ld / (nn as f64));
        zn.clear();
        i = 0;
        while i < nn * nn {
            zn.push(0.5 * (d * z[i] + zi[i] / d));
            i += 1;
        }
        if !ar_finite(zn) {
            return empty_arr(), fault(F_VALUE, tag, "the iteration of the sign function diverges");
        }
        var change: f64 = 0.0;
        i = 0;
        while i < nn * nn {
            let v = zn[i] - z[i];
            change += v * v;
            i += 1;
        }
        change = mathshim::root(change);
        i = 0;
        while i < nn * nn { z[i] = zn[i]; i += 1; }
        let size = ar_norm1(z, nn, nn);
        if change <= 1.0e-14 * size { converged = true; }
        it += 1;
    }
    if !converged {
        return empty_arr(), fault(F_VALUE, tag, "the iteration of the sign function does not converge");
    }
    // The stable subspace gives the solution: x = -z21 (i - z11)^-1.
    var mi: f64[>..] = [];
    var imz: f64[>..] = [];
    i = 0;
    while i < n * n { imz.push((if i / n == i % n { 1.0 } else { 0.0 }) - z[(i / n) * nn + (i % n)]); i += 1; }
    var ld: f64 = 0.0;
    if !ar_inv(imz, n, mi, ld) {
        return empty_arr(), fault(F_VALUE, tag, "the pencil has eigenvalues too close to the imaginary axis");
    }
    var z21: f64[>..] = [];
    i = 0;
    while i < n * n { z21.push(z[(n + i / n) * nn + (i % n)]); i += 1; }
    let yb = ar_mul(z21, n, n, mi, n);
    var y: f64[>..] = [];
    i = 0;
    while i < n * n { y.push(-yb[i]); i += 1; }
    let y2 = ar_finish(y, ei, n);
    // The residual of the answer in the original equation.
    let res = ar_res_care(y2, a.data, b.data, q.data, ri, e.data, s.data, n, m, e.data.len > 0, s.data.len > 0);
    let scale = ar_norm1(q.data, n, n) * ar_norm1(y2, n, n);
    let sc = (if scale > 1.0 { scale } else { 1.0 });
    if res > 1.0e-9 * sc {
        return empty_arr(), fault(F_VALUE, tag, "the equation has no solution of the stabilizing kind near this data");
    }
    var out: f64[>..] = [];
    i = 0;
    while i < n * n { out.push(y2[i]); i += 1; }
    return mat2(copy(out), n, n), no_fault();
}

// solve_discrete_are solves the discrete algebraic Riccati equation
// a^T x a - e^T x e - (a^T x b + s) (r + b^T x b)^-1 (b^T x a + s^T)
// + q = 0 for the stabilizing solution x. The matrices e and s hold no
// elements when the call gives none, and then e is the identity and s
// is zero. The matrices q and r must be symmetric. The matrix r, the
// matrix e and the matrix r plus the product of the transposed b, x
// and b must be nonsingular. The method gives a fault when the
// residual of the answer stays above its scale by more than a small
// part, which happens when the equation has no solution of the
// stabilizing kind.
fn solve_discrete_are(a, b, q, r, e, s) -> Arr, Fault {
    let tag = "solve_discrete_are";
    let f = ar_args(a, b, q, r, e, s, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let n = a.shape[0];
    let m = b.shape[1];
    var a2: f64[>..] = [];
    var g: f64[>..] = [];
    var h: f64[>..] = [];
    var ei: f64[>..] = [];
    var ri: f64[>..] = [];
    if !ar_reduce(a.data, b.data, q.data, r.data, e.data, s.data, n, m, e.data.len > 0, s.data.len > 0, a2, g, h, ei, ri) {
        return empty_arr(), fault(F_VALUE, tag, "the matrix r or the matrix e is singular");
    }
    // The structure preserving doubling.
    var aa: f64[>..] = [];
    var at: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { aa.push(a2[i]); i += 1; }
    var wi: f64[>..] = [];
    var an: f64[>..] = [];
    var gn: f64[>..] = [];
    var hn: f64[>..] = [];
    var hn2: f64[>..] = [];
    var converged = false;
    var it: i64 = 0;
    while it < 100 && !converged {
        let gh = ar_mul(g, n, n, h, n);
        var w: f64[>..] = [];
        i = 0;
        while i < n * n {
            w.push(gh[i] + (if i / n == i % n { 1.0 } else { 0.0 }));
            i += 1;
        }
        var ld: f64 = 0.0;
        if !ar_inv(w, n, wi, ld) {
            return empty_arr(), fault(F_VALUE, tag, "the doubling meets a singular matrix");
        }
        at = ar_tr(aa, n, n);
        let awi = ar_mul(aa, n, n, wi, n);
        an = ar_mul(awi, n, n, aa, n);
        gn = ar_mul(ar_mul(awi, n, n, g, n), n, n, at, n);
        let hw = ar_mul(ar_mul(at, n, n, h, n), n, n, wi, n);
        hn = ar_mul(hw, n, n, aa, n);
        hn2.clear();
        i = 0;
        while i < n * n { hn2.push(h[i] + hn[i]); i += 1; }
        if !ar_finite(hn2) {
            return empty_arr(), fault(F_VALUE, tag, "the doubling diverges");
        }
        let change = ar_norm1(hn, n, n);
        let size = ar_norm1(hn2, n, n);
        i = 0;
        while i < n * n {
            h[i] = hn2[i];
            aa[i] = an[i];
            g[i] = g[i] + gn[i];
            i += 1;
        }
        if change <= 1.0e-14 * (if size > 1.0 { size } else { 1.0 }) { converged = true; }
        it += 1;
    }
    if !converged {
        return empty_arr(), fault(F_VALUE, tag, "the doubling does not converge");
    }
    let y2 = ar_finish(h, ei, n);
    // The residual of the answer in the original equation.
    var res: f64 = 0.0;
    if !ar_res_dare(y2, a.data, b.data, q.data, r.data, e.data, s.data, n, m, e.data.len > 0, s.data.len > 0, res) {
        return empty_arr(), fault(F_VALUE, tag, "the matrix r plus the product of the transposed b, x and b is singular");
    }
    let scale = ar_norm1(q.data, n, n) * ar_norm1(y2, n, n);
    let sc = (if scale > 1.0 { scale } else { 1.0 });
    if res > 1.0e-9 * sc {
        return empty_arr(), fault(F_VALUE, tag, "the equation has no solution of the stabilizing kind near this data");
    }
    var out: f64[>..] = [];
    i = 0;
    while i < n * n { out.push(y2[i]); i += 1; }
    return mat2(copy(out), n, n), no_fault();
}

// solve_continuous_are with four operands solves the equation with e
// as the identity and s as zero.
fn solve_continuous_are(a, b, q, r) -> Arr, Fault {
    let none = empty_arr();
    return solve_continuous_are(a, b, q, r, none, none);
}

// solve_discrete_are with four operands solves the equation with e as
// the identity and s as zero.
fn solve_discrete_are(a, b, q, r) -> Arr, Fault {
    let none = empty_arr();
    return solve_discrete_are(a, b, q, r, none, none);
}
