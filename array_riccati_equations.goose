// The continuous and discrete algebraic Riccati equations, and the pencil form.
//
namespace scientific;

import mathshim;

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

// ar_dare_ok tells whether y2 solves the discrete equation. The
// residual must stay small against the scale of q and y2.
fn ar_dare_ok(y2: f64[:], a, b, q, r, e, s, n: i64, m: i64) -> bool {
    var res: f64 = 0.0;
    if !ar_res_dare(y2, a.data, b.data, q.data, r.data, e.data, s.data, n, m, e.data.len > 0, s.data.len > 0, res) {
        return false;
    }
    let scale = ar_norm1(q.data, n, n) * ar_norm1(y2, n, n);
    let sc = (if scale > 1.0 { scale } else { 1.0 });
    return res <= 1.0e-9 * sc;
}

// ar_dare_doubling runs the structure preserving doubling on the
// reduced data. It fills out with the solution and gives false when
// the doubling meets a singular matrix, diverges or does not converge.
fn ar_dare_doubling(a2: f64[:], g0: f64[:], h0: f64[:], ei: f64[:], n: i64, out: f64[>..]&) -> bool {
    var aa: f64[>..] = [];
    var g: f64[>..] = [];
    var h: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { aa.push(a2[i]); g.push(g0[i]); h.push(h0[i]); i += 1; }
    var wi: f64[>..] = [];
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
        if !ar_inv(w, n, wi, ld) { return false; }
        let at = ar_tr(aa, n, n);
        let awi = ar_mul(aa, n, n, wi, n);
        let an = ar_mul(awi, n, n, aa, n);
        let gn = ar_mul(ar_mul(awi, n, n, g, n), n, n, at, n);
        let hn = ar_mul(ar_mul(ar_mul(at, n, n, h, n), n, n, wi, n), n, n, aa, n);
        hn2.clear();
        i = 0;
        while i < n * n { hn2.push(h[i] + hn[i]); i += 1; }
        if !ar_finite(hn2) { return false; }
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
    if !converged { return false; }
    out.clear();
    let y = ar_finish(h, ei, n);
    i = 0;
    while i < n * n { out.push(y[i]); i += 1; }
    return true;
}

// ar_pencil solves the discrete equation with the extended symplectic
// pencil of side 2n + m. It needs no inverse of r. The stable
// deflating subspace comes from the first n Schur vectors, which the
// ordered generalized Schur form puts there. The pencil holds m
// infinite eigenvalues, which lie outside the unit circle.
fn ar_pencil(a, b, q, r, e, s, n: i64, m: i64, tag: const u8[:], out: f64[>..]&) -> Fault {
    let side = 2 * n + m;
    var h: f64[>..] = [];
    var j: f64[>..] = [];
    var i: i64 = 0;
    while i < side * side { h.push(0.0); j.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var c: i64 = 0;
        while c < n {
            h[i * side + c] = a.data[i * n + c];
            h[(n + i) * side + c] = -q.data[i * n + c];
            var ev: f64 = (if i == c { 1.0 } else { 0.0 });
            if e.data.len > 0 { ev = e.data[i * n + c]; }
            j[i * side + c] = ev;
            h[(n + c) * side + n + i] = ev;
            j[(n + i) * side + n + c] = a.data[c * n + i];
            c += 1;
        }
        c = 0;
        while c < m {
            h[i * side + 2 * n + c] = b.data[i * m + c];
            var sv: f64 = 0.0;
            if s.data.len > 0 { sv = s.data[i * m + c]; }
            h[(n + i) * side + 2 * n + c] = -sv;
            h[(2 * n + c) * side + i] = sv;
            c += 1;
        }
        i += 1;
    }
    i = 0;
    while i < m {
        var c: i64 = 0;
        while c < n {
            // the rows of -b^T in the pencil J
            j[(2 * n + i) * side + n + c] = -b.data[c * m + i];
            c += 1;
        }
        c = 0;
        while c < m {
            h[(2 * n + i) * side + 2 * n + c] = r.data[i * m + c];
            c += 1;
        }
        i += 1;
    }
    let aa, bb, alpha, beta, qq, zz, f = ordqz(mat2(copy(h), side, side), mat2(copy(j), side, side), "iuc");
    if f.kind != F_NONE {
        return fault(F_VALUE, tag, "the pencil of the equation has no ordered Schur form");
    }
    var inside: i64 = 0;
    i = 0;
    while i < side {
        let re = alpha.data[2 * i];
        let im = alpha.data[2 * i + 1];
        if re * re + im * im < beta.data[i] * beta.data[i] { inside += 1; }
        i += 1;
    }
    if inside != n {
        return fault(F_VALUE, tag, "the equation has no solution of the stabilizing kind near this data");
    }
    var u0: f64[>..] = [];
    var u1: f64[>..] = [];
    i = 0;
    while i < n {
        var c: i64 = 0;
        while c < n {
            u0.push(zz.data[i * side + c]);
            u1.push(zz.data[(n + i) * side + c]);
            c += 1;
        }
        i += 1;
    }
    // With a matrix e the costate relates to e times the first block.
    if e.data.len > 0 {
        let eu = ar_mul(e.data, n, n, u0, n);
        u0 = copy(eu);
    }
    var u0i: f64[>..] = [];
    var ld: f64 = 0.0;
    if !ar_inv(u0, n, u0i, ld) {
        return fault(F_VALUE, tag, "the equation has no solution of the stabilizing kind near this data");
    }
    let x = ar_mul(u1, n, n, u0i, n);
    if !ar_finite(x) {
        return fault(F_VALUE, tag, "the equation has no solution of the stabilizing kind near this data");
    }
    out.clear();
    i = 0;
    while i < n {
        var c: i64 = 0;
        while c < n { out.push(0.5 * (x[i * n + c] + x[c * n + i])); c += 1; }
        i += 1;
    }
    return no_fault();
}

// ar_pencil_solve runs the pencil and the residual test.
fn ar_pencil_solve(a, b, q, r, e, s, n: i64, m: i64, tag: const u8[:]) -> Arr, Fault {
    var y: f64[>..] = [];
    let f = ar_pencil(a, b, q, r, e, s, n, m, tag, y);
    if f.kind != F_NONE { return empty_arr(), f; }
    if !ar_dare_ok(y, a, b, q, r, e, s, n, m) {
        return empty_arr(), fault(F_VALUE, tag, "the equation has no solution of the stabilizing kind near this data");
    }
    return mat2(copy(y), n, n), no_fault();
}

// solve_discrete_are solves the discrete algebraic Riccati equation
// a^T x a - e^T x e - (a^T x b + s) (r + b^T x b)^-1 (b^T x a + s^T)
// + q = 0 for the stabilizing solution x. The matrices e and s hold no
// elements when the call gives none, and then e is the identity and s
// is zero. The matrices q and r must be symmetric. The matrix e must
// be nonsingular. The matrix r may be singular when r plus the product
// of the transposed b, x and b is nonsingular. The call tries the
// doubling first, which needs the inverse of r. When r is singular or
// the doubling fails, it solves the extended symplectic pencil with
// the ordered generalized Schur form, like SciPy. The call gives a
// fault when the residual of the answer stays above its scale by more
// than a small part, which happens when the equation has no solution
// of the stabilizing kind.
fn solve_discrete_are(a, b, q, r, e, s) -> Arr, Fault {
    let tag = "solve_discrete_are";
    let f = ar_args(a, b, q, r, e, s, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let n = a.shape[0];
    let m = b.shape[1];
    if e.data.len > 0 {
        var ei0: f64[>..] = [];
        var ld: f64 = 0.0;
        if !ar_inv(e.data, n, ei0, ld) {
            return empty_arr(), fault(F_VALUE, tag, "the matrix e is singular");
        }
    }
    var a2: f64[>..] = [];
    var g: f64[>..] = [];
    var h: f64[>..] = [];
    var ei: f64[>..] = [];
    var ri: f64[>..] = [];
    if ar_reduce(a.data, b.data, q.data, r.data, e.data, s.data, n, m, e.data.len > 0, s.data.len > 0, a2, g, h, ei, ri) {
        var y: f64[>..] = [];
        if ar_dare_doubling(a2, g, h, ei, n, y) && ar_dare_ok(y, a, b, q, r, e, s, n, m) {
            return mat2(copy(y), n, n), no_fault();
        }
    }
    return ar_pencil_solve(a, b, q, r, e, s, n, m, tag);
}

// solve_discrete_are_pencil solves the same equation with the pencil
// alone. It needs no inverse of r and gives the same matrix.
fn solve_discrete_are_pencil(a, b, q, r, e, s) -> Arr, Fault {
    let tag = "solve_discrete_are_pencil";
    let f = ar_args(a, b, q, r, e, s, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    return ar_pencil_solve(a, b, q, r, e, s, a.shape[0], b.shape[1], tag);
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

// solve_discrete_are_pencil with four operands solves the equation
// with e as the identity and s as zero.
fn solve_discrete_are_pencil(a, b, q, r) -> Arr, Fault {
    let none = empty_arr();
    return solve_discrete_are_pencil(a, b, q, r, none, none);
}
