// The biconjugate gradient solvers.
//
// The four methods solve a square system with a sparse matrix. The
// method of the biconjugate gradient and the method of the quasi
// minimal residual use the product of the transposed matrix. The
// method of the conjugate gradient squared and the transpose free
// method use only the product of the matrix.
//
// The methods follow the updates of SciPy, so the solution matches. The
// path of the iteration may differ.
namespace scientific;

// The smallest number that the methods of SciPy treat as a breakdown.
const KRYLOV_EPS: f64 = 2.220446049250313e-16;

// csr_matvec_t gives the product of the transposed matrix and a vector.
// The result holds one value per row of the matrix.
fn csr_matvec_t(s, x: f64[:]) -> Arr, Fault {
    if x.len != s.rows {
        return empty_arr(), fault(F_SHAPE, "csr_matvec_t", "the vector does not hold one value per row");
    }
    var out: f64[>..] = nd_zeros(s.cols);
    var i: i64 = 0;
    while i < s.rows {
        var k = csr_ptr(s, i);
        let e = csr_ptr(s, i + 1);
        while k < e {
            let j = csr_col(s, k);
            out[j] = out[j] + csr_val(s, k) * x[i];
            k += 1;
        }
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// krylov_check checks the operands of a solver of the sparse matrices.
fn krylov_check(s, b: f64[:], rtol: f64, tag: const u8[:]) -> Fault {
    if s.rows != s.cols || b.len != s.rows || b.len < 1 {
        return fault(F_SHAPE, tag, "the matrix is not square or the right side does not fit");
    }
    if !(rtol >= 0.0) || rtol - rtol != 0.0 {
        return fault(F_VALUE, tag, "the tolerance is negative or not finite");
    }
    // A norm that is not finite makes the first test pass, so the solver
    // would return zero as a solution. An overflow of the norm does the same.
    let bn = krylov_norm(b);
    if bn - bn != 0.0 {
        return fault(F_VALUE, tag, "the right side holds a value that is not finite or too large");
    }
    return no_fault();
}

// bicg solves a x = b for a square sparse matrix by the method of the
// biconjugate gradient, from a start of zero. It gives the solution and
// a count: zero when the residual fell below rtol times the size of b,
// and the count of turns when maxiter ran out. A breakdown is a fault.
fn bicg(s, b: f64[:], rtol: f64 = 1.0e-5, maxiter: i64 = 0) -> Arr, i64, Fault {
    let c = krylov_check(s, b, rtol, "bicg");
    if c.kind != F_NONE { return empty_arr(), 0, c; }
    let n = b.len;
    var limit = maxiter;
    if limit < 1 { limit = 10 * n; }
    let bn = krylov_norm(b);
    if bn == 0.0 { return line_arr(copy_slice(b)), 0, no_fault(); }
    var x: f64[>..] = [];
    var r: f64[>..] = [];
    var rt: f64[>..] = [];
    var p: f64[>..] = [];
    var pt: f64[>..] = [];
    var i: i64 = 0;
    while i < n { x.push(0.0); r.push(b[i]); rt.push(b[i]); p.push(0.0); pt.push(0.0); i += 1; }
    var rho_prev: f64 = 0.0;
    var turn: i64 = 0;
    while turn < limit {
        if krylov_norm(r) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
        var rho: f64 = 0.0;
        i = 0;
        while i < n { rho += rt[i] * r[i]; i += 1; }
        if mathshim::abs_of(rho) < KRYLOV_EPS * KRYLOV_EPS {
            return empty_arr(), 0, fault(F_VALUE, "bicg", "the method broke down");
        }
        if turn > 0 {
            let beta = rho / rho_prev;
            i = 0;
            while i < n { p[i] = beta * p[i] + r[i]; pt[i] = beta * pt[i] + rt[i]; i += 1; }
        } else {
            i = 0;
            while i < n { p[i] = r[i]; pt[i] = rt[i]; i += 1; }
        }
        let q, fq = csr_matvec(s, p);
        if fq.kind != F_NONE { return empty_arr(), 0, fq; }
        let qt, ft = csr_matvec_t(s, pt);
        if ft.kind != F_NONE { return empty_arr(), 0, ft; }
        var rv: f64 = 0.0;
        i = 0;
        while i < n { rv += pt[i] * q.data[i]; i += 1; }
        if rv == 0.0 {
            return empty_arr(), 0, fault(F_VALUE, "bicg", "the method broke down");
        }
        let alpha = rho / rv;
        i = 0;
        while i < n {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * q.data[i];
            rt[i] = rt[i] - alpha * qt.data[i];
            i += 1;
        }
        rho_prev = rho;
        turn += 1;
    }
    if krylov_norm(r) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
    return line_arr(copy(x)), turn, no_fault();
}

// cgs solves a x = b for a square sparse matrix by the method of the
// conjugate gradient squared, from a start of zero. It gives the
// solution and a count: zero when the residual fell below rtol times the
// size of b, and the count of turns when maxiter ran out. A breakdown is
// a fault.
fn cgs(s, b: f64[:], rtol: f64 = 1.0e-5, maxiter: i64 = 0) -> Arr, i64, Fault {
    let c = krylov_check(s, b, rtol, "cgs");
    if c.kind != F_NONE { return empty_arr(), 0, c; }
    let n = b.len;
    var limit = maxiter;
    if limit < 1 { limit = 10 * n; }
    let bn = krylov_norm(b);
    if bn == 0.0 { return line_arr(copy_slice(b)), 0, no_fault(); }
    var x: f64[>..] = [];
    var r: f64[>..] = [];
    var p: f64[>..] = [];
    var u: f64[>..] = [];
    var q: f64[>..] = [];
    var i: i64 = 0;
    while i < n { x.push(0.0); r.push(b[i]); p.push(0.0); u.push(0.0); q.push(0.0); i += 1; }
    var rho_prev: f64 = 0.0;
    var turn: i64 = 0;
    while turn < limit {
        if krylov_norm(r) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
        var rho: f64 = 0.0;
        i = 0;
        while i < n { rho += b[i] * r[i]; i += 1; }
        if mathshim::abs_of(rho) < KRYLOV_EPS * KRYLOV_EPS {
            return empty_arr(), 0, fault(F_VALUE, "cgs", "the method broke down");
        }
        if turn > 0 {
            let beta = rho / rho_prev;
            i = 0;
            while i < n { u[i] = r[i] + beta * q[i]; i += 1; }
            i = 0;
            while i < n { p[i] = beta * p[i] + q[i]; p[i] = beta * p[i] + u[i]; i += 1; }
        } else {
            i = 0;
            while i < n { p[i] = r[i]; u[i] = r[i]; i += 1; }
        }
        let v, fv = csr_matvec(s, p);
        if fv.kind != F_NONE { return empty_arr(), 0, fv; }
        var rv: f64 = 0.0;
        i = 0;
        while i < n { rv += b[i] * v.data[i]; i += 1; }
        if rv == 0.0 {
            return empty_arr(), 0, fault(F_VALUE, "cgs", "the method broke down");
        }
        let alpha = rho / rv;
        i = 0;
        while i < n { q[i] = u[i] - alpha * v.data[i]; i += 1; }
        i = 0;
        while i < n { x[i] = x[i] + alpha * (u[i] + q[i]); i += 1; }
        let mx, fm = csr_matvec(s, x);
        if fm.kind != F_NONE { return empty_arr(), 0, fm; }
        i = 0;
        while i < n { r[i] = b[i] - mx.data[i]; i += 1; }
        rho_prev = rho;
        turn += 1;
    }
    if krylov_norm(r) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
    return line_arr(copy(x)), turn, no_fault();
}

// qmr solves a x = b for a square sparse matrix by the method of the
// quasi minimal residual, from a start of zero. It gives the solution
// and a count: zero when the residual fell below rtol times the size of
// b, and the count of turns when maxiter ran out. A breakdown is a
// fault.
fn qmr(s, b: f64[:], rtol: f64 = 1.0e-5, maxiter: i64 = 0) -> Arr, i64, Fault {
    let c = krylov_check(s, b, rtol, "qmr");
    if c.kind != F_NONE { return empty_arr(), 0, c; }
    let n = b.len;
    var limit = maxiter;
    if limit < 1 { limit = 10 * n; }
    let bn = krylov_norm(b);
    if bn == 0.0 { return line_arr(copy_slice(b)), 0, no_fault(); }
    var x: f64[>..] = [];
    var r: f64[>..] = [];
    var p: f64[>..] = [];
    var q: f64[>..] = [];
    var d: f64[>..] = [];
    var sm: f64[>..] = [];
    var v: f64[>..] = [];
    var w: f64[>..] = [];
    var vt: f64[>..] = [];
    var wt: f64[>..] = [];
    var pt: f64[>..] = [];
    var y: f64[>..] = [];
    var z: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        x.push(0.0); r.push(b[i]); p.push(0.0); q.push(0.0); d.push(0.0); sm.push(0.0);
        v.push(0.0); w.push(0.0); vt.push(r[i]); wt.push(r[i]); pt.push(0.0);
        y.push(r[i]); z.push(r[i]);
        i += 1;
    }
    var rho = krylov_norm(y);
    var xi = krylov_norm(z);
    var gamma: f64 = 1.0;
    var eta: f64 = -1.0;
    var theta: f64 = 0.0;
    var epsilon: f64 = 0.0;
    var turn: i64 = 0;
    while turn < limit {
        if krylov_norm(r) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
        if mathshim::abs_of(rho) < KRYLOV_EPS {
            return empty_arr(), 0, fault(F_VALUE, "qmr", "the method broke down");
        }
        if mathshim::abs_of(xi) < KRYLOV_EPS {
            return empty_arr(), 0, fault(F_VALUE, "qmr", "the method broke down");
        }
        i = 0;
        while i < n { v[i] = vt[i] / rho; y[i] = y[i] / rho; w[i] = wt[i] / xi; z[i] = z[i] / xi; i += 1; }
        var delta: f64 = 0.0;
        i = 0;
        while i < n { delta += z[i] * y[i]; i += 1; }
        if mathshim::abs_of(delta) < KRYLOV_EPS {
            return empty_arr(), 0, fault(F_VALUE, "qmr", "the method broke down");
        }
        if turn > 0 {
            let f1 = xi * delta / epsilon;
            let f2 = rho * delta / epsilon;
            i = 0;
            while i < n { p[i] = y[i] - f1 * p[i]; q[i] = z[i] - f2 * q[i]; i += 1; }
        } else {
            i = 0;
            while i < n { p[i] = y[i]; q[i] = z[i]; i += 1; }
        }
        let pv, fpv = csr_matvec(s, p);
        if fpv.kind != F_NONE { return empty_arr(), 0, fpv; }
        i = 0;
        while i < n { pt[i] = pv.data[i]; i += 1; }
        epsilon = 0.0;
        i = 0;
        while i < n { epsilon += q[i] * pt[i]; i += 1; }
        if mathshim::abs_of(epsilon) < KRYLOV_EPS {
            return empty_arr(), 0, fault(F_VALUE, "qmr", "the method broke down");
        }
        let beta = epsilon / delta;
        if mathshim::abs_of(beta) < KRYLOV_EPS {
            return empty_arr(), 0, fault(F_VALUE, "qmr", "the method broke down");
        }
        i = 0;
        while i < n { vt[i] = pt[i] - beta * v[i]; y[i] = vt[i]; i += 1; }
        let rho_prev = rho;
        rho = krylov_norm(y);
        let qt, fqt = csr_matvec_t(s, q);
        if fqt.kind != F_NONE { return empty_arr(), 0, fqt; }
        i = 0;
        while i < n { wt[i] = -beta * w[i] + qt.data[i]; z[i] = wt[i]; i += 1; }
        xi = krylov_norm(z);
        let gamma_prev = gamma;
        let theta_prev = theta;
        theta = rho / (gamma_prev * mathshim::abs_of(beta));
        gamma = 1.0 / mathshim::root(1.0 + theta * theta);
        if mathshim::abs_of(gamma) < KRYLOV_EPS {
            return empty_arr(), 0, fault(F_VALUE, "qmr", "the method broke down");
        }
        eta = eta * (-(rho_prev / beta)) * (gamma / gamma_prev) * (gamma / gamma_prev);
        if turn > 0 {
            let f3 = (theta_prev * gamma) * (theta_prev * gamma);
            i = 0;
            while i < n { d[i] = d[i] * f3 + eta * p[i]; sm[i] = sm[i] * f3 + eta * pt[i]; i += 1; }
        } else {
            i = 0;
            while i < n { d[i] = eta * p[i]; sm[i] = eta * pt[i]; i += 1; }
        }
        i = 0;
        while i < n { x[i] = x[i] + d[i]; r[i] = r[i] - sm[i]; i += 1; }
        turn += 1;
    }
    if krylov_norm(r) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
    return line_arr(copy(x)), turn, no_fault();
}

// tfqmr solves a x = b for a square sparse matrix by the transpose free
// method of the quasi minimal residual, from a start of zero. It gives
// the solution and a count: zero when the measure of the method fell
// below rtol times the size of b, and the count of turns when maxiter
// ran out. A breakdown is a fault.
fn tfqmr(s, b: f64[:], rtol: f64 = 1.0e-5, maxiter: i64 = 0) -> Arr, i64, Fault {
    let c = krylov_check(s, b, rtol, "tfqmr");
    if c.kind != F_NONE { return empty_arr(), 0, c; }
    let n = b.len;
    var limit = maxiter;
    if limit < 1 { limit = 10 * n; }
    let bn = krylov_norm(b);
    if bn == 0.0 { return line_arr(copy_slice(b)), 0, no_fault(); }
    var x: f64[>..] = [];
    var u: f64[>..] = [];
    var w: f64[>..] = [];
    var v: f64[>..] = [];
    var uh: f64[>..] = [];
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n { x.push(0.0); u.push(b[i]); w.push(b[i]); v.push(0.0); uh.push(0.0); d.push(0.0); i += 1; }
    let pv, fpv = csr_matvec(s, b);
    if fpv.kind != F_NONE { return empty_arr(), 0, fpv; }
    i = 0;
    while i < n { v[i] = pv.data[i]; uh[i] = pv.data[i]; i += 1; }
    var rho: f64 = 0.0;
    i = 0;
    while i < n { rho += b[i] * b[i]; i += 1; }
    var rho_last = rho;
    var tau = mathshim::root(rho);
    var theta: f64 = 0.0;
    var eta: f64 = 0.0;
    var alpha: f64 = 0.0;
    var turn: i64 = 0;
    while turn < limit {
        let even = (turn % 2) == 0;
        var un: f64[>..] = [];
        if even {
            var vtr: f64 = 0.0;
            i = 0;
            while i < n { vtr += b[i] * v[i]; i += 1; }
            if vtr == 0.0 {
                return empty_arr(), 0, fault(F_VALUE, "tfqmr", "the method broke down");
            }
            alpha = rho / vtr;
            i = 0;
            while i < n { un.push(u[i] - alpha * v[i]); i += 1; }
        }
        i = 0;
        while i < n { w[i] = w[i] - alpha * uh[i]; i += 1; }
        let f3 = (theta * theta) / alpha;
        i = 0;
        while i < n { d[i] = u[i] + f3 * eta * d[i]; i += 1; }
        theta = krylov_norm(w) / tau;
        let cc = mathshim::root(1.0 / (1.0 + theta * theta));
        tau = tau * theta * cc;
        eta = cc * cc * alpha;
        i = 0;
        while i < n { x[i] = x[i] + eta * d[i]; i += 1; }
        if tau * mathshim::root((turn + 1) as f64) <= rtol * bn {
            return line_arr(copy(x)), 0, no_fault();
        }
        if !even {
            var rho2: f64 = 0.0;
            i = 0;
            while i < n { rho2 += b[i] * w[i]; i += 1; }
            rho = rho2;
            let beta = rho / rho_last;
            i = 0;
            while i < n { v[i] = beta * uh[i] + beta * beta * v[i]; u[i] = w[i] + beta * u[i]; i += 1; }
            let pu, fpu = csr_matvec(s, u);
            if fpu.kind != F_NONE { return empty_arr(), 0, fpu; }
            i = 0;
            while i < n { v[i] = v[i] + pu.data[i]; uh[i] = pu.data[i]; i += 1; }
        } else {
            let pu, fpu = csr_matvec(s, un);
            if fpu.kind != F_NONE { return empty_arr(), 0, fpu; }
            i = 0;
            while i < n { uh[i] = pu.data[i]; u[i] = un[i]; i += 1; }
            rho_last = rho;
        }
        turn += 1;
    }
    if tau * mathshim::root(turn as f64) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
    return line_arr(copy(x)), turn, no_fault();
}
