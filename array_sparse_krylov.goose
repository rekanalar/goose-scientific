// The Krylov solvers bicgstab and gmres of sparse matrices.
//
namespace scientific;

import mathshim;

// krylov_norm gives the length of a vector.
fn krylov_norm(v: f64[:]) -> f64 {
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < v.len { t += v[i] * v[i]; i += 1; }
    return mathshim::root(t);
}

// bicgstab solves a x = b for a square sparse matrix by the method of
// the stabilized biconjugate gradients, from a start of zero. It gives
// the solution and a count: zero when the residual fell below rtol times
// the size of b, and the count of turns when maxiter ran out. A
// breakdown is a fault.
fn bicgstab(s, b: f64[:], rtol: f64 = 1.0e-5, maxiter: i64 = 0) -> Arr, i64, Fault {
    if s.rows != s.cols || b.len != s.rows || b.len < 1 {
        return empty_arr(), 0, fault(F_SHAPE, "bicgstab", "the matrix is not square or the right side does not fit");
    }
    if !(rtol >= 0.0) {
        return empty_arr(), 0, fault(F_VALUE, "bicgstab", "the tolerance holds a negative value");
    }
    let n = b.len;
    var limit = maxiter;
    if limit < 1 { limit = 10 * n; }
    let bn = krylov_norm(b);
    var x: f64[>..] = [];
    var r: f64[>..] = [];
    var rh: f64[>..] = [];
    var p: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < n { x.push(0.0); r.push(b[i]); rh.push(b[i]); p.push(0.0); v.push(0.0); i += 1; }
    var rho: f64 = 1.0;
    var alpha: f64 = 1.0;
    var omega: f64 = 1.0;
    var turn: i64 = 0;
    while turn < limit {
        if krylov_norm(r) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
        var rho1: f64 = 0.0;
        i = 0;
        while i < n { rho1 += rh[i] * r[i]; i += 1; }
        if rho1 == 0.0 || omega == 0.0 {
            return empty_arr(), 0, fault(F_VALUE, "bicgstab", "the method broke down");
        }
        let beta = (rho1 / rho) * (alpha / omega);
        i = 0;
        while i < n { p[i] = r[i] + beta * (p[i] - omega * v[i]); i += 1; }
        let vv, f1 = csr_matvec(s, p);
        if f1.kind != F_NONE { return empty_arr(), 0, f1; }
        var rv: f64 = 0.0;
        i = 0;
        while i < n { v[i] = vv.data[i]; rv += rh[i] * v[i]; i += 1; }
        if rv == 0.0 { return empty_arr(), 0, fault(F_VALUE, "bicgstab", "the method broke down"); }
        alpha = rho1 / rv;
        var sv: f64[>..] = [];
        i = 0;
        while i < n { sv.push(r[i] - alpha * v[i]); i += 1; }
        if krylov_norm(sv) <= rtol * bn {
            i = 0;
            while i < n { x[i] = x[i] + alpha * p[i]; i += 1; }
            return line_arr(copy(x)), 0, no_fault();
        }
        let t, f2 = csr_matvec(s, sv);
        if f2.kind != F_NONE { return empty_arr(), 0, f2; }
        var ts: f64 = 0.0;
        var tt: f64 = 0.0;
        i = 0;
        while i < n { ts += t.data[i] * sv[i]; tt += t.data[i] * t.data[i]; i += 1; }
        if tt == 0.0 { return empty_arr(), 0, fault(F_VALUE, "bicgstab", "the method broke down"); }
        omega = ts / tt;
        i = 0;
        while i < n {
            x[i] = x[i] + alpha * p[i] + omega * sv[i];
            r[i] = sv[i] - omega * t.data[i];
            i += 1;
        }
        rho = rho1;
        turn += 1;
    }
    if krylov_norm(r) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
    return line_arr(copy(x)), turn, no_fault();
}

// gmres solves a x = b for a square sparse matrix by the method of the
// generalized minimal residual, restarted every restart turns, from a
// start of zero. It gives the solution and a count: zero when the
// residual fell below rtol times the size of b, and the count of
// restarts when maxiter ran out.
fn gmres(s, b: f64[:], rtol: f64 = 1.0e-5, restart: i64 = 20, maxiter: i64 = 0) -> Arr, i64, Fault {
    if s.rows != s.cols || b.len != s.rows || b.len < 1 {
        return empty_arr(), 0, fault(F_SHAPE, "gmres", "the matrix is not square or the right side does not fit");
    }
    if !(rtol >= 0.0) || restart < 1 {
        return empty_arr(), 0, fault(F_VALUE, "gmres", "the tolerance is negative or the restart is below one");
    }
    let n = b.len;
    var m = restart;
    if m > n { m = n; }
    var limit = maxiter;
    if limit < 1 { limit = 10 * n; }
    let bn = krylov_norm(b);
    var x: f64[>..] = [];
    var i: i64 = 0;
    while i < n { x.push(0.0); i += 1; }
    var cycle: i64 = 0;
    while cycle < limit {
        let ax, f0 = csr_matvec(s, x);
        if f0.kind != F_NONE { return empty_arr(), 0, f0; }
        var r: f64[>..] = [];
        i = 0;
        while i < n { r.push(b[i] - ax.data[i]); i += 1; }
        let beta = krylov_norm(r);
        if beta <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
        // The basis holds m + 1 vectors of n values. H holds the
        // Hessenberg matrix by columns of m + 1 values.
        var basis: f64[>..] = [];
        i = 0;
        while i < n { basis.push(r[i] / beta); i += 1; }
        var h: f64[>..] = [];
        i = 0;
        while i < (m + 1) * m { h.push(0.0); i += 1; }
        var cs: f64[>..] = [];
        var sn: f64[>..] = [];
        var g: f64[>..] = [];
        i = 0;
        while i < m + 1 { g.push(0.0); i += 1; }
        i = 0;
        while i < m { cs.push(0.0); sn.push(0.0); i += 1; }
        g[0] = beta;
        var used: i64 = 0;
        var j: i64 = 0;
        while j < m {
            var vj: f64[>..] = [];
            i = 0;
            while i < n { vj.push(basis[j * n + i]); i += 1; }
            let w0, f1 = csr_matvec(s, vj);
            if f1.kind != F_NONE { return empty_arr(), 0, f1; }
            var w: f64[>..] = [];
            i = 0;
            while i < n { w.push(w0.data[i]); i += 1; }
            var k: i64 = 0;
            while k <= j {
                var d: f64 = 0.0;
                i = 0;
                while i < n { d += w[i] * basis[k * n + i]; i += 1; }
                h[j * (m + 1) + k] = d;
                i = 0;
                while i < n { w[i] = w[i] - d * basis[k * n + i]; i += 1; }
                k += 1;
            }
            let hn = krylov_norm(w);
            h[j * (m + 1) + j + 1] = hn;
            i = 0;
            while i < n {
                if hn > 0.0 { basis.push(w[i] / hn); } else { basis.push(0.0); }
                i += 1;
            }
            // The turns of Givens keep H triangular.
            k = 0;
            while k < j {
                let t = cs[k] * h[j * (m + 1) + k] + sn[k] * h[j * (m + 1) + k + 1];
                h[j * (m + 1) + k + 1] = -sn[k] * h[j * (m + 1) + k] + cs[k] * h[j * (m + 1) + k + 1];
                h[j * (m + 1) + k] = t;
                k += 1;
            }
            let a1 = h[j * (m + 1) + j];
            let a2 = h[j * (m + 1) + j + 1];
            let den = mathshim::root(a1 * a1 + a2 * a2);
            if den == 0.0 { cs[j] = 1.0; sn[j] = 0.0; }
            else { cs[j] = a1 / den; sn[j] = a2 / den; }
            h[j * (m + 1) + j] = den;
            h[j * (m + 1) + j + 1] = 0.0;
            g[j + 1] = -sn[j] * g[j];
            g[j] = cs[j] * g[j];
            used = j + 1;
            if mathshim::abs_of(g[j + 1]) <= rtol * bn { break; }
            j += 1;
        }
        // Back substitution for the coefficients, then the update.
        var yv: f64[>..] = [];
        i = 0;
        while i < used { yv.push(0.0); i += 1; }
        i = used - 1;
        while i >= 0 {
            var t = g[i];
            var k = i + 1;
            while k < used { t -= h[k * (m + 1) + i] * yv[k]; k += 1; }
            let dd = h[i * (m + 1) + i];
            if dd == 0.0 { return empty_arr(), 0, fault(F_VALUE, "gmres", "the method broke down"); }
            yv[i] = t / dd;
            i -= 1;
        }
        i = 0;
        while i < used {
            var q: i64 = 0;
            while q < n { x[q] = x[q] + yv[i] * basis[i * n + q]; q += 1; }
            i += 1;
        }
        cycle += 1;
    }
    let ax2, f3 = csr_matvec(s, x);
    if f3.kind != F_NONE { return empty_arr(), 0, f3; }
    var rr: f64[>..] = [];
    i = 0;
    while i < n { rr.push(b[i] - ax2.data[i]); i += 1; }
    if krylov_norm(rr) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
    return line_arr(copy(x)), cycle, no_fault();
}

