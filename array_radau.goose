// The method Radau of solve_ivp.
//
// The method is the implicit Runge and Kutta method Radau IIA of the
// order five. The Newton iteration solves the system of the three
// stages through one real and one complex linear system. The Jacobian
// comes from finite differences of the function. The state of the
// method lives in plain buffers that the caller keeps from one step to
// the next.
namespace scientific;

import mathshim;

// rad_consts fills the constants of the method: the nodes, the weights
// of the error, the two eigenvalues, the matrices T and T inverse, and
// the matrix P of the interpolation. The buffer cm holds the nodes, em
// the weights of the error, mu the real eigenvalue and the two parts of
// the complex one, tm the matrix T by rows, ti the inverse by rows and
// pm the matrix P by rows.
fn rad_consts(cm: f64[>..]&, em: f64[>..]&, mu: f64[>..]&, tm: f64[>..]&, ti: f64[>..]&, pm: f64[>..]&) {
    let s6 = mathshim::root(6.0);
    cm.push((4.0 - s6) / 10.0); cm.push((4.0 + s6) / 10.0); cm.push(1.0);
    em.push((-13.0 - 7.0 * s6) / 3.0); em.push((-13.0 + 7.0 * s6) / 3.0); em.push(-1.0 / 3.0);
    let c13 = mathshim::power(3.0, 1.0 / 3.0);
    let c23 = mathshim::power(3.0, 2.0 / 3.0);
    mu.push(3.0 + c23 - c13);
    mu.push(3.0 + 0.5 * (c13 - c23));
    mu.push(-0.5 * (mathshim::power(3.0, 5.0 / 6.0) + mathshim::power(3.0, 7.0 / 6.0)));
    let t: f64[9] = [0.09443876248897524, -0.14125529502095421, 0.03002919410514742,
        0.25021312296533332, 0.20412935229379994, -0.38294211275726192,
        1.0, 1.0, 0.0];
    let tii: f64[9] = [4.17871859155190428, 0.32768282076106237, 0.52337644549944951,
        -4.17871859155190428, -0.32768282076106237, 0.47662355450055044,
        0.50287263494578682, -2.57192694985560522, 0.59603920482822492];
    let p: f64[9] = [13.0 / 3.0 + 7.0 * s6 / 3.0, -23.0 / 3.0 - 22.0 * s6 / 3.0, 10.0 / 3.0 + 5.0 * s6,
        13.0 / 3.0 - 7.0 * s6 / 3.0, -23.0 / 3.0 + 22.0 * s6 / 3.0, 10.0 / 3.0 - 5.0 * s6,
        1.0 / 3.0, -8.0 / 3.0, 10.0 / 3.0];
    var i: i64 = 0;
    while i < 9 { tm.push(t[i]); ti.push(tii[i]); pm.push(p[i]); i += 1; }
}

// rad_clu factors a complex square matrix in place with the pivots of
// the rows. The parts of the matrix lie in two buffers. It gives false
// when a pivot holds zero.
fn rad_clu(ar: f64[>..]&, ai: f64[>..]&, n: i64, piv: i64[>..]&) -> bool {
    var k: i64 = 0;
    while k < n {
        var p = k;
        var big = mathshim::abs_of(ar[k * n + k]) + mathshim::abs_of(ai[k * n + k]);
        var i: i64 = k + 1;
        while i < n {
            let v = mathshim::abs_of(ar[i * n + k]) + mathshim::abs_of(ai[i * n + k]);
            if v > big { big = v; p = i; }
            i += 1;
        }
        piv[k] = p;
        if !(big > 0.0) { return false; }
        if p != k {
            var j: i64 = 0;
            while j < n {
                let tr = ar[k * n + j]; ar[k * n + j] = ar[p * n + j]; ar[p * n + j] = tr;
                let ti = ai[k * n + j]; ai[k * n + j] = ai[p * n + j]; ai[p * n + j] = ti;
                j += 1;
            }
        }
        let cr = ar[k * n + k];
        let ci = ai[k * n + k];
        let den = cr * cr + ci * ci;
        i = k + 1;
        while i < n {
            let xr = ar[i * n + k];
            let xi = ai[i * n + k];
            let mr = (xr * cr + xi * ci) / den;
            let mi = (xi * cr - xr * ci) / den;
            ar[i * n + k] = mr;
            ai[i * n + k] = mi;
            var j: i64 = k + 1;
            while j < n {
                let yr = ar[k * n + j];
                let yi = ai[k * n + j];
                ar[i * n + j] = ar[i * n + j] - (mr * yr - mi * yi);
                ai[i * n + j] = ai[i * n + j] - (mr * yi + mi * yr);
                j += 1;
            }
            i += 1;
        }
        k += 1;
    }
    return true;
}

// rad_clu_solve solves a complex system with the factors of rad_clu.
// The right side goes in br and bi, and the answer comes out there.
fn rad_clu_solve(ar: f64[:], ai: f64[:], n: i64, piv: i64[:], br: f64[>..]&, bi: f64[>..]&) {
    var k: i64 = 0;
    while k < n {
        let p = piv[k];
        if p != k {
            let tr = br[k]; br[k] = br[p]; br[p] = tr;
            let ti = bi[k]; bi[k] = bi[p]; bi[p] = ti;
        }
        k += 1;
    }
    var i: i64 = 1;
    while i < n {
        var sr = br[i];
        var si = bi[i];
        var j: i64 = 0;
        while j < i {
            sr -= ar[i * n + j] * br[j] - ai[i * n + j] * bi[j];
            si -= ar[i * n + j] * bi[j] + ai[i * n + j] * br[j];
            j += 1;
        }
        br[i] = sr;
        bi[i] = si;
        i += 1;
    }
    i = n - 1;
    while i >= 0 {
        var sr = br[i];
        var si = bi[i];
        var j: i64 = i + 1;
        while j < n {
            sr -= ar[i * n + j] * br[j] - ai[i * n + j] * bi[j];
            si -= ar[i * n + j] * bi[j] + ai[i * n + j] * br[j];
            j += 1;
        }
        let cr = ar[i * n + i];
        let ci = ai[i * n + i];
        let den = cr * cr + ci * ci;
        br[i] = (sr * cr + si * ci) / den;
        bi[i] = (si * cr - sr * ci) / den;
        i -= 1;
    }
}

// rad_predict gives the factor by which the step changes. A negative
// old step or a negative old error selects the rule of one step.
fn rad_predict(h_abs: f64, h_old: f64, err: f64, err_old: f64) -> f64 {
    var mult: f64 = 1.0;
    if !(err_old < 0.0 || h_old < 0.0 || err == 0.0) {
        mult = h_abs / h_old * mathshim::power(err_old / err, 0.25);
    }
    if mult > 1.0 { mult = 1.0; }
    if err == 0.0 { return 1.0 / 0.0; }
    return mult * mathshim::power(err, -0.25);
}

// rad_newton solves the system of the three stages by the iteration of
// Newton. It gives whether the iteration converged. The solution Z holds
// one row of n values per stage. The turns and the last rate go in
// info.
fn rad_newton<F: fn(f64, f64[:]) -> f64[:]>(t: f64, y: f64[:], h: f64, z0: f64[:], scale: f64[:], tol: f64, n: i64, nev: i64, lur: f64[:], pivr: i64[:], lcr: f64[:], lci: f64[:], pivc: i64[:], z: f64[>..]&, info: f64[>..]&) -> bool, Fault {
    var cm: f64[>..] = [];
    var em: f64[>..] = [];
    var mu: f64[>..] = [];
    var tm: f64[>..] = [];
    var ti: f64[>..] = [];
    var pm: f64[>..] = [];
    rad_consts(cm, em, mu, tm, ti, pm);
    let mr = mu[0] / h;
    let mcr = mu[1] / h;
    let mci = mu[2] / h;
    var w: f64[>..] = [];
    var i: i64 = 0;
    var r: i64 = 0;
    z = [];
    while i < 3 * n { z.push(z0[i]); i += 1; }
    r = 0;
    while r < 3 {
        i = 0;
        while i < n {
            var s: f64 = 0.0;
            var k: i64 = 0;
            while k < 3 { s += ti[r * 3 + k] * z0[k * n + i]; k += 1; }
            w.push(s);
            i += 1;
        }
        r += 1;
    }
    var old: f64 = -1.0;
    var rate: f64 = -1.0;
    var converged = false;
    var k: i64 = 0;
    var used: i64 = 0;
    while k < 6 {
        used = k + 1;
        var fm: f64[>..] = [];
        var finite = true;
        var s: i64 = 0;
        while s < 3 {
            var yy: f64[>..] = [];
            i = 0;
            while i < n { yy.push(y[i] + z[s * n + i]); i += 1; }
            let fv = F(t + h * cm[s], copy(yy));
            if fv.len != n + nev {
                return false, fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
            }
            i = 0;
            while i < n { fm.push(fv[i]); if !(fv[i] - fv[i] == 0.0) { finite = false; } i += 1; }
            s += 1;
        }
        if !finite { break; }
        var fr: f64[>..] = [];
        var fcr: f64[>..] = [];
        var fci: f64[>..] = [];
        i = 0;
        while i < n {
            var a: f64 = 0.0;
            var b: f64 = 0.0;
            var c: f64 = 0.0;
            s = 0;
            while s < 3 {
                a += ti[s] * fm[s * n + i];
                b += ti[3 + s] * fm[s * n + i];
                c += ti[6 + s] * fm[s * n + i];
                s += 1;
            }
            fr.push(a - mr * w[i]);
            fcr.push(b - (mcr * w[n + i] - mci * w[2 * n + i]));
            fci.push(c - (mcr * w[2 * n + i] + mci * w[n + i]));
            i += 1;
        }
        bdf_lu_solve(lur, n, pivr, fr);
        rad_clu_solve(lcr, lci, n, pivc, fcr, fci);
        var sq: f64 = 0.0;
        i = 0;
        while i < n {
            let a = fr[i] / scale[i];
            let b = fcr[i] / scale[i];
            let c = fci[i] / scale[i];
            sq += a * a + b * b + c * c;
            i += 1;
        }
        let dn = mathshim::root(sq / (3 * n));
        if old >= 0.0 { rate = dn / old; }
        if rate >= 0.0 && (rate >= 1.0 || mathshim::power(rate, 6.0 - k) / (1.0 - rate) * dn > tol) { break; }
        i = 0;
        while i < n {
            w[i] = w[i] + fr[i];
            w[n + i] = w[n + i] + fcr[i];
            w[2 * n + i] = w[2 * n + i] + fci[i];
            i += 1;
        }
        r = 0;
        while r < 3 {
            i = 0;
            while i < n {
                var a: f64 = 0.0;
                var q: i64 = 0;
                while q < 3 { a += tm[r * 3 + q] * w[q * n + i]; q += 1; }
                z[r * n + i] = a;
                i += 1;
            }
            r += 1;
        }
        if dn == 0.0 || (rate >= 0.0 && rate / (1.0 - rate) * dn < tol) { converged = true; break; }
        old = dn;
        k += 1;
    }
    info[0] = used;
    info[1] = rate;
    return converged, no_fault();
}

// rad_dense gives the state of the polynomial of a step at a fraction
// of the step. The rows hold the three columns of the matrix Q.
fn rad_dense(n: i64, y: f64[:], q: f64[:], xs: f64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var p = xs;
        var a: f64 = 0.0;
        var k: i64 = 0;
        while k < 3 { a += q[k * n + i] * p; p = p * xs; k += 1; }
        out.push(y[i] + a);
        i += 1;
    }
    return out;
}

// rad_step takes one step of the method from the time t. The buffers
// hold the state between the steps: fst holds the step, the old step,
// the old error, the start of the last step and its size; ist holds
// whether the last step left a polynomial, whether the factors of the
// matrices are valid and whether the Jacobian is current; rq holds the
// three columns Q of the last step and ryo its start. The function at
// the start is in f. The new state goes in ynew, the function at the
// new point goes in fb and the new time goes in tn.
fn rad_step<F: fn(f64, f64[:]) -> f64[:]>(t: f64, t_bound: f64, dir: f64, n: i64, nev: i64, atol: f64, rtol: f64, ntol: f64, max_step: f64, y: f64[:], f: f64[:], fst: f64[>..]&, ist: i64[>..]&, rq: f64[>..]&, ryo: f64[>..]&, jm: f64[>..]&, jfac: f64[>..]&, lur: f64[>..]&, pivr: i64[>..]&, lcr: f64[>..]&, lci: f64[>..]&, pivc: i64[>..]&, ynew: f64[>..]&, fb: f64[>..]&, tn: f64[>..]&) -> Fault {
    var cm: f64[>..] = [];
    var em: f64[>..] = [];
    var mu: f64[>..] = [];
    var tm: f64[>..] = [];
    var ti: f64[>..] = [];
    var pm: f64[>..] = [];
    rad_consts(cm, em, mu, tm, ti, pm);
    var min_step = 10.0 * mathshim::abs_of(t) * 2.220446049250313e-16;
    if min_step < 5.0e-323 { min_step = 5.0e-323; }
    var h_abs = fst[0];
    var h_old = fst[1];
    var err_old = fst[2];
    if h_abs > max_step {
        h_abs = max_step; h_old = -1.0; err_old = -1.0;
    } else if h_abs < min_step {
        h_abs = min_step; h_old = -1.0; err_old = -1.0;
    }
    var lu_ok = ist[1] == 1;
    var current_jac = ist[2] == 1;
    var rejected = false;
    var accepted = false;
    var t_new: f64 = t;
    var z: f64[>..] = [];
    var info: f64[>..] = [0.0, -1.0];
    var error_norm: f64 = 0.0;
    var safety: f64 = 0.0;
    var i: i64 = 0;
    while !accepted {
        if h_abs < min_step {
            return fault(F_VALUE, "solve_ivp", "the step fell below the smallest step");
        }
        var h = h_abs * dir;
        t_new = t + h;
        if dir * (t_new - t_bound) > 0.0 { t_new = t_bound; }
        h = t_new - t;
        h_abs = mathshim::abs_of(h);
        var z0: f64[>..] = [];
        var s: i64 = 0;
        while s < 3 {
            i = 0;
            while i < n {
                if ist[0] == 0 {
                    z0.push(0.0);
                } else {
                    let ts = t + h * cm[s];
                    let xs = (ts - fst[3]) / fst[4];
                    let yv = rad_dense(n, ryo, rq, xs);
                    z0.push(yv[i] - y[i]);
                }
                i += 1;
            }
            s += 1;
        }
        var scale: f64[>..] = [];
        i = 0;
        while i < n { scale.push(atol + mathshim::abs_of(y[i]) * rtol); i += 1; }
        var converged = false;
        var singular = false;
        while !converged {
            if !lu_ok {
                var q: i64 = 0;
                while q < n * n {
                    lur[q] = 0.0 - jm[q];
                    lcr[q] = 0.0 - jm[q];
                    lci[q] = 0.0;
                    q += 1;
                }
                q = 0;
                while q < n {
                    lur[q * n + q] = lur[q * n + q] + mu[0] / h;
                    lcr[q * n + q] = lcr[q * n + q] + mu[1] / h;
                    lci[q * n + q] = lci[q * n + q] + mu[2] / h;
                    q += 1;
                }
                singular = !bdf_lu(lur, n, pivr);
                if !singular { singular = !rad_clu(lcr, lci, n, pivc); }
                lu_ok = true;
            }
            if singular {
                converged = false;
            } else {
                let cv, cf = rad_newton(t, y, h, z0, scale, ntol, n, nev, lur, pivr, lcr, lci, pivc, z, info) { tt, yy => F(tt, copy(yy)) };
                if cf.kind != F_NONE { return cf; }
                converged = cv;
            }
            if !converged {
                if current_jac { break; }
                var yc: f64[>..] = [];
                i = 0;
                while i < n { yc.push(y[i]); i += 1; }
                let jf = bdf_jac(t, yc, n, nev, atol, jfac, jm) { tt, yy => F(tt, copy(yy)) };
                if jf.kind != F_NONE { return jf; }
                current_jac = true;
                lu_ok = false;
                singular = false;
            }
        }
        if !converged {
            h_abs = h_abs * 0.5;
            lu_ok = false;
            continue;
        }
        ynew = [];
        i = 0;
        while i < n { ynew.push(y[i] + z[2 * n + i]); i += 1; }
        var ze: f64[>..] = [];
        i = 0;
        while i < n {
            ze.push((z[i] * em[0] + z[n + i] * em[1] + z[2 * n + i] * em[2]) / h);
            i += 1;
        }
        var er: f64[>..] = [];
        i = 0;
        while i < n { er.push(f[i] + ze[i]); i += 1; }
        bdf_lu_solve(lur, n, pivr, er);
        scale = [];
        i = 0;
        while i < n {
            var big = mathshim::abs_of(y[i]);
            if mathshim::abs_of(ynew[i]) > big { big = mathshim::abs_of(ynew[i]); }
            scale.push(atol + big * rtol);
            i += 1;
        }
        var sq: f64[>..] = [];
        i = 0;
        while i < n { sq.push(er[i] / scale[i]); i += 1; }
        error_norm = ivp_rms(sq);
        safety = 0.9 * 13.0 / (12.0 + info[0]);
        if rejected && error_norm > 1.0 {
            var yy: f64[>..] = [];
            i = 0;
            while i < n { yy.push(y[i] + er[i]); i += 1; }
            let fv = F(t, copy(yy));
            if fv.len != n + nev {
                return fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
            }
            er = [];
            i = 0;
            while i < n { er.push(fv[i] + ze[i]); i += 1; }
            bdf_lu_solve(lur, n, pivr, er);
            sq = [];
            i = 0;
            while i < n { sq.push(er[i] / scale[i]); i += 1; }
            error_norm = ivp_rms(sq);
        }
        if error_norm > 1.0 {
            let factor = rad_predict(h_abs, h_old, error_norm, err_old);
            var fm = safety * factor;
            if fm < 0.2 { fm = 0.2; }
            h_abs = h_abs * fm;
            lu_ok = false;
            rejected = true;
        } else {
            accepted = true;
        }
    }
    let recompute = info[0] > 2.0 && info[1] > 0.001;
    var factor = rad_predict(h_abs, h_old, error_norm, err_old);
    factor = safety * factor;
    if factor > 10.0 { factor = 10.0; }
    if !recompute && factor < 1.2 {
        factor = 1.0;
    } else {
        lu_ok = false;
    }
    let fv = F(t_new, copy(ynew));
    if fv.len != n + nev {
        return fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
    }
    fb = [];
    i = 0;
    while i < n + nev { fb.push(fv[i]); i += 1; }
    if recompute {
        var yc: f64[>..] = [];
        i = 0;
        while i < n { yc.push(ynew[i]); i += 1; }
        let jf = bdf_jac(t_new, yc, n, nev, atol, jfac, jm) { tt, yy => F(tt, copy(yy)) };
        if jf.kind != F_NONE { return jf; }
        current_jac = true;
    } else {
        current_jac = false;
    }
    // The polynomial of the step.
    i = 0;
    while i < n {
        var k: i64 = 0;
        while k < 3 {
            var a: f64 = 0.0;
            var s: i64 = 0;
            while s < 3 { a += z[s * n + i] * pm[s * 3 + k]; s += 1; }
            rq[k * n + i] = a;
            k += 1;
        }
        ryo[i] = y[i];
        i += 1;
    }
    fst[1] = fst[0];
    fst[2] = error_norm;
    fst[0] = h_abs * factor;
    fst[3] = t;
    fst[4] = t_new - t;
    ist[0] = 1;
    ist[1] = 0;
    if lu_ok { ist[1] = 1; }
    ist[2] = 0;
    if current_jac { ist[2] = 1; }
    tn[0] = t_new;
    return no_fault();
}
