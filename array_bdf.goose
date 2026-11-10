// The method BDF of solve_ivp.
//
// The method is the variable order method of the backward differentiation
// formulas, with the modified formulas of Shampine and Reichelt. The
// order runs from one to five. The Jacobian comes from finite differences
// of the function, with the control of the step of the difference that
// SciPy uses. The state of the method lives in plain buffers that the
// caller keeps from one step to the next.
namespace scientific;

import mathshim;

// bdf_consts fills the constants of the method: the numbers gamma, alpha
// and the error constants, each with one value for every order from zero
// to five. The error constants hold six values.
fn bdf_consts(gamma: f64[>..]&, alpha: f64[>..]&, errc: f64[>..]&) {
    let kappa: f64[6] = [0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0];
    gamma.push(0.0);
    var acc: f64 = 0.0;
    var i: i64 = 1;
    while i <= 5 { acc += 1.0 / i; gamma.push(acc); i += 1; }
    i = 0;
    while i < 6 { alpha.push((1.0 - kappa[i]) * gamma[i]); i += 1; }
    i = 0;
    while i < 6 { errc.push(kappa[i] * gamma[i] + 1.0 / (i + 1)); i += 1; }
}

// bdf_compute_r gives the matrix that changes the array of differences
// when the step changes by a factor. The matrix holds order plus one
// rows and columns.
fn bdf_compute_r(order: i64, factor: f64, r: f64[>..]&) {
    let w = order + 1;
    var i: i64 = 0;
    while i < w * w { r.push(0.0); i += 1; }
    var j: i64 = 0;
    while j < w { r[j] = 1.0; j += 1; }
    i = 1;
    while i < w {
        j = 1;
        while j < w { r[i * w + j] = (i - 1.0 - factor * j) / i; j += 1; }
        i += 1;
    }
    i = 1;
    while i < w {
        j = 0;
        while j < w { r[i * w + j] = r[i * w + j] * r[(i - 1) * w + j]; j += 1; }
        i += 1;
    }
}

// bdf_change_d changes the array of differences in place when the step
// changes by a factor. The array holds one row of n values per
// difference.
fn bdf_change_d(dm: f64[>..]&, n: i64, order: i64, factor: f64) {
    var r: f64[>..] = [];
    var u: f64[>..] = [];
    bdf_compute_r(order, factor, r);
    bdf_compute_r(order, 1.0, u);
    let w = order + 1;
    var ru: f64[>..] = [];
    var a: i64 = 0;
    while a < w {
        var b: i64 = 0;
        while b < w {
            var s: f64 = 0.0;
            var k: i64 = 0;
            while k < w { s += r[a * w + k] * u[k * w + b]; k += 1; }
            ru.push(s);
            b += 1;
        }
        a += 1;
    }
    var nd: f64[>..] = [];
    a = 0;
    while a < w {
        var c: i64 = 0;
        while c < n {
            var s: f64 = 0.0;
            var b: i64 = 0;
            while b < w { s += ru[b * w + a] * dm[b * n + c]; b += 1; }
            nd.push(s);
            c += 1;
        }
        a += 1;
    }
    var q: i64 = 0;
    while q < w * n { dm[q] = nd[q]; q += 1; }
}

// bdf_lu factors a square matrix in place with the pivots of the rows.
// It gives false when a pivot holds zero.
fn bdf_lu(a: f64[>..]&, n: i64, piv: i64[>..]&) -> bool {
    var k: i64 = 0;
    while k < n {
        var p = k;
        var big = mathshim::abs_of(a[k * n + k]);
        var i: i64 = k + 1;
        while i < n {
            let v = mathshim::abs_of(a[i * n + k]);
            if v > big { big = v; p = i; }
            i += 1;
        }
        piv[k] = p;
        if !(big > 0.0) { return false; }
        if p != k {
            var j: i64 = 0;
            while j < n {
                let tmp = a[k * n + j];
                a[k * n + j] = a[p * n + j];
                a[p * n + j] = tmp;
                j += 1;
            }
        }
        i = k + 1;
        while i < n {
            let m = a[i * n + k] / a[k * n + k];
            a[i * n + k] = m;
            var j: i64 = k + 1;
            while j < n { a[i * n + j] = a[i * n + j] - m * a[k * n + j]; j += 1; }
            i += 1;
        }
        k += 1;
    }
    return true;
}

// bdf_lu_solve solves a system with the factors of bdf_lu. The right
// side goes in b and the answer comes out in b.
fn bdf_lu_solve(a: f64[:], n: i64, piv: i64[:], b: f64[>..]&) {
    var k: i64 = 0;
    while k < n {
        let p = piv[k];
        if p != k { let tmp = b[k]; b[k] = b[p]; b[p] = tmp; }
        k += 1;
    }
    var i: i64 = 1;
    while i < n {
        var s = b[i];
        var j: i64 = 0;
        while j < i { s -= a[i * n + j] * b[j]; j += 1; }
        b[i] = s;
        i += 1;
    }
    i = n - 1;
    while i >= 0 {
        var s = b[i];
        var j: i64 = i + 1;
        while j < n { s -= a[i * n + j] * b[j]; j += 1; }
        b[i] = s / a[i * n + i];
        i -= 1;
    }
}

// bdf_jac gives the Jacobian of the function at a point by forward
// differences. The factors of the steps of the differences go from one
// call to the next, as in SciPy. An empty buffer of factors starts them.
// The matrix holds one row per component of the function.
fn bdf_jac<F: fn(f64, f64[:]) -> f64[:]>(t: f64, y: f64[>..]&, n: i64, nev: i64, threshold: f64, jfac: f64[>..]&, jm: f64[>..]&) -> Fault {
    let eps: f64 = 2.220446049250313e-16;
    // A negative first factor marks a Jacobian that the caller gives. The
    // call then holds a state of one more value, and the function answers
    // with the matrix in row-major order.
    if jfac.len > 0 && jfac[0] < 0.0 {
        var ext: f64[>..] = copy_slice(y);
        ext.push(0.0);
        let jv = F(t, copy(ext));
        if jv.len != n * n {
            return fault(F_SHAPE, "solve_ivp_jac", "the Jacobian does not hold the square of the count of values");
        }
        var q: i64 = 0;
        while q < n * n {
            if !(jv[q] - jv[q] == 0.0) {
                return fault(F_VALUE, "solve_ivp_jac", "the Jacobian holds a value that is not finite");
            }
            jm[q] = jv[q];
            q += 1;
        }
        return no_fault();
    }
    let f0 = F(t, copy(y));
    if f0.len != n + nev {
        return fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
    }
    var i: i64 = 0;
    if jfac.len == 0 {
        while i < n { jfac.push(mathshim::root(eps)); i += 1; }
    }
    var ysc: f64[>..] = [];
    var h: f64[>..] = [];
    i = 0;
    while i < n {
        var sg: f64 = -1.0;
        if f0[i] >= 0.0 { sg = 1.0; }
        var big = mathshim::abs_of(y[i]);
        if threshold > big { big = threshold; }
        ysc.push(sg * big);
        var hi = (y[i] + jfac[i] * ysc[i]) - y[i];
        while hi == 0.0 {
            jfac[i] = jfac[i] * 10.0;
            hi = (y[i] + jfac[i] * ysc[i]) - y[i];
        }
        h.push(hi);
        i += 1;
    }
    // The columns of the differences, one column per component.
    var diff: f64[>..] = [];
    i = 0;
    while i < n * n { diff.push(0.0); i += 1; }
    var maxd: f64[>..] = [];
    var scl: f64[>..] = [];
    var c: i64 = 0;
    while c < n {
        var yp: f64[>..] = [];
        i = 0;
        while i < n { yp.push(y[i]); i += 1; }
        yp[c] = yp[c] + h[c];
        let fc = F(t, copy(yp));
        if fc.len != n + nev {
            return fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
        }
        var mi: i64 = 0;
        var md: f64 = -1.0;
        i = 0;
        while i < n {
            diff[i * n + c] = fc[i] - f0[i];
            let av = mathshim::abs_of(diff[i * n + c]);
            if av > md { md = av; mi = i; }
            i += 1;
        }
        maxd.push(md);
        var sa = mathshim::abs_of(f0[mi]);
        let sb = mathshim::abs_of(fc[mi]);
        if sb > sa { sa = sb; }
        scl.push(sa);
        c += 1;
    }
    // A column whose difference holds too few digits gets a larger step.
    c = 0;
    while c < n {
        if maxd[c] < mathshim::power(eps, 0.875) * scl[c] {
            let nf = 10.0 * jfac[c];
            let hn = (y[c] + nf * ysc[c]) - y[c];
            var yp: f64[>..] = [];
            i = 0;
            while i < n { yp.push(y[i]); i += 1; }
            yp[c] = yp[c] + hn;
            let fc = F(t, copy(yp));
            if fc.len != n + nev {
                return fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
            }
            var mi: i64 = 0;
            var md: f64 = -1.0;
            i = 0;
            while i < n {
                let av = mathshim::abs_of(fc[i] - f0[i]);
                if av > md { md = av; mi = i; }
                i += 1;
            }
            var sa = mathshim::abs_of(f0[mi]);
            let sb = mathshim::abs_of(fc[mi]);
            if sb > sa { sa = sb; }
            if maxd[c] * sa < md * scl[c] {
                jfac[c] = nf;
                h[c] = hn;
                i = 0;
                while i < n { diff[i * n + c] = fc[i] - f0[i]; i += 1; }
                scl[c] = sa;
                maxd[c] = md;
            }
        }
        c += 1;
    }
    c = 0;
    while c < n {
        i = 0;
        while i < n { diff[i * n + c] = diff[i * n + c] / h[c]; i += 1; }
        if maxd[c] < mathshim::power(eps, 0.75) * scl[c] { jfac[c] = jfac[c] * 10.0; }
        if maxd[c] > mathshim::power(eps, 0.25) * scl[c] { jfac[c] = jfac[c] * 0.1; }
        let lo = 1000.0 * eps;
        if jfac[c] < lo { jfac[c] = lo; }
        c += 1;
    }
    i = 0;
    while i < n * n { jm[i] = diff[i]; i += 1; }
    return no_fault();
}

// bdf_newton solves the algebraic system of one step of the method by
// the iteration of Newton. It gives whether the iteration converged and
// the count of its turns. The new state goes in y and the change goes
// in d.
fn bdf_newton<F: fn(f64, f64[:]) -> f64[:]>(t_new: f64, yp: f64[:], c: f64, psi: f64[:], lum: f64[:], piv: i64[:], scale: f64[:], tol: f64, n: i64, nev: i64, y: f64[>..]&, d: f64[>..]&, turns: i64[>..]&) -> bool, Fault {
    var i: i64 = 0;
    while i < n { y.push(yp[i]); d.push(0.0); i += 1; }
    var old: f64 = -1.0;
    var converged = false;
    var k: i64 = 0;
    var used: i64 = 0;
    while k < 4 {
        used = k + 1;
        let f = F(t_new, copy(y));
        if f.len != n + nev {
            return false, fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
        }
        var finite = true;
        i = 0;
        while i < n { if !(f[i] - f[i] == 0.0) { finite = false; } i += 1; }
        if !finite { break; }
        var dy: f64[>..] = [];
        i = 0;
        while i < n { dy.push(c * f[i] - psi[i] - d[i]); i += 1; }
        bdf_lu_solve(lum, n, piv, dy);
        var sq: f64[>..] = [];
        i = 0;
        while i < n { sq.push(dy[i] / scale[i]); i += 1; }
        let dn = ivp_rms(sq);
        var rate: f64 = 0.0;
        if old >= 0.0 { rate = dn / old; }
        if old >= 0.0 && (rate >= 1.0 || mathshim::power(rate, 4.0 - k) / (1.0 - rate) * dn > tol) { break; }
        i = 0;
        while i < n { y[i] = y[i] + dy[i]; d[i] = d[i] + dy[i]; i += 1; }
        if dn == 0.0 || (old >= 0.0 && rate / (1.0 - rate) * dn < tol) { converged = true; break; }
        old = dn;
        k += 1;
    }
    turns[0] = used;
    return converged, no_fault();
}

// bdf_step takes one step of the method from the time t. The state
// lives in the buffers: dm holds the differences, fst[0] holds the step,
// ist holds the order, the count of equal steps and whether the factors
// of the matrix are valid, jm holds the Jacobian, lum and piv hold the
// factors of the matrix. The new time goes in tn[0] and the new state
// goes in ynew. A step that falls below the smallest step is a fault.
fn bdf_step<F: fn(f64, f64[:]) -> f64[:]>(t: f64, t_bound: f64, dir: f64, n: i64, nev: i64, atol: f64, rtol: f64, newton_tol: f64, max_step: f64, dm: f64[>..]&, fst: f64[>..]&, ist: i64[>..]&, jm: f64[>..]&, jfac: f64[>..]&, lum: f64[>..]&, piv: i64[>..]&, ynew: f64[>..]&, tn: f64[>..]&) -> Fault {
    var gamma: f64[>..] = [];
    var alpha: f64[>..] = [];
    var errc: f64[>..] = [];
    bdf_consts(gamma, alpha, errc);
    var min_step = 10.0 * mathshim::abs_of(t) * 2.220446049250313e-16;
    if min_step < 5.0e-323 { min_step = 5.0e-323; }
    var h_abs = fst[0];
    var order = ist[0];
    if h_abs > max_step {
        bdf_change_d(dm, n, order, max_step / h_abs);
        ist[1] = 0;
        h_abs = max_step;
    } else if h_abs < min_step {
        bdf_change_d(dm, n, order, min_step / h_abs);
        ist[1] = 0;
        h_abs = min_step;
    }
    var lu_ok = ist[2] == 1;
    var current_jac = false;
    var accepted = false;
    var t_new: f64 = t;
    var d: f64[>..] = [];
    var error_norm: f64 = 0.0;
    var safety: f64 = 0.0;
    var scale: f64[>..] = [];
    var i: i64 = 0;
    while !accepted {
        if h_abs < min_step {
            return fault(F_VALUE, "solve_ivp", "the step fell below the smallest step");
        }
        var h = h_abs * dir;
        t_new = t + h;
        if dir * (t_new - t_bound) > 0.0 {
            t_new = t_bound;
            bdf_change_d(dm, n, order, mathshim::abs_of(t_new - t) / h_abs);
            ist[1] = 0;
            lu_ok = false;
        }
        h = t_new - t;
        h_abs = mathshim::abs_of(h);
        var yp: f64[>..] = [];
        var psi: f64[>..] = [];
        scale = [];
        i = 0;
        while i < n {
            var s: f64 = 0.0;
            var j: i64 = 0;
            while j <= order { s += dm[j * n + i]; j += 1; }
            yp.push(s);
            scale.push(atol + rtol * mathshim::abs_of(s));
            var p: f64 = 0.0;
            j = 1;
            while j <= order { p += dm[j * n + i] * gamma[j]; j += 1; }
            psi.push(p / alpha[order]);
            i += 1;
        }
        var converged = false;
        let c = h / alpha[order];
        var turns: i64[>..] = [0];
        var singular = false;
        while !converged {
            if !lu_ok {
                var q: i64 = 0;
                while q < n * n { lum[q] = 0.0 - c * jm[q]; q += 1; }
                q = 0;
                while q < n { lum[q * n + q] = lum[q * n + q] + 1.0; q += 1; }
                singular = !bdf_lu(lum, n, piv);
                lu_ok = true;
            }
            if singular {
                converged = false;
            } else {
                ynew = [];
                d = [];
                let cv, cf = bdf_newton(t_new, yp, c, psi, lum, piv, scale, newton_tol, n, nev, ynew, d, turns) { tt, yy => F(tt, copy(yy)) };
                if cf.kind != F_NONE { return cf; }
                converged = cv;
            }
            if !converged {
                if current_jac { break; }
                let jf = bdf_jac(t_new, yp, n, nev, atol, jfac, jm) { tt, yy => F(tt, copy(yy)) };
                if jf.kind != F_NONE { return jf; }
                lu_ok = false;
                singular = false;
                current_jac = true;
            }
        }
        if !converged {
            h_abs = h_abs * 0.5;
            bdf_change_d(dm, n, order, 0.5);
            ist[1] = 0;
            lu_ok = false;
            continue;
        }
        safety = 0.9 * 9.0 / (8.0 + turns[0]);
        var er: f64[>..] = [];
        scale = [];
        i = 0;
        while i < n {
            scale.push(atol + rtol * mathshim::abs_of(ynew[i]));
            er.push(errc[order] * d[i] / scale[i]);
            i += 1;
        }
        error_norm = ivp_rms(er);
        if error_norm > 1.0 {
            var factor = safety * mathshim::power(error_norm, -1.0 / (order + 1.0));
            if factor < 0.2 { factor = 0.2; }
            h_abs = h_abs * factor;
            bdf_change_d(dm, n, order, factor);
            ist[1] = 0;
        } else {
            accepted = true;
        }
    }
    ist[1] = ist[1] + 1;
    tn[0] = t_new;
    // The differences of the new point.
    i = 0;
    while i < n {
        dm[(order + 2) * n + i] = d[i] - dm[(order + 1) * n + i];
        dm[(order + 1) * n + i] = d[i];
        i += 1;
    }
    var r = order;
    while r >= 0 {
        i = 0;
        while i < n { dm[r * n + i] = dm[r * n + i] + dm[(r + 1) * n + i]; i += 1; }
        r -= 1;
    }
    ist[2] = 1;
    if !lu_ok { ist[2] = 0; }
    fst[0] = h_abs;
    if ist[1] < order + 1 { return no_fault(); }
    // The order and the step for the next turn.
    var em: f64 = 1.0 / 0.0;
    var ep: f64 = 1.0 / 0.0;
    var v: f64[>..] = [];
    if order > 1 {
        i = 0;
        while i < n { v.push(errc[order - 1] * dm[order * n + i] / scale[i]); i += 1; }
        em = ivp_rms(v);
    }
    if order < 5 {
        v = [];
        i = 0;
        while i < n { v.push(errc[order + 1] * dm[(order + 2) * n + i] / scale[i]); i += 1; }
        ep = ivp_rms(v);
    }
    var fac: f64[>..] = [];
    var q: i64 = 0;
    while q < 3 {
        var en = em;
        if q == 1 { en = error_norm; }
        if q == 2 { en = ep; }
        var fv: f64 = 0.0;
        if en == 0.0 { fv = 1.0 / 0.0; }
        else if !(en - en == 0.0) { fv = 0.0; }
        else { fv = mathshim::power(en, -1.0 / (order + q)); }
        fac.push(fv);
        q += 1;
    }
    var best: i64 = 0;
    q = 1;
    while q < 3 { if fac[q] > fac[best] { best = q; } q += 1; }
    order = order + best - 1;
    ist[0] = order;
    var fmax = fac[best];
    var factor = safety * fmax;
    if factor > 10.0 { factor = 10.0; }
    fst[0] = fst[0] * factor;
    bdf_change_d(dm, n, order, factor);
    ist[1] = 0;
    ist[2] = 0;
    return no_fault();
}

// bdf_poly gives the state of a step of the method at a time. The rows
// hold the differences and then the end of the step, the step and the
// order of the step.
fn bdf_poly(n: i64, kb: f64[:], time: f64) -> f64[>..] {
    let tn = kb[6 * n];
    let h = kb[7 * n];
    let order = kb[8 * n] as i64;
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n { out.push(kb[i]); i += 1; }
    var p: f64 = 1.0;
    var k: i64 = 0;
    while k < order {
        p = p * ((time - (tn - h * k)) / (h * (k + 1)));
        i = 0;
        while i < n { out[i] = out[i] + kb[(k + 1) * n + i] * p; i += 1; }
        k += 1;
    }
    return out;
}
