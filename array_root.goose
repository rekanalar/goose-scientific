// The root finding of several equations. The function root follows
// scipy.optimize.root for the methods that the package holds.
namespace scientific;

import mathshim;

// rt_norm gives the largest absolute value of a vector.
fn rt_norm(v: f64[:]) -> f64 {
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < v.len {
        let a = mathshim::abs_of(v[i]);
        if a != a { return a; }
        if a > s { s = a; }
        i += 1;
    }
    return s;
}

// rt_inverse_jacobian gives the inverse of the matrix of the
// derivatives, taken by differences. A matrix without an inverse gives
// the false flag.
fn rt_inverse_jacobian<F: fn(f64[:]) -> f64[:]>(x: f64[:], fx: f64[:], n: i64) -> f64[>..], bool {
    var jac: f64[>..] = [];
    var c: i64 = 0;
    while c < n * n { jac.push(0.0); c += 1; }
    var d: i64 = 0;
    while d < n {
        let h = 0.0000001 * (1.0 + mathshim::abs_of(x[d]));
        var probe: f64[>..] = copy_slice(x);
        probe[d] = probe[d] + h;
        let fp = F(copy(probe));
        if fp.len != n { return [], false; }
        var i: i64 = 0;
        while i < n { jac[i * n + d] = (fp[i] - fx[i]) / h; i += 1; }
        d += 1;
    }
    var sh: i64[..8] = [];
    sh.push(n);
    sh.push(n);
    let inv_arr, fi = inv(Arr { ndim: 2, shape: sh, data: copy(jac) });
    if fi.kind != F_NONE { return [], false; }
    return copy_slice(inv_arr.data), true;
}

// rt_broyden runs the method of Broyden. The flag good selects the
// first method of Broyden, and the flag false selects the second. The
// inverse of the derivatives starts from the differences and each step
// updates it by a rank one change.
fn rt_broyden<F: fn(f64[:]) -> f64[:]>(x0: f64[:], good: bool, tol: f64, maxiter: i64, n: i64) -> f64[>..], f64[>..], bool, bool {
    var x: f64[>..] = copy_slice(x0);
    let fx0 = F(copy(x));
    if fx0.len != n { return [], [], false, false; }
    var fx: f64[>..] = copy_slice(fx0);
    let h0, ok0 = rt_inverse_jacobian(x, fx, n) { v => F(copy(v)) };
    if !ok0 { return x, fx, false, true; }
    var h: f64[>..] = copy_slice(h0);
    var turn: i64 = 0;
    while turn < maxiter {
        if rt_norm(fx) <= tol { return x, fx, true, true; }
        // The step is minus the inverse times the values.
        var dx: f64[>..] = [];
        var i: i64 = 0;
        while i < n {
            var t: f64 = 0.0;
            var j: i64 = 0;
            while j < n { t += h[i * n + j] * fx[j]; j += 1; }
            dx.push(-t);
            i += 1;
        }
        // The line search halves the step while the error grows.
        let size = rt_norm(fx);
        var factor: f64 = 1.0;
        var xn: f64[>..] = [];
        var fn2: f64[>..] = [];
        var trial_f = F(copy(x));
        var tries: i64 = 0;
        while tries < 30 {
            xn = [];
            i = 0;
            while i < n { xn.push(x[i] + factor * dx[i]); i += 1; }
            trial_f = F(copy(xn));
            if trial_f.len != n { return [], [], false, false; }
            fn2 = copy_slice(trial_f);
            let sz = rt_norm(fn2);
            if sz == sz && sz < size { break; }
            factor = factor / 2.0;
            tries += 1;
        }
        var step: f64[>..] = [];
        var df: f64[>..] = [];
        i = 0;
        while i < n { step.push(xn[i] - x[i]); df.push(fn2[i] - fx[i]); i += 1; }
        // The vector h times df.
        var hdf: f64[>..] = [];
        i = 0;
        while i < n {
            var t: f64 = 0.0;
            var j: i64 = 0;
            while j < n { t += h[i * n + j] * df[j]; j += 1; }
            hdf.push(t);
            i += 1;
        }
        var denom: f64 = 0.0;
        var row: f64[>..] = [];
        if good {
            // The row is the step times h.
            var j: i64 = 0;
            while j < n {
                var t: f64 = 0.0;
                i = 0;
                while i < n { t += step[i] * h[i * n + j]; i += 1; }
                row.push(t);
                denom += t * df[j];
                j += 1;
            }
        } else {
            i = 0;
            while i < n { row.push(df[i]); denom += df[i] * df[i]; i += 1; }
        }
        if denom != 0.0 && denom == denom {
            i = 0;
            while i < n {
                let u = (step[i] - hdf[i]) / denom;
                var j: i64 = 0;
                while j < n { h[i * n + j] = h[i * n + j] + u * row[j]; j += 1; }
                i += 1;
            }
        }
        x = copy_slice(xn);
        fx = copy_slice(fn2);
        turn += 1;
    }
    return x, fx, rt_norm(fx) <= tol, true;
}

// rt_mixing runs a simple iteration of SciPy. The mode 0 is linear
// mixing, 1 is diagonal Broyden, 2 is exciting mixing and 3 is Anderson
// mixing. Each mode keeps a cheap model of the inverse derivative. The
// step is minus the model times the values, and a halving line search
// keeps the error falling. SciPy takes the opposite sign of the first
// guess for these methods and diverges on common problems. This
// function takes the sign that descends.
fn rt_mixing<F: fn(f64[:]) -> f64[:]>(x0: f64[:], mode: i64, tol: f64, maxiter: i64, n: i64) -> f64[>..], bool, bool {
    var x: f64[>..] = copy_slice(x0);
    let f0 = F(copy(x));
    if f0.len != n { return [], false, false; }
    var fx: f64[>..] = copy_slice(f0);
    var alpha: f64 = 1.0;
    let nf0 = rt_norm(fx);
    var nx0: f64 = 1.0;
    var k: i64 = 0;
    while k < n { let a = mathshim::abs_of(x[k]); if a > nx0 { nx0 = a; } k += 1; }
    if nf0 > 0.0 { alpha = 0.5 * nx0 / nf0; }
    var d: f64[>..] = [];
    var beta: f64[>..] = [];
    k = 0;
    while k < n { d.push(-1.0 / alpha); beta.push(alpha); k += 1; }
    // The history of Anderson mixing: M pairs of steps and changes.
    var hx: f64[>..] = [];
    var hf: f64[>..] = [];
    var hn: i64 = 0;
    let mem: i64 = 5;
    var turn: i64 = 0;
    while turn < maxiter {
        if rt_norm(fx) <= tol { return x, true, true; }
        var step: f64[>..] = [];
        k = 0;
        if mode == 0 {
            while k < n { step.push(-alpha * fx[k]); k += 1; }
        } else if mode == 1 {
            while k < n { step.push(fx[k] / d[k]); k += 1; }
        } else if mode == 2 {
            while k < n { step.push(-beta[k] * fx[k]); k += 1; }
        } else {
            while k < n { step.push(-alpha * fx[k]); k += 1; }
            if hn > 0 {
                var a: f64[>..] = [];
                var rhs: f64[>..] = [];
                var i: i64 = 0;
                while i < hn {
                    var t: f64 = 0.0;
                    k = 0;
                    while k < n { t += hf[i * n + k] * fx[k]; k += 1; }
                    rhs.push(t);
                    var j: i64 = 0;
                    while j < hn {
                        var v: f64 = 0.0;
                        k = 0;
                        while k < n { v += hf[i * n + k] * hf[j * n + k]; k += 1; }
                        if i == j { v = v * 1.0001; }
                        a.push(v);
                        j += 1;
                    }
                    i += 1;
                }
                var sa: i64[..8] = [];
                sa.push(hn);
                sa.push(hn);
                var sb: i64[..8] = [];
                sb.push(hn);
                let g, fg = solve(Arr { ndim: 2, shape: sa, data: copy(a) }, Arr { ndim: 1, shape: sb, data: copy(rhs) });
                if fg.kind != F_NONE {
                    hn = 0;
                    hx = [];
                    hf = [];
                } else {
                    i = 0;
                    while i < hn {
                        k = 0;
                        while k < n {
                            step[k] = step[k] - g.data[i] * (hx[i * n + k] - alpha * hf[i * n + k]);
                            k += 1;
                        }
                        i += 1;
                    }
                }
            }
        }
        let size = rt_norm(fx);
        var factor: f64 = 1.0;
        var xn: f64[>..] = [];
        var fn2: f64[>..] = [];
        var tries: i64 = 0;
        while tries < 30 {
            xn = [];
            k = 0;
            while k < n { xn.push(x[k] + factor * step[k]); k += 1; }
            let tf = F(copy(xn));
            if tf.len != n { return [], false, false; }
            fn2 = copy_slice(tf);
            let sz = rt_norm(fn2);
            if sz == sz && sz < size { break; }
            factor = factor / 2.0;
            tries += 1;
        }
        var dxv: f64[>..] = [];
        var dfv: f64[>..] = [];
        var dn: f64 = 0.0;
        k = 0;
        while k < n {
            dxv.push(xn[k] - x[k]);
            dfv.push(fn2[k] - fx[k]);
            dn += dxv[k] * dxv[k];
            k += 1;
        }
        if mode == 1 {
            if dn > 0.0 {
                k = 0;
                while k < n { d[k] = d[k] - (dfv[k] + d[k] * dxv[k]) * dxv[k] / dn; k += 1; }
            }
            // A zero entry has no inverse. The start value replaces it.
            k = 0;
            while k < n { if d[k] == 0.0 { d[k] = -1.0 / alpha; } k += 1; }
        } else if mode == 2 {
            k = 0;
            while k < n {
                if fn2[k] * fx[k] > 0.0 { beta[k] = beta[k] + alpha; } else { beta[k] = alpha; }
                if beta[k] > 1.0 { beta[k] = 1.0; }
                k += 1;
            }
        } else if mode == 3 {
            if hn == mem {
                var tx: f64[>..] = [];
                var tf2: f64[>..] = [];
                var q: i64 = n;
                while q < hx.len { tx.push(hx[q]); tf2.push(hf[q]); q += 1; }
                hx = copy_slice(tx);
                hf = copy_slice(tf2);
                hn -= 1;
            }
            k = 0;
            while k < n { hx.push(dxv[k]); hf.push(dfv[k]); k += 1; }
            hn += 1;
        }
        x = copy_slice(xn);
        fx = copy_slice(fn2);
        turn += 1;
    }
    return x, rt_norm(fx) <= tol, true;
}

// rt_sumsq gives the sum of the squares of a vector.
fn rt_sumsq(v: f64[:]) -> f64 {
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < v.len { s += v[i] * v[i]; i += 1; }
    return s;
}

// rt_dfsane runs the spectral residual method of La Cruz, Martinez and
// Raydan with the non-monotone line search of the same authors. It uses
// no derivative. The merit is the squared norm of the values. The
// count of evaluations is at most maxfev.
fn rt_dfsane<F: fn(f64[:]) -> f64[:]>(x0: f64[:], tol: f64, maxfev: i64, n: i64) -> f64[>..], bool, bool {
    var x: f64[>..] = copy_slice(x0);
    let f0v = F(copy(x));
    if f0v.len != n { return [], false, false; }
    var fv: f64[>..] = copy_slice(f0v);
    var fk: f64 = rt_sumsq(fv);
    let f0 = fk;
    let norm0 = mathshim::root(f0);
    var sigma: f64 = 1.0;
    var prev: f64[>..] = [fk];
    var nfev: i64 = 1;
    var k: i64 = 0;
    while true {
        if mathshim::root(fk) < tol * norm0 + 1.0e-300 { return x, true, true; }
        if mathshim::abs_of(sigma) > 1.0e10 {
            if sigma < 0.0 { sigma = -1.0e10; } else { sigma = 1.0e10; }
        } else if mathshim::abs_of(sigma) < 1.0e-10 {
            sigma = 1.0e-10;
        }
        var d: f64[>..] = [];
        var q: i64 = 0;
        while q < n { d.push(-sigma * fv[q]); q += 1; }
        let kk = 1.0 + 1.0 * k;
        let eta = f0 / (kk * kk);
        var fbar: f64 = prev[0];
        q = 1;
        while q < prev.len { if prev[q] > fbar { fbar = prev[q]; } q += 1; }
        var ap: f64 = 1.0;
        var am: f64 = 1.0;
        var xp: f64[>..] = [];
        var fpv: f64[>..] = [];
        var fp: f64 = 0.0;
        while true {
            if nfev >= maxfev { return x, false, true; }
            xp = [];
            q = 0;
            while q < n { xp.push(x[q] + ap * d[q]); q += 1; }
            let t1 = F(copy(xp));
            if t1.len != n { return [], false, false; }
            fpv = copy_slice(t1);
            fp = rt_sumsq(fpv);
            nfev += 1;
            if fp <= fbar + eta - 1.0e-4 * ap * ap * fk { break; }
            let atp = ap * ap * fk / (fp + (2.0 * ap - 1.0) * fk);
            if nfev >= maxfev { return x, false, true; }
            xp = [];
            q = 0;
            while q < n { xp.push(x[q] - am * d[q]); q += 1; }
            let t2 = F(copy(xp));
            if t2.len != n { return [], false, false; }
            fpv = copy_slice(t2);
            fp = rt_sumsq(fpv);
            nfev += 1;
            if fp <= fbar + eta - 1.0e-4 * am * am * fk { break; }
            let atm = am * am * fk / (fp + (2.0 * am - 1.0) * fk);
            var np = atp;
            if np < 0.1 * ap { np = 0.1 * ap; }
            if np > 0.5 * ap { np = 0.5 * ap; }
            var nm = atm;
            if nm < 0.1 * am { nm = 0.1 * am; }
            if nm > 0.5 * am { nm = 0.5 * am; }
            ap = np;
            am = nm;
        }
        var ss: f64 = 0.0;
        var sy: f64 = 0.0;
        q = 0;
        while q < n {
            let sq = xp[q] - x[q];
            ss += sq * sq;
            sy += sq * (fpv[q] - fv[q]);
            q += 1;
        }
        sigma = ss / sy;
        if sigma != sigma { sigma = 1.0; }
        x = copy_slice(xp);
        fv = copy_slice(fpv);
        fk = fp;
        prev.push(fp);
        if prev.len > 10 {
            var tp: f64[>..] = [];
            q = 1;
            while q < prev.len { tp.push(prev[q]); q += 1; }
            prev = copy_slice(tp);
        }
        k += 1;
    }
    return x, false, true;
}

// rt_dot_off gives the dot product of two parts of two flat arrays. The
// offset holds the first element of the part.
fn rt_dot_off(a: f64[:], aoff: i64, b: f64[:], boff: i64, n: i64) -> f64 {
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < n { s += a[aoff + i] * b[boff + i]; i += 1; }
    return s;
}

// rt_norm_off gives the length of a part of a flat array. The length
// is the square root of the sum of the squares. The Krylov method needs
// this length and not the largest value.
fn rt_norm_off(a: f64[:], off: i64, n: i64) -> f64 {
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < n { s += a[off + i] * a[off + i]; i += 1; }
    return mathshim::root(s);
}

// rt_krylov_matvec gives the product of the derivative of the function
// and a vector. The product comes from a small difference of the
// function, like the method of SciPy. The vector sits at the offset in
// the flat array. The flag is false when the function gives a different
// count of values.
fn rt_krylov_matvec<F: fn(f64[:]) -> f64[:]>(x: f64[:], f0: f64[:], v: f64[:], off: i64, n: i64, omega: f64) -> f64[>..], bool {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n { out.push(0.0); i += 1; }
    let nv = rt_norm_off(v, off, n);
    if nv == 0.0 { return out, true; }
    let sc = omega / nv;
    var probe: f64[>..] = copy_slice(x);
    i = 0;
    while i < n { probe[i] = probe[i] + sc * v[off + i]; i += 1; }
    let fp = F(copy(probe));
    if fp.len != n { return out, false; }
    i = 0;
    while i < n { out[i] = (fp[i] - f0[i]) / sc; i += 1; }
    return out, true;
}

// rt_krylov runs the Newton-Krylov method of SciPy. Each outer step is
// a Newton step. The linear system of the step is solved by a Krylov
// iteration, so the package never builds the derivative. The inner
// iteration restarts twenty times at most. The line search of Armijo
// takes the step.
fn rt_krylov<F: fn(f64[:]) -> f64[:]>(x0: f64[:], tol: f64, maxiter: i64, n: i64) -> f64[>..], bool, bool {
    var x: f64[>..] = copy_slice(x0);
    let m: i64 = 20;
    let rdiff: f64 = 1.4901161193847656e-08;
    var turn: i64 = 0;
    while turn < maxiter {
        let fxv = F(copy(x));
        if fxv.len != n { return [], false, false; }
        let f0: f64[>..] = copy_slice(fxv);
        let fnorm = rt_norm(f0);
        if fnorm <= tol { return x, true, true; }
        // The step of the difference scales with the point and the
        // values.
        var mx: f64 = 1.0;
        let xn = rt_norm(x);
        if xn > 1.0 { mx = xn; }
        var mf: f64 = 1.0;
        if fnorm > 1.0 { mf = fnorm; }
        let omega: f64 = rdiff * mx / mf;
        // The right side of the linear system.
        var b: f64[>..] = [];
        var i: i64 = 0;
        while i < n { b.push(-f0[i]); i += 1; }
        let bn = rt_norm_off(b, 0, n);
        var dx: f64[>..] = [];
        i = 0;
        while i < n { dx.push(0.0); i += 1; }
        var cycle: i64 = 0;
        while cycle < 5 {
            // The residual of the linear system.
            let jd, jok = rt_krylov_matvec(x, f0, dx, 0, n, omega) { v => F(copy(v)) };
            if !jok { return [], false, false; }
            var r: f64[>..] = [];
            i = 0;
            while i < n { r.push(b[i] - jd[i]); i += 1; }
            let beta = rt_norm_off(r, 0, n);
            if beta <= 1.0e-10 * bn { break; }
            // The basis of the Krylov space and the matrix of the
            // coefficients.
            var v: f64[>..] = [];
            i = 0;
            while i < (m + 1) * n { v.push(0.0); i += 1; }
            i = 0;
            while i < n { v[i] = r[i] / beta; i += 1; }
            var h: f64[>..] = [];
            i = 0;
            while i < (m + 1) * m { h.push(0.0); i += 1; }
            var cs: f64[>..] = [];
            var sn: f64[>..] = [];
            var g: f64[>..] = [];
            i = 0;
            while i < m { cs.push(0.0); sn.push(0.0); g.push(0.0); i += 1; }
            g.push(0.0);
            g[0] = beta;
            var j: i64 = 0;
            var used: i64 = 0;
            while j < m {
                // The next basis vector comes from the product of the
                // derivative and the last vector.
                let wres, wok = rt_krylov_matvec(x, f0, v, j * n, n, omega) { v => F(copy(v)) };
                if !wok { return [], false, false; }
                var w: f64[>..] = copy_slice(wres);
                var q: i64 = 0;
                while q <= j {
                    let hv = rt_dot_off(v, q * n, w, 0, n);
                    h[q * m + j] = hv;
                    var t: i64 = 0;
                    while t < n { w[t] = w[t] - hv * v[q * n + t]; t += 1; }
                    q += 1;
                }
                let hn = rt_norm_off(w, 0, n);
                h[(j + 1) * m + j] = hn;
                // The space of the Krylov method closes when the last
                // vector leaves nothing. The count of vectors stops
                // after the rotation of this column.
                let closed: bool = hn <= 1.0e-14 * bn;
                if !closed {
                    var t: i64 = 0;
                    while t < n { v[(j + 1) * n + t] = w[t] / hn; t += 1; }
                }
                // The rotations of the earlier steps.
                q = 0;
                while q < j {
                    let upper = h[q * m + j];
                    let lower = h[(q + 1) * m + j];
                    h[q * m + j] = cs[q] * upper + sn[q] * lower;
                    h[(q + 1) * m + j] = -sn[q] * upper + cs[q] * lower;
                    q += 1;
                }
                // The new rotation.
                let up = h[j * m + j];
                let lo = h[(j + 1) * m + j];
                let den = mathshim::root(up * up + lo * lo);
                if den != 0.0 {
                    cs[j] = up / den;
                    sn[j] = lo / den;
                } else {
                    cs[j] = 1.0;
                    sn[j] = 0.0;
                }
                h[j * m + j] = cs[j] * up + sn[j] * lo;
                h[(j + 1) * m + j] = 0.0;
                let gj = g[j];
                g[j] = cs[j] * gj;
                g[j + 1] = -sn[j] * gj;
                j += 1;
                used = j;
                if closed { break; }
                if mathshim::abs_of(g[j]) <= 1.0e-10 * bn { break; }
            }
            // The solution of the small system gives the step.
            var y: f64[>..] = [];
            i = 0;
            while i < used { y.push(0.0); i += 1; }
            i = used - 1;
            while i >= 0 {
                var s: f64 = g[i];
                var q: i64 = i + 1;
                while q < used { s -= h[i * m + q] * y[q]; q += 1; }
                if h[i * m + i] != 0.0 { y[i] = s / h[i * m + i]; } else { y[i] = 0.0; }
                i -= 1;
            }
            i = 0;
            while i < used {
                var t: i64 = 0;
                while t < n { dx[t] = dx[t] + y[i] * v[i * n + t]; t += 1; }
                i += 1;
            }
            cycle += 1;
        }
        // The line search of Armijo takes the step that lowers the
        // squared norm of the values.
        let phi0 = fnorm * fnorm;
        var a: f64 = 1.0;
        var taken: bool = false;
        var tries: i64 = 0;
        while tries < 40 {
            var probe: f64[>..] = copy_slice(x);
            var t: i64 = 0;
            while t < n { probe[t] = probe[t] + a * dx[t]; t += 1; }
            let fp = F(copy(probe));
            if fp.len != n { return [], false, false; }
            let pn = rt_norm(fp);
            if pn * pn <= phi0 * (1.0 - 1.0e-4 * a) { taken = true; break; }
            a = a / 2.0;
            tries += 1;
        }
        if !taken { a = 1.0; }
        var t: i64 = 0;
        while t < n { x[t] = x[t] + a * dx[t]; t += 1; }
        turn += 1;
    }
    let fxv = F(copy(x));
    if fxv.len != n { return [], false, false; }
    return x, rt_norm(fxv) <= tol, true;
}

// root finds the point where several equations hold zero. The block
// takes a vector and gives a vector of the same length. The methods
// are "hybr", "lm", "broyden1", "broyden2", "linearmixing",
// "diagbroyden", "excitingmixing", "anderson", "df-sane" and "krylov". The result holds the
// point and the values of the equations at the point. A call that does
// not reach the tolerance is a fault.
fn root<F: fn(f64[:]) -> f64[:]>(x0: f64[:], method: const u8[:] = "hybr", tol: f64 = 1.0e-8, maxiter: i64 = 200) -> Arr, Arr, Fault {
    let n = x0.len;
    if n < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "root", "the start holds no element");
    }
    if tol <= 0.0 || tol != tol {
        return empty_arr(), empty_arr(), fault(F_VALUE, "root", "the tolerance holds a value of zero or below");
    }
    if maxiter < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "root", "the count of turns holds a value below one");
    }
    var i: i64 = 0;
    while i < n {
        if !(x0[i] - x0[i] == 0.0) {
            return empty_arr(), empty_arr(), fault(F_VALUE, "root", "the start holds a value that is not finite");
        }
        i += 1;
    }
    if method != "hybr" && method != "lm" && method != "broyden1" && method != "broyden2" && method != "linearmixing" && method != "diagbroyden" && method != "excitingmixing" && method != "anderson" && method != "df-sane" && method != "krylov" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "root", "the method holds an unknown name");
    }
    var x: f64[>..] = [];
    if method == "hybr" {
        let r, f = fsolve(x0, tol, maxiter) { v => F(copy(v)) };
        if f.kind != F_NONE { return empty_arr(), empty_arr(), fault(f.kind, "root", f.detail); }
        x = copy_slice(r.data);
    } else if method == "lm" {
        let r, f = least_squares(x0, tol, maxiter) { v => F(copy(v)) };
        if f.kind != F_NONE { return empty_arr(), empty_arr(), fault(f.kind, "root", f.detail); }
        x = copy_slice(r.data);
    } else if method == "krylov" {
        let rx, reached, shape_ok = rt_krylov(x0, tol, maxiter, n) { v => F(copy(v)) };
        if !shape_ok {
            return empty_arr(), empty_arr(), fault(F_SHAPE, "root", "the function holds a different count of values");
        }
        if !reached {
            return empty_arr(), empty_arr(), fault(F_LIMIT, "root", "the method did not reach the tolerance in the count of turns");
        }
        x = copy_slice(rx);
    } else if method == "df-sane" {
        let rx, reached, shape_ok = rt_dfsane(x0, tol, 5 * maxiter + 1000, n) { v => F(copy(v)) };
        if !shape_ok {
            return empty_arr(), empty_arr(), fault(F_SHAPE, "root", "the function holds a different count of values");
        }
        if !reached {
            return empty_arr(), empty_arr(), fault(F_LIMIT, "root", "the method did not reach the tolerance in the count of turns");
        }
        x = copy_slice(rx);
    } else if method == "linearmixing" || method == "diagbroyden" || method == "excitingmixing" || method == "anderson" {
        var mode: i64 = 0;
        if method == "diagbroyden" { mode = 1; }
        if method == "excitingmixing" { mode = 2; }
        if method == "anderson" { mode = 3; }
        let rx, reached, shape_ok = rt_mixing(x0, mode, tol, maxiter, n) { v => F(copy(v)) };
        if !shape_ok {
            return empty_arr(), empty_arr(), fault(F_SHAPE, "root", "the function holds a different count of values");
        }
        if !reached {
            return empty_arr(), empty_arr(), fault(F_LIMIT, "root", "the method did not reach the tolerance in the count of turns");
        }
        x = copy_slice(rx);
    } else {
        let rx, rf, reached, shape_ok = rt_broyden(x0, method == "broyden1", tol, maxiter, n) { v => F(copy(v)) };
        if !shape_ok {
            return empty_arr(), empty_arr(), fault(F_SHAPE, "root", "the function holds a different count of values");
        }
        if !reached {
            return empty_arr(), empty_arr(), fault(F_LIMIT, "root", "the method did not reach the tolerance in the count of turns");
        }
        x = copy_slice(rx);
    }
    let fv = F(copy(x));
    var s: i64[..8] = [];
    s.push(n);
    var s2: i64[..8] = [];
    s2.push(fv.len);
    if method == "lm" && rt_norm(fv) > 1.0e-4 {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "root", "the method stopped at a point that is not a root");
    }
    return Arr { ndim: 1, shape: s, data: x }, Arr { ndim: 1, shape: s2, data: copy_slice(fv) }, no_fault();
}
