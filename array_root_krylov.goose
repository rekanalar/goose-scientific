// The Krylov solver of the root finder and the public call root.
//
namespace scientific;

import mathshim;

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
