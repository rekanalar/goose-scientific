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

// root finds the point where several equations hold zero. The block
// takes a vector and gives a vector of the same length. The methods
// are "hybr", "lm", "broyden1", "broyden2", "linearmixing",
// "diagbroyden", "excitingmixing" and "anderson". The result holds the
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
    if method != "hybr" && method != "lm" && method != "broyden1" && method != "broyden2" && method != "linearmixing" && method != "diagbroyden" && method != "excitingmixing" && method != "anderson" {
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
