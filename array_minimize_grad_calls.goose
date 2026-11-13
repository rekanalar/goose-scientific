// The public calls minimize_grad and minimize_constrained.
//
namespace scientific;

import mathshim;

// minimize_grad finds the smallest point of a function with a method
// that uses the gradient. The block of the caller takes a vector. It
// gives the value first and the gradient after it. The methods are
// "bfgs", "cg", "lbfgsb" and "newton-cg". The bounds hold one value per variable.
// The start that lies outside the bounds moves to the border. The
// result holds the point and the value.
fn minimize_grad<F: fn(f64[:]) -> f64[:]>(x0: f64[:], method: const u8[:] = "bfgs", bounded: bool = false, lb: f64[:] = [], ub: f64[:] = [], xtol: f64 = 1.0e-8, ftol: f64 = 1.0e-8, gtol: f64 = 1.0e-8, turns: i64 = 200) -> Arr, Arr, Fault {
    let n = x0.len;
    if n < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "minimize_grad", "the start holds no element");
    }
    if xtol <= 0.0 || ftol <= 0.0 || gtol <= 0.0 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "minimize_grad", "a tolerance holds a value of zero or below");
    }
    if turns < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "minimize_grad", "the count of turns holds a value below one");
    }
    if method != "bfgs" && method != "cg" && method != "lbfgsb" && method != "newton-cg" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "minimize_grad", "the method holds an unknown name");
    }
    if bounded {
        if lb.len != n || ub.len != n {
            return empty_arr(), empty_arr(), fault(F_SHAPE, "minimize_grad", "the bounds must hold one value per variable");
        }
        var i: i64 = 0;
        while i < n {
            if lb[i] > ub[i] {
                return empty_arr(), empty_arr(), fault(F_VALUE, "minimize_grad", "a lower bound lies above the upper bound");
            }
            i += 1;
        }
    }
    // SciPy replaces the methods BFGS and CG with L-BFGS-B when bounds
    // are given, because the two methods cannot hold the bounds. The
    // package does the same, so the result matches.
    var use: const u8[:] = method;
    if bounded && method == "newton-cg" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "minimize_grad", "the method newton-cg holds no bounds");
    }
    if bounded && (method == "bfgs" || method == "cg") { use = "lbfgsb"; }
    var x: f64[>..] = [];
    var f: f64 = 0.0;
    var g: f64[>..] = [];
    if use == "bfgs" {
        let xx, ff, gg = mg_bfgs(x0, lb, ub, bounded, xtol, ftol, gtol, turns, n) { v => F(copy(v)) };
        x = copy(xx);
        f = ff;
        g = copy(gg);
    } else if use == "newton-cg" {
        let xx, ff, gg = mg_newtoncg(x0, xtol, turns, n) { v => F(copy(v)) };
        x = copy(xx);
        f = ff;
        g = copy(gg);
    } else if use == "cg" {
        let xx, ff, gg = mg_cg(x0, lb, ub, bounded, xtol, ftol, gtol, turns, n) { v => F(copy(v)) };
        x = copy(xx);
        f = ff;
        g = copy(gg);
    } else {
        let xx, ff, gg = mg_lbfgs(x0, lb, ub, bounded, xtol, ftol, gtol, turns, n) { v => F(copy(v)) };
        x = copy(xx);
        f = ff;
        g = copy(gg);
    }
    if x.len != n {
        return empty_arr(), empty_arr(), fault(F_VALUE, "minimize_grad", "the function holds a wrong count of values");
    }
    var s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: copy(x) }, line_arr([f]), no_fault();
}

// mc_phi gives the value of the augmented function of the method of the
// multipliers. It holds the value first and the gradient after it. The
// block F gives the value, the gradient and then the constraints. The
// gradient of a constraint comes from a central difference.
fn mc_phi<F: fn(f64[:]) -> f64[:]>(x: f64[:], lam: f64[:], mu: f64, neq: i64, n: i64, m: i64) -> f64[>..] {
    let r = F(copy_slice(x));
    if r.len != n + 1 + m { return []; }
    var phi = r[0];
    var g: f64[>..] = [];
    var i: i64 = 0;
    while i < n { g.push(r[1 + i]); i += 1; }
    var k: i64 = 0;
    while k < m {
        let ck = r[n + 1 + k];
        var w: f64 = 0.0;
        var act = true;
        if k < neq {
            phi += lam[k] * ck + 0.5 * mu * ck * ck;
            w = lam[k] + mu * ck;
        } else {
            var t = lam[k] - mu * ck;
            if t < 0.0 { t = 0.0; act = false; }
            phi += (t * t - lam[k] * lam[k]) / (2.0 * mu);
            w = -t;
        }
        if act {
            var d: i64 = 0;
            while d < n {
                let h = 0.000001 * (1.0 + mathshim::abs_of(x[d]));
                var up: f64[>..] = copy_slice(x);
                var dn: f64[>..] = copy_slice(x);
                up[d] += h;
                dn[d] -= h;
                let cu = F(copy(up));
                let cd = F(copy(dn));
                g[d] += w * (cu[n + 1 + k] - cd[n + 1 + k]) / (2.0 * h);
                d += 1;
            }
        }
        k += 1;
    }
    var out: f64[>..] = [phi];
    i = 0;
    while i < n { out.push(g[i]); i += 1; }
    return out;
}

// minimize_constrained finds the smallest point of a function under
// equality constraints, inequality constraints and bounds. The block
// takes a vector. It gives the value, then the gradient, then the
// values of the constraints. The first neq constraints must equal
// zero. The others must be zero or above. The method of the
// multipliers with a penalty runs L-BFGS-B on each round. The
// gradients of the constraints come from central differences.
fn minimize_constrained<F: fn(f64[:]) -> f64[:]>(x0: f64[:], neq: i64, lb: f64[:] = [], ub: f64[:] = [], tol: f64 = 1.0e-8, turns: i64 = 60) -> Arr, Arr, Fault {
    let n = x0.len;
    if n < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "minimize_constrained", "the start holds no element");
    }
    if !(tol > 0.0) || turns < 1 || neq < 0 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "minimize_constrained", "the tolerance, the count of turns or neq is not valid");
    }
    var bounded = false;
    if lb.len > 0 || ub.len > 0 {
        bounded = true;
        if lb.len != n || ub.len != n {
            return empty_arr(), empty_arr(), fault(F_SHAPE, "minimize_constrained", "the bounds must hold one value per variable");
        }
        var i: i64 = 0;
        while i < n {
            if lb[i] > ub[i] {
                return empty_arr(), empty_arr(), fault(F_VALUE, "minimize_constrained", "a lower bound lies above the upper bound");
            }
            i += 1;
        }
    }
    let r0 = F(copy_slice(x0));
    if r0.len < n + 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "minimize_constrained", "the function holds a wrong count of values");
    }
    let m = r0.len - n - 1;
    if neq > m {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "minimize_constrained", "neq stands above the count of constraints");
    }
    var x: f64[>..] = mg_project(x0, lb, ub, bounded);
    var lam: f64[>..] = [];
    var k: i64 = 0;
    while k < m { lam.push(0.0); k += 1; }
    var mu: f64 = 10.0;
    var prev: f64 = 1.0e300;
    var turn: i64 = 0;
    while turn < turns {
        let lam_now: f64[>..] = copy_slice(lam);
        let xx, ff, gg = mg_lbfgs(x, lb, ub, bounded, 1.0e-12, 1.0e-14, 1.0e-9, 500, n) { v => mc_phi(v, lam_now, mu, neq, n, m) { w => F(copy(w)) } };
        if xx.len != n {
            return empty_arr(), empty_arr(), fault(F_VALUE, "minimize_constrained", "the function holds a wrong count of values");
        }
        var moved: f64 = 0.0;
        k = 0;
        while k < n { moved += mathshim::abs_of(xx[k] - x[k]); k += 1; }
        x = copy_slice(xx);
        var viol: f64 = 0.0;
        if m > 0 {
            let r = F(copy_slice(x));
            k = 0;
            while k < m {
                let ck = r[n + 1 + k];
                var v: f64 = 0.0;
                if k < neq {
                    lam[k] = lam[k] + mu * ck;
                    v = mathshim::abs_of(ck);
                } else {
                    var t = lam[k] - mu * ck;
                    if t < 0.0 { t = 0.0; }
                    lam[k] = t;
                    if ck < 0.0 { v = -ck; }
                }
                if v > viol { viol = v; }
                k += 1;
            }
        }
        if viol < tol && moved < tol { break; }
        if viol > 0.25 * prev && mu < 1.0e9 { mu = mu * 10.0; }
        prev = viol;
        turn += 1;
    }
    let rf = F(copy_slice(x));
    var s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: copy(x) }, line_arr([rf[0]]), no_fault();
}
