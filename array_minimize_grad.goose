// The methods of minimization that use the gradient of the function.
// The methods are the method of Broyden, Fletcher and Goldfarb, the
// method of Fletcher and Reeves, and the limited memory form of the
// first method. The method of Nelder and Mead sits beside them in the
// file of the minimization.
namespace scientific;

import mathshim;

// The count of pairs that the limited memory method keeps.
const LBFGS_MEMORY: i64 = 10;

// mg_split takes the vector of the caller and gives the value and the
// gradient. The vector holds the value first and the gradient after
// it. A vector of a wrong count gives false.
fn mg_split(r: f64[:], n: i64) -> f64, f64[>..], bool {
    if r.len != n + 1 { return 0.0, [], false; }
    var g: f64[>..] = [];
    var i: i64 = 1;
    while i < r.len { g.push(r[i]); i += 1; }
    return r[0], g, true;
}

// mg_dot gives the product of the scalar product of two vectors.
fn mg_dot(a: f64[:], b: f64[:]) -> f64 {
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < a.len { s += a[i] * b[i]; i += 1; }
    return s;
}

// mg_norm gives the length of a vector.
fn mg_norm(a: f64[:]) -> f64 {
    return mathshim::root(mg_dot(a, a));
}

// mg_project clips a point to the bounds. A point outside the bounds
// moves to the border.
fn mg_project(x: f64[:], lb: f64[:], ub: f64[:], bounded: bool) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len {
        var v = x[i];
        if bounded {
            if v < lb[i] { v = lb[i]; }
            if v > ub[i] { v = ub[i]; }
        }
        out.push(v);
        i += 1;
    }
    return out;
}

// mg_feasible says whether a point lies inside the bounds.
fn mg_feasible(x: f64[:], lb: f64[:], ub: f64[:], bounded: bool) -> bool {
    if !bounded { return true; }
    var i: i64 = 0;
    while i < x.len {
        if x[i] < lb[i] || x[i] > ub[i] { return false; }
        i += 1;
    }
    return true;
}

// mg_line gives the step of the line search. It starts at the whole
// step and halves it until the value falls enough. The test of Armijo
// needs a fall of a part of the slope.
fn mg_line<F: fn(f64[:]) -> f64[:]>(x: f64[:], p: f64[:], f0: f64, slope0: f64, c2: f64, lb: f64[:], ub: f64[:], bounded: bool, n: i64) -> f64, f64[>..], f64 {
    // The conditions of Wolfe ask for a fall of a part of the slope and
    // a slope that falls to a part of the slope of the start.
    let c1: f64 = 0.0001;
    var lo: f64 = 0.0;
    var hi: f64 = 0.0;
    var flo: f64 = f0;
    var dlo: f64 = slope0;
    var dhi: f64 = slope0;
    var alpha: f64 = 1.0;
    var mode: i64 = 0;
    var best_alpha: f64 = 0.0;
    var best_f: f64 = f0;
    var best_g: f64[>..] = [];
    var turn: i64 = 0;
    while turn < 60 {
        var a = alpha;
        if mode == 1 {
            // The step of the secant between the two ends. It stays
            // inside the interval, or the middle of it is taken.
            a = lo - dlo * (hi - lo) / (dhi - dlo);
            let margin = 0.1 * (hi - lo);
            if !(a > lo + margin && a < hi - margin) { a = 0.5 * (lo + hi); }
        }
        let trial = mg_trial(x, p, a, lb, ub, bounded, n);
        let r = F(copy(trial));
        let v, gv, ok = mg_split(r, n);
        if !ok {
            if mode == 0 { alpha = alpha * 0.5; turn += 1; continue; }
            hi = 0.5 * (lo + hi);
            turn += 1;
            continue;
        }
        let d = mg_dot(gv, p);
        if v <= f0 + c1 * a * slope0 && v < best_f {
            best_alpha = a;
            best_f = v;
            best_g = copy(gv);
        }
        if mode == 0 {
            if v > f0 + c1 * a * slope0 || (v >= flo && a > lo) {
                hi = a;
                dhi = d;
                mode = 1;
            } else {
                if mathshim::abs_of(d) <= c2 * mathshim::abs_of(slope0) { return a, gv, v; }
                lo = a;
                flo = v;
                dlo = d;
                alpha = a * 2.0;
                if alpha > 1024.0 { alpha = 1024.0; }
            }
        } else {
            if v > f0 + c1 * a * slope0 || v >= flo {
                hi = a;
                dhi = d;
            } else {
                if mathshim::abs_of(d) <= c2 * mathshim::abs_of(slope0) { return a, gv, v; }
                if d * (hi - lo) >= 0.0 {
                    hi = lo;
                    dhi = dlo;
                }
                lo = a;
                flo = v;
                dlo = d;
            }
        }
        turn += 1;
    }
    return best_alpha, best_g, best_f;
}

// mg_block drops the parts of a direction that push out of the bounds.
// A variable at its bound that the gradient pulls outward stays.
fn mg_block(x: f64[:], p: f64[:], g: f64[:], lb: f64[:], ub: f64[:], bounded: bool) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len {
        var v = p[i];
        if bounded {
            if x[i] <= lb[i] && g[i] > 0.0 { v = 0.0; }
            if x[i] >= ub[i] && g[i] < 0.0 { v = 0.0; }
        }
        out.push(v);
        i += 1;
    }
    return out;
}

// mg_trial gives the point of a step, clipped to the bounds.
fn mg_trial(x: f64[:], p: f64[:], alpha: f64, lb: f64[:], ub: f64[:], bounded: bool, n: i64) -> f64[>..] {
    var trial: f64[>..] = [];
    var i: i64 = 0;
    while i < n { trial.push(x[i] + alpha * p[i]); i += 1; }
    let clipped = mg_project(trial, lb, ub, bounded);
    return clipped;
}

// mg_bfgs_update gives the new estimate of the inverse of the Hessian.
// The formula is the update of Broyden, Fletcher and Goldfarb.
fn mg_bfgs_update(h: f64[:], s: f64[:], y: f64[:], sy: f64, n: i64) -> f64[>..] {
    let rho = 1.0 / sy;
    // The product of the matrix with the change of the gradient.
    var hy: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var t: f64 = 0.0;
        var j: i64 = 0;
        while j < n { t += h[i * n + j] * y[j]; j += 1; }
        hy.push(t);
        i += 1;
    }
    // The right factor: the matrix minus rho times y s transpose.
    var hb: f64[>..] = [];
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n { hb.push(h[i * n + j] - rho * hy[i] * s[j]); j += 1; }
        i += 1;
    }
    // The left factor multiplies from the left.
    var out: f64[>..] = [];
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            var t: f64 = hb[i * n + j];
            var k: i64 = 0;
            while k < n { t -= rho * s[i] * y[k] * hb[k * n + j]; k += 1; }
            out.push(t + rho * s[i] * s[j]);
            j += 1;
        }
        i += 1;
    }
    return out;
}

// mg_bfgs runs the method of Broyden, Fletcher and Goldfarb. The
// matrix holds the inverse of the Hessian by the estimate of the
// secant pair.
fn mg_bfgs<F: fn(f64[:]) -> f64[:]>(x0: f64[:], lb: f64[:], ub: f64[:], bounded: bool, xtol: f64, ftol: f64, gtol: f64, turns: i64, n: i64) -> f64[>..], f64, f64[>..] {
    let xstart = mg_project(x0, lb, ub, bounded);
    var x = copy(xstart);
    let r = F(copy(x));
    var f, g, ok = mg_split(r, n);
    if !ok { return [], 0.0, []; }
    // The matrix starts as the identity.
    var h: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n {
        if i % (n + 1) == 0 { h.push(1.0); } else { h.push(0.0); }
        i += 1;
    }
    var turn: i64 = 0;
    while turn < turns {
        var gn: f64 = mathshim::root(mg_dot(g, g));
        if gn < gtol { break; }
        // The direction of the step.
        var p: f64[>..] = [];
        i = 0;
        while i < n {
            var s: f64 = 0.0;
            var j: i64 = 0;
            while j < n { s += h[i * n + j] * g[j]; j += 1; }
            p.push(-s);
            i += 1;
        }
        let pblock = mg_block(x, p, g, lb, ub, bounded);
        p = copy(pblock);
        var slope = mg_dot(g, p);
        if slope >= 0.0 {
            // The estimate lost the way down. The gradient takes over.
            let q = mg_block(x, g, g, lb, ub, bounded);
            i = 0;
            while i < n { p[i] = -q[i]; i += 1; }
            slope = mg_dot(g, p);
        }
        let alpha, gnew, fnew = mg_line(x, p, f, slope, 0.9, lb, ub, bounded, n) { t => F(copy(t)) };
        if gnew.len != n { break; }
        // The step and the change of the gradient.
        var xnew: f64[>..] = [];
        i = 0;
        while i < n { xnew.push(x[i] + alpha * p[i]); i += 1; }
        let xclip = mg_project(xnew, lb, ub, bounded);
        xnew = copy(xclip);
        var s: f64[>..] = [];
        var y: f64[>..] = [];
        i = 0;
        while i < n { s.push(xnew[i] - x[i]); y.push(gnew[i] - g[i]); i += 1; }
        let sy = mg_dot(s, y);
        let dx = mg_norm(s);
        let df = mathshim::abs_of(fnew - f);
        x = copy(xnew);
        f = fnew;
        g = copy(gnew);
        if sy > 0.00000000000000001 {
            let hn = mg_bfgs_update(h, s, y, sy, n);
            h = copy(hn);
        }
        if dx < xtol * (xtol + mg_norm(x)) { break; }
        turn += 1;
    }
    return x, f, g;
}

// mg_cg runs the method of Fletcher and Reeves. The direction restarts
// when it no longer goes down.
fn mg_cg<F: fn(f64[:]) -> f64[:]>(x0: f64[:], lb: f64[:], ub: f64[:], bounded: bool, xtol: f64, ftol: f64, gtol: f64, turns: i64, n: i64) -> f64[>..], f64, f64[>..] {
    let xstart = mg_project(x0, lb, ub, bounded);
    var x = copy(xstart);
    let r = F(copy(x));
    var f, g, ok = mg_split(r, n);
    if !ok { return [], 0.0, []; }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n { d.push(-g[i]); i += 1; }
    var turn: i64 = 0;
    while turn < turns {
        if mathshim::root(mg_dot(g, g)) < gtol { break; }
        let dblock = mg_block(x, d, g, lb, ub, bounded);
        d = copy(dblock);
        var slope = mg_dot(g, d);
        if slope >= 0.0 {
            let q = mg_block(x, g, g, lb, ub, bounded);
            i = 0;
            while i < n { d[i] = -q[i]; i += 1; }
            slope = mg_dot(g, d);
        }
        let alpha, gnew, fnew = mg_line(x, d, f, slope, 0.4, lb, ub, bounded, n) { t => F(copy(t)) };
        if gnew.len != n { break; }
        var xnew: f64[>..] = [];
        i = 0;
        while i < n { xnew.push(x[i] + alpha * d[i]); i += 1; }
        let xclip = mg_project(xnew, lb, ub, bounded);
        xnew = copy(xclip);
        let gg = mg_dot(g, g);
        // The rule of Polak and Ribiere with a part that holds no
        // negative value.
        var beta: f64 = 0.0;
        if gg > 0.0 {
            var diff: f64[>..] = [];
            i = 0;
            while i < n { diff.push(gnew[i] - g[i]); i += 1; }
            beta = mg_dot(diff, gnew) / gg;
            if beta < 0.0 { beta = 0.0; }
        }
        var dnew: f64[>..] = [];
        i = 0;
        while i < n { dnew.push(-gnew[i] + beta * d[i]); i += 1; }
        // The direction must fall enough, else the direction restarts.
        if mg_dot(dnew, gnew) > -0.01 * mg_dot(gnew, gnew) {
            i = 0;
            while i < n { dnew[i] = -gnew[i]; i += 1; }
        }
        let df = mathshim::abs_of(fnew - f);
        let dx = mg_norm(xnew);
        x = copy(xnew);
        f = fnew;
        g = copy(gnew);
        d = copy(dnew);
        if dx < xtol * (xtol + mg_norm(x)) { break; }
        turn += 1;
    }
    return x, f, g;
}

// mg_lbfgs runs the limited memory form of the method of Broyden,
// Fletcher and Goldfarb. The pairs of the steps and the changes of the
// gradient give the direction. A point outside the bounds moves to the
// border.
fn mg_lbfgs<F: fn(f64[:]) -> f64[:]>(x0: f64[:], lb: f64[:], ub: f64[:], bounded: bool, xtol: f64, ftol: f64, gtol: f64, turns: i64, n: i64) -> f64[>..], f64, f64[>..] {
    let xstart = mg_project(x0, lb, ub, bounded);
    var x = copy(xstart);
    let r = F(copy(x));
    var f, g, ok = mg_split(r, n);
    if !ok { return [], 0.0, []; }
    var sv: f64[>..] = [];
    var yv: f64[>..] = [];
    var rho: f64[>..] = [];
    var turn: i64 = 0;
    while turn < turns {
        if mathshim::root(mg_dot(g, g)) < gtol { break; }
        // The two loops that give the direction.
        var q: f64[>..] = copy(g);
        var count = sv.len / n;
        var alpha: f64[>..] = [];
        var ii: i64 = 0;
        while ii < count { alpha.push(0.0); ii += 1; }
        var k: i64 = count - 1;
        while k >= 0 {
            var srow: f64[>..] = [];
            var j: i64 = 0;
            while j < n { srow.push(sv[k * n + j]); j += 1; }
            let a = rho[k] * mg_dot(srow, q);
            alpha[k] = a;
            j = 0;
            while j < n { q[j] -= a * yv[k * n + j]; j += 1; }
            k -= 1;
        }
        var scale: f64 = 1.0;
        if count > 0 {
            let last = count - 1;
            var srow: f64[>..] = [];
            var yrow: f64[>..] = [];
            var j: i64 = 0;
            while j < n { srow.push(sv[last * n + j]); yrow.push(yv[last * n + j]); j += 1; }
            let sy = mg_dot(srow, yrow);
            if sy > 0.0 { scale = mg_dot(yrow, yrow) / sy; }
        }
        var i: i64 = 0;
        while i < n { q[i] = scale * q[i]; i += 1; }
        k = 0;
        while k < count {
            var yrow: f64[>..] = [];
            var j: i64 = 0;
            while j < n { yrow.push(yv[k * n + j]); j += 1; }
            let beta = rho[k] * mg_dot(yrow, q);
            var srow: f64[>..] = [];
            j = 0;
            while j < n { srow.push(sv[k * n + j]); j += 1; }
            j = 0;
            while j < n { q[j] += srow[j] * (alpha[k] - beta); j += 1; }
            k += 1;
        }
        var p: f64[>..] = [];
        i = 0;
        while i < n { p.push(-q[i]); i += 1; }
        let pblock = mg_block(x, p, g, lb, ub, bounded);
        p = copy(pblock);
        var slope = mg_dot(g, p);
        if slope >= 0.0 {
            let q = mg_block(x, g, g, lb, ub, bounded);
            i = 0;
            while i < n { p[i] = -q[i]; i += 1; }
            slope = mg_dot(g, p);
        }
        let alpha, gnew, fnew = mg_line(x, p, f, slope, 0.9, lb, ub, bounded, n) { t => F(copy(t)) };
        if gnew.len != n { break; }
        var xnew: f64[>..] = [];
        i = 0;
        while i < n { xnew.push(x[i] + alpha * p[i]); i += 1; }
        let xclip = mg_project(xnew, lb, ub, bounded);
        xnew = copy(xclip);
        var s: f64[>..] = [];
        var y: f64[>..] = [];
        i = 0;
        while i < n { s.push(xnew[i] - x[i]); y.push(gnew[i] - g[i]); i += 1; }
        let sy = mg_dot(s, y);
        let df = mathshim::abs_of(fnew - f);
        let dx = mg_norm(s);
        x = copy(xnew);
        f = fnew;
        g = copy(gnew);
        if sy > 0.00000000000000001 {
            if sv.len / n >= LBFGS_MEMORY {
                // The oldest pair leaves the list.
                var ns: f64[>..] = [];
                var ny: f64[>..] = [];
                var nr: f64[>..] = [];
                var j: i64 = 0;
                while j < LBFGS_MEMORY - 1 {
                    var q2: i64 = 0;
                    while q2 < n { ns.push(sv[(j + 1) * n + q2]); ny.push(yv[(j + 1) * n + q2]); q2 += 1; }
                    nr.push(rho[j + 1]);
                    j += 1;
                }
                sv = copy(ns);
                yv = copy(ny);
                rho = copy(nr);
            }
            var j: i64 = 0;
            while j < n { sv.push(s[j]); yv.push(y[j]); j += 1; }
            rho.push(1.0 / sy);
        }
        if df < ftol { break; }
        if dx < xtol * (xtol + mg_norm(x)) { break; }
        turn += 1;
    }
    return x, f, g;
}

// mg_newtoncg runs the method of Newton with the conjugate gradient
// for the step. The product of the Hessian and a vector comes from a
// small difference of two gradients. The inner loop stops on a
// negative curvature, as SciPy does.
fn mg_newtoncg<F: fn(f64[:]) -> f64[:]>(x0: f64[:], xtol: f64, turns: i64, n: i64) -> f64[>..], f64, f64[>..] {
    var x: f64[>..] = copy_slice(x0);
    let r = F(copy(x));
    var f, g, ok = mg_split(r, n);
    if !ok { return [], 0.0, []; }
    let eps: f64 = 1.4901161193847656e-08;
    let limit = xtol * n;
    var upd: f64 = 2.0 * limit + 1.0;
    var turn: i64 = 0;
    while upd > limit && turn < turns {
        // The inner loop solves H p = -g.
        var mag: f64 = 0.0;
        var i: i64 = 0;
        while i < n { mag += mathshim::abs_of(g[i]); i += 1; }
        var term = 0.5;
        if mathshim::root(mag) < term { term = mathshim::root(mag); }
        term = term * mag;
        var xs: f64[>..] = [];
        var ri: f64[>..] = copy_slice(g);
        var ps: f64[>..] = [];
        i = 0;
        while i < n { xs.push(0.0); ps.push(-g[i]); i += 1; }
        var dri0 = mg_dot(ri, ri);
        var k: i64 = 0;
        while true {
            var rsum: f64 = 0.0;
            i = 0;
            while i < n { rsum += mathshim::abs_of(ri[i]); i += 1; }
            if rsum <= term { break; }
            var probe: f64[>..] = [];
            i = 0;
            while i < n { probe.push(x[i] + eps * ps[i]); i += 1; }
            let rp = F(copy(probe));
            let fp, gp, okp = mg_split(rp, n);
            if !okp { return [], 0.0, []; }
            var ap: f64[>..] = [];
            i = 0;
            while i < n { ap.push((gp[i] - g[i]) / eps); i += 1; }
            let curv = mg_dot(ps, ap);
            if curv == 0.0 { break; }
            if curv < 0.0 {
                if k == 0 {
                    i = 0;
                    while i < n { xs[i] = dri0 / (-curv) * (-g[i]); i += 1; }
                }
                break;
            }
            let alpha = dri0 / curv;
            i = 0;
            while i < n { xs[i] += alpha * ps[i]; ri[i] += alpha * ap[i]; i += 1; }
            let dri1 = mg_dot(ri, ri);
            let beta = dri1 / dri0;
            i = 0;
            while i < n { ps[i] = -ri[i] + beta * ps[i]; i += 1; }
            dri0 = dri1;
            k += 1;
            if k > 20 * n + 100 { break; }
        }
        let slope = mg_dot(g, xs);
        if slope >= 0.0 { break; }
        let alpha, gnew, fnew = mg_line(x, xs, f, slope, 0.9, x0, x0, false, n) { t => F(copy(t)) };
        if gnew.len != n { break; }
        upd = 0.0;
        i = 0;
        while i < n {
            let u = alpha * xs[i];
            x[i] += u;
            upd += mathshim::abs_of(u);
            i += 1;
        }
        f = fnew;
        g = copy(gnew);
        turn += 1;
    }
    return x, f, g;
}

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
