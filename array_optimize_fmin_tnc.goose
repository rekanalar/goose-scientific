// The method of truncated Newton for a function under bounds.
//
// The method takes the Newton step from a conjugate gradient solve of
// the system of the Hessian. The solve stops inside a trust region,
// and it stops on a direction of negative curvature. Components of
// the step that push out of the bounds stay at zero. A line search
// with the rule of Armijo accepts the step. The projected gradient
// tells when the point is at its end.
//
// The Hessian comes from a forward difference of the gradient. The
// block of the caller takes a vector and gives a list that holds the
// value of the function first and the gradient after it.
namespace scientific;

import mathshim;

// tn_clip holds a point inside the bounds.
fn tn_clip(x: f64[:], lb: f64[:], ub: f64[:], bounded: bool, n: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
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

// tn_pgrad gives the projected gradient. The point of the projection
// is the point that the step of the gradient would hold. The projected
// gradient is the distance between the point and that projection.
fn tn_pgrad(x: f64[:], g: f64[:], lb: f64[:], ub: f64[:], bounded: bool, n: i64) -> f64[>..] {
    var t: f64[>..] = [];
    var i: i64 = 0;
    while i < n { t.push(x[i] - g[i]); i += 1; }
    let p = tn_clip(t, lb, ub, bounded, n);
    var out: f64[>..] = [];
    i = 0;
    while i < n { out.push(x[i] - p[i]); i += 1; }
    return out;
}

// tn_cg runs the conjugate gradient inside a trust region. The matrix
// h holds the Hessian by rows. The vector g holds the gradient. The
// call gives the step of the Newton system, cut at the border of the
// trust region or at a direction of negative curvature.
fn tn_cg(h: f64[:], g: f64[:], n: i64, delta: f64, maxcg: i64) -> f64[>..] {
    var r: f64[>..] = [];
    var p: f64[>..] = [];
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        r.push(-g[i]);
        p.push(-g[i]);
        d.push(0.0);
        i += 1;
    }
    var rr = mg_dot(r, r);
    var k: i64 = 0;
    while k < maxcg {
        k += 1;
        // The product of the Hessian with the direction.
        var hp: f64[>..] = [];
        i = 0;
        while i < n {
            var s: f64 = 0.0;
            var j: i64 = 0;
            while j < n { s += h[i * n + j] * p[j]; j += 1; }
            hp.push(s);
            i += 1;
        }
        let curv = mg_dot(p, hp);
        let pn = mathshim::root(mg_dot(p, p));
        if curv <= 0.000000000000000001 * pn * pn {
            // The curvature is negative. The step goes to the border of
            // the trust region.
            let dn = mathshim::root(mg_dot(d, d));
            let rest = delta * delta - dn * dn;
            if rest > 0.0 && pn > 0.0 {
                let a = mathshim::root(rest) / pn;
                i = 0;
                while i < n { d[i] += a * p[i]; i += 1; }
            }
            break;
        }
        let alpha = rr / curv;
        // The trial point of the step.
        var new: f64[>..] = [];
        i = 0;
        while i < n { new.push(d[i] + alpha * p[i]); i += 1; }
        if mathshim::root(mg_dot(new, new)) >= delta {
            // The step reaches the border of the trust region.
            let dn = mathshim::root(mg_dot(d, d));
            let rest = delta * delta - dn * dn;
            if rest > 0.0 && pn > 0.0 {
                let a = mathshim::root(rest) / pn;
                i = 0;
                while i < n { d[i] += a * p[i]; i += 1; }
            }
            break;
        }
        i = 0;
        while i < n {
            d[i] = new[i];
            r[i] -= alpha * hp[i];
            i += 1;
        }
        let rr1 = mg_dot(r, r);
        // The residual is small. The system holds its answer.
        if mathshim::root(rr1) < 0.0000000001 * (1.0 + mathshim::root(mg_dot(g, g))) {
            break;
        }
        let beta = rr1 / rr;
        i = 0;
        while i < n { p[i] = -r[i] + beta * p[i]; i += 1; }
        rr = rr1;
    }
    return d;
}

// fmin_tnc finds the point where a function of several variables is at
// its smallest under bounds. The method is the method of truncated
// Newton. The block of the caller takes a vector. It gives the value
// of the function first and the gradient after it. An empty list for
// lb or ub means no bound on that side. A negative value of ftol, of
// xtol or of pgtol takes the default of SciPy. The call gives the
// point of the minimum and a fault. The fault of the limit means that
// the method did not reach the tolerance in the count of calls.
fn fmin_tnc<F: fn(f64[:]) -> f64[:]>(x0: f64[:], lb: f64[:] = [], ub: f64[:] = [], epsilon: f64 = 0.00000001, maxfun: i64 = 0, ftol: f64 = -1.0, xtol: f64 = -1.0, pgtol: f64 = -1.0) -> Arr, Fault {
    let n = x0.len;
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "fmin_tnc", "the start holds no element");
    }
    if !all_finite(x0) {
        return empty_arr(), fault(F_VALUE, "fmin_tnc", "the start holds a value that is not a number");
    }
    // SciPy takes a pair of bounds for every variable, so the package
    // takes both lists or neither.
    if (lb.len > 0 || ub.len > 0) && (lb.len != n || ub.len != n) {
        return empty_arr(), fault(F_SHAPE, "fmin_tnc", "the bounds must hold one value per variable");
    }
    var bounded = lb.len > 0 || ub.len > 0;
    if bounded {
        if (lb.len > 0 && !all_finite(lb)) || (ub.len > 0 && !all_finite(ub)) {
            return empty_arr(), fault(F_VALUE, "fmin_tnc", "a bound holds a value that is not a number");
        }
        var i: i64 = 0;
        while i < n {
            var lo: f64 = -1.0e308;
            var hi: f64 = 1.0e308;
            if lb.len > 0 { lo = lb[i]; }
            if ub.len > 0 { hi = ub[i]; }
            if lo > hi {
                return empty_arr(), fault(F_VALUE, "fmin_tnc", "a lower bound lies above the upper bound");
            }
            i += 1;
        }
    }
    if !(epsilon > 0.0) {
        return empty_arr(), fault(F_VALUE, "fmin_tnc", "the step of the difference holds a value of zero or below");
    }
    if ftol - ftol != 0.0 || xtol - xtol != 0.0 || pgtol - pgtol != 0.0 {
        return empty_arr(), fault(F_VALUE, "fmin_tnc", "a tolerance holds a value that is not a number");
    }
    // The defaults of SciPy for the tolerances.
    var tol_f = ftol;
    if tol_f < 0.0 { tol_f = 0.0; }
    var tol_x = xtol;
    if tol_x < 0.0 { tol_x = 0.000000014901161193847656; }
    var tol_g = pgtol;
    if tol_g < 0.0 { tol_g = 0.01 * mathshim::root(0.000000014901161193847656); }
    var limit = maxfun;
    if limit < 1 { limit = 10 * n; if limit < 100 { limit = 100; } }
    var maxcg: i64 = n / 2;
    if maxcg > 50 { maxcg = 50; }
    if maxcg < 1 { maxcg = 1; }
    // The start point sits inside the bounds.
    var x: f64[>..] = tn_clip(x0, lb, ub, bounded, n);
    let r = F(copy(x));
    var f: f64 = 0.0;
    var g: f64[>..] = [];
    var okf: bool = false;
    f, g, okf = mg_split(r, n);
    if !okf {
        return empty_arr(), fault(F_VALUE, "fmin_tnc", "the function holds a wrong count of values");
    }
    var nf: i64 = 1;
    var done = false;
    while nf < limit && !done {
        // The projected gradient.
        let pg = tn_pgrad(x, g, lb, ub, bounded, n);
        var pgn: f64 = 0.0;
        var i: i64 = 0;
        while i < n {
            let v = mathshim::abs_of(pg[i]);
            if v > pgn { pgn = v; }
            i += 1;
        }
        if pgn <= tol_g { done = true; break; }
        // The Hessian from a forward difference of the gradient. The
        // matrix holds one row per row of the Hessian.
        var h: f64[>..] = [];
        i = 0;
        while i < n * n { h.push(0.0); i += 1; }
        var j: i64 = 0;
        while j < n {
            let hh = epsilon * (1.0 + mathshim::abs_of(x[j]));
            var probe: f64[>..] = copy_slice(x);
            probe[j] += hh;
            let rp = F(copy(probe));
            var fp: f64 = 0.0;
            var gp: f64[>..] = [];
            var okp: bool = false;
            fp, gp, okp = mg_split(rp, n);
            if !okp {
                return empty_arr(), fault(F_VALUE, "fmin_tnc", "the function holds a wrong count of values");
            }
            nf += 1;
            i = 0;
            while i < n {
                h[i * n + j] = (gp[i] - g[i]) / hh;
                i += 1;
            }
            j += 1;
        }
        // The symmetric part of the Hessian.
        i = 0;
        while i < n {
            j = i + 1;
            while j < n {
                let v = 0.5 * (h[i * n + j] + h[j * n + i]);
                h[i * n + j] = v;
                h[j * n + i] = v;
                j += 1;
            }
            i += 1;
        }
        // The step of the Newton system in the trust region.
        var d: f64[>..] = tn_cg(h, g, n, 1.0, maxcg);
        // A component that pushes out of a bound stays at zero.
        i = 0;
        while i < n {
            if bounded {
                if x[i] <= lb[i] && d[i] < 0.0 { d[i] = 0.0; }
                if x[i] >= ub[i] && d[i] > 0.0 { d[i] = 0.0; }
            }
            i += 1;
        }
        // The line search with the rule of Armijo. The trial point sits
        // inside the bounds.
        let f0 = f;
        var taken = false;
        var t: f64 = 1.0;
        var tries: i64 = 0;
        var xnew: f64[>..] = copy_slice(x);
        var fnew: f64 = f;
        var gnew: f64[>..] = copy_slice(g);
        let slope = mg_dot(g, d);
        while tries < 20 {
            tries += 1;
            var trial: f64[>..] = [];
            i = 0;
            while i < n { trial.push(x[i] + t * d[i]); i += 1; }
            xnew = tn_clip(trial, lb, ub, bounded, n);
            let rt = F(copy(xnew));
            var okt: bool = false;
            fnew, gnew, okt = mg_split(rt, n);
            if okt {
                nf += 1;
                if fnew <= f0 + 0.0001 * t * slope {
                    taken = true;
                    break;
                }
            }
            t *= 0.5;
        }
        // The size of the step and of the fall.
        var step: f64 = 0.0;
        var xmax: f64 = 0.0;
        i = 0;
        while i < n {
            let v = mathshim::abs_of(xnew[i] - x[i]);
            if v > step { step = v; }
            let w = mathshim::abs_of(xnew[i]);
            if w > xmax { xmax = w; }
            i += 1;
        }
        x = copy_slice(xnew);
        f = fnew;
        g = copy_slice(gnew);
        if step <= tol_x * (1.0 + xmax) { done = true; }
        let df = mathshim::abs_of(f0 - f);
        var scale = mathshim::abs_of(f0);
        if scale < 1.0 { scale = 1.0; }
        if df <= tol_f * scale { done = true; }
    }
    if !done {
        return empty_arr(), fault(F_LIMIT, "fmin_tnc", "the method did not reach the tolerance in the count of calls");
    }
    return line_arr(copy(x)), no_fault();
}
