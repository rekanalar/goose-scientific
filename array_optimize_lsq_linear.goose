// The linear least squares with bounds of SciPy lsq_linear.
//
// The method trf holds the point inside the bounds and walks the
// gradient. The method bvls holds an active set and moves one
// variable at a time. Both stop at the same point, because the
// problem is convex. The unconstrained solution is the answer when it
// lies inside the bounds, as SciPy returns it at once.
namespace scientific;

import mathshim;

// lsq_bounds_at gives the lower and the upper bound at one variable.
// An empty list means the bound of minus or plus infinity.
fn lsq_bounds_at(lower: f64[:], upper: f64[:], j: i64) -> f64, f64 {
    var lo = -1.0 / 0.0;
    var hi = 1.0 / 0.0;
    if lower.len > 0 { lo = lower[j]; }
    if upper.len > 0 { hi = upper[j]; }
    return lo, hi;
}

// lsq_bounds_ok holds the checks of the bounds of lsq_linear.
fn lsq_bounds_ok(lower: f64[:], upper: f64[:], n: i64, tag: const u8[:]) -> Fault {
    if lower.len != 0 && lower.len != n {
        return fault(F_SHAPE, tag, "the lower bounds do not hold one value per variable");
    }
    if upper.len != 0 && upper.len != n {
        return fault(F_SHAPE, tag, "the upper bounds do not hold one value per variable");
    }
    var i: i64 = 0;
    while i < n {
        let lo, hi = lsq_bounds_at(lower, upper, i);
        if lo != lo {
            return fault(F_VALUE, tag, "a lower bound holds a value that is not finite and not infinite");
        }
        if hi != hi {
            return fault(F_VALUE, tag, "an upper bound holds a value that is not finite and not infinite");
        }
        if lo > hi {
            return fault(F_VALUE, tag, "a lower bound lies above its upper bound");
        }
        i += 1;
    }
    return no_fault();
}

// lsq_lsol gives the least squares solution of the system on the
// columns of the list free, with the columns of the list held fixed at
// the values of x.
fn lsq_lsol(a, b: f64[:], x: f64[:], held: bool[:], free: i64[>..]&) -> Arr, Fault {
    let m = a.shape[0];
    let n = a.shape[1];
    var rhs: f64[>..] = [];
    var i: i64 = 0;
    while i < m {
        var t = b[i];
        var j: i64 = 0;
        while j < n {
            if held[j] { t -= a.data[i * n + j] * x[j]; }
            j += 1;
        }
        rhs.push(t);
        i += 1;
    }
    var k: i64 = 0;
    var sub: f64[>..] = [];
    i = 0;
    while i < m {
        var q: i64 = 0;
        while q < free.len {
            sub.push(a.data[i * n + free[q]]);
            q += 1;
        }
        i += 1;
    }
    k = free.len;
    return lstsq(mat2(copy(sub), m, k), vec1(copy(rhs)));
}

// lsq_residual gives the residuals, the cost and the gradient of the
// system at the point x.
fn lsq_residual(a, b: f64[:], x: f64[:], n: i64) -> f64[>..], f64, f64[>..] {
    let m = a.shape[0];
    var r: f64[>..] = [];
    var i: i64 = 0;
    while i < m {
        var t = b[i];
        var j: i64 = 0;
        while j < n { t -= a.data[i * n + j] * x[j]; j += 1; }
        r.push(t);
        i += 1;
    }
    var cost: f64 = 0.0;
    i = 0;
    while i < m { cost += r[i] * r[i]; i += 1; }
    cost = 0.5 * cost;
    var g: f64[>..] = [];
    var j: i64 = 0;
    while j < n {
        var t: f64 = 0.0;
        i = 0;
        while i < m { t += a.data[i * n + j] * r[i]; i += 1; }
        g.push(t);
        j += 1;
    }
    return r, cost, g;
}

// lsq_linear solves the linear least squares with bounds on the
// variables. The matrix a holds m rows and n columns, and b holds m
// values. The lists lower and upper hold one bound per variable, or
// no value for the bounds of plus and minus infinity. The method trf
// walks the gradient and holds the point inside the bounds. The
// method bvls holds an active set and moves one variable at a time.
// Both stop when the gradient of the bounds holds a size below tol,
// or when the cost stops to fall.
fn lsq_linear(a, b: f64[:], lower: f64[:], upper: f64[:], method: const u8[:] = "trf", tol: f64 = 1.0e-10, max_iter: i64 = 0) -> Arr, Fault {
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, "lsq_linear");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    if a.ndim != 2 || a.shape.len != 2 {
        return empty_arr(), fault(F_SHAPE, "lsq_linear", "the matrix is not a matrix of two dimensions");
    }
    let m = a.shape[0];
    let n = a.shape[1];
    if m < 1 || n < 1 || m > MAX_PROGRAM_SIDE || n > MAX_PROGRAM_SIDE {
        return empty_arr(), fault(F_SHAPE, "lsq_linear", "a side of the matrix holds a count out of range");
    }
    if b.len != m {
        return empty_arr(), fault(F_SHAPE, "lsq_linear", "the right side does not agree with the matrix");
    }
    if method != "trf" && method != "bvls" {
        return empty_arr(), fault(F_VALUE, "lsq_linear", "the method holds none of trf and bvls");
    }
    if !(tol > 0.0) {
        return empty_arr(), fault(F_VALUE, "lsq_linear", "the tolerance holds a value of zero or below");
    }
    if max_iter < 0 {
        return empty_arr(), fault(F_VALUE, "lsq_linear", "the count of turns holds a value below zero");
    }
    if max_iter > 1000000 {
        return empty_arr(), fault(F_LIMIT, "lsq_linear", "the count of turns holds a value above the limit");
    }
    let gb = lsq_bounds_ok(lower, upper, n, "lsq_linear");
    if gb.kind != F_NONE { return empty_arr(), gb; }
    if !all_finite(a.data) || !all_finite(b) {
        return empty_arr(), fault(F_VALUE, "lsq_linear", "a value of the system is not finite");
    }
    // The unconstrained solution. When it lies inside the bounds it is
    // the answer, as SciPy returns it at once.
    let z, fz = lstsq(a, vec1(copy_slice(b)));
    if fz.kind != F_NONE { return empty_arr(), fz; }
    var inside = true;
    var j: i64 = 0;
    while j < n {
        let lo, hi = lsq_bounds_at(lower, upper, j);
        if z.data[j] < lo || z.data[j] > hi { inside = false; }
        j += 1;
    }
    if inside { return z, no_fault(); }
    if method == "bvls" {
        return lsq_bvls(a, b, z, lower, upper, tol, max_iter);
    }
    return lsq_trf(a, b, z, lower, upper, tol, max_iter);
}

// lsq_bvls holds the active set method of Stark and Parker. It starts
// from the variables that violate a bound, and it moves one variable
// at a time into the free set while the gradient of the bounds is
// large.
fn lsq_bvls(a, b: f64[:], z, lower: f64[:], upper: f64[:], tol: f64, max_iter: i64) -> Arr, Fault {
    let n = a.shape[1];
    var x: f64[>..] = [];
    var on: i64[>..] = [];
    var j: i64 = 0;
    while j < n {
        let lo, hi = lsq_bounds_at(lower, upper, j);
        if z.data[j] <= lo { x.push(lo); on.push(-1); }
        else if z.data[j] >= hi { x.push(hi); on.push(1); }
        else { x.push(z.data[j]); on.push(0); }
        j += 1;
    }
    var held: bool[>..] = [];
    j = 0;
    while j < n { held.push(on[j] != 0); j += 1; }
    var iter: i64 = 0;
    // The loop of the start. It sets the variables that violate a
    // bound to their bound until the free solution holds inside.
    var free: i64[>..] = [];
    j = 0;
    while j < n { if on[j] == 0 { free.push(j); } j += 1; }
    while free.len > 0 {
        iter += 1;
        let zz, fz = lsq_lsol(a, b, x, held, free);
        if fz.kind != F_NONE { return empty_arr(), fz; }
        var hit = false;
        j = 0;
        while j < free.len {
            let k = free[j];
            let lo, hi = lsq_bounds_at(lower, upper, k);
            if zz.data[j] <= lo {
                x[k] = lo; on[k] = -1; held[k] = true; hit = true;
            } else if zz.data[j] >= hi {
                x[k] = hi; on[k] = 1; held[k] = true; hit = true;
            } else {
                x[k] = zz.data[j];
            }
            j += 1;
        }
        free = [];
        j = 0;
        while j < n { if on[j] == 0 { free.push(j); } j += 1; }
        if !hit { break; }
    }
    // The main loop of the method. It moves the variable with the
    // largest gradient against its bound into the free set.
    var limit = max_iter;
    if limit == 0 { limit = n; }
    limit += iter;
    let _, cost0, grad0 = lsq_residual(a, b, x, n);
    var cost = cost0;
    var grad = copy(grad0);
    var turns: i64 = 0;
    while turns < limit {
        turns += 1;
        // The size of the gradient of the bounds.
        var opt: f64 = 0.0;
        j = 0;
        while j < n {
            var t: f64 = 0.0;
            if on[j] == 0 { t = mathshim::abs_of(grad[j]); }
            else { t = grad[j] * (on[j] as f64); }
            if t > opt { opt = t; }
            j += 1;
        }
        if opt < tol { break; }
        var pick: i64 = 0;
        var best: f64 = -1.0e308;
        j = 0;
        while j < n {
            let t = grad[j] * (on[j] as f64);
            if t > best { best = t; pick = j; }
            j += 1;
        }
        on[pick] = 0;
        held[pick] = false;
        // The inner loop. It walks the free solution toward the
        // bounds and holds the first variable that reaches a bound.
        var go = true;
        while go {
            free = [];
            j = 0;
            while j < n { if on[j] == 0 { free.push(j); } j += 1; }
            let zz, fz = lsq_lsol(a, b, x, held, free);
            if fz.kind != F_NONE { return empty_arr(), fz; }
            var viol: i64 = -1;
            var alpha: f64 = 1.0e308;
            var at: i64 = -1;
            j = 0;
            while j < free.len {
                let k = free[j];
                let lo, hi = lsq_bounds_at(lower, upper, k);
                if zz.data[j] <= lo {
                    let t = (lo - x[k]) / (zz.data[j] - x[k]);
                    if t < alpha { alpha = t; viol = -1; at = k; }
                } else if zz.data[j] >= hi {
                    let t = (hi - x[k]) / (zz.data[j] - x[k]);
                    if t < alpha { alpha = t; viol = 1; at = k; }
                }
                j += 1;
            }
            if at >= 0 {
                j = 0;
                while j < free.len {
                    let k = free[j];
                    x[k] = x[k] + alpha * (zz.data[j] - x[k]);
                    j += 1;
                }
                x[at] = (if viol < 0 { lower[at] } else { upper[at] });
                on[at] = viol;
                held[at] = true;
            } else {
                j = 0;
                while j < free.len {
                    x[free[j]] = zz.data[j];
                    j += 1;
                }
                go = false;
            }
        }
        let _, cost2, g2 = lsq_residual(a, b, x, n);
        grad = copy(g2);
        let change = cost - cost2;
        cost = cost2;
        if change < tol * cost { break; }
    }
    var out: f64[>..] = [];
    j = 0;
    while j < n { out.push(x[j]); j += 1; }
    return vec1(copy(out)), no_fault();
}

// lsq_trf holds the method of the trust region that reflects. The
// point stays inside the bounds. The step solves the normal equations
// with a damping on the directions that stand near a bound, and a
// backtracking holds the cost falling.
fn lsq_trf(a, b: f64[:], z, lower: f64[:], upper: f64[:], tol: f64, max_iter: i64) -> Arr, Fault {
    let n = a.shape[1];
    var x: f64[>..] = [];
    var j: i64 = 0;
    while j < n {
        var lo = -1.0;
        var hi = 1.0;
        if lower.len > 0 && !(lower[j] == -1.0 / 0.0) { lo = lower[j]; }
        if upper.len > 0 && !(upper[j] == 1.0 / 0.0) { hi = upper[j]; }
        var t = 0.5 * (lo + hi);
        if lower.len > 0 && t < lower[j] { t = lower[j]; }
        if upper.len > 0 && t > upper[j] { t = upper[j]; }
        x.push(t);
        j += 1;
    }
    var damping: f64 = 1.0e-3;
    var iter: i64 = 0;
    var limit = max_iter;
    if limit == 0 { limit = 100; }
    while iter < limit {
        iter += 1;
        var cost: f64 = 0.0;
        var grad: f64[>..] = [];
        let _, cv, gv = lsq_residual(a, b, x, n);
        cost = cv;
        grad = copy(gv);
        // The gradient of the bounds. A direction into a bound holds
        // no size, and a direction with no bound holds the size one.
        var opt: f64 = 0.0;
        j = 0;
        while j < n {
            let lo, hi = lsq_bounds_at(lower, upper, j);
            var t: f64 = 0.0;
            if grad[j] < 0.0 {
                var d = 1.0;
                if !(hi == 1.0 / 0.0) { d = hi - x[j]; }
                t = grad[j] * d;
            } else {
                var d = 1.0;
                if !(lo == -1.0 / 0.0) { d = x[j] - lo; }
                t = -grad[j] * d;
            }
            if t > opt { opt = t; }
            j += 1;
        }
        if opt <= tol { break; }
        // The normal matrix and the damping. A direction that stands
        // near a bound holds a large damping.
        let m = a.shape[0];
        var sys: f64[>..] = [];
        var i: i64 = 0;
        while i < n * n { sys.push(0.0); i += 1; }
        i = 0;
        while i < n {
            var q: i64 = 0;
            while q < n {
                var t: f64 = 0.0;
                var row: i64 = 0;
                while row < m { t += a.data[row * n + i] * a.data[row * n + q]; row += 1; }
                sys[i * n + q] = t;
                q += 1;
            }
            i += 1;
        }
        i = 0;
        while i < n {
            let lo, hi = lsq_bounds_at(lower, upper, i);
            var d: f64 = 1.0;
            if !(lo == -1.0 / 0.0) { d = x[i] - lo; }
            if !(hi == 1.0 / 0.0) {
                let e = hi - x[i];
                if e < d { d = e; }
            }
            if d < 1.0e-8 { d = 1.0e-8; }
            if d > 1.0 { d = 1.0; }
            sys[i * n + i] += damping / (d * d);
            i += 1;
        }
        var rhs: f64[>..] = [];
        j = 0;
        while j < n { rhs.push(-grad[j]); j += 1; }
        let p, fp = solve(mat2(copy(sys), n, n), vec1(copy(rhs)));
        if fp.kind != F_NONE {
            damping = damping * 10.0;
            continue;
        }
        // The backtracking holds the point inside the bounds and the
        // cost falling.
        var alpha: f64 = 1.0;
        var ok = false;
        var tries: i64 = 0;
        while tries < 60 {
            var trial: f64[>..] = [];
            j = 0;
            while j < n {
                var t = x[j] + alpha * p.data[j];
                if lower.len > 0 && t < lower[j] { t = lower[j]; }
                if upper.len > 0 && t > upper[j] { t = upper[j]; }
                trial.push(t);
                j += 1;
            }
            let _, ct, _ = lsq_residual(a, b, trial, n);
            var gd: f64 = 0.0;
            j = 0;
            while j < n { gd += grad[j] * p.data[j]; j += 1; }
            if ct <= cost - 1.0e-4 * alpha * gd || ct < cost {
                j = 0;
                while j < n { x[j] = trial[j]; j += 1; }
                let change = cost - ct;
                cost = ct;
                if change < tol * cost { iter = limit; }
                ok = true;
                break;
            }
            alpha = alpha * 0.5;
            tries += 1;
        }
        if !ok {
            damping = damping * 10.0;
            if damping > 1.0e30 { break; }
        } else {
            damping = damping * 0.5;
            if damping < 1.0e-20 { damping = 1.0e-20; }
        }
    }
    var out: f64[>..] = [];
    j = 0;
    while j < n { out.push(x[j]); j += 1; }
    return vec1(copy(out)), no_fault();
}
