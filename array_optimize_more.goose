// The plain calls of the optimization of SciPy.
//
// The file holds the calls that need no solver of a method. The
// function of Rosenbrock comes with its gradient and its Hessian. The
// check of a gradient compares a gradient with a finite difference. The
// search of a fixed point, the search of a bracket and the search on a
// grid find a point without a gradient. The regression that must not
// fall fits a monotone function.
namespace scientific;

import mathshim;

// The step of the finite difference of the check of a gradient. It is
// the square root of the smallest difference of two numbers.
const CHECK_GRAD_STEP: f64 = 1.4901161193847656e-8;

// The limit of the count of points of the grid of the search on a grid.
const BRUTE_POINTS_MAX: i64 = 1000000;

// rosen gives the value of the function of Rosenbrock. The function
// holds at least two values.
fn rosen(x: f64[:]) -> f64 {
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i + 1 < x.len {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        s += 100.0 * a * a + b * b;
        i += 1;
    }
    return s;
}

// rosen_der gives the gradient of the function of Rosenbrock. The
// result holds one value per value of the point. A point of fewer than
// two values gives an empty result.
fn rosen_der(x: f64[:]) -> Arr {
    let n = x.len;
    if n < 2 { return empty_arr(); }
    var g: f64[>..] = [];
    var i: i64 = 0;
    while i < n { g.push(0.0); i += 1; }
    i = 1;
    while i + 1 < n {
        g[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
             - 400.0 * (x[i + 1] - x[i] * x[i]) * x[i]
             - 2.0 * (1.0 - x[i]);
        i += 1;
    }
    g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
    g[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);
    return flt_arr(copy(g));
}

// rosen_hess gives the Hessian of the function of Rosenbrock. The
// result holds a matrix of the side of the point. A point of fewer than
// two values gives an empty result.
fn rosen_hess(x: f64[:]) -> Arr {
    let n = x.len;
    if n < 2 { return empty_arr(); }
    var h: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { h.push(0.0); i += 1; }
    h[0] = 1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0;
    h[(n - 1) * n + (n - 1)] = 200.0;
    i = 1;
    while i + 1 < n {
        h[i * n + i] = 202.0 + 1200.0 * x[i] * x[i] - 400.0 * x[i + 1];
        i += 1;
    }
    // The entries beside the diagonal hold the value of the row above.
    i = 0;
    while i + 1 < n {
        h[i * n + i + 1] = -400.0 * x[i];
        h[(i + 1) * n + i] = -400.0 * x[i];
        i += 1;
    }
    return mat2(copy(h), n, n);
}

// rosen_hess_prod gives the product of the Hessian of the function of
// Rosenbrock and a vector. The result holds one value per value of the
// point.
fn rosen_hess_prod(x: f64[:], p: f64[:]) -> Arr {
    let n = x.len;
    if n < 2 || p.len != n { return empty_arr(); }
    var h: f64[>..] = [];
    var i: i64 = 0;
    while i < n { h.push(0.0); i += 1; }
    h[0] = (1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0) * p[0] - 400.0 * x[0] * p[1];
    h[n - 1] = 200.0 * p[n - 1] - 400.0 * x[n - 2] * p[n - 2];
    i = 1;
    while i + 1 < n {
        h[i] = -400.0 * x[i - 1] * p[i - 1]
             + (202.0 + 1200.0 * x[i] * x[i] - 400.0 * x[i + 1]) * p[i]
             - 400.0 * x[i] * p[i + 1];
        i += 1;
    }
    return flt_arr(copy(h));
}

// check_grad gives the distance between the gradient of a function and
// the gradient of the finite difference. The block of the caller takes
// a vector. It gives the value first and the gradient after it, as
// minimize_grad does. SciPy gives the same value.
fn check_grad<F: fn(f64[:]) -> f64[:]>(x: f64[:]) -> f64, Fault {
    let n = x.len;
    if n < 1 {
        return 0.0, fault(F_VALUE, "check_grad", "the point holds no element");
    }
    let r = F(x);
    if r.len != n + 1 {
        return 0.0, fault(F_SHAPE, "check_grad", "the block does not give the value and the gradient");
    }
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        var p: f64[>..] = [];
        var j: i64 = 0;
        while j < n { p.push(x[j]); j += 1; }
        p[i] = p[i] + CHECK_GRAD_STEP;
        let r2 = F(copy(p));
        if r2.len < 1 {
            return 0.0, fault(F_SHAPE, "check_grad", "the block does not give the value");
        }
        let num = (r2[0] - r[0]) / CHECK_GRAD_STEP;
        let d = r[i + 1] - num;
        s += d * d;
        i += 1;
    }
    return mathshim::root(s), no_fault();
}

// fixed_point finds the point that a function does not move. The block
// of the caller takes a vector and gives the image of the vector. The
// rule is the rule of Steffensen with the second difference, as SciPy
// uses by default.
fn fixed_point<F: fn(f64[:]) -> f64[:]>(x0: f64[:], xtol: f64 = 1.0e-8, maxiter: i64 = 500) -> Arr, Fault {
    let n = x0.len;
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "fixed_point", "the start holds no element");
    }
    if xtol <= 0.0 {
        return empty_arr(), fault(F_VALUE, "fixed_point", "the tolerance holds a value of zero or below");
    }
    if maxiter < 1 {
        return empty_arr(), fault(F_VALUE, "fixed_point", "the count of turns holds a value below one");
    }
    var p0: f64[>..] = [];
    var i: i64 = 0;
    while i < n { p0.push(x0[i]); i += 1; }
    var p: f64[>..] = [];
    var turn: i64 = 0;
    while turn < maxiter {
        let p1 = F(copy(p0));
        if p1.len != n {
            return empty_arr(), fault(F_SHAPE, "fixed_point", "the block does not give a vector of the same count");
        }
        let p2 = F(copy(p1));
        if p2.len != n {
            return empty_arr(), fault(F_SHAPE, "fixed_point", "the block does not give a vector of the same count");
        }
        // The second difference gives the rule of Steffensen.
        p = [];
        i = 0;
        while i < n {
            let d = p2[i] - 2.0 * p1[i] + p0[i];
            if d != 0.0 {
                let q = p1[i] - p0[i];
                p.push(p0[i] - q * q / d);
            } else {
                p.push(p2[i]);
            }
            i += 1;
        }
        // The error is the change of every value, relative to the start.
        var small = true;
        i = 0;
        while i < n {
            var e = p[i];
            if p0[i] != 0.0 { e = (p[i] - p0[i]) / p0[i]; }
            if !(mathshim::abs_of(e) < xtol) { small = false; }
            i += 1;
        }
        if small { return flt_arr(copy(p)), no_fault(); }
        p0 = copy(p);
        turn += 1;
    }
    return flt_arr(copy(p)), fault(F_VALUE, "fixed_point", "the search did not reach the tolerance");
}

// line_search gives the length of the step of the search of the line.
// The block of the caller gives the value and the gradient of a point.
// The conditions are the strong Wolfe conditions, as SciPy uses, and the
// rule of the search is the rule of Wright and Nocedal, so the step of
// the package matches the step of SciPy. The point moves by the step
// along the direction.
fn line_search<F: fn(f64[:]) -> f64[:]>(x: f64[:], p: f64[:], f0: f64, slope0: f64, c2: f64 = 0.9) -> f64, Fault {
    let n = x.len;
    if n < 1 {
        return 0.0, fault(F_VALUE, "line_search", "the point holds no element");
    }
    if p.len != n {
        return 0.0, fault(F_SHAPE, "line_search", "the direction does not match the point");
    }
    if c2 <= 0.0 || c2 >= 1.0 {
        return 0.0, fault(F_VALUE, "line_search", "the condition of the slope holds a value outside zero and one");
    }
    let a, ok = ls_wolfe2(x, p, f0, slope0, 1.0e-4, c2, n) { t => F(copy(t)) };
    if !ok {
        return 0.0, fault(F_LIMIT, "line_search", "the search found no step that holds the conditions of Wolfe");
    }
    return a, no_fault();
}

// bracket finds three points that hold the minimum of a function
// between them. The search goes down the hill from the two points of
// the caller. The result holds the three points and their values.
fn bracket<F: fn(f64) -> f64>(x0: f64, x1: f64, grow_limit: f64 = 110.0, maxiter: i64 = 1000) -> Arr, Arr, Fault {
    if maxiter < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "bracket", "the count of turns holds a value below one");
    }
    // The golden ratio of SciPy.
    let gold: f64 = 1.618034;
    let small: f64 = 1.0e-21;
    var xa = x0;
    var xb = x1;
    var fa = F(xa);
    var fb = F(xb);
    if fa < fb {
        let t = xa; xa = xb; xb = t;
        let v = fa; fa = fb; fb = v;
    }
    var xc = xb + gold * (xb - xa);
    var fc = F(xc);
    var turn: i64 = 0;
    while fc < fb {
        let tmp1 = (xb - xa) * (fb - fc);
        let tmp2 = (xb - xc) * (fb - fa);
        let val = tmp2 - tmp1;
        var den = 2.0 * val;
        if mathshim::abs_of(val) < small { den = 2.0 * small; }
        var w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / den;
        let wlim = xb + grow_limit * (xc - xb);
        if turn > maxiter {
            return empty_arr(), empty_arr(), fault(F_LIMIT, "bracket", "the search found no bracket");
        }
        turn += 1;
        var fw: f64 = 0.0;
        if (w - xc) * (xb - w) > 0.0 {
            fw = F(w);
            if fw < fc {
                xa = xb; xb = w; fa = fb; fb = fw;
                break;
            }
            if fw > fb {
                xc = w; fc = fw;
                break;
            }
            w = xc + gold * (xc - xb);
            fw = F(w);
        } else if (w - wlim) * (wlim - xc) >= 0.0 {
            w = wlim;
            fw = F(w);
        } else if (w - wlim) * (xc - w) > 0.0 {
            fw = F(w);
            if fw < fc {
                xb = xc; xc = w;
                w = xc + gold * (xc - xb);
                fb = fc; fc = fw;
                fw = F(w);
            }
        } else {
            w = xc + gold * (xc - xb);
            fw = F(w);
        }
        xa = xb; xb = xc; xc = w;
        fa = fb; fb = fc; fc = fw;
    }
    // The three conditions of a valid bracket of SciPy.
    let c1 = (fb < fc && fb <= fa) || (fb < fa && fb <= fc);
    let c2 = (xa < xb && xb < xc) || (xc < xb && xb < xa);
    let c3 = all_finite([xa, xb, xc]);
    if !(c1 && c2 && c3) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "bracket", "the search found no bracket");
    }
    var pts: f64[>..] = [];
    var vals: f64[>..] = [];
    pts.push(xa); pts.push(xb); pts.push(xc);
    vals.push(fa); vals.push(fb); vals.push(fc);
    return flt_arr(copy(pts)), flt_arr(copy(vals)), no_fault();
}

// brute searches the smallest value of a function on a grid. The grid
// holds the same count of points in every direction, from the lower
// bound to the upper bound. The result holds the point and the value.
fn brute<F: fn(f64[:]) -> f64>(lower: f64[:], upper: f64[:], ns: i64 = 10) -> Arr, f64, Fault {
    let n = lower.len;
    if n < 1 {
        return empty_arr(), 0.0, fault(F_VALUE, "brute", "the bounds hold no element");
    }
    if upper.len != n {
        return empty_arr(), 0.0, fault(F_SHAPE, "brute", "the bounds do not hold the same count");
    }
    if ns < 1 {
        return empty_arr(), 0.0, fault(F_VALUE, "brute", "the count of points holds a value below one");
    }
    var i: i64 = 0;
    while i < n {
        if lower[i] > upper[i] {
            return empty_arr(), 0.0, fault(F_VALUE, "brute", "a lower bound lies above the upper bound");
        }
        i += 1;
    }
    // The count of points of the grid must stay small.
    var total: i64 = 1;
    i = 0;
    while i < n {
        total *= ns;
        if total > BRUTE_POINTS_MAX {
            return empty_arr(), 0.0, fault(F_LIMIT, "brute", "the grid holds more points than the limit");
        }
        i += 1;
    }
    // The grid of every direction.
    var steps: f64[>..] = [];
    i = 0;
    while i < n {
        if ns > 1 { steps.push((upper[i] - lower[i]) / (ns - 1.0)); }
        else { steps.push(0.0); }
        i += 1;
    }
    var at: i64[>..] = [];
    i = 0;
    while i < n { at.push(0); i += 1; }
    var best: f64[>..] = [];
    var bestv: f64 = 0.0;
    var first = true;
    var count: i64 = 0;
    while count < total {
        var p: f64[>..] = [];
        i = 0;
        while i < n { p.push(lower[i] + steps[i] * (at[i] as f64)); i += 1; }
        let v = F(copy(p));
        if first || v < bestv {
            bestv = v;
            best = copy(p);
            first = false;
        }
        // The last direction moves the fastest, as NumPy does.
        var k = n - 1;
        while k >= 0 {
            at[k] = at[k] + 1;
            if at[k] < ns { break; }
            at[k] = 0;
            k -= 1;
        }
        count += 1;
    }
    return flt_arr(copy(best)), bestv, no_fault();
}

// bfgs_hessian_update gives the next estimate of the Hessian from the
// current one, the step of the point and the step of the gradient. The
// formula is B + y y^T / (y^T s) - (B s)(B s)^T / (s^T B s), the
// Hessian form of the update of Broyden, Fletcher and Goldfarb, as
// SciPy holds it. A pair of steps without curvature gives a fault.
fn bfgs_hessian_update(b, s: f64[:], y: f64[:]) -> Arr, Fault {
    if b.ndim != 2 || b.shape.len != 2 || b.shape[0] != b.shape[1] || b.shape[0] < 1 {
        return empty_arr(), fault(F_SHAPE, "bfgs_hessian_update", "the Hessian is not square");
    }
    let n = b.shape[0];
    if n > 4096 {
        return empty_arr(), fault(F_LIMIT, "bfgs_hessian_update", "the side holds a value above the limit");
    }
    if s.len != n || y.len != n {
        return empty_arr(), fault(F_SHAPE, "bfgs_hessian_update", "a step does not match the Hessian");
    }
    let fb = check_stack(b.ndim, b.shape, b.data.len, 1, "bfgs_hessian_update");
    if fb.kind != F_NONE { return empty_arr(), fb; }
    if !all_finite(b.data) || !all_finite(s) || !all_finite(y) {
        return empty_arr(), fault(F_VALUE, "bfgs_hessian_update", "a value is not finite");
    }
    var ys: f64 = 0.0;
    var i: i64 = 0;
    while i < n { ys += s[i] * y[i]; i += 1; }
    if ys == 0.0 {
        return empty_arr(), fault(F_VALUE, "bfgs_hessian_update", "the steps hold no curvature");
    }
    var bs: f64[>..] = [];
    var sbs: f64 = 0.0;
    i = 0;
    while i < n {
        var t: f64 = 0.0;
        var j: i64 = 0;
        while j < n { t += b.data[i * n + j] * s[j]; j += 1; }
        sbs += s[i] * t;
        bs.push(t);
        i += 1;
    }
    if sbs == 0.0 {
        return empty_arr(), fault(F_VALUE, "bfgs_hessian_update", "the step holds no curvature of the Hessian");
    }
    var out: f64[>..] = [];
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            out.push(b.data[i * n + j] + y[i] * y[j] / ys - bs[i] * bs[j] / sbs);
            j += 1;
        }
        i += 1;
    }
    if !all_finite(out) {
        return empty_arr(), fault(F_VALUE, "bfgs_hessian_update", "the update overflows");
    }
    return mat2(copy(out), n, n), no_fault();
}

// sr1_hessian_update gives the next estimate of the Hessian from the
// current one, the step of the point and the step of the gradient. The
// rule is the update of the symmetric rank one. When the denominator
// is small against the sizes of the steps the update does not run, and
// the call gives the current Hessian back, as SciPy skips it.
fn sr1_hessian_update(b, s: f64[:], y: f64[:], min_denominator: f64 = 1.0e-8) -> Arr, Fault {
    if b.ndim != 2 || b.shape.len != 2 || b.shape[0] != b.shape[1] || b.shape[0] < 1 {
        return empty_arr(), fault(F_SHAPE, "sr1_hessian_update", "the Hessian is not square");
    }
    let n = b.shape[0];
    if n > 4096 {
        return empty_arr(), fault(F_LIMIT, "sr1_hessian_update", "the side holds a value above the limit");
    }
    if s.len != n || y.len != n {
        return empty_arr(), fault(F_SHAPE, "sr1_hessian_update", "a step does not match the Hessian");
    }
    let fb = check_stack(b.ndim, b.shape, b.data.len, 1, "sr1_hessian_update");
    if fb.kind != F_NONE { return empty_arr(), fb; }
    if !all_finite(b.data) || !all_finite(s) || !all_finite(y) || !(min_denominator >= 0.0) {
        return empty_arr(), fault(F_VALUE, "sr1_hessian_update", "a value is not finite");
    }
    var z: f64[>..] = [];
    var ns: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        var t: f64 = 0.0;
        var j: i64 = 0;
        while j < n { t += b.data[i * n + j] * s[j]; j += 1; }
        z.push(y[i] - t);
        ns += s[i] * s[i];
        i += 1;
    }
    ns = mathshim::root(ns);
    var nz: f64 = 0.0;
    var den: f64 = 0.0;
    i = 0;
    while i < n {
        nz += z[i] * z[i];
        den += s[i] * z[i];
        i += 1;
    }
    nz = mathshim::root(nz);
    // The rule of the skip of SciPy.
    if mathshim::abs_of(den) <= min_denominator * ns * nz {
        var keep: f64[>..] = [];
        i = 0;
        while i < b.data.len { keep.push(b.data[i]); i += 1; }
        return mat2(copy(keep), n, n), no_fault();
    }
    var out: f64[>..] = [];
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            out.push(b.data[i * n + j] + z[i] * z[j] / den);
            j += 1;
        }
        i += 1;
    }
    if !all_finite(out) {
        return empty_arr(), fault(F_VALUE, "sr1_hessian_update", "the update overflows");
    }
    return mat2(copy(out), n, n), no_fault();
}

// isotonic_regression fits a function that must not fall. The rule is
// the pool of the neighbours that break the order. The result holds the
// fitted values and the weight of every block.
fn isotonic_regression(y: f64[:], w: f64[:] = [], increasing: bool = true) -> Arr, Arr, Fault {
    let n = y.len;
    if n < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "isotonic_regression", "the values hold no element");
    }
    if w.len > 0 && w.len != n {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "isotonic_regression", "the weights do not match the values");
    }
    // The values and the weights in the order of the work.
    var vals: f64[>..] = [];
    var wts: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        if w.len > 0 {
            if w[i] <= 0.0 {
                return empty_arr(), empty_arr(), fault(F_VALUE, "isotonic_regression", "a weight holds a value of zero or below");
            }
            wts.push(w[i]);
        } else {
            wts.push(1.0);
        }
        if increasing { vals.push(y[i]); } else { vals.push(y[n - 1 - i]); }
        i += 1;
    }
    // The blocks of the pool. Each block holds its value, its weight
    // and its count of values. The count of the blocks is the count of
    // the used places of the buffers. A merge takes the last two
    // blocks and leaves one block, so the count of the used places
    // falls while the count of the buffers stays.
    var bv: f64[>..] = [];
    var bw: f64[>..] = [];
    var bc: i64[>..] = [];
    var i: i64 = 0;
    while i < n {
        bv.push(0.0);
        bw.push(0.0);
        bc.push(0);
        i += 1;
    }
    var nb: i64 = 0;
    i = 0;
    while i < n {
        bv[nb] = vals[i];
        bw[nb] = wts[i];
        bc[nb] = 1;
        nb += 1;
        // The pool of the last two blocks runs while the block before
        // the last one holds a value that is not below the last one.
        while nb > 1 && bv[nb - 2] >= bv[nb - 1] {
            let w1 = bw[nb - 1];
            let w2 = bw[nb - 2];
            let ww = w1 + w2;
            bv[nb - 2] = (bv[nb - 2] * w2 + bv[nb - 1] * w1) / ww;
            bw[nb - 2] = ww;
            bc[nb - 2] = bc[nb - 2] + bc[nb - 1];
            nb -= 1;
        }
        i += 1;
    }
    // The fitted values hold the value of their block.
    var out: f64[>..] = [];
    var bweights: f64[>..] = [];
    i = 0;
    while i < nb {
        var k: i64 = 0;
        while k < bc[i] { out.push(bv[i]); k += 1; }
        bweights.push(bw[i]);
        i += 1;
    }
    if !increasing {
        var rev: f64[>..] = [];
        i = out.len - 1;
        while i >= 0 { rev.push(out[i]); i -= 1; }
        out = copy(rev);
        rev = [];
        i = bweights.len - 1;
        while i >= 0 { rev.push(bweights[i]); i -= 1; }
        bweights = copy(rev);
    }
    return flt_arr(copy(out)), flt_arr(copy(bweights)), no_fault();
}
