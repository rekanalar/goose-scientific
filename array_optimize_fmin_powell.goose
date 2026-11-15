// The method of Powell for the minimization of a function.
//
// The method holds a set of directions. It runs a line search along
// each direction of the set in turn. After the searches the method
// takes an extrapolated step along the mean of the last two
// directions. When that step improves the point, the direction with
// the largest fall leaves the set and the new direction joins it.
// Each line search is the method of Brent: a bracket search followed
// by a golden section step and steps of a parabola through three
// points.
//
// The block of the caller takes a vector and gives a list that holds
// the value of the function first. The method uses the values only.
namespace scientific;

import mathshim;

// PwBracket holds the three points of a bracket and the values of the
// function at them.
struct PwBracket {
    xa: f64, xb: f64, xc: f64,
    fa: f64, fb: f64, fc: f64,
}

// pw_value gives the value of the function of the caller at the point
// x plus alpha times the direction.
fn pw_value<F: fn(f64[:]) -> f64[:]>(x: f64[:], d: f64[:], alpha: f64, n: i64) -> f64 {
    var p: f64[>..] = [];
    var i: i64 = 0;
    while i < n { p.push(x[i] + alpha * d[i]); i += 1; }
    let r = F(copy(p));
    if r.len < 1 { return 1.0e300; }
    return r[0];
}

// pw_bracket finds three points along a direction that hold the
// minimum between them. The search starts at the points of alpha zero
// and one and walks in the direction where the function falls. The
// rule is the rule of the function bracket of SciPy.
fn pw_bracket<F: fn(f64[:]) -> f64[:]>(x: f64[:], d: f64[:], n: i64) -> PwBracket {
    let gold: f64 = 1.618034;
    let tiny: f64 = 1.0e-21;
    var xa: f64 = 0.0;
    var xb: f64 = 1.0;
    var fa = pw_value(x, d, xa, n) { p => F(copy(p)) };
    var fb = pw_value(x, d, xb, n) { p => F(copy(p)) };
    if fa < fb {
        let t = xa; xa = xb; xb = t;
        let s = fa; fa = fb; fb = s;
    }
    var xc = xb + gold * (xb - xa);
    var fc = pw_value(x, d, xc, n) { p => F(copy(p)) };
    var it: i64 = 0;
    while fc < fb && it < 1000 {
        it += 1;
        let tmp1 = (xb - xa) * (fb - fc);
        let tmp2 = (xb - xc) * (fb - fa);
        let val = tmp2 - tmp1;
        var denom: f64 = 2.0 * val;
        if mathshim::abs_of(val) < tiny { denom = 2.0 * tiny; }
        var w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
        let wlim = xb + 110.0 * (xc - xb);
        var fw: f64 = 0.0;
        if (w - xc) * (xb - w) > 0.0 {
            fw = pw_value(x, d, w, n) { p => F(copy(p)) };
            if fw < fc {
                xa = xb; xb = w; fa = fb; fb = fw;
                break;
            }
            if fw > fb {
                xc = w; fc = fw;
                break;
            }
            w = xc + gold * (xc - xb);
            fw = pw_value(x, d, w, n) { p => F(copy(p)) };
        } else if (w - wlim) * (wlim - xc) >= 0.0 {
            w = wlim;
            fw = pw_value(x, d, w, n) { p => F(copy(p)) };
        } else if (w - wlim) * (xc - w) > 0.0 {
            fw = pw_value(x, d, w, n) { p => F(copy(p)) };
            if fw < fc {
                xb = xc; xc = w; fb = fc; fc = fw;
                w = xc + gold * (xc - xb);
                fw = pw_value(x, d, w, n) { p => F(copy(p)) };
            }
        } else {
            w = xc + gold * (xc - xb);
            fw = pw_value(x, d, w, n) { p => F(copy(p)) };
        }
        xa = xb;
        xb = xc;
        xc = w;
        fa = fb;
        fb = fc;
        fc = fw;
    }
    return PwBracket { xa: xa, xb: xb, xc: xc, fa: fa, fb: fb, fc: fc };
}

// pw_brent finds the place of the smallest value of the function
// between the points of a bracket. The rule is the rule of Brent for
// a minimum: a step of the golden section when the last step was
// small, and a step of the parabola through the three points with the
// smallest values otherwise. The call gives the place, the value and
// a flag that tells whether the bracket held a minimum.
fn pw_brent<F: fn(f64[:]) -> f64[:]>(x: f64[:], d: f64[:], tol: f64, n: i64) -> f64, f64, bool {
    let br = pw_bracket(x, d, n) { p => F(copy(p)) };
    var xa = br.xa;
    var xc = br.xc;
    var fa = br.fa;
    var fb = br.fb;
    var fc = br.fc;
    // The bracket must hold the values in the form of a minimum.
    let cond1 = (fb < fc && fb <= fa) || (fb < fa && fb <= fc);
    let cond2 = (br.xa < br.xb && br.xb < xc) || (xc < br.xb && br.xb < br.xa);
    if !(cond1 && cond2) {
        // The bracket holds no minimum. The best of the three points
        // is the answer.
        var best = br.xa;
        var bf = fa;
        if fb < bf { best = br.xb; bf = fb; }
        if fc < bf { best = xc; bf = fc; }
        return best, bf, true;
    }
    if xa >= xc {
        let t = xa; xa = xc; xc = t;
        let s = fa; fa = fc; fc = s;
    }
    var xv = br.xb;
    var wv = br.xb;
    var vv = br.xb;
    var fx = fb;
    var fw = fb;
    var fv = fb;
    var a = xa;
    var b = xc;
    var deltax: f64 = 0.0;
    var rat: f64 = 0.0;
    var it: i64 = 0;
    while it < 500 {
        it += 1;
        let tol1 = tol * mathshim::abs_of(xv) + 0.00000000001;
        let tol2 = 2.0 * tol1;
        let xmid = 0.5 * (a + b);
        if mathshim::abs_of(xv - xmid) < (tol2 - 0.5 * (b - a)) { break; }
        var u: f64 = 0.0;
        if mathshim::abs_of(deltax) <= tol1 {
            // The step of the golden section.
            if xv >= xmid { deltax = a - xv; } else { deltax = b - xv; }
            rat = 0.381966 * deltax;
        } else {
            // The step of the parabola through three points.
            let tmp1 = (xv - wv) * (fx - fv);
            let tmp2 = (xv - vv) * (fx - fw);
            var p = (xv - vv) * tmp2 - (xv - wv) * tmp1;
            var t2 = 2.0 * (tmp2 - tmp1);
            if t2 > 0.0 { p = -p; }
            t2 = mathshim::abs_of(t2);
            let dx_temp = deltax;
            deltax = rat;
            if p > t2 * (a - xv) && p < t2 * (b - xv) && mathshim::abs_of(p) < mathshim::abs_of(0.5 * t2 * dx_temp) {
                rat = p / t2;
                u = xv + rat;
                if (u - a) < tol2 || (b - u) < tol2 {
                    if xmid - xv >= 0.0 { rat = tol1; } else { rat = -tol1; }
                }
            } else {
                if xv >= xmid { deltax = a - xv; } else { deltax = b - xv; }
                rat = 0.381966 * deltax;
            }
        }
        // The step must move the point by at least the tolerance.
        if mathshim::abs_of(rat) < tol1 {
            if rat >= 0.0 { u = xv + tol1; } else { u = xv - tol1; }
        } else {
            u = xv + rat;
        }
        let fu = pw_value(x, d, u, n) { p => F(copy(p)) };
        if fu > fx {
            // The new point holds a larger value. One end of the
            // bracket moves to it.
            if u < xv { a = u; } else { b = u; }
            if fu <= fw || wv == xv {
                vv = wv; wv = u; fv = fw; fw = fu;
            } else if fu <= fv || vv == xv || vv == wv {
                vv = u; fv = fu;
            }
        } else {
            // The new point holds the smallest value. It becomes the
            // point in the middle.
            if u >= xv { a = xv; } else { b = xv; }
            vv = wv;
            wv = xv;
            xv = u;
            fv = fw;
            fw = fx;
            fx = fu;
        }
    }
    return xv, fx, true;
}

// fmin_powell finds the point where a function of several variables is
// at its smallest. The method holds one direction for each variable.
// It runs a line search along each direction, and it takes an
// extrapolated step along the mean of the last two directions. The
// method uses the values of the function only. The call gives the
// point of the minimum and a fault. The fault of the limit means that
// the method did not reach the tolerance in the count of turns.
fn fmin_powell<F: fn(f64[:]) -> f64[:]>(x0: f64[:], xtol: f64 = 0.0001, ftol: f64 = 0.0001, maxiter: i64 = 0) -> Arr, Fault {
    let n = x0.len;
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "fmin_powell", "the start holds no element");
    }
    if !all_finite(x0) {
        return empty_arr(), fault(F_VALUE, "fmin_powell", "the start holds a value that is not a number");
    }
    if !(xtol > 0.0) || !(ftol > 0.0) {
        return empty_arr(), fault(F_VALUE, "fmin_powell", "a tolerance holds a value of zero or below");
    }
    var limit = maxiter;
    if limit < 1 { limit = n * 1000; }
    var x: f64[>..] = copy_slice(x0);
    // The directions form the identity matrix.
    var direc: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { direc.push(0.0); i += 1; }
    i = 0;
    while i < n { direc[i * n + i] = 1.0; i += 1; }
    let r = F(copy(x));
    if r.len < 1 {
        return empty_arr(), fault(F_VALUE, "fmin_powell", "the function holds no value");
    }
    var fval = r[0];
    var x1: f64[>..] = copy_slice(x);
    var iter: i64 = 0;
    var done = false;
    while iter < limit && !done {
        let fx = fval;
        var bigind: i64 = 0;
        var delta: f64 = 0.0;
        i = 0;
        while i < n {
            // The direction of this turn.
            var d: f64[>..] = [];
            var j: i64 = 0;
            while j < n { d.push(direc[i * n + j]); j += 1; }
            var zero = true;
            j = 0;
            while j < n {
                if d[j] != 0.0 { zero = false; }
                j += 1;
            }
            if !zero {
                let fx2 = fval;
                let alpha, fret, ok = pw_brent(x, d, xtol * 100.0, n) { p => F(copy(p)) };
                if !ok {
                    return empty_arr(), fault(F_VALUE, "fmin_powell", "the line search holds no point");
                }
                fval = fret;
                var pnt: f64[>..] = [];
                j = 0;
                while j < n { pnt.push(x[j] + alpha * d[j]); j += 1; }
                x = copy(pnt);
                if fx2 - fval > delta {
                    delta = fx2 - fval;
                    bigind = i;
                }
            }
            i += 1;
        }
        iter += 1;
        // The values of the function fall enough.
        let bnd = ftol * (mathshim::abs_of(fx) + mathshim::abs_of(fval)) + 1.0e-20;
        if 2.0 * (fx - fval) <= bnd { done = true; break; }
        if iter >= limit { break; }
        // The extrapolated step along the mean of the last directions.
        var d1: f64[>..] = [];
        i = 0;
        while i < n { d1.push(x[i] - x1[i]); i += 1; }
        x1 = copy_slice(x);
        var x2: f64[>..] = [];
        i = 0;
        while i < n { x2.push(x[i] + d1[i]); i += 1; }
        let r2 = F(copy(x2));
        if r2.len < 1 {
            return empty_arr(), fault(F_VALUE, "fmin_powell", "the function holds no value");
        }
        let fx2 = r2[0];
        if fx > fx2 {
            var t = 2.0 * (fx + fx2 - 2.0 * fval);
            let temp = fx - fval - delta;
            t *= temp * temp;
            let temp2 = fx - fx2;
            t -= delta * temp2 * temp2;
            if t < 0.0 {
                let alpha, fret, ok = pw_brent(x, d1, xtol * 100.0, n) { p => F(copy(p)) };
                if !ok {
                    return empty_arr(), fault(F_VALUE, "fmin_powell", "the line search holds no point");
                }
                fval = fret;
                var pnt: f64[>..] = [];
                i = 0;
                while i < n { pnt.push(x[i] + alpha * d1[i]); i += 1; }
                x = copy(pnt);
                // The direction of the step.
                var nd: f64[>..] = [];
                i = 0;
                while i < n { nd.push(alpha * d1[i]); i += 1; }
                var any = false;
                i = 0;
                while i < n {
                    if nd[i] != 0.0 { any = true; }
                    i += 1;
                }
                if any {
                    // The direction with the largest fall leaves the
                    // set and the new direction joins it at the end.
                    var j: i64 = 0;
                    while j < n {
                        direc[bigind * n + j] = direc[(n - 1) * n + j];
                        direc[(n - 1) * n + j] = nd[j];
                        j += 1;
                    }
                }
            }
        }
    }
    if !done {
        return empty_arr(), fault(F_LIMIT, "fmin_powell", "the method did not reach the tolerance in the count of turns");
    }
    return line_arr(copy(x)), no_fault();
}
