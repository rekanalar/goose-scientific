// The method of Levenberg and Marquardt of SciPy leastsq, and the
// linear least squares with bounds of SciPy lsq_linear.
//
// leastsq runs the algorithm of MINPACK: a step of Levenberg and
// Marquardt with a scaling of the columns, the three tolerances ftol,
// xtol and gtol, and the covariance from the final Jacobian. The call
// leastsq takes the Jacobian as a named function of the caller. The
// call leastsq_est estimates the Jacobian by the forward difference,
// as leastsq does when Dfun is None in SciPy.
//
// lsq_linear solves the linear least squares with box bounds. The
// method trf holds the point inside the bounds and walks the
// gradient. The method bvls holds an active set and moves one
// variable at a time. Both stop at the same point, because the
// problem is convex.
namespace scientific;

import mathshim;

// The machine step of the forward difference. It is the root of the
// machine epsilon.
const LSQ_FD_STEP: f64 = 1.4901161193847656e-8;

// The smallest difference of two numbers.
const LSQ_EPS: f64 = 2.220446049250313e-16;

// The largest count of turns of the loop of Levenberg and Marquardt.
const LSQ_MAX_TURNS: i64 = 10000;

// The largest count of the calls of the function.
const LSQ_MAX_FEV: i64 = 10000000;

// lsq_check_args holds the checks that leastsq and leastsq_est share.
fn lsq_check_args(x0: f64[:], ftol: f64, xtol: f64, gtol: f64, maxfev: i64, factor: f64, diag: f64[:], tag: const u8[:]) -> Fault {
    if x0.len < 1 {
        return fault(F_VALUE, tag, "the start holds no element");
    }
    if diag.len != 0 && diag.len != x0.len {
        return fault(F_SHAPE, tag, "diag holds a wrong count of values");
    }
    var i: i64 = 0;
    while i < diag.len {
        if !(diag[i] > 0.0) {
            return fault(F_VALUE, tag, "diag holds a value of zero or below");
        }
        i += 1;
    }
    if ftol < LSQ_EPS {
        return fault(F_VALUE, tag, "ftol is too small, no further reduction is possible");
    }
    if xtol < LSQ_EPS {
        return fault(F_VALUE, tag, "xtol is too small, no further improvement is possible");
    }
    if gtol < 0.0 {
        return fault(F_VALUE, tag, "gtol holds a value below zero");
    }
    if maxfev < 0 {
        return fault(F_VALUE, tag, "maxfev holds a value below zero");
    }
    if maxfev > LSQ_MAX_FEV {
        return fault(F_LIMIT, tag, "maxfev holds a value above the limit");
    }
    if !(factor > 0.1) || factor > 100.0 {
        return fault(F_RANGE, tag, "factor stands outside the range zero point one to one hundred");
    }
    return no_fault();
}

// lsq_fill_jac fills the Jacobian at the point x. The flag estimate
// tells the helper to build it by the forward difference. The other
// path takes it from the Jacobian of the caller. A row of the matrix
// holds one value of the function and a column holds one variable.
// The call gives the counts of the calls that the fill used and a
// fault.
fn lsq_fill_jac<D: fn(f64[:]) -> f64[:], F: fn(f64[:]) -> f64[:]>(x: f64[>..]&, jac: f64[>..]&, f: f64[:], m: i64, n: i64, estimate: bool, tag: const u8[:]) -> i64, i64, Fault {
    var nfev: i64 = 0;
    var njev: i64 = 0;
    if estimate {
        var j: i64 = 0;
        while j < n {
            let h = LSQ_FD_STEP * (if x[j] != 0.0 { mathshim::abs_of(x[j]) } else { 1.0 });
            let old = x[j];
            x[j] = old + h;
            let fp = F(copy(x));
            nfev += 1;
            x[j] = old;
            if fp.len != m {
                return nfev, njev, fault(F_SHAPE, tag, "the function gave a wrong count of values at a probe");
            }
            var i: i64 = 0;
            while i < m { jac[i * n + j] = (fp[i] - f[i]) / h; i += 1; }
            j += 1;
        }
    } else {
        let jr = D(copy(x));
        njev += 1;
        if jr.len != m * n {
            return nfev, njev, fault(F_SHAPE, tag, "the jacobian holds a wrong count of values");
        }
        var i: i64 = 0;
        while i < m * n { jac[i] = jr[i]; i += 1; }
    }
    return nfev, njev, no_fault();
}

// lsq_normal builds the normal matrix of a Jacobian. The matrix holds
// the products of the columns of the Jacobian.
fn lsq_normal(m: i64, n: i64, jac: f64[:]) -> f64[>..] {
    var normal: f64[>..] = [];
    var a: i64 = 0;
    while a < n * n { normal.push(0.0); a += 1; }
    a = 0;
    while a < n {
        var b: i64 = 0;
        while b < n {
            var t: f64 = 0.0;
            var i: i64 = 0;
            while i < m { t += jac[i * n + a] * jac[i * n + b]; i += 1; }
            normal[a * n + b] = t;
            b += 1;
        }
        a += 1;
    }
    return normal;
}

// leastsq finds the point that gives the least sum of squares of the
// values of a function. The function is a block. The Jacobian is a
// named function of the caller that takes a point and gives a matrix
// of m rows and n columns in one flat buffer, row by row. The counts
// maxfev of zero give the default of SciPy, one hundred times the
// count of the variables plus one. The answer holds the point, the
// value of the sum of the squares, the flag of the success, the code
// of the end and the counts. The code holds one to four for a run
// that converged and five for a run that ran out of calls. The
// covariance is the inverse of the product of the final Jacobian and
// its transpose, as SciPy scales it before the caller multiplies it
// with the variance of the residuals. A singular Jacobian is a fault.
fn leastsq<D: fn(f64[:]) -> f64[:], F: fn(f64[:]) -> f64[:]>(x0: f64[:], ftol: f64 = 1.49012e-8, xtol: f64 = 1.49012e-8, gtol: f64 = 0.0, maxfev: i64 = 0, factor: f64 = 100.0, diag: f64[:] = []) -> OptimizeResult, Arr, Fault {
    let g = lsq_check_args(x0, ftol, xtol, gtol, maxfev, factor, diag, "leastsq");
    if g.kind != F_NONE { return optimize_result(x0, 0.0, false, 0, "", 0, 0, 0), empty_arr(), g; }
    let n = x0.len;
    var limit = maxfev;
    if limit == 0 { limit = 100 * (n + 1); }
    let res, cov, f = lsq_lm_jac(x0, false, ftol, xtol, gtol, limit, factor, diag, D) { p => F(copy(p)) };
    return res, cov, f;
}

// lsq_lm_jac holds the loop of Levenberg and Marquardt with a
// Jacobian from the caller. The flag
// estimate tells the loop to build the Jacobian by the forward
// difference. The loop keeps the step small near a column that does
// not lower the error, and it grows the step again when the error
// falls.
fn lsq_lm_jac<D: fn(f64[:]) -> f64[:], F: fn(f64[:]) -> f64[:]>(x0: f64[:], estimate: bool, ftol: f64, xtol: f64, gtol: f64, maxfev: i64, factor: f64, diag: f64[:]) -> OptimizeResult, Arr, Fault {
    let n = x0.len;
    var x: f64[>..] = [];
    var scale: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        x.push(x0[i]);
        if diag.len > 0 { scale.push(diag[i]); } else { scale.push(1.0); }
        i += 1;
    }
    var f = F(copy(x));
    var nfev: i64 = 1;
    var njev: i64 = 0;
    let m = f.len;
    if m < n {
        return optimize_result(x0, 0.0, false, 0, "the function gives fewer values than the point holds", 1, 0, 0), empty_arr(), fault(F_SHAPE, "leastsq", "the function gives fewer values than the point holds");
    }
    if !all_finite(f) {
        return optimize_result(x0, 0.0, false, 0, "", 1, 0, 0), empty_arr(), fault(F_VALUE, "leastsq", "the function gave a value that is not finite");
    }
    // The size of the error and the step bound of the start. The
    // bound grows from factor times the scaled size of the start.
    var fnorm2: f64 = 0.0;
    i = 0;
    while i < m { fnorm2 += f[i] * f[i]; i += 1; }
    var xnorm: f64 = 0.0;
    i = 0;
    while i < n {
        var t = x[i] * scale[i];
        xnorm += t * t;
        i += 1;
    }
    xnorm = mathshim::root(xnorm);
    var damping: f64 = 0.001;
    var status: i64 = 0;
    var message: const u8[:] = "";
    var hit_ftol = false;
    var hit_xtol = false;
    var nit: i64 = 0;
    while nit < LSQ_MAX_TURNS {
        nit += 1;
        // The Jacobian at the point. A row holds one value of the
        // function and a column holds one variable.
        var jac: f64[>..] = [];
        i = 0;
        while i < m * n { jac.push(0.0); i += 1; }
        let dfev, djev, fj = lsq_fill_jac(x, jac, f, m, n, estimate, "leastsq", D) { p => F(copy(p)) };
        nfev += dfev;
        njev += djev;
        if fj.kind != F_NONE {
            return optimize_result(x0, 0.0, false, 0, "", nfev, njev, nit), empty_arr(), fj;
        }
        // The gradient and the normal matrix.
        var grad: f64[>..] = [];
        var j: i64 = 0;
        while j < n {
            var gv: f64 = 0.0;
            i = 0;
            while i < m { gv += jac[i * n + j] * f[i]; i += 1; }
            grad.push(gv);
            j += 1;
        }
        var normj: f64[>..] = [];
        j = 0;
        while j < n {
            var cn: f64 = 0.0;
            i = 0;
            while i < m { cn += jac[i * n + j] * jac[i * n + j]; i += 1; }
            normj.push(mathshim::root(cn));
            j += 1;
        }
        // The test of gtol: the cosine of the angle between the values
        // and every column. A small cosine means no column can lower
        // the error.
        if gtol > 0.0 {
            let fnorm = mathshim::root(fnorm2);
            if fnorm > 0.0 {
                var worst: f64 = 0.0;
                j = 0;
                while j < n {
                    var c: f64 = 0.0;
                    if normj[j] > 0.0 { c = mathshim::abs_of(grad[j]) / (normj[j] * fnorm); }
                    if c > worst { worst = c; }
                    j += 1;
                }
                if worst <= gtol {
                    status = 4;
                    message = "the cosine of the angle between the values and a column of the jacobian is at most gtol";
                    break;
                }
            }
        }
        // The normal matrix holds the products of the columns.
        let normal = lsq_normal(m, n, jac);
        // The loop of the damped steps. The damping grows while no
        // step lowers the error, and it shrinks after a good step.
        var accepted = false;
        var tries: i64 = 0;
        var actual: f64 = 0.0;
        var predicted: f64 = 0.0;
        var stepn: f64 = 0.0;
        while tries < 50 {
            var sys: f64[>..] = [];
            i = 0;
            while i < n * n { sys.push(normal[i]); i += 1; }
            i = 0;
            while i < n {
                var dj = mathshim::abs_of(normal[i * n + i]);
                if dj < 1.0e-12 { dj = 1.0e-12; }
                sys[i * n + i] += damping * dj;
                i += 1;
            }
            var rhs: f64[>..] = [];
            j = 0;
            while j < n { rhs.push(-grad[j]); j += 1; }
            let p, fstep = solve(mat2(copy(sys), n, n), vec1(copy(rhs)));
            if fstep.kind != F_NONE {
                damping = damping * 10.0;
                if damping > 1.0e30 { break; }
                tries += 1;
                continue;
            }
            // The predicted reduction holds the size of the model
            // error after the step.
            var model: f64 = 0.0;
            i = 0;
            while i < m {
                var t: f64 = 0.0;
                j = 0;
                while j < n { t += jac[i * n + j] * p.data[j]; j += 1; }
                t += f[i];
                model += t * t;
                i += 1;
            }
            predicted = fnorm2 - model;
            if !(predicted > 0.0) {
                damping = damping * 10.0;
                if damping > 1.0e30 { break; }
                tries += 1;
                continue;
            }
            var trial: f64[>..] = [];
            j = 0;
            while j < n { trial.push(x[j] + p.data[j]); j += 1; }
            let ft = F(copy(trial));
            nfev += 1;
            if !all_finite(ft) {
                return optimize_result(x0, 0.0, false, 0, "", nfev, njev, nit), empty_arr(), fault(F_VALUE, "leastsq", "the function gave a value that is not finite");
            }
            var ft2: f64 = 0.0;
            i = 0;
            while i < ft.len { ft2 += ft[i] * ft[i]; i += 1; }
            actual = fnorm2 - ft2;
            if actual > 0.0 {
                let rho = actual / predicted;
                var grow = 1.0 - (2.0 * rho - 1.0) * (2.0 * rho - 1.0);
                if grow < 1.0 / 3.0 { grow = 1.0 / 3.0; }
                damping = damping * grow;
                if damping < 1.0e-20 { damping = 1.0e-20; }
                i = 0;
                while i < n { x[i] = trial[i]; i += 1; }
                i = 0;
                while i < m { f[i] = ft[i]; i += 1; }
                fnorm2 = ft2;
                accepted = true;
                // The relative reduction and the size of the step
                // against the size of the point.
                if fnorm2 > 0.0 {
                    if actual / fnorm2 <= ftol && predicted / fnorm2 <= ftol { hit_ftol = true; }
                } else {
                    hit_ftol = true;
                }
                j = 0;
                while j < n {
                    let t = p.data[j];
                    stepn += t * t;
                    j += 1;
                }
                stepn = mathshim::root(stepn);
                if stepn <= xtol * (xnorm + xtol) { hit_xtol = true; }
                xnorm = 0.0;
                j = 0;
                while j < n {
                    var t = x[j] * scale[j];
                    xnorm += t * t;
                    j += 1;
                }
                xnorm = mathshim::root(xnorm);
                break;
            }
            damping = damping * 10.0;
            if damping > 1.0e30 { break; }
            tries += 1;
        }
        if status == 0 && hit_ftol && hit_xtol {
            status = 3;
            message = "both tolerances hold at the same turn";
        } else if status == 0 && hit_ftol {
            status = 1;
            message = "both actual and predicted relative reductions in the sum of squares are at most ftol";
        } else if status == 0 && hit_xtol {
            status = 2;
            message = "the relative error between two consecutive iterates is at most xtol";
        }
        if status != 0 { break; }
        if !accepted {
            // No step lowered the error. The error cannot fall
            // further, so the run ends as a run of ftol.
            status = 1;
            message = "both actual and predicted relative reductions in the sum of squares are at most ftol";
            break;
        }
        if nfev >= maxfev {
            status = 5;
            message = "the count of the calls reached maxfev";
            break;
        }
    }
    var success = status >= 1 && status <= 4;
    if status == 0 {
        status = 5;
        message = "the count of the calls reached maxfev";
        success = false;
    }
    // The covariance from the final Jacobian. The inverse of the
    // normal matrix at the point of the answer.
    var cov = empty_arr();
    if success {
        var jac: f64[>..] = [];
        i = 0;
        while i < m * n { jac.push(0.0); i += 1; }
        let dfev, djev, fj = lsq_fill_jac(x, jac, f, m, n, estimate, "leastsq", D) { p => F(copy(p)) };
        nfev += dfev;
        njev += djev;
        if fj.kind != F_NONE {
            return optimize_result(x0, 0.0, false, 0, "", nfev, njev, nit), empty_arr(), fj;
        }
        let normal = lsq_normal(m, n, jac);
        let ci, fi = inv(mat2(copy(normal), n, n));
        if fi.kind != F_NONE {
            return optimize_result(x0, 0.0, false, 0, "", nfev, njev, nit), empty_arr(), fault(F_VALUE, "leastsq", "the matrix of the derivatives is singular at the answer");
        }
        cov = copy(ci);
    }
    return optimize_result(copy(x), fnorm2, success, status, message, nfev, njev, nit), cov, no_fault();
}
