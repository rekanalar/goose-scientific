// The estimate form of the method of Levenberg and Marquardt of
// SciPy leastsq.
//
// leastsq_est runs the same loop as leastsq of the file
// array_optimize_lsq.goose, but it builds the Jacobian by the forward
// difference, as leastsq does when the caller gives no Jacobian.
namespace scientific;

import mathshim;

// lsq_no_jac is the Jacobian that the call leastsq_est passes to the
// shared loop. The loop never calls it when it estimates.
fn lsq_no_jac(x: f64[:]) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len { d.push(0.0); i += 1; }
    return d;
}

// leastsq_est runs the same loop as leastsq, but it estimates the
// Jacobian by the forward difference, as leastsq does without Dfun.
// The default count of the calls is two hundred times the count of
// the variables plus one.
fn leastsq_est<F: fn(f64[:]) -> f64[:]>(x0: f64[:], ftol: f64 = 1.49012e-8, xtol: f64 = 1.49012e-8, gtol: f64 = 0.0, maxfev: i64 = 0, factor: f64 = 100.0, diag: f64[:] = []) -> OptimizeResult, Arr, Fault {
    let g = lsq_check_args(x0, ftol, xtol, gtol, maxfev, factor, diag, "leastsq_est");
    if g.kind != F_NONE { return optimize_result(x0, 0.0, false, 0, "", 0, 0, 0), empty_arr(), g; }
    let n = x0.len;
    var limit = maxfev;
    if limit == 0 { limit = 200 * (n + 1); }
    let res, cov, f = lsq_lm_jac(x0, true, ftol, xtol, gtol, limit, factor, diag, lsq_no_jac, F);
    return res, cov, f;
}

