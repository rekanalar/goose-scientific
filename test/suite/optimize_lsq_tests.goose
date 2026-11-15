// The tests of the group optimize_lsq. Run them with
// SCIENTIFIC_SUITES=optimize_lsq.
//
// The calls leastsq, leastsq_est, lsq_linear, quadratic_assignment
// and shgo hold their answers against the pins of SciPy 1.18.1. The
// structs of the results and of the constraint hold their fields
// against the values that the tests set. Every fault path holds a
// test of its own.
namespace optimize_lsq_tests;

import scientific from "../..";
import optimize_lsq_pins;
import common;

// exp_y gives the right side of the problem of the exponential. It
// holds 2 exp(-t/2) with the small noise of the pins.
fn exp_y() -> f64[>..] {
    var d: f64[>..] = [];
    d.push(2.01);
    d.push(1.1930613194252668);
    d.push(0.7657588823428847);
    d.push(0.4362603202968596);
    d.push(0.2906705664732254);
    return d;
}

// poly_y gives the right side of the problem of the polynomial.
fn poly_y() -> f64[>..] {
    var d: f64[>..] = [];
    d.push(1.0);
    d.push(3.1);
    d.push(5.4);
    d.push(8.2);
    d.push(11.1);
    d.push(14.5);
    return d;
}

// gau_y gives the right side of the problem of the bell.
fn gau_y() -> f64[>..] {
    var d: f64[>..] = [];
    d.push(0.4260058497098381);
    d.push(1.8095919791373538);
    d.push(3.01);
    d.push(1.7895919791373538);
    d.push(0.4260058497098381);
    d.push(0.04329573117698889);
    d.push(-0.018992216326504592);
    return d;
}

// exp_res gives the residuals of the problem of the exponential.
fn exp_res(p: f64[:]) -> f64[>..] {
    let y = exp_y();
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 5 {
        let e = common::mathshim_exp(-p[1] * (i as f64));
        d.push(p[0] * e - y[i]);
        i += 1;
    }
    return d;
}

// exp_jac gives the Jacobian of the problem of the exponential. A row
// holds one value of the function and a column holds one variable.
fn exp_jac(p: f64[:]) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 5 {
        let e = common::mathshim_exp(-p[1] * (i as f64));
        d.push(e);
        d.push(-p[0] * (i as f64) * e);
        i += 1;
    }
    return d;
}

// short_jac gives a Jacobian of a wrong count of values.
fn short_jac(p: f64[:]) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 9 { d.push(0.0); i += 1; }
    return d;
}

// poly_res gives the residuals of the problem of the polynomial.
fn poly_res(p: f64[:]) -> f64[>..] {
    let y = poly_y();
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 6 {
        let t = i as f64;
        d.push(p[0] + p[1] * t + p[2] * t * t - y[i]);
        i += 1;
    }
    return d;
}

// poly_jac gives the Jacobian of the problem of the polynomial.
fn poly_jac(p: f64[:]) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 6 {
        let t = i as f64;
        d.push(1.0);
        d.push(t);
        d.push(t * t);
        i += 1;
    }
    return d;
}

// gau_res gives the residuals of the problem of the bell.
fn gau_res(p: f64[:]) -> f64[>..] {
    let y = gau_y();
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 7 {
        let t = i as f64;
        let e = common::mathshim_exp(-0.5 * (t - p[1]) * (t - p[1]));
        d.push(p[0] * e - y[i]);
        i += 1;
    }
    return d;
}

// gau_jac gives the Jacobian of the problem of the bell.
fn gau_jac(p: f64[:]) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 7 {
        let t = i as f64;
        let e = common::mathshim_exp(-0.5 * (t - p[1]) * (t - p[1]));
        d.push(e);
        d.push(p[0] * e * (t - p[1]));
        i += 1;
    }
    return d;
}

// one_dep gives the residuals of a problem whose second variable does
// nothing. The Jacobian holds a column of zeros, so the covariance is
// singular.
fn one_dep(p: f64[:]) -> f64[>..] {
    var d: f64[>..] = [];
    d.push(p[0] - 1.0);
    d.push(p[0] - 2.0);
    return d;
}

// one_dep_jac gives the Jacobian of the same problem.
fn one_dep_jac(p: f64[:]) -> f64[>..] {
    var d: f64[>..] = [];
    d.push(1.0);
    d.push(0.0);
    d.push(1.0);
    d.push(0.0);
    return d;
}

// shg_one gives the bowl of the first test of the global search.
fn shg_one(p: f64[:]) -> f64 {
    return (p[0] - 1.5) * (p[0] - 1.5) + (p[1] + 0.5) * (p[1] + 0.5);
}

// shg_two gives the wavy bowl of the second test.
fn shg_two(p: f64[:]) -> f64 {
    return (p[0] - 1.0) * (p[0] - 1.0) + (p[1] + 1.0) * (p[1] + 1.0)
        + 0.3 * common::mathshim_sin(4.0 * p[0]) * common::mathshim_sin(4.0 * p[1]);
}

// shg_three gives the wavy valley of the third test.
fn shg_three(p: f64[:]) -> f64 {
    return (p[0] - 2.0) * (p[0] - 2.0) + 0.1 * common::mathshim_sin(8.0 * p[0]);
}

// near_val tells whether two numbers agree to a relative tolerance.
fn near_val(a: f64, b: f64, tol: f64) -> bool {
    var d = a - b;
    if d < 0.0 { d = -d; }
    var s = b;
    if s < 0.0 { s = -s; }
    return d <= tol * (1.0 + s);
}

// row_near tells whether every value of an array agrees with a list.
fn row_near(a, v: f64[:], tol: f64) -> bool {
    if a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        if !near_val(a.data[i], v[i], tol) { return false; }
        i += 1;
    }
    return true;
}

// is_perm tells whether a list holds every value from zero to the
// count of the list minus one, once each.
fn is_perm(v: f64[:]) -> bool {
    var seen: bool[>..] = [];
    var i: i64 = 0;
    while i < v.len { seen.push(false); i += 1; }
    i = 0;
    while i < v.len {
        let k = v[i] as i64;
        if k < 0 || k >= v.len || seen[k] { return false; }
        seen[k] = true;
        i += 1;
    }
    return true;
}

// linmat builds a matrix from a list of values.
fn linmat(vals: f64[:], rows: i64, cols: i64) -> scientific::Arr {
    var c: f64[>..] = [];
    var i: i64 = 0;
    while i < vals.len { c.push(vals[i]); i += 1; }
    return scientific::Arr { ndim: 2, shape: [rows, cols], data: c };
}

fn test_leastsq() {
    // The problem of the exponential.
    let r1, c1, f1 = scientific::leastsq([1.5, 0.8], exp_jac) { p => exp_res(p) };
    assert(f1.kind == scientific::F_NONE);
    assert(r1.success && r1.status >= 1 && r1.status <= 4);
    assert(row_near(r1.x, optimize_lsq_pins::LSQ_EXP_X, 1.0e-6));
    assert(row_near(c1, optimize_lsq_pins::LSQ_EXP_COV, 1.0e-6));
    assert(common::close(r1.fun, optimize_lsq_pins::LSQ_EXP_FUN[0], 1.0e-5));
    assert(r1.nfev >= 1 && r1.njev >= 1);
    // The problem of the polynomial.
    let r2, c2, f2 = scientific::leastsq([0.0, 0.0, 0.0], poly_jac) { p => poly_res(p) };
    assert(f2.kind == scientific::F_NONE);
    assert(r2.success);
    assert(row_near(r2.x, optimize_lsq_pins::LSQ_POLY_X, 1.0e-6));
    assert(row_near(c2, optimize_lsq_pins::LSQ_POLY_COV, 1.0e-6));
    assert(common::close(r2.fun, optimize_lsq_pins::LSQ_POLY_FUN[0], 1.0e-5));
    // The problem of the bell.
    let r3, c3, f3 = scientific::leastsq([2.0, 1.5], gau_jac) { p => gau_res(p) };
    assert(f3.kind == scientific::F_NONE);
    assert(r3.success);
    assert(row_near(r3.x, optimize_lsq_pins::LSQ_GAU_X, 1.0e-6));
    assert(row_near(c3, optimize_lsq_pins::LSQ_GAU_COV, 1.0e-6));
    // The scaling of the variables holds the same answer.
    let r4, _, f4 = scientific::leastsq([1.5, 0.8], 1.49012e-8, 1.49012e-8, 0.0, 0, 100.0, [1.0, 1.0], exp_jac) { p => exp_res(p) };
    assert(f4.kind == scientific::F_NONE);
    assert(row_near(r4.x, optimize_lsq_pins::LSQ_EXP_X, 1.0e-6));
    // A small count of the calls gives no fault and no success.
    let r5, _, f5 = scientific::leastsq([1.5, 0.8], 1.49012e-8, 1.49012e-8, 0.0, 2, 100.0, [], exp_jac) { p => exp_res(p) };
    assert(f5.kind == scientific::F_NONE);
    assert(!r5.success && r5.status == 5);
    assert(r5.x.data.len == 2);
}

fn test_leastsq_est() {
    let r1, c1, f1 = scientific::leastsq_est([1.5, 0.8]) { p => exp_res(p) };
    assert(f1.kind == scientific::F_NONE);
    assert(r1.success);
    assert(row_near(r1.x, optimize_lsq_pins::EST_EXP_X, 1.0e-6));
    assert(row_near(c1, optimize_lsq_pins::EST_EXP_COV, 1.0e-6));
    assert(common::close(r1.fun, optimize_lsq_pins::EST_EXP_FUN[0], 1.0e-5));
    assert(r1.nfev >= 1);
    let r2, c2, f2 = scientific::leastsq_est([0.0, 0.0, 0.0]) { p => poly_res(p) };
    assert(f2.kind == scientific::F_NONE);
    assert(row_near(r2.x, optimize_lsq_pins::EST_POLY_X, 1.0e-6));
    assert(row_near(c2, optimize_lsq_pins::EST_POLY_COV, 1.0e-6));
    let r3, c3, f3 = scientific::leastsq_est([2.0, 1.5]) { p => gau_res(p) };
    assert(f3.kind == scientific::F_NONE);
    assert(row_near(r3.x, optimize_lsq_pins::EST_GAU_X, 1.0e-6));
    assert(row_near(c3, optimize_lsq_pins::EST_GAU_COV, 1.0e-6));
}

fn test_leastsq_faults() {
    // The start holds no element.
    let _, _, f1 = scientific::leastsq([], exp_jac) { p => exp_res(p) };
    assert(f1.kind == scientific::F_VALUE);
    // diag holds a wrong count of values.
    let _, _, f2 = scientific::leastsq([1.5, 0.8], 1.49012e-8, 1.49012e-8, 0.0, 0, 100.0, [1.0], exp_jac) { p => exp_res(p) };
    assert(f2.kind == scientific::F_SHAPE);
    // diag holds a value of zero.
    let _, _, f3 = scientific::leastsq([1.5, 0.8], 1.49012e-8, 1.49012e-8, 0.0, 0, 100.0, [1.0, 0.0], exp_jac) { p => exp_res(p) };
    assert(f3.kind == scientific::F_VALUE);
    // ftol holds a value below the smallest difference.
    let _, _, f4 = scientific::leastsq([1.5, 0.8], 0.0, 1.49012e-8, 0.0, 0, 100.0, [], exp_jac) { p => exp_res(p) };
    assert(f4.kind == scientific::F_VALUE);
    // xtol holds a value below the smallest difference.
    let _, _, f5 = scientific::leastsq([1.5, 0.8], 1.49012e-8, 0.0, 0.0, 0, 100.0, [], exp_jac) { p => exp_res(p) };
    assert(f5.kind == scientific::F_VALUE);
    // gtol holds a value below zero.
    let _, _, f6 = scientific::leastsq([1.5, 0.8], 1.49012e-8, 1.49012e-8, -1.0, 0, 100.0, [], exp_jac) { p => exp_res(p) };
    assert(f6.kind == scientific::F_VALUE);
    // maxfev holds a value below zero.
    let _, _, f7 = scientific::leastsq([1.5, 0.8], 1.49012e-8, 1.49012e-8, 0.0, -1, 100.0, [], exp_jac) { p => exp_res(p) };
    assert(f7.kind == scientific::F_VALUE);
    // factor stands outside its range.
    let _, _, f8 = scientific::leastsq([1.5, 0.8], 1.49012e-8, 1.49012e-8, 0.0, 0, 0.05, [], exp_jac) { p => exp_res(p) };
    assert(f8.kind == scientific::F_RANGE);
    let _, _, f9 = scientific::leastsq([1.5, 0.8], 1.49012e-8, 1.49012e-8, 0.0, 0, 200.0, [], exp_jac) { p => exp_res(p) };
    assert(f9.kind == scientific::F_RANGE);
    // The same checks hold for the call that estimates.
    let _, _, g1 = scientific::leastsq_est([]) { p => exp_res(p) };
    assert(g1.kind == scientific::F_VALUE);
    let _, _, g2 = scientific::leastsq_est([1.5, 0.8], 0.0) { p => exp_res(p) };
    assert(g2.kind == scientific::F_VALUE);
    // The Jacobian holds a wrong count of values.
    let _, _, g3 = scientific::leastsq([1.5, 0.8], short_jac) { p => exp_res(p) };
    assert(g3.kind == scientific::F_SHAPE);
    // The function gives fewer values than the point holds.
    let _, _, g4 = scientific::leastsq([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], poly_jac) { p => poly_res(p) };
    assert(g4.kind == scientific::F_SHAPE);
}

fn test_leastsq_singular() {
    // The second variable holds no information, so the covariance is
    // singular and the call gives a fault.
    let _, _, f = scientific::leastsq([0.0, 0.0], one_dep_jac) { p => one_dep(p) };
    assert(f.kind == scientific::F_VALUE);
}

fn test_lsq_linear() {
    // The first problem holds no bound, so the answer is the
    // unconstrained solution.
    let a1 = linmat([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], 4, 3);
    let x1, f1 = scientific::lsq_linear(a1, [1.0, 2.0, 3.0, 6.0], [], []);
    assert(f1.kind == scientific::F_NONE);
    assert(row_near(x1, optimize_lsq_pins::LIN_L1_TRF, 1.0e-6));
    // The second problem holds one bound.
    let b1 = linmat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let x2, f2 = scientific::lsq_linear(b1, [1.0, 2.0, 3.0], [], [1.0e308, 1.0]);
    assert(f2.kind == scientific::F_NONE);
    assert(row_near(x2, optimize_lsq_pins::LIN_L2_TRF, 1.0e-6));
    let x3, f3 = scientific::lsq_linear(b1, [1.0, 2.0, 3.0], [], [1.0e308, 1.0], "bvls");
    assert(f3.kind == scientific::F_NONE);
    assert(row_near(x3, optimize_lsq_pins::LIN_L2_BVLS, 1.0e-6));
    // The third problem holds the matrix and the right side of the
    // pins.
    let a3 = linmat([0.6285702027691996, 0.999277862440115, 1.1014983576233575,
                     0.5286890083719445, 0.6479260845774559, 1.4282110229603695,
                     0.5704205761541968, 0.629773949399298, 1.448328453291775,
                     1.1218835927963828, 0.868993123729791, 1.0113900218032628,
                     1.1628429525167991, 0.7753088157611293, 0.6379680728669553], 5, 3);
    let bv = [0.7880395945039919, 0.6703605841024838, 0.5123823134831604,
              0.8167364359696581, 0.5490752688700263];
    let x4, f4 = scientific::lsq_linear(a3, bv, [-1.0, -1.0, -1.0], [0.5, 2.0, 1.0], "trf");
    assert(f4.kind == scientific::F_NONE);
    assert(row_near(x4, optimize_lsq_pins::LIN_L3_TRF, 1.0e-6));
    let x5, f5 = scientific::lsq_linear(a3, bv, [-1.0, -1.0, -1.0], [0.5, 2.0, 1.0], "bvls");
    assert(f5.kind == scientific::F_NONE);
    assert(row_near(x5, optimize_lsq_pins::LIN_L3_BVLS, 1.0e-6));
    // A bound of infinity leaves the variable free.
    let x6, f6 = scientific::lsq_linear(a1, [1.0, 2.0, 3.0, 6.0], [-1.0e308, -1.0e308, -1.0e308], []);
    assert(f6.kind == scientific::F_NONE);
    assert(row_near(x6, optimize_lsq_pins::LIN_L1_TRF, 1.0e-6));
}

fn test_lsq_linear_faults() {
    let a1 = linmat([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 3);
    // The matrix holds one dimension.
    var v: f64[>..] = [];
    v.push(1.0);
    v.push(2.0);
    v.push(3.0);
    let v1 = scientific::Arr { ndim: 1, shape: [3], data: copy(v) };
    let _, f1 = scientific::lsq_linear(v1, [1.0, 2.0, 3.0], [], []);
    assert(f1.kind == scientific::F_SHAPE);
    // The right side does not agree with the matrix.
    let _, f2 = scientific::lsq_linear(a1, [1.0, 2.0], [], []);
    assert(f2.kind == scientific::F_SHAPE);
    // The method holds none of trf and bvls.
    let _, f3 = scientific::lsq_linear(a1, [1.0, 2.0, 3.0], [], [], "dogbox");
    assert(f3.kind == scientific::F_VALUE);
    // The tolerance holds a value of zero.
    let _, f4 = scientific::lsq_linear(a1, [1.0, 2.0, 3.0], [], [], "trf", 0.0);
    assert(f4.kind == scientific::F_VALUE);
    // The count of turns holds a value below zero.
    let _, f5 = scientific::lsq_linear(a1, [1.0, 2.0, 3.0], [], [], "trf", 1.0e-10, -1);
    assert(f5.kind == scientific::F_VALUE);
    // The lower bounds hold a wrong count of values.
    let _, f6 = scientific::lsq_linear(a1, [1.0, 2.0, 3.0], [0.0, 0.0], []);
    assert(f6.kind == scientific::F_SHAPE);
    // A lower bound lies above its upper bound.
    let _, f7 = scientific::lsq_linear(a1, [1.0, 2.0, 3.0], [0.0, 0.0, 5.0], [1.0, 1.0, 1.0]);
    assert(f7.kind == scientific::F_VALUE);
    // The system holds a value that is not finite.
    let _, f8 = scientific::lsq_linear(a1, [1.0, 2.0, 0.0 / 0.0], [], []);
    assert(f8.kind == scientific::F_VALUE);
}

fn test_qap_faq() {
    let a1 = linmat([0.0, 80.0, 150.0, 170.0, 80.0, 0.0, 130.0, 100.0,
                     150.0, 130.0, 0.0, 120.0, 170.0, 100.0, 120.0, 0.0], 4, 4);
    let b1 = linmat([0.0, 5.0, 2.0, 7.0, 0.0, 0.0, 3.0, 8.0,
                     0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], 4, 4);
    let p1, f1 = scientific::quadratic_assignment(a1, b1);
    assert(f1.kind == scientific::F_NONE);
    assert(row_near(p1, optimize_lsq_pins::QAP_Q1_COL, 0.0));
    let p2, f2 = scientific::quadratic_assignment(a1, b1, "faq");
    assert(f2.kind == scientific::F_NONE);
    assert(row_near(p2, optimize_lsq_pins::QAP_Q1_COL, 0.0));
    // The second pair holds the matrices of the pins.
    let a2 = linmat([3.0, 69.0, 42.0, 64.0, 26.0, 12.0, 60.0, 11.0, 12.0, 65.0,
                     65.0, 85.0, 43.0, 20.0, 20.0, 21.0, 10.0, 71.0, 76.0, 47.0,
                     68.0, 41.0, 73.0, 34.0, 47.0], 5, 5);
    let b2 = linmat([6.0, 18.0, 45.0, 15.0, 30.0, 1.0, 38.0, 13.0, 54.0, 33.0,
                     68.0, 7.0, 62.0, 4.0, 74.0, 28.0, 1.0, 27.0, 65.0, 29.0,
                     54.0, 94.0, 85.0, 11.0, 93.0], 5, 5);
    let p3, f3 = scientific::quadratic_assignment(a2, b2);
    assert(f3.kind == scientific::F_NONE);
    assert(row_near(p3, optimize_lsq_pins::QAP_Q2_COL, 0.0));
}

fn test_qap_2opt() {
    let a1 = linmat([0.0, 80.0, 150.0, 170.0, 80.0, 0.0, 130.0, 100.0,
                     150.0, 130.0, 0.0, 120.0, 170.0, 100.0, 120.0, 0.0], 4, 4);
    let b1 = linmat([0.0, 5.0, 2.0, 7.0, 0.0, 0.0, 3.0, 8.0,
                     0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], 4, 4);
    let p1, f1 = scientific::quadratic_assignment(a1, b1, "2opt");
    assert(f1.kind == scientific::F_NONE);
    assert(is_perm(p1.data));
    // The method 2opt starts from a random permutation, so its answer
    // holds a permutation of the rows and no more.
    let p2, f2 = scientific::quadratic_assignment(a1, b1, "2opt", 3);
    assert(f2.kind == scientific::F_NONE);
    assert(is_perm(p2.data));
}

fn test_qap_faults() {
    let bad = linmat([1.0, 2.0, 3.0], 1, 3);
    let good = linmat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
    // The first matrix is not square.
    let _, f1 = scientific::quadratic_assignment(bad, good);
    assert(f1.kind == scientific::F_SHAPE);
    // The matrices do not hold the same side.
    let _, f2 = scientific::quadratic_assignment(good, linmat([1.0, 2.0, 3.0, 4.0], 2, 2));
    assert(f2.kind == scientific::F_SHAPE);
    // The method holds none of faq and 2opt.
    let _, f3 = scientific::quadratic_assignment(good, good, "rust");
    assert(f3.kind == scientific::F_VALUE);
    // A value of the matrices is not finite.
    let _, f4 = scientific::quadratic_assignment(good, linmat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0 / 0.0], 3, 3));
    assert(f4.kind == scientific::F_VALUE);
    // The side holds a value above the limit of the method 2opt.
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < 65 * 65 { v.push(1.0); i += 1; }
    let b65 = linmat(v, 65, 65);
    let _, f5 = scientific::quadratic_assignment(b65, b65, "2opt");
    assert(f5.kind == scientific::F_LIMIT);
    // The side holds a value above the limit of the method faq.
    var w: f64[>..] = [];
    i = 0;
    while i < 257 * 257 { w.push(1.0); i += 1; }
    let b257 = linmat(w, 257, 257);
    let _, f6 = scientific::quadratic_assignment(b257, b257);
    assert(f6.kind == scientific::F_LIMIT);
}

fn test_shgo() {
    let x1, f1 = scientific::shgo([-5.0, -5.0], [5.0, 5.0], 100, 1) { p => shg_one(p) };
    assert(f1.kind == scientific::F_NONE);
    assert(row_near(x1, optimize_lsq_pins::SHG_S1_X, 1.0e-6));
    let x2, f2 = scientific::shgo([-5.0, -5.0], [5.0, 5.0], 100, 1) { p => shg_two(p) };
    assert(f2.kind == scientific::F_NONE);
    assert(row_near(x2, optimize_lsq_pins::SHG_S2_X, 1.0e-6));
    let x3, f3 = scientific::shgo([-5.0], [5.0], 100, 1) { p => shg_three(p) };
    assert(f3.kind == scientific::F_NONE);
    assert(row_near(x3, optimize_lsq_pins::SHG_S3_X, 1.0e-6));
    // The second round holds the same point with more samples.
    let x4, f4 = scientific::shgo([-5.0, -5.0], [5.0, 5.0], 40, 2) { p => shg_one(p) };
    assert(f4.kind == scientific::F_NONE);
    assert(row_near(x4, optimize_lsq_pins::SHG_S1_X, 1.0e-6));
}

fn test_shgo_faults() {
    // The bounds hold no element.
    let _, f1 = scientific::shgo([], [], 10, 1) { p => shg_one(p) };
    assert(f1.kind == scientific::F_VALUE);
    // The bounds do not hold the same count.
    let _, f2 = scientific::shgo([-5.0], [5.0, 5.0], 10, 1) { p => shg_one(p) };
    assert(f2.kind == scientific::F_SHAPE);
    // The count of the samples holds a value below the variables plus one.
    let _, f3 = scientific::shgo([-5.0, -5.0], [5.0, 5.0], 2, 1) { p => shg_one(p) };
    assert(f3.kind == scientific::F_VALUE);
    // The count of the rounds holds a value below one.
    let _, f4 = scientific::shgo([-5.0, -5.0], [5.0, 5.0], 10, 0) { p => shg_one(p) };
    assert(f4.kind == scientific::F_VALUE);
    // A lower bound lies above its upper bound.
    let _, f5 = scientific::shgo([5.0, -5.0], [-5.0, 5.0], 10, 1) { p => shg_one(p) };
    assert(f5.kind == scientific::F_VALUE);
    // The count of the rounds holds a value above the limit.
    let _, f6 = scientific::shgo([-5.0], [5.0], 10, 101) { p => shg_three(p) };
    assert(f6.kind == scientific::F_LIMIT);
}

fn test_optimize_result_structs() {
    let r = scientific::optimize_result([1.0, 2.0], 3.5, true, 2, "done", 7, 5, 4);
    assert(r.x.data.len == 2 && r.x.data[0] == 1.0 && r.x.data[1] == 2.0);
    assert(r.fun == 3.5 && r.success && r.status == 2);
    assert(r.message == "done");
    assert(r.nfev == 7 && r.njev == 5 && r.nit == 4);
    let s = scientific::root_result([0.5], 9, 11, false, "brentq");
    assert(s.root.data.len == 1 && s.root.data[0] == 0.5);
    assert(s.iterations == 9 && s.function_calls == 11);
    assert(!s.converged && s.flag == "brentq");
    // The struct of a failed run holds the flag of the failure and the
    // last point.
    let t = scientific::root_result([1.5, 2.5], 100, 200, false, "no convergence");
    assert(!t.converged && t.root.data.len == 2);
}

fn test_nl_constraint() {
    let c1, f1 = scientific::nl_constraint(0.0, 1.0, "ineq", true);
    assert(f1.kind == scientific::F_NONE);
    assert(c1.lb == 0.0 && c1.ub == 1.0 && c1.kind == "ineq" && c1.keep_feasible);
    let c2, f2 = scientific::nl_constraint(2.0, 2.0, "eq", false);
    assert(f2.kind == scientific::F_NONE);
    assert(c2.lb == 2.0 && c2.ub == 2.0 && c2.kind == "eq");
    let c3, f3 = scientific::nl_constraint(-1.0, 1.0, "both", false);
    assert(f3.kind == scientific::F_NONE);
    assert(c3.kind == "both");
    // The kind holds none of eq, ineq and both.
    let _, f4 = scientific::nl_constraint(0.0, 1.0, "less", false);
    assert(f4.kind == scientific::F_VALUE);
    // No point satisfies the bounds.
    let _, f5 = scientific::nl_constraint(1.0, 0.0, "ineq", false);
    assert(f5.kind == scientific::F_VALUE);
    // A constraint of the kind eq needs equal bounds.
    let _, f6 = scientific::nl_constraint(0.0, 1.0, "eq", false);
    assert(f6.kind == scientific::F_VALUE);
}
