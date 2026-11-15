// The tests of the four methods of minimization of SciPy. Run them
// with SCIENTIFIC_SUITES=optimize_fmin.
//
// The calls match the pins of SciPy 1.18.1 in the file
// optimize_fmin_pins. Each row of the pins holds a code for the
// problem, the start point, the answer of SciPy, the value of the
// function there and the tolerance of the comparison. The problems
// are:
//
// 1 the sphere x0^2 + 2 x1^2
// 2 the function of Rosenbrock
// 3 the square to the point (1, 2) with the constraint x0 + x1 - 2
// 4 the square to the point (-0.5, 0.25) under bounds
// 6 the function of Rosenbrock with the equality x1 - x0^3 - 1
// 7 the square to the point (2, 1) under bounds
// 8 the square with a cross term x0^2 + x0 x1 + x1^2
namespace optimize_fmin_tests;

import scientific from "../..";
import optimize_fmin_pins;
import common;

// row_at gives the seven numbers of a row of a list of pins.
fn row_at(pins: f64[:], i: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var k: i64 = 0;
    while k < 7 { out.push(pins[7 * i + k]); k += 1; }
    return out;
}

// vec2 gives a list of two numbers.
fn vec2(a: f64, b: f64) -> f64[>..] {
    var o: f64[>..] = [];
    o.push(a);
    o.push(b);
    return o;
}

// rosen_val gives the value of the function of Rosenbrock.
fn rosen_val(v: f64[:]) -> f64[>..] {
    let t = v[1] - v[0] * v[0];
    return vec2((1.0 - v[0]) * (1.0 - v[0]) + 100.0 * t * t, 0.0);
}

// powell_fun gives the value of the function of a case of Powell.
fn powell_fun(case: i64, v: f64[:]) -> f64[>..] {
    if case == 1 {
        var o: f64[>..] = [];
        o.push(v[0] * v[0] + 2.0 * v[1] * v[1]);
        return o;
    }
    return rosen_val(v);
}

// cobyla_fun gives the values of the function and of the constraints
// of a case of the method of Powell with constraints.
fn cobyla_fun(case: i64, v: f64[:]) -> f64[>..] {
    var o: f64[>..] = [];
    if case == 3 {
        o.push((v[0] - 1.0) * (v[0] - 1.0) + (v[1] - 2.0) * (v[1] - 2.0));
        o.push(v[0] + v[1] - 2.0);
        return o;
    }
    o.push(v[0] * v[0] + 2.0 * v[1] * v[1]);
    return o;
}

// slsqp_fun gives the values of the function and of the constraints
// of a case of the method of the least squares.
fn slsqp_fun(case: i64, v: f64[:]) -> f64[>..] {
    var o: f64[>..] = [];
    if case == 3 {
        o.push((v[0] - 1.0) * (v[0] - 1.0) + (v[1] - 2.0) * (v[1] - 2.0));
        o.push(v[0] + v[1] - 2.0);
        return o;
    }
    if case == 6 {
        let t = v[1] - v[0] * v[0];
        o.push((1.0 - v[0]) * (1.0 - v[0]) + 100.0 * t * t);
        o.push(v[1] - v[0] * v[0] * v[0] - 1.0);
        return o;
    }
    o.push((v[0] + 0.5) * (v[0] + 0.5) + (v[1] - 0.25) * (v[1] - 0.25));
    return o;
}

// tnc_fun gives the value and the gradient of a case of the method of
// the truncated Newton.
fn tnc_fun(case: i64, v: f64[:]) -> f64[>..] {
    var o: f64[>..] = [];
    if case == 7 {
        o.push((v[0] - 2.0) * (v[0] - 2.0) + (v[1] - 1.0) * (v[1] - 1.0));
        o.push(2.0 * (v[0] - 2.0));
        o.push(2.0 * (v[1] - 1.0));
        return o;
    }
    if case == 1 {
        o.push(v[0] * v[0] + 2.0 * v[1] * v[1]);
        o.push(2.0 * v[0]);
        o.push(4.0 * v[1]);
        return o;
    }
    o.push(v[0] * v[0] + v[0] * v[1] + v[1] * v[1]);
    o.push(2.0 * v[0] + v[1]);
    o.push(v[0] + 2.0 * v[1]);
    return o;
}

// check_row holds the checks that all four tests share. The point and
// the value must sit within the tolerance of the pin.
fn check_row(ans: scientific::Arr&, fr: scientific::Fault&, r: f64[:]) {
    assert(fr.kind == scientific::F_NONE);
    assert(ans.data.len == 2);
    let tol = r[6];
    assert(common::close(ans.data[0], r[3], tol));
    assert(common::close(ans.data[1], r[4], tol));
}

fn test_fmin_powell_pins() {
    let pins = optimize_fmin_pins::POWELL;
    var i: i64 = 0;
    while i < pins.len / 7 {
        let r = row_at(pins, i);
        let case = r[0] as i64;
        let ans, fr = scientific::fmin_powell(vec2(r[1], r[2]), 0.0001, 0.0001) { v => powell_fun(case, v) };
        check_row(ans, fr, r);
        assert(common::close(powell_fun(case, ans.data)[0], r[5], r[6]));
        i += 1;
    }
}

fn test_fmin_cobyla_pins() {
    let pins = optimize_fmin_pins::COBYLA;
    var i: i64 = 0;
    while i < pins.len / 7 {
        let r = row_at(pins, i);
        let case = r[0] as i64;
        let ans, fr = scientific::fmin_cobyla(vec2(r[1], r[2]), 1.0, 0.00000001, 5000, 0.00000001) { v => cobyla_fun(case, v) };
        check_row(ans, fr, r);
        assert(common::close(cobyla_fun(case, ans.data)[0], r[5], r[6]));
        i += 1;
    }
}

fn test_fmin_slsqp_pins() {
    let pins = optimize_fmin_pins::SLSQP;
    var i: i64 = 0;
    while i < pins.len / 7 {
        let r = row_at(pins, i);
        let case = r[0] as i64;
        var neq: i64 = 0;
        var lb: f64[>..] = [];
        var ub: f64[>..] = [];
        if case == 6 { neq = 1; }
        if case == 4 {
            lb = vec2(-1.0, -1.0);
            ub = vec2(0.3, 0.1);
        }
        let ans, fr = scientific::fmin_slsqp(vec2(r[1], r[2]), neq, lb, ub, 0.00000001, 100) { v => slsqp_fun(case, v) };
        check_row(ans, fr, r);
        assert(common::close(slsqp_fun(case, ans.data)[0], r[5], r[6]));
        i += 1;
    }
}

fn test_fmin_tnc_pins() {
    let pins = optimize_fmin_pins::TNC;
    var i: i64 = 0;
    while i < pins.len / 7 {
        let r = row_at(pins, i);
        let case = r[0] as i64;
        var lb: f64[>..] = [];
        var ub: f64[>..] = [];
        var ftol: f64 = -1.0;
        var xtol: f64 = -1.0;
        var pgtol: f64 = -1.0;
        if case == 7 {
            lb = vec2(0.0, -5.0);
            ub = vec2(1.0, 5.0);
        }
        let ans, fr = scientific::fmin_tnc(vec2(r[1], r[2]), lb, ub, 0.00000001, 1000, ftol, xtol, pgtol) { v => tnc_fun(case, v) };
        check_row(ans, fr, r);
        assert(common::close(tnc_fun(case, ans.data)[0], r[5], r[6]));
        i += 1;
    }
}

fn test_fmin_faults() {
    // The start of Powell holds no element.
    let a1, f1 = scientific::fmin_powell(common::empty_row()) { v => powell_fun(1, v) };
    assert(f1.kind == scientific::F_VALUE);
    // A tolerance of Powell holds a value of zero or below.
    let a2, f2 = scientific::fmin_powell(vec2(2.0, 3.0), 0.0, 0.0001) { v => powell_fun(1, v) };
    assert(f2.kind == scientific::F_VALUE);
    let a3, f3 = scientific::fmin_powell(vec2(2.0, 3.0), 0.0001, -1.0) { v => powell_fun(1, v) };
    assert(f3.kind == scientific::F_VALUE);
    // The function of Powell holds no value.
    let a4, f4 = scientific::fmin_powell(vec2(2.0, 3.0)) { v => common::empty_row() };
    assert(f4.kind == scientific::F_VALUE);
    // The method of Powell reaches its count of turns.
    let a5, f5 = scientific::fmin_powell(vec2(-1.2, 1.0), 0.0001, 0.0001, 1) { v => powell_fun(2, v) };
    assert(f5.kind == scientific::F_LIMIT);
    // The start of Powell holds a value that is not a number.
    let nn = 0.0 / 0.0;
    let a6, f6 = scientific::fmin_powell(vec2(nn, 1.0)) { v => powell_fun(1, v) };
    assert(f6.kind == scientific::F_VALUE);

    // The start of the method with constraints holds no element.
    let b1, g1 = scientific::fmin_cobyla(common::empty_row()) { v => cobyla_fun(3, v) };
    assert(g1.kind == scientific::F_VALUE);
    // A radius of the method with constraints is not valid.
    let b2, g2 = scientific::fmin_cobyla(vec2(0.0, 0.0), 0.0, 0.0001) { v => cobyla_fun(3, v) };
    assert(g2.kind == scientific::F_VALUE);
    let b3, g3 = scientific::fmin_cobyla(vec2(0.0, 0.0), 0.001, 0.01) { v => cobyla_fun(3, v) };
    assert(g3.kind == scientific::F_VALUE);
    // The tolerance of the constraints is not valid.
    let b4, g4 = scientific::fmin_cobyla(vec2(0.0, 0.0), 1.0, 0.0001, 1000, -1.0) { v => cobyla_fun(3, v) };
    assert(g4.kind == scientific::F_VALUE);
    // The count of calls is not valid.
    let b5, g5 = scientific::fmin_cobyla(vec2(0.0, 0.0), 1.0, 0.0001, 0) { v => cobyla_fun(3, v) };
    assert(g5.kind == scientific::F_VALUE);
    // The function holds no value.
    let b6, g6 = scientific::fmin_cobyla(vec2(0.0, 0.0)) { v => common::empty_row() };
    assert(g6.kind == scientific::F_VALUE);
    // The count of calls runs out before the radius reaches its end.
    let b7, g7 = scientific::fmin_cobyla(vec2(0.0, 0.0), 1.0, 0.00000001, 5, 0.00000001) { v => cobyla_fun(3, v) };
    assert(g7.kind == scientific::F_LIMIT);

    // The start of the method of the least squares holds no element.
    let c1, h1 = scientific::fmin_slsqp(common::empty_row(), 0) { v => slsqp_fun(3, v) };
    assert(h1.kind == scientific::F_VALUE);
    // The accuracy or the count of turns is not valid.
    let c2, h2 = scientific::fmin_slsqp(vec2(0.0, 0.0), 0, common::empty_row(), common::empty_row(), 0.0) { v => slsqp_fun(3, v) };
    assert(h2.kind == scientific::F_VALUE);
    let c3, h3 = scientific::fmin_slsqp(vec2(0.0, 0.0), 0, common::empty_row(), common::empty_row(), 0.000001, 0) { v => slsqp_fun(3, v) };
    assert(h3.kind == scientific::F_VALUE);
    let c4, h4 = scientific::fmin_slsqp(vec2(0.0, 0.0), -1) { v => slsqp_fun(3, v) };
    assert(h4.kind == scientific::F_VALUE);
    // neq stands above the count of the constraints.
    let c5, h5 = scientific::fmin_slsqp(vec2(0.0, 0.0), 2) { v => slsqp_fun(3, v) };
    assert(h5.kind == scientific::F_SHAPE);
    // The bounds hold a wrong count of values.
    let c6, h6 = scientific::fmin_slsqp(vec2(0.0, 0.0), 0, vec2(0.0, 0.0)) { v => slsqp_fun(3, v) };
    assert(h6.kind == scientific::F_SHAPE);
    // A lower bound lies above an upper bound.
    let c7, h7 = scientific::fmin_slsqp(vec2(0.0, 0.0), 0, vec2(1.0, 0.0), vec2(0.0, 1.0)) { v => slsqp_fun(3, v) };
    assert(h7.kind == scientific::F_VALUE);
    // The method of the least squares reaches its count of turns.
    let c8, h8 = scientific::fmin_slsqp(vec2(0.5, 2.0), 1, common::empty_row(), common::empty_row(), 0.000000000001, 2) { v => slsqp_fun(6, v) };
    assert(h8.kind == scientific::F_LIMIT);

    // The start of the method of the truncated Newton holds no
    // element.
    let d1, k1 = scientific::fmin_tnc(common::empty_row()) { v => tnc_fun(1, v) };
    assert(k1.kind == scientific::F_VALUE);
    // The bounds hold a wrong count of values.
    let d2, k2 = scientific::fmin_tnc(vec2(0.0, 0.0), vec2(0.0, 0.0)) { v => tnc_fun(1, v) };
    assert(k2.kind == scientific::F_SHAPE);
    // A lower bound lies above an upper bound.
    let d3, k3 = scientific::fmin_tnc(vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 0.0)) { v => tnc_fun(1, v) };
    assert(k3.kind == scientific::F_VALUE);
    // The step of the difference holds a value of zero or below.
    let d4, k4 = scientific::fmin_tnc(vec2(0.0, 0.0), common::empty_row(), common::empty_row(), 0.0) { v => tnc_fun(1, v) };
    assert(k4.kind == scientific::F_VALUE);
    // The function holds a wrong count of values.
    let d5, k5 = scientific::fmin_tnc(vec2(0.0, 0.0)) { v => vec2(v[0], v[1]) };
    assert(k5.kind == scientific::F_VALUE);
    // The count of calls runs out before the end.
    let d6, k6 = scientific::fmin_tnc(vec2(2.0, 3.0), common::empty_row(), common::empty_row(), 0.00000001, 1) { v => tnc_fun(1, v) };
    assert(k6.kind == scientific::F_LIMIT);
    // A tolerance of the method of the truncated Newton is not a
    // number.
    let nn2 = 0.0 / 0.0;
    let d7, k7 = scientific::fmin_tnc(vec2(2.0, 3.0), common::empty_row(), common::empty_row(), 0.00000001, 0, nn2) { v => tnc_fun(1, v) };
    assert(k7.kind == scientific::F_VALUE);
}
