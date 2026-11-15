// The tests of the group integrate. Run them with SCIENTIFIC_SUITES=integrate.
//
// The pins come from the generated file of the group. The values of
// the solver of the equations agree to the tolerance, so the tests
// hold the tolerance of 1e-6 and not the last digit.
namespace integrate_tests;

import scientific from "../..";
import os;
import common;

import integrate_pins;

// osc_f gives the right side of the oscillator.
fn osc_f(x: f64, y: f64[:], p: f64[:]) -> f64[>..] {
    var out: f64[>..] = [];
    out.push(y[1]);
    out.push(0.0 - y[0]);
    return out;
}

// osc_bc gives the conditions of the oscillator at the two ends.
fn osc_bc(ya: f64[:], yb: f64[:], p: f64[:]) -> f64[>..] {
    var out: f64[>..] = [];
    out.push(ya[0] - 1.0);
    out.push(yb[0] + 1.0);
    return out;
}

// par_f gives the right side of the problem with a parameter.
fn par_f(x: f64, y: f64[:], p: f64[:]) -> f64[>..] {
    var out: f64[>..] = [];
    out.push(p[0] * y[0]);
    return out;
}

// par_bc gives the conditions of the problem with a parameter.
fn par_bc(ya: f64[:], yb: f64[:], p: f64[:]) -> f64[>..] {
    var out: f64[>..] = [];
    out.push(ya[0] - 1.0);
    out.push(yb[0] - 2.718281828459045);
    return out;
}

// bad_bc gives a count of conditions that the system does not hold.
fn bad_bc(ya: f64[:], yb: f64[:], p: f64[:]) -> f64[>..] {
    var out: f64[>..] = [];
    out.push(ya[0]);
    return out;
}

// stiff_f gives the stiff pair with the eigenvalues -1 and -1000.
fn stiff_f(t: f64, y: f64[:]) -> f64[>..] {
    var out: f64[>..] = [];
    out.push(998.0 * y[0] + 1998.0 * y[1]);
    out.push(-999.0 * y[0] - 1999.0 * y[1]);
    return out;
}

// wave_f gives the derivative of the wave z' = i*z.
fn wave_f(t: f64, z: scientific::Cplx[>..]) -> scientific::Cplx[>..] {
    var out: scientific::Cplx[>..] = [];
    var i: i64 = 0;
    while i < z.len { out.push(scientific::Cplx { re: 0.0 - z[i].im, im: z[i].re }); i += 1; }
    return out;
}

// damp_f gives the derivative of the damped wave z' = (-1 + i)*z.
fn damp_f(t: f64, z: scientific::Cplx[>..]) -> scientific::Cplx[>..] {
    var out: scientific::Cplx[>..] = [];
    var i: i64 = 0;
    while i < z.len {
        out.push(scientific::Cplx { re: -z[i].re - z[i].im, im: z[i].re - z[i].im });
        i += 1;
    }
    return out;
}

// bad_f gives a value that is not finite.
fn bad_f(t: f64, y: f64[:]) -> f64[>..] {
    var out: f64[>..] = [];
    out.push(0.0 / 0.0);
    return out;
}

// pin_at tells whether the values of the pin agree with the answer at
// the node at of the first mesh. The mesh of the answer keeps every
// node of the first mesh, so the test finds the node by its place.
fn pin_at(bx: scientific::Arr&, by: scientific::Arr&, x0: f64[:], at: i64, n: i64, pin_y: f64[:]) -> bool {
    var i: i64 = 0;
    while i < bx.data.len {
        if bx.data[i] == x0[at] {
            var j: i64 = 0;
            while j < n {
                if !common::close(by.data[i * n + j], pin_y[at * n + j], 1.0e-6) { return false; }
                j += 1;
            }
            return true;
        }
        i += 1;
    }
    return false;
}

// RK23 takes the steps of SciPy, so the values agree to the last
// digits. The test holds the tolerance of the group.
fn test_integrate_rk23() {
    let t1, y1, f1 = scientific::solve_ivp(0.0, 1.0, [1.0, 0.0], 1.0e-8, 1.0e-10, "RK23", 1.0e308, scientific::copy_slice(integrate_pins::RK23_T), 0.0) { tt, yy => osc_f(tt, yy, []) };
    assert(f1.kind == scientific::F_NONE);
    assert(t1.data.len == integrate_pins::RK23_T.len);
    var i: i64 = 0;
    while i < integrate_pins::RK23_Y.len {
        assert(common::close(y1.data[i], integrate_pins::RK23_Y[i], 1.0e-6));
        i += 1;
    }
    // The method with an unknown name is a fault.
    let t2, y2, f2 = scientific::solve_ivp(0.0, 1.0, [1.0, 0.0], 1.0e-8, 1.0e-10, "RK12", 1.0e308, [], 0.0) { tt, yy => osc_f(tt, yy, []) };
    assert(f2.kind == scientific::F_VALUE);
    return;
}

// LSODA runs the Adams method and the BDF method, so the values agree
// to the tolerance and the steps differ from SciPy.
fn test_integrate_lsoda() {
    let t1, y1, f1 = scientific::solve_ivp(0.0, 2.0, [1.0, 0.0], 1.0e-8, 1.0e-10, "LSODA", 1.0e308, scientific::copy_slice(integrate_pins::LSODA_T), 0.0) { tt, yy => stiff_f(tt, yy) };
    assert(f1.kind == scientific::F_NONE);
    assert(t1.data.len == integrate_pins::LSODA_T.len);
    var i: i64 = 0;
    while i < integrate_pins::LSODA_Y.len {
        assert(common::close(y1.data[i], integrate_pins::LSODA_Y[i], 1.0e-6));
        i += 1;
    }
    // A function that holds a value that is not finite is a fault.
    let t2, y2, f2 = scientific::solve_ivp(0.0, 1.0, [1.0], 1.0e-8, 1.0e-10, "LSODA", 1.0e308, [], 0.0) { tt, yy => bad_f(tt, yy) };
    assert(f2.kind == scientific::F_VALUE || f2.kind == scientific::F_LIMIT);
    // A first step outside the span is a fault.
    let t3, y3, f3 = scientific::solve_ivp(0.0, 2.0, [1.0, 0.0], 1.0e-8, 1.0e-10, "LSODA", 1.0e308, [], 4.0) { tt, yy => stiff_f(tt, yy) };
    assert(f3.kind == scientific::F_VALUE);
    return;
}

// solve_bvp solves the oscillator and the problem with a parameter.
// The mesh of the answer keeps every node of the first mesh.
fn test_integrate_bvp() {
    let x0 = scientific::copy_slice(integrate_pins::BVP_X);
    let y0, y0f = scientific::full([5, 2], 0.0);
    assert(y0f.kind == scientific::F_NONE);
    let bx, by, bp, br, bf = scientific::solve_bvp(x0, copy(y0), [], 1.0e-8, 100000, osc_bc) { xx, yy, pp => osc_f(xx, yy, pp) };
    assert(bf.kind == scientific::F_NONE);
    var at: i64 = 0;
    while at < 5 {
        assert(pin_at(bx, by, x0, at, 2, integrate_pins::BVP_Y));
        at += 1;
    }
    // The problem with the unknown parameter.
    let x1 = scientific::copy_slice(integrate_pins::BVP_P_X);
    let y1, y1f = scientific::full([5, 1], 1.0);
    assert(y1f.kind == scientific::F_NONE);
    let px, py, pp2, pr, pf = scientific::solve_bvp(x1, copy(y1), [0.5], 1.0e-6, 100000, par_bc) { xx, yy, ppq => par_f(xx, yy, ppq) };
    assert(pf.kind == scientific::F_NONE);
    assert(common::close(pp2.data[0], integrate_pins::BVP_P[0], 1.0e-6));
    at = 0;
    while at < 5 {
        assert(pin_at(px, py, x1, at, 1, integrate_pins::BVP_P_Y));
        at += 1;
    }
    // A mesh that does not rise strictly is a fault.
    let g1, g1f = scientific::full([3, 1], 1.0);
    assert(g1f.kind == scientific::F_NONE);
    let bx2, by2, bp3, br2, bf2 = scientific::solve_bvp([0.0, 0.5, 0.25], copy(g1), [], 1.0e-6, 1000, par_bc) { xx, yy, pp => par_f(xx, yy, pp) };
    assert(bf2.kind == scientific::F_VALUE);
    // A guess of another count of nodes is a fault.
    let g2, g2f = scientific::full([4, 2], 0.0);
    assert(g2f.kind == scientific::F_NONE);
    let bx3, by3, bp4, br3, bf3 = scientific::solve_bvp(x0, copy(g2), [], 1.0e-6, 1000, osc_bc) { xx, yy, pp => osc_f(xx, yy, pp) };
    assert(bf3.kind == scientific::F_SHAPE);
    // A count of nodes that stays below the need of the mesh is a fault.
    let bx4, by4, bp5, br4, bf4 = scientific::solve_bvp(x0, copy(y0), [], 1.0e-6, 5, osc_bc) { xx, yy, pp => osc_f(xx, yy, pp) };
    assert(bf4.kind == scientific::F_LIMIT);
    // Conditions that hold another count of values are a fault.
    let bx5, by5, bp6, br5, bf5 = scientific::solve_bvp(x0, copy(y0), [], 1.0e-6, 1000, bad_bc) { xx, yy, pp => osc_f(xx, yy, pp) };
    assert(bf5.kind == scientific::F_SHAPE);
    return;
}

// cubature integrates over a box of two and of three dimensions.
fn test_integrate_cubature() {
    let v1, e1, f1 = scientific::cubature([0.0, 0.0], [1.0, 1.0], 1000) { x => common::mathshim_exp(0.0 - x[0] - x[1]) };
    assert(f1.kind == scientific::F_NONE && common::close(v1, integrate_pins::CUB_2D[0], 1.0e-6));
    let v2, e2, f2 = scientific::cubature([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1000) { x => common::mathshim_exp(0.0 - x[0] - x[1] - x[2]) };
    assert(f2.kind == scientific::F_NONE && common::close(v2, integrate_pins::CUB_3D[0], 1.0e-6));
    let v3, e3, f3 = scientific::cubature([0.0, 0.0], [3.141592653589793, 1.5707963267948966], 1000) { x => common::mathshim_sin(x[0]) * common::mathshim_cos(x[1]) };
    assert(f3.kind == scientific::F_NONE && common::close(v3, integrate_pins::CUB_TRIG[0], 1.0e-6));
    // Corners of a different count of values are a fault.
    let v4, e4, f4 = scientific::cubature([0.0], [1.0, 1.0], 1000) { x => x[0] };
    assert(f4.kind == scientific::F_SHAPE);
    // A box of nine dimensions is a fault.
    let v5, e5, f5 = scientific::cubature([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1000) { x => x[0] };
    assert(f5.kind == scientific::F_LIMIT);
    // A corner that is not finite is a fault.
    let v6, e6, f6 = scientific::cubature([0.0 / 0.0, 0.0], [1.0, 1.0], 1000) { x => x[0] };
    assert(f6.kind == scientific::F_VALUE);
    return;
}

// lebedev_rule gives the points and the weights of the rule of the
// sphere. The weights sum to the area of the unit sphere.
fn test_integrate_lebedev() {
    let x1, w1, f1 = scientific::lebedev_rule(3);
    assert(f1.kind == scientific::F_NONE);
    assert(common::same_shape(x1, [3, 6]));
    assert(common::close(x1.data[0], integrate_pins::LEB3[0], 1.0e-6) && common::close(x1.data[1], integrate_pins::LEB3[1], 1.0e-6) && common::close(x1.data[2], integrate_pins::LEB3[2], 1.0e-6));
    assert(common::close(w1.data[0], integrate_pins::LEB3[3], 1.0e-6));
    var sum: f64 = 0.0;
    var i: i64 = 0;
    while i < w1.data.len { sum += w1.data[i]; i += 1; }
    assert(common::close(sum, integrate_pins::LEB3_SUM[0], 1.0e-6));
    let x9, w9, f2 = scientific::lebedev_rule(9);
    assert(f2.kind == scientific::F_NONE);
    sum = 0.0;
    i = 0;
    while i < w9.data.len { sum += w9.data[i]; i += 1; }
    assert(common::close(sum, integrate_pins::LEB9_SUM[0], 1.0e-6));
    let x41, w41, f3 = scientific::lebedev_rule(41);
    assert(f3.kind == scientific::F_NONE);
    sum = 0.0;
    i = 0;
    while i < w41.data.len { sum += w41.data[i]; i += 1; }
    assert(common::close(sum, integrate_pins::LEB41_SUM[0], 1.0e-6));
    let x131, w131, f4 = scientific::lebedev_rule(131);
    assert(f4.kind == scientific::F_NONE && w131.data.len == 5810);
    assert(common::close(x131.data[0], integrate_pins::LEB131[0], 1.0e-6) && common::close(w131.data[0], integrate_pins::LEB131[3], 1.0e-6));
    sum = 0.0;
    i = 0;
    while i < w131.data.len { sum += w131.data[i]; i += 1; }
    assert(common::close(sum, integrate_pins::LEB131_SUM[0], 1.0e-6));
    // An order without a rule is a fault.
    let x4, w4, f5 = scientific::lebedev_rule(4);
    assert(f5.kind == scientific::F_VALUE);
    return;
}

// complex_ode solves the system of the complex values.
fn test_integrate_complex() {
    let y0, fy0 = scientific::complex_from_values([1.0], [0.0]);
    assert(fy0.kind == scientific::F_NONE);
    let t1, z1, f1 = scientific::complex_ode(0.0, 1.0, copy(y0), 1.0e-10, 1.0e-12, "RK45", 1.0e308, []) { tt, zz => wave_f(tt, copy(zz)) };
    assert(f1.kind == scientific::F_NONE);
    assert(t1.data.len == 1);
    assert(common::close(z1.data[0], integrate_pins::CPLX_A[0], 1.0e-6) && common::close(z1.data[1], integrate_pins::CPLX_A[1], 1.0e-6));
    let y2, fy2 = scientific::complex_from_values([1.0, -0.5], [1.0, 0.25]);
    assert(fy2.kind == scientific::F_NONE);
    let t2, z2, f2 = scientific::complex_ode(0.0, 2.0, copy(y2), 1.0e-10, 1.0e-12, "RK45", 1.0e308, []) { tt, zz => damp_f(tt, copy(zz)) };
    assert(f2.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < integrate_pins::CPLX_B.len {
        assert(common::close(z2.data[i], integrate_pins::CPLX_B[i], 1.0e-6));
        i += 1;
    }
    // A start of another count of dimensions is a fault.
    let t3, z3, f3 = scientific::complex_ode(0.0, 1.0, scientific::carr_make([2, 2], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]), 1.0e-10, 1.0e-12, "RK45", 1.0e308, []) { tt, zz => wave_f(tt, copy(zz)) };
    assert(f3.kind == scientific::F_SHAPE);
    // A start that holds a value that is not finite is a fault.
    let y4, fy4 = scientific::complex_from_values([0.0 / 0.0], [0.0]);
    assert(fy4.kind == scientific::F_NONE);
    let t4, z4, f4 = scientific::complex_ode(0.0, 1.0, copy(y4), 1.0e-10, 1.0e-12, "RK45", 1.0e308, []) { tt, zz => wave_f(tt, copy(zz)) };
    assert(f4.kind == scientific::F_VALUE);
    return;
}

// standard_deviation gives the deviation of the image over the
// regions of the labels.
fn test_integrate_std() {
    let img = common::arr_of([3, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let lab = common::arr_of([3, 3], [1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 2.0, 0.0]);
    let r1, f1 = scientific::standard_deviation(img, scientific::empty_arr());
    assert(f1.kind == scientific::F_NONE && common::close(r1.data[0], integrate_pins::STD_ALL[0], 1.0e-6));
    let r2, f2 = scientific::standard_deviation(img, lab, [1, 2]);
    assert(f2.kind == scientific::F_NONE);
    assert(common::close(r2.data[0], integrate_pins::STD_LAB[0], 1.0e-6) && common::close(r2.data[1], integrate_pins::STD_LAB[1], 1.0e-6));
    // Labels of another shape are a fault.
    let lab2 = common::arr_of([9], [1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 2.0, 0.0]);
    let r3, f3 = scientific::standard_deviation(img, lab2, [1]);
    assert(f3.kind == scientific::F_SHAPE);
    // A value of the labels that is not whole is a fault.
    let lab3 = common::arr_of([3, 3], [1.0, 1.0, 0.0, 2.0, 0.5, 1.0, 2.0, 2.0, 0.0]);
    let r4, f4 = scientific::standard_deviation(img, lab3, [1]);
    assert(f4.kind == scientific::F_VALUE);
    return;
}
