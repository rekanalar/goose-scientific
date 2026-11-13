// The tests of the group solve_ivp. Run them with SCIENTIFIC_SUITES=solve_ivp.
namespace solve_ivp_tests;

import scientific from "../..";
import os;
import common;

fn test_solve_ivp_jac() {
    // The Robertson problem. The count of steps and the last state match
    // SciPy with the same Jacobian.
    let t1, y1, f1 = scientific::solve_ivp_jac(0.0, 40.0, [1.0, 0.0, 0.0], 1.0e-6, 1.0e-9, "BDF", 1.0e308, [], 0.0) { tt, yy, mode => common::rob_jac(yy, mode, 0) };
    assert(f1.kind == scientific::F_NONE);
    assert(t1.data.len == 130);
    assert(common::mathshim_abs(y1.data[387] - 0.7158277520684803) < 1.0e-9);
    assert(common::mathshim_abs(y1.data[388] - 9.185561290303603e-06) < 1.0e-12);
    assert(common::mathshim_abs(y1.data[389] - 0.28416306237022904) < 1.0e-9);
    let t2, y2, f2 = scientific::solve_ivp_jac(0.0, 40.0, [1.0, 0.0, 0.0], 1.0e-6, 1.0e-9, "Radau", 1.0e308, [], 0.0) { tt, yy, mode => common::rob_jac(yy, mode, 0) };
    assert(f2.kind == scientific::F_NONE);
    assert(t2.data.len == 67);
    assert(common::mathshim_abs(y2.data[198] - 0.7158270747550006) < 1.0e-9);
    assert(common::mathshim_abs(y2.data[199] - 9.18553473413238e-06) < 1.0e-12);
    assert(common::mathshim_abs(y2.data[200] - 0.284163739710265) < 1.0e-9);
    // A Jacobian of the wrong size or with a value that is not finite is a fault.
    let t3, y3, f3 = scientific::solve_ivp_jac(0.0, 40.0, [1.0, 0.0, 0.0], 1.0e-6, 1.0e-9, "BDF", 1.0e308, [], 0.0) { tt, yy, mode => common::rob_jac(yy, mode, 1) };
    assert(f3.kind == scientific::F_SHAPE);
    let t4, y4, f4 = scientific::solve_ivp_jac(0.0, 40.0, [1.0, 0.0, 0.0], 1.0e-6, 1.0e-9, "Radau", 1.0e308, [], 0.0) { tt, yy, mode => common::rob_jac(yy, mode, 2) };
    assert(f4.kind == scientific::F_VALUE);
    // A method without a Jacobian ignores it.
    let t5, y5, f5 = scientific::solve_ivp_jac(0.0, 1.0, [1.0, 0.0, 0.0], 1.0e-6, 1.0e-9, "RK45", 1.0e308, [], 0.0) { tt, yy, mode => common::rob_jac(yy, mode, 0) };
    assert(f5.kind == scientific::F_NONE);
}
