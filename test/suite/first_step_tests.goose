// The tests of the group first_step. Run them with SCIENTIFIC_SUITES=first_step.
namespace first_step_tests;

import scientific from "../..";
import os;
import common;

// The argument first_step sets the first step, as SciPy does.
fn test_first_step() {
    let t1, y1, f1 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "RK45", 1.0e308, [], 0.01) { t, v => [v[1], -v[0]] };
    assert(f1.kind == scientific::F_NONE && t1.data.len == 9);
    assert(common::mathshim_abs(t1.data[1] - 0.01) < 1.0e-15 && common::mathshim_abs(t1.data[3] - 0.7582343760009075) < 1.0e-9);
    assert(common::mathshim_abs(y1.data[16] - 0.2847042338359484) < 1.0e-9);
    let t2, y2, f2 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "DOP853", 1.0e308, [], 0.05) { t, v => [v[1], -v[0]] };
    assert(f2.kind == scientific::F_NONE && t2.data.len == 5);
    assert(common::mathshim_abs(t2.data[3] - 3.0469539039081424) < 1.0e-8 && common::mathshim_abs(y2.data[8] - 0.28341575655185514) < 1.0e-8);
    // A first step below zero or beyond the span is a fault.
    let t3, y3, f3 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "RK45", 1.0e308, [], -1.0) { t, v => [v[1], -v[0]] };
    assert(f3.kind == scientific::F_VALUE);
    let t4, y4, f4 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "RK45", 1.0e308, [], 6.0) { t, v => [v[1], -v[0]] };
    assert(f4.kind == scientific::F_VALUE);
}
