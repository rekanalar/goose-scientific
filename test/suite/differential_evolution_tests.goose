// The tests of the group differential_evolution. Run them with SCIENTIFIC_SUITES=differential_evolution.
namespace differential_evolution_tests;

import scientific from "../..";
import os;
import common;

// differential_evolution finds the global minimum of two test functions.
fn test_differential_evolution() {
    // The sphere: the minimum is zero at the origin.
    let a, f1 = scientific::differential_evolution([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], 3, 2000, 15, 1e-8) { v => v[0] * v[0] + v[1] * v[1] + v[2] * v[2] };
    assert(f1.kind == scientific::F_NONE && a.data.len == 4 && a.data[3] < 1e-6);
    // The Rosenbrock function: the minimum is zero at one, one.
    let b, f2 = scientific::differential_evolution([-5.0, -5.0], [5.0, 5.0], 7, 3000, 15, 1e-10) { v => 100.0 * (v[1] - v[0] * v[0]) * (v[1] - v[0] * v[0]) + (1.0 - v[0]) * (1.0 - v[0]) };
    assert(f2.kind == scientific::F_NONE && b.data[2] < 1e-5);
    // The Rastrigin function has many local minima; the global one is zero at the origin.
    let c, f3 = scientific::differential_evolution([-5.12, -5.12], [5.12, 5.12], 11, 3000, 40, 1e-10) { v => 20.0 + v[0] * v[0] - 10.0 * common::mathshim_cos(6.283185307179586 * v[0]) + v[1] * v[1] - 10.0 * common::mathshim_cos(6.283185307179586 * v[1]) };
    assert(f3.kind == scientific::F_NONE && c.data[2] < 1e-4);
    // The same seed gives the same result.
    let d, f4 = scientific::differential_evolution([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], 3, 2000, 15, 1e-8) { v => v[0] * v[0] + v[1] * v[1] + v[2] * v[2] };
    assert(d.data[0] == a.data[0] && d.data[3] == a.data[3]);
    let e, f5 = scientific::differential_evolution([1.0], [0.0]) { v => v[0] };
    assert(f5.kind == scientific::F_VALUE);
    let g, f6 = scientific::differential_evolution([0.0, 1.0], [1.0]) { v => v[0] };
    assert(f6.kind == scientific::F_SHAPE);
    return;
}
