// The tests of the group ivp_t. Run them with SCIENTIFIC_SUITES=ivp_t.
namespace ivp_t_tests;

import scientific from "../..";
import os;
import common;

fn test_ivp_t_eval() {
    let e1 = [1.0, 0.0, 0.9553364803050641, -0.29552023491137414, 0.540302079728339, -0.8414711243630892, 0.020794742986437575, -0.9997835787518795, -0.9899921548382596, -0.1411198096921278];
    let t1, y1, f1 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1.0e-6, 1.0e-9, "RK45", 1.0e308, [0.0, 0.3, 1.0, 1.55, 3.0]) { t, v => [v[1], -v[0]] };
    assert(f1.kind == scientific::F_NONE && t1.data.len == 5 && y1.data.len == 10);
    var i: i64 = 0;
    while i < 10 { assert(common::mathshim_abs(y1.data[i] - e1[i]) < 1.0e-8); i += 1; }
    let e2 = [0.8775627709376573, -0.479416876047604, 0.07071226557404034, -0.9973594812519597, -0.8009810027734747, -0.5983202949277295];
    let t2, y2, f2 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1.0e-4, 1.0e-6, "RK23", 1.0e308, [0.5, 1.5, 2.5]) { t, v => [v[1], -v[0]] };
    assert(f2.kind == scientific::F_NONE && y2.data.len == 6);
    i = 0;
    while i < 6 { assert(common::mathshim_abs(y2.data[i] - e2[i]) < 1.0e-8); i += 1; }
    let e3 = [1.0, 0.0, 0.540302079728339, 0.841471124363089, -0.66627600604739, 0.7457049606986244, -0.9899921548382591, 0.14111980969212792];
    let t3, y3, f3 = scientific::solve_ivp(3.0, 0.0, [1.0, 0.0], 1.0e-6, 1.0e-9, "RK45", 1.0e308, [3.0, 2.0, 0.7, 0.0]) { t, v => [v[1], -v[0]] };
    assert(f3.kind == scientific::F_NONE && y3.data.len == 8);
    i = 0;
    while i < 8 { assert(common::mathshim_abs(y3.data[i] - e3[i]) < 1.0e-8); i += 1; }
    let t4, y4, f4 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1.0e-6, 1.0e-9, "RK45", 1.0e308, [4.0]) { t, v => [v[1], -v[0]] };
    assert(f4.kind == scientific::F_VALUE);
    let t5, y5, f5 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1.0e-6, 1.0e-9, "RK45", 1.0e308, [2.0, 1.0]) { t, v => [v[1], -v[0]] };
    assert(f5.kind == scientific::F_VALUE);
}
