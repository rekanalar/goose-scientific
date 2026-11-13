// The tests of the group odeint. Run them with SCIENTIFIC_SUITES=odeint.
namespace odeint_tests;

import scientific from "../..";
import os;
import common;

// odeint is pinned against SciPy at tight tolerances.
fn test_odeint() {
    let ts = [0.0, 0.5, 1.0, 2.0, 3.5];
    let r1, f1 = scientific::odeint([1.0, 0.0], ts, 1.0e-11, 1.0e-11) { y, t => [y[1], -y[0] - 0.2 * y[1]] };
    assert(f1.kind == scientific::F_NONE && r1.shape[0] == 5 && r1.shape[1] == 2);
    assert(common::all_loose(r1, [1.0, 0.0, 0.8815464028676221, -0.45623696587095147, 0.5689718911319457, -0.7627576785286944, -0.2580702634045758, -0.7516155021919233, -0.6878213068949068, 0.2367646744259087]));
    let r2, f2 = scientific::odeint([0.5], ts, 1.0e-11, 1.0e-11) { y, t => [-2.0 * y[0] + common::mathshim_sin(t)] };
    assert(f2.kind == scientific::F_NONE);
    assert(common::all_loose(r2, [0.5, 0.27376931184737185, 0.3232626309846345, 0.4597692852088899, 0.047616363747107364]));
    let r3, f3 = scientific::odeint([1.0], [2.0, 1.0, 0.0], 1.0e-11, 1.0e-11) { y, t => [-y[0]] };
    assert(f3.kind == scientific::F_NONE && common::all_loose(r3, [1.0, 2.718281828459045, 7.38905609893065]));
    let b1, g1 = scientific::odeint([1.0], [0.0], 1.0e-8, 1.0e-8) { y, t => [-y[0]] };
    assert(g1.kind == scientific::F_SHAPE);
    let b2, g2 = scientific::odeint([1.0], [0.0, 1.0, 0.5], 1.0e-8, 1.0e-8) { y, t => [-y[0]] };
    assert(g2.kind == scientific::F_VALUE);
    return;
}
