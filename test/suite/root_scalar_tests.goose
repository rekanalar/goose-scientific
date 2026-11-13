// The tests of the group root_scalar. Run them with SCIENTIFIC_SUITES=root_scalar.
namespace root_scalar_tests;

import scientific from "../..";
import os;
import common;

// root_scalar picks the bracket method by name.
fn test_root_scalar() {
    let r1, f1 = scientific::root_scalar(0.0, 2.0) { x => x * x - 2.0 };
    assert(f1.kind == scientific::F_NONE && common::near(r1, 1.4142135623730951));
    let r2, f2 = scientific::root_scalar(0.0, 2.0, "brenth") { x => x * x - 2.0 };
    assert(f2.kind == scientific::F_NONE && common::near(r2, 1.4142135623730951));
    let r3, f3 = scientific::root_scalar(0.0, 2.0, "bisect") { x => x * x - 2.0 };
    assert(f3.kind == scientific::F_NONE && common::near(r3, 1.4142135623730951));
    let r4, f4 = scientific::root_scalar(0.0, 2.0, "ridder") { x => x * x - 2.0 };
    assert(f4.kind == scientific::F_NONE && common::near(r4, 1.4142135623730951));
    let b1, g1 = scientific::root_scalar(0.0, 2.0, "secant") { x => x * x - 2.0 };
    assert(g1.kind == scientific::F_VALUE);
    let b2, g2 = scientific::root_scalar(3.0, 4.0) { x => x * x - 2.0 };
    assert(g2.kind != scientific::F_NONE);
    return;
}
