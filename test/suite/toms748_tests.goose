// The tests of the group toms748. Run them with SCIENTIFIC_SUITES=toms748.
namespace toms748_tests;

import scientific from "../..";
import os;
import common;

// test_toms748 pins the method of Alefeld, Potra and Shi against
// SciPy.
fn test_toms748() {
    let t1, t1f = scientific::toms748(0.0, 2.0) { x => x * x - 2.0 };
    assert(t1f.kind == scientific::F_NONE && common::close(t1, 1.414213562373095, 1e-14));
    let t2, t2f = scientific::toms748(-3.0, 0.0) { x => x * x * x - 2.0 * x + 2.0 };
    assert(t2f.kind == scientific::F_NONE && common::close(t2, -1.7692923542386314, 1e-14));
    let t3, t3f = scientific::toms748(0.0, 1.0) { x => cos(x) - x };
    assert(t3f.kind == scientific::F_NONE && common::close(t3, 0.7390851332151607, 1e-14));
    let t4, t4f = scientific::toms748(-2.0, 2.0) { x => exp(x) - 2.0 };
    assert(t4f.kind == scientific::F_NONE && common::close(t4, 0.6931471805599453, 1e-14));
    let t5, t5f = scientific::toms748(0.0, 2.0) { x => x * x * x * x * x - 2.0 * x * x * x + x - 1.0 };
    assert(t5f.kind == scientific::F_NONE && common::close(t5, 1.3625985776649348, 1e-14));
    // The count of steps of Newton type changes the path, not the root.
    let t6, t6f = scientific::toms748(0.0, 2.0, 2.0e-12, 8.881784197001252e-16, 100, 1) { x => x * x - 2.0 };
    assert(t6f.kind == scientific::F_NONE && common::close(t6, 1.414213562373095, 1e-14));
    let t7, t7f = scientific::toms748(0.0, 2.0, 2.0e-12, 8.881784197001252e-16, 100, 3) { x => x * x - 2.0 };
    assert(t7f.kind == scientific::F_NONE && common::close(t7, 1.414213562373095, 1e-14));
    // The faults.
    let b1, b1f = scientific::toms748(0.0, 2.0, 0.0) { x => x * x - 2.0 };
    assert(b1f.kind != scientific::F_NONE);
    let b2, b2f = scientific::toms748(0.0, 2.0, 2.0e-12, -1.0) { x => x * x - 2.0 };
    assert(b2f.kind != scientific::F_NONE);
    let b3, b3f = scientific::toms748(0.0, 2.0, 2.0e-12, 8.881784197001252e-16, 0) { x => x * x - 2.0 };
    assert(b3f.kind != scientific::F_NONE);
    let b4, b4f = scientific::toms748(0.0, 2.0) { x => x * x + 1.0 };
    assert(b4f.kind != scientific::F_NONE);
    let b5, b5f = scientific::toms748(0.0, 2.0) { x => 0.0 / 0.0 };
    assert(b5f.kind != scientific::F_NONE);
}
