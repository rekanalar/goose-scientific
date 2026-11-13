// The tests of the group residue_complex. Run them with SCIENTIFIC_SUITES=residue_complex.
namespace residue_complex_tests;

import scientific from "../..";
import os;
import common;

// residue with complex coefficients, pinned against SciPy.
fn test_residue_complex() {
    // Distinct poles.
    let r1, i1, p1, q1, k1, l1, f1 = scientific::residue_complex([1.0, 2.0], [1.0, 0.0], [1.0, 2.0, 3.0], [0.0, 1.0, -1.0]);
    assert(f1.kind == scientific::F_NONE);
    assert(common::all_near(r1, [0.24957156983195342, 0.7504284301680465]) && common::all_near(i1, [-0.05766500760348732, 1.0576650076034873]));
    assert(common::all_near(p1, [-0.3833969491244492, -1.6166030508755513]) && common::all_near(q1, [1.1217889265712222, -2.121788926571223]));
    assert(k1.data.len == 0 && l1.data.len == 0);
    // A double pole.
    let r2, i2, p2, q2, k2, l2, f2 = scientific::residue_complex([1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [1.0, 1.0, -9.0, -9.0], [0.0, -4.0, -8.0, 12.0]);
    assert(f2.kind == scientific::F_NONE);
    assert(common::all_close(r2, [0.88, -0.2, 0.12], 1.0e-9) && common::all_close(i2, [0.66, 1.6, -0.66], 1.0e-9));
    assert(common::all_close(p2, [1.0, 1.0, -3.0], 1.0e-9) && common::all_close(q2, [2.0, 2.0, 0.0], 1.0e-9));
    // Direct terms.
    let r3, i3, p3, q3, k3, l3, f3 = scientific::residue_complex([0.0, 2.0, 3.0, 1.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [0.0, -1.0, 0.0]);
    assert(f3.kind == scientific::F_NONE);
    assert(common::all_near(r3, [0.18393716584651643, 2.816062834153484]) && common::all_near(i3, [-0.7151455433308532, 0.7151455433308532]));
    assert(common::all_near(p3, [-0.32456794362370617, -0.6754320563762941]) && common::all_near(q3, [-0.9250531240639468, 1.925053124063947]));
    assert(common::all_near(k3, [0.0, 1.0]) && common::all_near(l3, [1.0, -1.0]));
    // Real data in the complex call.
    let r4, i4, p4, q4, k4, l4, f4 = scientific::residue_complex([1.0, 2.0], [0.0, 0.0], [1.0, 3.0, 2.0], [0.0, 0.0, 0.0]);
    assert(f4.kind == scientific::F_NONE && common::all_near(r4, [1.0, 0.0]) && common::all_near(p4, [-1.0, -2.0]));
    // A zero numerator.
    let r5, i5, p5, q5, k5, l5, f5 = scientific::residue_complex([0.0], [0.0], [1.0, 2.0, 0.5], [0.0, 1.0, 0.0]);
    assert(f5.kind == scientific::F_NONE && common::all_near(r5, [0.0, 0.0]) && common::all_near(p5, [-0.19975740977987952, -1.8002425902201198]) && common::all_near(q5, [0.12481053384382654, -1.1248105338438266]));
    // Faults.
    let e1, e2, e3, e4, e5, e6, g1 = scientific::residue_complex([1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]);
    assert(g1.kind == scientific::F_SHAPE);
    let k1x, k2x, k3x, k4x, k5x, k6x, g2 = scientific::residue_complex([1.0], [0.0], [0.0, 0.0], [0.0, 0.0]);
    assert(g2.kind == scientific::F_VALUE);
    let h1, h2, h3, h4, h5, h6, g3 = scientific::residue_complex([1.0], [0.0], [1.0, 1.0], [0.0, 0.0], 0.001, "other");
    assert(g3.kind == scientific::F_VALUE);
    let j1, j2, j3, j4, j5, j6, g4 = scientific::residue_complex([1.0], [1.0 / 0.0], [1.0, 1.0], [0.0, 0.0]);
    assert(g4.kind == scientific::F_VALUE);
}
