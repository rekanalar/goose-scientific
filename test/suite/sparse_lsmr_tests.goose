// The tests of the group sparse_lsmr. Run them with SCIENTIFIC_SUITES=sparse_lsmr.
namespace sparse_lsmr_tests;

import scientific from "../..";
import os;
import common_3;
import sparse_tests;

fn test_sparse_lsmr_pins() {
    sparse_tests::test_sparse_lsmr_pins0();
    sparse_tests::test_sparse_lsmr_pins1();
    sparse_tests::test_sparse_lsmr_pins2();
    sparse_tests::test_sparse_lsmr_pins3();
}

// test_sparse_lsmr_hostile checks the faults of lsmr.
fn test_sparse_lsmr_hostile() {
    let a, fa = scientific::csr_from_dense(common_3::ni_nd([3, 2], [1.0, 0.0, 0.0, 2.0, 1.0, 1.0]));
    assert(fa.kind == scientific::F_NONE);
    let r1, g1 = scientific::lsmr(a, [1.0, 2.0]);
    assert(g1.kind == scientific::F_SHAPE);
    let r2, g2 = scientific::lsmr(a, [1.0, 2.0, 3.0], -1.0);
    assert(g2.kind == scientific::F_VALUE);
    let r3, g3 = scientific::lsmr(a, [1.0, 2.0, 3.0], 0.0, 0.0 / 0.0);
    assert(g3.kind == scientific::F_VALUE);
    let r4, g4 = scientific::lsmr(a, [1.0, 1.0 / 0.0, 3.0]);
    assert(g4.kind == scientific::F_VALUE);
    let r5, g5 = scientific::lsmr(a, [1.0, 2.0, 3.0], 0.0, 1.0e-6, 1.0e-6, 0.0 / 0.0);
    assert(g5.kind == scientific::F_VALUE);
    // A right side of zeros gives the zero solution and the reason zero.
    let r6, g6 = scientific::lsmr(a, [0.0, 0.0, 0.0]);
    assert(g6.kind == scientific::F_NONE && r6.istop == 0 && r6.itn == 0 && common_3::ms_same(r6.x, [0.0, 0.0]));
    // A right side so small that its norm falls to zero is a fault.
    let r7, g7 = scientific::lsmr(a, [1.0e-170, 1.0e-170, 1.0e-170]);
    assert(g7.kind == scientific::F_VALUE);
    // A right side so large that its norm overflows is a fault.
    let r8, g8 = scientific::lsmr(a, [1.0e200, 2.0, 3.0]);
    assert(g8.kind == scientific::F_VALUE);
}
