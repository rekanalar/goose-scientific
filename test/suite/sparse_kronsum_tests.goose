// The tests of the group sparse_kronsum. Run them with SCIENTIFIC_SUITES=sparse_kronsum.
namespace sparse_kronsum_tests;

import scientific from "../..";
import os;
import common;
import sparse_tests;

fn test_sparse_kronsum_pins() {
    sparse_tests::test_sparse_kronsum_pins0();
    sparse_tests::test_sparse_kronsum_pins1();
    sparse_tests::test_sparse_kronsum_pins2();
    sparse_tests::test_sparse_kronsum_pins3();
    sparse_tests::test_sparse_kronsum_pins4();
    sparse_tests::test_sparse_kronsum_pins5();
    sparse_tests::test_sparse_kronsum_pins6();
    sparse_tests::test_sparse_kronsum_pins7();
    sparse_tests::test_sparse_kronsum_pins8();
    sparse_tests::test_sparse_kronsum_pins9();
    sparse_tests::test_sparse_kronsum_pins10();
    sparse_tests::test_sparse_kronsum_pins11();
    sparse_tests::test_sparse_kronsum_pins12();
    sparse_tests::test_sparse_kronsum_pins_zero();
    return;
}

// csr_kronsum refuses an operand that is not square, a side that is too
// large and a count of stored values that is too large.
fn test_sparse_kronsum_hostile() {
    let sq, f0 = scientific::csr_from_dense(common::mx(2, 2, [1.0, 2.0, 0.0, 3.0]));
    let wide, f1 = scientific::csr_from_dense(common::mx(2, 3, [1.0, 0.0, 2.0, 0.0, 0.0, 3.0]));
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);
    let k1, g1 = scientific::csr_kronsum(wide, sq);
    assert(g1.kind == scientific::F_SHAPE);
    let k2, g2 = scientific::csr_kronsum(sq, wide);
    assert(g2.kind == scientific::F_SHAPE);
    let big, f2 = scientific::csr_eye(40000);
    assert(f2.kind == scientific::F_NONE);
    let k3, g3 = scientific::csr_kronsum(big, big);
    assert(g3.kind == scientific::F_RANGE);
    let mid, f3 = scientific::csr_eye(12000);
    assert(f3.kind == scientific::F_NONE);
    let k4, g4 = scientific::csr_kronsum(mid, mid);
    assert(g4.kind == scientific::F_LIMIT);
    return;
}
