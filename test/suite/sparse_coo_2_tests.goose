// The tests of the group sparse_coo. Run them with SCIENTIFIC_SUITES=sparse_coo.
namespace sparse_coo_2_tests;

import scientific from "../..";
import os;
import common_3;
import sparse_coo_tests;

fn test_sparse_coo_pins15() {
    let qc30, qg30 = scientific::coo_from_dense(common_3::ni_nd([1, 5], [-0.17, 0.0, 0.0, -2.094, 0.0]));
    assert(qg30.kind == scientific::F_NONE && qc30.nnz == 2);
    let qd30, qh30 = scientific::coo_to_dense(qc30);
    assert(qh30.kind == scientific::F_NONE);
    assert(qd30.shape == [1, 5] && common_3::ms_same(qd30, [-0.17, 0.0, 0.0, -2.094, 0.0]));
    let qc31, qg31 = scientific::coo_from_dense(common_3::ni_nd([5, 2], [0.0, 0.0, 0.0, -1.444, 0.679, -2.221, 0.0, 0.0, 0.0, 0.0]));
    assert(qg31.kind == scientific::F_NONE && qc31.nnz == 3);
    let qd31, qh31 = scientific::coo_to_dense(qc31);
    assert(qh31.kind == scientific::F_NONE);
    assert(qd31.shape == [5, 2] && common_3::ms_same(qd31, [0.0, 0.0, 0.0, -1.444, 0.679, -2.221, 0.0, 0.0, 0.0, 0.0]));
    let qc32, qg32 = scientific::coo_from_dense(common_3::ni_nd([2, 1], [0.0, 0.0]));
    assert(qg32.kind == scientific::F_NONE && qc32.nnz == 0);
    let qd32, qh32 = scientific::coo_to_dense(qc32);
    assert(qh32.kind == scientific::F_NONE);
    assert(qd32.shape == [2, 1] && common_3::ms_same(qd32, [0.0, 0.0]));
    let qc33, qg33 = scientific::coo_from_dense(common_3::ni_nd([1, 5], [0.0, 0.0, 2.962, 0.0, -1.145]));
    assert(qg33.kind == scientific::F_NONE && qc33.nnz == 2);
    let qd33, qh33 = scientific::coo_to_dense(qc33);
    assert(qh33.kind == scientific::F_NONE);
    assert(qd33.shape == [1, 5] && common_3::ms_same(qd33, [0.0, 0.0, 2.962, 0.0, -1.145]));
    let qc34, qg34 = scientific::coo_from_dense(common_3::ni_nd([1, 3], [0.784, 0.0, 0.113]));
    assert(qg34.kind == scientific::F_NONE && qc34.nnz == 2);
    let qd34, qh34 = scientific::coo_to_dense(qc34);
    assert(qh34.kind == scientific::F_NONE);
    assert(qd34.shape == [1, 3] && common_3::ms_same(qd34, [0.784, 0.0, 0.113]));
    let qc35, qg35 = scientific::coo_from_dense(common_3::ni_nd([2, 2], [2.563, -1.711, 0.0, 0.142]));
    assert(qg35.kind == scientific::F_NONE && qc35.nnz == 3);
    let qd35, qh35 = scientific::coo_to_dense(qc35);
    assert(qh35.kind == scientific::F_NONE);
    assert(qd35.shape == [2, 2] && common_3::ms_same(qd35, [2.563, -1.711, 0.0, 0.142]));
    let qc36, qg36 = scientific::coo_from_dense(common_3::ni_nd([4, 5], [2.7, 0.0, 2.56, 0.707, 0.0, 1.57, -1.344, 0.0, -0.766, 0.0, 1.031, 0.0, 0.0, -1.568, -1.802, 0.0, 2.565, -2.856, 0.0, 0.0]));
    assert(qg36.kind == scientific::F_NONE && qc36.nnz == 11);
    let qd36, qh36 = scientific::coo_to_dense(qc36);
    assert(qh36.kind == scientific::F_NONE);
    assert(qd36.shape == [4, 5] && common_3::ms_same(qd36, [2.7, 0.0, 2.56, 0.707, 0.0, 1.57, -1.344, 0.0, -0.766, 0.0, 1.031, 0.0, 0.0, -1.568, -1.802, 0.0, 2.565, -2.856, 0.0, 0.0]));
    let qc37, qg37 = scientific::coo_from_dense(common_3::ni_nd([1, 2], [-1.157, -2.298]));
    assert(qg37.kind == scientific::F_NONE && qc37.nnz == 2);
    let qd37, qh37 = scientific::coo_to_dense(qc37);
    assert(qh37.kind == scientific::F_NONE);
    assert(qd37.shape == [1, 2] && common_3::ms_same(qd37, [-1.157, -2.298]));
}

fn test_sparse_coo_pins() {
    sparse_coo_tests::test_sparse_coo_pins0();
    sparse_coo_tests::test_sparse_coo_pins1();
    sparse_coo_tests::test_sparse_coo_pins2();
    sparse_coo_tests::test_sparse_coo_pins3();
    sparse_coo_tests::test_sparse_coo_pins4();
    sparse_coo_tests::test_sparse_coo_pins5();
    sparse_coo_tests::test_sparse_coo_pins6();
    sparse_coo_tests::test_sparse_coo_pins7();
    sparse_coo_tests::test_sparse_coo_pins8();
    sparse_coo_tests::test_sparse_coo_pins9();
    sparse_coo_tests::test_sparse_coo_pins10();
    sparse_coo_tests::test_sparse_coo_pins11();
    sparse_coo_tests::test_sparse_coo_pins12();
    sparse_coo_tests::test_sparse_coo_pins13();
    sparse_coo_tests::test_sparse_coo_pins14();
    sparse_coo_2_tests::test_sparse_coo_pins15();
}

// test_sparse_coo_hostile checks the faults of the coordinates.
fn test_sparse_coo_hostile() {
    // A side below one.
    let c1, g1 = scientific::coo_from_triplets(0, 2, [], [], []);
    assert(g1.kind == scientific::F_RANGE);
    // An entry outside the matrix.
    let c2, g2 = scientific::coo_from_triplets(2, 2, [0.0, 5.0], [0.0, 1.0], [1.0, 2.0]);
    assert(g2.kind == scientific::F_RANGE);
    // A negative entry and a fractional entry.
    let c3, g3 = scientific::coo_from_triplets(2, 2, [-1.0], [0.0], [1.0]);
    assert(g3.kind == scientific::F_RANGE);
    let c4, g4 = scientific::coo_from_triplets(2, 2, [0.5], [0.0], [1.0]);
    assert(g4.kind == scientific::F_RANGE);
    // An entry that is not a number.
    let c5, g5 = scientific::coo_from_triplets(2, 2, [0.0, 0.0 / 0.0], [0.0, 0.0], [1.0, 2.0]);
    assert(g5.kind == scientific::F_RANGE);
    // The rows, the columns and the values differ in count.
    let c6, g6 = scientific::coo_from_triplets(2, 2, [0.0], [0.0, 1.0], [1.0]);
    assert(g6.kind == scientific::F_SHAPE);
    // A matrix of one entry.
    let c7, g7 = scientific::coo_from_triplets(2, 2, [0.0], [0.0], [1.0]);
    assert(g7.kind == scientific::F_NONE);
    // A place outside the matrix.
    let x7, h7 = scientific::coo_get(c7, 2, 0);
    assert(h7.kind == scientific::F_RANGE);
    // A vector of the wrong length.
    let x8, h8 = scientific::coo_matvec(c7, [1.0, 2.0, 3.0]);
    assert(h8.kind == scientific::F_SHAPE);
    // A matrix that is not a matrix.
    let c9, g9 = scientific::coo_from_dense(common_3::ni_nd([3], [1.0, 2.0, 3.0]));
    assert(g9.kind == scientific::F_SHAPE);
    // The duplicates stay and count.
    let c10, g10 = scientific::coo_from_triplets(2, 2, [0.0, 0.0], [0.0, 0.0], [1.0, 2.0]);
    assert(g10.kind == scientific::F_NONE && c10.nnz == 2);
    let x10, h10 = scientific::coo_get(c10, 0, 0);
    assert(h10.kind == scientific::F_NONE && x10 == 3.0);
}
