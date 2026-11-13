// The tests of the group sparse_bsr. Run them with SCIENTIFIC_SUITES=sparse_bsr.
namespace sparse_bsr_2_tests;

import scientific from "../..";
import os;
import common;
import sparse_bsr_tests;

fn test_sparse_bsr_pins_zero() {
    let c, f0 = scientific::csr_from_triplets(2, 4, [0.0, 1.0], [0.0, 3.0], [0.0, 2.0]);
    assert(f0.kind == scientific::F_NONE);
    let s, f1 = scientific::bsr_from_csr(c, 2, 2);
    assert(f1.kind == scientific::F_NONE);
    let sp, si, sd, sf = scientific::bsr_blocks(s);
    assert(sf.kind == scientific::F_NONE && common::all_near(sp, [0.0, 2.0]) && common::all_near(si, [0.0, 1.0]) && common::all_near(sd, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]));
    assert(scientific::bsr_nnz(s) == 8 && s.br == 2 && s.bc == 2);
    let k, f2 = scientific::bsr_to_csr(s);
    assert(f2.kind == scientific::F_NONE && k.nnz == 8);
    return;
}

fn test_sparse_bsr_pins() {
    sparse_bsr_tests::test_sparse_bsr_pins0();
    sparse_bsr_tests::test_sparse_bsr_pins1();
    sparse_bsr_tests::test_sparse_bsr_pins2();
    sparse_bsr_tests::test_sparse_bsr_pins3();
    sparse_bsr_tests::test_sparse_bsr_pins4();
    sparse_bsr_tests::test_sparse_bsr_pins5();
    sparse_bsr_tests::test_sparse_bsr_pins6();
    sparse_bsr_tests::test_sparse_bsr_pins7();
    sparse_bsr_tests::test_sparse_bsr_pins8();
    sparse_bsr_tests::test_sparse_bsr_pins9();
    sparse_bsr_tests::test_sparse_bsr_pins10();
    sparse_bsr_tests::test_sparse_bsr_pins11();
    sparse_bsr_tests::test_sparse_bsr_pins12();
    sparse_bsr_tests::test_sparse_bsr_pins13();
    sparse_bsr_tests::test_sparse_bsr_pins14();
    sparse_bsr_tests::test_sparse_bsr_pins15();
    sparse_bsr_2_tests::test_sparse_bsr_pins_zero();
    return;
}

// The block matrices refuse a bad block size, a bad size, a bad value and a bad
// operand.
fn test_sparse_bsr_hostile() {
    let a = common::mx(4, 4, [1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 6.0]);
    let s, f0 = scientific::bsr_from_dense(a, 2, 2);
    assert(f0.kind == scientific::F_NONE);
    // The block size must be at least one and divide the shape.
    let b1, g1 = scientific::bsr_from_dense(a, 0, 2);
    assert(g1.kind == scientific::F_RANGE);
    let b2, g2 = scientific::bsr_from_dense(a, 2, -1);
    assert(g2.kind == scientific::F_RANGE);
    let b3, g3 = scientific::bsr_from_dense(a, 3, 2);
    assert(g3.kind == scientific::F_RANGE);
    let b4, g4 = scientific::bsr_from_dense(a, 2, 3);
    assert(g4.kind == scientific::F_RANGE);
    // A side above the limit.
    let e0: f64[>..] = [];
    let b5, g5 = scientific::bsr_from_entries(1073741825, 4, 1, 1, e0, e0, e0);
    assert(g5.kind == scientific::F_RANGE);
    let b6, g6 = scientific::bsr_from_entries(0, 4, 1, 1, e0, e0, e0);
    assert(g6.kind == scientific::F_RANGE);
    // A count of stored values above the limit: one block holds 2^30 values.
    let b7, g7 = scientific::bsr_from_entries(32768, 32768, 32768, 32768, [0.0], [0.0], [1.0]);
    assert(g7.kind == scientific::F_LIMIT);
    // A dense result above the limit.
    let b8, g8 = scientific::bsr_from_entries(20000, 20000, 1, 1, [0.0], [0.0], [1.0]);
    assert(g8.kind == scientific::F_NONE);
    let d8, h8 = scientific::bsr_to_dense(b8);
    assert(h8.kind == scientific::F_LIMIT);
    // An entry outside the matrix, and a fractional place.
    let b9, g9 = scientific::bsr_from_entries(4, 4, 2, 2, [4.0], [0.0], [1.0]);
    assert(g9.kind == scientific::F_RANGE);
    let b10, g10 = scientific::bsr_from_entries(4, 4, 2, 2, [0.5], [0.0], [1.0]);
    assert(g10.kind == scientific::F_RANGE);
    let b11, g11 = scientific::bsr_from_entries(4, 4, 2, 2, [0.0, 1.0], [0.0], [1.0]);
    assert(g11.kind == scientific::F_SHAPE);
    // A value that is not finite, and a factor that is not finite.
    let nan = 0.0 / 0.0;
    let b12, g12 = scientific::bsr_from_entries(4, 4, 2, 2, [0.0], [0.0], [nan]);
    assert(g12.kind == scientific::F_VALUE);
    let b13, g13 = scientific::bsr_scale(s, nan);
    assert(g13.kind == scientific::F_VALUE);
    let b14, g14 = scientific::bsr_scale(s, 1.0 / 0.0);
    assert(g14.kind == scientific::F_VALUE);
    // A finite factor that makes the product not finite is a fault.
    let big, g19 = scientific::bsr_from_entries(2, 2, 2, 2, [0.0], [0.0], [1.0e308]);
    assert(g19.kind == scientific::F_NONE);
    let b19, h19 = scientific::bsr_scale(big, 10.0);
    assert(h19.kind == scientific::F_VALUE);
    // The sum of two duplicates that overflows is a fault, not an infinity.
    let b15, g15 = scientific::bsr_from_entries(2, 2, 2, 2, [0.0, 0.0], [0.0, 0.0], [1.0e308, 1.0e308]);
    assert(g15.kind == scientific::F_VALUE);
    // A place outside the matrix, and a vector that does not fit.
    let x1, g16 = scientific::bsr_get(s, 4, 0);
    assert(g16.kind == scientific::F_RANGE);
    let x2, g17 = scientific::bsr_get(s, 0, -1);
    assert(g17.kind == scientific::F_RANGE);
    let y, g18 = scientific::bsr_matvec(s, [1.0, 2.0, 3.0]);
    assert(g18.kind == scientific::F_SHAPE);
    // A dense operand that is not a matrix, and one that holds a NaN.
    let v, fv = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    assert(fv.kind == scientific::F_NONE);
    let b19, g19 = scientific::bsr_from_dense(v, 1, 1);
    assert(g19.kind == scientific::F_SHAPE);
    let w = common::mx(2, 2, [1.0, nan, 0.0, 1.0]);
    let b20, g20 = scientific::bsr_from_dense(w, 1, 1);
    assert(g20.kind == scientific::F_VALUE);
    // Two entries of one place add up.
    let b21, g21 = scientific::bsr_from_entries(2, 2, 2, 2, [0.0, 0.0], [1.0, 1.0], [1.0, 2.0]);
    assert(g21.kind == scientific::F_NONE && scientific::bsr_nnz(b21) == 4);
    let u, g22 = scientific::bsr_get(b21, 0, 1);
    assert(g22.kind == scientific::F_NONE && u == 3.0);
    // The coordinates build a block matrix with the same checks.
    let c0, g23 = scientific::coo_from_triplets(4, 4, [0.0, 3.0], [0.0, 3.0], [1.0, 2.0]);
    assert(g23.kind == scientific::F_NONE);
    let b22, g24 = scientific::bsr_from_coo(c0, 0, 2);
    assert(g24.kind == scientific::F_RANGE);
    let b23, g25 = scientific::bsr_from_coo(c0, 2, 3);
    assert(g25.kind == scientific::F_RANGE);
    let b24, g26 = scientific::bsr_from_coo(c0, 2, 2);
    assert(g26.kind == scientific::F_NONE && scientific::bsr_nnz(b24) == 8);
    // The coordinates of a block matrix hold every value of every block.
    let c1, g27 = scientific::bsr_to_coo(b24);
    assert(g27.kind == scientific::F_NONE && c1.nnz == 8);
    let d1, g28 = scientific::coo_to_dense(c1);
    assert(g28.kind == scientific::F_NONE && common::all_near(d1, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]));
    return;
}
