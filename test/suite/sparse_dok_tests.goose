// The tests of the group sparse_dok. Run them with SCIENTIFIC_SUITES=sparse_dok.
namespace sparse_dok_tests;

import scientific from "../..";
import os;
import common_3;
import common_4;
import common_5;
import common_6;
import common_7;

fn test_sparse_dok_pins() {
    common_3::dok_pin_p1();
    common_3::dok_pin_p2();
    common_3::dok_pin_p3();
    common_3::dok_pin_p4();
    common_3::dok_pin_p5();
    common_3::dok_pin_p10_0();
    common_3::dok_pin_p10_1();
    common_3::dok_pin_p10_2();
    common_3::dok_pin_p10_3();
    common_3::dok_pin_p10_4();
    common_4::dok_pin_p10_5();
    common_4::dok_pin_p10_6();
    common_4::dok_pin_p10_edge_a();
    common_4::dok_pin_p10_edge_b();
    common_4::dok_pin_p10_long();
    common_4::dok_pin_p11a();
    common_4::dok_pin_p11b();
    common_4::dok_pin_r0();
    common_4::dok_pin_r1();
    common_4::dok_pin_r2();
    common_4::dok_pin_r3();
    common_4::dok_pin_r4();
    common_4::dok_pin_r5();
    common_4::dok_pin_r6();
    common_4::dok_pin_r7();
    common_4::dok_pin_r8();
    common_4::dok_pin_r9();
    common_4::dok_pin_r10();
    common_4::dok_pin_r11();
    common_4::dok_pin_r12();
    common_5::dok_pin_r13();
    common_5::dok_pin_r14();
    common_5::dok_pin_r15();
    common_5::dok_pin_r16();
    common_5::dok_pin_r17();
    common_5::dok_pin_r18();
    common_5::dok_pin_r19();
    common_5::dok_pin_r20();
    common_5::dok_pin_r21();
    common_5::dok_pin_r22();
    common_5::dok_pin_r23();
    common_5::dok_pin_r24();
    common_5::dok_pin_r25();
    common_5::dok_pin_r26();
    common_5::dok_pin_r27();
    common_5::dok_pin_r28();
    common_5::dok_pin_r29();
    common_5::dok_pin_r30();
    common_5::dok_pin_r31();
    common_6::dok_pin_r32();
    common_6::dok_pin_r33();
    common_6::dok_pin_r34();
    common_6::dok_pin_r35();
    common_6::dok_pin_r36();
    common_6::dok_pin_r37();
    common_6::dok_pin_r38();
    common_6::dok_pin_r39();
    common_6::dok_pin_r40();
    common_6::dok_pin_r41();
    common_6::dok_pin_r42();
    common_6::dok_pin_r43();
    common_6::dok_pin_r44();
    common_6::dok_pin_r45();
    common_6::dok_pin_r46();
    common_6::dok_pin_r47();
    common_7::dok_pin_r48();
    common_7::dok_pin_r49();
    common_7::dok_pin_r50();
    common_7::dok_pin_r51();
    common_7::dok_pin_r52();
    common_7::dok_pin_r53();
    common_7::dok_pin_r54();
    common_7::dok_pin_r55();
    common_7::dok_pin_r56();
    common_7::dok_pin_r57();
    common_7::dok_pin_r58();
    common_7::dok_pin_r59();
    common_7::dok_pin_conv();
}

// The hostile input of the sparse matrices of keys.
fn test_sparse_dok_hostile() {
    let nan = 0.0 / 0.0;
    let inf = 1.0e308 * 10.0;
    // A side of zero or above the limit is a fault.
    let z0, h0 = scientific::dok_new(0, 3);
    assert(h0.kind == scientific::F_RANGE);
    let z1, h1 = scientific::dok_new(3, scientific::MAX_SPARSE_SIDE + 1);
    assert(h1.kind == scientific::F_RANGE);
    let d0, h2 = scientific::dok_new(3, 4);
    assert(h2.kind == scientific::F_NONE);
    var d = d0;
    assert(scientific::dok_set(d, 1.0, 1.0, 5.0).kind == scientific::F_NONE);
    // A place that is not a whole place inside the matrix is a fault.
    assert(scientific::dok_set(d, nan, 0.0, 1.0).kind == scientific::F_RANGE);
    assert(scientific::dok_set(d, inf, 0.0, 1.0).kind == scientific::F_RANGE);
    assert(scientific::dok_set(d, 0.5, 0.0, 1.0).kind == scientific::F_RANGE);
    assert(scientific::dok_set(d, -1.0, 0.0, 1.0).kind == scientific::F_RANGE);
    assert(scientific::dok_set(d, 3.0, 0.0, 1.0).kind == scientific::F_RANGE);
    assert(scientific::dok_set(d, 0.0, 4.0, 1.0).kind == scientific::F_RANGE);
    assert(scientific::dok_set(d, 1.0e300, 0.0, 1.0).kind == scientific::F_RANGE);
    let g0, k0 = scientific::dok_get(d, -1.0, 0.0);
    assert(k0.kind == scientific::F_RANGE);
    let g1, k1 = scientific::dok_pop(d, 0.0, nan);
    assert(k1.kind == scientific::F_RANGE);
    // A value that is not finite is a fault.
    assert(scientific::dok_set(d, 0.0, 0.0, nan).kind == scientific::F_VALUE);
    assert(scientific::dok_set(d, 0.0, 0.0, inf).kind == scientific::F_VALUE);
    // A fault leaves the matrix as it was.
    assert(scientific::dok_nnz(d) == 1);
    assert(common_3::dok_is(d, [1.0], [1.0], [5.0]));
    // A sum that overflows is a fault.
    let ov, h3 = scientific::coo_from_triplets(1, 1, [0.0, 0.0], [0.0, 0.0], [1.0e308, 1.0e308]);
    assert(h3.kind == scientific::F_NONE);
    let ovd, h4 = scientific::dok_from_coo(ov);
    assert(h4.kind == scientific::F_VALUE);
    var bd, h5 = scientific::dok_new(1, 2);
    assert(scientific::dok_set(bd, 0.0, 0.0, 1.0e308).kind == scientific::F_NONE);
    assert(scientific::dok_set(bd, 0.0, 1.0, 1.0e308).kind == scientific::F_NONE);
    let s0, h6 = scientific::dok_sum(bd);
    assert(h6.kind == scientific::F_VALUE);
    // A factor or a product that is not finite is a fault.
    let sc0, h7 = scientific::dok_scale(bd, nan);
    assert(h7.kind == scientific::F_VALUE);
    let sc1, h8 = scientific::dok_scale(bd, 10.0);
    assert(h8.kind == scientific::F_VALUE);
    // A vector of a wrong length is a fault.
    let mv, h9 = scientific::dok_matvec(d, [1.0, 2.0]);
    assert(h9.kind == scientific::F_SHAPE);
    // A diagonal outside the matrix is a fault, and the matrix stays.
    assert(scientific::dok_setdiag(d, 4, [1.0]).kind == scientific::F_RANGE);
    assert(scientific::dok_setdiag(d, -3, [1.0]).kind == scientific::F_RANGE);
    assert(scientific::dok_setdiag(d, 9223372036854775807, [1.0]).kind == scientific::F_RANGE);
    assert(scientific::dok_setdiag(d, 0, [1.0, nan]).kind == scientific::F_VALUE);
    assert(common_3::dok_is(d, [1.0], [1.0], [5.0]));
    assert(scientific::dok_resize(d, 0, 2).kind == scientific::F_RANGE);
    assert(common_3::dok_is(d, [1.0], [1.0], [5.0]));
    // A dense result above the limit is a fault.
    let huge, h10 = scientific::dok_new(100000, 100000);
    assert(h10.kind == scientific::F_NONE);
    let hd, h11 = scientific::dok_to_dense(huge);
    assert(h11.kind == scientific::F_LIMIT);
    // A dense operand that is not a matrix is a fault.
    let line, h12 = scientific::from_values([1.0, 2.0]);
    let ld, h13 = scientific::dok_from_dense(line);
    assert(h13.kind == scientific::F_SHAPE);
    // A long series of sets and reads stays fast and exact.
    var many, h14 = scientific::dok_new(1000, 1000);
    var n: i64 = 0;
    while n < 100000 {
        assert(scientific::dok_set(many, (n % 1000) as f64, (n / 1000 * 7 % 1000) as f64, 1.0 + (n % 5) as f64).kind == scientific::F_NONE);
        n += 1;
    }
    assert(scientific::dok_nnz(many) == 100000);
    let mg, h15 = scientific::dok_get(many, 3.0, 0.0);
    assert(h15.kind == scientific::F_NONE && mg == 4.0);
    // Places that sit on a line of a power of two collide in a fixed hash.
    var line2, h16 = scientific::dok_new(1073741824, 1073741824);
    n = 0;
    while n < 100000 {
        assert(scientific::dok_set(line2, (n * 8192) as f64, (n * 8192) as f64, 1.0).kind == scientific::F_NONE);
        n += 1;
    }
    assert(scientific::dok_nnz(line2) == 100000);
    // A matrix built by hand with a full table is a fault, not a hang.
    var fbuf: f64[>..] = [];
    var fi: i64 = 0;
    while fi < 8 { fbuf.push(1.0); fi += 1; }
    fbuf.push(0.0); fbuf.push(0.0); fbuf.push(1.0);
    var forged = scientific::Dok { rows: 3, cols: 3, n: 1, cap: 8, salt: 0, buf: copy(fbuf) };
    assert(scientific::dok_set(forged, 2.0, 2.0, 7.0).kind == scientific::F_VALUE);
    // A stored zero stays: a sum of duplicates that gives zero, a scale by
    // zero and a compressed row with a stored zero.
    let zc, hz0 = scientific::coo_from_triplets(2, 2, [1.0, 1.0], [1.0, 1.0], [1.0, -1.0]);
    assert(hz0.kind == scientific::F_NONE);
    let zd, hz1 = scientific::dok_from_coo(zc);
    assert(hz1.kind == scientific::F_NONE && scientific::dok_nnz(zd) == 1);
    let zs, hz2 = scientific::dok_scale(zd, 0.0);
    assert(hz2.kind == scientific::F_NONE && scientific::dok_nnz(zs) == 1);
    let zr, hz4 = scientific::csr_from_triplets(2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 2.0]);
    assert(hz4.kind == scientific::F_NONE);
    let zrd, hz5 = scientific::dok_from_csr(zr);
    assert(hz5.kind == scientific::F_NONE && scientific::dok_nnz(zrd) == 2);
    // A set of zero on a missing place changes nothing.
    var zm, hz3 = scientific::dok_new(2, 2);
    assert(scientific::dok_set(zm, 0.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::dok_nnz(zm) == 0);
    // The count at the limit: a new place is a fault, an old place is not.
    var lbuf: f64[>..] = [];
    var li: i64 = 0;
    while li < 8 { lbuf.push(0.0); li += 1; }
    var lim = scientific::Dok { rows: 3, cols: 3, n: scientific::MAX_SPARSE_NNZ, cap: 8, salt: 7, buf: copy(lbuf) };
    assert(scientific::dok_set(lim, 1.0, 1.0, 1.0).kind == scientific::F_LIMIT);
    assert(lim.n == scientific::MAX_SPARSE_NNZ && lim.buf.len == 8);
    // The time of a series of sets grows in a straight line.
    var t0 = clock();
    var ta, ha = scientific::dok_new(1000, 1000);
    n = 0;
    while n < 50000 { scientific::dok_set(ta, (n % 1000) as f64, (n / 1000 * 7 % 1000) as f64, 1.0); n += 1; }
    var t1 = clock();
    var tb, hb = scientific::dok_new(1000, 1000);
    n = 0;
    while n < 100000 { scientific::dok_set(tb, (n % 1000) as f64, (n / 1000 * 7 % 1000) as f64, 1.0); n += 1; }
    var t2 = clock();
    assert(t2 - t1 < 4.0 * (t1 - t0) + 1.0);
}
