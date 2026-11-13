// The tests of the group sparse_lil. Run them with SCIENTIFIC_SUITES=sparse_lil.
namespace sparse_lil_tests;

import scientific from "../..";
import os;
import common_10;
import common_11;
import common_7;
import common_8;
import common_9;

fn test_sparse_lil_pins() {
    common_7::lil_pin_p1();
    common_7::lil_pin_p2();
    common_7::lil_pin_p3();
    common_7::lil_pin_p3z();
    common_7::lil_pin_p4();
    common_7::lil_pin_p5_0();
    common_7::lil_pin_p5_1();
    common_7::lil_pin_p5_2();
    common_8::lil_pin_p5_3();
    common_8::lil_pin_p5_4();
    common_8::lil_pin_p5_5();
    common_8::lil_pin_p5_6();
    common_8::lil_pin_p5_edge_a();
    common_8::lil_pin_p5_edge_b();
    common_8::lil_pin_p5_long();
    common_8::lil_pin_p6a();
    common_8::lil_pin_p6b();
    common_8::lil_pin_r0();
    common_8::lil_pin_r1();
    common_8::lil_pin_r2();
    common_8::lil_pin_r3();
    common_8::lil_pin_r4();
    common_8::lil_pin_r5();
    common_8::lil_pin_r6();
    common_8::lil_pin_r7();
    common_8::lil_pin_r8();
    common_8::lil_pin_r9();
    common_8::lil_pin_r10();
    common_8::lil_pin_r11();
    common_8::lil_pin_r12();
    common_8::lil_pin_r13();
    common_9::lil_pin_r14();
    common_9::lil_pin_r15();
    common_9::lil_pin_r16();
    common_9::lil_pin_r17();
    common_9::lil_pin_r18();
    common_9::lil_pin_r19();
    common_9::lil_pin_r20();
    common_9::lil_pin_r21();
    common_9::lil_pin_r22();
    common_9::lil_pin_r23();
    common_9::lil_pin_r24();
    common_9::lil_pin_r25();
    common_9::lil_pin_r26();
    common_9::lil_pin_r27();
    common_9::lil_pin_r28();
    common_9::lil_pin_r29();
    common_9::lil_pin_r30();
    common_9::lil_pin_r31();
    common_9::lil_pin_r32();
    common_9::lil_pin_r33();
    common_10::lil_pin_r34();
    common_10::lil_pin_r35();
    common_10::lil_pin_r36();
    common_10::lil_pin_r37();
    common_10::lil_pin_r38();
    common_10::lil_pin_r39();
    common_10::lil_pin_r40();
    common_10::lil_pin_r41();
    common_10::lil_pin_r42();
    common_10::lil_pin_r43();
    common_10::lil_pin_r44();
    common_10::lil_pin_r45();
    common_10::lil_pin_r46();
    common_10::lil_pin_r47();
    common_10::lil_pin_r48();
    common_10::lil_pin_r49();
    common_10::lil_pin_r50();
    common_10::lil_pin_r51();
    common_11::lil_pin_r52();
    common_11::lil_pin_r53();
    common_11::lil_pin_r54();
    common_11::lil_pin_r55();
    common_11::lil_pin_r56();
    common_11::lil_pin_r57();
    common_11::lil_pin_r58();
    common_11::lil_pin_r59();
    common_11::lil_pin_conv();
}

fn test_sparse_lil_hostile() {
    let nan = 0.0 / 0.0;
    let inf = 1.0e308 * 10.0;
    // A side of zero or above the limit is a fault.
    let z0, h0 = scientific::lil_new(0, 3);
    assert(h0.kind == scientific::F_RANGE);
    let z1, h1 = scientific::lil_new(3, scientific::MAX_SPARSE_SIDE + 1);
    assert(h1.kind == scientific::F_RANGE);
    // A buffer above the limit is a fault before any allocation.
    let z2, h1a = scientific::lil_new(scientific::MAX_SPARSE_SIDE, 3);
    assert(h1a.kind == scientific::F_LIMIT);
    let d0, h2 = scientific::lil_new(3, 4);
    assert(h2.kind == scientific::F_NONE);
    var d = d0;
    assert(scientific::lil_resize(d, scientific::MAX_SPARSE_SIDE, 4).kind == scientific::F_LIMIT);
    assert(scientific::lil_set(d, 1.0, 1.0, 5.0).kind == scientific::F_NONE);
    // A place that is not a whole place inside the matrix is a fault.
    assert(scientific::lil_set(d, nan, 0.0, 1.0).kind == scientific::F_RANGE);
    assert(scientific::lil_set(d, inf, 0.0, 1.0).kind == scientific::F_RANGE);
    assert(scientific::lil_set(d, 0.5, 0.0, 1.0).kind == scientific::F_RANGE);
    assert(scientific::lil_set(d, -1.0, 0.0, 1.0).kind == scientific::F_RANGE);
    assert(scientific::lil_set(d, 3.0, 0.0, 1.0).kind == scientific::F_RANGE);
    assert(scientific::lil_set(d, 0.0, 4.0, 1.0).kind == scientific::F_RANGE);
    assert(scientific::lil_set(d, 1.0e300, 0.0, 1.0).kind == scientific::F_RANGE);
    let g0, k0 = scientific::lil_get(d, -1.0, 0.0);
    assert(k0.kind == scientific::F_RANGE);
    // A bad row of lil_row is a fault.
    let q0, q1, k1 = scientific::lil_row(d, nan);
    assert(k1.kind == scientific::F_RANGE);
    let q2, q3, k2 = scientific::lil_row(d, 0.5);
    assert(k2.kind == scientific::F_RANGE);
    let q4, q5, k3 = scientific::lil_row(d, -1.0);
    assert(k3.kind == scientific::F_RANGE);
    let q6, q7, k4 = scientific::lil_row(d, 3.0);
    assert(k4.kind == scientific::F_RANGE);
    // A value that is not finite is a fault.
    assert(scientific::lil_set(d, 0.0, 0.0, nan).kind == scientific::F_VALUE);
    assert(scientific::lil_set(d, 0.0, 0.0, inf).kind == scientific::F_VALUE);
    // A fault leaves the matrix as it was.
    assert(scientific::lil_nnz(d) == 1);
    assert(common_7::lil_is(d, [1.0], [1.0], [5.0]));
    // A sum that overflows is a fault.
    let ov, h3 = scientific::coo_from_triplets(1, 1, [0.0, 0.0], [0.0, 0.0], [1.0e308, 1.0e308]);
    assert(h3.kind == scientific::F_NONE);
    let ovd, h4 = scientific::lil_from_coo(ov);
    assert(h4.kind == scientific::F_VALUE);
    var bd, h5 = scientific::lil_new(1, 2);
    assert(scientific::lil_set(bd, 0.0, 0.0, 1.0e308).kind == scientific::F_NONE);
    assert(scientific::lil_set(bd, 0.0, 1.0, 1.0e308).kind == scientific::F_NONE);
    let s0, h6 = scientific::lil_sum(bd);
    assert(h6.kind == scientific::F_VALUE);
    // A factor or a product that is not finite is a fault.
    let sc0, h7 = scientific::lil_scale(bd, nan);
    assert(h7.kind == scientific::F_VALUE);
    let sc1, h8 = scientific::lil_scale(bd, 10.0);
    assert(h8.kind == scientific::F_VALUE);
    // A vector of a wrong length is a fault.
    let mv, h9 = scientific::lil_matvec(d, [1.0, 2.0]);
    assert(h9.kind == scientific::F_SHAPE);
    // A diagonal outside the matrix is a fault, and the matrix stays.
    assert(scientific::lil_setdiag(d, 4, [1.0]).kind == scientific::F_RANGE);
    assert(scientific::lil_setdiag(d, -3, [1.0]).kind == scientific::F_RANGE);
    assert(scientific::lil_setdiag(d, 9223372036854775807, [1.0]).kind == scientific::F_RANGE);
    assert(scientific::lil_setdiag(d, 0, [1.0, nan]).kind == scientific::F_VALUE);
    assert(common_7::lil_is(d, [1.0], [1.0], [5.0]));
    assert(scientific::lil_resize(d, 0, 2).kind == scientific::F_RANGE);
    assert(common_7::lil_is(d, [1.0], [1.0], [5.0]));
    // A dense result above the limit is a fault.
    let huge, h10 = scientific::lil_new(100000, 100000);
    assert(h10.kind == scientific::F_NONE);
    let hd, h11 = scientific::lil_to_dense(huge);
    assert(h11.kind == scientific::F_LIMIT);
    // A dense operand that is not a matrix is a fault.
    let line, h12 = scientific::from_values([1.0, 2.0]);
    let ld, h13 = scientific::lil_from_dense(line);
    assert(h13.kind == scientific::F_SHAPE);
    // A set of zero on a missing place changes nothing.
    var zm, hz3 = scientific::lil_new(2, 2);
    assert(scientific::lil_set(zm, 0.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_nnz(zm) == 0);
    // A stored zero stays: a sum of duplicates that gives zero, a scale by
    // zero and a compressed row with a stored zero.
    let zc, hz0 = scientific::coo_from_triplets(2, 2, [1.0, 1.0], [1.0, 1.0], [1.0, -1.0]);
    assert(hz0.kind == scientific::F_NONE);
    let zd, hz1 = scientific::lil_from_coo(zc);
    assert(hz1.kind == scientific::F_NONE && scientific::lil_nnz(zd) == 1);
    let zs, hz2 = scientific::lil_scale(zd, 0.0);
    assert(hz2.kind == scientific::F_NONE && scientific::lil_nnz(zs) == 1);
    let zr, hz4 = scientific::csr_from_triplets(2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 2.0]);
    assert(hz4.kind == scientific::F_NONE);
    let zrd, hz5 = scientific::lil_from_csr(zr);
    assert(hz5.kind == scientific::F_NONE && scientific::lil_nnz(zrd) == 2);
    // The count at the limit: a new place is a fault, an old place is not.
    var lim = scientific::Lil { rows: 3, cols: 3, n: scientific::MAX_SPARSE_NNZ, buf: [0.0, 1.0, 1.0, 1.0, 1.0, 2.0] };
    assert(scientific::lil_set(lim, 0.0, 2.0, 1.0).kind == scientific::F_LIMIT);
    assert(lim.n == scientific::MAX_SPARSE_NNZ && lim.buf.len == 6);
    assert(scientific::lil_set(lim, 0.0, 1.0, 5.0).kind == scientific::F_NONE);
    assert(lim.buf[5] == 5.0);
    // One thousand sets in random order give sorted rows and the right count.
    var many, h14 = scientific::lil_new(20, 50);
    var mirror: f64[>..] = [];
    var n: i64 = 0;
    while n < 1000 { mirror.push(0.0); n += 1; }
    var seed: i64 = 12345;
    n = 0;
    while n < 1000 {
        seed = (seed * 1103515245 + 12345) % 2147483648;
        let r = (seed / 65536) % 20;
        seed = (seed * 1103515245 + 12345) % 2147483648;
        let c = (seed / 65536) % 50;
        var v = ((seed / 7) % 5) as f64;
        mirror[r * 50 + c] = v;
        assert(scientific::lil_set(many, r as f64, c as f64, v).kind == scientific::F_NONE);
        n += 1;
    }
    var cnt: i64 = 0;
    n = 0;
    while n < 1000 { if mirror[n] != 0.0 { cnt += 1; } n += 1; }
    assert(scientific::lil_nnz(many) == cnt);
    var ri: i64 = 0;
    while ri < 20 {
        let rc, rv, hr = scientific::lil_row(many, ri as f64);
        assert(hr.kind == scientific::F_NONE);
        var q: i64 = 0;
        while q < rc.data.len {
            if q > 0 { assert(rc.data[q] > rc.data[q - 1]); }
            assert(mirror[ri * 50 + (rc.data[q] as i64)] == rv.data[q]);
            q += 1;
        }
        ri += 1;
    }
}
