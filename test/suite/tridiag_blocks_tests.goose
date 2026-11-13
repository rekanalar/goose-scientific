// The tests of the group tridiag_blocks. Run them with SCIENTIFIC_SUITES=tridiag_blocks.
//
// A tridiagonal matrix with a zero or a negligible off diagonal value falls
// into blocks. The pins come from the generated file of the group.
namespace tridiag_blocks_tests;

import scientific from "../..";
import os;
import common;
import tridiag_blocks_pins;

// tb_close tells whether two values agree to the tolerance.
fn tb_close(a: f64, b: f64, tol: f64) -> bool {
    return common::mathshim_abs(a - b) <= tol * (1.0 + common::mathshim_abs(b));
}

// tb_gram gives the largest error of the orthonormality of the columns of v.
fn tb_gram(v: scientific::Arr&, n: i64) -> f64 {
    let nc = v.shape[1];
    var worst: f64 = 0.0;
    var j: i64 = 0;
    while j < nc {
        var j2: i64 = 0;
        while j2 < nc {
            var dot: f64 = 0.0;
            var i: i64 = 0;
            while i < n { dot += v.data[i * nc + j] * v.data[i * nc + j2]; i += 1; }
            var want: f64 = 0.0;
            if j == j2 { want = 1.0; }
            let er = common::mathshim_abs(dot - want);
            if er > worst { worst = er; }
            j2 += 1;
        }
        j += 1;
    }
    return worst;
}

// tb_residual gives the largest error of T v = w v over the columns of v.
fn tb_residual(d: f64[:], e: f64[:], w: scientific::Arr&, v: scientific::Arr&) -> f64 {
    let n = d.len;
    let nc = v.shape[1];
    var worst: f64 = 0.0;
    var j: i64 = 0;
    while j < nc {
        var i: i64 = 0;
        while i < n {
            var acc = d[i] * v.data[i * nc + j];
            if i > 0 { acc += e[i - 1] * v.data[(i - 1) * nc + j]; }
            if i < n - 1 { acc += e[i] * v.data[(i + 1) * nc + j]; }
            let er = common::mathshim_abs(acc - w.data[j] * v.data[i * nc + j]);
            if er > worst { worst = er; }
            i += 1;
        }
        j += 1;
    }
    return worst;
}

// tb_solve solves a matrix, checks it against the pinned values and checks
// the orthonormality and the residual of the vectors.
fn tb_solve(d: f64[:], e: f64[:], want: f64[:]) {
    let w, v, f = scientific::eigh_tridiagonal(d, e);
    assert(f.kind == scientific::F_NONE);
    assert(w.data.len == want.len);
    var i: i64 = 0;
    while i < want.len { assert(tb_close(w.data[i], want[i], 1.0e-12)); i += 1; }
    assert(v.shape[0] == d.len && v.shape[1] == want.len);
    assert(tb_gram(v, d.len) <= 1.0e-10);
    assert(tb_residual(d, e, w, v) <= 1.0e-10);
    let w2, f2 = scientific::eigvalsh_tridiagonal(d, e);
    assert(f2.kind == scientific::F_NONE);
    i = 0;
    while i < want.len { assert(tb_close(w2.data[i], w.data[i], 1.0e-12)); i += 1; }
}

// The four matrices that gave parallel vectors, a blocked matrix and an
// unreduced one.
fn test_tridiag_blocks_vectors() {
    tb_solve(tridiag_blocks_pins::D_M1, tridiag_blocks_pins::E_M1, tridiag_blocks_pins::W_M1);
    tb_solve(tridiag_blocks_pins::D_M2, tridiag_blocks_pins::E_M2, tridiag_blocks_pins::W_M2);
    tb_solve(tridiag_blocks_pins::D_M3, tridiag_blocks_pins::E_M3, tridiag_blocks_pins::W_M3);
    tb_solve(tridiag_blocks_pins::D_M4, tridiag_blocks_pins::E_M4, tridiag_blocks_pins::W_M4);
    tb_solve(tridiag_blocks_pins::D_M5, tridiag_blocks_pins::E_M5, tridiag_blocks_pins::W_M5);
    tb_solve(tridiag_blocks_pins::D_M6, tridiag_blocks_pins::E_M6, tridiag_blocks_pins::W_M6);
    tb_solve(tridiag_blocks_pins::D_M8, tridiag_blocks_pins::E_M8, tridiag_blocks_pins::W_M8);
}

// A large value beside two tiny ones must not hide the coupling of the
// tiny ones. The values must match the ones of SciPy.
fn test_tridiag_blocks_hidden_coupling() {
    let d = tridiag_blocks_pins::D_M7;
    let e = tridiag_blocks_pins::E_M7;
    let want = tridiag_blocks_pins::W_M7;
    let w, v, f = scientific::eigh_tridiagonal(d, e);
    assert(f.kind == scientific::F_NONE);
    assert(w.data.len == 3);
    var i: i64 = 0;
    while i < 3 {
        assert(tb_close(w.data[i], want[i], 1.0e-6));
        i += 1;
    }
    // The coupling of the two tiny values is the whole story of them.
    assert(common::mathshim_abs(w.data[0]) > 1.0e-8);
    assert(common::mathshim_abs(w.data[1]) > 1.0e-8);
}

// The choices of the call give the pairs of the whole matrix by rank.
fn test_tridiag_blocks_select() {
    let d = tridiag_blocks_pins::D_M5;
    let e = tridiag_blocks_pins::E_M5;
    let want = tridiag_blocks_pins::W_M5;
    let n = d.len;
    // A rank range.
    let w, v, f = scientific::eigh_tridiagonal(d, e, false, "i", [3.0, 9.0]);
    assert(f.kind == scientific::F_NONE);
    assert(w.data.len == 7 && v.shape[0] == n && v.shape[1] == 7);
    var i: i64 = 0;
    while i < 7 { assert(tb_close(w.data[i], want[3 + i], 1.0e-12)); i += 1; }
    assert(tb_gram(v, n) <= 1.0e-10);
    assert(tb_residual(d, e, w, v) <= 1.0e-10);
    // A range of values: the values above the low bound up to the high bound.
    let wv, vv, fv = scientific::eigh_tridiagonal(d, e, false, "v", [1.9, 3.2]);
    assert(fv.kind == scientific::F_NONE);
    var cnt: i64 = 0;
    i = 0;
    while i < n { if want[i] > 1.9 && want[i] <= 3.2 { cnt += 1; } i += 1; }
    assert(wv.data.len == cnt && vv.shape[1] == cnt);
    assert(tb_gram(vv, n) <= 1.0e-10);
    assert(tb_residual(d, e, wv, vv) <= 1.0e-10);
    // The values only call.
    let wo, vo, fo = scientific::eigh_tridiagonal(d, e, true);
    assert(fo.kind == scientific::F_NONE && vo.data.len == 0 && wo.data.len == n);
    i = 0;
    while i < n { assert(tb_close(wo.data[i], want[i], 1.0e-12)); i += 1; }
}

// A matrix of one value gives its value and a unit vector.
fn test_tridiag_blocks_one_value() {
    let w, v, f = scientific::eigh_tridiagonal([3.5], []);
    assert(f.kind == scientific::F_NONE);
    assert(w.data.len == 1 && v.data.len == 1 && common::mathshim_abs(v.data[0]) == 1.0);
    assert(tb_close(w.data[0], 3.5, 1.0e-12));
}

// A matrix of 4096 zero off diagonal values needs no solve of a block.
fn test_tridiag_blocks_wide() {
    var d: f64[>..] = [];
    var e: f64[>..] = [];
    var i: i64 = 0;
    while i < 4096 {
        d.push(((i % 7) as f64) + 1.0);
        if i < 4095 { e.push(0.0); }
        i += 1;
    }
    let w, f = scientific::eigvalsh_tridiagonal(d, e);
    assert(f.kind == scientific::F_NONE && w.data.len == 4096);
    assert(tb_close(w.data[0], 1.0, 1.0e-12) && tb_close(w.data[4095], 7.0, 1.0e-12));
    let w2, v2, f2 = scientific::eigh_tridiagonal(d, e, true);
    assert(f2.kind == scientific::F_NONE && w2.data[0] == 1.0 && w2.data[4095] == 7.0);
}

// The count of the values below a bound works on a matrix with a zero
// off diagonal value. The count of the whole matrix loses every sign
// change then, so the count of each block must add up.
fn test_tridiag_blocks_count() {
    // The values are 1, 1, 3 and 3.
    let d = [3.0, 1.0, 1.0, 3.0];
    let e = [0.0, 0.0, 0.0];
    // The range holds the two values of 1.
    let w, v, f = scientific::eigh_tridiagonal(d, e, false, "v", [1.0, 3.0]);
    assert(f.kind == scientific::F_NONE);
    assert(w.data.len == 2);
    assert(tb_close(w.data[0], 1.0, 1.0e-12) && tb_close(w.data[1], 1.0, 1.0e-12));
    // The range holds every value.
    let w2, v2, f2 = scientific::eigh_tridiagonal(d, e, false, "v", [0.0, 4.0]);
    assert(f2.kind == scientific::F_NONE && w2.data.len == 4);
    // The values only call gives the same count.
    let w3, f3 = scientific::eigvalsh_tridiagonal(d, e, "v", [1.0, 3.0]);
    assert(f3.kind == scientific::F_NONE && w3.data.len == 2);
}

// The faults of the call stay as they are.
fn test_tridiag_blocks_faults() {
    let w, v, f = scientific::eigh_tridiagonal([1.0, 1.0], [0.0, 0.0]);
    assert(f.kind == scientific::F_SHAPE);
    let w2, v2, f2 = scientific::eigh_tridiagonal([1.0, 0.0 / 0.0], [0.0]);
    assert(f2.kind == scientific::F_VALUE);
    let w3, v3, f3 = scientific::eigh_tridiagonal([1.0, 1.0, 1.0], [0.0, 0.0], false, "i", [2.0, 1.0]);
    assert(f3.kind != scientific::F_NONE);
    // A side above the limit of the side.
    var big: f64[>..] = [];
    var bige: f64[>..] = [];
    var i: i64 = 0;
    while i < 4097 {
        big.push(1.0);
        if i < 4096 { bige.push(0.0); }
        i += 1;
    }
    let w4, v4, f4 = scientific::eigh_tridiagonal(big, bige);
    assert(f4.kind == scientific::F_LIMIT);
}
