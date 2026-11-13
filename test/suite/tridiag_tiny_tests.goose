// The tests of the group tridiag_tiny. Run them with SCIENTIFIC_SUITES=tridiag_tiny.
//
// A tridiagonal matrix of any scale gives the values of SciPy, in the units
// of the caller. The pins come from the generated file of the group.
namespace tridiag_tiny_tests;

import scientific from "../..";
import common;
import tridiag_tiny_pins;

// tt_close tells whether two values agree to a relative tolerance.
fn tt_close(a: f64, b: f64, tol: f64) -> bool {
    return common::mathshim_abs(a - b) <= tol * common::mathshim_abs(b);
}

// tt_scaled gives the unit entries times the scale s.
fn tt_scaled(v: f64[:], s: f64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < v.len { out.push(v[i] * s); i += 1; }
    return out;
}

// tt_want gives the values of SciPy of the matrix m (0 or 1) at the scale i.
fn tt_want(m: i64, i: i64) -> f64[>..] {
    if m == 0 {
        if i == 0 { return scientific::copy_slice(tridiag_tiny_pins::S4_W0); }
        if i == 1 { return scientific::copy_slice(tridiag_tiny_pins::S4_W1); }
        if i == 2 { return scientific::copy_slice(tridiag_tiny_pins::S4_W2); }
        if i == 3 { return scientific::copy_slice(tridiag_tiny_pins::S4_W3); }
        return scientific::copy_slice(tridiag_tiny_pins::S4_W4);
    }
    if i == 0 { return scientific::copy_slice(tridiag_tiny_pins::S200_W0); }
    if i == 1 { return scientific::copy_slice(tridiag_tiny_pins::S200_W1); }
    if i == 2 { return scientific::copy_slice(tridiag_tiny_pins::S200_W2); }
    if i == 3 { return scientific::copy_slice(tridiag_tiny_pins::S200_W3); }
    return scientific::copy_slice(tridiag_tiny_pins::S200_W4);
}

// tt_residual gives the largest error of T v = w v over the columns of v.
fn tt_residual(d: f64[:], e: f64[:], w: scientific::Arr&, v: scientific::Arr&) -> f64 {
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

// tt_gram gives the largest error of the orthonormality of the columns of v.
fn tt_gram(v: scientific::Arr&, n: i64) -> f64 {
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

// tt_run checks the values and the vectors of one matrix at one scale.
fn tt_run(d0: f64[:], e0: f64[:], m: i64, i: i64) {
    let s = tridiag_tiny_pins::SCALES[i];
    let d = tt_scaled(d0, s);
    let e = tt_scaled(e0, s);
    let want = tt_want(m, i);
    let w, f = scientific::eigvalsh_tridiagonal(d, e);
    assert(f.kind == scientific::F_NONE && w.data.len == want.len);
    var q: i64 = 0;
    while q < want.len { assert(tt_close(w.data[q], want[q], 1.0e-12)); q += 1; }
    let w2, v, f2 = scientific::eigh_tridiagonal(d, e);
    assert(f2.kind == scientific::F_NONE);
    q = 0;
    while q < want.len { assert(tt_close(w2.data[q], want[q], 1.0e-12)); q += 1; }
    assert(tt_gram(v, d.len) <= 1.0e-10);
    assert(tt_residual(d, e, w2, v) <= 1.0e-10 * 4.0 * s);
}

// The values at each scale match SciPy, and the vectors stay orthonormal.
fn test_tridiag_tiny_scales() {
    var i: i64 = 0;
    while i < 5 {
        tt_run(tridiag_tiny_pins::S4_D, tridiag_tiny_pins::S4_E, 0, i);
        tt_run(tridiag_tiny_pins::S200_D, tridiag_tiny_pins::S200_E, 1, i);
        i += 1;
    }
}

// A diagonal matrix, a zero matrix and a matrix of one tiny coupling.
fn test_tridiag_tiny_special() {
    let w, f = scientific::eigvalsh_tridiagonal([1.0e-20, 2.0e-20, 3.0e-20], [0.0, 0.0]);
    assert(f.kind == scientific::F_NONE);
    assert(tt_close(w.data[0], 1.0e-20, 1.0e-12) && tt_close(w.data[1], 2.0e-20, 1.0e-12) && tt_close(w.data[2], 3.0e-20, 1.0e-12));
    let w2, f2 = scientific::eigvalsh_tridiagonal([0.0, 0.0], [1.0e-300]);
    assert(f2.kind == scientific::F_NONE);
    assert(tt_close(w2.data[0], -1.0e-300, 1.0e-12) && tt_close(w2.data[1], 1.0e-300, 1.0e-12));
    let w3, v3, f3 = scientific::eigh_tridiagonal([0.0, 0.0, 0.0], [0.0, 0.0]);
    assert(f3.kind == scientific::F_NONE && w3.data[0] == 0.0 && w3.data[2] == 0.0);
}

// The largest and the smallest finite values fault nothing and finish.
fn test_tridiag_tiny_extremes() {
    let w, f = scientific::eigvalsh_tridiagonal([1.7e308, 1.7e308], [0.0]);
    assert(f.kind == scientific::F_NONE && tt_close(w.data[1], 1.7e308, 1.0e-12));
    let w2, v2, f2 = scientific::eigh_tridiagonal([5.0e-324, 1.0e-323], [0.0]);
    assert(f2.kind == scientific::F_NONE && w2.data[0] == 5.0e-324 && w2.data[1] == 1.0e-323);
    let w3, f3 = scientific::eigvalsh_tridiagonal([1.0e-310, 2.0e-310, 3.0e-310], [1.0e-310, 1.0e-310]);
    assert(f3.kind == scientific::F_NONE && w3.data.len == 3 && w3.data[0] > -1.0e-310 && w3.data[2] < 5.0e-310);
}

// The select by value stays in the units of the caller, and so does a
// block that has a scale of its own beside a large block.
fn test_tridiag_tiny_select_blocks() {
    let d = tt_scaled(tridiag_tiny_pins::S4_D, 1.0e-20);
    let e = tt_scaled(tridiag_tiny_pins::S4_E, 1.0e-20);
    let w, f = scientific::eigvalsh_tridiagonal(d, e, "v", [1.0e-20, 3.0e-20]);
    assert(f.kind == scientific::F_NONE && w.data.len == 2);
    assert(tt_close(w.data[0], tridiag_tiny_pins::S4_W0[1], 1.0e-12));
    let w2, f2 = scientific::eigvalsh_tridiagonal(d, e, "i", [1.0, 2.0]);
    assert(f2.kind == scientific::F_NONE && w2.data.len == 2);
    assert(tt_close(w2.data[1], tridiag_tiny_pins::S4_W0[2], 1.0e-12));
    // A large block beside a tiny block.
    let w3, v3, f3 = scientific::eigh_tridiagonal([1.0e10, 1.0e-10, 2.0e-10], [0.0, 1.0e-10]);
    assert(f3.kind == scientific::F_NONE);
    assert(tt_close(w3.data[2], 1.0e10, 1.0e-12));
    assert(w3.data[0] < 1.0e-9 && w3.data[0] > 0.0);
}

// A matrix of normal scale gives the unit values of SciPy.
fn test_tridiag_tiny_normal() {
    let w, f = scientific::eigvalsh_tridiagonal(tridiag_tiny_pins::S4_D, tridiag_tiny_pins::S4_E);
    assert(f.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < 4 { assert(tt_close(w.data[i], tridiag_tiny_pins::S4_WU[i], 1.0e-14)); i += 1; }
    let w2, f2 = scientific::eigvalsh_tridiagonal(tridiag_tiny_pins::S200_D, tridiag_tiny_pins::S200_E);
    assert(f2.kind == scientific::F_NONE);
    i = 0;
    // The bisection holds an absolute error of the machine epsilon times the
    // largest value, so a small value is compared to that size.
    while i < 200 { assert(common::mathshim_abs(w2.data[i] - tridiag_tiny_pins::S200_WU[i]) <= 1.0e-14 * w2.data[199]); i += 1; }
}

// A tiny matrix with a zero off-diagonal value keeps orthonormal vectors.
fn test_tridiag_tiny_zero_coupling() {
    let d = [1.0e-100, 2.0e-100, 3.0e-100, 4.0e-100];
    let e = [5.0e-101, 0.0, 5.0e-101];
    let w, v, f = scientific::eigh_tridiagonal(d, e);
    assert(f.kind == scientific::F_NONE);
    assert(tt_gram(v, 4) <= 1.0e-10);
    assert(tt_residual(d, e, w, v) <= 1.0e-10 * 4.0e-100);
}

// Two diagonal values near the limit of the number format keep their
// coupling. The scale of the place does not overflow.
fn test_tridiag_tiny_large_diagonal() {
    let w, f = scientific::eigvalsh_tridiagonal([1.5e308, 1.5e308], [1.0e307]);
    assert(f.kind == scientific::F_NONE);
    assert(common::mathshim_abs(w.data[0] - 1.4e308) <= 1.0e-14 * 1.4e308);
    assert(common::mathshim_abs(w.data[1] - 1.6e308) <= 1.0e-14 * 1.6e308);
    let w2, f2 = scientific::eigvalsh_tridiagonal([1.0e308, -1.0e308], [1.0e308]);
    assert(f2.kind == scientific::F_NONE);
    assert(common::mathshim_abs(w2.data[0] + 1.4142135623730951e308) <= 1.0e-14 * 1.4142135623730951e308);
    assert(common::mathshim_abs(w2.data[1] - 1.4142135623730951e308) <= 1.0e-14 * 1.4142135623730951e308);
}
