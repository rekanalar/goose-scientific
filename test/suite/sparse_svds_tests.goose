// The tests of the group sparse_svds. Run them with SCIENTIFIC_SUITES=sparse_svds.
//
// The pins come from the generated file of the group.
namespace sparse_svds_tests;

import scientific from "../..";
import os;
import common;
import sparse_svds_pins;

// The tolerance of every value of a pin.
const SVDS_TOL: f64 = 1.0e-8;

// svds_csr builds a matrix of compressed rows from its rows, its columns and
// its values. The values stand in row major order.
fn svds_csr(rows: i64, cols: i64, vals: f64[:]) -> scientific::Csr {
    let s, f = scientific::csr_from_dense(common::mat(vals, rows, cols));
    assert(f.kind == scientific::F_NONE);
    return s;
}

// svds_a85 gives the matrix A85 of the pins.
fn svds_a85() -> scientific::Csr {
    return svds_csr(8, 5, sparse_svds_pins::A85);
}

// svds_a85_t gives the transpose of the matrix A85 of the pins.
fn svds_a85_t() -> scientific::Csr {
    let s, f = scientific::csr_transpose(svds_a85());
    assert(f.kind == scientific::F_NONE);
    return s;
}

// svds_diag gives a diagonal matrix of n values.
fn svds_diag(v: f64[:]) -> scientific::Csr {
    let s, f = scientific::csr_diags(v, 0, v.len, v.len);
    assert(f.kind == scientific::F_NONE);
    return s;
}

// svds_close tells whether two values agree to the tolerance.
fn svds_close(a: f64, b: f64) -> bool {
    let d = common::mathshim_abs(a - b);
    return d <= SVDS_TOL * (1.0 + common::mathshim_abs(b));
}

// svds_match checks the values of a result against a pin.
fn svds_match(got: f64[:], want: f64[:]) {
    assert(got.len == want.len);
    var i: i64 = 0;
    while i < got.len {
        assert(svds_close(got[i], want[i]));
        i += 1;
    }
}

// svds_gram_err gives the largest error of the orthonormality of k vectors.
// The vector v holds its values at the places vstride + e * estride.
fn svds_gram_err(a: f64[:], k: i64, n: i64, vstride: i64, estride: i64) -> f64 {
    var worst: f64 = 0.0;
    var i: i64 = 0;
    while i < k {
        var j: i64 = 0;
        while j < k {
            var t: f64 = 0.0;
            var e: i64 = 0;
            while e < n { t += a[i * vstride + e * estride] * a[j * vstride + e * estride]; e += 1; }
            var want = 0.0;
            if i == j { want = 1.0; }
            let d = common::mathshim_abs(t - want);
            if d > worst { worst = d; }
            j += 1;
        }
        i += 1;
    }
    return worst;
}

// svds_orth_cols gives the largest error of the orthonormality of the
// columns of a matrix of rows by k.
fn svds_orth_cols(a: f64[:], rows: i64, k: i64) -> f64 {
    return svds_gram_err(a, k, rows, 1, k);
}

// svds_orth_rows gives the largest error of the orthonormality of the rows of
// a matrix of k by cols.
fn svds_orth_rows(a: f64[:], k: i64, cols: i64) -> f64 {
    return svds_gram_err(a, k, cols, cols, 1);
}

// svds_residual gives the largest error of the relation A v = s u. The rows of
// vt hold the vectors v and the columns of u hold the vectors u.
fn svds_residual(s: scientific::Csr&, u: f64[:], sv: f64[:], vt: f64[:], k: i64) -> f64 {
    var worst: f64 = 0.0;
    var i: i64 = 0;
    while i < k {
        let y, f = scientific::csr_matvec(s, vt[i * s.cols..i * s.cols + s.cols]);
        assert(f.kind == scientific::F_NONE);
        var r: i64 = 0;
        while r < s.rows {
            let e = common::mathshim_abs(y.data[r] - sv[i] * u[r * k + i]);
            if e > worst { worst = e; }
            r += 1;
        }
        i += 1;
    }
    return worst;
}

// The largest singular values of a matrix, with their vectors.
fn test_sparse_svds_largest() {
    let s = svds_a85();
    let ua, sva, vta, fa = scientific::svds(s, 1);
    assert(fa.kind == scientific::F_NONE);
    svds_match(sva.data, sparse_svds_pins::A85_LM1);
    let ub, svb, vtb, fb = scientific::svds(s, 2);
    assert(fb.kind == scientific::F_NONE);
    svds_match(svb.data, sparse_svds_pins::A85_LM2);
    let uc, svc, vtc, fc = scientific::svds(s, 4);
    assert(fc.kind == scientific::F_NONE);
    svds_match(svc.data, sparse_svds_pins::A85_LM4);
    // The transpose of the matrix gives the same values.
    let st = svds_a85_t();
    let ud, svd, vtd, fd = scientific::svds(st, 2);
    assert(fd.kind == scientific::F_NONE);
    svds_match(svd.data, sparse_svds_pins::A85T_LM2);
    // A diagonal matrix gives its largest values.
    let sd = svds_diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ue, sve, vte, fe = scientific::svds(sd, 2);
    assert(fe.kind == scientific::F_NONE);
    svds_match(sve.data, sparse_svds_pins::DIAG6_LM2);
    // A diagonal matrix with a repeated value gives every copy of it.
    let sr = svds_diag([1.0, 2.0, 2.0, 2.0, 5.0, 5.0]);
    let uf, svf, vtf, ff = scientific::svds(sr, 3);
    assert(ff.kind == scientific::F_NONE);
    svds_match(svf.data, sparse_svds_pins::DIAGREP_LM3);
}

// The smallest singular values of a matrix.
fn test_sparse_svds_smallest() {
    let s = svds_a85();
    let ua, sva, vta, fa = scientific::svds(s, 2, "SM");
    assert(fa.kind == scientific::F_NONE);
    svds_match(sva.data, sparse_svds_pins::A85_SM2);
    let st = svds_a85_t();
    let ub, svb, vtb, fb = scientific::svds(st, 2, "SM");
    assert(fb.kind == scientific::F_NONE);
    svds_match(svb.data, sparse_svds_pins::A85T_SM2);
    let sd = svds_diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let uc, svc, vtc, fc = scientific::svds(sd, 2, "SM");
    assert(fc.kind == scientific::F_NONE);
    svds_match(svc.data, sparse_svds_pins::DIAG6_SM2);
    // A matrix of rank four gives the zeros of its two missing values.
    let sz = svds_diag([0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    let ud, svd, vtd, fd = scientific::svds(sz, 2, "SM");
    assert(fd.kind == scientific::F_NONE);
    svds_match(svd.data, sparse_svds_pins::DIAG4_SM2);
    // The vectors of the zero values stand orthonormal.
    assert(svds_orth_cols(ud.data, 6, 2) <= SVDS_TOL);
    assert(svds_orth_rows(vtd.data, 2, 6) <= SVDS_TOL);
}

// A small value against a large one keeps its relative accuracy and its vectors.
fn test_sparse_svds_small_values() {
    let s = svds_diag(sparse_svds_pins::LADDER);
    let ua, sva, vta, fa = scientific::svds(s, 2, "SM");
    assert(fa.kind == scientific::F_NONE);
    assert(sva.data.len == 2);
    var i: i64 = 0;
    while i < 2 {
        let want = sparse_svds_pins::LADDER_SM2[i];
        assert(common::mathshim_abs(sva.data[i] - want) <= SVDS_TOL * want);
        i += 1;
    }
    assert(svds_orth_cols(ua.data, 40, 2) <= SVDS_TOL);
    assert(svds_orth_rows(vta.data, 2, 40) <= SVDS_TOL);
    // The matrix is diagonal, so the relation can use its own values.
    assert(svds_residual(s, ua.data, sva.data, vta.data, 2) <= SVDS_TOL);
    // The values only mode gives the same accurate values.
    let ub, svb, vtb, fb = scientific::svds(s, 2, "SM", 0.0, [], 0, 0, scientific::SVDS_VALUES);
    assert(fb.kind == scientific::F_NONE);
    assert(common::mathshim_abs(svb.data[0] - sparse_svds_pins::LADDER_SM2[0]) <= SVDS_TOL * sparse_svds_pins::LADDER_SM2[0]);
}

// The vectors of a result stand orthonormal and hold the relation.
fn test_sparse_svds_vectors() {
    let s = svds_a85();
    let ua, sva, vta, fa = scientific::svds(s, 2);
    assert(fa.kind == scientific::F_NONE);
    assert(ua.shape[0] == 8 && ua.shape[1] == 2);
    assert(vta.shape[0] == 2 && vta.shape[1] == 5);
    assert(svds_orth_cols(ua.data, 8, 2) <= SVDS_TOL);
    assert(svds_orth_rows(vta.data, 2, 5) <= SVDS_TOL);
    assert(svds_residual(s, ua.data, sva.data, vta.data, 2) <= SVDS_TOL);
    // The product of the result equals the product of SciPy.
    var r: i64 = 0;
    while r < 8 {
        var c: i64 = 0;
        while c < 5 {
            var t: f64 = 0.0;
            var j: i64 = 0;
            while j < 2 { t += ua.data[r * 2 + j] * sva.data[j] * vta.data[j * 5 + c]; j += 1; }
            assert(svds_close(t, sparse_svds_pins::A85_PRODUCT[r * 5 + c]));
            c += 1;
        }
        r += 1;
    }
    // The smallest values give the same invariants.
    let ub, svb, vtb, fb = scientific::svds(s, 2, "SM");
    assert(fb.kind == scientific::F_NONE);
    assert(svds_orth_cols(ub.data, 8, 2) <= SVDS_TOL);
    assert(svds_orth_rows(vtb.data, 2, 5) <= SVDS_TOL);
    assert(svds_residual(s, ub.data, svb.data, vtb.data, 2) <= SVDS_TOL);
}

// A wide matrix gives the same invariants.
fn test_sparse_svds_wide() {
    let s = svds_a85_t();
    let ua, sva, vta, fa = scientific::svds(s, 2);
    assert(fa.kind == scientific::F_NONE);
    assert(ua.shape[0] == 5 && ua.shape[1] == 2);
    assert(vta.shape[0] == 2 && vta.shape[1] == 8);
    assert(svds_orth_cols(ua.data, 5, 2) <= SVDS_TOL);
    assert(svds_orth_rows(vta.data, 2, 8) <= SVDS_TOL);
    assert(svds_residual(s, ua.data, sva.data, vta.data, 2) <= SVDS_TOL);
}

// The call gives the same result on every run.
fn test_sparse_svds_same() {
    let s = svds_a85();
    let ua, sva, vta, fa = scientific::svds(s, 2);
    let ub, svb, vtb, fb = scientific::svds(s, 2);
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < 2 {
        assert(sva.data[i] == svb.data[i]);
        assert(ua.data[i] == ub.data[i]);
        assert(vta.data[i] == vtb.data[i]);
        i += 1;
    }
    // A start vector of the right length gives the same values.
    let uc, svc, vtc, fc = scientific::svds(s, 2, "LM", 0.0, [1.0, 0.0, 0.0, 0.0, 0.0]);
    assert(fc.kind == scientific::F_NONE);
    svds_match(svc.data, sparse_svds_pins::A85_LM2);
    // A start vector of a huge scale gives the same values.
    let sd = svds_diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ue, sve, vte, fe = scientific::svds(sd, 2, "SM", 0.0, [1.0e308, 1.0e308, 1.0e308, 1.0e308, 1.0e308, 1.0e308], 3);
    assert(fe.kind == scientific::F_NONE);
    svds_match(sve.data, sparse_svds_pins::DIAG6_SM2);
    // A space of a given size gives the same values.
    let ud, svd, vtd, fd = scientific::svds(s, 2, "LM", 0.0, [], 4);
    assert(fd.kind == scientific::F_NONE);
    svds_match(svd.data, sparse_svds_pins::A85_LM2);
}

// The four modes of the result.
fn test_sparse_svds_modes() {
    let s = svds_a85();
    let ua, sva, vta, fa = scientific::svds(s, 2, "LM", 0.0, [], 0, 0, scientific::SVDS_BOTH);
    assert(fa.kind == scientific::F_NONE);
    assert(ua.data.len == 16 && sva.data.len == 2 && vta.data.len == 10);
    let ub, svb, vtb, fb = scientific::svds(s, 2, "LM", 0.0, [], 0, 0, scientific::SVDS_VALUES);
    assert(fb.kind == scientific::F_NONE);
    assert(ub.data.len == 0 && svb.data.len == 2 && vtb.data.len == 0);
    svds_match(svb.data, sparse_svds_pins::A85_LM2);
    let uc, svc, vtc, fc = scientific::svds(s, 2, "LM", 0.0, [], 0, 0, scientific::SVDS_LEFT);
    assert(fc.kind == scientific::F_NONE);
    assert(uc.data.len == 16 && svc.data.len == 2 && vtc.data.len == 0);
    let ud, svd, vtd, fd = scientific::svds(s, 2, "LM", 0.0, [], 0, 0, scientific::SVDS_RIGHT);
    assert(fd.kind == scientific::F_NONE);
    assert(ud.data.len == 0 && svd.data.len == 2 && vtd.data.len == 10);
}

// The faults of the counts and of the choices.
fn test_sparse_svds_faults() {
    let s = svds_a85();
    let ua, sva, vta, fa = scientific::svds(s, 0);
    assert(fa.kind == scientific::F_RANGE && ua.data.len == 0 && sva.data.len == 0 && vta.data.len == 0);
    let ub, svb, vtb, fb = scientific::svds(s, 5);
    assert(fb.kind == scientific::F_RANGE);
    let uc, svc, vtc, fc = scientific::svds(s, 2, "XX");
    assert(fc.kind == scientific::F_VALUE);
    let ud, svd, vtd, fd = scientific::svds(s, 2, "LM", -1.0);
    assert(fd.kind == scientific::F_VALUE);
    let ue, sve, vte, fe = scientific::svds(s, 2, "LM", 0.0, [], -1);
    assert(fe.kind == scientific::F_RANGE);
    let uf, svf, vtf, ff = scientific::svds(s, 2, "LM", 0.0, [], 2);
    assert(ff.kind == scientific::F_RANGE);
    let ug, svg, vtg, fg = scientific::svds(s, 2, "LM", 0.0, [], 5);
    assert(fg.kind == scientific::F_RANGE);
    let uh, svh, vth, fh = scientific::svds(s, 2, "LM", 0.0, [], 0, -1);
    assert(fh.kind == scientific::F_RANGE);
    let ui, svi, vti, fi = scientific::svds(s, 2, "LM", 0.0, [], 0, 0, 4);
    assert(fi.kind == scientific::F_RANGE);
}

// The faults of the operands.
fn test_sparse_svds_bad_operands() {
    let s = svds_a85();
    // A start vector of the wrong length gives a fault of shape.
    let ua, sva, vta, fa = scientific::svds(s, 2, "LM", 0.0, [1.0, 0.0]);
    assert(fa.kind == scientific::F_SHAPE);
    // A start vector that holds a value that is not finite gives a fault of value.
    let ub, svb, vtb, fb = scientific::svds(s, 2, "LM", 0.0, [1.0, 0.0, 0.0, 0.0, 0.0 / 0.0]);
    assert(fb.kind == scientific::F_VALUE);
    // A start vector of zeros gives a fault of value.
    let uc, svc, vtc, fc = scientific::svds(s, 2, "LM", 0.0, [0.0, 0.0, 0.0, 0.0, 0.0]);
    assert(fc.kind == scientific::F_VALUE);
    // A matrix that holds a value that is not finite gives a fault of value.
    let bad = svds_csr(2, 2, [1.0, 0.0, 0.0, 0.0 / 0.0]);
    let ud, svd, vtd, fd = scientific::svds(bad, 1);
    assert(fd.kind == scientific::F_VALUE);
    // A matrix with a broken layout gives a fault of shape.
    var buf: f64[>..] = [0.0, 2.0, 0.0, 1.0, 1.0];
    let broken = scientific::Csr { rows: 2, cols: 2, nnz: 1, buf: copy(buf) };
    let ue, sve, vte, fe = scientific::svds(broken, 1);
    assert(fe.kind == scientific::F_SHAPE);
}

// The faults of the limits.
fn test_sparse_svds_limits() {
    // A diagonal matrix of 600 values. Its side stands above the space limit.
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    var i: i64 = 0;
    while i < 600 {
        ri.push(i as f64);
        ci.push(i as f64);
        vv.push(1.0 + (i as f64) * 0.01);
        i += 1;
    }
    let s, f0 = scientific::csr_from_triplets(600, 600, ri, ci, vv);
    assert(f0.kind == scientific::F_NONE);
    // A count of values at the limit of the space gives a fault of limit.
    let ua, sva, vta, fa = scientific::svds(s, scientific::MAX_SVDS_SPACE);
    assert(fa.kind == scientific::F_LIMIT);
    // A space above the limit of the space gives a fault of limit.
    let ub, svb, vtb, fb = scientific::svds(s, 1, "LM", 0.0, [], scientific::MAX_SVDS_SPACE + 38);
    assert(fb.kind == scientific::F_LIMIT);
    // A count of rebuilds above the limit gives a fault of limit.
    let uc, svc, vtc, fc = scientific::svds(s, 1, "LM", 0.0, [], 0, scientific::MAX_KRYLOV_ITER + 1);
    assert(fc.kind == scientific::F_LIMIT);
    // A run that does not converge within the rebuilds gives a fault of limit.
    let ud, svd, vtd, fd = scientific::svds(s, 1, "LM", 0.0, [], 0, 1);
    assert(fd.kind == scientific::F_LIMIT);
}

// A matrix with two large values and a cluster of equal small ones breaks the
// iteration down. The vectors must stay exact and the values must stay right.
fn test_sparse_svds_cluster() {
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < 30 {
        var x = 1.0;
        if i == 0 { x = 1000.0; }
        if i == 1 { x = 500.0; }
        v.push(x);
        i += 1;
    }
    let s = svds_diag(v);
    var ncv: i64 = 0;
    while ncv < 29 {
        let ua, sva, vta, fa = scientific::svds(s, 2, "LM", 0.0, [], ncv);
        assert(fa.kind == scientific::F_NONE);
        assert(common::mathshim_abs(sva.data[0] - 500.0) <= SVDS_TOL * 500.0);
        assert(common::mathshim_abs(sva.data[1] - 1000.0) <= SVDS_TOL * 1000.0);
        assert(svds_orth_cols(ua.data, 30, 2) <= SVDS_TOL);
        assert(svds_orth_rows(vta.data, 2, 30) <= SVDS_TOL);
        assert(svds_residual(s, ua.data, sva.data, vta.data, 2) <= SVDS_TOL * 1000.0);
        if ncv == 0 { ncv = 3; } else { ncv += 5; }
    }
}

// The faults of the hostile matrix and of the hostile numbers.
fn test_sparse_svds_hostile() {
    let s = svds_a85();
    // A tol of NaN and a mode below zero.
    let ua, sva, vta, fa = scientific::svds(s, 2, "LM", 0.0 / 0.0);
    assert(fa.kind == scientific::F_VALUE && ua.data.len == 0);
    let ub, svb, vtb, fb = scientific::svds(s, 2, "LM", 0.0, [], 0, 0, -1);
    assert(fb.kind == scientific::F_RANGE);
    // A side above the limit of the sides.
    var b1: f64[>..] = [0.0, 0.0];
    let wide = scientific::Csr { rows: 1, cols: scientific::MAX_SPARSE_SIDE + 1, nnz: 0, buf: copy(b1) };
    let uc, svc, vtc, fc = scientific::svds(wide, 1);
    assert(fc.kind == scientific::F_RANGE);
    // A count of stored values above the limit.
    var b2: f64[>..] = [0.0, 0.0];
    let many = scientific::Csr { rows: 1, cols: 3, nnz: scientific::MAX_SPARSE_NNZ + 1, buf: copy(b2) };
    let ud, svd, vtd, fd = scientific::svds(many, 1);
    assert(fd.kind == scientific::F_LIMIT);
}

// A matrix of zeros gives the zero values and orthonormal vectors.
fn test_sparse_svds_zero_matrix() {
    let s = svds_csr(3, 2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let ua, sva, vta, fa = scientific::svds(s, 1);
    assert(fa.kind == scientific::F_NONE);
    assert(sva.data.len == 1 && sva.data[0] == 0.0);
    assert(svds_orth_cols(ua.data, 3, 1) <= SVDS_TOL);
    assert(svds_orth_rows(vta.data, 1, 2) <= SVDS_TOL);
}

// A matrix of any finite scale gives the scaled values.
fn test_sparse_svds_scales() {
    let s = svds_a85();
    let ua, sva, vta, fa = scientific::svds(s, 2);
    assert(fa.kind == scientific::F_NONE);
    // A matrix of a small scale gives the scaled values.
    var small: f64[>..] = [];
    var i: i64 = 0;
    while i < 40 { small.push(sparse_svds_pins::A85[i] * 1.0e-300); i += 1; }
    let s2 = svds_csr(8, 5, small);
    let ub, svb, vtb, fb = scientific::svds(s2, 2);
    assert(fb.kind == scientific::F_NONE);
    assert(common::mathshim_abs(svb.data[0] - sva.data[0] * 1.0e-300) <= 1.0e-300);
    assert(common::mathshim_abs(svb.data[1] - sva.data[1] * 1.0e-300) <= 1.0e-300);
    // A matrix of a large scale gives the scaled values.
    var large: f64[>..] = [];
    i = 0;
    while i < 40 { large.push(sparse_svds_pins::A85[i] * 1.0e300); i += 1; }
    let s3 = svds_csr(8, 5, large);
    let uc, svc, vtc, fc = scientific::svds(s3, 2);
    assert(fc.kind == scientific::F_NONE);
    assert(common::mathshim_abs(svc.data[0] - sva.data[0] * 1.0e300) <= 1.0e292);
    assert(common::mathshim_abs(svc.data[1] - sva.data[1] * 1.0e300) <= 1.0e292);
}

// The call never builds the dense matrix of a wide matrix.
fn test_sparse_svds_sparse_only() {
    // A diagonal matrix of 20000 values. Its dense matrix holds more values
    // than the limit of the package.
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < 20000 {
        var x = 1.0;
        if i == 0 { x = 1000.0; }
        if i == 1 { x = 500.0; }
        v.push(x);
        i += 1;
    }
    let s = svds_diag(v);
    let ua, sva, vta, fa = scientific::svds(s, 2);
    assert(fa.kind == scientific::F_NONE);
    svds_match(sva.data, sparse_svds_pins::BIG_LM2);
}
