// The tests of the group geneig. Run them with SCIENTIFIC_SUITES=geneig.
namespace geneig_tests;

import scientific from "../..";
import os;
import common;
import common_2;
import common_3;

// test_ceig_pair pins the eigenvalues and the vectors of SciPy and checks the
// eigen equation of random pairs.
fn test_ceig_pair() {
    let a1 = common_3::gm(2, [1.0, 2.0, 3.0, 4.0], [1.0, 0.0, -2.0, 1.0]);
    let b1 = common_3::gm(2, [2.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, -1.0]);
    let al, be, v, f = scientific::ceig_pair(a1, b1);
    assert(f.kind == scientific::F_NONE);
    assert(al.shape[0] == 2 && v.shape[0] == 2 && v.shape[1] == 2);
    let k0 = common_3::pair_find(al, be, 2, -0.7499999999999999, 0.7499999999999998);
    let k1 = common_3::pair_find(al, be, 2, 2.9999999999999987, 0.9999999999999992);
    assert(k0 >= 0 && k1 >= 0 && k0 != k1);
    // The columns of the vectors match the pinned columns.
    let pr = [0.7453559924999299, 0.35478743759344955, -0.596284793999944, 0.7687061147858075];
    let pi = [0.0, -0.5321811563901746, 0.2981423969999719, 0.0];
    var i: i64 = 0;
    while i < 2 {
        assert(common_3::gclose(v.data[2 * (i * 2 + k0)], pr[i * 2], 1.0e-10));
        assert(common_3::gclose(v.data[2 * (i * 2 + k0) + 1], pi[i * 2], 1.0e-10));
        assert(common_3::gclose(v.data[2 * (i * 2 + k1)], pr[i * 2 + 1], 1.0e-10));
        assert(common_3::gclose(v.data[2 * (i * 2 + k1) + 1], pi[i * 2 + 1], 1.0e-10));
        i += 1;
    }
    assert(common_3::pair_residual(a1, b1, al, be, v) < 1.0e-12);
    // ceigvals_pair gives alpha divided by beta.
    let w, fw = scientific::ceigvals_pair(a1, b1);
    assert(fw.kind == scientific::F_NONE && w.shape[0] == 2);
    // The pair of side 3.
    let a2 = common_3::gm(3, [1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 1.0], [0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, -2.0]);
    let b2 = common_3::gm(3, [2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]);
    let al2, be2, v2, f2 = scientific::ceig_pair(a2, b2);
    assert(f2.kind == scientific::F_NONE);
    assert(common_3::pair_find(al2, be2, 3, -0.09375557560673803, -1.1446409764073708) >= 0);
    assert(common_3::pair_find(al2, be2, 3, 0.19926103939419426, 0.08798889161740063) >= 0);
    assert(common_3::pair_find(al2, be2, 3, 2.1802088219268305, 1.0566520847899712) >= 0);
    assert(common_3::pair_residual(a2, b2, al2, be2, v2) < 1.0e-12);
    // A singular B gives an infinite eigenvalue.
    let a3 = common_3::gm(2, [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]);
    let b3 = common_3::gm(2, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]);
    let w3, f3 = scientific::ceigvals_pair(a3, b3);
    assert(f3.kind == scientific::F_NONE);
    var finite = 0;
    var infinite = 0;
    i = 0;
    while i < 2 {
        if w3.data[2 * i] == 1.0 / 0.0 {
            assert(w3.data[2 * i + 1] == 0.0);
            infinite += 1;
        } else {
            assert(common_3::gclose(w3.data[2 * i], -0.5, 1.0e-12));
            finite += 1;
        }
        i += 1;
    }
    assert(finite == 1 && infinite == 1);
    // B equal to the unit matrix gives the eigenvalues of ceig.
    let eye = common_3::gm(2, [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]);
    let wp, fp = scientific::ceigvals_pair(a1, eye);
    let wc, fc = scientific::ceigvals(a1);
    assert(fp.kind == scientific::F_NONE && fc.kind == scientific::F_NONE);
    var matched = 0;
    i = 0;
    while i < 2 {
        var j: i64 = 0;
        while j < 2 {
            if common_3::gclose(wp.data[2 * i], wc.data[2 * j], 1.0e-12) && common_3::gclose(wp.data[2 * i + 1], wc.data[2 * j + 1], 1.0e-12) { matched += 1; }
            j += 1;
        }
        i += 1;
    }
    assert(matched == 2);
    // Random pairs of several sides.
    var rng = scientific::default_rng(298);
    var round: i64 = 0;
    while round < 20 {
        var n: i64 = 4;
        if round == 20 { n = 4; }
        let ra = common_2::random_complex(rng, n, n);
        let rb = common_2::random_complex(rng, n, n);
        let ral, rbe, rv, rf = scientific::ceig_pair(ra, rb);
        assert(rf.kind == scientific::F_NONE);
        assert(common_3::pair_residual(ra, rb, ral, rbe, rv) < 1.0e-10);
        round += 1;
    }
    var sides = [1, 2, 3, 6];
    i = 0;
    while i < 4 {
        let n = sides[i];
        let ra = common_2::random_complex(rng, n, n);
        let rb = common_2::random_complex(rng, n, n);
        let ral, rbe, rv, rf = scientific::ceig_pair(ra, rb);
        assert(rf.kind == scientific::F_NONE);
        assert(common_3::pair_residual(ra, rb, ral, rbe, rv) < 1.0e-10);
        i += 1;
    }
    return;
}

// test_ceig_pair_scale checks a pair near the limits of the number format.
fn test_ceig_pair_scale() {
    var scales = [1.0e300, 1.0e-300];
    var s: i64 = 0;
    while s < 2 {
        let k = scales[s];
        let a1 = common_3::gm(2, [1.0 * k, 2.0 * k, 3.0 * k, 4.0 * k], [1.0 * k, 0.0, -2.0 * k, 1.0 * k]);
        let b1 = common_3::gm(2, [2.0 * k, 0.0, 0.0, 1.0 * k], [0.0, 1.0 * k, 0.0, -1.0 * k]);
        let al, be, v, f = scientific::ceig_pair(a1, b1);
        assert(f.kind == scientific::F_NONE);
        let w, fw = scientific::ceigvals_pair(a1, b1);
        assert(fw.kind == scientific::F_NONE);
        assert(common_3::pair_find(al, be, 2, -0.7499999999999999, 0.7499999999999998) >= 0);
        assert(common_3::pair_find(al, be, 2, 2.9999999999999987, 0.9999999999999992) >= 0);
        s += 1;
    }
    return;
}

// test_eig_pair_real checks that the real calls equal the complex calls.
fn test_eig_pair_real() {
    let a = common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let b = common::mx(2, 2, [2.0, 1.0, 0.0, 3.0]);
    let al, be, v, f = scientific::eig_pair(a, b);
    assert(f.kind == scientific::F_NONE);
    let ca = common_3::gm(2, [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]);
    let cb = common_3::gm(2, [2.0, 1.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0]);
    let cal, cbe, cv, cf = scientific::ceig_pair(ca, cb);
    assert(cf.kind == scientific::F_NONE);
    assert(common_3::gnorm(al.data) > 0.0);
    var i: i64 = 0;
    while i < al.data.len {
        assert(common_3::gclose(al.data[i], cal.data[i], 1.0e-12));
        assert(common_3::gclose(be.data[i], cbe.data[i], 1.0e-12));
        i += 1;
    }
    i = 0;
    while i < v.data.len {
        assert(common_3::gclose(v.data[i], cv.data[i], 1.0e-12));
        i += 1;
    }
    let w, fw = scientific::eigvals_pair(a, b);
    let cw, fcw = scientific::ceigvals_pair(ca, cb);
    assert(fw.kind == scientific::F_NONE && fcw.kind == scientific::F_NONE);
    i = 0;
    while i < w.data.len {
        assert(common_3::gclose(w.data[i], cw.data[i], 1.0e-12));
        i += 1;
    }
    return;
}

// test_ceigh_pair pins the values of SciPy and checks random definite pairs.
fn test_ceigh_pair() {
    let da = common_3::gm(2, [2.0, 1.0, 1.0, 3.0], [0.0, -1.0, 1.0, 0.0]);
    let db = common_3::gm(2, [4.0, 0.0, 0.0, 2.0], [0.0, 1.0, -1.0, 0.0]);
    let w, v, f = scientific::ceigh_pair(da, db, "L");
    assert(f.kind == scientific::F_NONE);
    assert(common_3::gclose(w.data[0], 0.24569858724564023, 1.0e-12));
    assert(common_3::gclose(w.data[1], 2.3257299841829298, 1.0e-12));
    let pr = [0.41508017227708693, 0.09697678117867867, -0.16546269024986568, 0.7082136686698423];
    let pi = [0.0, -0.3225185889354786, -0.20611663948612066, 0.0];
    var i: i64 = 0;
    while i < 4 {
        assert(common_3::gclose(v.data[2 * i], pr[i], 1.0e-10));
        assert(common_3::gclose(v.data[2 * i + 1], pi[i], 1.0e-10));
        i += 1;
    }
    common_3::check_definite(da, db, w, v);
    // The upper triangle gives the same values, in either case.
    let wu, vu, fu = scientific::ceigh_pair(da, db, "U");
    let wl, vl, fl = scientific::ceigh_pair(da, db, "l");
    assert(fu.kind == scientific::F_NONE && fl.kind == scientific::F_NONE);
    assert(common_3::gclose(wu.data[0], w.data[0], 1.0e-12) && common_3::gclose(wl.data[1], w.data[1], 1.0e-12));
    let wv, fv = scientific::ceigvalsh_pair(da, db);
    assert(fv.kind == scientific::F_NONE && common_3::gclose(wv.data[1], 2.3257299841829298, 1.0e-12));
    // The pair of side 3.
    let a2 = common_3::gm(3, [2.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0], [0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0]);
    let b2 = common_3::gm(3, [3.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0]);
    let w2, v2, f2 = scientific::ceigh_pair(a2, b2);
    assert(f2.kind == scientific::F_NONE);
    assert(common_3::gclose(w2.data[0], 0.135548598119074, 1.0e-12));
    assert(common_3::gclose(w2.data[1], 0.48438238014813345, 1.0e-12));
    assert(common_3::gclose(w2.data[2], 1.791833727615146, 1.0e-12));
    common_3::check_definite(a2, b2, w2, v2);
    // A repeated eigenvalue: A equal to two times B.
    let w3, v3, f3 = scientific::ceigh_pair(db, db);
    assert(f3.kind == scientific::F_NONE);
    assert(common_3::gclose(w3.data[0], 1.0, 1.0e-12) && common_3::gclose(w3.data[1], 1.0, 1.0e-12));
    common_3::check_definite(db, db, w3, v3);
    // Random definite pairs.
    var rng = scientific::default_rng(299);
    var round: i64 = 0;
    while round < 20 {
        let ra = common_3::random_hermitian(rng, 4);
        let rb = common_3::random_definite(rng, 4);
        let rw, rv, rf = scientific::ceigh_pair(ra, rb);
        assert(rf.kind == scientific::F_NONE);
        common_3::check_definite(ra, rb, rw, rv);
        round += 1;
    }
    // B that is not positive definite gives a fault.
    let neg = common_3::gm(2, [1.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]);
    let wn, vn, fn1 = scientific::ceigh_pair(da, neg);
    assert(fn1.kind == scientific::F_VALUE);
    let wz, fz = scientific::ceigvalsh_pair(da, common_3::gm(2, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]));
    assert(fz.kind == scientific::F_VALUE);
    return;
}

// test_eigh_pair_real checks that the real calls give the complex values.
fn test_eigh_pair_real() {
    let a = common::mx(2, 2, [2.0, 1.0, 1.0, 3.0]);
    let b = common::mx(2, 2, [4.0, 1.0, 1.0, 2.0]);
    let w, v, f = scientific::eigh_pair(a, b);
    assert(f.kind == scientific::F_NONE);
    let ca = common_3::gm(2, [2.0, 1.0, 1.0, 3.0], [0.0, 0.0, 0.0, 0.0]);
    let cb = common_3::gm(2, [4.0, 1.0, 1.0, 2.0], [0.0, 0.0, 0.0, 0.0]);
    let cw, cv, cf = scientific::ceigh_pair(ca, cb);
    assert(cf.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < 2 {
        assert(common_3::gclose(w.data[i], cw.data[i], 1.0e-12));
        i += 1;
    }
    i = 0;
    while i < 4 {
        assert(common_3::gclose(v.data[i], cv.data[2 * i], 1.0e-12));
        assert(cv.data[2 * i + 1] == 0.0);
        i += 1;
    }
    let wv, fv = scientific::eigvalsh_pair(a, b, "U");
    assert(fv.kind == scientific::F_NONE && common_3::gclose(wv.data[0], w.data[0], 1.0e-12));
    return;
}

// test_ceigh_pair_hostile covers the faults of the Hermitian pair calls.
fn test_ceigh_pair_hostile() {
    let m = common_3::gm(2, [2.0, 1.0, 1.0, 3.0], [0.0, -1.0, 1.0, 0.0]);
    let w1, v1, g1 = scientific::ceigh_pair(m, m, "X");
    assert(g1.kind == scientific::F_VALUE);
    let w2, g2 = scientific::ceigvalsh_pair(m, m, "");
    assert(g2.kind == scientific::F_VALUE);
    let one = scientific::CArr { ndim: 1, shape: [4], data: [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0] };
    let w3, v3, g3 = scientific::ceigh_pair(one, m);
    assert(g3.kind == scientific::F_SHAPE);
    let short = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0] };
    let w4, v4, g4 = scientific::ceigh_pair(m, short);
    assert(g4.kind == scientific::F_SHAPE);
    let tiny = common_3::gm(1, [1.0], [0.0]);
    let w5, v5, g5 = scientific::ceigh_pair(m, tiny);
    assert(g5.kind == scientific::F_SHAPE);
    let nan = 0.0 / 0.0;
    let bad = scientific::CArr { ndim: 2, shape: [1, 1], data: [nan, 0.0] };
    let w6, v6, g6 = scientific::ceigh_pair(tiny, bad);
    assert(g6.kind == scientific::F_VALUE);
    var big: f64[>..] = [];
    var bi: i64 = 0;
    while bi < 2 * 257 * 257 { big.push(0.0); bi += 1; }
    let wide = scientific::CArr { ndim: 2, shape: [257, 257], data: copy(big) };
    let w7, v7, g7 = scientific::ceigh_pair(wide, wide);
    assert(g7.kind == scientific::F_LIMIT);
    let w8, g8 = scientific::ceigvalsh_pair(wide, wide);
    assert(g8.kind == scientific::F_LIMIT);
    let ra = common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let w9, v9, g9 = scientific::eigh_pair(ra, common::mx(1, 1, [1.0]));
    assert(g9.kind == scientific::F_SHAPE);
    return;
}
