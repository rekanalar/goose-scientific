// The tests of the group schur_funm. Run them with SCIENTIFIC_SUITES=schur_funm.
namespace schur_funm_tests;

import scientific from "../..";
import os;
import common;

// schur and funm and ldl are pinned against SciPy 1.18.1.
fn test_schur_funm_ldl() {
    let A = scientific::from_rows(5, 5, [4.0, 1.0, -2.0, 2.0, 0.0, 1.0, 3.0, 1.0, -1.0, 2.0, -2.0, 1.0, 5.0, 1.0, -1.0, 2.0, -1.0, 1.0, 3.0, 1.0, 0.0, 2.0, -1.0, 1.0, 4.0]);
    let ta, za, fa = scientific::schur(A);
    assert(fa.kind == scientific::F_NONE);
    // The change of the basis is orthogonal.
    let zta = scientific::transpose(za);
    let ga = scientific::matmul(zta, za);
    var orth: f64 = 0.0;
    var i: i64 = 0;
    while i < ga.shape[0] {
        var j: i64 = 0;
        while j < ga.shape[1] {
            let g = if j == i { 1.0 } else { 0.0 };
            var d = ga.data[i * ga.shape[1] + j] - g;
            if d < 0.0 { d = -d; }
            if d > orth { orth = d; }
            j += 1;
        }
        i += 1;
    }
    assert(orth < 1.0e-13);
    // The form rebuilds the matrix.
    let ra = scientific::matmul(za, ta);
    let ra2 = scientific::matmul(ra, zta);
    assert(common::maxdiff(ra2, A.data) < 1.0e-13);
    // The matrix is symmetric, so the form is diagonal and holds the
    // eigenvalues.
    var low: f64 = 0.0;
    i = 1;
    while i < ta.shape[0] {
        var d = ta.data[i * ta.shape[1] + (i - 1)];
        if d < 0.0 { d = -d; }
        if d > low { low = d; }
        i += 1;
    }
    assert(low == 0.0);
    let eva = scientific::eigvalsh(A);
    var diag_ok = true;
    i = 0;
    while i < 5 {
        var found = false;
        var j: i64 = 0;
        while j < 5 {
            if !found && common::loose(ta.data[i * 5 + i], eva.data[j]) { found = true; }
            j += 1;
        }
        if !found { diag_ok = false; }
        i += 1;
    }
    assert(diag_ok);
    // B holds a pair of complex values, and the form keeps a block of
    // two rows for it.
    let B = scientific::from_rows(4, 4, [1.0, 2.0, 0.0, 3.0, 4.0, 1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 2.0, 2.0, 0.0, 4.0, 1.0]);
    let tb, zb, fb = scientific::schur(B);
    assert(fb.kind == scientific::F_NONE);
    let ztb = scientific::transpose(zb);
    let rb = scientific::matmul(zb, tb);
    let rb2 = scientific::matmul(rb, ztb);
    assert(common::maxdiff(rb2, B.data) < 1.0e-13);
    // Below the subdiagonal the form is zero.
    var low2: f64 = 0.0;
    i = 0;
    while i < tb.shape[0] {
        var j: i64 = 0;
        while j < i - 1 {
            var d = tb.data[i * 4 + j];
            if d < 0.0 { d = -d; }
            if d > low2 { low2 = d; }
            j += 1;
        }
        i += 1;
    }
    assert(low2 == 0.0);
    // The block of two rows holds the values 1 plus and minus the
    // square root of two.
    let tr = tb.data[2 * 4 + 2] + tb.data[3 * 4 + 3];
    let det = tb.data[2 * 4 + 2] * tb.data[3 * 4 + 3] - tb.data[2 * 4 + 3] * tb.data[3 * 4 + 2];
    assert(common::loose(tr, 2.0) && common::loose(det, 3.0));
    // funm of exp and cos matches expm and cosm.
    let fa0, faf = scientific::funm(A) { x => mathshim::power_e(x) };
    assert(faf.kind == scientific::F_NONE && common::maxdiff(fa0, [434.58822489091307, 73.88495387687904, -407.5164308328381, 169.17283558549684, 224.6102821473738, 73.88495387687907, 117.08605144571604, -27.88445872891634, 30.18979999344893, 140.13141871107828, -407.51643083283807, -27.88445872891626, 524.0714715355521, -91.10898574982119, -202.02920946801962, 169.17283558549673, 30.189799993448926, -91.1089857498211, 115.40237007044733, 90.13466826248785, 224.61028214737374, 140.13141871107823, -202.02920946801967, 90.13466826248786, 249.96568795400466]) < 1.0e-6);
    let fc0, fcf = scientific::funm(A) { x => mathshim::cosine(x) };
    assert(fcf.kind == scientific::F_NONE && common::maxdiff(fc0, [0.15704861258425307, -0.31309477452687295, -0.305029467260153, 0.1971229086803145, 0.4862098577314571, -0.31309477452687307, 0.0763608808454167, -0.016327546660809586, 0.41569737623326863, 0.4941064682994968, -0.305029467260153, -0.016327546660809905, 0.5397328805261301, -0.06275755414680262, 0.14701263854926788, 0.19712290868031437, 0.4156973762332685, -0.06275755414680259, 0.03501849847974753, -0.16205918215711979, 0.48620985773145714, 0.4941064682994968, 0.1470126385492677, -0.16205918215711962, 0.1417857372232823]) < 1.0e-12);
    // funm of sin on a matrix with real values matches SciPy.
    let C = scientific::from_rows(3, 3, [2.0, 1.0, 0.5, 0.3, 5.0, 1.0, 0.1, 0.2, 7.0]);
    let fs0, fsf = scientific::funm(C) { x => mathshim::sine(x) };
    assert(fsf.kind == scientific::F_NONE && common::maxdiff(fs0, [0.8988026555927285, -0.5607175842086072, 0.24946869705817168, -0.14919582693367267, -0.8241060577378951, 0.8113997595198518, 0.011854842753815629, 0.18129940023287966, 0.6492549387175257]) < 1.0e-12);
    // Two equal values on the diagonal are a fault.
    let eye = scientific::eye(3);
    let b0, bf = scientific::funm(eye) { x => mathshim::power_e(x) };
    assert(bf.kind == scientific::F_VALUE);
    // A pair of complex values is a fault of funm.
    let bB0, bBf = scientific::funm(B) { x => mathshim::power_e(x) };
    assert(bBf.kind == scientific::F_VALUE);
    // ldl against SciPy on a matrix that needs no pivoting.
    let S = scientific::from_rows(5, 5, [10.0, 1.0, -2.0, 2.0, 0.0, 1.0, 9.0, 1.0, -1.0, 2.0, -2.0, 1.0, 11.0, 1.0, -1.0, 2.0, -1.0, 1.0, 9.0, 1.0, 0.0, 2.0, -1.0, 1.0, 10.0]);
    let lo, di, fl = scientific::ldl(S);
    assert(fl.kind == scientific::F_NONE);
    assert(common::maxdiff(lo, [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, -0.2, 0.1348314606741573, 1.0, 0.0, 0.0, 0.2, -0.1348314606741573, 0.1496232508073197, 1.0, 0.0, 0.0, 0.2247191011235955, -0.12163616792249732, 0.17790606140120704, 1.0]) < 1.0e-14);
    assert(common::maxdiff(di, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.9, 0.0, 0.0, 0.0, 0.0, 0.0, 10.438202247191011, 0.0, 0.0, 0.0, 0.0, 0.0, 8.204520990312163, 0.0, 0.0, 0.0, 0.0, 0.0, 9.136447126738389]) < 1.0e-14);
    let lt = scientific::transpose(lo);
    let ld = scientific::matmul(lo, di);
    let rec = scientific::matmul(ld, lt);
    assert(common::maxdiff(rec, S.data) < 1.0e-13);
    // A matrix that needs pivoting is a fault.
    let P0 = scientific::from_rows(2, 2, [0.0, 1.0, 1.0, 0.0]);
    let pl, pd, pf = scientific::ldl(P0);
    assert(pf.kind == scientific::F_VALUE);
    // A matrix that is not symmetric is a fault.
    let ns = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let nl, nd, nf = scientific::ldl(ns);
    assert(nf.kind == scientific::F_VALUE);
    // A matrix that is not square is a fault of schur.
    let rect = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rl, rz, rf = scientific::schur(rect);
    assert(rf.kind == scientific::F_SHAPE);
}
