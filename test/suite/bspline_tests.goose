// The tests of the group bspline. Run them with SCIENTIFIC_SUITES=bspline.
namespace bspline_tests;

import scientific from "../..";
import os;
import balance_tests;
import banded_tests;
import common;
import cossin_tests;
import delaunay_tests;
import qrinsert_tests;
import qrtools_tests;
import qrupdate_tests;
import riccati_tests;
import schur_cplx_tests;
import schur_stacks_tests;
import solve_ivp_tests;
import special_matrices_tests;
import splrep_smoothing_tests;
import structured_tests;

// The B-spline functions are pinned against SciPy for the degrees 1,
// 2, 3 and 5, with derivatives, extrapolation and the faults.
fn test_bspline() {
    let x = [0.0, 1.0, 2.5, 4.0, 5.0, 6.5, 8.0];
    let y = [1.0, 3.0, 2.0, 5.0, 1.0, 0.5, 2.0];
    let pts = [0.0, 0.5, 1.7, 4.4, 8.0, 9.0, -1.0];
    let ks = [1, 2, 3, 5];
    let b1, f1 = scientific::make_interp_spline(x, y, 1);
    assert(f1.kind == scientific::F_NONE);
    let v1, fv1 = scientific::splev(pts, b1);
    let e1 = [1.0, 2.0, 2.533333333333333, 3.3999999999999986, 2.0, 2.9999999999999996, -1.0];
    let d1, fd1 = scientific::splev([0.5, 4.4], b1, 1);
    let ed1 = [2.0, -4.0];
    let s1, fs1 = scientific::splev([0.5, 4.4], b1, 2);
    let es1 = [0.0, 0.0];
    var i1: i64 = 0;
    while i1 < 7 { assert(common::mathshim_abs(v1.data[i1] - e1[i1]) < 1.0e-8 * (1.0 + common::mathshim_abs(e1[i1]))); i1 += 1; }
    assert(common::mathshim_abs(d1.data[0] - ed1[0]) < 1.0e-8 && common::mathshim_abs(d1.data[1] - ed1[1]) < 1.0e-8);
    assert(common::mathshim_abs(s1.data[0] - es1[0]) < 1.0e-7 && common::mathshim_abs(s1.data[1] - es1[1]) < 1.0e-7);
    let b2, f2 = scientific::make_interp_spline(x, y, 2);
    assert(f2.kind == scientific::F_NONE);
    let v2, fv2 = scientific::splev(pts, b2);
    let e2 = [1.0, 2.4070264444072773, 2.4625541246213594, 3.707898385088548, 2.0, 3.394621917451822, -4.2562115552582185];
    let d2, fd2 = scientific::splev([0.5, 4.4], b2, 1);
    let ed2 = [2.0, -4.810139190871756];
    let s2, fs2 = scientific::splev([0.5, 4.4], b2, 2);
    let es2 = [-3.2562115552582185, -7.899425767965655];
    var i2: i64 = 0;
    while i2 < 7 { assert(common::mathshim_abs(v2.data[i2] - e2[i2]) < 1.0e-8 * (1.0 + common::mathshim_abs(e2[i2]))); i2 += 1; }
    assert(common::mathshim_abs(d2.data[0] - ed2[0]) < 1.0e-8 && common::mathshim_abs(d2.data[1] - ed2[1]) < 1.0e-8);
    assert(common::mathshim_abs(s2.data[0] - es2[0]) < 1.0e-7 && common::mathshim_abs(s2.data[1] - es2[1]) < 1.0e-7);
    let b3, f3 = scientific::make_interp_spline(x, y, 3);
    assert(f3.kind == scientific::F_NONE);
    let v3, fv3 = scientific::splev(pts, b3);
    let e3 = [1.0, 2.7759259259259257, 2.1610370370370373, 3.7092444444444412, 2.0, -1.76954732510288, -10.26296296296296];
    let d3, fd3 = scientific::splev([0.5, 4.4], b3, 1);
    let ed3 = [1.74537037037037, -4.399185185185186];
    let s3, fs3 = scientific::splev([0.5, 4.4], b3, 2);
    let es3 = [-6.207407407407406, -3.6718518518518444];
    var i3: i64 = 0;
    while i3 < 7 { assert(common::mathshim_abs(v3.data[i3] - e3[i3]) < 1.0e-8 * (1.0 + common::mathshim_abs(e3[i3]))); i3 += 1; }
    assert(common::mathshim_abs(d3.data[0] - ed3[0]) < 1.0e-8 && common::mathshim_abs(d3.data[1] - ed3[1]) < 1.0e-8);
    assert(common::mathshim_abs(s3.data[0] - es3[0]) < 1.0e-7 && common::mathshim_abs(s3.data[1] - es3[1]) < 1.0e-7);
    let b5, f5 = scientific::make_interp_spline(x, y, 5);
    assert(f5.kind == scientific::F_NONE);
    let v5, fv5 = scientific::splev(pts, b5);
    let e5 = [1.0, 4.084333544764113, 1.0722625024332686, 3.8503072596962076, 2.0, -62.79030311967932, -45.52148214932095];
    let d5, fd5 = scientific::splev([0.5, 4.4], b5, 1);
    let ed5 = [0.39162517302690636, -3.9602456578765066];
    let s5, fs5 = scientific::splev([0.5, 4.4], b5, 2);
    let es5 = [-15.832241676460812, -4.525612894168046];
    var i5: i64 = 0;
    while i5 < 7 { assert(common::mathshim_abs(v5.data[i5] - e5[i5]) < 1.0e-8 * (1.0 + common::mathshim_abs(e5[i5]))); i5 += 1; }
    assert(common::mathshim_abs(d5.data[0] - ed5[0]) < 1.0e-8 && common::mathshim_abs(d5.data[1] - ed5[1]) < 1.0e-8);
    assert(common::mathshim_abs(s5.data[0] - es5[0]) < 1.0e-7 && common::mathshim_abs(s5.data[1] - es5[1]) < 1.0e-7);
    // The knots and the coefficients of the cubic match SciPy.
    let kn = [0.0, 0.0, 0.0, 0.0, 2.5, 4.0, 5.0, 8.0, 8.0, 8.0, 8.0];
    let co = [1.0, 5.677469135802468, -2.276851851851851, 7.829629629629628, -4.521604938271605, 2.9796296296296294, 2.0];
    let bk = scientific::bs_knots_of(b3);
    let bc = scientific::bs_coefs_of(b3);
    assert(bk.len == kn.len && bc.len == co.len);
    var q: i64 = 0;
    while q < kn.len { assert(common::mathshim_abs(bk[q] - kn[q]) < 1.0e-12); q += 1; }
    q = 0;
    while q < co.len { assert(common::mathshim_abs(bc[q] - co[q]) < 1.0e-8); q += 1; }
    // A spline built from knots and coefficients gives the same values.
    let bb, fb = scientific::bspline(bk, bc, 3);
    assert(fb.kind == scientific::F_NONE);
    let vb, fvb = scientific::bspline_eval(bb, pts);
    assert(common::mathshim_abs(vb.data[2] - v3.data[2]) < 1.0e-12);
    // A derivative above the degree is zero.
    let z, fz = scientific::splev([1.0], b1, 2);
    assert(common::mathshim_abs(z.data[0]) < 1.0e-12);
    let sr, fsr = scientific::splrep(x, y, 3);
    assert(fsr.kind == scientific::F_NONE);
    splrep_smoothing_tests::test_splrep_smoothing();
    delaunay_tests::test_griddata();
    solve_ivp_tests::test_solve_ivp_jac();
    special_matrices_tests::test_special_matrices();
    structured_tests::test_structured();
    banded_tests::test_banded();
    riccati_tests::test_riccati();
    qrtools_tests::test_qrtools();
    qrupdate_tests::test_qrupdate();
    qrinsert_tests::test_qrinsert();
    schur_cplx_tests::test_schur_cplx();
    schur_stacks_tests::test_schur_stacks();
    cossin_tests::test_cossin();
    balance_tests::test_balance();
    let e1, g1 = scientific::make_interp_spline([0.0, 1.0], [1.0, 2.0], 3);
    assert(g1.kind == scientific::F_SHAPE);
    let e2, g2 = scientific::make_interp_spline([0.0, 1.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0], 3);
    assert(g2.kind == scientific::F_VALUE);
    let e3, g3 = scientific::make_interp_spline(x, y, 21);
    assert(g3.kind == scientific::F_RANGE);
    let e4, g4 = scientific::bspline([0.0, 0.0, 1.0], [1.0, 2.0], 1);
    assert(g4.kind == scientific::F_SHAPE);
    let e5, g5 = scientific::bspline([0.0, 1.0, 0.5, 1.0], [1.0, 2.0], 1);
    assert(g5.kind == scientific::F_VALUE);
    let e6, g6 = scientific::splev([], b3);
    assert(g6.kind == scientific::F_SHAPE);
    let e7, g7 = scientific::splev([1.0], b3, -1);
    assert(g7.kind == scientific::F_RANGE);
}
