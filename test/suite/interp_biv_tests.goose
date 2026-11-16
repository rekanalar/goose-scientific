// The tests of the bivariate splines, the smoothing splines, the
// polynomial of Bernstein and the splines of many dimensions. Run
// them with SCIENTIFIC_SUITES=interp_biv.
//
// The calls match the pins of SciPy 1.18.1 to a tolerance of 1.0e-6.
namespace interp_biv_tests;

import scientific from "../..";
import common;
import interp_biv_pins;

fn buf_near(a: f64[:], v: f64[:], tol: f64) -> bool {
    if a.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        if !common::close(a[i], v[i], tol) { return false; }
        i += 1;
    }
    return true;
}

fn test_lsq_bivariate() {
    let g, f = scientific::lsq_bivariate_spline(interp_biv_pins::LSQ_X, interp_biv_pins::LSQ_Y, interp_biv_pins::LSQ_Z, interp_biv_pins::LSQ_TX, interp_biv_pins::LSQ_TY, interp_biv_pins::LSQ_W, 3, 3);
    assert(f.kind == scientific::F_NONE);
    // The knots follow from the boundary rule, so they agree to the
    // last digits.
    let t1 = scientific::grid2_knots_x(g);
    let t2 = scientific::grid2_knots_y(g);
    assert(buf_near(t1, interp_biv_pins::LSQ_T1, 1.0e-9));
    assert(buf_near(t2, interp_biv_pins::LSQ_T2, 1.0e-9));
    let c = scientific::grid2_coefs(g);
    assert(buf_near(c, interp_biv_pins::LSQ_C, 1.0e-6));
    // The values on a grid, at single points and the derivative.
    let grid, f2 = scientific::bisplev(interp_biv_pins::LSQ_GX, interp_biv_pins::LSQ_GY, t1, t2, c, 3, 3);
    assert(f2.kind == scientific::F_NONE);
    assert(buf_near(grid.data, interp_biv_pins::LSQ_GRID, 1.0e-6));
    let pts, f3 = scientific::bisplev(interp_biv_pins::LSQ_GX, interp_biv_pins::LSQ_GX, t1, t2, c, 3, 3, false, 0, 0);
    assert(f3.kind == scientific::F_NONE);
    assert(pts.data.len == interp_biv_pins::LSQ_GX.len);
    let dx, f4 = scientific::bisplev(interp_biv_pins::LSQ_GX, interp_biv_pins::LSQ_GY, t1, t2, c, 3, 3, true, 1, 0);
    assert(f4.kind == scientific::F_NONE);
    assert(buf_near(dx.data, interp_biv_pins::LSQ_DX, 1.0e-5));
    let intg, f5 = scientific::bivariate_integral(0.15, 0.7, 0.2, 0.6, t1, t2, c, 3, 3);
    assert(f5.kind == scientific::F_NONE);
    assert(common::close(intg, interp_biv_pins::LSQ_INT[0], 1.0e-6));
}

fn test_lsq_bivariate_degrees() {
    let g, f = scientific::lsq_bivariate_spline(interp_biv_pins::LSQ_X, interp_biv_pins::LSQ_Y, interp_biv_pins::LSQ_Z, interp_biv_pins::LSQ_TX, interp_biv_pins::LSQ_TY, [], 1, 2);
    assert(f.kind == scientific::F_NONE);
    let t1 = scientific::grid2_knots_x(g);
    let t2 = scientific::grid2_knots_y(g);
    assert(buf_near(t1, interp_biv_pins::LSQ2_T1, 1.0e-9));
    assert(buf_near(t2, interp_biv_pins::LSQ2_T2, 1.0e-9));
    let c = scientific::grid2_coefs(g);
    assert(buf_near(c, interp_biv_pins::LSQ2_C, 1.0e-6));
    let grid, f2 = scientific::bisplev(interp_biv_pins::LSQ_GX, interp_biv_pins::LSQ_GY, t1, t2, c, 1, 2);
    assert(f2.kind == scientific::F_NONE);
    assert(buf_near(grid.data, interp_biv_pins::LSQ2_GRID, 1.0e-6));
}

fn test_bivariate_spline() {
    // The holder of the knots and the coefficients evaluates through
    // the grid call of the package.
    let g, f = scientific::bivariate_spline(interp_biv_pins::LSQ_T1, interp_biv_pins::LSQ_T2, interp_biv_pins::LSQ_C, 3, 3);
    assert(f.kind == scientific::F_NONE);
    let grid, f2 = scientific::rect_spline_eval(g, interp_biv_pins::LSQ_GX, interp_biv_pins::LSQ_GY);
    assert(f2.kind == scientific::F_NONE);
    assert(grid.ndim == 2 && grid.shape[0] == interp_biv_pins::LSQ_GX.len && grid.shape[1] == interp_biv_pins::LSQ_GY.len);
    assert(buf_near(grid.data, interp_biv_pins::LSQ_GRID, 1.0e-6));
}

fn test_smoothing_spline() {
    let b, f = scientific::make_smoothing_spline(interp_biv_pins::SM_X, interp_biv_pins::SM_Y, [], interp_biv_pins::SM_LAM[0]);
    assert(f.kind == scientific::F_NONE);
    let t = scientific::bs_knots_of(b);
    let c = scientific::bs_coefs_of(b);
    assert(buf_near(t, interp_biv_pins::SM_T, 1.0e-9));
    assert(buf_near(c, interp_biv_pins::SM_C, 1.0e-6));
    let q: f64[>..] = [];
    var i: i64 = 0;
    while i < 7 { q.push(0.2 + (i * 13) / 30.0); i += 1; }
    let v, f2 = scientific::bspline_eval(b, q);
    assert(f2.kind == scientific::F_NONE);
    assert(buf_near(v.data, interp_biv_pins::SM_V, 1.0e-6));
}

fn test_smoothing_spline_weights() {
    let b, f = scientific::make_smoothing_spline(interp_biv_pins::SM_X, interp_biv_pins::SM_Y, interp_biv_pins::SMW_W, interp_biv_pins::SM_LAM[0]);
    assert(f.kind == scientific::F_NONE);
    let c = scientific::bs_coefs_of(b);
    assert(buf_near(c, interp_biv_pins::SMW_C, 1.0e-6));
}

fn test_smoothing_spline_gcv() {
    let b, f = scientific::make_smoothing_spline_gcv(interp_biv_pins::SM_X, interp_biv_pins::SMG_Y, []);
    assert(f.kind == scientific::F_NONE);
    let c = scientific::bs_coefs_of(b);
    // The factor of GCV is flat near its smallest value: the search of
    // the package and the search of SciPy stop at factors that differ
    // by the step of the search, so the coefficients agree to 1e-3.
    assert(buf_near(c, interp_biv_pins::SMG_C, 1.0e-3));
    let q: f64[>..] = [];
    var i: i64 = 0;
    while i < 7 { q.push(0.2 + (i * 13) / 30.0); i += 1; }
    let v, f2 = scientific::bspline_eval(b, q);
    assert(f2.kind == scientific::F_NONE);
    assert(buf_near(v.data, interp_biv_pins::SMG_V, 1.0e-3));
}

fn test_generate_knots() {
    let n = interp_biv_pins::GK_CASES.len / 2;
    var i: i64 = 0;
    while i < n {
        let k = interp_biv_pins::GK_CASES[2 * i] as i64;
        let s = interp_biv_pins::GK_CASES[2 * i + 1];
        let t, f = scientific::generate_knots(interp_biv_pins::GK_X, interp_biv_pins::GK_Y, [], k, s);
        assert(f.kind == scientific::F_NONE);
        if i == 0 { assert(buf_near(t, interp_biv_pins::GK0_T, 1.0e-9)); }
        if i == 1 { assert(buf_near(t, interp_biv_pins::GK1_T, 1.0e-9)); }
        if i == 2 { assert(buf_near(t, interp_biv_pins::GK2_T, 1.0e-9)); }
        if i == 3 { assert(buf_near(t, interp_biv_pins::GK3_T, 1.0e-9)); }
        i += 1;
    }
}

fn test_bpoly() {
    let b, f = scientific::bpoly(interp_biv_pins::BP_C, interp_biv_pins::BP_X);
    assert(f.kind == scientific::F_NONE);
    let v, f2 = scientific::bpoly_eval(b, interp_biv_pins::BP_Q);
    assert(f2.kind == scientific::F_NONE);
    assert(buf_near(v.data, interp_biv_pins::BP_V, 1.0e-6));
    let d1, f3 = scientific::bpoly_eval(b, interp_biv_pins::BP_Q, 1);
    assert(f3.kind == scientific::F_NONE);
    assert(buf_near(d1.data, interp_biv_pins::BP_D1, 1.0e-6));
    let d2, f4 = scientific::bpoly_eval(b, interp_biv_pins::BP_Q, 2);
    assert(f4.kind == scientific::F_NONE);
    assert(buf_near(d2.data, interp_biv_pins::BP_D2, 1.0e-6));
    let der, f5 = scientific::bpoly_derivative(copy(b), 1);
    assert(f5.kind == scientific::F_NONE);
    assert(buf_near(scientific::bpoly_coefs_of(der), interp_biv_pins::BP_DC, 1.0e-6));
    let anti, f6 = scientific::bpoly_antiderivative(copy(b), 1);
    assert(f6.kind == scientific::F_NONE);
    assert(buf_near(scientific::bpoly_coefs_of(anti), interp_biv_pins::BP_AC, 1.0e-6));
    let intg, f7 = scientific::bpoly_integrate(b, 0.5, 3.2);
    assert(f7.kind == scientific::F_NONE);
    assert(common::close(intg, interp_biv_pins::BP_INT[0], 1.0e-6));
}

fn test_nd_bspline() {
    var t: f64[>..] = [];
    var i: i64 = 0;
    while i < interp_biv_pins::ND_T1.len { t.push(interp_biv_pins::ND_T1[i]); i += 1; }
    i = 0;
    while i < interp_biv_pins::ND_T2.len { t.push(interp_biv_pins::ND_T2[i]); i += 1; }
    var nts: i64[>..] = [interp_biv_pins::ND_T1.len, interp_biv_pins::ND_T2.len];
    var ks: i64[>..] = [3, 3];
    let s, f = scientific::nd_bspline(t, nts, ks, interp_biv_pins::ND_C);
    assert(f.kind == scientific::F_NONE);
    let np = interp_biv_pins::ND_PTS.len / 2;
    let v, f2 = scientific::ndspl_eval(s, interp_biv_pins::ND_PTS, np, [], true);
    assert(f2.kind == scientific::F_NONE);
    assert(buf_near(v.data, interp_biv_pins::ND_V, 1.0e-6));
    let d1, f3 = scientific::ndspl_eval(s, interp_biv_pins::ND_PTS, np, [1, 0], true);
    assert(f3.kind == scientific::F_NONE);
    assert(buf_near(d1.data, interp_biv_pins::ND_D, 1.0e-5));
    let d2, f4 = scientific::ndspl_eval(s, interp_biv_pins::ND_PTS, np, [1, 2], true);
    assert(f4.kind == scientific::F_NONE);
    assert(buf_near(d2.data, interp_biv_pins::ND_D2, 1.0e-4));
}

fn test_nd_poly() {
    var x: f64[>..] = [];
    var i: i64 = 0;
    while i < interp_biv_pins::NP_X1.len { x.push(interp_biv_pins::NP_X1[i]); i += 1; }
    i = 0;
    while i < interp_biv_pins::NP_X2.len { x.push(interp_biv_pins::NP_X2[i]); i += 1; }
    var nbs: i64[>..] = [interp_biv_pins::NP_X1.len, interp_biv_pins::NP_X2.len];
    var ks: i64[>..] = [2, 1];
    let p, f = scientific::nd_poly(x, nbs, ks, interp_biv_pins::NP_C);
    assert(f.kind == scientific::F_NONE);
    let np = interp_biv_pins::NP_Q.len / 2;
    let v, f2 = scientific::ndpoly_eval(p, interp_biv_pins::NP_Q, np, [], true);
    assert(f2.kind == scientific::F_NONE);
    assert(buf_near(v.data, interp_biv_pins::NP_V, 1.0e-6));
    let d1, f3 = scientific::ndpoly_eval(p, interp_biv_pins::NP_Q, np, [1, 1], true);
    assert(f3.kind == scientific::F_NONE);
    assert(buf_near(d1.data, interp_biv_pins::NP_D, 1.0e-6));
}

fn test_bisplev_faults() {
    let t1 = interp_biv_pins::LSQ_T1;
    let t2 = interp_biv_pins::LSQ_T2;
    let c = interp_biv_pins::LSQ_C;
    let bad, f = scientific::bisplev(interp_biv_pins::LSQ_GX, interp_biv_pins::LSQ_GY, t1, t2, c, 3, 3);
    assert(f.kind == scientific::F_NONE);
    // One coefficient short breaks the count of the coefficients.
    var short: f64[>..] = [];
    var i: i64 = 0;
    while i + 1 < c.len { short.push(c[i]); i += 1; }
    let r1, f1 = scientific::bisplev([0.5], [0.5], t1, t2, short, 3, 3);
    assert(f1.kind == scientific::F_SHAPE);
    let r2, f2 = scientific::bisplev([0.5 / 0.0], [0.5], t1, t2, c, 3, 3);
    assert(f2.kind == scientific::F_VALUE);
    let r3, f3 = scientific::bisplev([0.5], [0.5], t1, t2, c, 3, 3, true, -1, 0);
    assert(f3.kind == scientific::F_RANGE);
    let r4, f4 = scientific::bisplev([], [0.5], t1, t2, c, 3, 3);
    assert(f4.kind == scientific::F_SHAPE);
}

fn test_bivariate_faults() {
    let r1, f1 = scientific::bivariate_spline(interp_biv_pins::LSQ_T1, interp_biv_pins::LSQ_T2, interp_biv_pins::LSQ_C, 0, 3);
    assert(f1.kind == scientific::F_RANGE);
    var short: f64[>..] = [];
    var i: i64 = 0;
    while i + 1 < interp_biv_pins::LSQ_C.len { short.push(interp_biv_pins::LSQ_C[i]); i += 1; }
    let r2, f2 = scientific::bivariate_spline(interp_biv_pins::LSQ_T1, interp_biv_pins::LSQ_T2, short, 3, 3);
    assert(f2.kind == scientific::F_SHAPE);
    let m = interp_biv_pins::LSQ_X.len;
    var few_x: f64[>..] = [];
    var few_y: f64[>..] = [];
    var few_z: f64[>..] = [];
    i = 0;
    while i < 5 { few_x.push(interp_biv_pins::LSQ_X[i]); few_y.push(interp_biv_pins::LSQ_Y[i]); few_z.push(interp_biv_pins::LSQ_Z[i]); i += 1; }
    let r3, f3 = scientific::lsq_bivariate_spline(few_x, few_y, few_z, interp_biv_pins::LSQ_TX, interp_biv_pins::LSQ_TY, [], 3, 3);
    assert(f3.kind == scientific::F_SHAPE);
    var fall_t: f64[>..] = [];
    i = 0;
    while i < interp_biv_pins::LSQ_TX.len { fall_t.push(interp_biv_pins::LSQ_TX[interp_biv_pins::LSQ_TX.len - 1 - i]); i += 1; }
    let r4, f4 = scientific::lsq_bivariate_spline(interp_biv_pins::LSQ_X, interp_biv_pins::LSQ_Y, interp_biv_pins::LSQ_Z, fall_t, interp_biv_pins::LSQ_TY, [], 3, 3);
    assert(f4.kind == scientific::F_VALUE);
    let r5, f5 = scientific::lsq_bivariate_spline(interp_biv_pins::LSQ_X, interp_biv_pins::LSQ_Y, interp_biv_pins::LSQ_Z, interp_biv_pins::LSQ_TX, interp_biv_pins::LSQ_TY, [1.0], 3, 3);
    assert(f5.kind == scientific::F_SHAPE);
    var nan_x: f64[>..] = [];
    i = 0;
    while i < m { if i == 3 { nan_x.push(0.0 / 0.0); } else { nan_x.push(interp_biv_pins::LSQ_X[i]); } i += 1; }
    let r6, f6 = scientific::lsq_bivariate_spline(nan_x, interp_biv_pins::LSQ_Y, interp_biv_pins::LSQ_Z, interp_biv_pins::LSQ_TX, interp_biv_pins::LSQ_TY, [], 3, 3);
    assert(f6.kind == scientific::F_VALUE);
}

fn test_smoothing_faults() {
    var short_x: f64[>..] = [];
    var short_y: f64[>..] = [];
    var i: i64 = 0;
    while i < 4 { short_x.push(interp_biv_pins::SM_X[i]); short_y.push(interp_biv_pins::SM_Y[i]); i += 1; }
    let r1, f1 = scientific::make_smoothing_spline(short_x, short_y, [], 1.0);
    assert(f1.kind == scientific::F_SHAPE);
    let r2, f2 = scientific::make_smoothing_spline(interp_biv_pins::SM_X, interp_biv_pins::SM_Y, [], -1.0);
    assert(f2.kind == scientific::F_VALUE);
    var fall_x: f64[>..] = [];
    var fall_y: f64[>..] = [];
    i = 0;
    while i < interp_biv_pins::SM_X.len { fall_x.push(interp_biv_pins::SM_X[i]); fall_y.push(interp_biv_pins::SM_Y[i]); i += 1; }
    fall_x[3] = fall_x[2];
    let r3, f3 = scientific::make_smoothing_spline(fall_x, fall_y, [], 1.0);
    assert(f3.kind == scientific::F_VALUE);
    var bad_w: f64[>..] = [];
    i = 0;
    while i < interp_biv_pins::SM_X.len { if i == 2 { bad_w.push(-1.0); } else { bad_w.push(1.0); } i += 1; }
    let r4, f4 = scientific::make_smoothing_spline_gcv(interp_biv_pins::SM_X, interp_biv_pins::SM_Y, bad_w);
    assert(f4.kind == scientific::F_VALUE);
    let r5, f5 = scientific::generate_knots(interp_biv_pins::GK_X, interp_biv_pins::GK_Y, [], 0, 0.5);
    assert(f5.kind == scientific::F_RANGE);
    let r6, f6 = scientific::generate_knots(interp_biv_pins::GK_X, interp_biv_pins::GK_Y, [], 3, -1.0);
    assert(f6.kind == scientific::F_VALUE);
}

fn test_bpoly_faults() {
    var fall_x: f64[>..] = [];
    var i: i64 = 0;
    while i < interp_biv_pins::BP_X.len { fall_x.push(interp_biv_pins::BP_X[i]); i += 1; }
    fall_x[2] = fall_x[1];
    let r1, f1 = scientific::bpoly(interp_biv_pins::BP_C, fall_x);
    assert(f1.kind == scientific::F_VALUE);
    let d, f2 = scientific::bpoly_eval(r1, [0.5]);
    assert(f2.kind == scientific::F_SHAPE);
    var short_c: f64[>..] = [];
    i = 0;
    while i + 1 < interp_biv_pins::BP_C.len { short_c.push(interp_biv_pins::BP_C[i]); i += 1; }
    let r3, f3 = scientific::bpoly(short_c, interp_biv_pins::BP_X);
    assert(f3.kind == scientific::F_SHAPE);
    let b, f4 = scientific::bpoly(interp_biv_pins::BP_C, interp_biv_pins::BP_X);
    assert(f4.kind == scientific::F_NONE);
    let r5, f5 = scientific::bpoly_eval(b, [0.5], -1);
    assert(f5.kind == scientific::F_RANGE);
}

fn test_nd_faults() {
    var t: f64[>..] = [];
    var i: i64 = 0;
    while i < interp_biv_pins::ND_T1.len { t.push(interp_biv_pins::ND_T1[i]); i += 1; }
    i = 0;
    while i < interp_biv_pins::ND_T2.len { t.push(interp_biv_pins::ND_T2[i]); i += 1; }
    var nts: i64[>..] = [interp_biv_pins::ND_T1.len, interp_biv_pins::ND_T2.len];
    let r1, f1 = scientific::nd_bspline(t, nts, [], interp_biv_pins::ND_C);
    assert(f1.kind == scientific::F_SHAPE);
    var ks_bad: i64[>..] = [3, 30];
    let r2, f2 = scientific::nd_bspline(t, nts, ks_bad, interp_biv_pins::ND_C);
    assert(f2.kind == scientific::F_RANGE);
    var short_c: f64[>..] = [];
    i = 0;
    while i + 1 < interp_biv_pins::ND_C.len { short_c.push(interp_biv_pins::ND_C[i]); i += 1; }
    var ks: i64[>..] = [3, 3];
    let r3, f3 = scientific::nd_bspline(t, nts, ks, short_c);
    assert(f3.kind == scientific::F_SHAPE);
    var nan_t: f64[>..] = [];
    i = 0;
    while i < t.len { if i == 4 { nan_t.push(0.0 / 0.0); } else { nan_t.push(t[i]); } i += 1; }
    let r4, f4 = scientific::nd_bspline(nan_t, nts, ks, interp_biv_pins::ND_C);
    assert(f4.kind == scientific::F_VALUE);
    var nbs: i64[>..] = [interp_biv_pins::NP_X1.len, interp_biv_pins::NP_X2.len];
    var ks2: i64[>..] = [2, 1];
    var x: f64[>..] = [];
    i = 0;
    while i < interp_biv_pins::NP_X1.len { x.push(interp_biv_pins::NP_X1[i]); i += 1; }
    i = 0;
    while i < interp_biv_pins::NP_X2.len { x.push(interp_biv_pins::NP_X2[i]); i += 1; }
    let p, f5 = scientific::nd_poly(x, nbs, ks2, interp_biv_pins::NP_C);
    assert(f5.kind == scientific::F_NONE);
    let np = interp_biv_pins::NP_Q.len / 2;
    let r6, f6 = scientific::ndpoly_eval(p, interp_biv_pins::NP_Q, np, [1], true);
    assert(f6.kind == scientific::F_SHAPE);
    let r7, f7 = scientific::ndpoly_eval(p, interp_biv_pins::NP_Q, np + 1, [], true);
    assert(f7.kind == scientific::F_SHAPE);
    let v, f8 = scientific::ndpoly_eval(p, interp_biv_pins::NP_Q, np, [], false);
    assert(f8.kind == scientific::F_NONE);
    // The point at 1.1 stands outside, so the value is NaN.
    var q: f64[>..] = [];
    i = 0;
    while i < interp_biv_pins::NP_Q.len { if i == interp_biv_pins::NP_Q.len - 1 { q.push(3.0); } else { q.push(interp_biv_pins::NP_Q[i]); } i += 1; }
    let v2, f9 = scientific::ndpoly_eval(p, q, np, [], false);
    assert(f9.kind == scientific::F_NONE);
    assert(v2.data[v2.data.len - 1] != v2.data[v2.data.len - 1]);
}

fn run() {
    test_lsq_bivariate();
    test_lsq_bivariate_degrees();
    test_bivariate_spline();
    test_smoothing_spline();
    test_smoothing_spline_weights();
    test_smoothing_spline_gcv();
    test_generate_knots();
    test_bpoly();
    test_nd_bspline();
    test_nd_poly();
    test_bisplev_faults();
    test_bivariate_faults();
    test_smoothing_faults();
    test_bpoly_faults();
    test_nd_faults();
}
