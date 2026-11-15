// The tests of the group interp_nd. Run them with SCIENTIFIC_SUITES=interp_nd.
//
// The interpolators of scattered data of the package are pinned
// against SciPy 1.18.1: the radial basis of the legacy class and the
// class of the modern one, the rational approximation of AAA and the
// piecewise cubic of Clough and Tocher.
namespace interp_nd_tests;

import scientific from "../..";
import os;
import common;
import interp_nd_pins;

// buf_near6 tells whether the values of a buffer agree with a list.
fn buf_near6(v: f64[:], want: f64[:]) -> bool {
    if v.len != want.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        if !common::close(v[i], want[i], 1.0e-6) { return false; }
        i += 1;
    }
    return true;
}

// The legacy interpolator of the radial basis gives the values at the
// points of the query.
fn test_rbf() {
    let pts = common::arr_of([12, 2], interp_nd_pins::RBF_PTS);
    let vals = common::arr_of([12], interp_nd_pins::RBF_VALS);
    let query = common::arr_of([5, 2], interp_nd_pins::RBF_QUERY);
    let v, f = scientific::rbf(pts, vals, query);
    assert(f.kind == scientific::F_NONE && buf_near6(v.data, interp_nd_pins::RBF_MQ_VALUE));
    // The Gaussian with a shape and a smoothing.
    let g, gf = scientific::rbf(pts, vals, query, "gaussian", 0.8, 0.1);
    assert(gf.kind == scientific::F_NONE && buf_near6(g.data, interp_nd_pins::RBF_GAUSS_VALUE));
    // The thin plate spline of one dimension.
    let nodes = common::arr_of([9], interp_nd_pins::RBF_TPS_1D_NODES);
    let one = common::arr_of([9], interp_nd_pins::RBF_TPS_1D_VALS);
    let q = common::arr_of([4], interp_nd_pins::RBF_TPS_1D_QUERY);
    let t, tf = scientific::rbf(nodes, one, q, "thin_plate");
    assert(tf.kind == scientific::F_NONE && buf_near6(t.data, interp_nd_pins::RBF_TPS_1D_VALUE));
    return;
}

// The fault paths of the legacy interpolator.
fn test_rbf_faults() {
    let pts = common::arr_of([12, 2], interp_nd_pins::RBF_PTS);
    let vals = common::arr_of([12], interp_nd_pins::RBF_VALS);
    let query = common::arr_of([5, 2], interp_nd_pins::RBF_QUERY);
    // A function of no name gives a fault.
    let e0, f0 = scientific::rbf(pts, vals, query, "spline");
    assert(f0.kind == scientific::F_VALUE);
    // A shape of no width gives a fault.
    let e1, f1 = scientific::rbf(pts, vals, query, "multiquadric", 0.0);
    assert(f1.kind == scientific::F_VALUE);
    // The values must hold one value for each point.
    let short = common::arr_of([4], interp_nd_pins::RBF_VALS);
    let e2, f2 = scientific::rbf(pts, short, query);
    assert(f2.kind == scientific::F_SHAPE);
    // The query must hold the same count of coordinates.
    let bad = common::arr_of([5, 3], interp_nd_pins::RBF_QUERY);
    let e3, f3 = scientific::rbf(pts, vals, bad);
    assert(f3.kind == scientific::F_SHAPE);
    // A point that is not finite gives a fault.
    let nan = 0.0 / 0.0;
    var broken: f64[>..] = [];
    var i: i64 = 0;
    while i < 24 { broken.push(interp_nd_pins::RBF_PTS[i]); i += 1; }
    broken[3] = nan;
    let badpts = common::arr_of([12, 2], broken);
    let e4, f4 = scientific::rbf(badpts, vals, query);
    assert(f4.kind == scientific::F_VALUE);
    return;
}

// The interpolator of the radial basis gives the values at the points
// of the query, with the kernels of the reference.
fn test_rbf_interpolate() {
    let pts = common::arr_of([12, 2], interp_nd_pins::RBF_PTS);
    let vals = common::arr_of([12], interp_nd_pins::RBF_VALS);
    let query = common::arr_of([5, 2], interp_nd_pins::RBF_QUERY);
    // The thin plate spline of the default.
    let v, f = scientific::rbf_interpolate(pts, vals, query);
    assert(f.kind == scientific::F_NONE && buf_near6(v.data, interp_nd_pins::RBFI_TPS_VALUE));
    // The multiquadric with a shape.
    let m, mf = scientific::rbf_interpolate(pts, vals, query, -1, 0.0, "multiquadric", 2.0);
    assert(mf.kind == scientific::F_NONE && buf_near6(m.data, interp_nd_pins::RBFI_MQ_VALUE));
    // The linear kernel with no polynomial.
    let l, lf = scientific::rbf_interpolate(pts, vals, query, -1, 0.0, "linear", -1.0, 0);
    assert(lf.kind == scientific::F_NONE && buf_near6(l.data, interp_nd_pins::RBFI_LINEAR_VALUE));
    // The Gaussian with a shape and no polynomial.
    let g, gf = scientific::rbf_interpolate(pts, vals, query, -1, 0.0, "gaussian", 1.5, -1);
    assert(gf.kind == scientific::F_NONE && buf_near6(g.data, interp_nd_pins::RBFI_GAUSS_VALUE));
    // The smoothing relaxes the fit of the data.
    let s, sf = scientific::rbf_interpolate(pts, vals, query, -1, 0.5);
    assert(sf.kind == scientific::F_NONE && buf_near6(s.data, interp_nd_pins::RBFI_SMOOTH_VALUE));
    return;
}

// The nearest points of the query bound the system of the
// interpolator.
fn test_rbf_interpolate_neighbors() {
    let pts = common::arr_of([24, 2], interp_nd_pins::RBFI_NB_PTS);
    let vals = common::arr_of([24], interp_nd_pins::RBFI_NB_VALS);
    let query = common::arr_of([4, 2], interp_nd_pins::RBFI_NB_QUERY);
    let v, f = scientific::rbf_interpolate(pts, vals, query, 8);
    assert(f.kind == scientific::F_NONE && buf_near6(v.data, interp_nd_pins::RBFI_NB_VALUE));
    return;
}

// The centers hold one dimension, and the result holds one value for
// each point of the query.
fn test_rbf_interpolate_one_dimension() {
    let pts = common::arr_of([11, 1], interp_nd_pins::RBFI_1D_PTS);
    let vals = common::arr_of([11], interp_nd_pins::RBFI_1D_VALS);
    let query = common::arr_of([3, 1], interp_nd_pins::RBFI_1D_QUERY);
    let v, f = scientific::rbf_interpolate(pts, vals, query);
    assert(f.kind == scientific::F_NONE && buf_near6(v.data, interp_nd_pins::RBFI_1D_VALUE));
    return;
}

// The values of the centers hold two columns, and the result holds
// two columns too.
fn test_rbf_interpolate_columns() {
    let pts = common::arr_of([12, 2], interp_nd_pins::RBF_PTS);
    let vals = common::arr_of([12, 2], interp_nd_pins::RBFI_MULTI_VALS);
    let query = common::arr_of([5, 2], interp_nd_pins::RBF_QUERY);
    let v, f = scientific::rbf_interpolate(pts, vals, query);
    assert(f.kind == scientific::F_NONE && v.ndim == 2 && v.shape[0] == 5 && v.shape[1] == 2);
    assert(buf_near6(v.data, interp_nd_pins::RBFI_MULTI_VALUE));
    return;
}

// The fault paths of the interpolator.
fn test_rbf_interpolate_faults() {
    let pts = common::arr_of([12, 2], interp_nd_pins::RBF_PTS);
    let vals = common::arr_of([12], interp_nd_pins::RBF_VALS);
    let query = common::arr_of([5, 2], interp_nd_pins::RBF_QUERY);
    // A kernel of no name gives a fault.
    let e0, f0 = scientific::rbf_interpolate(pts, vals, query, -1, 0.0, "cubic_spline");
    assert(f0.kind == scientific::F_VALUE);
    // The centers must hold a matrix.
    let flat = common::arr_of([12], interp_nd_pins::RBF_PTS);
    let e1, f1 = scientific::rbf_interpolate(flat, vals, query);
    assert(f1.kind == scientific::F_SHAPE);
    // The values must hold one row for each center.
    let short = common::arr_of([4], interp_nd_pins::RBF_VALS);
    let e2, f2 = scientific::rbf_interpolate(pts, short, query);
    assert(f2.kind == scientific::F_SHAPE);
    // A kernel that needs a shape gives a fault without one.
    let e3, f3 = scientific::rbf_interpolate(pts, vals, query, -1, 0.0, "multiquadric");
    assert(f3.kind == scientific::F_VALUE);
    // A degree below -2 gives a fault.
    let e4, f4 = scientific::rbf_interpolate(pts, vals, query, -1, 0.0, "thin_plate_spline", -1.0, -3);
    assert(f4.kind == scientific::F_RANGE);
    // The centers must hold more points than the monomials need.
    let two = common::arr_of([2, 2], interp_nd_pins::RBF_PTS);
    let twov = common::arr_of([2], interp_nd_pins::RBF_VALS);
    let e5, f5 = scientific::rbf_interpolate(two, twov, query);
    assert(f5.kind == scientific::F_SHAPE);
    // A count of neighbors below -1 gives a fault.
    let e6, f6 = scientific::rbf_interpolate(pts, vals, query, -2);
    assert(f6.kind == scientific::F_RANGE);
    // A center that is not finite gives a fault.
    let nan = 0.0 / 0.0;
    var broken: f64[>..] = [];
    var i: i64 = 0;
    while i < 24 { broken.push(interp_nd_pins::RBF_PTS[i]); i += 1; }
    broken[5] = nan;
    let badpts = common::arr_of([12, 2], broken);
    let e7, f7 = scientific::rbf_interpolate(badpts, vals, query);
    assert(f7.kind == scientific::F_VALUE);
    return;
}

// The rational approximation of AAA gives the values at the points of
// the query.
fn test_aaa_interpolate() {
    let v, f = scientific::aaa_interpolate(interp_nd_pins::AAA_RUNGE_X,
                                            interp_nd_pins::AAA_RUNGE_F,
                                            interp_nd_pins::AAA_RUNGE_QUERY);
    assert(f.kind == scientific::F_NONE && buf_near6(v.data, interp_nd_pins::AAA_RUNGE_VALUE));
    let e, ef = scientific::aaa_interpolate(interp_nd_pins::AAA_EXP_X,
                                            interp_nd_pins::AAA_EXP_F,
                                            interp_nd_pins::AAA_EXP_QUERY);
    assert(ef.kind == scientific::F_NONE && buf_near6(e.data, interp_nd_pins::AAA_EXP_VALUE));
    // The sample holds a pole outside of it.
    let p, pf = scientific::aaa_interpolate(interp_nd_pins::AAA_POLE_X,
                                            interp_nd_pins::AAA_POLE_F,
                                            interp_nd_pins::AAA_POLE_QUERY);
    assert(pf.kind == scientific::F_NONE && buf_near6(p.data, interp_nd_pins::AAA_POLE_VALUE));
    return;
}

// The fault paths of the rational approximation.
fn test_aaa_faults() {
    let x = interp_nd_pins::AAA_RUNGE_X;
    let y = interp_nd_pins::AAA_RUNGE_F;
    let q = interp_nd_pins::AAA_RUNGE_QUERY;
    // The sample and the values need equal counts.
    let e0, f0 = scientific::aaa_interpolate(x, interp_nd_pins::AAA_EXP_F, q);
    assert(f0.kind == scientific::F_SHAPE);
    // The list of points must hold a value.
    let e1, f1 = scientific::aaa_interpolate(x, y, []);
    assert(f1.kind == scientific::F_SHAPE);
    // A count of terms below one gives a fault.
    let e2, f2 = scientific::aaa_interpolate(x, y, q, -1.0, 0);
    assert(f2.kind == scientific::F_RANGE);
    // A count of terms above the limit gives a fault.
    let e3, f3 = scientific::aaa_interpolate(x, y, q, -1.0, 100000);
    assert(f3.kind == scientific::F_LIMIT);
    // A sample that is not finite gives a fault.
    let nan = 0.0 / 0.0;
    var broken: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len { broken.push(x[i]); i += 1; }
    broken[2] = nan;
    let e4, f4 = scientific::aaa_interpolate(broken, y, q);
    assert(f4.kind == scientific::F_VALUE);
    // The values hold no finite entry.
    var allnan: f64[>..] = [];
    i = 0;
    while i < y.len { allnan.push(nan); i += 1; }
    let e5, f5 = scientific::aaa_interpolate(x, allnan, q);
    assert(f5.kind == scientific::F_VALUE);
    return;
}

// The interpolator of Clough and Tocher gives the values at the
// points of the query.
fn test_clough_tocher() {
    let pts = common::arr_of([40, 2], interp_nd_pins::CT_PTS);
    let vals = common::arr_of([40], interp_nd_pins::CT_VALS);
    let query = common::arr_of([5, 2], interp_nd_pins::CT_QUERY);
    let v, f = scientific::clough_tocher_interpolate(pts, vals, query);
    assert(f.kind == scientific::F_NONE && buf_near6(v.data, interp_nd_pins::CT_VALUE));
    // A point outside the hull takes the fill value.
    let out = common::arr_of([1, 2], interp_nd_pins::CT_OUT_QUERY);
    let w, wf = scientific::clough_tocher_interpolate(pts, vals, out);
    assert(wf.kind == scientific::F_NONE && w.data.len == 1 && w.data[0] != w.data[0]);
    return;
}

// The values of the points hold two columns, and the result holds two
// columns too.
fn test_clough_tocher_columns() {
    let pts = common::arr_of([40, 2], interp_nd_pins::CT_PTS);
    let vals = common::arr_of([40, 2], interp_nd_pins::CT_MULTI_VALS);
    let query = common::arr_of([5, 2], interp_nd_pins::CT_QUERY);
    let v, f = scientific::clough_tocher_interpolate(pts, vals, query);
    assert(f.kind == scientific::F_NONE && v.ndim == 2 && v.shape[0] == 5 && v.shape[1] == 2);
    assert(buf_near6(v.data, interp_nd_pins::CT_MULTI_VALUE));
    return;
}

// The rescale moves the axes of a different width into the unit
// width.
fn test_clough_tocher_rescale() {
    var sp: f64[>..] = [];
    var i: i64 = 0;
    while i < 40 {
        sp.push(interp_nd_pins::CT_PTS[i * 2] * 1000.0);
        sp.push(interp_nd_pins::CT_PTS[i * 2 + 1]);
        i += 1;
    }
    let pts = common::arr_of([40, 2], sp);
    let vals = common::arr_of([40], interp_nd_pins::CT_VALS);
    let query = common::arr_of([5, 2], interp_nd_pins::CT_RESCALE_QUERY);
    let v, f = scientific::clough_tocher_interpolate(pts, vals, query, 0.0 / 0.0, 1.0e-6, 400, true);
    assert(f.kind == scientific::F_NONE && buf_near6(v.data, interp_nd_pins::CT_RESCALE_VALUE));
    return;
}

// The fault paths of the interpolator.
fn test_clough_tocher_faults() {
    let pts = common::arr_of([40, 2], interp_nd_pins::CT_PTS);
    let vals = common::arr_of([40], interp_nd_pins::CT_VALS);
    let query = common::arr_of([5, 2], interp_nd_pins::CT_QUERY);
    // The points must hold two columns.
    let onecol = common::arr_of([40], interp_nd_pins::CT_PTS);
    let e0, f0 = scientific::clough_tocher_interpolate(onecol, vals, query);
    assert(f0.kind == scientific::F_SHAPE);
    // The query must hold two columns.
    let badq = common::arr_of([5, 3], interp_nd_pins::CT_QUERY);
    let e1, f1 = scientific::clough_tocher_interpolate(pts, vals, badq);
    assert(f1.kind == scientific::F_SHAPE);
    // The values must hold one row for each point.
    let short = common::arr_of([4], interp_nd_pins::CT_VALS);
    let e2, f2 = scientific::clough_tocher_interpolate(pts, short, query);
    assert(f2.kind == scientific::F_SHAPE);
    // A coordinate that is not finite gives a fault.
    let nan = 0.0 / 0.0;
    var broken: f64[>..] = [];
    var i: i64 = 0;
    while i < 80 { broken.push(interp_nd_pins::CT_PTS[i]); i += 1; }
    broken[7] = nan;
    let badpts = common::arr_of([40, 2], broken);
    let e3, f3 = scientific::clough_tocher_interpolate(badpts, vals, query);
    assert(f3.kind == scientific::F_VALUE);
    // Fewer than three points give a fault.
    let two = common::arr_of([2, 2], interp_nd_pins::CT_PTS);
    let twov = common::arr_of([2], interp_nd_pins::CT_VALS);
    let e4, f4 = scientific::clough_tocher_interpolate(two, twov, query);
    assert(f4.kind == scientific::F_SHAPE);
    return;
}
