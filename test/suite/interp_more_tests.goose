// The tests of the group interp_more. Run them with SCIENTIFIC_SUITES=interp_more.
//
// The extra interpolation calls of the package are pinned against
// SciPy 1.18.1: the interpolators of Barycentric, Krogh and Floater
// and Hormann, the polynomial of Lagrange, the approximation of Pade,
// the Taylor polynomial, the algebra of the B-spline and the spline
// of two variables on a grid.
namespace interp_more_tests;

import scientific from "../..";
import os;
import common;
import interp_more_pins;

// buffer_near tells whether the values of a buffer agree with a list.
fn buffer_near(v: f64[:], want: f64[:]) -> bool {
    if v.len != want.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        if !common::near(v[i], want[i]) { return false; }
        i += 1;
    }
    return true;
}

// The interpolator of Barycentric gives the values and the two first
// derivatives at the points.
fn test_barycentric_interpolate() {
    let v, f = scientific::barycentric_interpolate(interp_more_pins::BARY_NODES, interp_more_pins::BARY_VALUES, interp_more_pins::BARY_QUERY);
    assert(f.kind == scientific::F_NONE && buffer_near(v.data, interp_more_pins::BARY_VALUE));
    let d1, f1 = scientific::barycentric_interpolate(interp_more_pins::BARY_NODES, interp_more_pins::BARY_VALUES, interp_more_pins::BARY_QUERY, 1);
    assert(f1.kind == scientific::F_NONE && buffer_near(d1.data, interp_more_pins::BARY_DER1));
    let d2, f2 = scientific::barycentric_interpolate(interp_more_pins::BARY_NODES, interp_more_pins::BARY_VALUES, interp_more_pins::BARY_QUERY, 2);
    assert(f2.kind == scientific::F_NONE && buffer_near(d2.data, interp_more_pins::BARY_DER2));
    // A node that repeats gives a fault.
    let e1, g1 = scientific::barycentric_interpolate([0.0, 1.0, 1.0], [1.0, 2.0, 3.0], [0.5]);
    assert(g1.kind == scientific::F_VALUE);
    // The counts must agree.
    let e2, g2 = scientific::barycentric_interpolate([0.0, 1.0], [1.0, 2.0, 3.0], [0.5]);
    assert(g2.kind == scientific::F_SHAPE);
    // The list of points must hold a value.
    let e3, g3 = scientific::barycentric_interpolate([0.0, 1.0], [1.0, 2.0], []);
    assert(g3.kind == scientific::F_SHAPE);
    // The order of the derivative stands below zero.
    let e4, g4 = scientific::barycentric_interpolate([0.0, 1.0], [1.0, 2.0], [0.5], -1);
    assert(g4.kind == scientific::F_RANGE);
    // A node that is not finite gives a fault.
    let nan = 0.0 / 0.0;
    let e5, g5 = scientific::barycentric_interpolate([0.0, nan], [1.0, 2.0], [0.5]);
    assert(g5.kind == scientific::F_VALUE);
    return;
}

// The interpolator of Krogh gives the values and the two first
// derivatives at the points. A node that repeats holds a derivative.
fn test_krogh_interpolate() {
    let v, f = scientific::krogh_interpolate(interp_more_pins::KROGH_NODES, interp_more_pins::KROGH_VALUES, interp_more_pins::KROGH_QUERY);
    assert(f.kind == scientific::F_NONE && buffer_near(v.data, interp_more_pins::KROGH_VALUE));
    let d1, f1 = scientific::krogh_interpolate(interp_more_pins::KROGH_NODES, interp_more_pins::KROGH_VALUES, interp_more_pins::KROGH_QUERY, 2);
    assert(f1.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < interp_more_pins::KROGH_QUERY.len {
        assert(common::near(d1.data[3 * i], interp_more_pins::KROGH_VALUE[i]));
        assert(common::near(d1.data[3 * i + 1], interp_more_pins::KROGH_DER1[i]));
        assert(common::near(d1.data[3 * i + 2], interp_more_pins::KROGH_DER2[i]));
        i += 1;
    }
    // The nodes must not fall.
    let e1, g1 = scientific::krogh_interpolate([1.0, 0.0], [1.0, 2.0], [0.5]);
    assert(g1.kind == scientific::F_VALUE);
    // The order of the derivative stands above the count of the nodes.
    let e2, g2 = scientific::krogh_interpolate([0.0, 1.0], [1.0, 2.0], [0.5], 5);
    assert(g2.kind == scientific::F_RANGE);
    return;
}

// The interpolator of Floater and Hormann gives the values for the
// depths zero, one and three.
fn test_floater_hormann() {
    let v0, f0 = scientific::floater_hormann(interp_more_pins::BARY_NODES, interp_more_pins::BARY_VALUES, interp_more_pins::BARY_QUERY, 0);
    assert(f0.kind == scientific::F_NONE && buffer_near(v0.data, interp_more_pins::FH_D0_VALUE));
    let v1, f1 = scientific::floater_hormann(interp_more_pins::BARY_NODES, interp_more_pins::BARY_VALUES, interp_more_pins::BARY_QUERY, 1);
    assert(f1.kind == scientific::F_NONE && buffer_near(v1.data, interp_more_pins::FH_D1_VALUE));
    let v3, f3 = scientific::floater_hormann(interp_more_pins::BARY_NODES, interp_more_pins::BARY_VALUES, interp_more_pins::BARY_QUERY, 3);
    assert(f3.kind == scientific::F_NONE && buffer_near(v3.data, interp_more_pins::FH_D3_VALUE));
    // The depth stands above the count of the nodes.
    let e1, g1 = scientific::floater_hormann([0.0, 1.0], [1.0, 2.0], [0.5], 4);
    assert(g1.kind == scientific::F_RANGE);
    // The nodes must not repeat. The depth of three needs four nodes
    // or more, so the call asks for the depth of one.
    let e2, g2 = scientific::floater_hormann([0.0, 1.0, 1.0], [1.0, 2.0, 3.0], [0.5], 1);
    assert(g2.kind == scientific::F_VALUE);
    return;
}

// The polynomial of Lagrange gives the coefficients from the highest
// power to the lowest one.
fn test_lagrange_poly() {
    let c, f = scientific::lagrange_poly(interp_more_pins::LAG_NODES, interp_more_pins::LAG_VALUES);
    assert(f.kind == scientific::F_NONE && buffer_near(c, interp_more_pins::LAG_COEF));
    // A node that repeats gives a fault.
    let e1, g1 = scientific::lagrange_poly([0.0, 1.0, 1.0], [1.0, 2.0, 3.0]);
    assert(g1.kind == scientific::F_VALUE);
    // The counts must agree.
    let e2, g2 = scientific::lagrange_poly([0.0, 1.0], [1.0, 2.0, 3.0]);
    assert(g2.kind == scientific::F_SHAPE);
    return;
}

// The approximation of Pade gives the two polynomials of the ratio.
fn test_pade_approx() {
    let p, q, f = scientific::pade_approx(interp_more_pins::PADE_COEFF, 2, 2);
    assert(f.kind == scientific::F_NONE);
    assert(buffer_near(p, interp_more_pins::PADE_2_2_P));
    assert(buffer_near(q, interp_more_pins::PADE_2_2_Q));
    let p1, q1, f1 = scientific::pade_approx(interp_more_pins::PADE_COEFF, 1, 2);
    assert(f1.kind == scientific::F_NONE);
    assert(buffer_near(p1, interp_more_pins::PADE_1_2_P));
    assert(buffer_near(q1, interp_more_pins::PADE_1_2_Q));
    let p3, q3, f3 = scientific::pade_approx(interp_more_pins::PADE_COEFF, 3, 1);
    assert(f3.kind == scientific::F_NONE);
    assert(buffer_near(p3, interp_more_pins::PADE_3_1_P));
    assert(buffer_near(q3, interp_more_pins::PADE_3_1_Q));
    // The coefficients are too few for the two orders.
    let e1, g1, h1 = scientific::pade_approx([1.0, 2.0], 2, 2);
    assert(h1.kind == scientific::F_SHAPE);
    // An order stands below zero.
    let e2, g2, h2 = scientific::pade_approx(interp_more_pins::PADE_COEFF, -1, 2);
    assert(h2.kind == scientific::F_RANGE);
    return;
}

// The Taylor polynomial of the function gives the coefficients from
// the highest power to the lowest one.
fn test_taylor_approximate() {
    let c, f = scientific::taylor_approximate(0.5, 3, 1.0) { xs => [common::mathshim_exp(xs[0]), common::mathshim_exp(xs[1]), common::mathshim_exp(xs[2]), common::mathshim_exp(xs[3])] };
    assert(f.kind == scientific::F_NONE && buffer_near(c, interp_more_pins::TAYLOR_COEF));
    // The degree stands below zero.
    let e1, g1 = scientific::taylor_approximate(0.5, -1, 1.0) { xs => xs };
    assert(g1.kind == scientific::F_RANGE);
    // The width of the arc must be positive.
    let e2, g2 = scientific::taylor_approximate(0.5, 3, 0.0) { xs => xs };
    assert(g2.kind == scientific::F_VALUE);
    return;
}

// The algebra of the spline gives the derivative, the antiderivative,
// the integral, every derivative and the roots.
fn test_spline_algebra() {
    let b, f = scientific::splrep(interp_more_pins::SPL_X, interp_more_pins::SPL_Y, 3);
    assert(f.kind == scientific::F_NONE);
    let v, fv = scientific::splev(interp_more_pins::SPL_QUERY, b);
    assert(fv.kind == scientific::F_NONE && common::all_near(v, interp_more_pins::SPL_VALUES));
    let d1, fd1 = scientific::splev(interp_more_pins::SPL_QUERY, b, 1);
    assert(fd1.kind == scientific::F_NONE && common::all_near(d1, interp_more_pins::SPL_DER1));
    // The derivative spline holds the knots and the coefficients of
    // the first derivative.
    let dv, fdv = scientific::splder(b, 1);
    assert(fdv.kind == scientific::F_NONE);
    let dk = scientific::bs_knots_of(dv);
    let dc = scientific::bs_coefs_of(dv);
    assert(dk.len == interp_more_pins::SPLDER_KNOTS.len);
    var i: i64 = 0;
    while i < dk.len {
        assert(common::near(dk[i], interp_more_pins::SPLDER_KNOTS[i]));
        i += 1;
    }
    i = 0;
    while i < dc.len {
        assert(common::near(dc[i], interp_more_pins::SPLDER_COEFS[i]));
        i += 1;
    }
    // The antiderivative spline holds the knots and the coefficients.
    let av, fav = scientific::splantider(b, 1);
    assert(fav.kind == scientific::F_NONE);
    let ak = scientific::bs_knots_of(av);
    let ac = scientific::bs_coefs_of(av);
    i = 0;
    while i < ak.len {
        assert(common::near(ak[i], interp_more_pins::SPLANTI_KNOTS[i]));
        i += 1;
    }
    i = 0;
    while i < ac.len {
        assert(common::near(ac[i], interp_more_pins::SPLANTI_COEFS[i]));
        i += 1;
    }
    // The integral of the spline over the interval.
    let s, fs = scientific::splint(0.5, 2.5, b);
    assert(fs.kind == scientific::F_NONE && common::near(s, interp_more_pins::SPL_INT[0]));
    // Every derivative at the points.
    let sd, fsd = scientific::spalde(b, interp_more_pins::SPL_QUERY);
    assert(fsd.kind == scientific::F_NONE && common::all_near(sd, interp_more_pins::SPL_SPALDE));
    // The roots of the spline.
    let r, fr = scientific::sproot(b);
    assert(fr.kind == scientific::F_NONE);
    assert(r.data.len == interp_more_pins::SPL_ROOT.len);
    i = 0;
    while i < r.data.len {
        assert(common::near(r.data[i], interp_more_pins::SPL_ROOT[i]));
        i += 1;
    }
    // The order of the derivative stands above the degree.
    let e1, g1 = scientific::splder(b, 5);
    assert(g1.kind == scientific::F_VALUE);
    // The ends of the interval must rise.
    let e2, g2 = scientific::splint(2.5, 0.5, b);
    assert(g2.kind == scientific::F_VALUE);
    // A point outside the base interval gives a fault.
    let e3, g3 = scientific::spalde(b, [4.0]);
    assert(g3.kind == scientific::F_RANGE);
    return;
}

// The fit with least squares holds the coefficients of the spline.
fn test_make_lsq_spline() {
    let b, f = scientific::make_lsq_spline(interp_more_pins::LSQ_X, interp_more_pins::LSQ_Y, interp_more_pins::LSQ_KNOTS, 3, interp_more_pins::LSQ_W);
    assert(f.kind == scientific::F_NONE);
    let c = scientific::bs_coefs_of(b);
    assert(buffer_near(c, interp_more_pins::LSQ_COEFS));
    // The knots are too few for the degree.
    let e1, g1 = scientific::make_lsq_spline(interp_more_pins::LSQ_X, interp_more_pins::LSQ_Y, [0.0, 1.0, 2.0], 3);
    assert(g1.kind == scientific::F_SHAPE);
    return;
}

// The spline of two variables on a grid holds the knots of the two
// axes and gives the values and the derivatives.
fn test_rect_spline() {
    let g, f = scientific::rect_spline(interp_more_pins::GRID_X, interp_more_pins::GRID_Y, interp_more_pins::GRID_Z, 3, 3);
    assert(f.kind == scientific::F_NONE);
    let kx = scientific::grid2_knots_x(g);
    let ky = scientific::grid2_knots_y(g);
    var i: i64 = 0;
    while i < kx.len {
        assert(common::near(kx[i], interp_more_pins::GRID_KNOTS_X[i]));
        i += 1;
    }
    i = 0;
    while i < ky.len {
        assert(common::near(ky[i], interp_more_pins::GRID_KNOTS_Y[i]));
        i += 1;
    }
    let v, fv = scientific::rect_spline_eval(g, interp_more_pins::GRID_QUERY_X, interp_more_pins::GRID_QUERY_Y);
    assert(fv.kind == scientific::F_NONE && common::all_near(v, interp_more_pins::GRID_VALUE));
    let dx, fdx = scientific::rect_spline_eval(g, interp_more_pins::GRID_QUERY_X, interp_more_pins::GRID_QUERY_Y, 1, 0);
    assert(fdx.kind == scientific::F_NONE && common::all_near(dx, interp_more_pins::GRID_DX));
    let dy, fdy = scientific::rect_spline_eval(g, interp_more_pins::GRID_QUERY_X, interp_more_pins::GRID_QUERY_Y, 0, 1);
    assert(fdy.kind == scientific::F_NONE && common::all_near(dy, interp_more_pins::GRID_DY));
    // The values need the product of the counts of the two axes.
    let e1, g1 = scientific::rect_spline(interp_more_pins::GRID_X, interp_more_pins::GRID_Y, [1.0, 2.0], 3, 3);
    assert(g1.kind == scientific::F_SHAPE);
    // The axes must rise.
    let e2, g2 = scientific::rect_spline([1.0, 0.0, 0.5, 0.75, 1.0], interp_more_pins::GRID_Y, interp_more_pins::GRID_Z, 3, 3);
    assert(g2.kind == scientific::F_VALUE);
    return;
}

// The one call form of PCHIP gives the values of the shape keeping
// spline.
fn test_pchip_interpolate() {
    let v, f = scientific::pchip_interpolate(interp_more_pins::BARY_NODES, interp_more_pins::BARY_VALUES, interp_more_pins::BARY_QUERY);
    assert(f.kind == scientific::F_NONE);
    // The spline passes through the points.
    let w, fw = scientific::pchip_interpolate(interp_more_pins::BARY_NODES, interp_more_pins::BARY_VALUES, interp_more_pins::BARY_NODES);
    assert(fw.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < interp_more_pins::BARY_NODES.len {
        assert(common::near(w.data[i], interp_more_pins::BARY_VALUES[i]));
        i += 1;
    }
    assert(common::all_near(v, interp_more_pins::PCHIP_VALUE));
    return;
}

// A spline that a caller built by hand and broke does not agree with
// its knots. The calls that read the buffer of a spline give a fault.
fn test_spline_broken() {
    // The degree stands above the count of the knots.
    let bad = scientific::BSpl { k: 3, nt: 2, buf: [0.0, 1.0, 2.0] };
    let s, f1 = scientific::splint(0.5, 2.5, bad);
    assert(f1.kind == scientific::F_SHAPE);
    let d, f2 = scientific::spalde(bad, [1.0]);
    assert(f2.kind == scientific::F_SHAPE);
    let r, f3 = scientific::sproot(bad);
    assert(f3.kind == scientific::F_SHAPE);
    // The buffer of the spline holds fewer values than its knots.
    let short = scientific::BSpl { k: 1, nt: 4, buf: [0.0] };
    let s2, f4 = scientific::splint(0.5, 2.5, short);
    assert(f4.kind == scientific::F_SHAPE);
}
