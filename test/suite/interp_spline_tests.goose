// The tests of the group interp_spline. Run them with SCIENTIFIC_SUITES=interp_spline.
namespace interp_spline_tests;

import scientific from "../..";
import os;
import common;

// B-spline interpolation with end derivatives, pinned against SciPy.
fn test_interp_spline_bc() {
    let kx = [0.0, 0.7, 1.5, 2.4, 4.0];
    let ky = [1.0, 2.0, 0.5, 3.0, 2.2];
    let q = [0.3, 1.0, 3.0, 4.0];
    let e0: i64[0] = [];
    let f0: f64[0] = [];
    let a, fa = scientific::make_interp_spline_bc(kx, ky, 3, [1], [0.0], [1], [0.0]);
    let va, ea = scientific::bspline_eval(a, q, 0);
    assert(fa.kind == scientific::F_NONE && ea.kind == scientific::F_NONE && common::all_near(va, [1.4051118583571318, 1.5108523428116984, 3.312696206422242, 2.2]));
    let b, fb = scientific::make_interp_spline_bc(kx, ky, 3, [2], [0.0], [2], [0.0]);
    let vb, eb = scientific::bspline_eval(b, q, 0);
    assert(fb.kind == scientific::F_NONE && common::all_near(vb, [1.7098999308782359, 1.4199408943790213, 3.694483387911039, 2.2]));
    let c, fc = scientific::make_interp_spline_bc(kx, ky, 3, [1], [0.4], [2], [-0.6]);
    let vc, ec = scientific::bspline_eval(c, q, 0);
    assert(fc.kind == scientific::F_NONE && common::all_near(vc, [1.4506489986030193, 1.5068019484608233, 3.7633453622296695, 2.2]));
    let d, fd = scientific::make_interp_spline_bc(kx, ky, 5, [1, 2], [0.4, 0.1], [1, 3], [0.2, 0.5]);
    let vd, ed = scientific::bspline_eval(d, q, 0);
    assert(fd.kind == scientific::F_NONE && common::all_near(vd, [1.3077339227890068, 1.7093460382631585, 3.7899517744721862, 2.2]));
    let e, fe = scientific::make_interp_spline_bc(kx, ky, 2, [1], [0.4], e0, f0);
    let ve, ee = scientific::bspline_eval(e, q, 0);
    assert(fe.kind == scientific::F_NONE && common::all_near(ve, [1.2522448979591836, 2.2497767857142863, 7.298511904761907, 2.2]));
    let g, fg = scientific::make_interp_spline_bc(kx, ky, 4, [1, 2], [0.4, 1.0], [1], [0.3]);
    let vg, eg = scientific::bspline_eval(g, q, 0);
    assert(fg.kind == scientific::F_NONE && common::all_near(vg, [1.2461846446754583, 2.1484239716784668, 6.7424247589120405, 2.2]));
    let h1, g1 = scientific::make_interp_spline_bc(kx, ky, 3, [1], [0.4], e0, f0);
    assert(g1.kind == scientific::F_SHAPE);
    let h2, g2 = scientific::make_interp_spline_bc(kx, ky, 3, [0], [0.4], [1], [0.0]);
    assert(g2.kind == scientific::F_VALUE);
    let h3, g3 = scientific::make_interp_spline_bc(kx, ky, 3, [1], [1.0 / 0.0], [1], [0.0]);
    assert(g3.kind == scientific::F_VALUE);
}
