// The tests of the group bspline_design. Run them with SCIENTIFIC_SUITES=bspline_design.
namespace bspline_design_tests;

import scientific from "../..";
import os;
import common;

// The design matrix of a B-spline, pinned against SciPy.
fn test_bspline_design_matrix() {
    let t = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
    let m, fm = scientific::bspline_design_matrix([-1.0, 0.0, 0.5, 2.5, 3.0, 5.0], t, 3, "yes");
    assert(fm.kind == scientific::F_NONE && m.rows == 6 && m.cols == 6);
    let d, fd = scientific::csr_to_dense(m);
    assert(fd.kind == scientific::F_NONE);
    assert(common::all_near(d, [8.0, -9.25, 2.4166666666666665, -0.16666666666666666, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.125, 0.59375, 0.26041666666666663, 0.020833333333333332, 0.0, 0.0, 0.0, 0.0, 0.020833333333333332, 0.26041666666666663, 0.59375, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.3333333333333333, 13.333333333333332, -38.0, 27.0]));
    let m2, f2 = scientific::bspline_design_matrix([1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);
    let d2, f3 = scientific::csr_to_dense(m2);
    assert(f2.kind == scientific::F_NONE && f3.kind == scientific::F_NONE);
    assert(common::all_near(d2, [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]));
    let m3, f4 = scientific::bspline_design_matrix([3.5, 4.5], t, 3, "periodic");
    let d3, f5 = scientific::csr_to_dense(m3);
    assert(f4.kind == scientific::F_NONE && f5.kind == scientific::F_NONE);
    assert(common::all_near(d3, [0.125, 0.59375, 0.26041666666666663, 0.020833333333333332, 0.0, 0.0, 0.0, 0.03125, 0.46875, 0.46875, 0.03125, 0.0]));
    // Faults.
    let e1, g1 = scientific::bspline_design_matrix([4.0], t, 3, "no");
    assert(g1.kind == scientific::F_RANGE);
    let e2, g2 = scientific::bspline_design_matrix([0.5], t, 3, "maybe");
    assert(g2.kind == scientific::F_VALUE);
    let e3, g3 = scientific::bspline_design_matrix([4.0], t, 3);
    assert(g3.kind == scientific::F_RANGE);
    let e4, g4 = scientific::bspline_design_matrix([0.5], [0.0, 1.0, 2.0], 3, "no");
    assert(g4.kind == scientific::F_SHAPE);
    let e5, g5 = scientific::bspline_design_matrix([0.5], t, -1, "no");
    assert(g5.kind == scientific::F_RANGE);
}
