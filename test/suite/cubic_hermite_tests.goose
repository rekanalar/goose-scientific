// The tests of the group cubic_hermite. Run them with SCIENTIFIC_SUITES=cubic_hermite.
namespace cubic_hermite_tests;

import scientific from "../..";
import os;
import common;

// cubic_hermite is pinned against SciPy CubicHermiteSpline.
fn test_cubic_hermite() {
    let x = [0.0, 1.0, 2.5, 4.0];
    let y = [1.0, 3.0, 2.0, 5.0];
    let d = [0.5, -1.0, 2.0, 0.0];
    let c, f = scientific::cubic_hermite(x, y, d);
    assert(f.kind == scientific::F_NONE);
    let v, fv = scientific::spline_eval(c, [0.0, 0.5, 1.0, 1.7, 2.5, 3.3, 4.0, 5.0]);
    assert(fv.kind == scientific::F_NONE);
    let ex = [1.0, 2.1875, 3.0, 2.0023703703703704, 2.0, 3.9982222222222212, 5.0, 1.4444444444444446];
    var i: i64 = 0;
    while i < 8 { assert(common::mathshim_abs(v.data[i] - ex[i]) < 1.0e-12); i += 1; }
    let d1, vf1 = scientific::spline_eval(c, [0.5], 1);
    assert(common::mathshim_abs(d1.data[0] - 3.125) < 1.0e-12);
    let b1, g1 = scientific::cubic_hermite(x, y, [0.0, 1.0]);
    assert(g1.kind == scientific::F_SHAPE);
    let b2, g2 = scientific::cubic_hermite(x, y, [0.0, 1.0, 1.0 / 0.0, 0.0]);
    assert(g2.kind == scientific::F_VALUE);
    let b3, g3 = scientific::cubic_hermite([0.0, 0.0], [1.0, 2.0], [0.0, 0.0]);
    assert(g3.kind == scientific::F_VALUE);
}
