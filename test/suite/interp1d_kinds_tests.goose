// The tests of the group interp1d_kinds. Run them with SCIENTIFIC_SUITES=interp1d_kinds.
namespace interp1d_kinds_tests;

import scientific from "../..";
import os;
import common;

// interp1d takes the names zero and slinear and the fill values of SciPy.
fn test_interp1d_kinds() {
    let x = [0.0, 1.0, 2.5, 3.0, 4.5];
    let y = [1.0, 3.0, 2.0, 5.0, 4.0];
    let xs = [0.0, 0.4, 1.0, 1.7, 2.5, 2.9, 3.6, 4.5];
    let z, f1 = scientific::interp1d(x, y, xs, "zero");
    assert(f1.kind == scientific::F_NONE && common::all_near(z, [1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 5.0, 4.0]));
    let sl, f2 = scientific::interp1d(x, y, xs, "slinear");
    assert(f2.kind == scientific::F_NONE && common::all_loose(sl, [1.0, 1.8, 3.0, 2.53333333, 2.0, 4.4, 4.6, 4.0]));
    let out = [-1.0, 5.0];
    let a, f3 = scientific::interp1d(x, y, out, "linear", false, true, -7.0, -8.0);
    assert(f3.kind == scientific::F_NONE && common::all_near(a, [-7.0, -8.0]));
    let b, f4 = scientific::interp1d(x, y, out, "zero", true);
    assert(f4.kind == scientific::F_NONE && common::all_near(b, [1.0, 4.0]));
    let qq, fq = scientific::interp1d(x, y, xs, "quadratic");
    assert(fq.kind == scientific::F_NONE && common::all_loose(qq, [1.0, 2.32643741, 3.0, 1.78974782, 2.0, 4.44104539, 6.7114718, 4.0]));
    let qe, fqe = scientific::interp1d(x, y, out, "quadratic", true);
    assert(fqe.kind == scientific::F_NONE && common::all_loose(qe, [-5.38697845, -0.2434663]));
    let c, f5 = scientific::interp1d(x, y, out, "linear");
    assert(f5.kind == scientific::F_RANGE);
    return;
}
