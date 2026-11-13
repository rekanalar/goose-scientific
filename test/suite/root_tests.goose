// The tests of the group root. Run them with SCIENTIFIC_SUITES=root.
namespace root_tests;

import scientific from "../..";
import os;
import common;

// root is pinned against SciPy for four methods and the faults.
fn test_root() {
    let ms = ["hybr", "lm", "broyden1", "broyden2"];
    var k: i64 = 0;
    while k < 4 {
        let x, fv, f = scientific::root([2.0, 0.5], ms[k], 1.0e-10, 200) { v => [v[0] * v[0] + v[1] * v[1] - 4.0, v[0] * v[1] - 1.0] };
        assert(f.kind == scientific::F_NONE);
        assert(common::mathshim_abs(x.data[0] - 1.93185165) < 1.0e-7 && common::mathshim_abs(x.data[1] - 0.51763809) < 1.0e-7);
        assert(common::mathshim_abs(fv.data[0]) < 1.0e-6 && common::mathshim_abs(fv.data[1]) < 1.0e-6);
        k += 1;
    }
    // The simple iterations reach the root that SciPy finds for this
    // system: 0.8411639, 0.1588361.
    let ms2 = ["linearmixing", "diagbroyden", "excitingmixing", "anderson"];
    k = 0;
    while k < 4 {
        let xm, fm, gm = scientific::root([0.0, 0.0], ms2[k], 1.0e-9, 5000) { v => [v[0] + 0.5 * (v[0] - v[1]) * (v[0] - v[1]) * (v[0] - v[1]) - 1.0, 0.5 * (v[1] - v[0]) * (v[1] - v[0]) * (v[1] - v[0]) + v[1]] };
        assert(gm.kind == scientific::F_NONE);
        assert(common::mathshim_abs(xm.data[0] - 0.8411639) < 1.0e-6 && common::mathshim_abs(xm.data[1] - 0.1588361) < 1.0e-6);
        k += 1;
    }
    // The method df-sane takes no derivative. Its tolerance is relative
    // to the first residual.
    let xd, fd, gd = scientific::root([0.0, 0.0], "df-sane", 1.0e-10, 500) { v => [v[0] + 0.5 * (v[0] - v[1]) * (v[0] - v[1]) * (v[0] - v[1]) - 1.0, 0.5 * (v[1] - v[0]) * (v[1] - v[0]) * (v[1] - v[0]) + v[1]] };
    assert(gd.kind == scientific::F_NONE);
    assert(common::mathshim_abs(xd.data[0] - 0.8411639) < 1.0e-6 && common::mathshim_abs(xd.data[1] - 0.1588361) < 1.0e-6);
    // The method krylov uses a Krylov iteration for the linear step.
    // SciPy reaches the same root.
    let xk, fk2, gk = scientific::root([2.0, 0.5], "krylov", 1.0e-10, 200) { v => [v[0] * v[0] + v[1] * v[1] - 4.0, v[0] * v[1] - 1.0] };
    assert(gk.kind == scientific::F_NONE);
    assert(common::mathshim_abs(xk.data[0] - 1.93185165) < 1.0e-6 && common::mathshim_abs(xk.data[1] - 0.51763809) < 1.0e-6);
    let m1, m2, mg = scientific::root([1.0, 2.0], "anderson") { v => [v[0]] };
    assert(mg.kind == scientific::F_SHAPE);
    let x3, f3v, g3 = scientific::root([1.0, 1.0, 1.0]) { v => [v[0] + v[1] + v[2] - 6.0, v[0] * v[1] - 2.0, v[2] - v[0] - 1.0] };
    assert(g3.kind == scientific::F_NONE);
    assert(common::mathshim_abs(x3.data[0] - 2.0) < 1.0e-6 && common::mathshim_abs(x3.data[1] - 1.0) < 1.0e-6 && common::mathshim_abs(x3.data[2] - 3.0) < 1.0e-6);
    let b1, b2, g4 = scientific::root([1.0], "newton") { v => [v[0]] };
    assert(g4.kind == scientific::F_VALUE);
    let b3, b4, g5 = scientific::root([1.0], "hybr", 0.0) { v => [v[0]] };
    assert(g5.kind == scientific::F_VALUE);
    let b5, b6, g6 = scientific::root([1.0, 2.0], "broyden1") { v => [v[0]] };
    assert(g6.kind == scientific::F_SHAPE);
    let b7, b8, g7 = scientific::root([1.0], "broyden1", 1.0e-12, 3) { v => [v[0] * v[0] * v[0] - 1000.0] };
    assert(g7.kind == scientific::F_LIMIT);
    let b9, b10, g8 = scientific::root([], "hybr") { v => [v[0]] };
    assert(g8.kind == scientific::F_VALUE);
}
