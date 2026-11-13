// The tests of the group tplquad. Run them with SCIENTIFIC_SUITES=tplquad.
namespace tplquad_tests;

import scientific from "../..";
import os;
import common;

// tplquad is pinned against SciPy.
fn test_tplquad() {
    let v1, e1, f1 = scientific::tplquad(0.0, 1.0, 1.49e-8, 1.49e-8, 50, common::tq_zero, common::tq_one, common::tq_zero_xy, common::tq_one_xy) { z, y, x => x * y * z };
    assert(f1.kind == scientific::F_NONE && common::near(v1, 0.12499999999999999));
    let v2, e2, f2 = scientific::tplquad(0.0, 1.0, 1.49e-8, 1.49e-8, 50, common::tq_half_x, common::tq_one_x, common::tq_half_y, common::tq_y_half) { z, y, x => common::mathshim_exp(-x - y - z) };
    assert(f2.kind == scientific::F_NONE && common::near(v2, 0.14202386703691938));
    let v3, e3, f3 = scientific::tplquad(0.0, 1.0, 1.49e-8, 1.49e-8, 50, common::tq_m1, common::tq_p1, common::tq_m1_xy, common::tq_p1_xy) { z, y, x => 1.0 / (1.0 + x * x + y * y + z * z) };
    assert(f3.kind == scientific::F_NONE && common::near(v3, 2.143427031150922));
    // A bad tolerance is a fault.
    let v4, e4, f4 = scientific::tplquad(0.0, 1.0, -1.0, 1.49e-8, 50, common::tq_zero, common::tq_one, common::tq_zero_xy, common::tq_one_xy) { z, y, x => x * y * z };
    assert(f4.kind == scientific::F_VALUE);
    return;
}
