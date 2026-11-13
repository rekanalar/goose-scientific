// The tests of the group dblquad. Run them with SCIENTIFIC_SUITES=dblquad.
namespace dblquad_tests;

import scientific from "../..";
import os;
import common;

// dblquad is pinned against SciPy.
fn test_dblquad() {
    let v1, e1, f1 = scientific::dblquad(0.0, 2.0, common::dbl_lo1, common::dbl_hi1) { y, x => common::mathshim_exp(-x * y) * common::mathshim_sin(x + y) };
    assert(f1.kind == scientific::F_NONE && common::near(v1, 0.7360958517002549) && e1 >= 0.0);
    let v2, e2, f2 = scientific::dblquad(1.0, 3.0, common::dbl_zero, common::dbl_sq) { y, x => x * y * y };
    assert(f2.kind == scientific::F_NONE && common::near(v2, 273.33333333333337));
    let v3, e3, f3 = scientific::dblquad(0.0, 1.0, common::dbl_m1, common::dbl_p1) { y, x => 1.0 / (1.0 + x * x + y * y) };
    assert(f3.kind == scientific::F_NONE && common::near(v3, 1.2790207037406225));
    // The limits in the wrong order change the sign.
    let v4, e4, f4 = scientific::dblquad(3.0, 1.0, common::dbl_zero, common::dbl_sq) { y, x => x * y * y };
    assert(f4.kind == scientific::F_NONE && common::near(v4, -273.33333333333337));
    // A bad tolerance is a fault.
    let v5, e5, f5 = scientific::dblquad(0.0, 1.0, -1.0, 1.49e-8, 50, common::dbl_zero, common::dbl_sq) { y, x => x * y };
    assert(f5.kind == scientific::F_VALUE);
    // Infinite limits of x and of y.
    let v6, e6, f6 = scientific::dblquad(0.0, 1.0 / 0.0, common::dbl_zero, common::dbl_inf) { y, x => common::mathshim_exp(-x - y) };
    assert(f6.kind == scientific::F_NONE && common::mathshim_abs(v6 - 1.0) < 1.0e-7);
    let v7, e7, f7 = scientific::dblquad(-(1.0 / 0.0), 1.0 / 0.0, common::dbl_zero, common::dbl_inf) { y, x => common::mathshim_exp(-x * x - y) };
    assert(f7.kind == scientific::F_NONE && common::mathshim_abs(v7 - 1.772453850905264) < 1.0e-6);
    // A missing value of a limit is a fault.
    let v8, e8, f8 = scientific::dblquad(0.0 / 0.0, 1.0, common::dbl_zero, common::dbl_sq) { y, x => x * y };
    assert(f8.kind == scientific::F_VALUE);
    return;
}
