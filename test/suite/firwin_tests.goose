// The tests of the group firwin. Run them with SCIENTIFIC_SUITES=firwin.
namespace firwin_tests;

import scientific from "../..";
import os;
import common;

// firwin2 is pinned against SciPy.
fn test_firwin2() {
    let a, f1 = scientific::firwin2(9, [0.0, 0.3, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0]);
    assert(f1.kind == scientific::F_NONE);
    assert(common::all_loose(a, [-0.004709708691207965, -0.01152260218143383, 0.048068435342338324, 0.25764638184042365, 0.3984375, 0.2576463818404236, 0.04806843534233829, -0.011522602181433838, -0.004709708691207965]));
    let b, f2 = scientific::firwin2(8, [0.0, 0.5, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0], 0, "none");
    assert(f2.kind == scientific::F_NONE);
    assert(common::all_loose(b, [-0.05385075661518586, -0.08268148379274896, 0.14568866650727913, 0.44871097697629203, 0.44871097697629203, 0.1456886665072791, -0.08268148379274898, -0.053850756615185874]));
    let c, f3 = scientific::firwin2(11, [0.0, 0.5, 1.0], [0.0, 1.0, 0.0], 64, "hamming", true);
    assert(f3.kind == scientific::F_NONE && common::loose(c.data[0], 0.001293533458552159) && common::loose(c.data[2], -0.017899195736421628) && common::loose(c.data[4], 0.3696412747256351) && c.data[5] == 0.0 && common::loose(c.data[6], -0.36964127472563496));
    let d, f4 = scientific::firwin2(11, [0.0, 500.0, 1000.0], [0.0, 1.0, 0.0], 0, "hann", false, 2000.0);
    assert(f4.kind == scientific::F_NONE && d.data.len == 11 && common::loose(d.data[3], -0.13434882372563778) && common::loose(d.data[5], 0.5) && common::loose(d.data[7], -0.13434882372563778));
    let e, f5 = scientific::firwin2(9, [0.1, 1.0], [1.0, 0.0]);
    assert(f5.kind == scientific::F_VALUE);
    return;
}
