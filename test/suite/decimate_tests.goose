// The tests of the group decimate. Run them with SCIENTIFIC_SUITES=decimate.
namespace decimate_tests;

import scientific from "../..";
import os;
import common;

// decimate is pinned against SciPy.
fn test_decimate() {
    let x = common::dc_signal();
    let a, f1 = scientific::decimate(x, 3);
    assert(f1.kind == scientific::F_NONE && a.data.len == 20);
    assert(common::loose(a.data[1], 1.0729684936013582) && common::loose(a.data[2], 1.5557205343558305) && common::loose(a.data[5], 0.5152653616743713));
    let b, f2 = scientific::decimate(x, 2, "fir");
    assert(f2.kind == scientific::F_NONE && b.data.len == 30);
    assert(common::loose(b.data[0], 0.06383926367780504) && common::loose(b.data[1], 0.754324189814132) && common::loose(b.data[5], 1.1396273776282466));
    let c, f3 = scientific::decimate(x, 4, "iir", 6);
    assert(f3.kind == scientific::F_NONE && common::loose(c.data[1], 1.326462621204827) && common::loose(c.data[3], 0.7518117215075588));
    let d, f4 = scientific::decimate(x, 0);
    assert(f4.kind == scientific::F_RANGE);
    let e, f5 = scientific::decimate(x, 2, "bessel");
    assert(f5.kind == scientific::F_VALUE);
    return;
}
