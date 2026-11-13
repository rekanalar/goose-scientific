// The tests of the group cfactor. Run them with SCIENTIFIC_SUITES=cfactor.
namespace cfactor_tests;

import scientific from "../..";
import os;
import common;

// test_cfactor_scale tests entries near the limits of the number format.
fn test_cfactor_scale() {
    let big = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0e300, 1.0e300, 1.0e300, 0.0, 0.0, 1.0e300, 1.0e300, 1.0e300] };
    let q, r, f = scientific::cqr(big);
    assert(f.kind == scientific::F_NONE);
    assert(common::close(r.data[0] / 1.0e300, -1.7320508075688772, 1.0e-12));
    assert(common::close(r.data[6] / 1.0e300, -0.5773502691896257, 1.0e-12));
    let u, s, vh, g = scientific::csvd(big);
    assert(g.kind == scientific::F_NONE);
    assert(common::close(s.data[0] / 1.0e300, 2.414213562373095, 1.0e-12));
    assert(common::close(s.data[1] / 1.0e300, 0.41421356237309503, 1.0e-12));
    let small = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0e-300, 1.0e-300, 1.0e-300, 0.0, 0.0, 1.0e-300, 1.0e-300, 1.0e-300] };
    let us, ss, vs, gs = scientific::csvd(small);
    assert(gs.kind == scientific::F_NONE);
    assert(common::close(ss.data[0] / 1.0e-300, 2.414213562373095, 1.0e-12));
    return;
}
