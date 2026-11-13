// The tests of the group interpolate. Run them with SCIENTIFIC_SUITES=interpolate.
namespace interpolate_tests;

import scientific from "../..";
import os;
import common;

fn test_interpolate() {
    // The spline through four points.
    let sp, f1 = scientific::spline([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 1.0]);
    assert(f1.kind == scientific::F_NONE);
    let v1, f2 = scientific::spline_value(sp, 0.5);
    assert(f2.kind == scientific::F_NONE && common::close(v1, 0.75));
    let v2, f3 = scientific::spline_value(sp, 1.5);
    assert(f3.kind == scientific::F_NONE && common::close(v2, 0.5));
    let v3, f4 = scientific::spline_value(sp, 2.5);
    assert(f4.kind == scientific::F_NONE && common::close(v3, 0.25));
    // The values at a row of points.
    let vs, f5 = scientific::spline_values(sp, [0.25, 1.75]);
    assert(f5.kind == scientific::F_NONE);
    assert(common::close(vs.data[0], 0.40625) && common::close(vs.data[1], 0.1875));
    // The spline passes through the points of the signal.
    let v4, f6 = scientific::spline_value(sp, 1.0);
    assert(f6.kind == scientific::F_NONE && v4 == 1.0);
    let v5, f7 = scientific::spline_value(sp, 3.0);
    assert(f7.kind == scientific::F_NONE && v5 == 1.0);
    // Points that do not rise are a fault.
    let bad1, fb1 = scientific::spline([0.0, 1.0, 1.0], [0.0, 1.0, 2.0]);
    assert(fb1.kind == scientific::F_VALUE);
    // Values of a different count are a fault.
    let bad2, fb2 = scientific::spline([0.0, 1.0, 2.0], [0.0, 1.0]);
    assert(fb2.kind == scientific::F_SHAPE);
    // A signal of too few points is a fault.
    let bad3, fb3 = scientific::spline([0.0, 1.0], [0.0, 1.0]);
    assert(fb3.kind == scientific::F_VALUE);
    // A point outside the signal is a fault.
    let bad4, fb4 = scientific::spline_value(sp, -1.0);
    assert(fb4.kind == scientific::F_RANGE);
    let bad5, fb5 = scientific::spline_value(sp, 4.0);
    assert(fb5.kind == scientific::F_RANGE);
}
