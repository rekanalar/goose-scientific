// The tests of the group minimize. Run them with SCIENTIFIC_SUITES=minimize.
namespace minimize_tests;

import scientific from "../..";
import os;
import common;

fn test_minimize() {
    // The smallest point of a parabola.
    let m1, f1 = scientific::minimize_scalar(-2.0, 2.0, 1e-10, 200) { x => (x - 0.5) * (x - 0.5) + 1.0 };
    assert(f1.kind == scientific::F_NONE);
    assert(common::close(m1.data[0], 0.4999999894781067));
    // The smallest point of a parabola with a slope.
    let m2, f2 = scientific::minimize_scalar(-5.0, 5.0, 1e-10, 200) { x => (x - 3.0) * (x - 3.0) + (x + 1.0) * (x + 1.0) };
    assert(f2.kind == scientific::F_NONE);
    assert(common::close(m2.data[0], 0.9999999865310867));
    // The smallest point of the exponential.
    let m3, f3 = scientific::minimize_scalar(0.0, 3.0, 1e-10, 200) { x => mathshim::power_e(x) - 2.0 * x };
    assert(f3.kind == scientific::F_NONE);
    assert(common::close(m3.data[0], 0.6931471680308768));
    // The smallest point of two variables.
    let m4, f4 = scientific::minimize([1.0, 1.0], 1e-10, 500) { v => [(v[0] - 1.0) * (v[0] - 1.0) + (v[1] + 2.0) * (v[1] + 2.0)] };
    assert(f4.kind == scientific::F_NONE);
    assert(m4.data.len == 2);
    assert(common::close(m4.data[0], 1.0000000000153233) && common::close(m4.data[1], -2.0000000000334754));
    // The fit of two equations.
    let l1, f5 = scientific::least_squares([0.0, 0.0], 1e-10, 200) { v => [v[0] + v[1] - 3.0, v[0] - v[1] - 1.0] };
    assert(f5.kind == scientific::F_NONE);
    assert(l1.data.len == 2);
    assert(common::close(l1.data[0], 1.999999999998008) && common::close(l1.data[1], 0.999999999999004));
    // A tolerance of zero is a fault.
    let bad1, fb1 = scientific::minimize_scalar(-2.0, 2.0, 0.0, 200) { x => x * x };
    assert(fb1.kind == scientific::F_VALUE);
    // A count of turns below one is a fault.
    let bad2, fb2 = scientific::minimize_scalar(-2.0, 2.0, 1e-10, 0) { x => x * x };
    assert(fb2.kind == scientific::F_VALUE);
    let bad3, fb3 = scientific::minimize([1.0], 0.0, 200) { v => [v[0] * v[0]] };
    assert(fb3.kind == scientific::F_VALUE);
    let bad4, fb4 = scientific::minimize([1.0], 1e-10, 0) { v => [v[0] * v[0]] };
    assert(fb4.kind == scientific::F_VALUE);
    // A start that holds no element is a fault.
    let bad5, fb5 = scientific::minimize([], 1e-10, 200) { v => [v[0] * v[0]] };
    assert(fb5.kind == scientific::F_VALUE);
    let bad6, fb6 = scientific::least_squares([], 1e-10, 200) { v => [v[0] * v[0]] };
    assert(fb6.kind == scientific::F_VALUE);
}
