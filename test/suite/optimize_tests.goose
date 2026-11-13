// The tests of the group optimize. Run them with SCIENTIFIC_SUITES=optimize.
namespace optimize_tests;

import scientific from "../..";
import os;
import common;

fn test_optimize() {
    // The root of the square of x and two.
    let r1, f1 = scientific::brentq(0.0, 2.0, 1e-12, 100) { x => x * x - 2.0 };
    assert(f1.kind == scientific::F_NONE);
    assert(common::close(r1, 1.4142135623731364));
    let r2, f2 = scientific::brenth(0.0, 2.0, 1e-12, 100) { x => x * x - 2.0 };
    assert(f2.kind == scientific::F_NONE);
    assert(common::close(r2, 1.414213562373095));
    // The root of a cosine, which lies between the two points.
    let r3, f3 = scientific::brentq(1.0, 4.0, 1e-12, 100) { x => mathshim::cosine(x) };
    assert(f3.kind == scientific::F_NONE);
    assert(common::close(r3, 1.570796326794891));
    // The root of a cubic, which lies between the two points.
    let r5, f5 = scientific::brentq(1.0, 3.0, 1e-12, 100) { x => x * x * x - 2.0 * x - 5.0 };
    assert(f5.kind == scientific::F_NONE);
    assert(common::close(r5, 2.0945514815423265));
    // The method of Newton gives the same root.
    let r4, f4 = scientific::newton(1.0, 1e-12, 100, common::dquad) { x => x * x - 2.0 };
    assert(f4.kind == scientific::F_NONE);
    assert(common::close(r4, 1.4142135623730951));
    // The points that hold no change of sign are a fault.
    let bad1, fb1 = scientific::brentq(3.0, 4.0, 1e-12, 100) { x => x * x - 2.0 };
    assert(fb1.kind == scientific::F_VALUE);
    let bad2, fb2 = scientific::brenth(3.0, 4.0, 1e-12, 100) { x => x * x - 2.0 };
    assert(fb2.kind == scientific::F_VALUE);
    // A tolerance of zero is a fault.
    let bad3, fb3 = scientific::brentq(0.0, 2.0, 0.0, 100) { x => x * x - 2.0 };
    assert(fb3.kind == scientific::F_VALUE);
    // A count of turns below one is a fault.
    let bad4, fb4 = scientific::brentq(0.0, 2.0, 1e-12, 0) { x => x * x - 2.0 };
    assert(fb4.kind == scientific::F_VALUE);
    // A count of turns that is too small is a fault.
    let bad5, fb5 = scientific::brentq(0.0, 2.0, 1e-12, 2) { x => x * x - 2.0 };
    assert(fb5.kind == scientific::F_LIMIT);
    // A derivative of zero is a fault.
    let bad6, fb6 = scientific::newton(1.0, 1e-12, 100, common::dflat) { x => x * x - 2.0 };
    assert(fb6.kind == scientific::F_VALUE);
    // The root of two equations at once.
    // The secant method takes two points.
    let r4b, f4b = scientific::secant(1.0, 1.5, 1e-12, 100) { x => x * x - 2.0 };
    assert(f4b.kind == scientific::F_NONE && common::close(r4b, 1.4142135623730951, 1e-10));
    let r4c, f4c = scientific::secant(0.0, 1.0, 1e-12, 100) { x => x * x * x - 2.0 * x + 2.0 };
    assert(f4c.kind == scientific::F_NONE && common::close(r4c, -1.7692923542386314, 1e-10));
    // The method of Halley takes the second derivative.
    let r4d, f4d = scientific::halley(1.0, 1e-12, 100, common::dquad, common::dsecond) { x => x * x - 2.0 };
    assert(f4d.kind == scientific::F_NONE && common::close(r4d, 1.4142135623730951, 1e-10));
    let bad4b, fb4b = scientific::secant(1.0, 1.0, 1e-12, 100) { x => x * x - 2.0 };
    assert(fb4b.kind != scientific::F_NONE);
    let bad4c, fb4c = scientific::secant(1.0, 1.5, 0.0, 100) { x => x * x - 2.0 };
    assert(fb4c.kind != scientific::F_NONE);
    let bad4d, fb4d = scientific::secant(1.0, 1.5, 1e-12, 0) { x => x * x - 2.0 };
    assert(fb4d.kind != scientific::F_NONE);
    let bad4e, fb4e = scientific::halley(1.0, 1e-12, 100, common::dflat, common::dsecond) { x => x * x - 2.0 };
    assert(fb4e.kind != scientific::F_NONE);
    let bad4f, fb4f = scientific::halley(1.0, 0.0, 100, common::dquad, common::dsecond) { x => x * x - 2.0 };
    assert(fb4f.kind != scientific::F_NONE);
    let bad4g, fb4g = scientific::halley(1.0, 1e-12, 0, common::dquad, common::dsecond) { x => x * x - 2.0 };
    assert(fb4g.kind != scientific::F_NONE);
    let r5, f5 = scientific::fsolve([1.0, 1.0], 1e-10, 50) { v => [v[0] * v[0] + v[1] * v[1] - 4.0, v[0] - v[1]] };
    assert(f5.kind == scientific::F_NONE);
    assert(r5.data.len == 2 && common::close(r5.data[0], 1.4142135623730951));
    assert(common::close(r5.data[1], 1.4142135623730951));
    // A function that gives a different count of values is a fault.
    let bad7, fb7 = scientific::fsolve([1.0, 1.0], 1e-10, 50) { v => [v[0] * v[0] - 2.0] };
    assert(fb7.kind == scientific::F_SHAPE);
    // A count of turns that is too small is a fault.
    let bad8, fb8 = scientific::fsolve([1.0, 1.0], 1e-14, 1) { v => [v[0] * v[0] + v[1] * v[1] - 4.0, v[0] - v[1]] };
    assert(fb8.kind == scientific::F_LIMIT);
}
