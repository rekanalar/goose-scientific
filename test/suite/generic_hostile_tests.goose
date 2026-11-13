// The tests of the group generic_hostile. Run them with SCIENTIFIC_SUITES=generic_hostile.
namespace generic_hostile_tests;

import scientific from "../..";
import os;
import common_3;

// test_generic_hostile checks the faults of the filters that take a block.
fn test_generic_hostile() {
    let a = common_3::ni_nd([3, 4], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let big = 9223372036854775807;
    let small = -big - 1;
    let r1, h1 = scientific::generic_filter(a, [3, 3], "bogus") { w => common_3::gsum(w) };
    assert(h1.kind == scientific::F_VALUE);
    let r2, h2 = scientific::generic_filter(a, []) { w => common_3::gsum(w) };
    assert(h2.kind == scientific::F_SHAPE);
    let r3, h3 = scientific::generic_filter(a, [0, 3]) { w => common_3::gsum(w) };
    assert(h3.kind == scientific::F_VALUE);
    let r4, h4 = scientific::generic_filter(a, [3, 3], "reflect", 0.0, [2, 0]) { w => common_3::gsum(w) };
    assert(h4.kind == scientific::F_RANGE);
    let r5, h5 = scientific::generic_filter(a, [3, 3], "reflect", 0.0, [big, small]) { w => common_3::gsum(w) };
    assert(h5.kind == scientific::F_RANGE);
    let r6, h6 = scientific::generic_filter(a, [100000, 100000]) { w => common_3::gsum(w) };
    assert(h6.kind == scientific::F_LIMIT);
    let r7, h7 = scientific::generic_filter(a, [big, big]) { w => common_3::gsum(w) };
    assert(h7.kind != scientific::F_NONE);
    let r8, h8 = scientific::generic_filter_fp(a, common_3::ni_nd([2, 2], [0.0, 0.0, 0.0, 0.0])) { w => common_3::gsum(w) };
    assert(h8.kind == scientific::F_VALUE);
    let r9, h9 = scientific::generic_filter_fp(a, common_3::ni_nd([3], [1.0, 1.0, 1.0])) { w => common_3::gsum(w) };
    assert(h9.kind == scientific::F_SHAPE);
    var lie = scientific::clone_of(a);
    lie.data.pop();
    let r10, h10 = scientific::generic_filter(lie, [3, 3]) { w => common_3::gsum(w) };
    assert(h10.kind == scientific::F_SHAPE);
    var huge = scientific::clone_of(a);
    huge.shape[0] = 4611686018427387904;
    let r11, h11 = scientific::generic_filter(huge, [3, 3]) { w => common_3::gsum(w) };
    assert(h11.kind != scientific::F_NONE);
    // A block that gives NaN or an infinity passes the value through.
    let nan = 0.0 / 0.0;
    let r12, h12 = scientific::generic_filter(a, [1, 1]) { w => nan };
    assert(h12.kind == scientific::F_NONE && r12.data[0] != r12.data[0]);
    // A derivative block that gives an array of another size is a fault.
    let r13, h13 = scientific::generic_laplace(a) { x, ax, md, cv => common_3::gshort(copy(x), ax, md, cv) };
    assert(h13.kind == scientific::F_SHAPE);
    let r14, h14 = scientific::generic_gradient_magnitude(a) { x, ax, md, cv => common_3::gshort(copy(x), ax, md, cv) };
    assert(h14.kind == scientific::F_SHAPE);
    let r15, h15 = scientific::generic_laplace(a, "bogus") { x, ax, md, cv => common_3::gsecond(copy(x), ax, md, cv) };
    assert(h15.kind == scientific::F_VALUE);
    let r16, h16 = scientific::generic_laplace(lie) { x, ax, md, cv => common_3::gsecond(copy(x), ax, md, cv) };
    assert(h16.kind == scientific::F_SHAPE);
}
