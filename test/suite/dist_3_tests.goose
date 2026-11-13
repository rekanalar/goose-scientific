// The tests of the group dist. Run them with SCIENTIFIC_SUITES=dist.
namespace dist_3_tests;

import scientific from "../..";
import os;
import common_3;

fn test_dist_pins26() {
    let v428, c428, w428, g428 = scientific::value_indices(common_3::ni_nd([1, 4, 5], [0.0, -1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 2.0, -2.0, -1.0, 0.0, -2.0, 2.0, 3.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0]));
    assert(g428.kind == scientific::F_NONE);
    assert(v428.shape == [6] && common_3::ms_same(v428, [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]));
    assert(c428.shape == [6] && common_3::ms_same(c428, [2.0, 5.0, 5.0, 1.0, 6.0, 1.0]));
    assert(w428.shape == [20, 3] && common_3::ms_same(w428, [0.0, 1.0, 3.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 4.0, 0.0, 2.0, 4.0, 0.0, 3.0, 1.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0, 3.0, 2.0, 0.0, 2.0, 3.0]));
    let v429, c429, w429, g429 = scientific::value_indices(common_3::ni_nd([1, 4, 5], [0.0, -1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 2.0, -2.0, -1.0, 0.0, -2.0, 2.0, 3.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0]), -1.0, true);
    assert(g429.kind == scientific::F_NONE);
    assert(v429.shape == [5] && common_3::ms_same(v429, [-2.0, 0.0, 1.0, 2.0, 3.0]));
    assert(c429.shape == [5] && common_3::ms_same(c429, [2.0, 5.0, 1.0, 6.0, 1.0]));
    assert(w429.shape == [15, 3] && common_3::ms_same(w429, [0.0, 1.0, 3.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0, 3.0, 2.0, 0.0, 2.0, 3.0]));
    let v430, c430, w430, g430 = scientific::value_indices(common_3::ni_nd([5], [0.0, 1.0, -2.0, 2.0, 0.0]));
    assert(g430.kind == scientific::F_NONE);
    assert(v430.shape == [4] && common_3::ms_same(v430, [-2.0, 0.0, 1.0, 2.0]));
    assert(c430.shape == [4] && common_3::ms_same(c430, [1.0, 2.0, 1.0, 1.0]));
    assert(w430.shape == [5, 1] && common_3::ms_same(w430, [2.0, 0.0, 4.0, 1.0, 3.0]));
    let v431, c431, w431, g431 = scientific::value_indices(common_3::ni_nd([5], [0.0, 1.0, -2.0, 2.0, 0.0]), 3.0, true);
    assert(g431.kind == scientific::F_NONE);
    assert(v431.shape == [4] && common_3::ms_same(v431, [-2.0, 0.0, 1.0, 2.0]));
    assert(c431.shape == [4] && common_3::ms_same(c431, [1.0, 2.0, 1.0, 1.0]));
    assert(w431.shape == [5, 1] && common_3::ms_same(w431, [2.0, 0.0, 4.0, 1.0, 3.0]));
    let v432, c432, w432, g432 = scientific::value_indices(common_3::ni_nd([3, 1], [3.0, 2.0, 1.0]));
    assert(g432.kind == scientific::F_NONE);
    assert(v432.shape == [3] && common_3::ms_same(v432, [1.0, 2.0, 3.0]));
    assert(c432.shape == [3] && common_3::ms_same(c432, [1.0, 1.0, 1.0]));
    assert(w432.shape == [3, 2] && common_3::ms_same(w432, [2.0, 0.0, 1.0, 0.0, 0.0, 0.0]));
    let v433, c433, w433, g433 = scientific::value_indices(common_3::ni_nd([3, 1], [3.0, 2.0, 1.0]), 2.0, true);
    assert(g433.kind == scientific::F_NONE);
    assert(v433.shape == [2] && common_3::ms_same(v433, [1.0, 3.0]));
    assert(c433.shape == [2] && common_3::ms_same(c433, [1.0, 1.0]));
    assert(w433.shape == [2, 2] && common_3::ms_same(w433, [2.0, 0.0, 0.0, 0.0]));
    let v434, c434, w434, g434 = scientific::value_indices(common_3::ni_nd([5, 5, 4], [-1.0, -1.0, 1.0, 1.0, -1.0, -2.0, 1.0, 0.0, 0.0, 1.0, 3.0, -1.0, 2.0, -2.0, 0.0, 2.0, 1.0, 1.0, -2.0, -2.0, 1.0, 2.0, -1.0, -2.0, 0.0, -2.0, -2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 3.0, 0.0, 3.0, -1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.0, 2.0, 2.0, -2.0, 1.0, 2.0, 2.0, 3.0, -2.0, -2.0, 3.0, -2.0, -2.0, 1.0, -2.0, 2.0, -1.0, -2.0, 1.0, 0.0, -2.0, -1.0, 3.0, 1.0, -2.0, -1.0, -2.0, 3.0, 2.0, 3.0, 3.0, 0.0, 0.0, 3.0, -2.0, 3.0, 1.0, 0.0, -2.0, 3.0, 1.0, -1.0, 3.0, 1.0, 3.0, 2.0, 2.0, 2.0, 1.0, -2.0, 2.0, 2.0, -1.0, 2.0, -2.0, -2.0, 2.0, 2.0, -1.0]));
    assert(g434.kind == scientific::F_NONE);
    assert(v434.shape == [6] && common_3::ms_same(v434, [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]));
    assert(c434.shape == [6] && common_3::ms_same(c434, [23.0, 12.0, 12.0, 19.0, 19.0, 15.0]));
    assert(w434.shape == [100, 3] && common_3::ms_same(w434, [0.0, 1.0, 1.0, 0.0, 3.0, 1.0, 0.0, 4.0, 2.0, 0.0, 4.0, 3.0, 1.0, 0.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 2.0, 3.0, 0.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 3.0, 4.0, 2.0, 2.0, 4.0, 3.0, 3.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 1.0, 0.0, 2.0, 1.0, 3.0, 3.0, 2.0, 4.0, 1.0, 3.0, 0.0, 2.0, 3.0, 1.0, 2.0, 4.0, 0.0, 2.0, 4.0, 3.0, 1.0, 4.0, 4.0, 3.0, 0.0, 1.0, 3.0, 0.0, 2.0, 0.0, 0.0, 3.0, 2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 3.0, 3.0, 0.0, 3.0, 3.0, 1.0, 3.0, 4.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 4.0, 0.0, 0.0, 4.0, 1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 2.0, 3.0, 1.0, 4.0, 3.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0, 2.0, 3.0, 2.0, 2.0, 4.0, 3.0, 3.0, 1.0, 0.0, 3.0, 4.0, 1.0, 4.0, 0.0, 1.0, 4.0, 1.0, 0.0, 4.0, 2.0, 1.0, 0.0, 3.0, 0.0, 0.0, 3.0, 3.0, 1.0, 0.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 3.0, 2.0, 1.0, 2.0, 2.0, 1.0, 3.0, 2.0, 4.0, 0.0, 3.0, 2.0, 1.0, 4.0, 1.0, 2.0, 4.0, 1.0, 3.0, 4.0, 2.0, 0.0, 4.0, 2.0, 3.0, 4.0, 3.0, 0.0, 4.0, 3.0, 2.0, 4.0, 4.0, 1.0, 4.0, 4.0, 2.0, 0.0, 2.0, 2.0, 1.0, 3.0, 0.0, 1.0, 3.0, 2.0, 1.0, 4.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 3.0, 3.0, 0.0, 3.0, 3.0, 2.0, 0.0, 3.0, 2.0, 2.0, 3.0, 2.0, 3.0, 3.0, 3.0, 2.0, 3.0, 4.0, 0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 3.0, 4.0, 1.0, 1.0]));
    let v435, c435, w435, g435 = scientific::value_indices(common_3::ni_nd([5, 5, 4], [-1.0, -1.0, 1.0, 1.0, -1.0, -2.0, 1.0, 0.0, 0.0, 1.0, 3.0, -1.0, 2.0, -2.0, 0.0, 2.0, 1.0, 1.0, -2.0, -2.0, 1.0, 2.0, -1.0, -2.0, 0.0, -2.0, -2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 3.0, 0.0, 3.0, -1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.0, 2.0, 2.0, -2.0, 1.0, 2.0, 2.0, 3.0, -2.0, -2.0, 3.0, -2.0, -2.0, 1.0, -2.0, 2.0, -1.0, -2.0, 1.0, 0.0, -2.0, -1.0, 3.0, 1.0, -2.0, -1.0, -2.0, 3.0, 2.0, 3.0, 3.0, 0.0, 0.0, 3.0, -2.0, 3.0, 1.0, 0.0, -2.0, 3.0, 1.0, -1.0, 3.0, 1.0, 3.0, 2.0, 2.0, 2.0, 1.0, -2.0, 2.0, 2.0, -1.0, 2.0, -2.0, -2.0, 2.0, 2.0, -1.0]), 2.0, true);
    assert(g435.kind == scientific::F_NONE);
    assert(v435.shape == [5] && common_3::ms_same(v435, [-2.0, -1.0, 0.0, 1.0, 3.0]));
    assert(c435.shape == [5] && common_3::ms_same(c435, [23.0, 12.0, 12.0, 19.0, 15.0]));
    assert(w435.shape == [81, 3] && common_3::ms_same(w435, [0.0, 1.0, 1.0, 0.0, 3.0, 1.0, 0.0, 4.0, 2.0, 0.0, 4.0, 3.0, 1.0, 0.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 0.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 2.0, 3.0, 0.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 3.0, 4.0, 2.0, 2.0, 4.0, 3.0, 3.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 1.0, 0.0, 2.0, 1.0, 3.0, 3.0, 2.0, 4.0, 1.0, 3.0, 0.0, 2.0, 3.0, 1.0, 2.0, 4.0, 0.0, 2.0, 4.0, 3.0, 1.0, 4.0, 4.0, 3.0, 0.0, 1.0, 3.0, 0.0, 2.0, 0.0, 0.0, 3.0, 2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 3.0, 3.0, 0.0, 3.0, 3.0, 1.0, 3.0, 4.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 4.0, 0.0, 0.0, 4.0, 1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 2.0, 3.0, 1.0, 4.0, 3.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0, 2.0, 3.0, 2.0, 2.0, 4.0, 3.0, 3.0, 1.0, 0.0, 3.0, 4.0, 1.0, 4.0, 0.0, 1.0, 4.0, 1.0, 0.0, 4.0, 2.0, 1.0, 0.0, 2.0, 2.0, 1.0, 3.0, 0.0, 1.0, 3.0, 2.0, 1.0, 4.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 3.0, 3.0, 0.0, 3.0, 3.0, 2.0, 0.0, 3.0, 2.0, 2.0, 3.0, 2.0, 3.0, 3.0, 3.0, 2.0, 3.0, 4.0, 0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 3.0, 4.0, 1.0, 1.0]));
    let v436, c436, w436, g436 = scientific::value_indices(common_3::ni_nd([2], [3.0, -1.0]));
    assert(g436.kind == scientific::F_NONE);
    assert(v436.shape == [2] && common_3::ms_same(v436, [-1.0, 3.0]));
    assert(c436.shape == [2] && common_3::ms_same(c436, [1.0, 1.0]));
    assert(w436.shape == [2, 1] && common_3::ms_same(w436, [1.0, 0.0]));
    let v437, c437, w437, g437 = scientific::value_indices(common_3::ni_nd([2], [3.0, -1.0]), 1.0, true);
    assert(g437.kind == scientific::F_NONE);
    assert(v437.shape == [2] && common_3::ms_same(v437, [-1.0, 3.0]));
    assert(c437.shape == [2] && common_3::ms_same(c437, [1.0, 1.0]));
    assert(w437.shape == [2, 1] && common_3::ms_same(w437, [1.0, 0.0]));
    let v438, c438, w438, g438 = scientific::value_indices(common_3::ni_nd([3, 1], [0.0, 0.0, -1.0]));
    assert(g438.kind == scientific::F_NONE);
    assert(v438.shape == [2] && common_3::ms_same(v438, [-1.0, 0.0]));
    assert(c438.shape == [2] && common_3::ms_same(c438, [1.0, 2.0]));
    assert(w438.shape == [3, 2] && common_3::ms_same(w438, [2.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
    let v439, c439, w439, g439 = scientific::value_indices(common_3::ni_nd([3, 1], [0.0, 0.0, -1.0]), 3.0, true);
    assert(g439.kind == scientific::F_NONE);
    assert(v439.shape == [2] && common_3::ms_same(v439, [-1.0, 0.0]));
    assert(c439.shape == [2] && common_3::ms_same(c439, [1.0, 2.0]));
    assert(w439.shape == [3, 2] && common_3::ms_same(w439, [2.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
}

// test_dist_hostile checks the faults and the edge cases of the distances.
fn test_dist_hostile() {
    let a = common_3::ni_nd([3, 4], [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    let none = scientific::empty_arr();
    let ones = common_3::ni_nd([2, 2], [1.0, 1.0, 1.0, 1.0]);
    let zeros = common_3::ni_nd([2, 2], [0.0, 0.0, 0.0, 0.0]);
    let big = 9223372036854775807;
    // No background value: every call faults, SciPy gives sentinel values.
    let r1, h1 = scientific::distance_transform_cdt(ones);
    assert(h1.kind == scientific::F_VALUE);
    let r2, h2 = scientific::distance_transform_bf(ones);
    assert(h2.kind == scientific::F_VALUE);
    // No foreground value: the distances are zero.
    let r3, h3 = scientific::distance_transform_cdt(zeros);
    assert(h3.kind == scientific::F_NONE && common_3::ms_same(r3, [0.0, 0.0, 0.0, 0.0]));
    let r4, h4 = scientific::distance_transform_bf(zeros);
    assert(h4.kind == scientific::F_NONE && common_3::ms_same(r4, [0.0, 0.0, 0.0, 0.0]));
    // An unknown metric.
    let r5, h5 = scientific::distance_transform_cdt(a, "bogus");
    assert(h5.kind == scientific::F_VALUE);
    let r6, h6 = scientific::distance_transform_bf(a, "bogus");
    assert(h6.kind == scientific::F_VALUE);
    // A structure of the wrong size or count of dimensions.
    let r7, h7 = scientific::distance_transform_cdt_st(a, common_3::ni_nd([3, 2], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    assert(h7.kind == scientific::F_SHAPE);
    let r8, h8 = scientific::distance_transform_cdt_st(a, common_3::ni_nd([3], [1.0, 1.0, 1.0]));
    assert(h8.kind == scientific::F_SHAPE);
    // A structure of zeros reaches no value: the foreground keeps -1.
    let r9, h9 = scientific::distance_transform_cdt_st(a, common_3::ni_nd([3, 3], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(h9.kind == scientific::F_NONE && r9.data[0] == -1.0 && r9.data[3] == 0.0);
    // An integer structure counts by value, not by bytes (SciPy reads bytes).
    let r10, h10 = scientific::distance_transform_cdt_st(common_3::ni_nd([1, 3], [0.0, 1.0, 1.0]), common_3::ni_nd([3, 3], [0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0]));
    assert(h10.kind == scientific::F_NONE && common_3::ms_same(r10, [0.0, 1.0, 2.0]));
    // A sampling of the wrong count, zero, NaN, or too large.
    let r11, h11 = scientific::distance_transform_bf(a, "euclidean", [1.0, 1.0, 1.0]);
    assert(h11.kind == scientific::F_SHAPE);
    let r12, h12 = scientific::distance_transform_bf(a, "euclidean", [1.0, 0.0]);
    assert(h12.kind == scientific::F_VALUE);
    let r13, h13 = scientific::distance_transform_bf(a, "euclidean", [1.0, 0.0 / 0.0]);
    assert(h13.kind == scientific::F_VALUE);
    let r14, h14 = scientific::distance_transform_bf(a, "euclidean", [1.0, 1.0e300]);
    assert(h14.kind == scientific::F_VALUE);
    // The sampling counts for the euclidean metric only.
    let r15, h15 = scientific::distance_transform_bf(a, "taxicab", [1.0, 0.0]);
    assert(h15.kind == scientific::F_NONE);
    // A buffer that lies about its shape.
    var lie = scientific::clone_of(a);
    lie.data.pop();
    let r16, h16 = scientific::distance_transform_cdt(lie);
    assert(h16.kind == scientific::F_SHAPE);
    let r17, h17 = scientific::distance_transform_bf(lie);
    assert(h17.kind == scientific::F_SHAPE);
    let r18, rc18, rw18, h18 = scientific::value_indices(lie);
    assert(h18.kind == scientific::F_SHAPE);
    // value_indices: a value that is not a whole number or is too large.
    let v1, c1, w1, h19 = scientific::value_indices(common_3::ni_nd([2], [1.0, 1.5]));
    assert(h19.kind == scientific::F_VALUE);
    let v2, c2, w2, h20 = scientific::value_indices(common_3::ni_nd([2], [1.0, 1.0e19]));
    assert(h20.kind == scientific::F_VALUE);
    let v3, c3, w3, h21 = scientific::value_indices(common_3::ni_nd([2], [1.0, 0.0 / 0.0]));
    assert(h21.kind == scientific::F_VALUE);
    // The ignored value leaves no entry, and an array of that value leaves none.
    let v4, c4, w4, h22 = scientific::value_indices(common_3::ni_nd([3], [2.0, 2.0, 2.0]), 2.0, true);
    assert(h22.kind == scientific::F_NONE && v4.data.len == 0 && w4.data.len == 0);
    // Fault kinds are not changed by the metric: a unit array of one place.
    let r19, h23 = scientific::distance_transform_cdt(common_3::ni_nd([1], [1.0]));
    assert(h23.kind == scientific::F_VALUE);
    // A checkerboard of 600 by 600 holds too many pairs for the brute force search.
    var cb: f64[>..] = [];
    var q: i64 = 0;
    while q < 600 * 600 { cb.push(((q / 600 + q % 600) % 2) as f64); q += 1; }
    let r20, h25 = scientific::distance_transform_bf(scientific::arr_of(copy(cb), [600, 600]));
    assert(h25.kind == scientific::F_LIMIT);
    // The indices call gives a position of a background value for every value.
    let d, ix, h24 = scientific::distance_transform_bf_indices(a);
    assert(h24.kind == scientific::F_NONE && ix.shape == [2, 3, 4]);
    var p: i64 = 0;
    while p < 12 {
        let y = ix.data[p] as i64;
        let x = ix.data[12 + p] as i64;
        assert(a.data[y * 4 + x] == 0.0);
        p += 1;
    }
}
