// The tests of the group pad_stats. Run them with SCIENTIFIC_SUITES=pad_stats.
namespace pad_stats_tests;

import scientific from "../..";
import os;
import common;

// The statistic modes of pad_modes, pinned against NumPy.
fn test_pad_stats() {
    let q = common::av([3.0, 1.0, 4.0, 1.0, 5.0]);
    let m = common::mx(2, 3, [1.0, 2.0, 3.0, 4.0, 8.0, 6.0]);
    let r1, f1 = scientific::pad_modes(q, [3], [4], "linear_ramp", 7.0);
    assert(f1.kind == scientific::F_NONE && common::all_near(r1, [7.0, 5.666666666666667, 4.333333333333334, 3.0, 1.0, 4.0, 1.0, 5.0, 5.5, 6.0, 6.5, 7.0]));
    let r2, f2 = scientific::pad_modes(q, [3], [4], "maximum");
    assert(f2.kind == scientific::F_NONE && common::all_near(r2, [5.0, 5.0, 5.0, 3.0, 1.0, 4.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0]));
    let r3, f3 = scientific::pad_modes(q, [3], [4], "minimum", 0.0, 3);
    assert(f3.kind == scientific::F_NONE && common::all_near(r3, [1.0, 1.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0]));
    let r4, f4 = scientific::pad_modes(q, [3], [4], "mean", 0.0, 2);
    assert(f4.kind == scientific::F_NONE && common::all_near(r4, [2.0, 2.0, 2.0, 3.0, 1.0, 4.0, 1.0, 5.0, 3.0, 3.0, 3.0, 3.0]));
    let r5, f5 = scientific::pad_modes(q, [3], [4], "median");
    assert(f5.kind == scientific::F_NONE && common::all_near(r5, [3.0, 3.0, 3.0, 3.0, 1.0, 4.0, 1.0, 5.0, 3.0, 3.0, 3.0, 3.0]));
    let r6, f6 = scientific::pad_modes(m, [1, 2], [2, 1], "mean");
    assert(f6.kind == scientific::F_NONE && common::all_near(r6, [4.0, 4.0, 2.5, 5.0, 4.5, 4.0, 2.0, 2.0, 1.0, 2.0, 3.0, 2.0, 6.0, 6.0, 4.0, 8.0, 6.0, 6.0, 4.0, 4.0, 2.5, 5.0, 4.5, 4.0, 4.0, 4.0, 2.5, 5.0, 4.5, 4.0]));
    let r7, f7 = scientific::pad_modes(m, [2, 1], [1, 2], "linear_ramp", -1.0);
    assert(f7.kind == scientific::F_NONE && common::all_near(r7, [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.5, 1.0, 0.0, -1.0, -1.0, 1.0, 2.0, 3.0, 1.0, -1.0, -1.0, 4.0, 8.0, 6.0, 2.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]));
    let r8, f8 = scientific::pad_modes(m, [1, 1], [1, 1], "median", 0.0, 2);
    assert(f8.kind == scientific::F_NONE && common::all_near(r8, [3.75, 2.5, 5.0, 4.5, 4.75, 1.5, 1.0, 2.0, 3.0, 2.5, 6.0, 4.0, 8.0, 6.0, 7.0, 3.75, 2.5, 5.0, 4.5, 4.75]));
    let r9, f9 = scientific::pad_modes(q, [1], [1], "mean", 0.0, -1);
    assert(f9.kind == scientific::F_VALUE);
    return;
}
