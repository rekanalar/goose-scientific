// The tests of the group pad_hostile. Run them with SCIENTIFIC_SUITES=pad_hostile.
namespace pad_hostile_tests;

import scientific from "../..";
import os;
import common;

// test_pad_hostile covers the faults of the pad.
fn test_pad_hostile() {
    let a = common::mx(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    var bad = scientific::Arr { ndim: 3, shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
    let p1, f1 = scientific::pad_axes(bad, [1, 1, 1], [1, 1, 1]);
    assert(f1.kind == scientific::F_SHAPE);
    let sh = scientific::Arr { ndim: 2, shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0] };
    let p2, f2 = scientific::pad_axes(sh, [1, 1], [1, 1]);
    assert(f2.kind == scientific::F_SHAPE);
    let p3, f3 = scientific::pad_axes(a, [1, 1], [1, 1], "mean", [1, 2, 3]);
    assert(f3.kind == scientific::F_SHAPE);
    let p4, f4 = scientific::pad_axes(a, [1, 1], [1, 1], "linear_ramp", [], [1.0, 2.0, 3.0]);
    assert(f4.kind == scientific::F_SHAPE);
    let p5, f5 = scientific::pad_axes(a, [1], [1, 1]);
    assert(f5.kind == scientific::F_SHAPE);
    let p6, f6 = scientific::pad_axes(a, [-1, 1], [1, 1]);
    assert(f6.kind == scientific::F_VALUE);
    let p7, f7 = scientific::pad_axes(a, [1, 1], [1, 1], "mean", [0]);
    assert(f7.kind == scientific::F_VALUE);
    let p8, f8 = scientific::pad_axes(a, [1, 1], [1, 1], "mean", [2, -1]);
    assert(f8.kind == scientific::F_VALUE);
    let p9, f9 = scientific::pad_axes(a, [1, 1], [1, 1], "no such mode");
    assert(f9.kind == scientific::F_VALUE);
    let p10, f10 = scientific::pad_axes(a, [268435456, 0], [0, 0]);
    assert(f10.kind == scientific::F_LIMIT);
    // A count near the largest integer must not overflow the size test.
    let big = 9223372036854775807;
    let p11, f11 = scientific::pad_axes(a, [big, 0], [0, 0]);
    assert(f11.kind == scientific::F_LIMIT);
    let p12, f12 = scientific::pad_axes(a, [0, 0], [big, big], "edge");
    assert(f12.kind == scientific::F_LIMIT);
    let p13, f13 = scientific::pad_modes(a, [big - 1, 0], [2, 0]);
    assert(f13.kind == scientific::F_LIMIT);
    // A length above the width clips and succeeds.
    let p14, f14 = scientific::pad_axes(a, [1, 1], [1, 1], "mean", [50]);
    assert(f14.kind == scientific::F_NONE);
    // An input with a NaN passes the NaN into the pad.
    let nn = common::mx(1, 2, [0.0 / 0.0, 2.0]);
    let p15, f15 = scientific::pad_axes(nn, [0, 1], [0, 0], "edge");
    assert(f15.kind == scientific::F_NONE && p15.data[0] != p15.data[0]);
    // The call of pad_modes with the length zero takes the whole line.
    let p16, f16 = scientific::pad_modes(a, [0, 1], [0, 1], "mean", 0.0, 0);
    assert(f16.kind == scientific::F_NONE);
    return;
}
