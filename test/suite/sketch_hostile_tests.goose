// The tests of the group sketch_hostile. Run them with SCIENTIFIC_SUITES=sketch_hostile.
namespace sketch_hostile_tests;

import scientific from "../..";
import os;
import common;

// test_sketch_hostile covers the faults of the sketch.
fn test_sketch_hostile() {
    let a = common::mx(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let lie = scientific::Arr { ndim: 3, shape: [3, 2], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
    let r1, f1 = scientific::clarkson_woodruff_transform(lie, 2, 1);
    assert(f1.kind == scientific::F_SHAPE);
    let r2, f2 = scientific::clarkson_woodruff_transform(common::av([1.0, 2.0, 3.0]), 2, 1);
    assert(f2.kind == scientific::F_SHAPE);
    let sh = scientific::Arr { ndim: 2, shape: [3, 2], data: [1.0, 2.0, 3.0] };
    let r3, f3 = scientific::clarkson_woodruff_transform(sh, 2, 1);
    assert(f3.kind == scientific::F_SHAPE);
    let r4, f4 = scientific::clarkson_woodruff_transform(a, 0, 1);
    assert(f4.kind == scientific::F_VALUE);
    let r5, f5 = scientific::clarkson_woodruff_transform(a, -3, 1);
    assert(f5.kind == scientific::F_VALUE);
    let r6, f6 = scientific::clarkson_woodruff_transform(a, 134217729, 1);
    assert(f6.kind == scientific::F_LIMIT);
    let r7, f7 = scientific::clarkson_woodruff_transform(a, 9223372036854775807, 1);
    assert(f7.kind == scientific::F_LIMIT);
    let r8, f8 = scientific::clarkson_woodruff_transform(common::mx(1, 2, [1.0, 0.0 / 0.0]), 2, 1);
    assert(f8.kind == scientific::F_VALUE);
    let r9, f9 = scientific::clarkson_woodruff_transform(common::mx(1, 2, [1.0, 1.0 / 0.0]), 2, 1);
    assert(f9.kind == scientific::F_VALUE);
    return;
}
