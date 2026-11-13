// The tests of the group geometric_hostile. Run them with SCIENTIFIC_SUITES=geometric_hostile.
namespace geometric_hostile_tests;

import scientific from "../..";
import os;
import common_3;

// test_geometric_hostile checks the faults and the properties of the two
// calls that take a block.
fn test_geometric_hostile() {
    let a = common_3::ni_nd([3, 4], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    // An output shape above eight axes is a shape fault.
    let r1, h1 = scientific::geometric_transform(a, [1, 1, 1, 1, 1, 1, 1, 1, 1], 3, "constant", 0.0, true) { c, o => { o[0] = c[0]; } };
    assert(h1.kind == scientific::F_SHAPE);
    // An unknown mode is a value fault.
    let r2, h2 = scientific::geometric_transform(a, [], 3, "bogus", 0.0, true) { c, o => { o[0] = c[0]; } };
    assert(h2.kind == scientific::F_VALUE);
    // A bad order is a range fault.
    let r3, h3 = scientific::geometric_transform(a, [], -1, "constant", 0.0, true) { c, o => { o[0] = c[0]; } };
    assert(h3.kind == scientific::F_RANGE);
    // An empty output shape holds the shape of the input, and the mapping
    // identity reads the array back at order 1.
    let r4, h4 = scientific::geometric_transform(a, [], 1, "nearest", 0.0, true) { c, o => { var d = 0; while d < o.len { o[d] = c[d]; d += 1; } } };
    assert(h4.kind == scientific::F_NONE && r4.shape == [3, 4] && common_3::tr_same(r4, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]));
    // The block that writes outside the shape faults in the runtime. The
    // block here writes a coordinate below zero, and the border rule of the
    // mode constant turns it into the fill value.
    let r5, h5 = scientific::geometric_transform(a, [], 1, "constant", 7.0, true) { c, o => { o[0] = -5.0; o[1] = c[1]; } };
    assert(h5.kind == scientific::F_NONE && common_3::tr_same(r5, [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]));
    // generic_filter1d: a size below one, a bad axis, an origin off the
    // window, an unknown mode and a bad origin.
    let s1, e1 = scientific::generic_filter1d(a, 0, -1, "reflect", 0.0, 0) { line, out => { out[0] = line[0]; } };
    assert(e1.kind == scientific::F_VALUE);
    let s2, e2 = scientific::generic_filter1d(a, 3, 2, "reflect", 0.0, 0) { line, out => { out[0] = line[0]; } };
    assert(e2.kind == scientific::F_RANGE);
    let s3, e3 = scientific::generic_filter1d(a, 3, -1, "bogus", 0.0, 0) { line, out => { out[0] = line[0]; } };
    assert(e3.kind == scientific::F_VALUE);
    let s4, e4 = scientific::generic_filter1d(a, 3, -1, "reflect", 0.0, 2) { line, out => { out[0] = line[0]; } };
    assert(e4.kind == scientific::F_RANGE);
    let s5, e5 = scientific::generic_filter1d(a, 3, -1, "reflect", 0.0, -2) { line, out => { out[0] = line[0]; } };
    assert(e5.kind == scientific::F_RANGE);
    // A line sum with size 3 along the last axis of a row of four.
    let line_in = common_3::ni_nd([4], [1.0, 2.0, 3.0, 4.0]);
    let s6, e6 = scientific::generic_filter1d(line_in, 3, -1, "constant", 10.0, 0) { line, out => { var j = 0; while j < out.len { out[j] = line[j] + line[j + 1] + line[j + 2]; j += 1; } } };
    assert(e6.kind == scientific::F_NONE && common_3::tr_same(s6, [13.0, 6.0, 9.0, 17.0]));
    // The same sum along the axis 0 of a column.
    let col_in = common_3::ni_nd([4, 1], [1.0, 2.0, 3.0, 4.0]);
    let s7, e7 = scientific::generic_filter1d(col_in, 3, 0, "reflect", 0.0, 0) { line, out => { var j = 0; while j < out.len { out[j] = line[j] + line[j + 1] + line[j + 2]; j += 1; } } };
    assert(e7.kind == scientific::F_NONE && common_3::tr_same(s7, [4.0, 6.0, 9.0, 11.0]));
    // An origin of 1 moves the window one value forward.
    let s8, e8 = scientific::generic_filter1d(line_in, 3, -1, "reflect", 0.0, 1) { line, out => { var j = 0; while j < out.len { out[j] = line[j] * 1.0 + line[j + 1] * 2.0 + line[j + 2] * 3.0; j += 1; } } };
    assert(e8.kind == scientific::F_NONE && common_3::tr_same(s8, [7.0, 9.0, 14.0, 20.0]));
}
