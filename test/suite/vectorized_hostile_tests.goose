// The tests of the group vectorized_hostile. Run them with SCIENTIFIC_SUITES=vectorized_hostile.
namespace vectorized_hostile_tests;

import scientific from "../..";
import os;
import common_3;

// test_vectorized_hostile checks the faults of the vectorized filter.
fn test_vectorized_hostile() {
    let a = common_3::ni_nd([3, 4], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    // A window with a count that is not one for each axis.
    let r1, h1 = scientific::vectorized_filter(a, [3], "reflect", 0.0, []) { x, o => { o[0] = x.data[0]; } };
    assert(h1.kind == scientific::F_SHAPE);
    // A window below one.
    let r2, h2 = scientific::vectorized_filter(a, [3, 0], "reflect", 0.0, []) { x, o => { o[0] = x.data[0]; } };
    assert(h2.kind == scientific::F_VALUE);
    // An origin off the window.
    let r3, h3 = scientific::vectorized_filter(a, [3, 3], "reflect", 0.0, [2, 0]) { x, o => { o[0] = x.data[0]; } };
    assert(h3.kind == scientific::F_RANGE);
    // An origin with a count that is not one for each axis. An origin of
    // one value stands for every axis.
    let r4, h4 = scientific::vectorized_filter(a, [3, 3], "reflect", 0.0, [1, 0, 0]) { x, o => { o[0] = x.data[0]; } };
    assert(h4.kind == scientific::F_SHAPE);
    let r4b, h4b = scientific::vectorized_filter(a, [3, 3], "reflect", 0.0, [1]) { x, o => { o[0] = x.data[0]; } };
    assert(h4b.kind == scientific::F_NONE);
    // An unknown mode.
    let r5, h5 = scientific::vectorized_filter(a, [3, 3], "bogus", 0.0, []) { x, o => { o[0] = x.data[0]; } };
    assert(h5.kind == scientific::F_VALUE);
    // A window of side 1 gives the array back.
    let r6, h6 = scientific::vectorized_filter(a, [1, 1], "reflect", 0.0, []) { x, o => {
        var p = 0;
        while p < x.data.len { o[p] = x.data[p]; p += 1; }
    } };
    assert(h6.kind == scientific::F_NONE && common_3::tr_same(r6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]));
    // A window of side 3 with the mode constant and a fill value of 7 gives
    // the sums of the windows with the value 7 outside.
    let r7, h7 = scientific::vectorized_filter(a, [3, 3], "constant", 7.0, []) { x, o => {
        var h = x.ndim / 2;
        var d = 0;
        var n = 1;
        while d < h { n *= x.shape[d]; d += 1; }
        var w = 1;
        d = h;
        while d < x.ndim { w *= x.shape[d]; d += 1; }
        var p = 0;
        while p < n {
            var s = 0.0;
            var j = 0;
            while j < w { s += x.data[p * w + j]; j += 1; }
            o[p] = s;
            p += 1;
        }
    } };
    assert(h7.kind == scientific::F_NONE && common_3::tr_same(r7, [49.0, 45.0, 51.0, 57.0, 54.0, 54.0, 63.0, 66.0, 65.0, 69.0, 75.0, 73.0]));
}
