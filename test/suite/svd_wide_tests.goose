// The tests of the group svd_wide. Run them with SCIENTIFIC_SUITES=svd_wide.
namespace svd_wide_tests;

import scientific from "../..";
import os;
import common;

// A wide matrix of rank 3 holds columns that shrink to the size of an
// underflow. The singular values and the vectors stay finite.
fn test_svd_wide_rank() {
    let r = common::ct_orth(12, 17 + 31 * 21);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 3 { var j: i64 = 0; while j < 9 { d.push(r.data[i * 12 + j]); j += 1; } i += 1; }
    let w = scientific::Arr { ndim: 2, shape: [3, 9], data: copy(d) };
    let u, s, vt, f = scientific::svd(w);
    assert(f.kind == scientific::F_NONE);
    i = 0;
    while i < s.data.len { assert(s.data[i] - s.data[i] == 0.0); i += 1; }
    i = 0;
    while i < vt.data.len { assert(vt.data[i] - vt.data[i] == 0.0); i += 1; }
    i = 0;
    while i < u.data.len { assert(u.data[i] - u.data[i] == 0.0); i += 1; }
    // The right vectors hold nine columns. Six of them come from the
    // completion. Every column holds a length of one. The columns are
    // orthogonal to each other.
    var j: i64 = 0;
    while j < 9 {
        var norm: f64 = 0.0;
        var k: i64 = 0;
        while k < 9 { norm += vt.data[j * 9 + k] * vt.data[j * 9 + k]; k += 1; }
        assert(common::mathshim_abs(norm - 1.0) <= 1.0e-12);
        j += 1;
    }
    j = 0;
    while j < 9 {
        var k: i64 = 0;
        while k < 9 {
            var dot: f64 = 0.0;
            var r: i64 = 0;
            while r < 9 { dot += vt.data[j * 9 + r] * vt.data[k * 9 + r]; r += 1; }
            if j != k { assert(common::mathshim_abs(dot) <= 1.0e-12); }
            k += 1;
        }
        j += 1;
    }
}
