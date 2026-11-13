// The tests of the group qz_c. Run them with SCIENTIFIC_SUITES=qz_c.
namespace qz_c_tests;

import scientific from "../..";
import os;
import common_2;

fn test_qz_c_hostile() {
    // A count of dimensions that does not match the shape.
    let good = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let lie = scientific::CArr { ndim: 3, shape: [2, 2], data: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let f1a, f1b, f1c, f1d, f1 = scientific::qz_c(copy(lie), copy(good));
    assert(f1.kind == scientific::F_SHAPE);
    // A buffer that is too short.
    let short = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0, 0.0, 0.0, 0.0] };
    let f2a, f2b, f2c, f2d, f2 = scientific::qz_c(copy(short), copy(good));
    assert(f2.kind == scientific::F_SHAPE);
    // A buffer that holds the count of entries and not twice the count.
    let half = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0, 0.0, 0.0, 1.0] };
    let f3a, f3b, f3c, f3d, f3 = scientific::qz_c(copy(half), copy(good));
    assert(f3.kind == scientific::F_SHAPE);
    // A side above the limit.
    var big: f64[>..] = [];
    var i: i64 = 0;
    while i < 513 * 513 * 2 { big.push(0.0); i += 1; }
    let bg = scientific::CArr { ndim: 2, shape: [513, 513], data: copy(big) };
    let f4a, f4b, f4c, f4d, f4 = scientific::qz_c(copy(bg), copy(bg));
    assert(f4.kind == scientific::F_LIMIT);
    // A NaN in a real part and a NaN in an imaginary part.
    let nanr = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0, 0.0, 0.0 / 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let f5a, f5b, f5c, f5d, f5 = scientific::qz_c(copy(nanr), copy(good));
    assert(f5.kind == scientific::F_VALUE);
    let nani = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0, 0.0 / 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let f6a, f6b, f6c, f6d, f6 = scientific::qz_c(copy(nani), copy(good));
    assert(f6.kind == scientific::F_VALUE);
    // An infinite value.
    let inf = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0, 1.0 / 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let f7a, f7b, f7c, f7d, f7 = scientific::qz_c(copy(inf), copy(good));
    assert(f7.kind == scientific::F_VALUE);
    // Two different sides.
    let three = scientific::CArr { ndim: 2, shape: [3, 3], data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] };
    let f8a, f8b, f8c, f8d, f8 = scientific::qz_c(copy(good), copy(three));
    assert(f8.kind == scientific::F_SHAPE);
    // A batch that does not broadcast.
    var s1: f64[>..] = [];
    var s2: f64[>..] = [];
    i = 0;
    while i < 2 * 2 * 2 * 2 { s1.push(1.0); s2.push(1.0); i += 1; }
    let b1 = scientific::CArr { ndim: 3, shape: [2, 2, 2], data: copy(s1) };
    let b2 = scientific::CArr { ndim: 3, shape: [3, 2, 2], data: copy(s2) };
    let f9a, f9b, f9c, f9d, f9 = scientific::qz_c(copy(b1), copy(b2));
    assert(f9.kind == scientific::F_SHAPE);
    // A stack above the limit of elements.
    var l1: f64[>..] = [];
    i = 0;
    while i < 6000 * 9 * 2 { l1.push(1.0); i += 1; }
    let la = scientific::CArr { ndim: 4, shape: [6000, 1, 3, 3], data: copy(l1) };
    let lb = scientific::CArr { ndim: 4, shape: [1, 6000, 3, 3], data: copy(l1) };
    let f10a, f10b, f10c, f10d, f10 = scientific::qz_c(copy(la), copy(lb));
    assert(f10.kind == scientific::F_LIMIT);
    // A rule that is not a name of a rule.
    let f11a, f11b, f11c, f11d, f11e, f11f, f11 = scientific::ordqz_c(copy(good), copy(good), "abc");
    assert(f11.kind == scientific::F_VALUE);

    // A pair whose entries stand near the limit of the format. The
    // call gives a fault, or all results are finite.
    let hd: f64[>..] = [];
    i = 0;
    while i < 9 { hd.push(1.0e300 * (1.0 + (i as f64) / 7.0)); hd.push(0.0); i += 1; }
    let h1 = scientific::CArr { ndim: 2, shape: [3, 3], data: copy(hd) };
    let h2 = scientific::CArr { ndim: 2, shape: [3, 3], data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] };
    let ha, hb, hq, hz, hf = scientific::qz_c(copy(h1), copy(h2));
    if hf.kind == scientific::F_NONE {
        assert(common_2::qt_finite_all(ha.data) && common_2::qt_finite_all(hb.data) && common_2::qt_finite_all(hq.data) && common_2::qt_finite_all(hz.data));
    } else {
        assert(hf.kind == scientific::F_VALUE);
    }
    let md: f64[>..] = [];
    i = 0;
    while i < 9 { md.push(1.0e200 * (1.0 + (i as f64) / 7.0)); md.push(0.0); i += 1; }
    let m1 = scientific::CArr { ndim: 2, shape: [3, 3], data: copy(md) };
    let ma, mb, mq, mz, mf = scientific::qz_c(copy(m1), copy(h2));
    if mf.kind == scientific::F_NONE {
        assert(common_2::qt_finite_all(ma.data) && common_2::qt_finite_all(mb.data) && common_2::qt_finite_all(mq.data) && common_2::qt_finite_all(mz.data));
    } else {
        assert(mf.kind == scientific::F_VALUE);
    }
}
