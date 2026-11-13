// The tests of the group ordqz. Run them with SCIENTIFIC_SUITES=ordqz.
namespace ordqz_tests;

import scientific from "../..";
import os;
import common;
import common_2;

fn test_ordqz() {
    // The first group of each rule on the pinned pair.
    common_2::qt_group("lhp", [-15.928848409502654, -0.6126906030083017], [0.19514833489844835, 1.543360374582167]);
    common_2::qt_group("rhp", [0.19514833489844835, 1.543360374582167], [-15.928848409502654, -0.6126906030083017]);
    common_2::qt_group("iuc", [-0.6126906030083017, 0.19514833489844835], [-15.928848409502654, 1.543360374582167]);
    common_2::qt_group("ouc", [-15.928848409502654, 1.543360374582167], [-0.6126906030083017, 0.19514833489844835]);
    // Random pairs, each with the four rules.
    var n: i64 = 2;
    while n <= 10 {
        var k: i64 = 0;
        while k < 6 {
            let pa, pb = common::qt_pair(n, n * 131 + k, k % 3 % 2);
            common::qt_ord(pa, pb, n, "lhp");
            common::qt_ord(pa, pb, n, "rhp");
            common::qt_ord(pa, pb, n, "iuc");
            common::qt_ord(pa, pb, n, "ouc");
            k += 1;
        }
        n += 1;
    }
    // An infinite eigenvalue is outside the unit circle.
    let ia, ib = common::qt_pair(5, 4242, 1);
    let aa, bb, al, be, q, z, f = scientific::ordqz(ia, ib, "ouc");
    assert(f.kind == scientific::F_NONE);
    var small: i64 = -1;
    var i: i64 = 0;
    while i < 5 { if common::mathshim_abs(be.data[i]) < 1.0e-12 { small = i; } i += 1; }
    assert(small >= 0);
    i = 0;
    while i <= small { assert(common::qt_chosen("ouc", al.data[2 * i], al.data[2 * i + 1], be.data[i])); i += 1; }
    // A rule that is no rule.
    let ba1, ba2, ba3, ba4, ba5, ba6, bf = scientific::ordqz(ia, ib, "xyz");
    assert(bf.kind == scientific::F_VALUE);
    let ea1, ea2, ea3, ea4, ea5, ea6, ef = scientific::ordqz(ia, ib, "");
    assert(ef.kind == scientific::F_VALUE);

    // A stack with a broadcast operand. The values alpha and beta hold
    // the shape of the batch followed by the side.
    var sd: f64[>..] = [];
    var s: i64 = 0;
    while s < 2 {
        let p1, p2 = common::qt_pair(3, 210 + s, 0);
        var d: i64 = 0;
        while d < 9 { sd.push(p1.data[d]); d += 1; }
        s += 1;
    }
    let sb1, sb2 = common::qt_pair(3, 299, 0);
    let stack_a = scientific::Arr { ndim: 3, shape: [2, 3, 3], data: copy(sd) };
    let ta, tb, tal, tbe, tq, tz, tf = scientific::ordqz(stack_a, sb2, "rhp");
    assert(tf.kind == scientific::F_NONE);
    assert(tal.ndim == 2 && tal.shape[0] == 2 && tal.shape[1] == 3 && tbe.ndim == 2 && tbe.shape[1] == 3);
    assert(tal.data.len == 12 && tbe.data.len == 6 && tq.shape[0] == 2);
    s = 0;
    while s < 2 {
        let u1, u2 = common::qt_pair(3, 210 + s, 0);
        let ua, ub, ual, ube, uq, uz, uf = scientific::ordqz(u1, sb2, "rhp");
        var d2: i64 = 0;
        while d2 < 9 { assert(common::mathshim_abs(tq.data[s * 9 + d2] - uq.data[d2]) < 1.0e-12); d2 += 1; }
        d2 = 0;
        while d2 < 3 { assert(common::mathshim_abs(tbe.data[s * 3 + d2] - ube.data[d2]) < 1.0e-12); d2 += 1; }
        s += 1;
    }
    // A pair of zeros: no rule chooses an eigenvalue (0, 0), and the call
    // does not fault.
    let zero = scientific::Arr { ndim: 2, shape: [3, 3], data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] };
    let z1, z2, z3, z4, z5, z6, zf = scientific::ordqz(zero, zero, "ouc");
    assert(zf.kind == scientific::F_NONE);
    assert(!common::qt_chosen("ouc", z3.data[0], z3.data[1], z4.data[0]));
}
