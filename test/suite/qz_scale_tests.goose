// The tests of the group qz_scale. Run them with SCIENTIFIC_SUITES=qz_scale.
namespace qz_scale_tests;

import scientific from "../..";
import os;
import common;
import common_2;

// test_qz_scale decomposes pairs whose entries sit near the limits of the
// number format.
fn test_qz_scale() {
    let scales = [1.0e-300, 1.0e-150, 1.0e150, 1.0e300];
    var i: i64 = 0;
    while i < 4 {
        common_2::qt_scaled_case(scales[i], scales[i], 1.0, true);
        i += 1;
    }
    // Matrices of very different size: the eigenvalues move by the ratio.
    common_2::qt_scaled_case(1.0e-300, 1.0, 1.0e-300, true);
    common_2::qt_scaled_case(1.0, 1.0e-300, 1.0e300, true);
    // A pair whose scales sit 2^2000 apart. The ratio leaves the range of
    // the format, and no value turns NaN.
    common_2::qt_scaled_case(1.0715086071862673e301, 9.332636185032189e-302, 1.0, false);
    common_2::qt_scaled_case(9.332636185032189e-302, 1.0715086071862673e301, 1.0, false);

    // Two eigenvalues of size 1.0e-300 and of opposite sign. The rule lhp
    // must swap them.
    let ta = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0e-300, 0.0, 0.0, -1.0e-300] };
    let tb = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0, 0.0, 0.0, 1.0] };
    let t1, t2, t3, t4, t5, t6, tf = scientific::ordqz(ta, tb, "lhp");
    assert(tf.kind == scientific::F_NONE);
    assert(t3.data[0] < 0.0 && t3.data[2] > 0.0);
    assert(t5.data.len == 4);

    // A repeated eigenvalue in two blocks next to a chosen block. The call
    // gives a valid order or a fault. It never gives a wrong order.
    let ra = scientific::Arr { ndim: 2, shape: [3, 3], data: [-1.0, 0.3, 0.2, 0.0, 1.0, 0.4, 0.0, 0.0, -1.0] };
    let rb = scientific::Arr { ndim: 2, shape: [3, 3], data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
    let r1, r2, ral, rbe, r5, r6, rf = scientific::ordqz(ra, rb, "lhp");
    if rf.kind == scientific::F_NONE {
        assert(ral.data[0] < 0.0 && ral.data[2] < 0.0 && ral.data[4] > 0.0);
    } else {
        assert(rf.kind == scientific::F_VALUE);
    }

    // An eigenvalue of modulus exactly 1 lies in neither the inside nor the
    // outside of the unit circle. Each rule picks the same group as before.
    let ua = scientific::Arr { ndim: 2, shape: [3, 3], data: [1.0, 0.2, 0.1, 0.0, 0.5, 0.3, 0.0, 0.0, 2.0] };
    let ub = scientific::Arr { ndim: 2, shape: [3, 3], data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
    let u1, u2, ual, ube, u5, u6, uf = scientific::ordqz(ua, ub, "iuc");
    assert(uf.kind == scientific::F_NONE);
    assert(common::mathshim_abs(ual.data[0] / ube.data[0] - 0.5) < 1.0e-12);
    let v1, v2, val, vbe, v5, v6, vf = scientific::ordqz(ua, ub, "ouc");
    assert(vf.kind == scientific::F_NONE);
    assert(common::mathshim_abs(val.data[0] / vbe.data[0] - 2.0) < 1.0e-12);

    // A value that is not finite stays a fault at any scale.
    let na, nb = common_2::qt_pinned_scaled(1.0e300, 1.0e300);
    var nd = copy(na.data);
    nd[5] = 0.0 / 0.0;
    let nan_a = scientific::Arr { ndim: 2, shape: [4, 4], data: copy(nd) };
    let n1, n2, n3, n4, nf = scientific::qz(nan_a, nb);
    assert(nf.kind == scientific::F_VALUE);
    let m1, m2, m3, m4, m5, m6, mf = scientific::ordqz(nan_a, nb, "lhp");
    assert(mf.kind == scientific::F_VALUE);
}
