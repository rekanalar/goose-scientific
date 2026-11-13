// The tests of the group integrate. Run them with SCIENTIFIC_SUITES=integrate.
namespace integrate_tests;

import scientific from "../..";
import os;
import common;

fn test_integrate() {
    // The integral of a straight line over zero to one.
    let y = [0.0, 0.25, 0.5, 0.75, 1.0];
    let t1, f1 = scientific::trapezoid(y, 0.25);
    assert(f1.kind == scientific::F_NONE && common::close(t1, 0.5));
    let s1, f2 = scientific::simpson(y, 0.25);
    assert(f2.kind == scientific::F_NONE && common::close(s1, 0.5));
    let r1, f3 = scientific::romberg(y, 0.25);
    assert(f3.kind == scientific::F_NONE && common::close(r1, 0.5));
    let c1, f4 = scientific::cumulative_trapezoid(y, 0.25);
    assert(f4.kind == scientific::F_NONE && c1.data.len == 4);
    assert(common::close(c1.data[0], 0.03125) && common::close(c1.data[1], 0.125));
    assert(common::close(c1.data[2], 0.28125) && common::close(c1.data[3], 0.5));
    // The integral of a square over zero to two.
    let ones = [1.0, 1.0, 1.0, 1.0, 1.0];
    let t2, f5 = scientific::trapezoid(ones, 0.5);
    assert(f5.kind == scientific::F_NONE && common::close(t2, 2.0));
    // An even count of parts takes the correction of Cartwright on
    // the last part. The straight line gives the same value both ways.
    let s3, f6 = scientific::simpson([0.0, 0.5, 1.0, 1.5], 0.5);
    assert(f6.kind == scientific::F_NONE && common::close(s3, 1.125));
    // The correction of Cartwright on a signal that is not straight.
    let s4, f6b = scientific::simpson([1.0, 2.0, 4.0, 8.0], 0.5);
    assert(f6b.kind == scientific::F_NONE && common::close(s4, 5.083333333333333));
    // The integral of the sine over zero to pi.
    var y9: f64[>..] = [];
    var i: i64 = 0;
    while i <= 8 {
        y9.push(mathshim::sine(3.141592653589793 * i / 8.0));
        i += 1;
    }
    let s9, f7 = scientific::simpson(y9, 3.141592653589793 / 8.0);
    assert(f7.kind == scientific::F_NONE && common::close(s9, 2.000269169948388));
    let r9, f8 = scientific::romberg(y9, 3.141592653589793 / 8.0);
    assert(f8.kind == scientific::F_NONE && common::close(r9, 2.000005549979671));
    // A spacing of zero is a fault.
    let bad1, fb1 = scientific::trapezoid(y, 0.0);
    assert(fb1.kind == scientific::F_VALUE);
    let bad2, fb2 = scientific::simpson(y, 0.0);
    assert(fb2.kind == scientific::F_VALUE);
    let bad3, fb3 = scientific::cumulative_trapezoid(y, 0.0);
    assert(fb3.kind == scientific::F_VALUE);
    let bad4, fb4 = scientific::romberg(y, 0.0);
    assert(fb4.kind == scientific::F_VALUE);
    // A signal of fewer than two values gives zero, as SciPy does.
    let bad5, fb5 = scientific::trapezoid([1.0], 1.0);
    assert(fb5.kind == scientific::F_NONE && common::close(bad5, 0.0));
    let bad6, fb6 = scientific::simpson([1.0], 1.0);
    assert(fb6.kind == scientific::F_NONE && common::close(bad6, 0.0));
    // A signal of two values takes the rule of the trapezoid.
    let bad7, fb7 = scientific::simpson([1.0, 2.0], 1.0);
    assert(fb7.kind == scientific::F_NONE && common::close(bad7, 1.5));
    let bad8, fb8 = scientific::romberg([1.0, 2.0], 1.0);
    assert(fb8.kind == scientific::F_VALUE);
    // A count of the parts that is not a power of two is a fault.
    let bad9, fb9 = scientific::romberg([0.0, 1.0, 2.0, 3.0], 1.0);
    assert(fb9.kind == scientific::F_VALUE);
    // The integral of an array along one axis. The values of SciPy
    // 1.18 pin every number. The axis drops from the shape.
    let m, mf = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let a23, a23f = scientific::reshape(m, [2, 3]);
    assert(mf.kind == scientific::F_NONE && a23f.kind == scientific::F_NONE);
    let ta1, taf1 = scientific::trapezoid_axis(a23, 1);
    assert(taf1.kind == scientific::F_NONE && ta1.ndim == 1 && ta1.shape[0] == 2);
    assert(common::all_close(ta1, [4.0, 10.0], 1.0e-12));
    let ta0, taf0 = scientific::trapezoid_axis(a23, 0);
    assert(taf0.kind == scientific::F_NONE && common::all_close(ta0, [2.5, 3.5, 4.5], 1.0e-12));
    let sa1, saf1 = scientific::simpson_axis(a23, 1, 0.5);
    assert(saf1.kind == scientific::F_NONE && common::all_close(sa1, [2.0, 5.0], 1.0e-12));
    // The axis form of a signal of three dimensions.
    var t3: f64[>..] = [];
    var ti: i64 = 1;
    while ti <= 24 { t3.push(ti); ti += 1; }
    let m3, m3f = scientific::from_values(t3);
    let a234, a234f = scientific::reshape(m3, [2, 3, 4]);
    assert(m3f.kind == scientific::F_NONE && a234f.kind == scientific::F_NONE);
    let ta3, taf3 = scientific::trapezoid_axis(a234, 1, 2.0);
    assert(taf3.kind == scientific::F_NONE && ta3.ndim == 2 && ta3.shape[0] == 2 && ta3.shape[1] == 4);
    assert(common::all_close(ta3, [20.0, 24.0, 28.0, 32.0, 68.0, 72.0, 76.0, 80.0], 1.0e-12));
    let sa3, saf3 = scientific::simpson_axis(a234, 1);
    assert(saf3.kind == scientific::F_NONE && common::all_close(sa3, [10.0, 12.0, 14.0, 16.0, 34.0, 36.0, 38.0, 40.0], 1.0e-12));
    // The running total along one axis. The axis holds one value less.
    let ca1, caf1 = scientific::cumulative_trapezoid_axis(a23, 1, 0.5);
    assert(caf1.kind == scientific::F_NONE && ca1.ndim == 2 && ca1.shape[0] == 2 && ca1.shape[1] == 2);
    assert(common::all_close(ca1, [0.75, 2.0, 2.25, 5.0], 1.0e-12));
    let ca0, caf0 = scientific::cumulative_trapezoid_axis(a23, 0);
    assert(caf0.kind == scientific::F_NONE && ca0.shape[0] == 1 && common::all_close(ca0, [2.5, 3.5, 4.5], 1.0e-12));
    var c3: f64[>..] = [];
    var ci: i64 = 1;
    while ci <= 8 { c3.push(ci); ci += 1; }
    let mc, mcf = scientific::from_values(c3);
    let a222, a222f = scientific::reshape(mc, [2, 2, 2]);
    assert(mcf.kind == scientific::F_NONE && a222f.kind == scientific::F_NONE);
    let ca2, caf2 = scientific::cumulative_trapezoid_axis(a222, 2);
    assert(caf2.kind == scientific::F_NONE && ca2.shape[0] == 2 && ca2.shape[1] == 2 && ca2.shape[2] == 1);
    assert(common::all_close(ca2, [1.5, 3.5, 5.5, 7.5], 1.0e-12));
    // An axis outside the shape is a fault.
    let e1, ef1 = scientific::trapezoid_axis(a23, 2);
    assert(ef1.kind == scientific::F_RANGE);
    let e2, ef2 = scientific::simpson_axis(a23, -3);
    assert(ef2.kind == scientific::F_RANGE);
    let e3, ef3 = scientific::cumulative_trapezoid_axis(a23, 5);
    assert(ef3.kind == scientific::F_RANGE);
    // A spacing of zero is a fault.
    let e4, ef4 = scientific::trapezoid_axis(a23, 0, 0.0);
    assert(ef4.kind == scientific::F_VALUE);
    let e5, ef5 = scientific::cumulative_trapezoid_axis(a23, 0, 0.0);
    assert(ef5.kind == scientific::F_VALUE);
}
