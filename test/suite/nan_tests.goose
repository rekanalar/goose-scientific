// The tests of the group nan. Run them with SCIENTIFIC_SUITES=nan.
namespace nan_tests;

import scientific from "../..";
import os;
import common;

// The reductions that skip NaN hold the values of NumPy.
fn test_nan_reductions() {
    let nan = 0.0 / 0.0;
    let a, fa = scientific::from_values([1.0, nan, 3.0, nan, 8.0]);
    assert(fa.kind == scientific::F_NONE);
    let r1, f1 = scientific::nansum(a);
    let r2, f2 = scientific::nanprod(a);
    let r3, f3 = scientific::nanmin(a);
    let r4, f4 = scientific::nanmax(a);
    let r5, f5 = scientific::nanmean(a);
    assert(f1.kind == scientific::F_NONE && f2.kind == scientific::F_NONE);
    assert(f3.kind == scientific::F_NONE && f4.kind == scientific::F_NONE && f5.kind == scientific::F_NONE);
    assert(r1 == 12.0 && r2 == 24.0 && r3 == 1.0 && r4 == 8.0 && r5 == 4.0);
    let v0, fv0 = scientific::nanvar(a);
    let s0, fs0 = scientific::nanstd(a);
    let v1, fv1 = scientific::nanvar(a, 1);
    let s1, fs1 = scientific::nanstd(a, 1);
    let v3, fv3 = scientific::nanvar(a, 3);
    assert(fv0.kind == scientific::F_NONE && fs0.kind == scientific::F_NONE);
    assert(fv1.kind == scientific::F_NONE && fs1.kind == scientific::F_NONE && fv3.kind == scientific::F_NONE);
    assert(common::close(v0, 8.666666666666666) && common::close(s0, 2.943920288775949));
    assert(common::close(v1, 13.0) && common::close(s1, 3.605551275463989));
    // A divisor below one gives NaN, and a ddof below zero is a fault.
    assert(v3 != v3);
    let vb, fvb = scientific::nanvar(a, -1);
    assert(fvb.kind == scientific::F_VALUE);
    // An array of only NaN gives the neutral value or NaN.
    let z, fz = scientific::from_values([nan, nan]);
    let z1, fz1 = scientific::nansum(z);
    let z2, fz2 = scientific::nanprod(z);
    let z3, fz3 = scientific::nanmin(z);
    let z4, fz4 = scientific::nanmean(z);
    assert(fz.kind == scientific::F_NONE && fz1.kind == scientific::F_NONE);
    assert(z1 == 0.0 && z2 == 1.0 && z3 != z3 && z4 != z4);
    // The forms along an axis drop the axis.
    let m, fm = scientific::from_rows(2, 3, [1.0, nan, 3.0, 4.0, 5.0, nan]);
    assert(fm.kind == scientific::F_NONE);
    let c0, fc0 = scientific::nansum_axis(m, 0);
    let c1, fc1 = scientific::nansum_axis(m, 1);
    assert(fc0.kind == scientific::F_NONE && c0.ndim == 1 && c0.data.len == 3);
    assert(c0.data[0] == 5.0 && c0.data[1] == 5.0 && c0.data[2] == 3.0);
    assert(fc1.kind == scientific::F_NONE && c1.data[0] == 4.0 && c1.data[1] == 9.0);
    let p1, fp1 = scientific::nanprod_axis(m, 1);
    let n1, fn1 = scientific::nanmin_axis(m, 1);
    let x1, fx1 = scientific::nanmax_axis(m, 0);
    let e1, fe1 = scientific::nanmean_axis(m, 0);
    let d1, fd1 = scientific::nanvar_axis(m, 0);
    let t1, ft1 = scientific::nanstd_axis(m, 1);
    assert(fp1.kind == scientific::F_NONE && fn1.kind == scientific::F_NONE && fx1.kind == scientific::F_NONE);
    assert(fe1.kind == scientific::F_NONE && fd1.kind == scientific::F_NONE && ft1.kind == scientific::F_NONE);
    assert(p1.data[0] == 3.0 && p1.data[1] == 20.0 && n1.data[0] == 1.0 && n1.data[1] == 4.0);
    assert(x1.data[0] == 4.0 && x1.data[1] == 5.0 && x1.data[2] == 3.0);
    assert(e1.data[0] == 2.5 && e1.data[1] == 5.0 && e1.data[2] == 3.0);
    assert(d1.data[0] == 2.25 && d1.data[1] == 0.0 && t1.data[0] == 1.0 && t1.data[1] == 0.5);
    // A bad axis is a fault.
    let bad, fbad = scientific::nansum_axis(m, 2);
    assert(fbad.kind == scientific::F_RANGE && bad.data.len == 0);
    // nan_to_num replaces NaN and the infinities.
    let inf = 1.0 / 0.0;
    let q, fq = scientific::from_values([nan, inf, -inf, 2.0]);
    let r, fr = scientific::nan_to_num(q);
    assert(fr.kind == scientific::F_NONE && r.data[0] == 0.0);
    assert(r.data[1] == 1.7976931348623157e308 && r.data[2] == -1.7976931348623157e308 && r.data[3] == 2.0);
    let r6, fr6 = scientific::nan_to_num(q, -1.0, 9.0, -9.0);
    assert(fr6.kind == scientific::F_NONE && r6.data[0] == -1.0 && r6.data[1] == 9.0 && r6.data[2] == -9.0);
}
