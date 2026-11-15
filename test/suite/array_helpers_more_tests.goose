// The tests of the group array_helpers_more. Run them with SCIENTIFIC_SUITES=array_helpers_more.
//
// array_equiv, asarray_chkfinite, asmatrix, divmod, dsplit,
// empty_like, fromfunction, fromiter, iterable and real_if_close match
// the pins of NumPy 2.5.4.
namespace array_helpers_more_tests;

import scientific from "../..";
import common;
import array_helpers_more_pins;

fn test_array_equiv() {
    let lens = array_helpers_more_pins::EQ_A_L;
    var i: i64 = 0;
    while i < lens.len {
        let a = common::arr_of(common::shape_of(array_helpers_more_pins::EQ_A_SHAPE_D, array_helpers_more_pins::EQ_A_SHAPE_L, i), common::part_of(array_helpers_more_pins::EQ_A_D, lens, i));
        let b = common::arr_of(common::shape_of(array_helpers_more_pins::EQ_B_SHAPE_D, array_helpers_more_pins::EQ_B_SHAPE_L, i), common::part_of(array_helpers_more_pins::EQ_B_D, array_helpers_more_pins::EQ_B_L, i));
        let got, f = scientific::array_equiv(a, b);
        assert(f.kind == scientific::F_NONE);
        assert(got == (array_helpers_more_pins::EQ_OUT[i] == 1.0));
        i += 1;
    }
}

fn test_divmod() {
    let n = array_helpers_more_pins::DM_A.len;
    let m = array_helpers_more_pins::DM_B.len;
    let a = common::arr_of([n, 1], array_helpers_more_pins::DM_A);
    let b = common::arr_of([1, m], array_helpers_more_pins::DM_B);
    let q, r, f = scientific::divmod(a, b);
    assert(f.kind == scientific::F_NONE);
    assert(common::same_shape(q, [n, m]) && common::same_shape(r, [n, m]));
    assert(common::all_close(q, array_helpers_more_pins::DM_Q, 0.0));
    assert(common::all_close(r, array_helpers_more_pins::DM_R, 1.0e-12));
}

fn test_real_if_close() {
    let lens = array_helpers_more_pins::RIC_RE_L;
    var i: i64 = 0;
    while i < lens.len {
        let re = common::part_of(array_helpers_more_pins::RIC_RE_D, lens, i);
        let im = common::part_of(array_helpers_more_pins::RIC_IM_D, array_helpers_more_pins::RIC_IM_L, i);
        let c, fc = scientific::complex_from_values(re, im);
        assert(fc.kind == scientific::F_NONE);
        let r, ok, f = scientific::real_if_close(c, array_helpers_more_pins::RIC_TOL[i]);
        assert(f.kind == scientific::F_NONE);
        assert(ok == (array_helpers_more_pins::RIC_OUT[i] == 1.0));
        if ok { assert(common::all_close(r, re, 0.0)); }
        i += 1;
    }
}

// ds_check compares the pieces of one cut with the pins and gives the
// place of the next piece in the pins.
fn ds_check(p: scientific::Parts&, f: scientific::Fault, i: i64, g0: i64) -> i64 {
    assert(f.kind == scientific::F_NONE);
    assert(p.count == array_helpers_more_pins::DS_PIECES[i] as i64);
    var g = g0;
    var k: i64 = 0;
    while k < p.count {
        let x, fx = scientific::part(p, k);
        assert(fx.kind == scientific::F_NONE);
        assert(x.shape[2] == array_helpers_more_pins::DS_WIDTH[g] as i64);
        assert(common::all_close(x, common::part_of(array_helpers_more_pins::DS_OUT_D, array_helpers_more_pins::DS_OUT_L, g), 0.0));
        g += 1;
        k += 1;
    }
    return g;
}

fn test_dsplit() {
    var g: i64 = 0;
    var i: i64 = 0;
    while i < array_helpers_more_pins::DS_MODE.len {
        let shape = common::shape_of(array_helpers_more_pins::DS_SHAPE_D, array_helpers_more_pins::DS_SHAPE_L, i);
        var total: i64 = 1;
        var k: i64 = 0;
        while k < shape.len { total *= shape[k]; k += 1; }
        var v: f64[>..] = [];
        k = 0;
        while k < total { v.push(k); k += 1; }
        let a = common::arr_of(shape, v);
        let arg = common::shape_of(array_helpers_more_pins::DS_ARG_D, array_helpers_more_pins::DS_ARG_L, i);
        if array_helpers_more_pins::DS_MODE[i] == 1.0 {
            let p, f = scientific::dsplit(a, arg);
            g = ds_check(p, f, i, g);
        } else {
            let p, f = scientific::dsplit(a, arg[0]);
            g = ds_check(p, f, i, g);
        }
        i += 1;
    }
}

fn test_fromfunction_fromiter() {
    let a, fa = scientific::fromfunction([3, 4]) { p => p[0] * 10.0 + p[1] };
    assert(fa.kind == scientific::F_NONE && common::same_shape(a, [3, 4]));
    assert(common::all_close(a, array_helpers_more_pins::FF_A, 0.0));
    let b, fb = scientific::fromfunction([5]) { p => p[0] * p[0] + 0.5 };
    assert(fb.kind == scientific::F_NONE);
    assert(common::all_close(b, array_helpers_more_pins::FF_B, 0.0));
    let c, fc = scientific::fromfunction([2, 3, 2]) { p => p[0] * 100.0 + p[1] * 10.0 + p[2] };
    assert(fc.kind == scientific::F_NONE && common::same_shape(c, [2, 3, 2]));
    assert(common::all_close(c, array_helpers_more_pins::FF_C, 0.0));
    let d, fd = scientific::fromiter(7) { i => 0.5 * (i as f64) * (i as f64) - (i as f64) };
    assert(fd.kind == scientific::F_NONE && common::same_shape(d, [7]));
    assert(common::all_close(d, array_helpers_more_pins::FI_A, 0.0));
}

fn test_array_helpers_more_forms() {
    let a = common::arr_of([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let e, fe = scientific::empty_like(a);
    assert(fe.kind == scientific::F_NONE && common::same_shape(e, [2, 3]));
    // asmatrix gives a row for a vector and keeps two dimensions.
    let v = common::arr_of([3], [1.0, 2.0, 3.0]);
    let m, fm = scientific::asmatrix(v);
    assert(fm.kind == scientific::F_NONE && common::same_shape(m, [1, 3]));
    let m2, fm2 = scientific::asmatrix(a);
    assert(fm2.kind == scientific::F_NONE && common::same_shape(m2, [2, 3]));
    let t = common::arr_of([2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let m3, fm3 = scientific::asmatrix(t);
    assert(fm3.kind == scientific::F_SHAPE);
    // asarray_chkfinite copies a finite array and rejects NaN and infinity.
    let c, fc = scientific::asarray_chkfinite(a);
    assert(fc.kind == scientific::F_NONE && common::same_vals(c, [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let zero: f64 = 0.0;
    let bad = common::arr_of([2], [1.0, zero / zero]);
    let c2, fc2 = scientific::asarray_chkfinite(bad);
    assert(fc2.kind == scientific::F_VALUE);
    let inf = common::arr_of([2], [1.0, 1.0 / zero]);
    let c3, fc3 = scientific::asarray_chkfinite(inf);
    assert(fc3.kind == scientific::F_VALUE);
    // iterable is true for an array and false for a number.
    assert(scientific::iterable(a));
    assert(!scientific::iterable(1.5));
    // dsplit of a matrix is a fault.
    let sp, fp = scientific::dsplit(a, 1);
    assert(fp.kind == scientific::F_SHAPE);
    // The builds reject an empty shape and a count below one.
    let z, fz = scientific::fromfunction([0]) { q => 0.0 };
    assert(fz.kind == scientific::F_VALUE);
    let y, fy = scientific::fromiter(0) { i => 0.0 };
    assert(fy.kind == scientific::F_VALUE);
}
