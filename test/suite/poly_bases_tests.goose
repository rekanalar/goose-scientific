// The tests of the group poly_bases. Run them with SCIENTIFIC_SUITES=poly_bases.
//
// The values, the conversions, the derivatives, the integrals and the roots
// of the five orthogonal bases match the pins of NumPy 2.5.4.
namespace poly_bases_tests;

import scientific from "../..";
import common;
import poly_bases_pins;

// Family codes: 0 Legendre, 1 Chebyshev, 2 Hermite, 3 HermiteE, 4 Laguerre.

fn pin_val(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_VAL); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_VAL); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_VAL); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_VAL); }
    return scientific::copy_slice(poly_bases_pins::LAG_VAL);
}

fn pin_b20(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_B20); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_B20); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_B20); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_B20); }
    return scientific::copy_slice(poly_bases_pins::LAG_B20);
}

fn pin_tom_d(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_TOM_D); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_TOM_D); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_TOM_D); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_TOM_D); }
    return scientific::copy_slice(poly_bases_pins::LAG_TOM_D);
}

fn pin_tom_l(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_TOM_L); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_TOM_L); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_TOM_L); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_TOM_L); }
    return scientific::copy_slice(poly_bases_pins::LAG_TOM_L);
}

fn pin_frm_d(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_FRM_D); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_FRM_D); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_FRM_D); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_FRM_D); }
    return scientific::copy_slice(poly_bases_pins::LAG_FRM_D);
}

fn pin_frm_l(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_FRM_L); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_FRM_L); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_FRM_L); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_FRM_L); }
    return scientific::copy_slice(poly_bases_pins::LAG_FRM_L);
}

fn pin_der_d(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_DER_D); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_DER_D); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_DER_D); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_DER_D); }
    return scientific::copy_slice(poly_bases_pins::LAG_DER_D);
}

fn pin_der_l(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_DER_L); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_DER_L); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_DER_L); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_DER_L); }
    return scientific::copy_slice(poly_bases_pins::LAG_DER_L);
}

fn pin_int_d(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_INT_D); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_INT_D); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_INT_D); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_INT_D); }
    return scientific::copy_slice(poly_bases_pins::LAG_INT_D);
}

fn pin_int_l(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_INT_L); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_INT_L); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_INT_L); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_INT_L); }
    return scientific::copy_slice(poly_bases_pins::LAG_INT_L);
}

fn pin_re_d(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_RE_D); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_RE_D); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_RE_D); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_RE_D); }
    return scientific::copy_slice(poly_bases_pins::LAG_RE_D);
}

fn pin_re_l(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_RE_L); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_RE_L); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_RE_L); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_RE_L); }
    return scientific::copy_slice(poly_bases_pins::LAG_RE_L);
}

fn pin_im_d(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_IM_D); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_IM_D); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_IM_D); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_IM_D); }
    return scientific::copy_slice(poly_bases_pins::LAG_IM_D);
}

fn pin_im_l(f: i64) -> f64[>..] {
    if f == 0 { return scientific::copy_slice(poly_bases_pins::LEG_IM_L); }
    if f == 1 { return scientific::copy_slice(poly_bases_pins::CHEB_IM_L); }
    if f == 2 { return scientific::copy_slice(poly_bases_pins::HERM_IM_L); }
    if f == 3 { return scientific::copy_slice(poly_bases_pins::HERME_IM_L); }
    return scientific::copy_slice(poly_bases_pins::LAG_IM_L);
}

fn call_val(f: i64, x: f64, c: f64[:]) -> f64, scientific::Fault {
    if f == 0 { return scientific::legval(x, c); }
    if f == 1 { return scientific::chebval(x, c); }
    if f == 2 { return scientific::hermval(x, c); }
    if f == 3 { return scientific::hermeval(x, c); }
    return scientific::lagval(x, c);
}

fn call_tom(f: i64, c: f64[:]) -> scientific::Arr, scientific::Fault {
    if f == 0 { return scientific::leg2poly(c); }
    if f == 1 { return scientific::cheb2poly(c); }
    if f == 2 { return scientific::herm2poly(c); }
    if f == 3 { return scientific::herme2poly(c); }
    return scientific::lag2poly(c);
}

fn call_frm(f: i64, c: f64[:]) -> scientific::Arr, scientific::Fault {
    if f == 0 { return scientific::poly2leg(c); }
    if f == 1 { return scientific::poly2cheb(c); }
    if f == 2 { return scientific::poly2herm(c); }
    if f == 3 { return scientific::poly2herme(c); }
    return scientific::poly2lag(c);
}

fn call_der(f: i64, c: f64[:], m: i64, scl: f64) -> scientific::Arr, scientific::Fault {
    if f == 0 { return scientific::legder(c, m, scl); }
    if f == 1 { return scientific::chebder(c, m, scl); }
    if f == 2 { return scientific::hermder(c, m, scl); }
    if f == 3 { return scientific::hermeder(c, m, scl); }
    return scientific::lagder(c, m, scl);
}

fn call_int(f: i64, c: f64[:], m: i64, k: f64[:], lb: f64, scl: f64) -> scientific::Arr, scientific::Fault {
    if f == 0 { return scientific::legint(c, m, k, lb, scl); }
    if f == 1 { return scientific::chebint(c, m, k, lb, scl); }
    if f == 2 { return scientific::hermint(c, m, k, lb, scl); }
    if f == 3 { return scientific::hermeint(c, m, k, lb, scl); }
    return scientific::lagint(c, m, k, lb, scl);
}

fn call_roots(f: i64, c: f64[:]) -> scientific::Arr, scientific::Arr, scientific::Fault {
    if f == 0 { return scientific::legroots(c); }
    if f == 1 { return scientific::chebroots(c); }
    if f == 2 { return scientific::hermroots(c); }
    if f == 3 { return scientific::hermeroots(c); }
    return scientific::lagroots(c);
}
// part gives part i of a ragged list: the data d and the lengths l.
fn part(d: f64[:], l: f64[:], i: i64) -> f64[>..] {
    var off: i64 = 0;
    var j: i64 = 0;
    while j < i { off += l[j] as i64; j += 1; }
    var out: f64[>..] = [];
    var n = l[i] as i64;
    j = 0;
    while j < n { out.push(d[off + j]); j += 1; }
    return out;
}

// near tells whether got and want have one length and agree to tol, in
// units of one plus the largest magnitude of want.
fn near(got: f64[:], want: f64[:], tol: f64) -> bool {
    if got.len != want.len { return false; }
    var big: f64 = 1.0;
    var i: i64 = 0;
    while i < want.len {
        let a = common::mathshim_abs(want[i]);
        if a > big { big = a; }
        i += 1;
    }
    i = 0;
    while i < got.len {
        if !(common::mathshim_abs(got[i] - want[i]) <= tol * big) { return false; }
        i += 1;
    }
    return true;
}

fn ser(i: i64) -> f64[>..] { return part(poly_bases_pins::S_D, poly_bases_pins::S_L, i); }

fn zeros(n: i64) -> f64[>..] {
    var z: f64[>..] = [];
    var i: i64 = 0;
    while i < n { z.push(0.0); i += 1; }
    return z;
}

// The values of a series match NumPy at every point, for every family.
fn test_poly_bases_val() {
    let want = pin_val(0);
    var f: i64 = 0;
    while f < 5 {
        let pv = pin_val(f);
        var i: i64 = 0;
        while i < 20 {
            let c = ser(i);
            var j: i64 = 0;
            while j < 6 {
                let v, fl = call_val(f, poly_bases_pins::X[j], c);
                assert(fl.kind == scientific::F_NONE);
                assert(near([v], [pv[i * 6 + j]], 1.0e-12));
                j += 1;
            }
            i += 1;
        }
        f += 1;
    }
    assert(want.len == 120);
}

// Both conversions match NumPy, for every family.
fn test_poly_bases_convert() {
    var f: i64 = 0;
    while f < 5 {
        let td = pin_tom_d(f);
        let tl = pin_tom_l(f);
        let fd = pin_frm_d(f);
        let fl2 = pin_frm_l(f);
        var i: i64 = 0;
        while i < 20 {
            let a, fa = call_tom(f, ser(i));
            assert(fa.kind == scientific::F_NONE && near(a.data, part(td, tl, i), 1.0e-12));
            let b, fb = call_frm(f, ser(i));
            assert(fb.kind == scientific::F_NONE && near(b.data, part(fd, fl2, i), 1.0e-12));
            i += 1;
        }
        f += 1;
    }
}

// The derivatives match NumPy for each option set.
fn test_poly_bases_der() {
    var f: i64 = 0;
    while f < 5 {
        let dd = pin_der_d(f);
        let dl = pin_der_l(f);
        var cs: i64 = 0;
        while cs < 5 {
            var i: i64 = 0;
            while i < 20 {
                let a, fa = call_der(f, ser(i), poly_bases_pins::DER_M[cs] as i64, poly_bases_pins::DER_S[cs]);
                assert(fa.kind == scientific::F_NONE && near(a.data, part(dd, dl, cs * 20 + i), 1.0e-12));
                i += 1;
            }
            cs += 1;
        }
        f += 1;
    }
}

// The integrals match NumPy for each option set, with the constants and the bound.
fn test_poly_bases_int() {
    var f: i64 = 0;
    while f < 5 {
        let dd = pin_int_d(f);
        let dl = pin_int_l(f);
        var cs: i64 = 0;
        while cs < 6 {
            let k = part(poly_bases_pins::INT_K_D, poly_bases_pins::INT_K_L, cs);
            var i: i64 = 0;
            while i < 20 {
                let a, fa = call_int(f, ser(i), poly_bases_pins::INT_M[cs] as i64, k,
                                     poly_bases_pins::INT_LB[cs], poly_bases_pins::INT_S[cs]);
                assert(fa.kind == scientific::F_NONE && near(a.data, part(dd, dl, cs * 20 + i), 1.0e-12));
                i += 1;
            }
            cs += 1;
        }
        f += 1;
    }
}

// The roots match NumPy for series of degree 1 to 20, real and complex.
fn test_poly_bases_roots() {
    var f: i64 = 0;
    while f < 5 {
        let rd = pin_re_d(f);
        let rl = pin_re_l(f);
        let id = pin_im_d(f);
        let il = pin_im_l(f);
        var i: i64 = 0;
        while i < 26 {
            let c = part(poly_bases_pins::R_D, poly_bases_pins::R_L, i);
            let re, im, fa = call_roots(f, c);
            assert(fa.kind == scientific::F_NONE);
            assert(near(re.data, part(rd, rl, i), 1.0e-10) && near(im.data, part(id, il, i), 1.0e-10));
            i += 1;
        }
        f += 1;
    }
}

// The roots of the basis polynomial of degree 20 are the Gauss nodes.
fn test_poly_bases_roots_basis() {
    var f: i64 = 0;
    while f < 5 {
        let c = zeros(21);
        var c2 = c;
        c2[20] = 1.0;
        let re, im, fa = call_roots(f, c2);
        assert(fa.kind == scientific::F_NONE);
        assert(near(re.data, pin_b20(f), 1.0e-12));
        // The roots are all real, so the imaginary parts are exactly zero.
        assert(near(im.data, zeros(20), 0.0));
        f += 1;
    }
}

// A few values that a reader can check by hand.
fn test_poly_bases_known() {
    let c = [1.0, 2.0, 3.0];
    let v, f = scientific::legval(0.5, c);
    assert(f.kind == scientific::F_NONE && common::close(v, 1.625, 1.0e-14));
    let d, fd = scientific::legder(c);
    assert(fd.kind == scientific::F_NONE && near(d.data, [2.0, 9.0], 1.0e-14));
    let re, im, fr = scientific::legroots(c);
    assert(fr.kind == scientific::F_NONE && near(re.data, [-0.62283903, 0.17839459], 1.0e-8));
    let p, fp = scientific::leg2poly(c);
    assert(fp.kind == scientific::F_NONE && near(p.data, [-0.5, 2.0, 4.5], 1.0e-14));
    let q, fq = scientific::poly2leg(c);
    assert(fq.kind == scientific::F_NONE && near(q.data, [2.0, 2.0, 2.0], 1.0e-14));
    // One coefficient gives that coefficient at every point.
    let s, fs = scientific::lagval(7.5, [3.25]);
    assert(fs.kind == scientific::F_NONE && s == 3.25);
}

// The trailing zeros count in the value, the derivative and the integral
// and drop out of the conversions and the roots, as in NumPy.
fn test_poly_bases_trailing_zeros() {
    let c = [1.0, 2.0, 0.0];
    let v, f = scientific::legval(0.5, c);
    assert(f.kind == scientific::F_NONE && v == 2.0);
    let d, fd = scientific::legder(c);
    assert(fd.kind == scientific::F_NONE && near(d.data, [2.0, 0.0], 0.0));
    let t, ft = scientific::legint(c);
    assert(ft.kind == scientific::F_NONE && t.data.len == 4);
    let p, fp = scientific::leg2poly(c);
    assert(fp.kind == scientific::F_NONE && near(p.data, [1.0, 2.0], 0.0));
    let re, im, fr = scientific::legroots(c);
    assert(fr.kind == scientific::F_NONE && near(re.data, [-0.5], 1.0e-14) && near(im.data, [0.0], 0.0));
    // A series of zeros has the degree zero.
    let re0, im0, fz = scientific::chebroots([0.0, 0.0]);
    assert(fz.kind == scientific::F_NONE && re0.data.len == 0 && im0.data.len == 0);
    let z, z2, fz2 = scientific::hermeroots([5.0]);
    assert(fz2.kind == scientific::F_NONE && z.data.len == 0);
}

// The closed forms of the series of degree one.
fn test_poly_bases_degree_one() {
    let a, b2, fa2 = scientific::legroots([1.0, 2.0]);
    assert(fa2.kind == scientific::F_NONE && common::close(a.data[0], -0.5, 1.0e-15));
    let b, c2, fb2 = scientific::hermroots([1.0, 2.0]);
    assert(fb2.kind == scientific::F_NONE && common::close(b.data[0], -0.25, 1.0e-15));
    let c, d2, fc2 = scientific::lagroots([1.0, 2.0]);
    assert(fc2.kind == scientific::F_NONE && common::close(c.data[0], 1.5, 1.0e-15));
    let d, e2, fd2 = scientific::hermeroots([1.0, 2.0]);
    assert(fd2.kind == scientific::F_NONE && common::close(d.data[0], -0.5, 1.0e-15));
    let e, f2, fe2 = scientific::chebroots([1.0, 2.0]);
    assert(fe2.kind == scientific::F_NONE && common::close(e.data[0], -0.5, 1.0e-15));
}

// The order, the constants and the zero series.
fn test_poly_bases_order_edges() {
    let c = [1.0, 2.0, 3.0];
    // The order zero gives the series. An order at the length gives zero.
    let a, fa = scientific::chebder(c, 0);
    assert(fa.kind == scientific::F_NONE && near(a.data, c, 0.0));
    let b, fb = scientific::lagder(c, 3);
    assert(fb.kind == scientific::F_NONE && near(b.data, [0.0], 0.0));
    let b2, fb2 = scientific::hermder(c, 9);
    assert(fb2.kind == scientific::F_NONE && near(b2.data, [0.0], 0.0));
    let b3, fb3 = scientific::hermeder(c, 3);
    assert(fb3.kind == scientific::F_NONE && near(b3.data, [0.0], 0.0));
    let n, fnn = scientific::legint(c, 0);
    assert(fnn.kind == scientific::F_NONE && near(n.data, c, 0.0));
    // A zero series gets the constant only.
    let z, fz = scientific::hermint([0.0], 1, [4.0]);
    assert(fz.kind == scientific::F_NONE && near(z.data, [4.0], 0.0));
    // The integral at the bound equals the constant.
    let w, fw = scientific::chebint(c, 1, [2.5], 0.75);
    assert(fw.kind == scientific::F_NONE);
    let at, fat = scientific::chebval(0.75, w.data);
    assert(fat.kind == scientific::F_NONE && common::close(at, 2.5, 1.0e-13));
    // The derivative undoes the integral, up to the constant.
    let u, fu = scientific::lagint(c, 2, [], 0.0);
    let v, fv = scientific::lagder(u.data, 2);
    assert(fu.kind == scientific::F_NONE && fv.kind == scientific::F_NONE && near(v.data, c, 1.0e-12));
}

// The faults of the limits against hostile input.
fn test_poly_bases_faults() {
    let inf = 1.0 / 0.0;
    let nan = inf - inf;
    var f: i64 = 0;
    while f < 5 {
        let one = [1.0, 2.0, 3.0];
        var empty: f64[>..] = [];
        // An empty series.
        let v0, f0 = call_val(f, 0.5, empty);
        assert(f0.kind == scientific::F_SHAPE);
        let a0, fa0 = call_tom(f, empty);
        assert(fa0.kind == scientific::F_SHAPE);
        let b0, fb0 = call_frm(f, empty);
        assert(fb0.kind == scientific::F_SHAPE);
        let d0, fd0 = call_der(f, empty, 1, 1.0);
        assert(fd0.kind == scientific::F_SHAPE);
        let i0, fi0 = call_int(f, empty, 1, [], 0.0, 1.0);
        assert(fi0.kind == scientific::F_SHAPE);
        let r0, s0, fr0 = call_roots(f, empty);
        assert(fr0.kind == scientific::F_SHAPE);
        // A coefficient that is not finite.
        let v1, f1 = call_val(f, 0.5, [1.0, inf]);
        assert(f1.kind == scientific::F_VALUE);
        let a1, fa1 = call_tom(f, [1.0, nan, 2.0]);
        assert(fa1.kind == scientific::F_VALUE);
        let r1, s1, fr1 = call_roots(f, [1.0, nan, 2.0]);
        assert(fr1.kind == scientific::F_VALUE);
        // A point, a bound, a scale or a constant that is not finite.
        let v2, f2 = call_val(f, nan, one);
        assert(f2.kind == scientific::F_VALUE);
        let d2, fd2 = call_der(f, one, 1, inf);
        assert(fd2.kind == scientific::F_VALUE);
        let d3, fd3 = call_der(f, one, 1, 0.0);
        assert(fd3.kind == scientific::F_VALUE);
        let i2, fi2 = call_int(f, one, 1, [], inf, 1.0);
        assert(fi2.kind == scientific::F_VALUE);
        let i3, fi3 = call_int(f, one, 1, [nan], 0.0, 1.0);
        assert(fi3.kind == scientific::F_VALUE);
        let i4, fi4 = call_int(f, one, 1, [], 0.0, 0.0);
        assert(fi4.kind == scientific::F_VALUE);
        // More constants than the order.
        let i5, fi5 = call_int(f, one, 1, [1.0, 2.0], 0.0, 1.0);
        assert(fi5.kind == scientific::F_VALUE);
        // The order.
        let d4, fd4 = call_der(f, one, -1, 1.0);
        assert(fd4.kind == scientific::F_RANGE);
        let i6, fi6 = call_int(f, one, -1, [], 0.0, 1.0);
        assert(fi6.kind == scientific::F_RANGE);
        let d5, fd5 = call_der(f, one, 1001, 1.0);
        assert(fd5.kind == scientific::F_LIMIT);
        let i7, fi7 = call_int(f, one, 1001, [], 0.0, 1.0);
        assert(fi7.kind == scientific::F_LIMIT);
        // A series of 1002 coefficients.
        let long = zeros(1002);
        let v3, f3 = call_val(f, 0.5, long);
        assert(f3.kind == scientific::F_LIMIT);
        let a3, fa3 = call_tom(f, long);
        assert(fa3.kind == scientific::F_LIMIT);
        // A degree above 512.
        var deg = zeros(514);
        deg[513] = 1.0;
        let r3, s3, fr3 = call_roots(f, deg);
        assert(fr3.kind == scientific::F_LIMIT);
        // A result that is not finite.
        let v4, f4 = call_val(f, 10.0, [1.0e308, 1.0e308, 1.0e308]);
        assert(f4.kind == scientific::F_VALUE);
        let r4, s4, fr4 = call_roots(f, [1.0e300, 1.0, 1.0e-300]);
        assert(fr4.kind == scientific::F_VALUE);
        f += 1;
    }
}

// A series of the highest allowed length gives a result and no fault.
fn test_poly_bases_long_series() {
    var f: i64 = 0;
    while f < 5 {
        var c = zeros(1001);
        c[0] = 1.0;
        let v, fv = call_val(f, 0.5, c);
        assert(fv.kind == scientific::F_NONE && common::close(v, 1.0, 1.0e-15));
        let a, fa = call_tom(f, c);
        assert(fa.kind == scientific::F_NONE && near(a.data, [1.0], 0.0));
        let b, fb = call_frm(f, c);
        assert(fb.kind == scientific::F_NONE && near(b.data, [1.0], 0.0));
        // A high coefficient gives a value or a fault, and never a crash.
        c[1000] = 1.0;
        let v2, fv2 = call_val(f, 0.5, c);
        assert(fv2.kind == scientific::F_NONE || fv2.kind == scientific::F_VALUE);
        let a2, fa2 = call_tom(f, c);
        assert(fa2.kind == scientific::F_NONE || fa2.kind == scientific::F_VALUE);
        f += 1;
    }
}
