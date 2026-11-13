// The tests of the group ops. Run them with SCIENTIFIC_SUITES=ops.
namespace ops_tests;

import scientific from "../..";
import os;
import common;

// The elementwise operations broadcast and hold the IEEE rules.
fn test_ops() {
    let a = common::make23();
    let row, f0 = scientific::from_values([10.0, 20.0, 30.0]);
    // A matrix and a row broadcast over the last dimension.
    let s, f = scientific::add(a, row);
    assert(f.kind == scientific::F_NONE);
    assert(s.ndim == 2 && s.shape[0] == 2 && s.shape[1] == 3);
    assert(s.data[0] == 11.0 && s.data[2] == 33.0 && s.data[3] == 14.0 && s.data[5] == 36.0);
    let d, f1 = scientific::sub(a, row);
    assert(f1.kind == scientific::F_NONE && d.data[4] == -15.0);
    let m, f2 = scientific::mul(a, row);
    assert(f2.kind == scientific::F_NONE && m.data[1] == 40.0);
    let dv, f3 = scientific::div(row, row);
    assert(f3.kind == scientific::F_NONE);
    assert(dv.data[0] == 1.0 && dv.data[2] == 1.0);
    let mx, f4 = scientific::maximum(a, row);
    assert(f4.kind == scientific::F_NONE && mx.data[0] == 10.0 && mx.data[3] == 10.0);
    let mn, f5 = scientific::minimum(a, row);
    assert(f5.kind == scientific::F_NONE && mn.data[0] == 1.0 && mn.data[2] == 3.0);
    // Two shapes that hold 2 and 3 at one position are a fault.
    let other, f5b = scientific::from_rows(1, 2, [1.0, 2.0]);
    let bad, f6 = scientific::add(a, other);
    assert(f6.kind == scientific::F_SHAPE);
    // A column shape of [2, 1] broadcasts over the columns.
    let col, f7 = scientific::from_rows(2, 1, [100.0, 200.0]);
    let s2, f8 = scientific::add(a, col);
    assert(f8.kind == scientific::F_NONE && s2.data[0] == 101.0 && s2.data[5] == 206.0);
    // The unary operations.
    let n, f9 = scientific::neg(a);
    assert(f9.kind == scientific::F_NONE && n.data[0] == -1.0);
    let ab, f10 = scientific::abs(n);
    assert(f10.kind == scientific::F_NONE && ab.data[2] == 3.0);
    let rt, f11 = scientific::sqrt(row);
    assert(f11.kind == scientific::F_NONE && common::close(rt.data[1], 4.47213595));
    let ex, f12 = scientific::exp(row);
    assert(f12.kind == scientific::F_NONE && common::close(ex.data[0], 22026.465794807));
    let lg, f13 = scientific::log(row);
    assert(f13.kind == scientific::F_NONE && common::close(lg.data[2], 3.40119738));
    // scale and clip.
    let sc, f14 = scientific::scale(a, 2.0);
    assert(f14.kind == scientific::F_NONE && sc.data[5] == 12.0);
    let cl, f15 = scientific::clip(a, 2.0, 4.0);
    assert(f15.kind == scientific::F_NONE);
    assert(cl.data[0] == 2.0 && cl.data[2] == 3.0 && cl.data[5] == 4.0);
    let cl2, f16 = scientific::clip(a, 5.0, 2.0);
    assert(f16.kind == scientific::F_VALUE);
    // The IEEE rules: a division by zero gives an infinity, a log of
    // a number below zero gives NaN.
    let zero, f17 = scientific::from_values([0.0]);
    let inf, f18 = scientific::div(row, zero);
    assert(f18.kind == scientific::F_NONE);
    let inf_bit = inf.data[1];
    assert(inf_bit == inf_bit + 1.0);
    let neg1, f19 = scientific::full([1], -1.0);
    let nan, f20 = scientific::log(neg1);
    assert(f20.kind == scientific::F_NONE);
    let nan_bit = nan.data[0];
    assert(nan_bit != nan_bit);
}

// The reductions over the whole array.
fn test_reduce() {
    let a = common::make23();
    let s, f = scientific::sum(a);
    assert(f.kind == scientific::F_NONE && s == 21.0);
    let p, f2 = scientific::prod(a);
    assert(f2.kind == scientific::F_NONE && p == 720.0);
    let mn, f3 = scientific::min(a);
    assert(f3.kind == scientific::F_NONE && mn == 1.0);
    let mx, f4 = scientific::max(a);
    assert(f4.kind == scientific::F_NONE && mx == 6.0);
    let m, f5 = scientific::mean(a);
    assert(f5.kind == scientific::F_NONE && common::close(m, 3.5));
}

// The reductions along one axis, the flat positions of the extremes
// and the permutation of the dimensions.
fn test_axis_and_permute() {
    let a = common::make23();
    // Over the rows: one result per column.
    let sr, f = scientific::sum_axis(a, 0);
    assert(f.kind == scientific::F_NONE && sr.ndim == 1 && sr.shape[0] == 3);
    assert(sr.data[0] == 5.0 && sr.data[2] == 9.0);
    // Over the columns: one result per row.
    let sc, f2 = scientific::sum_axis(a, 1);
    assert(f2.kind == scientific::F_NONE && sc.shape[0] == 2);
    assert(sc.data[0] == 6.0 && sc.data[1] == 15.0);
    let pc, f3 = scientific::prod_axis(a, 1);
    assert(f3.kind == scientific::F_NONE && pc.data[0] == 6.0 && pc.data[1] == 120.0);
    let mnc, f4 = scientific::min_axis(a, 0);
    assert(f4.kind == scientific::F_NONE && mnc.data[1] == 2.0);
    let mxc, f5 = scientific::max_axis(a, 0);
    assert(f5.kind == scientific::F_NONE && mxc.data[1] == 5.0);
    let mc, f6 = scientific::mean_axis(a, 1);
    assert(f6.kind == scientific::F_NONE && common::close(mc.data[1], 5.0));
    // A three-dimensional array reduces to a matrix and to a row.
    let cu, f7 = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let deep, f8 = scientific::expand_dims(cu, 0);
    assert(f8.kind == scientific::F_NONE && deep.ndim == 3);
    let d0, f9 = scientific::sum_axis(deep, 0);
    assert(f9.kind == scientific::F_NONE && d0.ndim == 2 && d0.data[0] == 1.0 && d0.data[5] == 6.0);
    let d1, f10 = scientific::sum_axis(deep, 1);
    assert(f10.kind == scientific::F_NONE && d1.ndim == 2 && d1.shape[1] == 3 && d1.data[0] == 5.0 && d1.data[2] == 9.0);
    // A bad axis is a fault with an empty result.
    let bad, f11 = scientific::sum_axis(a, 2);
    assert(f11.kind == scientific::F_RANGE && bad.data.len == 0);
    // The flat position of the extremes. The first of equals wins.
    let v, f12 = scientific::from_values([3.0, 1.0, 4.0, 1.0, 5.0]);
    let imn, f13 = scientific::argmin(v);
    assert(f13.kind == scientific::F_NONE && imn == 1);
    let imx, f14 = scientific::argmax(v);
    assert(f14.kind == scientific::F_NONE && imx == 4);
    let ze, f16 = scientific::argmin(a);
    assert(f16.kind == scientific::F_NONE && ze == 0);
    // The general permutation.
    let c1, f17 = scientific::reshape(a, [1, 2, 3]);
    let p1, f18 = scientific::transpose_perm(c1, [2, 0, 1]);
    assert(f18.kind == scientific::F_NONE && p1.ndim == 3);
    assert(p1.shape[0] == 3 && p1.shape[1] == 1 && p1.shape[2] == 2);
    assert(p1.data[0] == 1.0 && p1.data[1] == 4.0);
    // The default order is the reverse, which matches the transpose of a matrix.
    let rev, f19 = scientific::transpose_perm(a, [1, 0]);
    assert(f19.kind == scientific::F_NONE && rev.data[3] == 5.0);
    // A permutation that repeats an axis or leaves one out is a fault.
    let bad1, f20 = scientific::transpose_perm(a, [0, 0]);
    assert(f20.kind == scientific::F_SHAPE);
    let bad2, f21 = scientific::transpose_perm(a, [0]);
    assert(f21.kind == scientific::F_SHAPE);
    let bad3, f22 = scientific::transpose_perm(a, [0, 2]);
    assert(f22.kind == scientific::F_RANGE);
}

// Elementwise math, comparisons, selection, running sums, spread.
fn test_logic_and_math() {
    let v, f = scientific::from_values([-2.5, -0.5, 0.5, 1.5, 2.5]);
    let fl, f1 = scientific::floor(v);
    assert(fl.data[0] == -3.0 && fl.data[1] == -1.0 && fl.data[3] == 1.0);
    let ce, f2 = scientific::ceil(v);
    assert(ce.data[0] == -2.0 && ce.data[1] == 0.0 && ce.data[3] == 2.0);
    let ro, f3 = scientific::round(v);
    assert(ro.data[0] == -2.0 && ro.data[2] == 0.0 && ro.data[3] == 2.0 && ro.data[4] == 2.0);
    let tr, f4 = scientific::trunc(v);
    assert(tr.data[0] == -2.0 && tr.data[1] == 0.0);
    let sg, f5 = scientific::sign(v);
    assert(sg.data[0] == -1.0 && sg.data[3] == 1.0);
    let sq, f6 = scientific::square(v);
    assert(sq.data[0] == 6.25);
    let p4, f7 = scientific::from_values([1.0, 4.0, 8.0]);
    let l2, f8 = scientific::log2(p4);
    assert(common::close(l2.data[1], 2.0) && common::close(l2.data[2], 3.0));
    let l10, f9 = scientific::log10(p4);
    assert(common::close(l10.data[0], 0.0));
    let two, f10 = scientific::full([1], 2.0);
    let pw, f11 = scientific::power(p4, two);
    assert(f11.kind == scientific::F_NONE && pw.data[2] == 64.0);
    // Comparisons give 1.0 and 0.0 and broadcast.
    let a = common::make23();
    let three, f12 = scientific::full([1], 3.0);
    let gt, f13 = scientific::greater(a, three);
    assert(f13.kind == scientific::F_NONE && gt.data[2] == 0.0 && gt.data[3] == 1.0);
    let ge, f14 = scientific::greater_equal(a, three);
    assert(ge.data[2] == 1.0 && ge.data[1] == 0.0);
    let eq, f15 = scientific::equal(a, three);
    assert(eq.data[2] == 1.0 && eq.data[3] == 0.0);
    let ne, f16 = scientific::not_equal(a, three);
    assert(ne.data[2] == 0.0);
    let lt, f17 = scientific::less(a, three);
    let le, f18 = scientific::less_equal(a, three);
    assert(lt.data[2] == 0.0 && le.data[2] == 1.0);
    let la, f19 = scientific::logical_and(gt, ge);
    assert(la.data[3] == 1.0 && la.data[2] == 0.0);
    let lo, f20 = scientific::logical_or(gt, eq);
    assert(lo.data[2] == 1.0 && lo.data[0] == 0.0);
    let lx, f21 = scientific::logical_xor(gt, ge);
    assert(lx.data[2] == 1.0 && lx.data[3] == 0.0);
    let ln, f22 = scientific::logical_not(gt);
    assert(ln.data[0] == 1.0 && ln.data[5] == 0.0);
    let anyv, f23 = scientific::any(gt);
    let allv, f24 = scientific::all(gt);
    let cnt, f25 = scientific::count_nonzero(gt);
    assert(anyv && !allv && cnt == 3);
    // isnan, isinf, isfinite.
    let zero, f26 = scientific::zeros([2]);
    let bad, f27 = scientific::div(zero, zero);
    let zero1, f28b = scientific::zeros([1]);
    let inf, f28 = scientific::div(p4, zero1);
    let nn, f29 = scientific::isnan(bad);
    let ni, f30 = scientific::isinf(inf);
    let nf, f31 = scientific::isfinite(inf);
    let nf2, g32 = scientific::isfinite(p4);
    assert(nn.data[0] == 1.0 && ni.data[0] == 1.0 && nf.data[0] == 0.0 && nf2.data[0] == 1.0);
    // where selects with broadcasting.
    let w, f33 = scientific::where(gt, a, three);
    assert(f33.kind == scientific::F_NONE && w.data[0] == 3.0 && w.data[3] == 4.0 && w.data[5] == 6.0);
    // cumulative sums, differences and flips.
    let cs, f34 = scientific::cumsum(a, 1);
    assert(cs.data[2] == 6.0 && cs.data[4] == 9.0 && cs.data[5] == 15.0);
    let cp, f35 = scientific::cumprod(a, 0);
    assert(cp.data[3] == 4.0 && cp.data[5] == 18.0);
    let df, f36 = scientific::diff(a, 1);
    assert(df.shape[1] == 2 && df.data[0] == 1.0 && df.data[3] == 1.0);
    let df2, f37 = scientific::diff(a, 0);
    assert(df2.shape[0] == 1 && df2.data[2] == 3.0);
    let fp, f38 = scientific::flip(a, 1);
    assert(fp.data[0] == 3.0 && fp.data[3] == 6.0);
    let fp0, f39 = scientific::flip(a, 0);
    assert(fp0.data[0] == 4.0 && fp0.data[5] == 3.0);
    let one, f40 = scientific::ones([1, 1]);
    let bd, f41 = scientific::diff(one, 0);
    assert(f41.kind == scientific::F_SHAPE);
    let bf, f42 = scientific::flip(a, 2);
    assert(f42.kind == scientific::F_RANGE);
    // variance, deviation and span.
    let sv, f43 = scientific::from_values([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    let va, f44 = scientific::variance(sv, 0);
    assert(f44.kind == scientific::F_NONE && common::close(va, 4.0));
    let sd, f45 = scientific::std(sv, 0);
    assert(common::close(sd, 2.0));
    let va1, f46 = scientific::variance(sv, 1);
    assert(common::close(va1, 32.0 / 7.0));
    let vb, f47 = scientific::variance(sv, 8);
    assert(f47.kind == scientific::F_VALUE);
    let vx, f48 = scientific::var_axis(a, 1, 0);
    assert(f48.kind == scientific::F_NONE && common::close(vx.data[0], 2.0 / 3.0) && common::close(vx.data[1], 2.0 / 3.0));
    let sx, f49 = scientific::std_axis(a, 0, 0);
    assert(common::close(sx.data[0], 1.5));
    let sp, f50 = scientific::ptp(a);
    assert(sp == 5.0);
}

// The elementwise math holds the values of NumPy.
fn test_elementwise_math() {
    let a, f = scientific::from_values([0.25, 0.5, 1.0, 2.0]);
    assert(f.kind == scientific::F_NONE);
    let s, fs = scientific::sin(a);
    assert(fs.kind == scientific::F_NONE);
    assert(common::close(s.data[0], 0.24740395925452294) && common::close(s.data[1], 0.479425538604203));
    assert(common::close(s.data[2], 0.8414709848078965) && common::close(s.data[3], 0.9092974268256817));
    let c, fc = scientific::cos(a);
    assert(fc.kind == scientific::F_NONE);
    assert(common::close(c.data[0], 0.9689124217106447) && common::close(c.data[1], 0.8775825618903728));
    assert(common::close(c.data[2], 0.5403023058681398) && common::close(c.data[3], -0.4161468365471424));
    let t, ft = scientific::tan(a);
    assert(ft.kind == scientific::F_NONE);
    assert(common::close(t.data[0], 0.25534192122103627) && common::close(t.data[1], 0.5463024898437905));
    assert(common::close(t.data[2], 1.5574077246549023) && common::close(t.data[3], -2.185039863261519));
    let asin2, fas = scientific::arcsin(a);
    assert(fas.kind == scientific::F_NONE);
    assert(common::close(asin2.data[0], 0.25268025514207865) && common::close(asin2.data[1], 0.5235987755982989));
    assert(common::close(asin2.data[2], 1.5707963267948966) && asin2.data[3] != asin2.data[3]);
    let ac, fac = scientific::arccos(a);
    assert(fac.kind == scientific::F_NONE);
    assert(common::close(ac.data[0], 1.318116071652818) && common::close(ac.data[1], 1.0471975511965979));
    assert(common::close(ac.data[2], 0.0) && ac.data[3] != ac.data[3]);
    let at, fat = scientific::arctan(a);
    assert(fat.kind == scientific::F_NONE);
    assert(common::close(at.data[0], 0.24497866312686414) && common::close(at.data[1], 0.4636476090008061));
    assert(common::close(at.data[2], 0.7853981633974483) && common::close(at.data[3], 1.1071487177940904));
    // The angle of a point of the plane comes from two arrays.
    let y, fy = scientific::from_values([1.0, -1.0, 0.0, 2.0]);
    let x, fx = scientific::from_values([1.0, 1.0, -1.0, -2.0]);
    let a2, fa2 = scientific::arctan2(y, x);
    assert(fa2.kind == scientific::F_NONE);
    assert(common::close(a2.data[0], 0.7853981633974483) && common::close(a2.data[1], -0.7853981633974483));
    assert(common::close(a2.data[2], 3.141592653589793) && common::close(a2.data[3], 2.356194490192345));
    // The hyperbolic functions hold the values of NumPy.
    let h, fh = scientific::from_values([-1.0, -0.5, 0.0, 0.5, 1.0]);
    assert(fh.kind == scientific::F_NONE);
    let sh, fsh = scientific::sinh(h);
    assert(fsh.kind == scientific::F_NONE);
    assert(common::close(sh.data[0], -1.1752011936438014) && common::close(sh.data[1], -0.5210953054937474));
    assert(common::close(sh.data[2], 0.0) && common::close(sh.data[3], 0.5210953054937474));
    assert(common::close(sh.data[4], 1.1752011936438014));
    let ch, fch = scientific::cosh(h);
    assert(fch.kind == scientific::F_NONE);
    assert(common::close(ch.data[0], 1.5430806348152437) && common::close(ch.data[1], 1.1276259652063807));
    assert(common::close(ch.data[2], 1.0) && common::close(ch.data[3], 1.1276259652063807));
    assert(common::close(ch.data[4], 1.5430806348152437));
    let th, fth = scientific::tanh(h);
    assert(fth.kind == scientific::F_NONE);
    assert(common::close(th.data[0], -0.7615941559557649) && common::close(th.data[1], -0.46211715726000974));
    assert(common::close(th.data[2], 0.0) && common::close(th.data[3], 0.46211715726000974));
    assert(common::close(th.data[4], 0.7615941559557649));
    let ash, fash = scientific::arcsinh(h);
    assert(fash.kind == scientific::F_NONE);
    assert(common::close(ash.data[0], -0.881373587019543) && common::close(ash.data[1], -0.48121182505960347));
    assert(common::close(ash.data[2], 0.0) && common::close(ash.data[3], 0.48121182505960347));
    assert(common::close(ash.data[4], 0.881373587019543));
    // The arc of the hyperbolic cosine holds no value below one.
    let ac2, fac2 = scientific::arccosh(scientific::from_values([2.0, 1.5, 1.0, 1.5, 2.0]));
    assert(fac2.kind == scientific::F_NONE);
    assert(common::close(ac2.data[0], 1.3169578969248168) && common::close(ac2.data[1], 0.9624236501192069));
    assert(common::close(ac2.data[2], 0.0) && common::close(ac2.data[3], 0.9624236501192069));
    assert(common::close(ac2.data[4], 1.3169578969248168));
    let ac3, fac3 = scientific::arccosh(scientific::from_values([0.5]));
    assert(fac3.kind == scientific::F_NONE && ac3.data[0] != ac3.data[0]);
    let ath, fath = scientific::arctanh(scientific::from_values([-0.9, -0.5, 0.0, 0.5, 0.9]));
    assert(fath.kind == scientific::F_NONE);
    assert(common::close(ath.data[0], -1.4722194895832204) && common::close(ath.data[1], -0.5493061443340549));
    assert(common::close(ath.data[2], 0.0) && common::close(ath.data[3], 0.5493061443340549));
    assert(common::close(ath.data[4], 1.4722194895832204));
    // The arc of the hyperbolic tangent holds an infinity at one.
    let ath2, fath2 = scientific::arctanh(scientific::from_values([1.0, 2.0]));
    assert(fath2.kind == scientific::F_NONE);
    assert(ath2.data[0] > 1000000.0 && ath2.data[1] != ath2.data[1]);
    // The power and the log of a small number hold more digits.
    let e, fe = scientific::from_values([-0.25, 0.0, 0.25, 0.75]);
    assert(fe.kind == scientific::F_NONE);
    let em, fem = scientific::expm1(e);
    assert(fem.kind == scientific::F_NONE);
    assert(common::close(em.data[0], -0.22119921692859512) && common::close(em.data[1], 0.0));
    assert(common::close(em.data[2], 0.2840254166877415) && common::close(em.data[3], 1.1170000166126748));
    let lp, flp = scientific::log1p(e);
    assert(flp.kind == scientific::F_NONE);
    assert(common::close(lp.data[0], -0.2876820724517809) && common::close(lp.data[1], 0.0));
    assert(common::close(lp.data[2], 0.22314355131420976) && common::close(lp.data[3], 0.5596157879354227));
    // The reciprocal of a number gives the number that multiplies to one.
    let r, fr = scientific::reciprocal(scientific::from_values([-2.0, 0.5, 4.0]));
    assert(fr.kind == scientific::F_NONE);
    assert(common::close(r.data[0], -0.5) && common::close(r.data[1], 2.0) && common::close(r.data[2], 0.25));
    // The degrees and the radians follow each other.
    let dr, fdr = scientific::deg2rad(scientific::from_values([0.0, 45.0, 90.0, 180.0]));
    assert(fdr.kind == scientific::F_NONE);
    assert(common::close(dr.data[0], 0.0) && common::close(dr.data[1], 0.7853981633974483));
    assert(common::close(dr.data[2], 1.5707963267948966) && common::close(dr.data[3], 3.141592653589793));
    let rd, frd = scientific::rad2deg(scientific::from_values([0.0, 0.7853981633974483, 1.5707963267948966, 3.141592653589793]));
    assert(frd.kind == scientific::F_NONE);
    assert(common::close(rd.data[0], 0.0) && common::close(rd.data[1], 45.0));
    assert(common::close(rd.data[2], 90.0) && common::close(rd.data[3], 180.0));
    // The sign of minus holds a zero of the negative side.
    let sb, fsb = scientific::signbit(scientific::from_values([1.0, -1.0, 0.0, 0.0 * -1.0]));
    assert(fsb.kind == scientific::F_NONE);
    assert(sb.data.len == 4 && sb.data[0] == 0.0 && sb.data[1] == 1.0);
    assert(sb.data[2] == 0.0 && sb.data[3] == 1.0);
    // The infinities hold their side.
    let inf = 1.0 / 0.0;
    let ni, fni = scientific::isneginf(scientific::from_values([inf, -inf, 0.0]));
    assert(fni.kind == scientific::F_NONE);
    assert(ni.data[0] == 0.0 && ni.data[1] == 1.0 && ni.data[2] == 0.0);
    let pi, fpi = scientific::isposinf(scientific::from_values([inf, -inf, 0.0]));
    assert(fpi.kind == scientific::F_NONE);
    assert(pi.data[0] == 1.0 && pi.data[1] == 0.0 && pi.data[2] == 0.0);
    // A number of a huge size holds no infinity.
    let ii, fii = scientific::isinf(scientific::from_values([1.0e300, 1.0]));
    assert(fii.kind == scientific::F_NONE && ii.data[0] == 0.0 && ii.data[1] == 0.0);
    let fi, ffi = scientific::isfinite(scientific::from_values([1.0e300, inf]));
    assert(ffi.kind == scientific::F_NONE && fi.data[0] == 1.0 && fi.data[1] == 0.0);
    // The quotient that rounds down follows the rule of NumPy.
    let qa, fqa = scientific::from_values([7.0, -7.0, 7.5, 1.0]);
    let qb, fqb = scientific::from_values([2.0, 2.0, 2.0, 0.1]);
    let q, fq = scientific::floor_divide(qa, qb);
    assert(fq.kind == scientific::F_NONE);
    assert(common::close(q.data[0], 3.0) && common::close(q.data[1], -4.0));
    assert(common::close(q.data[2], 3.0) && common::close(q.data[3], 9.0));
    // A division by zero holds an infinity.
    let q0, fq0 = scientific::floor_divide(scientific::from_values([1.0, -1.0, 0.0]),
                                            scientific::from_values([0.0, 0.0, 0.0]));
    assert(fq0.kind == scientific::F_NONE);
    assert(q0.data[0] > 0.0 && q0.data[1] < 0.0 && q0.data[2] != q0.data[2]);
    // Two shapes that hold no broadcast give a fault.
    let qbad, fqbad = scientific::floor_divide(scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                                                 scientific::from_values([1.0, 2.0]));
    assert(fqbad.kind == scientific::F_SHAPE && qbad.data.len == 0);
    // The names of NumPy give the same values.
    let nv, fnv = scientific::from_values([-2.0, 0.0, 3.0]);
    let sub2, fsub2 = scientific::subtract(nv, scientific::from_values([1.0, 1.0, 1.0]));
    assert(fsub2.kind == scientific::F_NONE);
    assert(common::close(sub2.data[0], -3.0) && common::close(sub2.data[1], -1.0) && common::close(sub2.data[2], 2.0));
    let mul2, fmul2 = scientific::multiply(nv, scientific::from_values([2.0, 2.0, 2.0]));
    assert(fmul2.kind == scientific::F_NONE);
    assert(common::close(mul2.data[0], -4.0) && common::close(mul2.data[1], 0.0) && common::close(mul2.data[2], 6.0));
    let div2, fdiv2 = scientific::divide(nv, scientific::from_values([2.0, 2.0, 2.0]));
    assert(fdiv2.kind == scientific::F_NONE);
    assert(common::close(div2.data[0], -1.0) && common::close(div2.data[1], 0.0) && common::close(div2.data[2], 1.5));
    let td, ftd = scientific::true_divide(nv, scientific::from_values([2.0, 2.0, 2.0]));
    assert(ftd.kind == scientific::F_NONE);
    assert(common::close(td.data[0], -1.0) && common::close(td.data[1], 0.0) && common::close(td.data[2], 1.5));
    let neg2, fneg2 = scientific::negative(nv);
    assert(fneg2.kind == scientific::F_NONE);
    assert(common::close(neg2.data[0], 2.0) && common::close(neg2.data[1], 0.0) && common::close(neg2.data[2], -3.0));
    let pos2, fpos2 = scientific::positive(nv);
    assert(fpos2.kind == scientific::F_NONE);
    assert(common::close(pos2.data[0], -2.0) && common::close(pos2.data[1], 0.0) && common::close(pos2.data[2], 3.0));
    let fab2, ffab2 = scientific::fabs(nv);
    assert(ffab2.kind == scientific::F_NONE);
    assert(common::close(fab2.data[0], 2.0) && common::close(fab2.data[1], 0.0) && common::close(fab2.data[2], 3.0));
    let abs2, fabs2 = scientific::absolute(nv);
    assert(fabs2.kind == scientific::F_NONE);
    assert(common::close(abs2.data[0], 2.0) && common::close(abs2.data[1], 0.0) && common::close(abs2.data[2], 3.0));
    // A half goes to the even neighbour, and the decimals give the
    // place of the last digit.
    let ar, far = scientific::around(scientific::from_values([0.125, 0.375, 2.675, -1.5]), 2);
    assert(far.kind == scientific::F_NONE);
    assert(common::close(ar.data[0], 0.12) && common::close(ar.data[1], 0.38));
    assert(common::close(ar.data[2], 2.68) && common::close(ar.data[3], -1.5));
    let ar2, far2 = scientific::around(scientific::from_values([1234.0, 1236.0]), -1);
    assert(far2.kind == scientific::F_NONE);
    assert(common::close(ar2.data[0], 1230.0) && common::close(ar2.data[1], 1240.0));
    // The name of NumPy for an array of values gives the same array.
    let arr, farr = scientific::array([1.0, 2.0, 3.0]);
    assert(farr.kind == scientific::F_NONE && arr.data.len == 3 && arr.data[2] == 3.0);
}

// The rest of the elementwise math holds the values of NumPy. The
// root of the cube holds the correctly rounded value; the C library
// behind NumPy gives the value next to it on some inputs.
fn test_elementwise_rest() {
    // A half goes to the even neighbour under both names of NumPy.
    let ri, fri = scientific::rint(scientific::from_values([0.5, -0.5, 2.5, -2.5, 1.5]));
    assert(fri.kind == scientific::F_NONE);
    assert(ri.data[0] == 0.0 && ri.data[1] == 0.0 && ri.data[2] == 2.0);
    assert(ri.data[3] == -2.0 && ri.data[4] == 2.0);
    let fx, ffx = scientific::fix(scientific::from_values([-1.7, 1.7, -0.5, 2.9]));
    assert(ffx.kind == scientific::F_NONE);
    assert(fx.data[0] == -1.0 && fx.data[1] == 1.0 && fx.data[2] == 0.0 && fx.data[3] == 2.0);
    // A result of zero keeps the sign of the number, as NumPy keeps it.
    let zs, fzs = scientific::signbit(scientific::from_values([ri.data[1], fx.data[2]]));
    assert(fzs.kind == scientific::F_NONE && zs.data[0] == 1.0 && zs.data[1] == 1.0);
    // The remainder holds the sign of the divisor, or of the number.
    let ma, fma = scientific::from_values([7.0, -7.0, 5.5, -5.5, 1.0]);
    let mb, fmb = scientific::from_values([3.0, 3.0, -2.5, -2.5, 0.1]);
    let mo, fmo = scientific::mod(ma, mb);
    assert(fmo.kind == scientific::F_NONE);
    assert(common::close(mo.data[0], 1.0) && common::close(mo.data[1], 2.0));
    assert(common::close(mo.data[2], -2.0) && common::close(mo.data[3], -0.5));
    assert(mo.data[4] == 0.09999999999999995);
    let fm, ffm = scientific::fmod(ma, mb);
    assert(ffm.kind == scientific::F_NONE);
    assert(common::close(fm.data[0], 1.0) && common::close(fm.data[1], -1.0));
    assert(common::close(fm.data[2], 0.5) && common::close(fm.data[3], -0.5));
    assert(fm.data[4] == 0.09999999999999995);
    // A divisor of zero holds no remainder.
    let mz, fmz = scientific::mod(scientific::from_values([5.0]), scientific::from_values([0.0]));
    assert(fmz.kind == scientific::F_NONE && mz.data[0] != mz.data[0]);
    let fz, ffz = scientific::fmod(scientific::from_values([5.0]), scientific::from_values([0.0]));
    assert(ffz.kind == scientific::F_NONE && fz.data[0] != fz.data[0]);
    // Two shapes that hold no broadcast give a fault.
    let mbad, fmbad = scientific::mod(scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                                      scientific::from_values([1.0, 2.0]));
    assert(fmbad.kind == scientific::F_SHAPE && mbad.data.len == 0);
    // The root of the cube holds the sign of the number.
    let cb, fcb = scientific::cbrt(scientific::from_values([27.0, -27.0, 0.5, 8.0]));
    assert(fcb.kind == scientific::F_NONE);
    assert(cb.data[0] == 3.0 && cb.data[1] == -3.0);
    assert(cb.data[2] == 0.7937005259840998 && cb.data[3] == 2.0);
    // The length of two numbers takes no harm from a huge square.
    let ha, fha = scientific::from_values([3.0, 1.0e300, 0.0, 5.0, -3.0]);
    let hb, fhb = scientific::from_values([4.0, 1.0e300, 0.0, 12.0, -4.0]);
    let hy, fhy = scientific::hypot(ha, hb);
    assert(fhy.kind == scientific::F_NONE);
    assert(hy.data[0] == 5.0 && hy.data[1] == 1.4142135623730952e300);
    assert(hy.data[2] == 0.0 && hy.data[3] == 13.0 && hy.data[4] == 5.0);
    // Two to the power of the number.
    let e2, fe2 = scientific::exp2(scientific::from_values([0.5, 10.0, -1.0, 0.0]));
    assert(fe2.kind == scientific::F_NONE);
    assert(e2.data[0] == 1.4142135623730951 && e2.data[1] == 1024.0);
    assert(e2.data[2] == 0.5 && e2.data[3] == 1.0);
    // The next number of the machine toward the second number.
    let nx, fnx = scientific::from_values([1.0, 2.0, 0.0, 0.0]);
    let ny, fny = scientific::from_values([2.0, 1.0, 1.0, -1.0]);
    let na, fna = scientific::nextafter(nx, ny);
    assert(fna.kind == scientific::F_NONE);
    assert(na.data[0] == 1.0000000000000002 && na.data[1] == 1.9999999999999998);
    assert(na.data[2] == 4.9406564584124654e-324 && na.data[3] == -4.9406564584124654e-324);
    // A number equals its own neighbour.
    let ns, fns = scientific::nextafter(scientific::from_values([7.0]), scientific::from_values([7.0]));
    assert(fns.kind == scientific::F_NONE && ns.data[0] == 7.0);
    // The sign of the second number goes onto the first.
    let cs, fcs = scientific::copysign(scientific::from_values([3.0, -3.0, 3.0, 3.0]),
                                       scientific::from_values([-1.0, 1.0, 0.0, 0.0 * -1.0]));
    assert(fcs.kind == scientific::F_NONE);
    assert(cs.data[0] == -3.0 && cs.data[1] == 3.0);
    assert(cs.data[2] == 3.0 && cs.data[3] == -3.0);
    // The distance to the next number of the machine above.
    let sp, fsp = scientific::spacing(scientific::from_values([1.0, 0.0, -1.0, 2.0]));
    assert(fsp.kind == scientific::F_NONE);
    assert(sp.data[0] == 2.220446049250313e-16 && sp.data[1] == 4.9406564584124654e-324);
    assert(sp.data[2] == -2.220446049250313e-16 && sp.data[3] == 4.440892098500626e-16);
    // The infinity holds no spacing. The largest number holds an infinity.
    let inf = 1.0 / 0.0;
    let sp2, fsp2 = scientific::spacing(scientific::from_values([inf, 1.7976931348623157e308]));
    assert(fsp2.kind == scientific::F_NONE && sp2.data[0] != sp2.data[0] && sp2.data[1] > 1.0e308);
    // The step from the smallest normal number to zero is one subnormal step.
    let nz, fnz = scientific::nextafter(scientific::from_values([2.2250738585072014e-308, 1.0]),
                                        scientific::from_values([0.0, 0.0]));
    assert(fnz.kind == scientific::F_NONE);
    assert(nz.data[0] == 2.225073858507201e-308 && nz.data[1] == 0.9999999999999999);
    // A remainder of zero holds the sign of the divisor.
    let mz, fmz = scientific::mod(scientific::from_values([6.0, -6.0]), scientific::from_values([-3.0, 3.0]));
    let mzs, fmzs = scientific::signbit(mz);
    assert(fmz.kind == scientific::F_NONE && fmzs.kind == scientific::F_NONE);
    assert(mzs.data[0] == 1.0 && mzs.data[1] == 0.0);
}
