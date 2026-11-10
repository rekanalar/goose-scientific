import scientific from "..";
import os;

// close tells whether two numbers agree to a small tolerance.
fn mathshim_abs(x: f64) -> f64 {
    if x < 0.0 { return -x; }
    return x;
}

fn close(a: f64, b: f64) -> bool {
    let d = a - b;
    if d < 0.0 { return -d < 0.0000001; }
    return d < 0.0000001;
}

fn make23() -> scientific::Arr {
    let a, f = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert(f.kind == scientific::F_NONE);
    return a;
}

// The creation functions hold the shapes and the values they claim.
fn test_create() {
    let a, f = scientific::zeros([2, 3]);
    assert(f.kind == scientific::F_NONE);
    assert(a.ndim == 2 && a.shape[0] == 2 && a.shape[1] == 3 && a.data.len == 6);
    assert(a.data[4] == 0.0);
    let o, f2 = scientific::ones([4]);
    assert(f2.kind == scientific::F_NONE && o.data.len == 4 && o.data[3] == 1.0);
    let fu, f3 = scientific::full([2, 2], 7.5);
    assert(f3.kind == scientific::F_NONE && fu.data[2] == 7.5);
    let e, f4 = scientific::eye(3);
    assert(f4.kind == scientific::F_NONE);
    assert(e.data[0] == 1.0 && e.data[1] == 0.0 && e.data[4] == 1.0 && e.data[8] == 1.0);
    let dg, f5 = scientific::diag([2.0, 3.0]);
    assert(f5.kind == scientific::F_NONE);
    assert(dg.data[0] == 2.0 && dg.data[1] == 0.0 && dg.data[3] == 3.0);
    let ar, f6 = scientific::arange(0.0, 5.0, 1.0);
    assert(f6.kind == scientific::F_NONE);
    assert(ar.data.len == 5 && ar.data[4] == 4.0);
    let ar2, f7 = scientific::arange(5.0, 1.0, -2.0);
    assert(f7.kind == scientific::F_NONE);
    assert(ar2.data.len == 2 && ar2.data[1] == 3.0);
    let ls, f8 = scientific::linspace(0.0, 1.0, 5);
    assert(f8.kind == scientific::F_NONE);
    assert(ls.data.len == 5 && ls.data[0] == 0.0 && ls.data[4] == 1.0 && close(ls.data[2], 0.5));
    let ls1, f9 = scientific::linspace(2.0, 9.0, 1);
    assert(f9.kind == scientific::F_NONE && ls1.data.len == 1 && ls1.data[0] == 2.0);
    let fv, f10 = scientific::from_values([3.0, 1.0, 4.0]);
    assert(f10.kind == scientific::F_NONE && fv.ndim == 1 && fv.data[1] == 1.0);
    let c, f11 = scientific::clone(make23());
    assert(f11.kind == scientific::F_NONE && c.data[5] == 6.0);
    // A bad shape, a bad count and a bad step are faults.
    let b1, g1 = scientific::zeros([0, 2]);
    assert(g1.kind == scientific::F_VALUE);
    let b2, g2 = scientific::zeros([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert(g2.kind == scientific::F_LIMIT);
    let b3, g3 = scientific::from_rows(2, 3, [1.0, 2.0]);
    assert(g3.kind == scientific::F_SHAPE);
    let b4, g4 = scientific::arange(0.0, 1.0, 0.0);
    assert(g4.kind == scientific::F_VALUE);
    let b5, g5 = scientific::linspace(0.0, 1.0, 0);
    assert(g5.kind == scientific::F_VALUE);
    let b6, g6 = scientific::eye(0);
    assert(g6.kind == scientific::F_VALUE);
    // A range over the limit is a fault.
    let b7, g7 = scientific::arange(0.0, 400000000.0, 0.5);
    assert(g7.kind == scientific::F_LIMIT);
    // A fault result holds an empty array.
    assert(b7.ndim == 0 && b7.data.len == 0);
}

// The shape operations hold the elements and the order.
fn test_shape() {
    let a = make23();
    // Reshape keeps the row-major order.
    let r, f = scientific::reshape(a, [3, 2]);
    assert(f.kind == scientific::F_NONE);
    assert(r.shape[0] == 3 && r.shape[1] == 2);
    assert(r.data[0] == 1.0 && r.data[1] == 2.0 && r.data[2] == 3.0);
    // One open dimension takes the length that fits.
    let r2, f2 = scientific::reshape(a, [-1, 2]);
    assert(f2.kind == scientific::F_NONE && r2.shape[0] == 3);
    let r3, f3 = scientific::reshape(a, [-1, -1]);
    assert(f3.kind == scientific::F_SHAPE);
    let r4, f4 = scientific::reshape(a, [4, 2]);
    assert(f4.kind == scientific::F_SHAPE);
    // Transpose swaps the dimensions and the elements.
    let t, f5 = scientific::transpose(a);
    assert(f5.kind == scientific::F_NONE);
    assert(t.shape[0] == 3 && t.shape[1] == 2);
    assert(t.data[1] == 4.0 && t.data[2] == 2.0 && t.data[3] == 5.0);
    let one_d, f5b = scientific::from_values([1.0, 2.0]);
    let t1, f5c = scientific::transpose(one_d);
    assert(f5c.kind == scientific::F_SHAPE);
    // Ravel gives one dimension.
    let v, f6 = scientific::ravel(a);
    assert(f6.kind == scientific::F_NONE && v.ndim == 1 && v.data.len == 6);
    // Squeeze drops the length-one dimensions and keeps one.
    let one, f7 = scientific::ones([1, 3, 1]);
    let sq, f8 = scientific::squeeze(one);
    assert(f8.kind == scientific::F_NONE && sq.ndim == 1 && sq.shape[0] == 3);
    let all1, f8b = scientific::ones([1, 1]);
    let sq2, f8c = scientific::squeeze(all1);
    assert(f8c.kind == scientific::F_NONE && sq2.ndim == 1 && sq2.shape[0] == 1);
    // expand_dims puts a length-one dimension at the axis.
    let ex, f9 = scientific::expand_dims(a, 1);
    assert(f9.kind == scientific::F_NONE && ex.ndim == 3 && ex.shape[0] == 2 && ex.shape[1] == 1 && ex.shape[2] == 3);
    let ex2, f9b = scientific::expand_dims(a, 3);
    assert(f9b.kind == scientific::F_RANGE);
    // concatenate joins along an axis.
    let b = make23();
    let cc, f10 = scientific::concatenate(0, a, b);
    assert(f10.kind == scientific::F_NONE && cc.shape[0] == 4 && cc.shape[1] == 3);
    assert(cc.data[3] == 4.0 && cc.data[6] == 1.0);
    let cc2, f11 = scientific::concatenate(1, a, b);
    assert(f11.kind == scientific::F_NONE && cc2.shape[0] == 2 && cc2.shape[1] == 6);
    assert(cc2.data[3] == 1.0 && cc2.data[4] == 2.0 && cc2.data[9] == 4.0);
    let cc3, f12 = scientific::concatenate(2, a, b);
    assert(f12.kind == scientific::F_RANGE);
    // stack puts a new first dimension.
    let st, f13 = scientific::stack(a, b);
    assert(f13.kind == scientific::F_NONE && st.ndim == 3 && st.shape[0] == 2);
    assert(st.data[0] == 1.0 && st.data[6] == 1.0);
    // index reads one element with its parts checked.
    let x, f14 = scientific::index(a, [1, 2]);
    assert(f14.kind == scientific::F_NONE && x == 6.0);
    let y, f15 = scientific::index(a, [2, 0]);
    assert(f15.kind == scientific::F_RANGE);
    let z, f16 = scientific::index(a, [1]);
    assert(f16.kind == scientific::F_RANGE);
    // broadcast_shapes holds the rule of NumPy.
    var bs: i64[>..] = [];
    let sh, f17 = scientific::broadcast_shapes([2, 1, 3], [4, 3]);
    assert(f17.kind == scientific::F_NONE && sh.len == 3 && sh[0] == 2 && sh[1] == 4 && sh[2] == 3);
    let sh2, f18 = scientific::broadcast_shapes([5, 3], [2, 3]);
    assert(f18.kind == scientific::F_SHAPE);
}

// The elementwise operations broadcast and hold the IEEE rules.
fn test_ops() {
    let a = make23();
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
    assert(f11.kind == scientific::F_NONE && close(rt.data[1], 4.47213595));
    let ex, f12 = scientific::exp(row);
    assert(f12.kind == scientific::F_NONE && close(ex.data[0], 22026.465794807));
    let lg, f13 = scientific::log(row);
    assert(f13.kind == scientific::F_NONE && close(lg.data[2], 3.40119738));
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
    let a = make23();
    let s, f = scientific::sum(a);
    assert(f.kind == scientific::F_NONE && s == 21.0);
    let p, f2 = scientific::prod(a);
    assert(f2.kind == scientific::F_NONE && p == 720.0);
    let mn, f3 = scientific::min(a);
    assert(f3.kind == scientific::F_NONE && mn == 1.0);
    let mx, f4 = scientific::max(a);
    assert(f4.kind == scientific::F_NONE && mx == 6.0);
    let m, f5 = scientific::mean(a);
    assert(f5.kind == scientific::F_NONE && close(m, 3.5));
}


// The reductions along one axis, the flat positions of the extremes
// and the permutation of the dimensions.
fn test_axis_and_permute() {
    let a = make23();
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
    assert(f6.kind == scientific::F_NONE && close(mc.data[1], 5.0));
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
    assert(close(l2.data[1], 2.0) && close(l2.data[2], 3.0));
    let l10, f9 = scientific::log10(p4);
    assert(close(l10.data[0], 0.0));
    let two, f10 = scientific::full([1], 2.0);
    let pw, f11 = scientific::power(p4, two);
    assert(f11.kind == scientific::F_NONE && pw.data[2] == 64.0);
    // Comparisons give 1.0 and 0.0 and broadcast.
    let a = make23();
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
    assert(f44.kind == scientific::F_NONE && close(va, 4.0));
    let sd, f45 = scientific::std(sv, 0);
    assert(close(sd, 2.0));
    let va1, f46 = scientific::variance(sv, 1);
    assert(close(va1, 32.0 / 7.0));
    let vb, f47 = scientific::variance(sv, 8);
    assert(f47.kind == scientific::F_VALUE);
    let vx, f48 = scientific::var_axis(a, 1, 0);
    assert(f48.kind == scientific::F_NONE && close(vx.data[0], 2.0 / 3.0) && close(vx.data[1], 2.0 / 3.0));
    let sx, f49 = scientific::std_axis(a, 0, 0);
    assert(close(sx.data[0], 1.5));
    let sp, f50 = scientific::ptp(a);
    assert(sp == 5.0);
}

// The slicing, the masks and the selection by position.
fn test_indexing() {
    let a = make23();
    // A slice keeps the elements between the bounds.
    let sl, f = scientific::slice_axis(a, 1, 1, 3, 1);
    assert(f.kind == scientific::F_NONE && sl.shape[1] == 2);
    assert(sl.data[0] == 2.0 && sl.data[1] == 3.0 && sl.data[2] == 5.0);
    // A slice of the rows.
    let sl2, f2 = scientific::slice_axis(a, 0, 0, 1, 1);
    assert(f2.kind == scientific::F_NONE && sl2.shape[0] == 1 && sl2.data[2] == 3.0);
    // A step of two keeps every second element.
    let sl3, f3 = scientific::slice_axis(a, 1, 0, 3, 2);
    assert(f3.kind == scientific::F_NONE && sl3.shape[1] == 2 && sl3.data[0] == 1.0 && sl3.data[1] == 3.0 && sl3.data[2] == 4.0);
    // A negative step reads the axis backwards.
    let sl4, f4 = scientific::slice_axis(a, 1, -1, -4, -1);
    assert(f4.kind == scientific::F_NONE && sl4.shape[1] == 3);
    assert(sl4.data[0] == 3.0 && sl4.data[1] == 2.0 && sl4.data[2] == 1.0);
    // A slice that holds no element is a fault.
    let bad, f5 = scientific::slice_axis(a, 1, 2, 2, 1);
    assert(f5.kind == scientific::F_SHAPE && bad.data.len == 0);
    let bad2, f6 = scientific::slice_axis(a, 1, 0, 3, 0);
    assert(f6.kind == scientific::F_VALUE);
    let bad3, f7 = scientific::slice_axis(a, 5, 0, 1, 1);
    assert(f7.kind == scientific::F_RANGE);
    // take gives the elements at the flat positions.
    let tk, f8 = scientific::take(a, [0, 5, 2]);
    assert(f8.kind == scientific::F_NONE && tk.data[0] == 1.0 && tk.data[1] == 6.0 && tk.data[2] == 3.0);
    let tk2, f9 = scientific::take(a, [6]);
    assert(f9.kind == scientific::F_RANGE && tk2.data.len == 0);
    // put writes into the elements at the flat positions.
    let b = make23();
    let pf = scientific::put(b, [0, 5], [9.0, 8.0]);
    assert(pf.kind == scientific::F_NONE && b.data[0] == 9.0 && b.data[5] == 8.0 && b.data[1] == 2.0);
    let b2 = make23();
    let pf2 = scientific::put(b2, [9], [1.0]);
    assert(pf2.kind == scientific::F_RANGE && b2.data[0] == 1.0);
    let b3 = make23();
    let pf3 = scientific::put(b3, [0, 1], [1.0]);
    assert(pf3.kind == scientific::F_SHAPE && b3.data[0] == 1.0);
    // nonzero gives the flat positions of the values that are not zero.
    let z, f10 = scientific::from_values([0.0, 2.0, 0.0, 4.0]);
    let nz, f11 = scientific::nonzero(z);
    assert(f11.kind == scientific::F_NONE && nz.data[0] == 1.0 && nz.data[1] == 3.0 && nz.data.len == 2);
    // a mask keeps the elements that hold true.
    let m, f12 = scientific::mask(z, z);
    assert(f12.kind == scientific::F_NONE && m.data[0] == 2.0 && m.data[1] == 4.0);
    let m2, f13 = scientific::mask(z, z);
    assert(m2.data.len == 2);
    let wrong, f14 = scientific::from_values([1.0]);
    let m3, f15 = scientific::mask(z, wrong);
    assert(f15.kind == scientific::F_SHAPE);
    // take_along_axis takes the elements at the positions of an axis.
    let idx, f16 = scientific::from_rows(2, 1, [2.0, 0.0]);
    let ta, f17 = scientific::take_along_axis(a, idx, 1);
    assert(f17.kind == scientific::F_NONE && ta.data[0] == 3.0 && ta.data[1] == 4.0);
    let badidx, f18 = scientific::from_rows(2, 1, [5.0, 0.0]);
    let ta2, f19 = scientific::take_along_axis(a, badidx, 1);
    assert(f19.kind == scientific::F_RANGE);
    // repeat repeats every element.
    let rp, f20 = scientific::repeat(a, 2);
    assert(f20.kind == scientific::F_NONE && rp.data.len == 12 && rp.data[0] == 1.0 && rp.data[1] == 1.0 && rp.data[2] == 2.0);
    let rp2, f21 = scientific::repeat(a, 0);
    assert(f21.kind == scientific::F_VALUE);
    // tile repeats the whole array along one axis.
    let tl, f22 = scientific::tile(a, 2, 0);
    assert(f22.kind == scientific::F_NONE && tl.shape[0] == 4 && tl.shape[1] == 3);
    assert(tl.data[0] == 1.0 && tl.data[3] == 4.0 && tl.data[6] == 1.0);
    let tl2, f23 = scientific::tile(a, 2, 1);
    assert(f23.kind == scientific::F_NONE && tl2.shape[1] == 6 && tl2.data[3] == 1.0);
    // flatten gives a copy in one dimension.
    let fnn2, f25 = scientific::flatten(a);
    assert(f25.kind == scientific::F_NONE && fnn2.data[5] == 6.0);
    // partition puts the values below the pivot first.
    let pv, f26 = scientific::from_values([5.0, 1.0, 4.0, 2.0]);
    let pt, f27 = scientific::partition(pv, 3.0);
    assert(f27.kind == scientific::F_NONE && pt.data[0] == 1.0 && pt.data[1] == 2.0 && pt.data[2] == 5.0 && pt.data[3] == 4.0);
}

// The sorting, the search and the order statistics.
fn test_sort_and_search() {
    let v, f = scientific::from_values([3.0, 1.0, 2.0, 1.0]);
    // The sort gives a copy in the order of the values.
    let so, f1 = scientific::sort(v);
    assert(f1.kind == scientific::F_NONE && so.data[0] == 1.0 && so.data[1] == 1.0 && so.data[2] == 2.0 && so.data[3] == 3.0);
    assert(v.data[0] == 3.0);
    // argsort gives the flat positions that order the values.
    let ar, f2 = scientific::argsort(v);
    assert(f2.kind == scientific::F_NONE && ar.data[0] == 1.0 && ar.data[1] == 3.0 && ar.data[2] == 2.0 && ar.data[3] == 0.0);
    // A sort along one axis.
    let a = make23();
    let sa, f3 = scientific::sort_axis(a, 1);
    assert(f3.kind == scientific::F_NONE && sa.data[0] == 1.0 && sa.data[1] == 2.0 && sa.data[2] == 3.0);
    assert(sa.data[3] == 4.0 && sa.data[5] == 6.0);
    let sb, f4 = scientific::sort_axis(a, 5);
    assert(f4.kind == scientific::F_RANGE);
    // unique gives the values in the order of their first appearance.
    let u, f5 = scientific::unique(v);
    assert(f5.kind == scientific::F_NONE && u.data[0] == 3.0 && u.data[1] == 1.0 && u.data[2] == 2.0);
    let us, f6 = scientific::unique_sorted(v);
    assert(f6.kind == scientific::F_NONE && us.data[0] == 1.0 && us.data[1] == 2.0 && us.data[2] == 3.0);
    // searchsorted gives the place where a value fits.
    let ss, f7 = scientific::searchsorted(so, [0.5, 1.0, 2.5, 9.0]);
    assert(f7.kind == scientific::F_NONE && ss.data[0] == 0.0 && ss.data[1] == 0.0 && ss.data[2] == 3.0 && ss.data[3] == 4.0);
    // The histogram counts the values in equal bins.
    let h, e, f8 = scientific::histogram(v, 2);
    assert(f8.kind == scientific::F_NONE && h.data.len == 2 && h.data[0] == 2.0 && h.data[1] == 2.0);
    assert(e.data.len == 3 && e.data[0] == 1.0 && e.data[2] == 3.0);
    let he, f9 = scientific::histogram_edges(v, 2);
    assert(f9.kind == scientific::F_NONE && he.data.len == 3);
    let hb, he2, f10 = scientific::histogram(v, 0);
    assert(f10.kind == scientific::F_VALUE);
    let one, f11 = scientific::from_values([2.0, 2.0]);
    let hc, he3, f12 = scientific::histogram(one, 2);
    assert(f12.kind == scientific::F_VALUE);
    // bincount counts the whole numbers.
    let w, f13 = scientific::from_values([0.0, 1.0, 1.0, 3.0]);
    let bc, f14 = scientific::bincount(w, 0);
    assert(f14.kind == scientific::F_NONE && bc.data[0] == 1.0 && bc.data[1] == 2.0 && bc.data[2] == 0.0 && bc.data[3] == 1.0);
    let bd, f15 = scientific::from_values([1.5]);
    let bc2, f16 = scientific::bincount(bd, 0);
    assert(f16.kind == scientific::F_VALUE);
    // digitize gives the bin of each number.
    let dg, f17 = scientific::digitize([0.5, 1.5, 2.5], [1.0, 2.0]);
    assert(f17.kind == scientific::F_NONE && dg.data[0] == 0.0 && dg.data[1] == 1.0 && dg.data[2] == 2.0);
    let dgb, f18 = scientific::digitize([1.0], [2.0, 1.0]);
    assert(f18.kind == scientific::F_VALUE);
    // The median and the percentiles.
    let md, f19 = scientific::median(v);
    assert(f19.kind == scientific::F_NONE && md == 1.5);
    let md2, f20 = scientific::median(so);
    assert(md2 == 1.5);
    let pc, f21 = scientific::percentile(v, 50.0);
    assert(close(pc, 1.5));
    let pc2, f22 = scientific::percentile(v, 0.0);
    assert(pc2 == 1.0);
    let pc3, f23 = scientific::percentile(v, 100.0);
    assert(pc3 == 3.0);
    let pb, f24 = scientific::percentile(v, 101.0);
    assert(f24.kind == scientific::F_VALUE);
    // roll moves the elements along one axis.
    let rl, f25 = scientific::roll(v, 1, 0);
    assert(f25.kind == scientific::F_NONE && rl.data[0] == 1.0 && rl.data[1] == 3.0 && rl.data[3] == 2.0);
    let rl2, f26 = scientific::roll(v, -1, 0);
    assert(rl2.data[0] == 1.0 && rl2.data[1] == 2.0 && rl2.data[3] == 3.0);
    let rb, f27 = scientific::roll(v, 1, 4);
    assert(f27.kind == scientific::F_RANGE);
}

// The products and the solving of a system.
fn test_linalg() {
    let a = make23();
    // matmul multiplies two matrices.
    let b, f0 = scientific::from_rows(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let p, f = scientific::matmul(a, b);
    assert(f.kind == scientific::F_NONE && p.shape[0] == 2 && p.shape[1] == 2);
    assert(p.data[0] == 22.0 && p.data[1] == 28.0 && p.data[2] == 49.0 && p.data[3] == 64.0);
    // The inner dimensions must agree.
    let bad, f2 = scientific::matmul(a, a);
    assert(f2.kind == scientific::F_SHAPE);
    let row, f3 = scientific::from_values([1.0, 2.0, 3.0]);
    let bad2, f4 = scientific::matmul(a, row);
    assert(f4.kind == scientific::F_SHAPE);
    // dot of two rows gives a number.
    let v, f5 = scientific::from_values([1.0, 2.0, 3.0]);
    let w, f6 = scientific::from_values([4.0, 5.0, 6.0]);
    let dt, f7 = scientific::dot(v, w);
    assert(f7.kind == scientific::F_NONE && dt.ndim == 0 && dt.data[0] == 32.0);
    // dot of a row and a matrix gives a row.
    let u, f8a = scientific::from_values([1.0, 2.0]);
    let dr, f8 = scientific::dot(u, a);
    assert(f8.kind == scientific::F_NONE && dr.data.len == 3 && dr.data[0] == 9.0 && dr.data[1] == 12.0 && dr.data[2] == 15.0);
    // dot of a matrix and a row gives a column.
    let dc, f9 = scientific::dot(a, u);
    assert(f9.kind == scientific::F_NONE && dc.data.len == 3 && dc.data[0] == 9.0 && dc.data[1] == 12.0 && dc.data[2] == 15.0);
    // The lengths must agree.
    let dw, f9a = scientific::dot(v, w);
    assert(f9a.kind == scientific::F_NONE && dw.data[0] == 32.0);
    // inner and outer.
    let inn, f10 = scientific::inner(v, v);
    assert(f10.kind == scientific::F_NONE && inn.data[0] == 14.0);
    let ou, f11 = scientific::outer(v, w);
    assert(f11.kind == scientific::F_NONE && ou.shape[0] == 3 && ou.shape[1] == 3);
    assert(ou.data[0] == 4.0 && ou.data[4] == 10.0 && ou.data[8] == 18.0);
    // kron gives the Kronecker product.
    let k1, f12 = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let k2, f13 = scientific::from_rows(1, 2, [5.0, 6.0]);
    let kr, f14 = scientific::kron(k1, k2);
    assert(f14.kind == scientific::F_NONE && kr.shape[0] == 2 && kr.shape[1] == 4);
    assert(kr.data[0] == 5.0 && kr.data[1] == 6.0 && kr.data[4] == 15.0 && kr.data[5] == 18.0);
    // trace and norm.
    let tr, f15 = scientific::trace(k1);
    assert(f15.kind == scientific::F_NONE && tr == 5.0);
    let tr2, f16 = scientific::trace(a);
    assert(f16.kind == scientific::F_SHAPE);
    let nm, f17 = scientific::norm(k1);
    assert(f17.kind == scientific::F_NONE && close(nm, 5.47722557));
    let nm2, f18 = scientific::norm(v);
    assert(f18.kind == scientific::F_NONE && close(nm2, 3.74165738));
    // solve gives the vector that solves the system.
    let m, f19 = scientific::from_rows(2, 2, [2.0, 1.0, 1.0, 3.0]);
    let rhs, f20 = scientific::from_values([3.0, 4.0]);
    let sol, f21 = scientific::solve(m, rhs);
    assert(f21.kind == scientific::F_NONE && close(sol.data[0], 1.0) && close(sol.data[1], 1.0));
    // A matrix without a pivot is a fault.
    let sing, f22 = scientific::from_rows(2, 2, [1.0, 2.0, 2.0, 4.0]);
    let bad3, f23 = scientific::solve(sing, rhs);
    assert(f23.kind == scientific::F_VALUE);
    let bad4, f24 = scientific::solve(a, rhs);
    assert(f24.kind == scientific::F_SHAPE);
    // inv gives the inverse.
    let iv, f25 = scientific::inv(m);
    assert(f25.kind == scientific::F_NONE && close(iv.data[0], 0.6) && close(iv.data[1], -0.2));
    assert(close(iv.data[2], -0.2) && close(iv.data[3], 0.4));
    // det gives the determinant.
    let dt2, f26 = scientific::det(m);
    assert(f26.kind == scientific::F_NONE && close(dt2, 5.0));
    let dt3, f27 = scientific::det(sing);
    assert(f27.kind == scientific::F_VALUE);
    // slogdet gives the sign and the logarithm.
    let sg, lg, f28 = scientific::slogdet(m);
    assert(f28.kind == scientific::F_NONE && sg == 1.0 && close(lg, 1.60943791));
    let sg2, lg2, f29 = scientific::slogdet(sing);
    assert(f29.kind == scientific::F_NONE && sg2 == 0.0 && lg2 == 0.0);
    // matrix_power raises a matrix to a power.
    let pw, f30 = scientific::matrix_power(m, 2);
    assert(f30.kind == scientific::F_NONE && close(pw.data[0], 5.0) && close(pw.data[1], 5.0));
    assert(close(pw.data[2], 5.0) && close(pw.data[3], 10.0));
    let pw0, f31 = scientific::matrix_power(m, 0);
    assert(f31.kind == scientific::F_NONE && pw0.data[0] == 1.0 && pw0.data[1] == 0.0 && pw0.data[3] == 1.0);
    let pwn, f33 = scientific::matrix_power(m, -1);
    assert(f33.kind == scientific::F_NONE && close(pwn.data[0], 0.6));
}

// The Cholesky, the QR and the eigenvalues.
fn test_decomp_and_eig() {
    // The Cholesky factor of a symmetric matrix.
    let m, f = scientific::from_rows(2, 2, [4.0, 2.0, 2.0, 3.0]);
    let ch, f1 = scientific::cholesky(m);
    assert(f1.kind == scientific::F_NONE);
    assert(close(ch.data[0], 2.0) && close(ch.data[2], 1.0) && close(ch.data[3], 1.41421356));
    // A matrix with a negative value holds no Cholesky factor.
    let neg, f2 = scientific::from_rows(2, 2, [1.0, 2.0, 2.0, 1.0]);
    let bad, f3 = scientific::cholesky(neg);
    assert(f3.kind == scientific::F_VALUE && bad.data.len == 0);
    let ns, f4 = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let bad2, f5 = scientific::cholesky(ns);
    assert(f5.kind == scientific::F_SHAPE);
    // The QR decomposition gives an orthogonal matrix and an upper
    // triangle. The product of the two gives the matrix.
    let q, r, f6 = scientific::qr(ns);
    assert(f6.kind == scientific::F_NONE && q.shape[0] == 2 && q.shape[1] == 2 && r.shape[0] == 2 && r.shape[1] == 3);
    let qr_prod, f7 = scientific::matmul(q, r);
    assert(f7.kind == scientific::F_NONE);
    assert(close(qr_prod.data[0], 1.0) && close(qr_prod.data[2], 3.0) && close(qr_prod.data[3], 4.0) && close(qr_prod.data[5], 6.0));
    let qn, rn, f8 = scientific::qr(m);
    assert(f8.kind == scientific::F_NONE);
    // The eigenvalues of a symmetric matrix come in the order of their
    // value.
    let sym, f9 = scientific::from_rows(2, 2, [2.0, 1.0, 1.0, 2.0]);
    let ev, f10 = scientific::eigvalsh(sym);
    assert(f10.kind == scientific::F_NONE && close(ev.data[0], 1.0) && close(ev.data[1], 3.0));
    // The eigenvectors are the columns of the result.
    let vals, vecs, f11 = scientific::eigh(sym);
    assert(f11.kind == scientific::F_NONE && close(vals.data[0], 1.0) && close(vals.data[1], 3.0));
    assert(close(mathshim_abs(vecs.data[1]), 0.70710678));
    // The eigenvalues of a matrix that is not symmetric.
    let gen, f12 = scientific::from_rows(2, 2, [4.0, 1.0, 2.0, 3.0]);
    let gev, f13 = scientific::eigvals(gen);
    assert(f13.kind == scientific::F_NONE && gev.data.len == 2);
    let sum = gev.data[0] + gev.data[1];
    assert(close(sum, 7.0));
    // The vector of an eigenvalue holds the matrix near zero.
    let gvals, gvecs, f12b = scientific::eig(gen);
    assert(f12b.kind == scientific::F_NONE);
    let g0 = gvecs.data[0];
    let g1 = gvecs.data[2];
    let av0 = gen.data[0] * g0 + gen.data[1] * g1;
    let av1 = gen.data[2] * g0 + gen.data[3] * g1;
    assert(close(av0, gvals.data[0] * g0) && close(av1, gvals.data[0] * g1));
    // A matrix with a complex pair holds no real eigenvalues.
    let rot, f14 = scientific::from_rows(2, 2, [0.0, -1.0, 1.0, 0.0]);
    let bad3, f15 = scientific::eigvals(rot);
    assert(f15.kind == scientific::F_VALUE);
    // A matrix that is not square is a fault.
    let bad4, f16 = scientific::eigvalsh(ns);
    assert(f16.kind == scientific::F_SHAPE);
    let bad5, bad6, f17 = scientific::eigh(ns);
    assert(f17.kind == scientific::F_SHAPE);
}

// The singular values, the least squares and the polynomials.
fn test_svd_and_poly() {
    // The singular values of a matrix come in the order of their size.
    let m, f = scientific::from_rows(2, 2, [4.0, 3.0, 6.0, 3.0]);
    let sv, f1 = scientific::svdvals(m);
    assert(f1.kind == scientific::F_NONE && sv.data.len == 2);
    assert(close(sv.data[0], 8.33557912) && close(sv.data[1], 0.71980602));
    // The product of the left vectors, the values and the right
    // vectors gives the matrix.
    let u, s, vt, f2 = scientific::svd(m);
    assert(f2.kind == scientific::F_NONE && u.shape[0] == 2 && u.shape[1] == 2);
    assert(vt.shape[0] == 2 && vt.shape[1] == 2);
    let prod, f3 = scientific::matmul(u, vt);
    assert(f3.kind == scientific::F_NONE);
    // The product scales the columns of the right vectors.
    let scaled, f4 = scientific::scale(vt, s.data[0]);
    assert(f4.kind == scientific::F_NONE);
    // The left vectors, the values and the right vectors give the
    // matrix back.
    let r00 = u.data[0] * s.data[0] * vt.data[0] + u.data[1] * s.data[1] * vt.data[2];
    let r01 = u.data[0] * s.data[0] * vt.data[1] + u.data[1] * s.data[1] * vt.data[3];
    let r10 = u.data[2] * s.data[0] * vt.data[0] + u.data[3] * s.data[1] * vt.data[2];
    let r11 = u.data[2] * s.data[0] * vt.data[1] + u.data[3] * s.data[1] * vt.data[3];
    assert(close(r00, 4.0) && close(r01, 3.0) && close(r10, 6.0) && close(r11, 3.0));
    // A matrix with fewer rows than columns holds a value for each
    // column.
    let wide, f5 = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let wsv, f6 = scientific::svdvals(wide);
    assert(f6.kind == scientific::F_NONE && wsv.data.len == 2);
    // The least squares fits the points.
    let pts, f7 = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 1.0]);
    let rhs, f8 = scientific::from_values([1.0, 2.0]);
    let sol, f9 = scientific::lstsq(pts, rhs);
    assert(f9.kind == scientific::F_NONE && sol.data.len == 2);
    assert(close(sol.data[0], 0.6) && close(sol.data[1], 0.2));
    // A system with more points than unknowns fits the points.
    let many, f7b = scientific::from_rows(3, 2, [1.0, 1.0, 2.0, 1.0, 3.0, 1.0]);
    let many_rhs, f8b = scientific::from_values([2.0, 3.0, 4.0]);
    let many_sol, f9b = scientific::lstsq(many, many_rhs);
    assert(f9b.kind == scientific::F_NONE && close(many_sol.data[0], 1.0) && close(many_sol.data[1], 1.0));
    // The rank and the condition of a matrix.
    let rk, f10 = scientific::matrix_rank(m);
    assert(f10.kind == scientific::F_NONE && rk == 2);
    let sing, f11 = scientific::from_rows(2, 2, [1.0, 2.0, 2.0, 4.0]);
    let rk2, f12 = scientific::matrix_rank(sing);
    assert(f12.kind == scientific::F_NONE && rk2 == 1);
    let cd, f13 = scientific::cond(m);
    assert(f13.kind == scientific::F_NONE && close(cd, 8.33557912 / 0.71980602));
    // The value of a polynomial.
    let pv = scientific::polyval([1.0, 0.0, -1.0], 2.0);
    assert(close(pv, 3.0));
    // The polynomial that fits the points.
    let xs, f14 = scientific::from_values([0.0, 1.0, 2.0]);
    let ys, f15 = scientific::from_values([1.0, 3.0, 5.0]);
    let fit, f16 = scientific::polyfit(xs.data, ys.data, 1);
    assert(f16.kind == scientific::F_NONE && close(fit.data[0], 2.0) && close(fit.data[1], 1.0));
    // The roots of a polynomial. The parts come in two arrays.
    let rt, rti, f17 = scientific::roots([1.0, -3.0, 2.0]);
    assert(f17.kind == scientific::F_NONE && rt.data.len == 2);
    assert(close(rt.data[0], 1.0) || close(rt.data[0], 2.0));
    assert(close(rt.data[1], 1.0) || close(rt.data[1], 2.0));
    // A polynomial with no term is a fault.
    let bad, badi, f18 = scientific::roots([0.0, 0.0]);
    assert(f18.kind == scientific::F_VALUE);
    // The roots of a polynomial with complex roots. The values come
    // from numpy.roots.
    let cr, cri, f20 = scientific::roots([1.0, 0.0, 1.0]);
    assert(f20.kind == scientific::F_NONE && cr.data.len == 2);
    assert(close(cr.data[0], 0.0) && close(cr.data[1], 0.0));
    assert((close(cri.data[0], 1.0) && close(cri.data[1], -1.0)) || (close(cri.data[0], -1.0) && close(cri.data[1], 1.0)));
    let cr2, cri2, f21 = scientific::roots([1.0, 0.0, 0.0, -1.0]);
    assert(f21.kind == scientific::F_NONE && cr2.data.len == 3);
    // The roots come in no fixed order.
    var one: i64 = -1;
    var k: i64 = 0;
    while k < 3 {
        if close(cr2.data[k], 1.0) { one = k; }
        k += 1;
    }
    assert(one >= 0 && close(cri2.data[one], 0.0));
    var j1: i64 = 0;
    while j1 == one { j1 += 1; }
    var j2 = j1 + 1;
    while j2 == one { j2 += 1; }
    assert(close(cr2.data[j1], -0.5) && close(cr2.data[j2], -0.5));
    assert((close(cri2.data[j1], 0.8660254037844386) && close(cri2.data[j2], -0.8660254037844386)) || (close(cri2.data[j1], -0.8660254037844386) && close(cri2.data[j2], 0.8660254037844386)));
    let cr3, cri3, f22 = scientific::roots([2.0, 3.0, 5.0]);
    assert(f22.kind == scientific::F_NONE && cr3.data.len == 2);
    assert(close(cr3.data[0], -0.75) && close(cr3.data[1], -0.75));
    assert((close(cri3.data[0], 1.3919410907075054) && close(cri3.data[1], -1.3919410907075054)) || (close(cri3.data[0], -1.3919410907075054) && close(cri3.data[1], 1.3919410907075054)));
    // A degree above the limit is a fault, before any work starts.
    var big: f64[>..] = [];
    var bi: i64 = 0;
    while bi < 300 { big.push(1.0); bi += 1; }
    let bigr, bigi, f23 = scientific::roots(big);
    assert(f23.kind == scientific::F_LIMIT);
    // A degree below zero is a fault.
    let bad2, f19 = scientific::polyfit(xs.data, ys.data, -1);
    assert(f19.kind == scientific::F_VALUE);
}

// The discrete Fourier transform.
fn test_fft() {
    // The transform of a signal that holds one value.
    let re, im, f = scientific::fft([1.0, 0.0, 0.0, 0.0]);
    assert(f.kind == scientific::F_NONE && re.data.len == 4);
    assert(re.data[0] == 1.0 && re.data[1] == 1.0 && im.data[0] == 0.0);
    // The transform of a signal of four values.
    let re2, im2, f2 = scientific::fft([1.0, 2.0, 3.0, 4.0]);
    assert(f2.kind == scientific::F_NONE);
    assert(close(re2.data[0], 10.0) && close(re2.data[1], -2.0) && close(im2.data[1], 2.0));
    assert(close(re2.data[2], -2.0) && close(re2.data[3], -2.0) && close(im2.data[3], -2.0));
    // A length that holds a factor that is not a power of two.
    let re6, im6, f3 = scientific::fft([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert(f3.kind == scientific::F_NONE && re6.data.len == 6);
    assert(close(re6.data[0], 21.0) && close(re6.data[1], -3.0) && close(im6.data[1], 5.19615242));
    assert(close(re6.data[2], -3.0) && close(im6.data[2], 1.73205081));
    assert(close(re6.data[3], -3.0) && close(im6.data[5], -5.19615242));
    // The signal comes back from its transform.
    let back, back_im, f4 = scientific::ifft(re2.data, im2.data);
    assert(f4.kind == scientific::F_NONE);
    assert(close(back.data[0], 1.0) && close(back.data[3], 4.0));
    let back6, back6_im, f5 = scientific::ifft(re6.data, im6.data);
    assert(f5.kind == scientific::F_NONE);
    assert(close(back6.data[0], 1.0) && close(back6.data[5], 6.0));
    // A signal that holds no element is a fault.
    let bad, bad_im, f6 = scientific::fft([]);
    assert(f6.kind == scientific::F_VALUE);
    // The frequencies of a transform.
    let fr, f7 = scientific::fftfreq(4, 1.0);
    assert(f7.kind == scientific::F_NONE);
    assert(close(fr.data[0], 0.0) && close(fr.data[1], 0.25) && close(fr.data[2], -0.5) && close(fr.data[3], -0.25));
    let fr2, f8 = scientific::fftfreq(4, 0.5);
    assert(close(fr2.data[1], 0.5));
    // The shift moves the zero frequency to the middle.
    let sh, f9 = scientific::fftshift(fr);
    assert(f9.kind == scientific::F_NONE);
    assert(close(sh.data[0], -0.5) && close(sh.data[1], -0.25) && close(sh.data[2], 0.0));
    // The transform of a matrix.
    let m = make23();
    let mre, mim, f10 = scientific::fft2(m);
    assert(f10.kind == scientific::F_NONE && mre.shape[0] == 2 && mre.shape[1] == 3);
    assert(close(mre.data[0], 21.0) && close(mre.data[3], -9.0) && close(mim.data[1], 1.73205081));
    // The transform with only the positive frequencies.
    let rre, rim, f11 = scientific::rfft([1.0, 2.0, 3.0, 4.0]);
    assert(f11.kind == scientific::F_NONE && rre.data.len == 3);
    assert(close(rre.data[0], 10.0) && close(rre.data[1], -2.0) && close(rim.data[1], 2.0));
    let rback, f12 = scientific::irfft(rre.data, rim.data, 4);
    assert(f12.kind == scientific::F_NONE);
    assert(close(rback.data[0], 1.0) && close(rback.data[3], 4.0));
    // A sine of three turns over eight places peaks at the place of
    // its frequency, with the sign of NumPy.
    var sine: f64[>..] = [];
    var i9: i64 = 0;
    while i9 < 8 {
        sine.push(mathshim::sine(6.283185307179586 * 3.0 * i9 / 8.0));
        i9 += 1;
    }
    let sre, sim, f13 = scientific::fft(sine);
    assert(f13.kind == scientific::F_NONE);
    assert(close(sre.data[3], 0.0) && close(sim.data[3], -4.0));
    // The peak stands in the positive half. The negative half holds
    // the mirror of it.
    var peak: i64 = 0;
    var best: f64 = 0.0;
    i9 = 1;
    while i9 < 4 {
        let mag = sre.data[i9] * sre.data[i9] + sim.data[i9] * sim.data[i9];
        if mag > best { best = mag; peak = i9; }
        i9 += 1;
    }
    assert(peak == 3 && best > 15.9 && close(sre.data[0], 0.0));
    // The transform of a count of dimensions that is not two is a
    // fault.
    let bad2, bad2_im, f14 = scientific::fft2(fr);
    assert(f14.kind == scientific::F_SHAPE);
    // Parts that hold a different count are a fault.
    let bad3, f15 = scientific::irfft(rre.data, [1.0], 4);
    assert(f15.kind == scientific::F_SHAPE);
}

fn test_random() {
    // The numbers of one seed are pinned. The same seed gives the
    // same numbers on every run.
    var r = scientific::default_rng(42);
    assert(scientific::uniform(r, 2.0, 5.0) == 4.22469463631547);
    assert(scientific::random_uniform(r) == 0.1599103928769201);
    let a, fa = scientific::integers(r, 1, 7);
    assert(fa.kind == scientific::F_NONE && a == 1);
    // The pair of Box and Muller gives one number and holds one.
    let n1, f1 = scientific::normal(r, 0.0, 1.0);
    let n2, f2 = scientific::normal(r, 0.0, 1.0);
    assert(f1.kind == scientific::F_NONE && f2.kind == scientific::F_NONE);
    assert(n1 == 0.892470392690831 && n2 == 0.21741001724982148);
    // A spread below zero is a fault.
    let n3, f3 = scientific::normal(r, 0.0, -1.0);
    assert(f3.kind == scientific::F_VALUE);
    // The exponential law takes the inverse of the law.
    let e, f4 = scientific::exponential(r, 2.0);
    assert(f4.kind == scientific::F_NONE && e == 4.053365406960153);
    let e2, f5 = scientific::exponential(r, -1.0);
    assert(f5.kind == scientific::F_VALUE);
    // The method of Knuth holds a rate of 30 and below.
    let p, f6 = scientific::poisson(r, 5.0);
    assert(f6.kind == scientific::F_NONE && p == 5);
    let p0, f7 = scientific::poisson(r, 0.0);
    assert(f7.kind == scientific::F_NONE && p0 == 0);
    let pbig, f8 = scientific::poisson(r, 31.0);
    assert(f8.kind == scientific::F_VALUE);
    // The order of a count holds every whole number of the count.
    let pc, f9 = scientific::permutation(r, 5);
    assert(f9.kind == scientific::F_NONE);
    assert(pc.data[0] == 1.0 && pc.data[1] == 0.0 && pc.data[2] == 2.0);
    assert(pc.data[3] == 4.0 && pc.data[4] == 3.0);
    // A range that holds no number is a fault.
    let a2, f10 = scientific::integers(r, 5, 5);
    assert(f10.kind == scientific::F_VALUE);
    // The picks of an array.
    var r2 = scientific::default_rng(7);
    let src = scientific::from_values([10.0, 20.0, 30.0, 40.0]);
    let c1, fc1 = scientific::choice(r2, src, 3, true);
    assert(fc1.kind == scientific::F_NONE);
    assert(c1.data[0] == 40.0 && c1.data[1] == 10.0 && c1.data[2] == 30.0);
    // A pick without a repeat holds no element twice.
    let c2, fc2 = scientific::choice(r2, src, 3, false);
    assert(fc2.kind == scientific::F_NONE);
    assert(c2.data[0] == 10.0 && c2.data[1] == 30.0 && c2.data[2] == 20.0);
    let c3, fc3 = scientific::choice(r2, src, 5, false);
    assert(fc3.kind == scientific::F_VALUE);
    // A shuffled copy leaves the array of the caller as it was.
    var r3 = scientific::default_rng(9);
    let src2 = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0]);
    let sh, fsh = scientific::shuffled(r3, src2);
    assert(fsh.kind == scientific::F_NONE);
    assert(sh.data[0] == 2.0 && sh.data[1] == 5.0 && sh.data[2] == 1.0);
    assert(sh.data[3] == 3.0 && sh.data[4] == 4.0);
    var same = true;
    var i: i64 = 0;
    while i < 5 {
        if src2.data[i] != i + 1 as f64 { same = false; }
        i += 1;
    }
    assert(same);
    // The rows of a matrix move as a whole.
    var r4 = scientific::default_rng(7);
    let m = scientific::from_rows(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let fr = scientific::shuffle_rows(r4, m);
    assert(fr.kind == scientific::F_NONE);
    assert(m.data[0] == 3.0 && m.data[1] == 4.0 && m.data[2] == 5.0);
    assert(m.data[3] == 6.0 && m.data[4] == 1.0 && m.data[5] == 2.0);
    // The same seed gives the same numbers.
    var ra = scientific::default_rng(11);
    var rb = scientific::default_rng(11);
    var i2: i64 = 0;
    while i2 < 4 {
        let ua = scientific::uniform(ra, 0.0, 1.0);
        let ub = scientific::uniform(rb, 0.0, 1.0);
        assert(ua == ub);
        i2 += 1;
    }
}

fn test_misc() {
    // The shape functions.
    let e, fe = scientific::empty([2, 3]);
    assert(fe.kind == scientific::F_NONE && e.data.len == 6 && e.data[0] == 0.0);
    let fl, ffl = scientific::full_like(e, 7.0);
    assert(ffl.kind == scientific::F_NONE && fl.data[0] == 7.0);
    assert(fl.shape[0] == 2 && fl.shape[1] == 3);
    let zz, fzz = scientific::zeros_like(e);
    assert(fzz.kind == scientific::F_NONE && zz.data[0] == 0.0);
    let oo, foo = scientific::ones_like(e);
    assert(foo.kind == scientific::F_NONE && oo.data[0] == 1.0);
    // The writes through a mask.
    var m = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    let fpm = scientific::putmask(m, scientific::from_values([1.0, 0.0, 1.0, 0.0]), [9.0]);
    assert(fpm.kind == scientific::F_NONE);
    assert(m.data[0] == 9.0 && m.data[1] == 2.0 && m.data[2] == 9.0 && m.data[3] == 4.0);
    var m2 = scientific::from_values([0.0, 0.0, 0.0, 0.0]);
    let fpl = scientific::place(m2, scientific::from_values([1.0, 0.0, 1.0, 1.0]), [5.0, 6.0]);
    assert(fpl.kind == scientific::F_NONE);
    assert(m2.data[0] == 5.0 && m2.data[2] == 6.0 && m2.data[3] == 5.0);
    // place cycles the values when they run out, as NumPy does.
    let fplbad = scientific::place(m2, scientific::from_values([1.0, 1.0, 1.0, 0.0]), [7.0]);
    assert(fplbad.kind == scientific::F_NONE);
    assert(m2.data[0] == 7.0 && m2.data[1] == 7.0 && m2.data[2] == 7.0);
    // values that hold no element are a fault, and the array stays.
    let fplnone = scientific::place(m2, scientific::from_values([1.0, 0.0, 0.0, 0.0]), []);
    assert(fplnone.kind == scientific::F_VALUE);
    // a mask of a different count is a fault.
    let fplshape = scientific::place(m2, scientific::from_values([1.0, 0.0]), [7.0]);
    assert(fplshape.kind == scientific::F_SHAPE);
    // The cuts.
    let a6 = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let sp, fsp = scientific::split(a6, 2, 0);
    assert(fsp.kind == scientific::F_NONE && sp.count == 2);
    let sp0, fsp0 = scientific::part(sp, 0);
    let sp1, fsp1 = scientific::part(sp, 1);
    assert(fsp0.kind == scientific::F_NONE && fsp1.kind == scientific::F_NONE);
    assert(sp0.data.len == 3 && sp0.data[0] == 1.0 && sp0.data[2] == 3.0);
    assert(sp1.data.len == 3 && sp1.data[0] == 4.0 && sp1.data[2] == 6.0);
    let spbad, fspb = scientific::part(sp, 2);
    assert(fspb.kind == scientific::F_RANGE);
    let spcut, fspcut = scientific::split(a6, 4, 0);
    assert(fspcut.kind == scientific::F_VALUE);
    // The uneven cut gives the first pieces one element more.
    let asp, fasp = scientific::array_split(a6, 4, 0);
    assert(fasp.kind == scientific::F_NONE && asp.count == 4);
    let ap0, fap0 = scientific::part(asp, 0);
    let ap1, fap1 = scientific::part(asp, 1);
    let ap2, fap2 = scientific::part(asp, 2);
    let ap3, fap3 = scientific::part(asp, 3);
    assert(fap0.kind == scientific::F_NONE);
    assert(ap0.data.len == 2 && ap0.data[0] == 1.0 && ap0.data[1] == 2.0);
    assert(ap1.data.len == 2 && ap1.data[0] == 3.0 && ap1.data[1] == 4.0);
    assert(ap2.data.len == 1 && ap2.data[0] == 5.0);
    assert(ap3.data.len == 1 && ap3.data[0] == 6.0);
    let m23 = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let hs, fhs = scientific::hsplit(m23, 3);
    assert(fhs.kind == scientific::F_NONE && hs.count == 3);
    let hp0, fhp0 = scientific::part(hs, 0);
    let hp1, fhp1 = scientific::part(hs, 1);
    let hp2, fhp2 = scientific::part(hs, 2);
    assert(fhp0.kind == scientific::F_NONE);
    assert(hp0.data.len == 2 && hp0.data[0] == 1.0 && hp0.data[1] == 4.0);
    assert(hp0.shape[0] == 2 && hp0.shape[1] == 1);
    assert(hp1.data.len == 2 && hp1.data[0] == 2.0 && hp1.data[1] == 5.0);
    assert(hp2.data.len == 2 && hp2.data[0] == 3.0 && hp2.data[1] == 6.0);
    let vs, fvs = scientific::vsplit(m23, 2);
    assert(fvs.kind == scientific::F_NONE && vs.count == 2);
    let vp0, fvp0 = scientific::part(vs, 0);
    assert(fvp0.kind == scientific::F_NONE);
    assert(vp0.data.len == 3 && vp0.data[0] == 1.0 && vp0.data[2] == 3.0);
    // The growth.
    let pd, fpd = scientific::pad_value(a6, 0, 2, 1, 0.5);
    assert(fpd.kind == scientific::F_NONE && pd.data.len == 9);
    assert(pd.data[0] == 0.5 && pd.data[1] == 0.5 && pd.data[2] == 1.0);
    assert(pd.data[8] == 0.5);
    let pdbad, fpdbad = scientific::pad_value(a6, 0, -1, 1, 0.0);
    assert(fpdbad.kind == scientific::F_VALUE);
    // The values between points.
    let ip, fip = scientific::interp([0.0, 10.0, 20.0], [0.0, 1.0, 3.0], [0.5, 2.0, -1.0, 5.0]);
    assert(fip.kind == scientific::F_NONE);
    assert(ip.data[0] == 5.0 && ip.data[1] == 15.0 && ip.data[2] == 0.0 && ip.data[3] == 20.0);
    let ipbad, fipbad = scientific::interp([0.0, 10.0], [1.0, 0.0], [0.5]);
    assert(fipbad.kind == scientific::F_VALUE);
    // The derivative.
    let gr, fgr = scientific::gradient(scientific::from_values([1.0, 4.0, 9.0, 16.0]), 1.0);
    assert(fgr.kind == scientific::F_NONE);
    assert(gr.data[0] == 3.0 && gr.data[1] == 4.0 && gr.data[2] == 6.0 && gr.data[3] == 7.0);
    let grbad, fgrbad = scientific::gradient(a6, 0.0);
    assert(fgrbad.kind == scientific::F_VALUE);
    // The trims and the triangles.
    let tz, ftz = scientific::trim_zeros(scientific::from_values([0.0, 0.0, 3.0, 1.0, 0.0, 2.0, 0.0]));
    assert(ftz.kind == scientific::F_NONE);
    assert(tz.data[0] == 3.0 && tz.data[1] == 1.0 && tz.data[2] == 0.0 && tz.data[3] == 2.0);
    let tzbad, ftzbad = scientific::trim_zeros(scientific::from_values([0.0, 0.0]));
    assert(ftzbad.kind == scientific::F_VALUE);
    let ins, fins = scientific::insert(a6, 2, [9.0, 9.5]);
    assert(fins.kind == scientific::F_NONE && ins.data.len == 8);
    assert(ins.data[0] == 1.0 && ins.data[1] == 2.0 && ins.data[2] == 9.0 && ins.data[3] == 9.5);
    assert(ins.data[4] == 3.0 && ins.data[7] == 6.0);
    let insbad, finsbad = scientific::insert(a6, 7, [1.0]);
    assert(finsbad.kind == scientific::F_RANGE);
    let tl, ftl = scientific::tril(m23, 0);
    let tu, ftu = scientific::triu(m23, 0);
    assert(ftl.kind == scientific::F_NONE && ftu.kind == scientific::F_NONE);
    assert(tl.data[0] == 1.0 && tl.data[1] == 0.0 && tl.data[2] == 0.0);
    assert(tl.data[3] == 4.0 && tl.data[4] == 5.0 && tl.data[5] == 0.0);
    assert(tu.data[0] == 1.0 && tu.data[1] == 2.0 && tu.data[2] == 3.0);
    assert(tu.data[3] == 0.0 && tu.data[4] == 5.0 && tu.data[5] == 6.0);
}

// dquad is the derivative of the square. The solvers that need a
// derivative take it as a named function.
fn dquad(x: f64) -> f64 { return 2.0 * x; }

// dflat is the derivative of a function that never changes.
fn dflat(x: f64) -> f64 { return 0.0; }

fn test_optimize() {
    // The root of the square of x and two.
    let r1, f1 = scientific::brentq(0.0, 2.0, 1e-12, 100) { x => x * x - 2.0 };
    assert(f1.kind == scientific::F_NONE);
    assert(close(r1, 1.4142135623731364));
    let r2, f2 = scientific::brenth(0.0, 2.0, 1e-12, 100) { x => x * x - 2.0 };
    assert(f2.kind == scientific::F_NONE);
    assert(close(r2, 1.414213562373095));
    // The root of a cosine, which lies between the two points.
    let r3, f3 = scientific::brentq(1.0, 4.0, 1e-12, 100) { x => mathshim::cosine(x) };
    assert(f3.kind == scientific::F_NONE);
    assert(close(r3, 1.570796326794891));
    // The root of a cubic, which lies between the two points.
    let r5, f5 = scientific::brentq(1.0, 3.0, 1e-12, 100) { x => x * x * x - 2.0 * x - 5.0 };
    assert(f5.kind == scientific::F_NONE);
    assert(close(r5, 2.0945514815423265));
    // The method of Newton gives the same root.
    let r4, f4 = scientific::newton(1.0, 1e-12, 100, dquad) { x => x * x - 2.0 };
    assert(f4.kind == scientific::F_NONE);
    assert(close(r4, 1.4142135623730951));
    // The points that hold no change of sign are a fault.
    let bad1, fb1 = scientific::brentq(3.0, 4.0, 1e-12, 100) { x => x * x - 2.0 };
    assert(fb1.kind == scientific::F_VALUE);
    let bad2, fb2 = scientific::brenth(3.0, 4.0, 1e-12, 100) { x => x * x - 2.0 };
    assert(fb2.kind == scientific::F_VALUE);
    // A tolerance of zero is a fault.
    let bad3, fb3 = scientific::brentq(0.0, 2.0, 0.0, 100) { x => x * x - 2.0 };
    assert(fb3.kind == scientific::F_VALUE);
    // A count of turns below one is a fault.
    let bad4, fb4 = scientific::brentq(0.0, 2.0, 1e-12, 0) { x => x * x - 2.0 };
    assert(fb4.kind == scientific::F_VALUE);
    // A count of turns that is too small is a fault.
    let bad5, fb5 = scientific::brentq(0.0, 2.0, 1e-12, 2) { x => x * x - 2.0 };
    assert(fb5.kind == scientific::F_LIMIT);
    // A derivative of zero is a fault.
    let bad6, fb6 = scientific::newton(1.0, 1e-12, 100, dflat) { x => x * x - 2.0 };
    assert(fb6.kind == scientific::F_VALUE);
    // The root of two equations at once.
    let r5, f5 = scientific::fsolve([1.0, 1.0], 1e-10, 50) { v => [v[0] * v[0] + v[1] * v[1] - 4.0, v[0] - v[1]] };
    assert(f5.kind == scientific::F_NONE);
    assert(r5.data.len == 2 && close(r5.data[0], 1.4142135623730951));
    assert(close(r5.data[1], 1.4142135623730951));
    // A function that gives a different count of values is a fault.
    let bad7, fb7 = scientific::fsolve([1.0, 1.0], 1e-10, 50) { v => [v[0] * v[0] - 2.0] };
    assert(fb7.kind == scientific::F_SHAPE);
    // A count of turns that is too small is a fault.
    let bad8, fb8 = scientific::fsolve([1.0, 1.0], 1e-14, 1) { v => [v[0] * v[0] + v[1] * v[1] - 4.0, v[0] - v[1]] };
    assert(fb8.kind == scientific::F_LIMIT);
}

fn test_minimize() {
    // The smallest point of a parabola.
    let m1, f1 = scientific::minimize_scalar(-2.0, 2.0, 1e-10, 200) { x => (x - 0.5) * (x - 0.5) + 1.0 };
    assert(f1.kind == scientific::F_NONE);
    assert(close(m1.data[0], 0.4999999894781067));
    // The smallest point of a parabola with a slope.
    let m2, f2 = scientific::minimize_scalar(-5.0, 5.0, 1e-10, 200) { x => (x - 3.0) * (x - 3.0) + (x + 1.0) * (x + 1.0) };
    assert(f2.kind == scientific::F_NONE);
    assert(close(m2.data[0], 0.9999999865310867));
    // The smallest point of the exponential.
    let m3, f3 = scientific::minimize_scalar(0.0, 3.0, 1e-10, 200) { x => mathshim::power_e(x) - 2.0 * x };
    assert(f3.kind == scientific::F_NONE);
    assert(close(m3.data[0], 0.6931471680308768));
    // The smallest point of two variables.
    let m4, f4 = scientific::minimize([1.0, 1.0], 1e-10, 500) { v => [(v[0] - 1.0) * (v[0] - 1.0) + (v[1] + 2.0) * (v[1] + 2.0)] };
    assert(f4.kind == scientific::F_NONE);
    assert(m4.data.len == 2);
    assert(close(m4.data[0], 1.0000000000153233) && close(m4.data[1], -2.0000000000334754));
    // The fit of two equations.
    let l1, f5 = scientific::least_squares([0.0, 0.0], 1e-10, 200) { v => [v[0] + v[1] - 3.0, v[0] - v[1] - 1.0] };
    assert(f5.kind == scientific::F_NONE);
    assert(l1.data.len == 2);
    assert(close(l1.data[0], 1.999999999998008) && close(l1.data[1], 0.999999999999004));
    // A tolerance of zero is a fault.
    let bad1, fb1 = scientific::minimize_scalar(-2.0, 2.0, 0.0, 200) { x => x * x };
    assert(fb1.kind == scientific::F_VALUE);
    // A count of turns below one is a fault.
    let bad2, fb2 = scientific::minimize_scalar(-2.0, 2.0, 1e-10, 0) { x => x * x };
    assert(fb2.kind == scientific::F_VALUE);
    let bad3, fb3 = scientific::minimize([1.0], 0.0, 200) { v => [v[0] * v[0]] };
    assert(fb3.kind == scientific::F_VALUE);
    let bad4, fb4 = scientific::minimize([1.0], 1e-10, 0) { v => [v[0] * v[0]] };
    assert(fb4.kind == scientific::F_VALUE);
    // A start that holds no element is a fault.
    let bad5, fb5 = scientific::minimize([], 1e-10, 200) { v => [v[0] * v[0]] };
    assert(fb5.kind == scientific::F_VALUE);
    let bad6, fb6 = scientific::least_squares([], 1e-10, 200) { v => [v[0] * v[0]] };
    assert(fb6.kind == scientific::F_VALUE);
}

fn test_integrate() {
    // The integral of a straight line over zero to one.
    let y = [0.0, 0.25, 0.5, 0.75, 1.0];
    let t1, f1 = scientific::trapezoid(y, 0.25);
    assert(f1.kind == scientific::F_NONE && close(t1, 0.5));
    let s1, f2 = scientific::simpson(y, 0.25);
    assert(f2.kind == scientific::F_NONE && close(s1, 0.5));
    let r1, f3 = scientific::romberg(y, 0.25);
    assert(f3.kind == scientific::F_NONE && close(r1, 0.5));
    let c1, f4 = scientific::cumulative_trapezoid(y, 0.25);
    assert(f4.kind == scientific::F_NONE && c1.data.len == 4);
    assert(close(c1.data[0], 0.03125) && close(c1.data[1], 0.125));
    assert(close(c1.data[2], 0.28125) && close(c1.data[3], 0.5));
    // The integral of a square over zero to two.
    let ones = [1.0, 1.0, 1.0, 1.0, 1.0];
    let t2, f5 = scientific::trapezoid(ones, 0.5);
    assert(f5.kind == scientific::F_NONE && close(t2, 2.0));
    // An even count of parts takes the trapezoid on the last part.
    let s3, f6 = scientific::simpson([0.0, 0.5, 1.0, 1.5], 0.5);
    assert(f6.kind == scientific::F_NONE && close(s3, 1.125));
    // The integral of the sine over zero to pi.
    var y9: f64[>..] = [];
    var i: i64 = 0;
    while i <= 8 {
        y9.push(mathshim::sine(3.141592653589793 * i / 8.0));
        i += 1;
    }
    let s9, f7 = scientific::simpson(y9, 3.141592653589793 / 8.0);
    assert(f7.kind == scientific::F_NONE && close(s9, 2.000269169948388));
    let r9, f8 = scientific::romberg(y9, 3.141592653589793 / 8.0);
    assert(f8.kind == scientific::F_NONE && close(r9, 2.000005549979671));
    // A spacing of zero is a fault.
    let bad1, fb1 = scientific::trapezoid(y, 0.0);
    assert(fb1.kind == scientific::F_VALUE);
    let bad2, fb2 = scientific::simpson(y, 0.0);
    assert(fb2.kind == scientific::F_VALUE);
    let bad3, fb3 = scientific::cumulative_trapezoid(y, 0.0);
    assert(fb3.kind == scientific::F_VALUE);
    let bad4, fb4 = scientific::romberg(y, 0.0);
    assert(fb4.kind == scientific::F_VALUE);
    // A signal of too few values is a fault.
    let bad5, fb5 = scientific::trapezoid([1.0], 1.0);
    assert(fb5.kind == scientific::F_VALUE);
    let bad6, fb6 = scientific::simpson([1.0, 2.0], 1.0);
    assert(fb6.kind == scientific::F_VALUE);
    let bad7, fb7 = scientific::romberg([1.0, 2.0], 1.0);
    assert(fb7.kind == scientific::F_VALUE);
    // A count of the parts that is not a power of two is a fault.
    let bad8, fb8 = scientific::romberg([0.0, 1.0, 2.0, 3.0], 1.0);
    assert(fb8.kind == scientific::F_VALUE);
}

fn test_interpolate() {
    // The spline through four points.
    let sp, f1 = scientific::spline([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 1.0]);
    assert(f1.kind == scientific::F_NONE);
    let v1, f2 = scientific::spline_value(sp, 0.5);
    assert(f2.kind == scientific::F_NONE && close(v1, 0.75));
    let v2, f3 = scientific::spline_value(sp, 1.5);
    assert(f3.kind == scientific::F_NONE && close(v2, 0.5));
    let v3, f4 = scientific::spline_value(sp, 2.5);
    assert(f4.kind == scientific::F_NONE && close(v3, 0.25));
    // The values at a row of points.
    let vs, f5 = scientific::spline_values(sp, [0.25, 1.75]);
    assert(f5.kind == scientific::F_NONE);
    assert(close(vs.data[0], 0.40625) && close(vs.data[1], 0.1875));
    // The spline passes through the points of the signal.
    let v4, f6 = scientific::spline_value(sp, 1.0);
    assert(f6.kind == scientific::F_NONE && v4 == 1.0);
    let v5, f7 = scientific::spline_value(sp, 3.0);
    assert(f7.kind == scientific::F_NONE && v5 == 1.0);
    // Points that do not rise are a fault.
    let bad1, fb1 = scientific::spline([0.0, 1.0, 1.0], [0.0, 1.0, 2.0]);
    assert(fb1.kind == scientific::F_VALUE);
    // Values of a different count are a fault.
    let bad2, fb2 = scientific::spline([0.0, 1.0, 2.0], [0.0, 1.0]);
    assert(fb2.kind == scientific::F_SHAPE);
    // A signal of too few points is a fault.
    let bad3, fb3 = scientific::spline([0.0, 1.0], [0.0, 1.0]);
    assert(fb3.kind == scientific::F_VALUE);
    // A point outside the signal is a fault.
    let bad4, fb4 = scientific::spline_value(sp, -1.0);
    assert(fb4.kind == scientific::F_RANGE);
    let bad5, fb5 = scientific::spline_value(sp, 4.0);
    assert(fb5.kind == scientific::F_RANGE);
}

fn test_signal() {
    // The overlap of two signals.
    let c1, f1 = scientific::convolve([1.0, 2.0, 3.0], [0.0, 1.0, 0.5]);
    assert(f1.kind == scientific::F_NONE && c1.data.len == 5);
    assert(c1.data[0] == 0.0 && c1.data[1] == 1.0 && c1.data[2] == 2.5);
    assert(c1.data[3] == 4.0 && c1.data[4] == 1.5);
    // The match of a pattern in a signal.
    let k1, f2 = scientific::correlate([1.0, 2.0, 3.0], [0.0, 1.0, 0.5]);
    assert(f2.kind == scientific::F_NONE && k1.data.len == 5);
    assert(k1.data[0] == 0.5 && k1.data[1] == 2.0 && k1.data[2] == 3.5);
    assert(k1.data[3] == 3.0 && k1.data[4] == 0.0);
    // The line under a straight signal comes out, so the signal holds
    // zeros.
    let d1, f3 = scientific::detrend([0.0, 1.0, 2.0, 3.0]);
    assert(f3.kind == scientific::F_NONE && d1.data.len == 4);
    assert(mathshim_abs(d1.data[0]) < 0.0000001 && mathshim_abs(d1.data[3]) < 0.0000001);
    // The size of the parts of a sine holds its frequency.
    var y: f64[>..] = [];
    var i: i64 = 0;
    while i < 64 { y.push(mathshim::sine(6.283185307179586 * 8.0 * i / 64.0)); i += 1; }
    let w, f4 = scientific::welch(y, 64);
    assert(f4.kind == scientific::F_NONE && w.data.len == 33);
    var best: i64 = 0;
    i = 1;
    while i < w.data.len {
        if w.data[i] > w.data[best] { best = i; }
        i += 1;
    }
    assert(best == 8 && close(w.data[best], 21.333333333333343));
    // A signal that holds no value is a fault.
    let bad1, fb1 = scientific::convolve([], [1.0]);
    assert(fb1.kind == scientific::F_VALUE);
    let bad2, fb2 = scientific::correlate([1.0], []);
    assert(fb2.kind == scientific::F_VALUE);
    // A signal of too few values is a fault.
    let bad3, fb3 = scientific::detrend([1.0]);
    assert(fb3.kind == scientific::F_VALUE);
    // A part of too few values is a fault.
    let bad4, fb4 = scientific::welch(y, 1);
    assert(fb4.kind == scientific::F_VALUE);
    // A signal that holds fewer values than a part is a fault.
    let bad5, fb5 = scientific::welch([1.0, 2.0], 8);
    assert(fb5.kind == scientific::F_VALUE);
}

// The elementwise math holds the values of NumPy.
fn test_elementwise_math() {
    let a, f = scientific::from_values([0.25, 0.5, 1.0, 2.0]);
    assert(f.kind == scientific::F_NONE);
    let s, fs = scientific::sin(a);
    assert(fs.kind == scientific::F_NONE);
    assert(close(s.data[0], 0.24740395925452294) && close(s.data[1], 0.479425538604203));
    assert(close(s.data[2], 0.8414709848078965) && close(s.data[3], 0.9092974268256817));
    let c, fc = scientific::cos(a);
    assert(fc.kind == scientific::F_NONE);
    assert(close(c.data[0], 0.9689124217106447) && close(c.data[1], 0.8775825618903728));
    assert(close(c.data[2], 0.5403023058681398) && close(c.data[3], -0.4161468365471424));
    let t, ft = scientific::tan(a);
    assert(ft.kind == scientific::F_NONE);
    assert(close(t.data[0], 0.25534192122103627) && close(t.data[1], 0.5463024898437905));
    assert(close(t.data[2], 1.5574077246549023) && close(t.data[3], -2.185039863261519));
    let asin2, fas = scientific::arcsin(a);
    assert(fas.kind == scientific::F_NONE);
    assert(close(asin2.data[0], 0.25268025514207865) && close(asin2.data[1], 0.5235987755982989));
    assert(close(asin2.data[2], 1.5707963267948966) && asin2.data[3] != asin2.data[3]);
    let ac, fac = scientific::arccos(a);
    assert(fac.kind == scientific::F_NONE);
    assert(close(ac.data[0], 1.318116071652818) && close(ac.data[1], 1.0471975511965979));
    assert(close(ac.data[2], 0.0) && ac.data[3] != ac.data[3]);
    let at, fat = scientific::arctan(a);
    assert(fat.kind == scientific::F_NONE);
    assert(close(at.data[0], 0.24497866312686414) && close(at.data[1], 0.4636476090008061));
    assert(close(at.data[2], 0.7853981633974483) && close(at.data[3], 1.1071487177940904));
    // The angle of a point of the plane comes from two arrays.
    let y, fy = scientific::from_values([1.0, -1.0, 0.0, 2.0]);
    let x, fx = scientific::from_values([1.0, 1.0, -1.0, -2.0]);
    let a2, fa2 = scientific::arctan2(y, x);
    assert(fa2.kind == scientific::F_NONE);
    assert(close(a2.data[0], 0.7853981633974483) && close(a2.data[1], -0.7853981633974483));
    assert(close(a2.data[2], 3.141592653589793) && close(a2.data[3], 2.356194490192345));
    // The hyperbolic functions hold the values of NumPy.
    let h, fh = scientific::from_values([-1.0, -0.5, 0.0, 0.5, 1.0]);
    assert(fh.kind == scientific::F_NONE);
    let sh, fsh = scientific::sinh(h);
    assert(fsh.kind == scientific::F_NONE);
    assert(close(sh.data[0], -1.1752011936438014) && close(sh.data[1], -0.5210953054937474));
    assert(close(sh.data[2], 0.0) && close(sh.data[3], 0.5210953054937474));
    assert(close(sh.data[4], 1.1752011936438014));
    let ch, fch = scientific::cosh(h);
    assert(fch.kind == scientific::F_NONE);
    assert(close(ch.data[0], 1.5430806348152437) && close(ch.data[1], 1.1276259652063807));
    assert(close(ch.data[2], 1.0) && close(ch.data[3], 1.1276259652063807));
    assert(close(ch.data[4], 1.5430806348152437));
    let th, fth = scientific::tanh(h);
    assert(fth.kind == scientific::F_NONE);
    assert(close(th.data[0], -0.7615941559557649) && close(th.data[1], -0.46211715726000974));
    assert(close(th.data[2], 0.0) && close(th.data[3], 0.46211715726000974));
    assert(close(th.data[4], 0.7615941559557649));
    let ash, fash = scientific::arcsinh(h);
    assert(fash.kind == scientific::F_NONE);
    assert(close(ash.data[0], -0.881373587019543) && close(ash.data[1], -0.48121182505960347));
    assert(close(ash.data[2], 0.0) && close(ash.data[3], 0.48121182505960347));
    assert(close(ash.data[4], 0.881373587019543));
    // The arc of the hyperbolic cosine holds no value below one.
    let ac2, fac2 = scientific::arccosh(scientific::from_values([2.0, 1.5, 1.0, 1.5, 2.0]));
    assert(fac2.kind == scientific::F_NONE);
    assert(close(ac2.data[0], 1.3169578969248168) && close(ac2.data[1], 0.9624236501192069));
    assert(close(ac2.data[2], 0.0) && close(ac2.data[3], 0.9624236501192069));
    assert(close(ac2.data[4], 1.3169578969248168));
    let ac3, fac3 = scientific::arccosh(scientific::from_values([0.5]));
    assert(fac3.kind == scientific::F_NONE && ac3.data[0] != ac3.data[0]);
    let ath, fath = scientific::arctanh(scientific::from_values([-0.9, -0.5, 0.0, 0.5, 0.9]));
    assert(fath.kind == scientific::F_NONE);
    assert(close(ath.data[0], -1.4722194895832204) && close(ath.data[1], -0.5493061443340549));
    assert(close(ath.data[2], 0.0) && close(ath.data[3], 0.5493061443340549));
    assert(close(ath.data[4], 1.4722194895832204));
    // The arc of the hyperbolic tangent holds an infinity at one.
    let ath2, fath2 = scientific::arctanh(scientific::from_values([1.0, 2.0]));
    assert(fath2.kind == scientific::F_NONE);
    assert(ath2.data[0] > 1000000.0 && ath2.data[1] != ath2.data[1]);
    // The power and the log of a small number hold more digits.
    let e, fe = scientific::from_values([-0.25, 0.0, 0.25, 0.75]);
    assert(fe.kind == scientific::F_NONE);
    let em, fem = scientific::expm1(e);
    assert(fem.kind == scientific::F_NONE);
    assert(close(em.data[0], -0.22119921692859512) && close(em.data[1], 0.0));
    assert(close(em.data[2], 0.2840254166877415) && close(em.data[3], 1.1170000166126748));
    let lp, flp = scientific::log1p(e);
    assert(flp.kind == scientific::F_NONE);
    assert(close(lp.data[0], -0.2876820724517809) && close(lp.data[1], 0.0));
    assert(close(lp.data[2], 0.22314355131420976) && close(lp.data[3], 0.5596157879354227));
    // The reciprocal of a number gives the number that multiplies to one.
    let r, fr = scientific::reciprocal(scientific::from_values([-2.0, 0.5, 4.0]));
    assert(fr.kind == scientific::F_NONE);
    assert(close(r.data[0], -0.5) && close(r.data[1], 2.0) && close(r.data[2], 0.25));
    // The degrees and the radians follow each other.
    let dr, fdr = scientific::deg2rad(scientific::from_values([0.0, 45.0, 90.0, 180.0]));
    assert(fdr.kind == scientific::F_NONE);
    assert(close(dr.data[0], 0.0) && close(dr.data[1], 0.7853981633974483));
    assert(close(dr.data[2], 1.5707963267948966) && close(dr.data[3], 3.141592653589793));
    let rd, frd = scientific::rad2deg(scientific::from_values([0.0, 0.7853981633974483, 1.5707963267948966, 3.141592653589793]));
    assert(frd.kind == scientific::F_NONE);
    assert(close(rd.data[0], 0.0) && close(rd.data[1], 45.0));
    assert(close(rd.data[2], 90.0) && close(rd.data[3], 180.0));
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
    assert(close(q.data[0], 3.0) && close(q.data[1], -4.0));
    assert(close(q.data[2], 3.0) && close(q.data[3], 9.0));
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
    assert(close(sub2.data[0], -3.0) && close(sub2.data[1], -1.0) && close(sub2.data[2], 2.0));
    let mul2, fmul2 = scientific::multiply(nv, scientific::from_values([2.0, 2.0, 2.0]));
    assert(fmul2.kind == scientific::F_NONE);
    assert(close(mul2.data[0], -4.0) && close(mul2.data[1], 0.0) && close(mul2.data[2], 6.0));
    let div2, fdiv2 = scientific::divide(nv, scientific::from_values([2.0, 2.0, 2.0]));
    assert(fdiv2.kind == scientific::F_NONE);
    assert(close(div2.data[0], -1.0) && close(div2.data[1], 0.0) && close(div2.data[2], 1.5));
    let td, ftd = scientific::true_divide(nv, scientific::from_values([2.0, 2.0, 2.0]));
    assert(ftd.kind == scientific::F_NONE);
    assert(close(td.data[0], -1.0) && close(td.data[1], 0.0) && close(td.data[2], 1.5));
    let neg2, fneg2 = scientific::negative(nv);
    assert(fneg2.kind == scientific::F_NONE);
    assert(close(neg2.data[0], 2.0) && close(neg2.data[1], 0.0) && close(neg2.data[2], -3.0));
    let pos2, fpos2 = scientific::positive(nv);
    assert(fpos2.kind == scientific::F_NONE);
    assert(close(pos2.data[0], -2.0) && close(pos2.data[1], 0.0) && close(pos2.data[2], 3.0));
    let fab2, ffab2 = scientific::fabs(nv);
    assert(ffab2.kind == scientific::F_NONE);
    assert(close(fab2.data[0], 2.0) && close(fab2.data[1], 0.0) && close(fab2.data[2], 3.0));
    let abs2, fabs2 = scientific::absolute(nv);
    assert(fabs2.kind == scientific::F_NONE);
    assert(close(abs2.data[0], 2.0) && close(abs2.data[1], 0.0) && close(abs2.data[2], 3.0));
    // A half goes to the even neighbour, and the decimals give the
    // place of the last digit.
    let ar, far = scientific::around(scientific::from_values([0.125, 0.375, 2.675, -1.5]), 2);
    assert(far.kind == scientific::F_NONE);
    assert(close(ar.data[0], 0.12) && close(ar.data[1], 0.38));
    assert(close(ar.data[2], 2.68) && close(ar.data[3], -1.5));
    let ar2, far2 = scientific::around(scientific::from_values([1234.0, 1236.0]), -1);
    assert(far2.kind == scientific::F_NONE);
    assert(close(ar2.data[0], 1230.0) && close(ar2.data[1], 1240.0));
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
    assert(close(mo.data[0], 1.0) && close(mo.data[1], 2.0));
    assert(close(mo.data[2], -2.0) && close(mo.data[3], -0.5));
    assert(mo.data[4] == 0.09999999999999995);
    let fm, ffm = scientific::fmod(ma, mb);
    assert(ffm.kind == scientific::F_NONE);
    assert(close(fm.data[0], 1.0) && close(fm.data[1], -1.0));
    assert(close(fm.data[2], 0.5) && close(fm.data[3], -0.5));
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

// The scales of numbers hold the values of NumPy.
fn test_scales() {
    let id, fid = scientific::identity(3);
    assert(fid.kind == scientific::F_NONE && id.data.len == 9);
    assert(id.data[0] == 1.0 && id.data[4] == 1.0 && id.data[8] == 1.0 && id.data[1] == 0.0);
    let vals, fvals = scientific::from_values([1.0, 2.0, 3.0]);
    assert(fvals.kind == scientific::F_NONE);
    let df, fdf = scientific::diagflat(vals.data, 1);
    assert(fdf.kind == scientific::F_NONE && df.data.len == 16);
    assert(df.data[1] == 1.0 && df.data[6] == 2.0 && df.data[11] == 3.0 && df.data[0] == 0.0);
    let df0, fdf0 = scientific::diagflat(vals.data);
    assert(fdf0.kind == scientific::F_NONE && df0.data.len == 9);
    assert(df0.data[0] == 1.0 && df0.data[4] == 2.0 && df0.data[8] == 3.0);
    // The matrix of the identity takes a count of columns and an
    // offset of the diagonal.
    let ey, fey = scientific::eye(3, 5, 1);
    assert(fey.kind == scientific::F_NONE && ey.data.len == 15);
    assert(ey.data[1] == 1.0 && ey.data[7] == 1.0 && ey.data[13] == 1.0 && ey.data[0] == 0.0);
    let ey2, fey2 = scientific::eye(3, 0, -1);
    assert(fey2.kind == scientific::F_NONE && ey2.data.len == 9);
    assert(ey2.data[3] == 1.0 && ey2.data[7] == 1.0 && ey2.data[0] == 0.0);
    // A size below one is a fault.
    let eybad, feybad = scientific::eye(0);
    assert(feybad.kind == scientific::F_VALUE && eybad.data.len == 0);
    // The scale of a power holds the power of the base.
    let ls, fls = scientific::logspace(0.0, 2.0, 3);
    assert(fls.kind == scientific::F_NONE);
    assert(close(ls.data[0], 1.0) && close(ls.data[1], 10.0) && close(ls.data[2], 100.0));
    let ls2, fls2 = scientific::logspace(0.0, 3.0, 4, 2.0);
    assert(fls2.kind == scientific::F_NONE);
    assert(close(ls2.data[0], 1.0) && close(ls2.data[1], 2.0));
    assert(close(ls2.data[2], 4.0) && close(ls2.data[3], 8.0));
    // A count below one is a fault.
    let lsbad, flsbad = scientific::logspace(0.0, 2.0, 0);
    assert(flsbad.kind == scientific::F_VALUE && lsbad.data.len == 0);
    // The scale of a geometric sequence rises by a factor.
    let gs, fgs = scientific::geomspace(1.0, 100.0, 3);
    assert(fgs.kind == scientific::F_NONE);
    assert(close(gs.data[0], 1.0) && close(gs.data[1], 10.0) && close(gs.data[2], 100.0));
    let gs2, fgs2 = scientific::geomspace(-1.0, -100.0, 3);
    assert(fgs2.kind == scientific::F_NONE);
    assert(close(gs2.data[0], -1.0) && close(gs2.data[1], -10.0) && close(gs2.data[2], -100.0));
    // A zero holds no geometric sequence.
    let gsbad, fgsbad = scientific::geomspace(0.0, 100.0, 3);
    assert(fgsbad.kind == scientific::F_VALUE && gsbad.data.len == 0);
    let gsbad2, fgsbad2 = scientific::geomspace(1.0, 0.0, 3);
    assert(fgsbad2.kind == scientific::F_VALUE && gsbad2.data.len == 0);
}

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
    assert(close(v0, 8.666666666666666) && close(s0, 2.943920288775949));
    assert(close(v1, 13.0) && close(s1, 3.605551275463989));
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

// The order statistics hold the values of NumPy.
fn test_order_statistics() {
    let a, fa = scientific::from_values([1.0, 2.0, 4.0, 7.0, 11.0, 12.0]);
    assert(fa.kind == scientific::F_NONE);
    // The quantile follows the method of NumPy.
    let q, fq = scientific::quantile(a, 0.25);
    assert(fq.kind == scientific::F_NONE && q == 2.5);
    let q2, fq2 = scientific::quantile(a, 0.25, "lower");
    assert(fq2.kind == scientific::F_NONE && q2 == 2.0);
    let q3, fq3 = scientific::quantile(a, 0.25, "higher");
    assert(fq3.kind == scientific::F_NONE && q3 == 4.0);
    let q4, fq4 = scientific::quantile(a, 0.25, "midpoint");
    assert(fq4.kind == scientific::F_NONE && q4 == 3.0);
    let q5, fq5 = scientific::quantile(a, 0.5, "nearest");
    assert(fq5.kind == scientific::F_NONE && q5 == 4.0);
    let q6, fq6 = scientific::quantile(a, 0.25, "inverted_cdf");
    assert(fq6.kind == scientific::F_NONE && q6 == 2.0);
    let q7, fq7 = scientific::quantile(a, 0.5, "averaged_inverted_cdf");
    assert(fq7.kind == scientific::F_NONE && q7 == 5.5);
    let q8, fq8 = scientific::quantile(a, 0.1, "hazen");
    assert(fq8.kind == scientific::F_NONE && close(q8, 1.1));
    let q9, fq9 = scientific::quantile(a, 0.25, "weibull");
    assert(fq9.kind == scientific::F_NONE && close(q9, 1.75));
    let q10, fq10 = scientific::quantile(a, 0.25, "median_unbiased");
    assert(fq10.kind == scientific::F_NONE && close(q10, 1.9166666666666667));
    let q11, fq11 = scientific::quantile(a, 0.25, "normal_unbiased");
    assert(fq11.kind == scientific::F_NONE && close(q11, 1.9375));
    let q12, fq12 = scientific::quantile(a, 0.1, "interpolated_inverted_cdf");
    assert(fq12.kind == scientific::F_NONE && close(q12, 1.0));
    let q13, fq13 = scientific::quantile(a, 0.5, "closest_observation");
    assert(fq13.kind == scientific::F_NONE && q13 == 4.0);
    // A quantile outside the range and an unknown method are faults.
    let qb, fqb = scientific::quantile(a, 1.5);
    assert(fqb.kind == scientific::F_VALUE && qb == 0.0);
    let qb2, fqb2 = scientific::quantile(a, 0.5, "unknown");
    assert(fqb2.kind == scientific::F_VALUE);
    // The percentage runs from zero to one hundred.
    let p, fp = scientific::percentile(a, 50.0);
    assert(fp.kind == scientific::F_NONE && p == 5.5);
    let p2, fp2 = scientific::percentile(a, 25.0);
    assert(fp2.kind == scientific::F_NONE && p2 == 2.5);
    let pb, fpb = scientific::percentile(a, 150.0);
    assert(fpb.kind == scientific::F_VALUE);
    // The middle value of an even count is the mean of the two.
    let med, fmed = scientific::median(a);
    assert(fmed.kind == scientific::F_NONE && med == 5.5);
    let med2, fmed2 = scientific::median(scientific::from_values([1.0, 2.0, 4.0, 7.0]));
    assert(fmed2.kind == scientific::F_NONE && med2 == 3.0);
    // The mode takes the value that comes most often.
    let m1, c1, fm1 = scientific::mode(scientific::from_values([1.0, 2.0, 2.0, 3.0, 3.0]));
    assert(fm1.kind == scientific::F_NONE && m1 == 2.0 && c1 == 2);
    let m2, c2, fm2 = scientific::mode(scientific::from_values([5.0, 5.0, 5.0, 1.0, 1.0]));
    assert(fm2.kind == scientific::F_NONE && m2 == 5.0 && c2 == 3);
    let m3, c3, fm3 = scientific::mode(scientific::from_values([2.0, 1.0, 1.0, 2.0, 3.0]));
    assert(fm3.kind == scientific::F_NONE && m3 == 1.0 && c3 == 2);
    // The weighted mean follows the weights.
    let w, fw = scientific::from_values([4.0, 3.0, 2.0, 1.0]);
    let av, fav = scientific::average(scientific::from_values([1.0, 2.0, 3.0, 4.0]), w);
    assert(fav.kind == scientific::F_NONE && av == 2.0);
    // Weights of a different shape and weights of zero are faults.
    let aw, faw = scientific::average(scientific::from_values([1.0, 2.0]), scientific::from_values([1.0]));
    assert(faw.kind == scientific::F_SHAPE);
    let aw2, faw2 = scientific::average(scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                                        scientific::from_values([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    assert(faw2.kind == scientific::F_SHAPE && aw2 == 0.0);
    let az, faz = scientific::average(scientific::from_values([1.0, 2.0]), scientific::from_values([0.0, 0.0]));
    assert(faz.kind == scientific::F_VALUE);
    // The forms along an axis drop the axis.
    let m4, fm4 = scientific::from_rows(2, 3, [1.0, 2.0, 4.0, 7.0, 11.0, 12.0]);
    assert(fm4.kind == scientific::F_NONE);
    let qa, fqa = scientific::quantile_axis(m4, 0.5, 0);
    let ma, fma = scientific::median_axis(m4, 1);
    let pa, fpa = scientific::percentile_axis(m4, 50.0, 0);
    assert(fqa.kind == scientific::F_NONE && fma.kind == scientific::F_NONE && fpa.kind == scientific::F_NONE);
    assert(qa.data[0] == 4.0 && qa.data[1] == 6.5 && qa.data[2] == 8.0);
    assert(ma.data[0] == 2.0 && ma.data[1] == 11.0);
    assert(pa.data[0] == 4.0 && pa.data[1] == 6.5 && pa.data[2] == 8.0);
    // The weighted mean along an axis takes one weight for each place.
    let aa, faa = scientific::average_axis(m4, scientific::from_values([1.0, 1.0]), 0);
    assert(faa.kind == scientific::F_NONE && aa.data[0] == 4.0 && aa.data[1] == 6.5 && aa.data[2] == 8.0);
    let ab, fab = scientific::average_axis(m4, scientific::from_values([1.0, 1.0, 1.0]), 1);
    assert(fab.kind == scientific::F_NONE && close(ab.data[0], 2.3333333333333335) && ab.data[1] == 10.0);
    // Weights of a different length are a fault.
    let ac, fac = scientific::average_axis(m4, scientific::from_values([1.0]), 0);
    assert(fac.kind == scientific::F_SHAPE && ac.data.len == 0);
    // A bad axis is a fault.
    let qb3, fqb3 = scientific::quantile_axis(m4, 0.5, 5);
    assert(fqb3.kind == scientific::F_RANGE && qb3.data.len == 0);
}

// The sets of numbers hold the values of NumPy.
fn test_sets() {
    let a, fa = scientific::from_values([3.0, 1.0, 2.0, 2.0]);
    let b, fb = scientific::from_values([2.0, 3.0, 4.0]);
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    // The union holds each value once, in the order of its value.
    let u, fu = scientific::union1d(a, b);
    assert(fu.kind == scientific::F_NONE && u.data.len == 4);
    assert(u.data[0] == 1.0 && u.data[1] == 2.0 && u.data[2] == 3.0 && u.data[3] == 4.0);
    // The common values of two rows.
    let i2, fi2 = scientific::intersect1d(a, b);
    assert(fi2.kind == scientific::F_NONE && i2.data.len == 2);
    assert(i2.data[0] == 2.0 && i2.data[1] == 3.0);
    // The values of the first row that the second holds no value of.
    let s, fs = scientific::setdiff1d(a, b);
    assert(fs.kind == scientific::F_NONE && s.data.len == 1 && s.data[0] == 1.0);
    // The membership of a flat row.
    let q, fq = scientific::in1d(a, b);
    assert(fq.kind == scientific::F_NONE && q.data.len == 4);
    assert(q.data[0] == 1.0 && q.data[1] == 0.0 && q.data[2] == 1.0 && q.data[3] == 1.0);
    // The membership with the answer turned around.
    let qi, fqi = scientific::in1d(a, b, true);
    assert(fqi.kind == scientific::F_NONE && qi.data[0] == 0.0 && qi.data[3] == 0.0);
    // The membership keeps the shape of the array.
    let m, fm = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let e, fe = scientific::isin(m, b);
    assert(fe.kind == scientific::F_NONE && e.ndim == 2 && e.data.len == 4);
    assert(e.data[0] == 0.0 && e.data[1] == 1.0 && e.data[2] == 1.0 && e.data[3] == 1.0);
    // A NaN holds no value, not even another NaN.
    let nan = 0.0 / 0.0;
    let na, fna = scientific::from_values([1.0, nan, nan, 2.0]);
    let nb, fnb = scientific::from_values([nan, 2.0, 3.0]);
    assert(fna.kind == scientific::F_NONE && fnb.kind == scientific::F_NONE);
    let nu, fnu = scientific::union1d(na, nb);
    assert(fnu.kind == scientific::F_NONE && nu.data.len == 4);
    assert(nu.data[0] == 1.0 && nu.data[1] == 2.0 && nu.data[2] == 3.0);
    assert(nu.data[3] != nu.data[3]);
    let ni, fni = scientific::intersect1d(na, nb);
    assert(fni.kind == scientific::F_NONE && ni.data.len == 1 && ni.data[0] == 2.0);
    let ns, fns = scientific::setdiff1d(na, nb);
    assert(fns.kind == scientific::F_NONE && ns.data.len == 2);
    assert(ns.data[0] == 1.0 && ns.data[1] != ns.data[1]);
    let nq, fnq = scientific::in1d(na, nb);
    assert(fnq.kind == scientific::F_NONE && nq.data[0] == 0.0);
    assert(nq.data[1] == 0.0 && nq.data[2] == 0.0 && nq.data[3] == 1.0);
}

// The stacks hold the values of NumPy.
fn line_sum(line: scientific::Arr&) -> f64 {
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < line.data.len { t += line.data[i]; i += 1; }
    return t;
}

fn line_reverse(line: scientific::Arr&) -> scientific::Arr {
    var d: f64[>..] = [];
    var i: i64 = line.data.len - 1;
    while i >= 0 { d.push(line.data[i]); i -= 1; }
    var s: i64[..8] = [];
    s.push(d.len);
    return scientific::Arr { ndim: 1, shape: s, data: d };
}

fn line_bounds(line: scientific::Arr&) -> scientific::Arr {
    var lo = line.data[0];
    var hi = line.data[0];
    var i: i64 = 1;
    while i < line.data.len {
        if line.data[i] < lo { lo = line.data[i]; }
        if line.data[i] > hi { hi = line.data[i]; }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(2);
    var d: f64[>..] = [];
    d.push(lo);
    d.push(hi);
    return scientific::Arr { ndim: 1, shape: s, data: d };
}

fn test_generators() {
    // The names of NumPy call the generator of the package. A name
    // gives the same numbers as the function under it.
    let r = scientific::seed(42);
    let r2 = scientific::default_rng(42);
    assert(scientific::random_sample(r) == scientific::random_uniform(r2));
    assert(scientific::randint(r, 1, 10) == scientific::integers(r2, 1, 10));
    let r3 = scientific::seed(7);
    let r4 = scientific::seed(7);
    var d: f64[>..] = [3.0, 1.0, 4.0, 1.5];
    var q = scientific::Arr { ndim: 1, shape: [4], data: copy(d) };
    let f = scientific::shuffle(r3, q);
    assert(f.kind == scientific::F_NONE);
    var d2: f64[>..] = [3.0, 1.0, 4.0, 1.5];
    scientific::mix(d2, r4.g);
    var i: i64 = 0;
    while i < 4 { assert(q.data[i] == d2[i]); i += 1; }
    // The shuffled copy of the plain values holds the same order.
    var d3: f64[>..] = [3.0, 1.0, 4.0, 1.5];
    let r5 = scientific::seed(7);
    let plain = scientific::Arr { ndim: 1, shape: [4], data: copy(d3) };
    let c, fc = scientific::shuffled(r5, plain);
    assert(fc.kind == scientific::F_NONE);
    i = 0;
    while i < 4 { assert(c.data[i] == d2[i]); i += 1; }
    let r6 = scientific::seed(1);
    let u, f2 = scientific::rand(r6, [2, 3]);
    assert(f2.kind == scientific::F_NONE && u.ndim == 2 && u.shape[0] == 2 && u.shape[1] == 3);
    assert(u.data[0] >= 0.0 && u.data[0] < 1.0 && u.data[5] >= 0.0 && u.data[5] < 1.0);
    let r7 = scientific::seed(1);
    let u2, f3 = scientific::rand(r7, [2, 3]);
    i = 0;
    while i < 6 { assert(u.data[i] == u2.data[i]); i += 1; }
    let n, f4 = scientific::randn(r6, [2, 3]);
    assert(f4.kind == scientific::F_NONE && n.data.len == 6);
    // The state round trips through the word.
    let r8 = scientific::seed(5);
    var out: u8[>..] = [];
    let g = scientific::bytes(r8, 4, out);
    assert(g.kind == scientific::F_NONE && out.len == 4);
    let st = scientific::get_state(r8);
    let r9 = scientific::set_state(st);
    assert(scientific::random_uniform(r9) == scientific::random_uniform(r8));
    // The faults.
    var bad: u8[>..] = [];
    let r10 = scientific::seed(3);
    let g2 = scientific::bytes(r10, 0, bad);
    assert(g2.kind == scientific::F_RANGE);
    let r11 = scientific::seed(3);
    let e1, g3 = scientific::rand(r11, []);
    assert(g3.kind == scientific::F_VALUE);
    let e2, g4 = scientific::randn(r11, [2, 0]);
    assert(g4.kind == scientific::F_VALUE);
    var m: f64[>..] = [1.0, 2.0, 3.0, 4.0];
    let q2 = scientific::Arr { ndim: 2, shape: [2, 2], data: copy(m) };
    let r12 = scientific::seed(3);
    let g5 = scientific::shuffle(r12, q2);
    assert(g5.kind == scientific::F_SHAPE);
    let badint = scientific::randint(r11, 5, 5);
}

fn close_row(a, at: i64, v: f64[:]) -> bool {
    var i: i64 = 0;
    while i < v.len {
        let d = a.data[at + i] - v[i];
        if d > 0.000000001 || d < -0.000000001 { return false; }
        i += 1;
    }
    return true;
}

fn close_all(a, v: f64[:]) -> bool {
    if a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        let d = a.data[i] - v[i];
        if d > 0.000000001 || d < -0.000000001 { return false; }
        i += 1;
    }
    return true;
}

fn mat(d: f64[:], rows: i64, cols: i64) -> scientific::Arr {
    var c: f64[>..] = [];
    var i: i64 = 0;
    while i < d.len { c.push(d[i]); i += 1; }
    var s: i64[..8] = [];
    s.push(rows);
    s.push(cols);
    return scientific::Arr { ndim: 2, shape: s, data: c };
}

fn test_stats() {
    // The values come from NumPy 2.5.3.
    let m = mat([1.0, 2.0, 4.0, 7.0, 2.0, 1.5, 3.0, 9.0, 5.0, 3.0, 1.0, 0.5], 3, 4);
    let c, f1 = scientific::cov(m);
    assert(f1.kind == scientific::F_NONE && c.shape[0] == 3 && c.shape[1] == 3);
    assert(close_all(c, [7.0, 8.583333333333332, -4.916666666666666, 8.583333333333332, 12.0625, -4.9375, -4.916666666666666, -4.9375, 4.229166666666666]));
    let c0, f2 = scientific::cov(m, true, 0);
    assert(f2.kind == scientific::F_NONE && close(c0.data[0], 5.25) && close(c0.data[4], 9.046875));
    let t = scientific::transpose(m);
    let ct, f3 = scientific::cov(t, false);
    assert(f3.kind == scientific::F_NONE && close(ct.data[1], 8.583333333333332) && close(ct.data[8], 4.229166666666666));
    let r, f4 = scientific::corrcoef(m);
    assert(f4.kind == scientific::F_NONE);
    assert(close_all(r, [1.0, 0.9340890810744523, -0.9036377079347127, 0.9340890810744522, 1.0, -0.6912912266696619, -0.9036377079347127, -0.6912912266696619, 1.0]));
    let row = mat([1.0, 2.0, 4.0, 7.0], 1, 4);
    let y = mat([3.0, 1.0, 2.0, 8.0], 1, 4);
    let cw, f5 = scientific::cov_with(row, y);
    assert(f5.kind == scientific::F_NONE && close_all(cw, [7.0, 6.666666666666666, 6.666666666666666, 9.666666666666666]));
    let rw, f6 = scientific::corrcoef_with(row, y);
    assert(f6.kind == scientific::F_NONE && close(rw.data[1], 0.8104408984731077) && close(rw.data[0], 1.0));
    let bad = mat([1.0, 2.0, 3.0], 1, 3);
    let e1, g1 = scientific::cov_with(row, bad);
    assert(g1.kind == scientific::F_SHAPE);
    // The NaN statistics.
    let nan = 0.0 / 0.0;
    var av: f64[>..] = [3.0, nan, 1.0, 7.0, nan, 5.0];
    let a = scientific::Arr { ndim: 1, shape: [6], data: copy(av) };
    let mx, g2 = scientific::nanargmax(a);
    let mn, g3 = scientific::nanargmin(a);
    assert(g2.kind == scientific::F_NONE && mx == 3 && g3.kind == scientific::F_NONE && mn == 2);
    let md, g4 = scientific::nanmedian(a);
    let pc, g5 = scientific::nanpercentile(a, 30.0);
    let qq, g6 = scientific::nanquantile(a, 0.9, "lower");
    assert(g4.kind == scientific::F_NONE && close(md, 4.0) && close(pc, 2.8) && close(qq, 5.0));
    var only: f64[>..] = [nan, nan];
    let allnan = scientific::Arr { ndim: 1, shape: [2], data: copy(only) };
    let nm, g7 = scientific::nanmedian(allnan);
    assert(g7.kind == scientific::F_NONE && nm != nm);
    let ng, g8 = scientific::nanargmax(allnan);
    assert(g8.kind == scientific::F_VALUE);
    var bv: f64[>..] = [1.0, nan, 3.0, 2.0, 5.0, nan];
    let b = mat(bv, 2, 3);
    let cs, g9 = scientific::nancumsum(b, 1);
    let cp, g10 = scientific::nancumprod(b, 0);
    assert(g9.kind == scientific::F_NONE && close_all(cs, [1.0, 1.0, 4.0, 2.0, 7.0, 7.0]));
    assert(g10.kind == scientific::F_NONE && close_all(cp, [1.0, 1.0, 3.0, 2.0, 5.0, 3.0]));
    // The running forms of the array API.
    let q = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let ci, g11 = scientific::cumulative_sum(q, 1, true);
    assert(g11.kind == scientific::F_NONE && ci.shape[1] == 4 && close_all(ci, [0.0, 1.0, 3.0, 6.0, 0.0, 4.0, 9.0, 15.0]));
    let cq, g12 = scientific::cumulative_prod(q, 0, true);
    assert(g12.kind == scientific::F_NONE && cq.shape[0] == 3 && close_all(cq, [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 10.0, 18.0]));
    let cn, g13 = scientific::cumulative_sum(q, 0);
    assert(g13.kind == scientific::F_NONE && close_all(cn, [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]));
    // The differences with the ends.
    var ev: f64[>..] = [1.0, 4.0, 9.0, 16.0];
    let e = scientific::Arr { ndim: 1, shape: [4], data: copy(ev) };
    let ed, g14 = scientific::ediff1d(e, [-1.0], [100.0, 200.0]);
    assert(g14.kind == scientific::F_NONE && close_all(ed, [-1.0, 3.0, 5.0, 7.0, 100.0, 200.0]));
    let one = scientific::Arr { ndim: 1, shape: [1], data: [5.0] };
    let e2, g15 = scientific::ediff1d(one);
    assert(g15.kind == scientific::F_SHAPE);
    // The edges of the bins.
    let he, g16 = scientific::histogram_bin_edges(e, 3);
    let he2, g17 = scientific::histogram_edges(e, 3);
    assert(g16.kind == scientific::F_NONE && close_all(he, he2.data));
}

fn test_sections() {
    // The values come from SciPy 1.18.1.
    let s3, f1 = scientific::butter_sos(3, [0.2], "lowpass");
    assert(f1.kind == scientific::F_NONE && s3.ndim == 2 && s3.shape[0] == 2 && s3.shape[1] == 6);
    assert(close_row(s3, 0, [0.018098933007514428, 0.036197866015028855, 0.018098933007514428, 1.0, -0.5095254494944288, 0.0]));
    assert(close_row(s3, 6, [1.0, 1.0, 0.0, 1.0, -1.2505164308487402, 0.5457233155094579]));
    let s2, f2 = scientific::butter_sos(2, [0.1, 0.4], "bandpass");
    assert(f2.kind == scientific::F_NONE && s2.shape[0] == 2);
    assert(close_row(s2, 0, [0.13110643991662602, 0.26221287983325203, 0.13110643991662602, 1.0, -0.578864087474264, 0.3841299062257851]));
    assert(close_row(s2, 6, [1.0, -2.0, 1.0, 1.0, -1.6017937511285352, 0.7086533318887278]));
    let s4, f3 = scientific::butter_sos(4, [0.2], "lowpass");
    assert(f3.kind == scientific::F_NONE);
    // The state and the zero phase filter of the sections.
    let zi, f4 = scientific::sosfilt_zi(s4);
    assert(f4.kind == scientific::F_NONE && zi.shape[0] == 2 && zi.shape[1] == 2);
    assert(close_row(zi, 0, [0.07313199715874633, -0.018261675197028267, 0.9220436594835375, -0.5547824523688142]));
    var x: f64[>..] = [];
    var i: i64 = 0;
    while i < 60 { x.push(mathshim::sine(0.3 * i) + 0.1 * i); i += 1; }
    let y, f5 = scientific::sosfiltfilt(s4, x);
    assert(f5.kind == scientific::F_NONE && y.data.len == 60);
    assert(close(y.data[0], -8.176916419887215e-05) && close(y.data[5], 1.4954761292656586));
    assert(close(y.data[30], 3.4110733197288363) && close(y.data[59], 4.994258219420487));
    // The sections from the zeros, the poles and the gain.
    let zs, f6 = scientific::zpk2sos([0.2, 0.2, -0.7, 0.4], [0.6, -0.6, 0.0, 0.0], [0.5, 0.5, -0.3, -0.3], [0.5, -0.5, 0.4, -0.4], 0.8);
    assert(f6.kind == scientific::F_NONE && zs.shape[0] == 2);
    assert(close_row(zs, 0, [0.8, 0.24, -0.224, 1.0, 0.6, 0.25]));
    assert(close_row(zs, 6, [1.0, -0.4, 0.4, 1.0, -1.0, 0.5]));
    // The sections from the numerator and the denominator.
    let ts, f7 = scientific::tf2sos([0.8, -0.08, 0.0, 0.1856, -0.0896], [1.0, -0.4, 0.15, 0.05, 0.125]);
    assert(f7.kind == scientific::F_NONE && ts.shape[0] == 2);
    assert(close_row(ts, 0, [0.8, 0.24, -0.224, 1.0, 0.6, 0.25]));
    assert(close_row(ts, 6, [1.0, -0.4, 0.4, 1.0, -1.0, 0.5]));
    // The faults.
    let e1, g1 = scientific::butter_sos(0, [0.2], "lowpass");
    assert(g1.kind == scientific::F_VALUE);
    let e2, g2 = scientific::butter_sos(2, [1.5], "lowpass");
    assert(g2.kind == scientific::F_RANGE);
    let e3, g3 = scientific::zpk2sos([0.2], [0.6], [0.5], [0.0], 1.0);
    assert(g3.kind == scientific::F_VALUE);
    let e4, g4 = scientific::zpk2sos([0.2], [], [0.5], [0.0], 1.0);
    assert(g4.kind == scientific::F_SHAPE);
    let e5, g5 = scientific::sosfiltfilt(s4, x[..5]);
    assert(g5.kind == scientific::F_SHAPE);
    let e6, g6 = scientific::tf2sos([1.0], [0.0, 1.0]);
    assert(g6.kind == scientific::F_VALUE);
}

fn test_polynomial() {
    // The values come from numpy.polynomial.Polynomial.
    let p = scientific::polynomial([1.0, 2.0, 3.0]);
    assert(scientific::polynomial_call(p, 2.0) == 17.0);
    assert(scientific::polynomial_degree(p) == 2);
    assert(scientific::polynomial_degree(scientific::polynomial([1.0, 2.0, 0.0])) == 1);
    let q = scientific::polynomial([1.0, 1.0]);
    let a = scientific::polynomial_add(p, q);
    let b = scientific::polynomial_sub(p, q);
    let m = scientific::polynomial_mul(p, q);
    assert(a.coe[0] == 2.0 && a.coe[2] == 3.0);
    assert(b.coe[0] == 0.0 && b.coe[2] == 3.0);
    assert(m.coe.len == 4 && m.coe[1] == 3.0 && m.coe[3] == 3.0);
    let quo, rem, f = scientific::polynomial_divmod(scientific::polynomial([1.0, 3.0, 5.0, 3.0]), q);
    assert(f.kind == scientific::F_NONE);
    assert(quo.coe.len == 3 && close(quo.coe[2], 3.0) && rem.coe.len == 1 && rem.coe[0] == 0.0);
    let d1, f1 = scientific::polynomial_der(p, 1);
    let d2, f2 = scientific::polynomial_der(p, 2);
    assert(f1.kind == scientific::F_NONE && d1.coe.len == 2 && d1.coe[1] == 6.0);
    assert(d2.coe.len == 1 && d2.coe[0] == 6.0);
    let ig = scientific::polynomial_integ(p);
    assert(ig.coe.len == 4 && ig.coe[0] == 0.0 && close(ig.coe[3], 1.0));
    let rp, rpi, fr = scientific::polynomial_roots(scientific::polynomial([-1.0, 0.0, 1.0]));
    assert(fr.kind == scientific::F_NONE);
    assert((close(rp.data[0], -1.0) && close(rp.data[1], 1.0)) || (close(rp.data[0], 1.0) && close(rp.data[1], -1.0)));
    assert(close(rpi.data[0], 0.0) && close(rpi.data[1], 0.0));
    let frp = scientific::polynomial_fromroots([1.0, -1.0]);
    assert(close(frp.coe[0], -1.0) && frp.coe[1] == 0.0 && close(frp.coe[2], 1.0));
    let pw, f3 = scientific::polynomial_pow(q, 2);
    let pw3, f4 = scientific::polynomial_pow(scientific::polynomial([0.0, 1.0]), 3);
    assert(f3.kind == scientific::F_NONE && pw.coe.len == 3 && close(pw.coe[1], 2.0));
    assert(pw3.coe.len == 4 && close(pw3.coe[3], 1.0));
    let bad, f5 = scientific::polynomial_pow(p, -1);
    assert(f5.kind == scientific::F_RANGE);
    let cd, f6 = scientific::polynomial_cutdeg(scientific::polynomial([1.0, 2.0, 3.0, 4.0]), 1);
    assert(f6.kind == scientific::F_NONE && cd.coe.len == 2 && cd.coe[1] == 2.0);
    let fp, f7 = scientific::polynomial_fit([0.0, 1.0, 2.0], [1.0, 3.0, 7.0], 2);
    assert(f7.kind == scientific::F_NONE);
    assert(close(fp.coe[0], 1.0) && close(fp.coe[1], 1.0) && close(fp.coe[2], 1.0));
    let xs, ys, f8 = scientific::polynomial_linspace(scientific::polynomial([0.0, 1.0]), -1.0, 1.0, 5);
    assert(f8.kind == scientific::F_NONE && xs.data.len == 5);
    assert(close(xs.data[1], -0.5) && close(ys.data[1], -0.5) && close(ys.data[4], 1.0));
    let noq, nor, f9 = scientific::polynomial_divmod(p, scientific::polynomial([0.0]));
    assert(f9.kind == scientific::F_VALUE);
    let noc, f10 = scientific::polynomial_cutdeg(p, -1);
    assert(f10.kind == scientific::F_RANGE);
}

fn test_filters() {
    // The values come from SciPy 1.18.1.
    let b, a, f = scientific::butter(3, [0.2], "lowpass");
    assert(f.kind == scientific::F_NONE);
    assert(close(b.data[0], 0.018098933007514428) && close(b.data[1], 0.05429679902254328));
    assert(close(a.data[1], -1.7600418803431688) && close(a.data[2], 1.182893262037831) && close(a.data[3], -0.27805991763454646));
    let hb, ha, f2 = scientific::butter(2, [0.3], "highpass");
    assert(f2.kind == scientific::F_NONE);
    assert(close(hb.data[0], 0.5050010290458776) && close(hb.data[1], -1.0100020580917553));
    assert(close(ha.data[1], -0.7477891782585034) && close(ha.data[2], 0.27221493792500717));
    let bb, ba, f3 = scientific::butter(2, [0.1, 0.4], "bandpass");
    assert(f3.kind == scientific::F_NONE);
    assert(bb.data.len == 5 && close(bb.data[0], 0.13110643991662596) && close(bb.data[2], -0.2622128798332519));
    let sb, sa, f4 = scientific::butter(2, [0.1, 0.4], "bandstop");
    assert(f4.kind == scientific::F_NONE);
    assert(close(sb.data[1], -1.6030843431581088) && close(sa.data[1], -2.1806578386027993) && close(sa.data[3], -1.025510847713418));
    var x: f64[>..] = [];
    var i: i64 = 0;
    while i < 40 { x.push(mathshim::sine(0.3 * i) + 0.1 * i); i += 1; }
    let y, f5 = scientific::lfilter(b.data, a.data, x);
    assert(f5.kind == scientific::F_NONE && y.data.len == 40);
    assert(close(y.data[0], 0.0) && close(y.data[5], 0.6242264193888316) && close(y.data[39], 2.6324122648517223));
    let w, f6 = scientific::filtfilt(b.data, a.data, x);
    assert(f6.kind == scientific::F_NONE && w.data.len == 40);
    assert(close(w.data[0], -0.0016527642520426794) && close(w.data[5], 1.4879722866741751));
    assert(close(w.data[20], 1.7229739986405581) && close(w.data[39], 3.1457717019846685));
    let zi, f7 = scientific::lfilter_zi(b.data, a.data);
    assert(f7.kind == scientific::F_NONE && zi.data.len == 3);
    assert(close(zi.data[0], 0.9819010669924851) && close(zi.data[2], 0.29615885064206077));
    let sos: f64[>..] = [0.0674552738890719, 0.1349105477781438, 0.0674552738890719, 1.0, -1.1429805025399011, 0.41280159809618877];
    let q = scientific::Arr { ndim: 2, shape: [1, 6], data: copy(sos) };
    let z, f8 = scientific::sosfilt(q, x);
    assert(f8.kind == scientific::F_NONE);
    assert(close(z.data[5], 0.9284098237835398) && close(z.data[39], 2.705008195057858));
    let hr, hi, f9 = scientific::hilbert(x[..7]);
    assert(f9.kind == scientific::F_NONE);
    assert(close(hr.data[3], x[3]) && close(hi.data[0], 0.731648828187718) && close(hi.data[6], 0.9299383059240987));
    // The faults.
    let e1, e2, g1 = scientific::butter(0, [0.2], "lowpass");
    assert(g1.kind == scientific::F_VALUE);
    let e3, e4, g2 = scientific::butter(2, [1.0], "lowpass");
    assert(g2.kind == scientific::F_RANGE);
    let e5, e6, g3 = scientific::butter(2, [0.2], "bandpass");
    assert(g3.kind == scientific::F_SHAPE);
    let e7, e8, g4 = scientific::butter(2, [0.4, 0.1], "bandstop");
    assert(g4.kind == scientific::F_RANGE);
    let e9, g5 = scientific::lfilter([1.0], [0.0], x);
    assert(g5.kind == scientific::F_VALUE);
    let e10, g6 = scientific::filtfilt(b.data, a.data, x[..5]);
    assert(g6.kind == scientific::F_SHAPE);
}

fn test_constants() {
    // SciPy 1.18.1 holds the values of CODATA 2022.
    assert(constants::c == 299792458.0);
    assert(constants::pi == 3.141592653589793);
    assert(constants::golden == 1.618033988749895);
    assert(constants::kilo == 1000.0 && constants::milli == 0.001 && constants::mega == 1000000.0);
    assert(constants::hour == 3600.0 && constants::day == 86400.0 && constants::week == 604800.0);
    assert(constants::inch == 0.0254 && constants::foot == 0.30479999999999996 && constants::mile == 1609.3439999999998);
    assert(constants::zero_Celsius == 273.15 && constants::atm == 101325.0);
    assert(constants::h == 6.62607015e-34 && constants::N_A == 6.02214076e+23 && constants::k == 1.380649e-23);
    assert(constants::m_e == 9.1093837139e-31 && constants::e == 1.602176634e-19);
    assert(constants::G == 6.6743e-11 && constants::g == 9.80665);
    // The table of the physical constants: the value, the unit and the
    // relative uncertainty by name.
    let me, fme = scientific::value("electron mass");
    assert(fme.kind == scientific::F_NONE && me == 9.1093837139e-31);
    let un, fun = scientific::unit("electron mass");
    assert(fun.kind == scientific::F_NONE && un == "kg");
    let vu, fvu = scientific::unit("fine-structure constant");
    assert(fvu.kind == scientific::F_NONE && vu.len == 0);
    let pr, fpr = scientific::precision("electron mass");
    assert(fpr.kind == scientific::F_NONE && pr > 3.0e-10 && pr < 3.2e-10);
    // The precision keeps the sign of the value, as SciPy gives it.
    let pg, fpg = scientific::precision("electron g factor");
    assert(fpg.kind == scientific::F_NONE && pg == -1.7979150438990604e-13);
    let _v, bad = scientific::value("no such constant");
    assert(bad.kind == scientific::F_VALUE);
    let _w, bad2 = scientific::value("Electron Mass");
    assert(bad2.kind == scientific::F_VALUE);
    let _u, bad3 = scientific::unit("no such constant");
    assert(bad3.kind == scientific::F_VALUE);
    let _x, bad4 = scientific::precision("no such constant");
    assert(bad4.kind == scientific::F_VALUE);
    // The search ignores the case and gives sorted names, one per line.
    let hits = scientific::find("ELECTRON MASS");
    assert(hits.len > 0 && hits[0] == 97);
    var lines: i64 = 1;
    var q: i64 = 0;
    while q < hits.len { if hits[q] == 10 { lines += 1; } q += 1; }
    assert(lines == 12);
    let none = scientific::find("zzzz");
    assert(none.len == 0);
    // The scales of temperature: 100 C is 212 F and 373.15 K. The short
    // names and the case do not matter.
    let t, ft = scientific::from_values([100.0, 0.0, -40.0]);
    assert(ft.kind == scientific::F_NONE);
    let f, ff = scientific::convert_temperature(t, "Celsius", "Fahrenheit");
    assert(ff.kind == scientific::F_NONE && f.data[0] == 212.0 && f.data[1] == 32.0 && f.data[2] == -40.0);
    let k, fk = scientific::convert_temperature(t, "c", "KELVIN");
    assert(fk.kind == scientific::F_NONE && k.data[0] == 373.15 && k.data[1] == 273.15);
    let r, fr = scientific::convert_temperature(t, "K", "R");
    assert(fr.kind == scientific::F_NONE && r.data[0] == 180.0);
    let _bt, fbt = scientific::convert_temperature(t, "X", "K");
    assert(fbt.kind == scientific::F_VALUE);
    // Wavelength and frequency: 500 nm is 599584916000000 Hz in SciPy.
    let l, fl = scientific::from_values([500.0e-9]);
    assert(fl.kind == scientific::F_NONE);
    let nu, fnu = scientific::lambda2nu(l);
    assert(fnu.kind == scientific::F_NONE && nu.data[0] == 299792458.0 / 500.0e-9);
    let back, fback = scientific::nu2lambda(nu);
    assert(fback.kind == scientific::F_NONE && back.data[0] == 299792458.0 / nu.data[0]);
}

fn test_moves() {
    // NumPy 2.5.3 values. The cube holds 0 to 23 in the shape 2, 3, 4.
    let flat, ff = scientific::arange(0.0, 24.0, 1.0);
    assert(ff.kind == scientific::F_NONE);
    let cube, fc = scientific::reshape(flat, [2, 3, 4]);
    assert(fc.kind == scientific::F_NONE);
    let m1, f1 = scientific::moveaxis(cube, 0, -1);
    assert(f1.kind == scientific::F_NONE && m1.shape[0] == 3 && m1.shape[1] == 4 && m1.shape[2] == 2);
    let m2, f2 = scientific::moveaxis(cube, -1, 0);
    assert(f2.kind == scientific::F_NONE && m2.shape[0] == 4 && m2.shape[1] == 2 && m2.shape[2] == 3);
    assert(m2.data[1 * 6 + 0 * 3 + 0] == 1.0 && m2.data[1 * 6 + 0 * 3 + 1] == 5.0 && m2.data[1 * 6 + 0 * 3 + 2] == 9.0);
    let m3, f3 = scientific::moveaxis_many(cube, [0, 1], [-1, -2]);
    assert(f3.kind == scientific::F_NONE && m3.shape[0] == 4 && m3.shape[1] == 3 && m3.shape[2] == 2);
    let _m4, f4 = scientific::moveaxis(cube, 3, 0);
    assert(f4.kind == scientific::F_RANGE);
    let _m5, f5 = scientific::moveaxis_many(cube, [0, 0], [1, 2]);
    assert(f5.kind == scientific::F_VALUE);
    // The flips and the turns of a matrix of two rows and three columns.
    let mat, fm = scientific::from_rows(2, 3, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert(fm.kind == scientific::F_NONE);
    let ud, fud = scientific::flipud(mat);
    assert(fud.kind == scientific::F_NONE && ud.data[0] == 3.0 && ud.data[3] == 0.0);
    let lr, flr = scientific::fliplr(mat);
    assert(flr.kind == scientific::F_NONE && lr.data[0] == 2.0 && lr.data[3] == 5.0);
    let r1, fr1 = scientific::rot90(mat);
    assert(fr1.kind == scientific::F_NONE && r1.shape[0] == 3 && r1.shape[1] == 2);
    assert(r1.data[0] == 2.0 && r1.data[1] == 5.0 && r1.data[2] == 1.0 && r1.data[5] == 3.0);
    let r2, fr2 = scientific::rot90(mat, 2);
    assert(fr2.kind == scientific::F_NONE && r2.data[0] == 5.0 && r2.data[5] == 0.0);
    let r3, fr3 = scientific::rot90(mat, 3);
    assert(fr3.kind == scientific::F_NONE && r3.data[0] == 3.0 && r3.data[1] == 0.0 && r3.data[5] == 2.0);
    let rm, frm = scientific::rot90(mat, -1);
    assert(frm.kind == scientific::F_NONE && rm.data[0] == 3.0 && rm.data[1] == 0.0);
    let r0, fr0 = scientific::rot90(mat, 0);
    assert(fr0.kind == scientific::F_NONE && r0.data[0] == 0.0 && r0.data[5] == 5.0);
    let rc, frc = scientific::rot90(cube);
    assert(frc.kind == scientific::F_NONE && rc.shape[0] == 3 && rc.shape[1] == 2 && rc.shape[2] == 4);
    let rd, frd = scientific::rot90(cube, 1, 1, 2);
    assert(frd.kind == scientific::F_NONE && rd.shape[0] == 2 && rd.shape[1] == 4 && rd.shape[2] == 3);
    let vec, fvec = scientific::from_values([1.0, 2.0]);
    assert(fvec.kind == scientific::F_NONE);
    let _rv, frv = scientific::rot90(vec);
    assert(frv.kind == scientific::F_SHAPE);
    let _lv, flv = scientific::fliplr(vec);
    assert(flv.kind == scientific::F_SHAPE);
    // The broadcast to a shape, and of two arrays.
    let row, frow = scientific::from_values([1.0, 2.0, 3.0]);
    assert(frow.kind == scientific::F_NONE);
    let bt, fbt = scientific::broadcast_to(row, [2, 3]);
    assert(fbt.kind == scientific::F_NONE && bt.shape[0] == 2 && bt.data[3] == 1.0 && bt.data[5] == 3.0);
    let _bad, fbad = scientific::broadcast_to(vec, [3]);
    assert(fbad.kind == scientific::F_SHAPE);
    let _bad2, fbad2 = scientific::broadcast_to(mat, [3]);
    assert(fbad2.kind == scientific::F_SHAPE);
    let col, fcol = scientific::from_rows(2, 1, [1.0, 2.0]);
    let wide, fwide = scientific::from_values([10.0, 20.0, 30.0]);
    assert(fcol.kind == scientific::F_NONE && fwide.kind == scientific::F_NONE);
    let bx, by, fba = scientific::broadcast_arrays(col, wide);
    assert(fba.kind == scientific::F_NONE && bx.shape[0] == 2 && bx.shape[1] == 3);
    assert(bx.data[0] == 1.0 && bx.data[3] == 2.0 && by.data[0] == 10.0 && by.data[5] == 30.0);
    // The choice: NumPy gives [1, 5, -1] for these two conditions.
    let conds, fcn = scientific::from_rows(2, 3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let picks, fpk = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert(fcn.kind == scientific::F_NONE && fpk.kind == scientific::F_NONE);
    let sel, fsel = scientific::select(conds, picks, -1.0);
    assert(fsel.kind == scientific::F_NONE && sel.ndim == 1 && sel.data[0] == 1.0 && sel.data[1] == 5.0 && sel.data[2] == -1.0);
    let _sel2, fsel2 = scientific::select(conds, row, -1.0);
    assert(fsel2.kind == scientific::F_SHAPE);
    // The function along an axis: NumPy gives sums [3, 12] on axis 1,
    // the reversed columns on axis 0, and the bounds [[0, 2], [3, 5]].
    let sums, fsu = scientific::apply_along_axis_scalar(mat, 1, line_sum);
    assert(fsu.kind == scientific::F_NONE && sums.ndim == 1 && sums.data[0] == 3.0 && sums.data[1] == 12.0);
    let rev, frev = scientific::apply_along_axis(mat, 0, line_reverse);
    assert(frev.kind == scientific::F_NONE && rev.shape[0] == 2 && rev.data[0] == 3.0 && rev.data[2] == 5.0 && rev.data[3] == 0.0);
    let bnd, fbn = scientific::apply_along_axis(mat, 1, line_bounds);
    assert(fbn.kind == scientific::F_NONE && bnd.shape[0] == 2 && bnd.shape[1] == 2);
    assert(bnd.data[0] == 0.0 && bnd.data[1] == 2.0 && bnd.data[2] == 3.0 && bnd.data[3] == 5.0);
    let _ax, fax = scientific::apply_along_axis(mat, 2, line_reverse);
    assert(fax.kind == scientific::F_RANGE);
}

fn test_fftn() {
    // NumPy 2.5.3: fftshift([0,1,2]) is [2,0,1] and ifftshift is [1,2,0].
    let v, fv = scientific::from_values([0.0, 1.0, 2.0]);
    assert(fv.kind == scientific::F_NONE);
    let s, fs = scientific::fftshift(v);
    assert(fs.kind == scientific::F_NONE && s.data[0] == 2.0 && s.data[1] == 0.0 && s.data[2] == 1.0);
    let t, ft = scientific::ifftshift(v);
    assert(ft.kind == scientific::F_NONE && t.data[0] == 1.0 && t.data[1] == 2.0 && t.data[2] == 0.0);
    // The shift of a matrix moves both axes. For [[0,1,2],[3,4,5]],
    // NumPy gives [[5,3,4],[2,0,1]].
    let m, fm = scientific::from_rows(2, 3, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert(fm.kind == scientific::F_NONE);
    let sm, fsm = scientific::fftshift(m);
    assert(fsm.kind == scientific::F_NONE && sm.data[0] == 5.0 && sm.data[1] == 3.0 && sm.data[2] == 4.0 && sm.data[3] == 2.0);
    let tm, ftm = scientific::ifftshift(sm);
    assert(ftm.kind == scientific::F_NONE && tm.data[0] == 0.0 && tm.data[5] == 5.0);
    // The transform of many axes: fftn of [[1,2],[3,4]] is [[10,-2],[-4,0]].
    let a, fa = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let z, fz = scientific::from_rows(2, 2, [0.0, 0.0, 0.0, 0.0]);
    assert(fa.kind == scientific::F_NONE && fz.kind == scientific::F_NONE);
    let fr, fi, ff = scientific::fftn(a, z);
    assert(ff.kind == scientific::F_NONE && fr.data[0] == 10.0 && fr.data[1] == -2.0 && fr.data[2] == -4.0 && fr.data[3] == 0.0);
    assert(fi.data[0] == 0.0 && fi.data[3] == 0.0);
    // The inverse gives the input back.
    let br, bi, fb = scientific::ifftn(fr, fi);
    assert(fb.kind == scientific::F_NONE);
    var k: i64 = 0;
    while k < 4 {
        var d = br.data[k] - a.data[k];
        if d < 0.0 { d = -d; }
        assert(d < 1.0e-12 && bi.data[k] < 1.0e-12 && bi.data[k] > -1.0e-12);
        k += 1;
    }
    let g2r, g2i, f2 = scientific::ifft2(fr, fi);
    assert(f2.kind == scientific::F_NONE && g2r.data[3] > 3.999999999 && g2r.data[3] < 4.000000001);
    let c2r, c2i, fc2 = scientific::fft2c(a, z);
    assert(fc2.kind == scientific::F_NONE && c2r.data[0] == 10.0 && c2i.data[1] == 0.0);
    // A transform of three axes, and the shapes that disagree.
    let cube, fcube = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert(fcube.kind == scientific::F_NONE);
    let c3, fc3 = scientific::reshape(cube, [2, 2, 2]);
    let z3, fz3 = scientific::zeros([2, 2, 2]);
    assert(fc3.kind == scientific::F_NONE && fz3.kind == scientific::F_NONE);
    let r3, i3, f3 = scientific::fftn(c3, z3);
    assert(f3.kind == scientific::F_NONE && r3.data[0] == 36.0 && r3.data[7] == 0.0);
    let _x, _y, bad = scientific::fftn(a, z3);
    assert(bad.kind == scientific::F_SHAPE);
    let _x2, _y2, bad2 = scientific::ifft2(c3, z3);
    assert(bad2.kind == scientific::F_SHAPE);
    // The frequencies of the real transform: NumPy 2.5.3 gives
    // rfftfreq(8, 1) = [0, 0.125, 0.25, 0.375, 0.5].
    let q, fq = scientific::rfftfreq(8, 1.0);
    assert(fq.kind == scientific::F_NONE && q.data.len == 5 && q.data[1] == 0.125 && q.data[4] == 0.5);
    let q7, fq7 = scientific::rfftfreq(7, 0.5);
    assert(fq7.kind == scientific::F_NONE && q7.data.len == 4);
    let _q0, bq = scientific::rfftfreq(0, 1.0);
    assert(bq.kind == scientific::F_VALUE);
    let _q1, bq2 = scientific::rfftfreq(4, 0.0);
    assert(bq2.kind == scientific::F_VALUE);
}

fn test_pinv() {
    // NumPy 2.5.3: the pseudo-inverse of a tall matrix, of a matrix
    // of rank one, of zeros and of a wide matrix.
    let a, fa = scientific::from_rows(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert(fa.kind == scientific::F_NONE);
    let p, fp = scientific::pinv(a);
    assert(fp.kind == scientific::F_NONE && p.shape[0] == 2 && p.shape[1] == 3);
    let want = [-1.3333333333333337, -0.33333333333333287, 0.6666666666666665, 1.0833333333333335, 0.3333333333333329, -0.4166666666666665];
    var i: i64 = 0;
    while i < 6 {
        var diff = p.data[i] - want[i];
        if diff < 0.0 { diff = -diff; }
        assert(diff < 1.0e-12);
        i += 1;
    }
    let b, fb = scientific::from_rows(2, 2, [1.0, 2.0, 2.0, 4.0]);
    assert(fb.kind == scientific::F_NONE);
    let pb, fpb = scientific::pinv(b);
    assert(fpb.kind == scientific::F_NONE);
    let wb = [0.04, 0.08, 0.08, 0.16];
    i = 0;
    while i < 4 {
        var diff = pb.data[i] - wb[i];
        if diff < 0.0 { diff = -diff; }
        assert(diff < 1.0e-12);
        i += 1;
    }
    let z, fz = scientific::from_rows(2, 3, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert(fz.kind == scientific::F_NONE);
    let pz, fpz = scientific::pinv(z);
    assert(fpz.kind == scientific::F_NONE && pz.shape[0] == 3 && pz.shape[1] == 2 && pz.data[5] == 0.0);
    // A singular value under the cutoff counts as zero.
    let c, fc = scientific::from_rows(2, 2, [1.0, 0.0, 0.0, 1.0e-20]);
    assert(fc.kind == scientific::F_NONE);
    let pc, fpc = scientific::pinv(c);
    assert(fpc.kind == scientific::F_NONE && pc.data[0] == 1.0 && pc.data[3] == 0.0);
    let pc2, fpc2 = scientific::pinv(c, 1.0e-30);
    assert(fpc2.kind == scientific::F_NONE && pc2.data[3] > 9.0e19);
    // The product of a matrix, its inverse and the matrix gives the matrix.
    let pa, fpa = scientific::matmul(a, p);
    assert(fpa.kind == scientific::F_NONE);
    let back, fback = scientific::matmul(pa, a);
    assert(fback.kind == scientific::F_NONE);
    i = 0;
    while i < 6 {
        var diff = back.data[i] - a.data[i];
        if diff < 0.0 { diff = -diff; }
        assert(diff < 1.0e-10);
        i += 1;
    }
    // A matrix whose first row is zero still gives the values from the
    // largest to the smallest, and the rank of NumPy. NumPy 2.5.3
    // gives the values 2519.1989127, 1136.8725513, 0 and the rank 2.
    let zr, fzr = scientific::from_rows(3, 3, [0.0, 0.0, 0.0, 610.6409992957679, 577.0754169394714, 2332.7120253825424, -336.52552992851207, -914.8298592004128, 735.687023006662]);
    assert(fzr.kind == scientific::F_NONE);
    let sz, fsz = scientific::svdvals(zr);
    assert(fsz.kind == scientific::F_NONE);
    assert(sz.data[0] > 2519.19 && sz.data[0] < 2519.20 && sz.data[1] > 1136.87 && sz.data[1] < 1136.88);
    assert(sz.data[2] < 1.0e-10);
    let rz, frz = scientific::matrix_rank(zr);
    assert(frz.kind == scientific::F_NONE && rz == 2);
    let pz2, fpz2 = scientific::pinv(zr);
    assert(fpz2.kind == scientific::F_NONE && pz2.data[0] == 0.0);
    let v, fv = scientific::from_values([1.0, 2.0]);
    assert(fv.kind == scientific::F_NONE);
    let _q, bad = scientific::pinv(v);
    assert(bad.kind == scientific::F_SHAPE);
}

fn test_buffer() {
    // NumPy 2.5.3: little endian numbers of 64 bits decode from bytes.
    var b: u8[>..] = [0, 0, 0, 0, 0, 0, 248, 63, 0, 0, 0, 0, 0, 0, 0, 192, 0, 0, 0, 0, 0, 0, 10, 64, 0, 0, 0, 0, 0, 0, 16, 64];
    let a, f = scientific::frombuffer(b);
    assert(f.kind == scientific::F_NONE && a.data.len == 4);
    assert(a.data[0] == 1.5 && a.data[1] == -2.0 && a.data[2] == 3.25 && a.data[3] == 4.0);
    let c, f2 = scientific::frombuffer(b, 2);
    assert(f2.kind == scientific::F_NONE && c.data.len == 2 && c.data[1] == -2.0);
    let r, f3 = scientific::frombuffer(b, -1, 8);
    assert(f3.kind == scientific::F_NONE && r.data.len == 3 && r.data[0] == -2.0);
    // An offset that no number holds, a rest of bytes and a count above
    // the rest are faults.
    let _x, bad = scientific::frombuffer(b, -1, 4);
    assert(bad.kind == scientific::F_VALUE);
    let _y, bad2 = scientific::frombuffer(b[..30]);
    assert(bad2.kind == scientific::F_VALUE);
    let _z, bad3 = scientific::frombuffer(b, 5);
    assert(bad3.kind == scientific::F_VALUE);
    let empty, f4 = scientific::frombuffer(b[..0]);
    assert(f4.kind == scientific::F_NONE && empty.data.len == 0);
    // The missing value, the zero below zero and the infinity keep
    // their meaning.
    var nb: u8[>..] = [0, 0, 0, 0, 0, 0, 248, 127, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 240, 127];
    let n, f5 = scientific::frombuffer(nb);
    assert(f5.kind == scientific::F_NONE && n.data[0] != n.data[0] && n.data[1] == 0.0 && 1.0 / n.data[1] < 0.0);
    assert(n.data[2] > 0.0 && n.data[2] == n.data[2] * 2.0);
    // The smallest numbers below the normal carry their value in the
    // digits alone.
    var sb: u8[>..] = [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
    let sm, f6 = scientific::frombuffer(sb);
    assert(f6.kind == scientific::F_NONE);
    assert(sm.data[0] == 4.9406564584124654e-324 && sm.data[1] == 2.0 * 4.9406564584124654e-324);
    // A file holds the same bytes. A count above the file takes what
    // the file holds.
    let ok = write_file("sci_f64_tmp.bin", b);
    assert(ok);
    let fa, ff = scientific::fromfile("sci_f64_tmp.bin");
    assert(ff.kind == scientific::F_NONE && fa.data.len == 4 && fa.data[3] == 4.0);
    let fc, ffc = scientific::fromfile("sci_f64_tmp.bin", 2);
    assert(ffc.kind == scientific::F_NONE && fc.data.len == 2);
    let fmore, ffmore = scientific::fromfile("sci_f64_tmp.bin", 9);
    assert(ffmore.kind == scientific::F_NONE && fmore.data.len == 4);
    // A file that does not hold a count of numbers drops the rest.
    let ok2 = write_file("sci_f64_tmp.bin", b[..20]);
    assert(ok2);
    let fp, ffp = scientific::fromfile("sci_f64_tmp.bin");
    assert(ffp.kind == scientific::F_NONE && fp.data.len == 2);
    let _gone, fgone = scientific::fromfile("sci_f64_no.bin");
    assert(fgone.kind == scientific::F_VALUE);
    assert(delete_file("sci_f64_tmp.bin"));
}

fn test_histogram2d() {
    // NumPy 2.5.3: the pairs fall in a grid of two bins on each axis.
    let x, fx = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    let y, fy = scientific::from_values([0.0, 5.0, 1.0, 2.0]);
    assert(fx.kind == scientific::F_NONE && fy.kind == scientific::F_NONE);
    let h, ex, ey, f = scientific::histogram2d(x, y, 2, 2);
    assert(f.kind == scientific::F_NONE && h.shape[0] == 2 && h.shape[1] == 2);
    assert(h.data[0] == 1.0 && h.data[1] == 1.0 && h.data[2] == 2.0 && h.data[3] == 0.0);
    assert(ex.data[0] == 1.0 && ex.data[1] == 2.5 && ex.data[2] == 4.0);
    assert(ey.data[0] == 0.0 && ey.data[1] == 2.5 && ey.data[2] == 5.0);
    // A value on the right edge of the range falls in the last bin.
    let e5, fe5 = scientific::from_values([1.0, 5.0]);
    let e3, fe3 = scientific::from_values([1.0, 2.0]);
    assert(fe5.kind == scientific::F_NONE && fe3.kind == scientific::F_NONE);
    let hr, _exr, _eyr, fr = scientific::histogram2d(e5, e3, 2, 2);
    assert(fr.kind == scientific::F_NONE && hr.data[0] == 1.0 && hr.data[1] == 0.0 && hr.data[2] == 0.0 && hr.data[3] == 1.0);
    // The inputs must hold the same count.
    let _h1, _e1, _e2, bad = scientific::histogram2d(x, e5, 2, 2);
    assert(bad.kind == scientific::F_SHAPE);
    let _h2, _e3, _e4, bad2 = scientific::histogram2d(x, y, 0, 2);
    assert(bad2.kind == scientific::F_VALUE);
    // The grid of points counts along every column.
    let pts, fpts = scientific::from_rows(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert(fpts.kind == scientific::F_NONE);
    let hd, ed, fd = scientific::histogramdd(pts, 2);
    assert(fd.kind == scientific::F_NONE && hd.ndim == 2 && hd.data.len == 4);
    assert(hd.data[0] == 1.0 && hd.data[1] == 0.0 && hd.data[2] == 0.0 && hd.data[3] == 2.0);
    assert(ed.shape[0] == 2 && ed.shape[1] == 3);
    assert(ed.data[0] == 1.0 && ed.data[2] == 5.0 && ed.data[3] == 2.0 && ed.data[5] == 6.0);
    // A column where every point holds one value is a fault, as the
    // histogram of one dimension faults too.
    let one, fone = scientific::from_rows(1, 3, [1.0, 2.0, 3.0]);
    assert(fone.kind == scientific::F_NONE);
    let _h1, _ed1, f1 = scientific::histogramdd(one, 2);
    assert(f1.kind == scientific::F_VALUE);
    // A row per dimension above the limit is a fault.
    let wide, fwide = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    assert(fwide.kind == scientific::F_NONE);
    let _h3, _ed3, fbad = scientific::histogramdd(wide, 0);
    assert(fbad.kind == scientific::F_VALUE);
    let same, fsame = scientific::from_rows(2, 2, [1.0, 5.0, 1.0, 6.0]);
    assert(fsame.kind == scientific::F_NONE);
    let _h4, _ed4, fbad2 = scientific::histogramdd(same, 2);
    assert(fbad2.kind == scientific::F_VALUE);
}

fn test_closeness() {
    // NumPy 2.5.3: the default sizes are a relative 1e-05 and an
    // absolute 1e-08.
    let a, fa = scientific::from_values([1.0e10, 1.0e-7, 1.0, 0.0]);
    let b, fb = scientific::from_values([1.00001e10, 1.0e-8, 1.0001, 0.0]);
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let c, fc = scientific::isclose(a, b);
    assert(fc.kind == scientific::F_NONE && c.data.len == 4);
    assert(c.data[0] == 1.0 && c.data[1] == 0.0 && c.data[2] == 0.0 && c.data[3] == 1.0);
    // The rule leans on the second number.
    let lo, flo = scientific::from_values([1.00001]);
    let one, fone = scientific::from_values([1.0]);
    assert(flo.kind == scientific::F_NONE && fone.kind == scientific::F_NONE);
    let n1, fn1 = scientific::isclose(lo, one, 1.0e-05, 0.0);
    let n2, fn2 = scientific::isclose(one, lo, 1.0e-05, 0.0);
    assert(fn1.kind == scientific::F_NONE && n1.data[0] == 0.0);
    assert(fn2.kind == scientific::F_NONE && n2.data[0] == 1.0);
    // The infinity equals itself. The missing value matches itself only
    // under the flag.
    let inf, finf = scientific::from_values([1.0 / 0.0, -1.0 / 0.0, 1.0 / 0.0, 0.0 / 0.0]);
    assert(finf.kind == scientific::F_NONE);
    let ie, fie = scientific::isclose(inf, inf);
    assert(fie.kind == scientific::F_NONE && ie.data[0] == 1.0 && ie.data[1] == 1.0 && ie.data[2] == 1.0);
    let nn, fnn = scientific::isclose(inf, inf, 1.0e-05, 1.0e-08, true);
    assert(fnn.kind == scientific::F_NONE && nn.data[3] == 1.0);
    let nm, fnm = scientific::isclose(inf, inf, 1.0e-05, 1.0e-08, false);
    assert(fnm.kind == scientific::F_NONE && nm.data[3] == 0.0);
    let pos, fpos = scientific::from_values([1.0 / 0.0]);
    let miss, fmiss = scientific::from_values([0.0 / 0.0]);
    assert(fpos.kind == scientific::F_NONE && fmiss.kind == scientific::F_NONE);
    let mixed, fmx = scientific::isclose(pos, miss);
    assert(fmx.kind == scientific::F_NONE && mixed.data[0] == 0.0);
    let all1, fall1 = scientific::allclose(a, b);
    assert(fall1.kind == scientific::F_NONE && all1 == false);
    let tw1, ftw1 = scientific::from_values([1.0, 2.0]);
    let tw2, ftw2 = scientific::from_values([1.0, 2.0]);
    assert(ftw1.kind == scientific::F_NONE && ftw2.kind == scientific::F_NONE);
    let ok, fok = scientific::allclose(tw1, tw2);
    assert(fok.kind == scientific::F_NONE && ok == true);
    // array_equal takes the shape and the missing value rule.
    let tri, ftri = scientific::from_values([1.0, 2.0, 3.0]);
    assert(ftri.kind == scientific::F_NONE);
    let e1, fe1 = scientific::array_equal(tw1, tri);
    assert(fe1.kind == scientific::F_NONE && e1 == false);
    let r, rr = scientific::from_rows(2, 2, [1.0, 0.0 / 0.0, 3.0, 4.0]);
    assert(rr.kind == scientific::F_NONE);
    let e2, fe2 = scientific::array_equal(r, r);
    assert(fe2.kind == scientific::F_NONE && e2 == false);
    let e3, fe3 = scientific::array_equal(r, r, true);
    assert(fe3.kind == scientific::F_NONE && e3 == true);
    let row12, frow12 = scientific::from_rows(1, 2, [1.0, 2.0]);
    assert(frow12.kind == scientific::F_NONE);
    let e4, fe4 = scientific::array_equal(tw1, row12);
    assert(fe4.kind == scientific::F_NONE && e4 == false);
}

fn test_positions() {
    // NumPy 2.5.3: argwhere and flatnonzero of a mask on a 3x3 grid.
    let a, fa = scientific::from_rows(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert(fa.kind == scientific::F_NONE);
    let m, fm = scientific::from_rows(3, 3, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    assert(fm.kind == scientific::F_NONE);
    let w, fw = scientific::argwhere(m);
    assert(fw.kind == scientific::F_NONE && w.shape[0] == 5 && w.shape[1] == 2);
    assert(w.data[0] == 0.0 && w.data[1] == 0.0 && w.data[2] == 0.0 && w.data[3] == 2.0);
    assert(w.data[4] == 1.0 && w.data[5] == 1.0 && w.data[6] == 2.0 && w.data[7] == 0.0);
    assert(w.data[8] == 2.0 && w.data[9] == 2.0);
    let flat, ff = scientific::flatnonzero(m);
    assert(ff.kind == scientific::F_NONE && flat.data.len == 5);
    assert(flat.data[0] == 0.0 && flat.data[1] == 2.0 && flat.data[2] == 4.0 && flat.data[3] == 6.0 && flat.data[4] == 8.0);
    let w2, fw2 = scientific::argwhere(a);
    assert(fw2.kind == scientific::F_NONE && w2.shape[0] == 9 && w2.shape[1] == 2);
    assert(w2.data[0] == 0.0 && w2.data[16] == 2.0 && w2.data[17] == 2.0);
    // A mask of one dimension gives one column.
    let v, fv = scientific::from_values([0.0, 3.0, 0.0, 5.0]);
    assert(fv.kind == scientific::F_NONE);
    let f2, ff2 = scientific::flatnonzero(v);
    assert(ff2.kind == scientific::F_NONE && f2.data.len == 2 && f2.data[0] == 1.0 && f2.data[1] == 3.0);
    let w3, fw3 = scientific::argwhere(v);
    assert(fw3.kind == scientific::F_NONE && w3.shape[0] == 2 && w3.shape[1] == 1);
    assert(w3.data[0] == 1.0 && w3.data[1] == 3.0);
    // No match gives zero rows.
    let z, fz = scientific::from_values([0.0, 0.0]);
    assert(fz.kind == scientific::F_NONE);
    let w4, fw4 = scientific::argwhere(z);
    assert(fw4.kind == scientific::F_NONE && w4.shape[0] == 0 && w4.shape[1] == 1 && w4.data.len == 0);
}

fn test_grids() {
    let x, fx = scientific::from_values([1.0, 2.0, 3.0]);
    let y, fy = scientific::from_values([10.0, 20.0]);
    assert(fx.kind == scientific::F_NONE && fy.kind == scientific::F_NONE);
    // The default indexing gives the shape (len(y), len(x)).
    let a, b, f = scientific::meshgrid(x, y);
    assert(f.kind == scientific::F_NONE && a.shape[0] == 2 && a.shape[1] == 3);
    assert(a.data[0] == 1.0 && a.data[2] == 3.0 && a.data[3] == 1.0 && a.data[5] == 3.0);
    assert(b.data[0] == 10.0 && b.data[2] == 10.0 && b.data[3] == 20.0 && b.data[5] == 20.0);
    // The ij indexing gives the shape (len(x), len(y)).
    let c, d, g = scientific::meshgrid(x, y, "ij");
    assert(g.kind == scientific::F_NONE && c.shape[0] == 3 && c.shape[1] == 2);
    assert(c.data[0] == 1.0 && c.data[1] == 1.0 && c.data[2] == 2.0 && c.data[5] == 3.0);
    assert(d.data[0] == 10.0 && d.data[1] == 20.0 && d.data[2] == 10.0 && d.data[5] == 20.0);
    // The sparse form holds one row and one column.
    let e, h, k = scientific::meshgrid(x, y, "xy", true);
    assert(k.kind == scientific::F_NONE && e.shape[0] == 1 && e.shape[1] == 3);
    assert(h.shape[0] == 2 && h.shape[1] == 1 && h.data[1] == 20.0);
    let e2, h2, k2 = scientific::meshgrid(x, y, "ij", true);
    assert(k2.kind == scientific::F_NONE && e2.shape[0] == 3 && e2.shape[1] == 1);
    assert(h2.shape[0] == 1 && h2.shape[1] == 2);
    // A bad indexing and an input of two dimensions are faults.
    let _p, _q, bad = scientific::meshgrid(x, y, "zz");
    assert(bad.kind == scientific::F_VALUE);
    let m, fm = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    assert(fm.kind == scientific::F_NONE);
    let _r, _s, bad2 = scientific::meshgrid(m, y);
    assert(bad2.kind == scientific::F_SHAPE);
    // The cross product of two vectors of three elements.
    let u, fu = scientific::from_values([1.0, 2.0, 3.0]);
    let v, fv = scientific::from_values([4.0, 5.0, 6.0]);
    assert(fu.kind == scientific::F_NONE && fv.kind == scientific::F_NONE);
    let w, fw = scientific::cross(u, v);
    assert(fw.kind == scientific::F_NONE && w.ndim == 1);
    assert(w.data[0] == -3.0 && w.data[1] == 6.0 && w.data[2] == -3.0);
    // Rows of vectors, and one vector against many rows.
    let ra, fra = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rb, frb = scientific::from_rows(2, 3, [7.0, 8.0, 9.0, 1.0, 2.0, 3.0]);
    assert(fra.kind == scientific::F_NONE && frb.kind == scientific::F_NONE);
    let rc, frc = scientific::cross(ra, rb);
    assert(frc.kind == scientific::F_NONE && rc.shape[0] == 2 && rc.shape[1] == 3);
    assert(rc.data[0] == -6.0 && rc.data[1] == 12.0 && rc.data[2] == -6.0);
    assert(rc.data[3] == 3.0 && rc.data[4] == -6.0 && rc.data[5] == 3.0);
    let rd, frd = scientific::cross(ra, v);
    assert(frd.kind == scientific::F_NONE && rd.shape[0] == 2 && rd.data[0] == -3.0);
    assert(rd.data[3] == 0.0 && rd.data[4] == 0.0 && rd.data[5] == 0.0);
    // A vector of two elements is a fault, as in NumPy 2.
    let t2, ft2 = scientific::from_values([1.0, 2.0]);
    assert(ft2.kind == scientific::F_NONE);
    let _z, bad3 = scientific::cross(t2, t2);
    assert(bad3.kind == scientific::F_SHAPE);
    let r3, fr3 = scientific::from_rows(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert(fr3.kind == scientific::F_NONE);
    let _z2, bad4 = scientific::cross(ra, r3);
    assert(bad4.kind == scientific::F_SHAPE);
}

fn test_stacks() {
    let a1, fa1 = scientific::from_values([1.0, 2.0]);
    let b1, fb1 = scientific::from_values([3.0, 4.0]);
    let a2, fa2 = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let b2, fb2 = scientific::from_rows(2, 2, [5.0, 6.0, 7.0, 8.0]);
    assert(fa1.kind == scientific::F_NONE && fb1.kind == scientific::F_NONE);
    assert(fa2.kind == scientific::F_NONE && fb2.kind == scientific::F_NONE);
    // The rows join under one another.
    let v1, fv1 = scientific::vstack(a1, b1);
    assert(fv1.kind == scientific::F_NONE && v1.ndim == 2 && v1.data.len == 4);
    assert(v1.data[0] == 1.0 && v1.data[1] == 2.0 && v1.data[2] == 3.0 && v1.data[3] == 4.0);
    let v2, fv2 = scientific::vstack(a2, b2);
    assert(fv2.kind == scientific::F_NONE && v2.ndim == 2 && v2.data.len == 8);
    assert(v2.shape[0] == 4 && v2.shape[1] == 2 && v2.data[4] == 5.0);
    // The rows join beside each other. A row of one dimension joins
    // along its own direction.
    let h1, fh1 = scientific::hstack(a1, b1);
    assert(fh1.kind == scientific::F_NONE && h1.ndim == 1 && h1.data.len == 4);
    assert(h1.data[0] == 1.0 && h1.data[2] == 3.0 && h1.data[3] == 4.0);
    let h2, fh2 = scientific::hstack(a2, b2);
    assert(fh2.kind == scientific::F_NONE && h2.ndim == 2 && h2.shape[1] == 4);
    assert(h2.data[0] == 1.0 && h2.data[2] == 5.0);
    // The planes join in a third dimension.
    let d1, fd1 = scientific::dstack(a1, b1);
    assert(fd1.kind == scientific::F_NONE && d1.ndim == 3 && d1.data.len == 4);
    assert(d1.data[0] == 1.0 && d1.data[1] == 3.0 && d1.data[2] == 2.0 && d1.data[3] == 4.0);
    let d2, fd2 = scientific::dstack(a2, b2);
    assert(fd2.kind == scientific::F_NONE && d2.ndim == 3 && d2.shape[2] == 2);
    assert(d2.data[0] == 1.0 && d2.data[1] == 5.0);
    // The columns join beside each other.
    let c1, fc1 = scientific::column_stack(a1, b1);
    assert(fc1.kind == scientific::F_NONE && c1.ndim == 2 && c1.data.len == 4);
    assert(c1.data[0] == 1.0 && c1.data[1] == 3.0 && c1.data[2] == 2.0 && c1.data[3] == 4.0);
    let c2, fc2 = scientific::column_stack(a2, b2);
    assert(fc2.kind == scientific::F_NONE && c2.ndim == 2 && c2.shape[1] == 4);
    assert(c2.data[0] == 1.0 && c2.data[2] == 5.0);
    // The stack takes the place of the new dimension.
    let s0, fs0 = scientific::stack(a2, b2, 0);
    let s1, fs1 = scientific::stack(a2, b2, 1);
    let s2, fs2 = scientific::stack(a2, b2, 2);
    let sm, fsm = scientific::stack(a2, b2, -1);
    assert(fs0.kind == scientific::F_NONE && fs1.kind == scientific::F_NONE);
    assert(fs2.kind == scientific::F_NONE && fsm.kind == scientific::F_NONE);
    assert(s0.data[0] == 1.0 && s0.data[4] == 5.0);
    assert(s1.data[0] == 1.0 && s1.data[2] == 5.0);
    assert(s2.data[0] == 1.0 && s2.data[1] == 5.0);
    assert(sm.data[1] == 5.0);
    // The default is the first dimension.
    let sd, fsd = scientific::stack(a2, b2);
    assert(fsd.kind == scientific::F_NONE && sd.data[0] == 1.0 && sd.data[4] == 5.0);
    // An axis outside the shape is a fault.
    let sb, fsb = scientific::stack(a2, b2, 3);
    assert(fsb.kind == scientific::F_RANGE && sb.data.len == 0);
    // Rows of different lengths hold no join.
    let cb, fcb = scientific::vstack(a1, scientific::from_values([1.0, 2.0, 3.0]));
    assert(fcb.kind == scientific::F_SHAPE && cb.data.len == 0);
    // The stacks join arrays of three dimensions in their own axes.
    let p, fp = scientific::reshape(scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), [2, 2, 2]);
    let q, fq = scientific::reshape(scientific::from_values([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]), [2, 2, 2]);
    assert(fp.kind == scientific::F_NONE && fq.kind == scientific::F_NONE);
    let d3, fd3 = scientific::dstack(p, q);
    assert(fd3.kind == scientific::F_NONE && d3.ndim == 3 && d3.shape[2] == 4);
    assert(d3.data[0] == 1.0 && d3.data[2] == 9.0 && d3.data[4] == 3.0);
    let v3, fv3 = scientific::vstack(p, q);
    assert(fv3.kind == scientific::F_NONE && v3.shape[0] == 4 && v3.shape[1] == 2);
    let h3, fh3 = scientific::hstack(p, q);
    assert(fh3.kind == scientific::F_NONE && h3.shape[1] == 4 && h3.shape[2] == 2);
    let c3, fc3 = scientific::column_stack(p, q);
    assert(fc3.kind == scientific::F_NONE && c3.shape[1] == 4 && c3.shape[2] == 2);
    // An array of the most dimensions holds no new dimension.
    let e8, fe8 = scientific::expand_dims(scientific::zeros([1, 1, 1, 1, 1, 1, 1, 2]), 0);
    assert(fe8.kind == scientific::F_LIMIT && e8.data.len == 0);
    // An array of one dimension below the most takes the new one.
    let e7, fe7 = scientific::expand_dims(scientific::zeros([1, 1, 1, 1, 1, 1, 1]), 7);
    assert(fe7.kind == scientific::F_NONE && e7.ndim == 8);
}

fn av(x: f64[:]) -> scientific::Arr {
    let a, f = scientific::from_values(x);
    assert(f.kind == scientific::F_NONE);
    return a;
}

// near tells whether two numbers agree to a relative tolerance.
fn near(a: f64, b: f64) -> bool {
    var d = a - b;
    if d < 0.0 { d = -d; }
    var s = b;
    if s < 0.0 { s = -s; }
    return d <= 0.000000001 * (1.0 + s);
}

// The rest of the elementwise math, pinned against NumPy.
fn test_ufunc() {
    let a = av([1.0, 0.0 / 0.0, -2.0, 3.0]);
    let b = av([0.0 / 0.0, 2.0, -5.0, 3.0]);
    let mx, g1 = scientific::fmax(a, b);
    assert(g1.kind == scientific::F_NONE);
    assert(mx.data[0] == 1.0 && mx.data[1] == 2.0 && mx.data[2] == -2.0 && mx.data[3] == 3.0);
    let mn, g2 = scientific::fmin(a, b);
    assert(g2.kind == scientific::F_NONE);
    assert(mn.data[0] == 1.0 && mn.data[1] == 2.0 && mn.data[2] == -5.0 && mn.data[3] == 3.0);
    let hv, g3 = scientific::heaviside(av([-1.5, 0.0, 2.0]), av([0.5]));
    assert(g3.kind == scientific::F_NONE);
    assert(hv.data[0] == 0.0 && hv.data[1] == 0.5 && hv.data[2] == 1.0);
    let la, g4 = scientific::logaddexp(av([1.0, -800.0]), av([2.0, -801.0]));
    assert(g4.kind == scientific::F_NONE);
    assert(near(la.data[0], 2.31326169) && near(la.data[1], -799.68673831));
    let l2, g5 = scientific::logaddexp2(av([1.0, 3.0]), av([2.0, 3.0]));
    assert(g5.kind == scientific::F_NONE);
    assert(near(l2.data[0], 2.5849625007211565) && near(l2.data[1], 4.0));
    let sc, g6 = scientific::sinc(av([0.0, 0.5, 1.5]));
    assert(g6.kind == scientific::F_NONE);
    assert(sc.data[0] == 1.0 && near(sc.data[1], 0.6366197723675814) && near(sc.data[2], -0.21220659078919377));
    let bi, g7 = scientific::i0(av([0.0, 1.0, 2.5]));
    assert(g7.kind == scientific::F_NONE);
    assert(bi.data[0] == 1.0 && near(bi.data[1], 1.2660658777520082) && near(bi.data[2], 3.2898391440501235));
    let fm, fe, g8 = scientific::frexp(av([8.0, -0.3, 0.0]));
    assert(g8.kind == scientific::F_NONE);
    assert(near(fm.data[0], 0.5) && near(fm.data[1], -0.6) && fm.data[2] == 0.0);
    assert(fe.data[0] == 4.0 && fe.data[1] == -1.0 && fe.data[2] == 0.0);
    let fr, ip, g9 = scientific::modf(av([3.7, -2.25]));
    assert(g9.kind == scientific::F_NONE);
    assert(near(fr.data[0], 0.7) && fr.data[1] == -0.25 && ip.data[0] == 3.0 && ip.data[1] == -2.0);
    let ld, g10 = scientific::ldexp(av([1.0, 3.0]), av([3.0, -2.0]));
    assert(g10.kind == scientific::F_NONE && ld.data[0] == 8.0 && ld.data[1] == 0.75);
    let ldb, g11 = scientific::ldexp(av([1.0]), av([0.5]));
    assert(g11.kind == scientific::F_VALUE && ldb.data.len == 0);
    // The functions of whole numbers.
    let g, g12 = scientific::gcd(av([12.0, -12.0]), av([18.0, 0.0]));
    assert(g12.kind == scientific::F_NONE && g.data[0] == 6.0 && g.data[1] == 12.0);
    let l, g13 = scientific::lcm(av([4.0, 0.0]), av([6.0, 3.0]));
    assert(g13.kind == scientific::F_NONE && l.data[0] == 12.0 && l.data[1] == 0.0);
    let x = av([12.0, 5.0]);
    let y = av([10.0, 3.0]);
    let an, g14 = scientific::bitwise_and(x, y);
    let bo, g15 = scientific::bitwise_or(x, y);
    let xo, g16 = scientific::bitwise_xor(x, y);
    assert(g14.kind == scientific::F_NONE && g15.kind == scientific::F_NONE && g16.kind == scientific::F_NONE);
    assert(an.data[0] == 8.0 && an.data[1] == 1.0 && bo.data[0] == 14.0 && bo.data[1] == 7.0);
    assert(xo.data[0] == 6.0 && xo.data[1] == 6.0);
    let inv, g17 = scientific::invert(av([0.0, 5.0]));
    assert(g17.kind == scientific::F_NONE && inv.data[0] == -1.0 && inv.data[1] == -6.0);
    let ls, g18 = scientific::left_shift(av([1.0, -3.0]), av([70.0, 2.0]));
    assert(g18.kind == scientific::F_NONE && ls.data[0] == 0.0 && ls.data[1] == -12.0);
    let rs, g19 = scientific::right_shift(av([-8.0, 8.0]), av([70.0, 1.0]));
    assert(g19.kind == scientific::F_NONE && rs.data[0] == -1.0 && rs.data[1] == 4.0);
    let bc, g20 = scientific::bitwise_count(av([-7.0, 255.0]));
    assert(g20.kind == scientific::F_NONE && bc.data[0] == 3.0 && bc.data[1] == 8.0);
    let bad, g21 = scientific::gcd(av([1.5]), av([2.0]));
    assert(g21.kind == scientific::F_VALUE && bad.data.len == 0);
    let big, g22 = scientific::invert(av([1.0e300]));
    assert(g22.kind == scientific::F_VALUE && big.data.len == 0);
    // The unwrap of a phase.
    let p = av([0.0, 1.5, 3.5, 6.5, 0.1, -3.0]);
    let u, g23 = scientific::unwrap(p);
    assert(g23.kind == scientific::F_NONE);
    assert(near(u.data[4], 6.383185307179586) && near(u.data[5], 3.283185307179586) && u.data[3] == 6.5);
    let m, gm = scientific::reshape(av([0.0, 3.5, 6.5, 0.5, 1.0, 7.5]), [3, 2]);
    assert(gm.kind == scientific::F_NONE);
    let m0, g24 = scientific::unwrap(m, 0.0 / 0.0, 0);
    assert(g24.kind == scientific::F_NONE);
    assert(near(m0.data[2], 0.21681469282041344) && near(m0.data[5], 1.2168146928204138));
    let m1, g25 = scientific::unwrap(m);
    assert(g25.kind == scientific::F_NONE);
    assert(near(m1.data[1], -2.7831853071795862) && near(m1.data[3], 6.783185307179586));
    let up, g26 = scientific::unwrap(av([0.0, 5.0, 10.0]), 0.0 / 0.0, -1, 4.0);
    assert(g26.kind == scientific::F_NONE && near(up.data[1], 1.0) && near(up.data[2], 2.0));
    let ub, g27 = scientific::unwrap(p, 0.0 / 0.0, 3);
    assert(g27.kind == scientific::F_RANGE && ub.data.len == 0);
    // The names of NumPy that stand for a function of the package.
    let fp, g28 = scientific::float_power(av([2.0]), av([3.0]));
    assert(g28.kind == scientific::F_NONE && fp.data[0] == 8.0);
    let rd, g29 = scientific::radians(av([180.0]));
    assert(g29.kind == scientific::F_NONE && near(rd.data[0], 3.141592653589793));
    let ac, g30 = scientific::acos(av([1.0]));
    assert(g30.kind == scientific::F_NONE && ac.data[0] == 0.0);
    let at, g31 = scientific::atan2(av([1.0]), av([1.0]));
    assert(g31.kind == scientific::F_NONE && near(at.data[0], 0.7853981633974483));
    let rm, g32 = scientific::remainder(av([-7.0]), av([3.0]));
    assert(g32.kind == scientific::F_NONE && rm.data[0] == 2.0);
    let bn, g33 = scientific::bitwise_not(av([5.0]));
    assert(g33.kind == scientific::F_NONE && bn.data[0] == -6.0);
    let am, g34 = scientific::amax(av([1.0, 9.0, 4.0]));
    assert(g34.kind == scientific::F_NONE && am == 9.0);
    assert(scientific::ndim(m) == 2);
}

// same tells whether an array holds the shape and the values.
fn same_vals(a: scientific::Arr&, shape: i64[:], v: f64[:]) -> bool {
    if a.ndim != shape.len || a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < shape.len { if a.shape[i] != shape[i] { return false; } i += 1; }
    i = 0;
    while i < v.len { if a.data[i] != v[i] { return false; } i += 1; }
    return true;
}

fn seq(n: i64) -> scientific::Arr {
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < n { v.push(i); i += 1; }
    let a, f = scientific::from_values(v);
    assert(f.kind == scientific::F_NONE);
    return a;
}

// The assembly of shapes, pinned against NumPy.
fn test_assemble() {
    let a6 = seq(6);
    let a, g0 = scientific::reshape(a6, [2, 3]);
    assert(g0.kind == scientific::F_NONE);
    let m, g1 = scientific::reshape(seq(24), [2, 3, 4]);
    assert(g1.kind == scientific::F_NONE);
    // axes
    let sw, g2 = scientific::swapaxes(m, 0, -1);
    assert(g2.kind == scientific::F_NONE && sw.shape[0] == 4 && sw.shape[2] == 2);
    assert(sw.data[3 * 6 + 2 * 2 + 1] == 23.0);
    let pd, g3 = scientific::permute_dims(m, [2, 0, 1]);
    assert(g3.kind == scientific::F_NONE && pd.shape[0] == 4 && pd.shape[1] == 2 && pd.shape[2] == 3);
    let pbad, g4 = scientific::permute_dims(m, [0, 1]);
    assert(g4.kind == scientific::F_SHAPE && pbad.data.len == 0);
    let a3, g5 = scientific::atleast_3d(seq(3));
    assert(g5.kind == scientific::F_NONE && a3.shape[0] == 1 && a3.shape[1] == 3 && a3.shape[2] == 1);
    let b3, g6 = scientific::atleast_3d(a);
    assert(g6.kind == scientific::F_NONE && b3.shape[0] == 2 && b3.shape[2] == 1);
    let b2, g7 = scientific::atleast_2d(seq(3));
    assert(g7.kind == scientific::F_NONE && b2.ndim == 2 && b2.shape[0] == 1);
    // growth and cut
    let ap, g8 = scientific::append(a, seq(2));
    assert(g8.kind == scientific::F_NONE && same_vals(ap, [8], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0]));
    let apx, g9 = scientific::append_axis(a, a, 1);
    assert(g9.kind == scientific::F_NONE && apx.shape[1] == 6 && apx.data[3] == 0.0);
    let da, g10 = scientific::delete_axis(a, [0, -1], 1);
    assert(g10.kind == scientific::F_NONE && same_vals(da, [2, 1], [1.0, 4.0]));
    let df, g11 = scientific::delete(a, [1, 4]);
    assert(g11.kind == scientific::F_NONE && same_vals(df, [4], [0.0, 2.0, 3.0, 5.0]));
    let dbad, g12 = scientific::delete(a, [6]);
    assert(g12.kind == scientific::F_RANGE && dbad.data.len == 0);
    let rz, g13 = scientific::resize(seq(3), [2, 4]);
    assert(g13.kind == scientific::F_NONE && same_vals(rz, [2, 4], [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0]));
    // padding
    let p1, g14 = scientific::pad_modes(a, [1, 2], [2, 1], "reflect");
    assert(g14.kind == scientific::F_NONE);
    assert(same_vals(p1, [5, 6], [5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 5.0, 4.0, 3.0, 4.0, 5.0, 4.0]));
    let q = av([1.0, 2.0, 3.0, 4.0]);
    let pr, g15 = scientific::pad_modes(q, [5], [6], "reflect");
    assert(g15.kind == scientific::F_NONE && same_vals(pr, [15], [2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0]));
    let ps, g16 = scientific::pad_modes(q, [5], [6], "symmetric");
    assert(g16.kind == scientific::F_NONE && same_vals(ps, [15], [4.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0]));
    let pw, g17 = scientific::pad_modes(q, [5], [6], "wrap");
    assert(g17.kind == scientific::F_NONE && same_vals(pw, [15], [4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0]));
    let pe, g18 = scientific::pad_modes(q, [2], [3], "edge");
    assert(g18.kind == scientific::F_NONE && same_vals(pe, [9], [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0]));
    let pc, g19 = scientific::pad_modes(av([1.0, 2.0]), [1], [2], "constant", 9.0);
    assert(g19.kind == scientific::F_NONE && same_vals(pc, [5], [9.0, 1.0, 2.0, 9.0, 9.0]));
    let pm, g20 = scientific::pad_modes(q, [1], [1], "mean");
    assert(g20.kind == scientific::F_VALUE && pm.data.len == 0);
    let pn, g21 = scientific::pad_modes(q, [-1], [1]);
    assert(g21.kind == scientific::F_VALUE && pn.data.len == 0);
    // diagonals
    let dg, g22 = scientific::diagonal(m, 1, 0, 2);
    assert(g22.kind == scientific::F_NONE && same_vals(dg, [3, 2], [1.0, 14.0, 5.0, 18.0, 9.0, 22.0]));
    let dh, g23 = scientific::diagonal(m, -1, 1, 2);
    assert(g23.kind == scientific::F_NONE && same_vals(dh, [2, 2], [4.0, 9.0, 16.0, 21.0]));
    let w, g24 = scientific::reshape(seq(12), [3, 4]);
    assert(g24.kind == scientific::F_NONE);
    let d2, g25 = scientific::diagonal(w, 2);
    assert(g25.kind == scientific::F_NONE && same_vals(d2, [2], [2.0, 7.0]));
    let d3, g26 = scientific::diagonal(w, 4);
    assert(g26.kind == scientific::F_VALUE && d3.data.len == 0);
    var z, gz = scientific::zeros([7, 3]);
    assert(gz.kind == scientific::F_NONE);
    assert(scientific::fill_diagonal(z, 7.0, true).kind == scientific::F_NONE);
    assert(z.data[0] == 7.0 && z.data[4] == 7.0 && z.data[8] == 7.0 && z.data[12] == 7.0 && z.data[20] == 7.0 && z.data[1] == 0.0);
    var z4, gz4 = scientific::zeros([4, 3]);
    assert(gz4.kind == scientific::F_NONE);
    assert(scientific::fill_diagonal(z4, 7.0).kind == scientific::F_NONE);
    assert(z4.data[0] == 7.0 && z4.data[4] == 7.0 && z4.data[8] == 7.0 && z4.data[9] == 0.0 && z4.data[12 - 1] == 0.0);
    var z3, gz3 = scientific::zeros([2, 2, 2]);
    assert(gz3.kind == scientific::F_NONE);
    assert(scientific::fill_diagonal(z3, 5.0).kind == scientific::F_NONE);
    assert(z3.data[0] == 5.0 && z3.data[7] == 5.0 && z3.data[3] == 0.0);
    // tables of places
    let t, g27 = scientific::tri(3, 4, 1);
    assert(g27.kind == scientific::F_NONE);
    assert(same_vals(t, [3, 4], [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]));
    let tl, g28 = scientific::tril_indices(3, -1, 4);
    assert(g28.kind == scientific::F_NONE && same_vals(tl, [2, 3], [1.0, 2.0, 2.0, 0.0, 0.0, 1.0]));
    let tu, g29 = scientific::triu_indices(3, 1, 4);
    assert(g29.kind == scientific::F_NONE && same_vals(tu, [2, 6], [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 3.0]));
    let di, g30 = scientific::diag_indices(3, 2);
    assert(g30.kind == scientific::F_NONE && same_vals(di, [2, 3], [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]));
    let ix, g31 = scientific::indices([2, 3]);
    assert(g31.kind == scientific::F_NONE);
    assert(same_vals(ix, [2, 2, 3], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]));
    let gc, gr, g32 = scientific::ix_([0, 2], [1, 3, 4]);
    assert(g32.kind == scientific::F_NONE && gc.shape[0] == 2 && gc.shape[1] == 1 && gr.shape[0] == 1 && gr.shape[1] == 3);
    let un, g33 = scientific::unravel_index([5, 11], [2, 3, 4]);
    assert(g33.kind == scientific::F_NONE && same_vals(un, [3, 2], [0.0, 0.0, 1.0, 2.0, 1.0, 3.0]));
    let ub, g34 = scientific::unravel_index([24], [2, 3, 4]);
    assert(g34.kind == scientific::F_RANGE && ub.data.len == 0);
    let rv, g35 = scientific::ravel_multi_index(un, [2, 3, 4]);
    assert(g35.kind == scientific::F_NONE && same_vals(rv, [2], [5.0, 11.0]));
    let mw, g36 = scientific::from_rows(2, 2, [5.0, -1.0, 0.0, 0.0]);
    assert(g36.kind == scientific::F_NONE);
    let rw, g37 = scientific::ravel_multi_index(mw, [3, 4], "wrap");
    assert(g37.kind == scientific::F_NONE && same_vals(rw, [2], [8.0, 8.0]));
    let rc, g38 = scientific::ravel_multi_index(mw, [3, 4], "clip");
    assert(g38.kind == scientific::F_NONE && same_vals(rc, [2], [8.0, 0.0]));
    let rb, g39 = scientific::ravel_multi_index(mw, [3, 4]);
    assert(g39.kind == scientific::F_RANGE && rb.data.len == 0);
    // pieces and blocks
    let us, g40 = scientific::unstack(a, 1);
    assert(g40.kind == scientific::F_NONE && us.count == 3);
    let u1, g41 = scientific::unstack_part(us, 1);
    assert(g41.kind == scientific::F_NONE && same_vals(u1, [2], [1.0, 4.0]));
    let bl, g42 = scientific::block_grid(a, a, a, a);
    assert(g42.kind == scientific::F_NONE && bl.shape[0] == 4 && bl.shape[1] == 6 && bl.data[9] == 3.0);
}

// Selection and order, pinned against NumPy.
fn test_select() {
    let k, g0 = scientific::from_rows(2, 5, [3.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 2.0, 1.0]);
    assert(g0.kind == scientific::F_NONE);
    let ls, g1 = scientific::lexsort(k);
    assert(g1.kind == scientific::F_NONE && same_vals(ls, [5], [2.0, 0.0, 4.0, 1.0, 3.0]));
    let kn, g2 = scientific::from_rows(1, 4, [2.0, 1.0, 0.0 / 0.0, 1.0]);
    assert(g2.kind == scientific::F_NONE);
    let ln, g3 = scientific::lexsort(kn);
    assert(g3.kind == scientific::F_NONE && same_vals(ln, [4], [1.0, 3.0, 0.0, 2.0]));
    let a, g4 = scientific::reshape(seq(6), [2, 3]);
    assert(g4.kind == scientific::F_NONE);
    let c1, g5 = scientific::compress_axis([0.0, 1.0, 1.0], a, 1);
    assert(g5.kind == scientific::F_NONE && same_vals(c1, [2, 2], [1.0, 2.0, 4.0, 5.0]));
    let c2, g6 = scientific::compress([1.0, 0.0, 1.0], a);
    assert(g6.kind == scientific::F_NONE && same_vals(c2, [2], [0.0, 2.0]));
    let c3, g7 = scientific::compress([0.0, 0.0], a);
    assert(g7.kind == scientific::F_VALUE && c3.data.len == 0);
    let cm, g8 = scientific::from_rows(2, 3, [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    assert(g8.kind == scientific::F_NONE);
    let ex, g9 = scientific::extract(cm, a);
    assert(g9.kind == scientific::F_NONE && same_vals(ex, [3], [0.0, 2.0, 5.0]));
    let ch, g10 = scientific::reshape(seq(12), [3, 4]);
    assert(g10.kind == scientific::F_NONE);
    let ix = av([5.0, -1.0, 1.0, 2.0]);
    let r1, g11 = scientific::choose(ix, ch, "wrap");
    assert(g11.kind == scientific::F_NONE && same_vals(r1, [4], [8.0, 9.0, 6.0, 11.0]));
    let r2, g12 = scientific::choose(ix, ch, "clip");
    assert(g12.kind == scientific::F_NONE && same_vals(r2, [4], [8.0, 1.0, 6.0, 11.0]));
    let r3, g13 = scientific::choose(ix, ch);
    assert(g13.kind == scientific::F_RANGE && r3.data.len == 0);
    let xs = av([-2.0, 0.0, 3.0, 5.0]);
    let cd, g14 = scientific::from_rows(2, 4, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    assert(g14.kind == scientific::F_NONE);
    let pw, g15 = scientific::piecewise(xs, cd, [-1.0]);
    assert(g15.kind == scientific::F_SHAPE && pw.data.len == 0);
    let cd2, g16 = scientific::reshape(cd, [2, 4]);
    assert(g16.kind == scientific::F_NONE);
    let pw2, g17 = scientific::piecewise(xs, cd2, [-1.0, 1.0, 0.5]);
    assert(g17.kind == scientific::F_NONE && same_vals(pw2, [4], [-1.0, 0.5, 1.0, 1.0]));
    // setxor1d keeps a NaN of both arrays twice.
    let sx, g18 = scientific::setxor1d(av([1.0, 2.0, 3.0, 0.0 / 0.0]), av([3.0, 4.0, 0.0 / 0.0]));
    assert(g18.kind == scientific::F_NONE && sx.data.len == 5);
    assert(sx.data[0] == 1.0 && sx.data[1] == 2.0 && sx.data[2] == 4.0 && sx.data[3] != sx.data[3] && sx.data[4] != sx.data[4]);
    let sy, g19 = scientific::setxor1d(av([1.0, 2.0, 2.0]), av([2.0, 5.0]));
    assert(g19.kind == scientific::F_NONE && same_vals(sy, [2], [1.0, 5.0]));
    let sz, g20 = scientific::setxor1d(av([1.0]), av([1.0]));
    assert(g20.kind == scientific::F_VALUE && sz.data.len == 0);
    // The unique functions keep each NaN apart.
    let u, ui, uv, uc, g21 = scientific::unique_all(av([2.0, 1.0, 2.0, 0.0 / 0.0, 0.0 / 0.0, 1.0]));
    assert(g21.kind == scientific::F_NONE && u.data.len == 4);
    assert(u.data[0] == 1.0 && u.data[1] == 2.0 && u.data[2] != u.data[2] && u.data[3] != u.data[3]);
    assert(same_vals(ui, [4], [1.0, 0.0, 3.0, 4.0]));
    assert(same_vals(uv, [6], [1.0, 0.0, 1.0, 2.0, 3.0, 0.0]));
    assert(same_vals(uc, [4], [2.0, 2.0, 1.0, 1.0]));
    let m, g22 = scientific::from_rows(2, 2, [3.0, 1.0, 3.0, 0.0 / 0.0]);
    assert(g22.kind == scientific::F_NONE);
    let um2, um3, g24 = scientific::unique_inverse(m);
    assert(g24.kind == scientific::F_NONE && same_vals(um3, [2, 2], [1.0, 0.0, 1.0, 2.0]));
    let cv, cc, g25 = scientific::unique_counts(av([5.0, 5.0, 1.0]));
    assert(g25.kind == scientific::F_NONE && same_vals(cv, [2], [1.0, 5.0]) && same_vals(cc, [2], [1.0, 2.0]));
    let vv, g26 = scientific::unique_values(av([3.0, 1.0, 3.0]));
    assert(g26.kind == scientific::F_NONE && same_vals(vv, [2], [1.0, 3.0]));
    // The partition holds the sorted element at kth.
    let pa, g27 = scientific::argpartition(av([3.0, 1.0, 2.0, 0.0]), 1);
    assert(g27.kind == scientific::F_NONE && same_vals(pa, [4], [3.0, 1.0, 2.0, 0.0]));
    let pb, g28 = scientific::argpartition(av([3.0, 1.0, 2.0, 0.0]), 4);
    assert(g28.kind == scientific::F_RANGE && pb.data.len == 0);
}

fn main() {
    test_create();
    test_shape();
    test_ops();
    test_reduce();
    test_axis_and_permute();
    test_logic_and_math();
    test_elementwise_math();
    test_elementwise_rest();
    test_scales();
    test_nan_reductions();
    test_order_statistics();
    test_sets();
    test_stacks();
    test_grids();
    test_positions();
    test_closeness();
    test_histogram2d();
    test_buffer();
    test_pinv();
    test_fftn();
    test_moves();
    test_constants();
    test_filters();
    test_polynomial();
    test_sections();
    test_stats();
    test_ufunc();
    test_assemble();
    test_select();
    test_generators();
    test_indexing();
    test_sort_and_search();
    test_linalg();
    test_decomp_and_eig();
    test_svd_and_poly();
    test_fft();
    test_random();
    test_misc();
    test_optimize();
    test_minimize();
    test_integrate();
    test_interpolate();
    test_signal();
    print("ok\n");
    return;
}
