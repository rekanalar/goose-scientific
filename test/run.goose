import scientific from "..";

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
    // The roots of a polynomial.
    let rt, f17 = scientific::roots([1.0, -3.0, 2.0]);
    assert(f17.kind == scientific::F_NONE && rt.data.len == 2);
    assert(close(rt.data[0], 1.0) || close(rt.data[0], 2.0));
    assert(close(rt.data[1], 1.0) || close(rt.data[1], 2.0));
    // A polynomial with no term is a fault.
    let bad, f18 = scientific::roots([0.0, 0.0]);
    assert(f18.kind == scientific::F_VALUE);
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

fn main() {
    test_create();
    test_shape();
    test_ops();
    test_reduce();
    test_axis_and_permute();
    test_logic_and_math();
    test_indexing();
    test_sort_and_search();
    test_linalg();
    test_decomp_and_eig();
    test_svd_and_poly();
    test_fft();
    test_random();
    print("ok\n");
    return;
}
