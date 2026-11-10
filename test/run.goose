import scientific from "..";
import os;

fn empty_row() -> f64[>..] { var r: f64[>..] = []; return r; }

// all_close tells whether every tap of a is within tol of v.
fn all_close(a: scientific::Arr&, v: f64[:], tol: f64) -> bool {
    if a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        var d = a.data[i] - v[i];
        if d < 0.0 { d = -d; }
        if d > tol { return false; }
        i += 1;
    }
    return true;
}

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
    let col, f4 = scientific::matmul(a, row);
    assert(f4.kind == scientific::F_NONE && same_vals(col, [2], [14.0, 32.0]));
    let rowp, f4b = scientific::matmul(row, b);
    assert(f4b.kind == scientific::F_NONE && same_vals(rowp, [2], [22.0, 28.0]));
    // dot of two rows gives a number.
    let v, f5 = scientific::from_values([1.0, 2.0, 3.0]);
    let w, f6 = scientific::from_values([4.0, 5.0, 6.0]);
    let dt, f7 = scientific::dot(v, w);
    assert(f7.kind == scientific::F_NONE && dt.ndim == 0 && dt.data[0] == 32.0);
    // dot of a row and a matrix gives a row.
    let u, f8a = scientific::from_values([1.0, 2.0]);
    let dr, f8 = scientific::dot(u, a);
    assert(f8.kind == scientific::F_NONE && dr.data.len == 3 && dr.data[0] == 9.0 && dr.data[1] == 12.0 && dr.data[2] == 15.0);
    // dot of a matrix and a row sums over the last axis of the matrix.
    let dc, f9 = scientific::dot(a, v);
    assert(f9.kind == scientific::F_NONE && dc.data.len == 2 && dc.data[0] == 14.0 && dc.data[1] == 32.0);
    let dcb, f9b = scientific::dot(a, u);
    assert(f9b.kind == scientific::F_SHAPE && dcb.data.len == 0);
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

// dsecond is the second derivative of the square of x and two.
fn dsecond(x: f64) -> f64 { return 2.0; }

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
    // The secant method takes two points.
    let r4b, f4b = scientific::secant(1.0, 1.5, 1e-12, 100) { x => x * x - 2.0 };
    assert(f4b.kind == scientific::F_NONE && close(r4b, 1.4142135623730951, 1e-10));
    let r4c, f4c = scientific::secant(0.0, 1.0, 1e-12, 100) { x => x * x * x - 2.0 * x + 2.0 };
    assert(f4c.kind == scientific::F_NONE && close(r4c, -1.7692923542386314, 1e-10));
    // The method of Halley takes the second derivative.
    let r4d, f4d = scientific::halley(1.0, 1e-12, 100, dquad, dsecond) { x => x * x - 2.0 };
    assert(f4d.kind == scientific::F_NONE && close(r4d, 1.4142135623730951, 1e-10));
    let bad4b, fb4b = scientific::secant(1.0, 1.0, 1e-12, 100) { x => x * x - 2.0 };
    assert(fb4b.kind != scientific::F_NONE);
    let bad4c, fb4c = scientific::secant(1.0, 1.5, 0.0, 100) { x => x * x - 2.0 };
    assert(fb4c.kind != scientific::F_NONE);
    let bad4d, fb4d = scientific::secant(1.0, 1.5, 1e-12, 0) { x => x * x - 2.0 };
    assert(fb4d.kind != scientific::F_NONE);
    let bad4e, fb4e = scientific::halley(1.0, 1e-12, 100, dflat, dsecond) { x => x * x - 2.0 };
    assert(fb4e.kind != scientific::F_NONE);
    let bad4f, fb4f = scientific::halley(1.0, 0.0, 100, dquad, dsecond) { x => x * x - 2.0 };
    assert(fb4f.kind != scientific::F_NONE);
    let bad4g, fb4g = scientific::halley(1.0, 1e-12, 0, dquad, dsecond) { x => x * x - 2.0 };
    assert(fb4g.kind != scientific::F_NONE);
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
    // An even count of parts takes the correction of Cartwright on
    // the last part. The straight line gives the same value both ways.
    let s3, f6 = scientific::simpson([0.0, 0.5, 1.0, 1.5], 0.5);
    assert(f6.kind == scientific::F_NONE && close(s3, 1.125));
    // The correction of Cartwright on a signal that is not straight.
    let s4, f6b = scientific::simpson([1.0, 2.0, 4.0, 8.0], 0.5);
    assert(f6b.kind == scientific::F_NONE && close(s4, 5.083333333333333));
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
    // A signal of fewer than two values gives zero, as SciPy does.
    let bad5, fb5 = scientific::trapezoid([1.0], 1.0);
    assert(fb5.kind == scientific::F_NONE && close(bad5, 0.0));
    let bad6, fb6 = scientific::simpson([1.0], 1.0);
    assert(fb6.kind == scientific::F_NONE && close(bad6, 0.0));
    // A signal of two values takes the rule of the trapezoid.
    let bad7, fb7 = scientific::simpson([1.0, 2.0], 1.0);
    assert(fb7.kind == scientific::F_NONE && close(bad7, 1.5));
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
    assert(all_close(ta1, [4.0, 10.0], 1.0e-12));
    let ta0, taf0 = scientific::trapezoid_axis(a23, 0);
    assert(taf0.kind == scientific::F_NONE && all_close(ta0, [2.5, 3.5, 4.5], 1.0e-12));
    let sa1, saf1 = scientific::simpson_axis(a23, 1, 0.5);
    assert(saf1.kind == scientific::F_NONE && all_close(sa1, [2.0, 5.0], 1.0e-12));
    // The axis form of a signal of three dimensions.
    var t3: f64[>..] = [];
    var ti: i64 = 1;
    while ti <= 24 { t3.push(ti); ti += 1; }
    let m3, m3f = scientific::from_values(t3);
    let a234, a234f = scientific::reshape(m3, [2, 3, 4]);
    assert(m3f.kind == scientific::F_NONE && a234f.kind == scientific::F_NONE);
    let ta3, taf3 = scientific::trapezoid_axis(a234, 1, 2.0);
    assert(taf3.kind == scientific::F_NONE && ta3.ndim == 2 && ta3.shape[0] == 2 && ta3.shape[1] == 4);
    assert(all_close(ta3, [20.0, 24.0, 28.0, 32.0, 68.0, 72.0, 76.0, 80.0], 1.0e-12));
    let sa3, saf3 = scientific::simpson_axis(a234, 1);
    assert(saf3.kind == scientific::F_NONE && all_close(sa3, [10.0, 12.0, 14.0, 16.0, 34.0, 36.0, 38.0, 40.0], 1.0e-12));
    // The running total along one axis. The axis holds one value less.
    let ca1, caf1 = scientific::cumulative_trapezoid_axis(a23, 1, 0.5);
    assert(caf1.kind == scientific::F_NONE && ca1.ndim == 2 && ca1.shape[0] == 2 && ca1.shape[1] == 2);
    assert(all_close(ca1, [0.75, 2.0, 2.25, 5.0], 1.0e-12));
    let ca0, caf0 = scientific::cumulative_trapezoid_axis(a23, 0);
    assert(caf0.kind == scientific::F_NONE && ca0.shape[0] == 1 && all_close(ca0, [2.5, 3.5, 4.5], 1.0e-12));
    var c3: f64[>..] = [];
    var ci: i64 = 1;
    while ci <= 8 { c3.push(ci); ci += 1; }
    let mc, mcf = scientific::from_values(c3);
    let a222, a222f = scientific::reshape(mc, [2, 2, 2]);
    assert(mcf.kind == scientific::F_NONE && a222f.kind == scientific::F_NONE);
    let ca2, caf2 = scientific::cumulative_trapezoid_axis(a222, 2);
    assert(caf2.kind == scientific::F_NONE && ca2.shape[0] == 2 && ca2.shape[1] == 2 && ca2.shape[2] == 1);
    assert(all_close(ca2, [1.5, 3.5, 5.5, 7.5], 1.0e-12));
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
    let pm, g20 = scientific::pad_modes(q, [1], [1], "unknown");
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

fn all_near(a: scientific::Arr&, v: f64[:]) -> bool {
    if a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len { if !near(a.data[i], v[i]) { return false; } i += 1; }
    return true;
}

fn win_is(name: const u8[:], n: i64, fftbins: bool, p: f64, v: f64[:]) -> bool {
    let w, f = scientific::get_window(name, n, fftbins, p);
    assert(f.kind == scientific::F_NONE);
    return all_near(w, v);
}

// The products, the windows and the old polynomial functions.
fn test_products() {
    let nan = 0.0 / 0.0;
    // products
    let vd, g0 = scientific::vdot(av([1.0, 2.0, 3.0]), av([4.0, 5.0, 6.0]));
    assert(g0.kind == scientific::F_NONE && vd == 32.0);
    let vx, g1 = scientific::vander([1.0, 2.0, 3.0], 4);
    assert(g1.kind == scientific::F_NONE && same_vals(vx, [3, 4], [1.0, 1.0, 1.0, 1.0, 8.0, 4.0, 2.0, 1.0, 27.0, 9.0, 3.0, 1.0]));
    let vi, g2 = scientific::vander([1.0, 2.0, 3.0], 0, true);
    assert(g2.kind == scientific::F_NONE && same_vals(vi, [3, 3], [1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0, 3.0, 9.0]));
    let a, ga = scientific::reshape(seq(6), [2, 3]);
    let b, gb = scientific::reshape(seq(12), [3, 4]);
    let c3, gc = scientific::reshape(seq(24), [2, 3, 4]);
    assert(ga.kind == scientific::F_NONE && gb.kind == scientific::F_NONE && gc.kind == scientific::F_NONE);
    let td, g3 = scientific::tensordot(c3, b, [1, 2], [0, 1]);
    assert(g3.kind == scientific::F_NONE && same_vals(td, [2], [506.0, 1298.0]));
    let tn, g4 = scientific::tensordot_n(a, b, 1);
    assert(g4.kind == scientific::F_NONE && same_vals(tn, [2, 4], [20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]));
    let tb, g5 = scientific::tensordot(a, b, [0], [0]);
    assert(g5.kind == scientific::F_SHAPE && tb.data.len == 0);
    let vc, g6 = scientific::vecdot(a, a);
    assert(g6.kind == scientific::F_NONE && same_vals(vc, [2], [5.0, 50.0]));
    let mv, g7 = scientific::matvec(a, av([1.0, 0.0, 1.0]));
    assert(g7.kind == scientific::F_NONE && same_vals(mv, [2], [2.0, 8.0]));
    let es, g8 = scientific::einsum("ij,jk->ik", a, b);
    assert(g8.kind == scientific::F_NONE && same_vals(es, [2, 4], [20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]));
    let sq, gq = scientific::reshape(seq(9), [3, 3]);
    assert(gq.kind == scientific::F_NONE);
    let e1, g9 = scientific::einsum1("ii", sq);
    assert(g9.kind == scientific::F_NONE && same_vals(e1, [], [12.0]));
    let e2, g10 = scientific::einsum1("ij->ji", a);
    assert(g10.kind == scientific::F_NONE && same_vals(e2, [3, 2], [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]));
    let e3, g11 = scientific::einsum("i,i", av([0.0, 1.0, 2.0]), av([0.0, 1.0, 2.0]));
    assert(g11.kind == scientific::F_NONE && same_vals(e3, [], [5.0]));
    let e4, g12 = scientific::einsum("ij,kj", a, a);
    assert(g12.kind == scientific::F_NONE && same_vals(e4, [2, 2], [5.0, 14.0, 14.0, 50.0]));
    let e5, g13 = scientific::einsum1("ii->i", sq);
    assert(g13.kind == scientific::F_NONE && same_vals(e5, [3], [0.0, 4.0, 8.0]));
    let e6, g14 = scientific::einsum1("ij", a);
    assert(g14.kind == scientific::F_NONE && same_vals(e6, [2, 3], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));
    let e7, g15 = scientific::einsum("ij,jk", a, a);
    assert(g15.kind == scientific::F_SHAPE && e7.data.len == 0);
    let e8, g16 = scientific::einsum1("i...", a);
    assert(g16.kind == scientific::F_NONE && same_vals(e8, [3, 2], [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]));
    // The rule of letters with an ellipsis and with three operands.
    let v3a, f_v3a = scientific::reshape(seq(3), [3]);
    assert(f_v3a.kind == scientific::F_NONE);
    let v4a, f_v4a = scientific::reshape(seq(4), [4]);
    assert(f_v4a.kind == scientific::F_NONE);
    let v45a, f_v45a = scientific::reshape(seq(20), [4, 5]);
    assert(f_v45a.kind == scientific::F_NONE);
    let v52a, f_v52a = scientific::reshape(seq(10), [5, 2]);
    assert(f_v52a.kind == scientific::F_NONE);
    let v42a, f_v42a = scientific::reshape(seq(8), [4, 2]);
    assert(f_v42a.kind == scientific::F_NONE);
    let v22a, f_v22a = scientific::reshape(seq(4), [2, 2]);
    assert(f_v22a.kind == scientific::F_NONE);
    let v245a, f_v245a = scientific::reshape(seq(40), [2, 4, 5]);
    assert(f_v245a.kind == scientific::F_NONE);
    let v145a, f_v145a = scientific::reshape(seq(20), [1, 4, 5]);
    assert(f_v145a.kind == scientific::F_NONE);
    let v345a, f_v345a = scientific::reshape(seq(60), [3, 4, 5]);
    assert(f_v345a.kind == scientific::F_NONE);
    let v24a, f_v24a = scientific::reshape(seq(8), [2, 4]);
    assert(f_v24a.kind == scientific::F_NONE);
    let v14a, f_v14a = scientific::reshape(seq(4), [1, 4]);
    assert(f_v14a.kind == scientific::F_NONE);
    let v214a, f_v214a = scientific::reshape(seq(8), [2, 1, 4]);
    assert(f_v214a.kind == scientific::F_NONE);
    let v13a, f_v13a = scientific::reshape(seq(3), [1, 3]);
    assert(f_v13a.kind == scientific::F_NONE);
    let v31a, f_v31a = scientific::reshape(seq(3), [3, 1]);
    assert(f_v31a.kind == scientific::F_NONE);
    let v11a, f_v11a = scientific::reshape(seq(1), [1, 1]);
    assert(f_v11a.kind == scientific::F_NONE);

    let e9, g17 = scientific::einsum("...i,...i->...", a, sq);
    assert(g17.kind == scientific::F_SHAPE);
    let e10, g18 = scientific::einsum("...j,...j->...", a, v3a);
    assert(g18.kind == scientific::F_NONE && same_vals(e10, [2], [5.0, 14.0]));
    let e11, g19 = scientific::einsum("i...j,...j->i...", a, v3a);
    assert(g19.kind == scientific::F_NONE && same_vals(e11, [2], [5.0, 14.0]));
    let e12, g20 = scientific::einsum("...ij,...jk->...ik", c3, v45a);
    assert(g20.kind == scientific::F_NONE && same_vals(e12, [2, 3, 5], [70.0, 76.0, 82.0, 88.0, 94.0, 190.0, 212.0, 234.0, 256.0, 278.0, 310.0, 348.0, 386.0, 424.0, 462.0, 430.0, 484.0, 538.0, 592.0, 646.0, 550.0, 620.0, 690.0, 760.0, 830.0, 670.0, 756.0, 842.0, 928.0, 1014.0]));
    let e13, g21 = scientific::einsum("bc,b", v11a, v3a);
    assert(g21.kind == scientific::F_NONE && same_vals(e13, [1], [0.0]));
    let e14, g22 = scientific::einsum("ij,ij->", v13a, v31a);
    assert(g22.kind == scientific::F_NONE && same_vals(e14, [], [9.0]));
    let e15, g23 = scientific::einsum3("ij,jk,kl->il", a, b, v42a);
    assert(g23.kind == scientific::F_NONE && same_vals(e15, [2, 2], [324.0, 422.0, 1008.0, 1304.0]));
    let e16, g24 = scientific::einsum3("ij,jk,kl", a, b, v42a);
    assert(g24.kind == scientific::F_NONE && same_vals(e16, [2, 2], [324.0, 422.0, 1008.0, 1304.0]));
    let e17, g25 = scientific::einsum3("...ij,...jk,...kl->...il", c3, v45a, v52a);
    assert(g25.kind == scientific::F_NONE && same_vals(e17, [2, 3, 2], [1760.0, 2170.0, 5120.0, 6290.0, 8480.0, 10410.0, 11840.0, 14530.0, 15200.0, 18650.0, 18560.0, 22770.0]));
    var es = scientific::eset_new();
    var es1 = scientific::eset_add(es, a);
    var es2 = scientific::eset_add(es1, b);
    var es3 = scientific::eset_add(es2, v42a);
    var es4 = scientific::eset_add(es3, v22a);
    let e18, g26 = scientific::einsum_set("ij,jk,kl,lm->im", es4);
    assert(g26.kind == scientific::F_NONE && same_vals(e18, [2, 2], [844.0, 1590.0, 2608.0, 4920.0]));

    // The product of two arrays of matrices with the batch rule.
    let m1, gm1 = scientific::matmul(c3, v45a);
    assert(gm1.kind == scientific::F_NONE && same_vals(m1, [2, 3, 5], [70.0, 76.0, 82.0, 88.0, 94.0, 190.0, 212.0, 234.0, 256.0, 278.0, 310.0, 348.0, 386.0, 424.0, 462.0, 430.0, 484.0, 538.0, 592.0, 646.0, 550.0, 620.0, 690.0, 760.0, 830.0, 670.0, 756.0, 842.0, 928.0, 1014.0]));
    let m2, gm2 = scientific::matmul(c3, v245a);
    assert(gm2.kind == scientific::F_NONE && same_vals(m2, [2, 3, 5], [70.0, 76.0, 82.0, 88.0, 94.0, 190.0, 212.0, 234.0, 256.0, 278.0, 310.0, 348.0, 386.0, 424.0, 462.0, 1510.0, 1564.0, 1618.0, 1672.0, 1726.0, 1950.0, 2020.0, 2090.0, 2160.0, 2230.0, 2390.0, 2476.0, 2562.0, 2648.0, 2734.0]));
    let m3, gm3 = scientific::matmul(c3, v145a);
    assert(gm3.kind == scientific::F_NONE && same_vals(m3, [2, 3, 5], [70.0, 76.0, 82.0, 88.0, 94.0, 190.0, 212.0, 234.0, 256.0, 278.0, 310.0, 348.0, 386.0, 424.0, 462.0, 430.0, 484.0, 538.0, 592.0, 646.0, 550.0, 620.0, 690.0, 760.0, 830.0, 670.0, 756.0, 842.0, 928.0, 1014.0]));
    let m4, gm4 = scientific::matmul(c3, v345a);
    assert(gm4.kind == scientific::F_SHAPE);
    let m5, gm5 = scientific::matmul(v3a, b);
    assert(gm5.kind == scientific::F_NONE && same_vals(m5, [4], [20.0, 23.0, 26.0, 29.0]));
    let m6, gm6 = scientific::matmul(a, v3a);
    assert(gm6.kind == scientific::F_NONE && same_vals(m6, [2], [5.0, 14.0]));
    let m7, gm7 = scientific::matmul(v3a, v3a);
    assert(gm7.kind == scientific::F_NONE && same_vals(m7, [], [5.0]));
    let m8, gm8 = scientific::matmul(c3, v4a);
    assert(gm8.kind == scientific::F_NONE && same_vals(m8, [2, 3], [14.0, 38.0, 62.0, 86.0, 110.0, 134.0]));

    // The product of a matrix and a row with the batch rule.
    let w1, gw1 = scientific::matvec(a, v3a);
    assert(gw1.kind == scientific::F_NONE && same_vals(w1, [2], [5.0, 14.0]));
    let w2, gw2 = scientific::matvec(c3, v24a);
    assert(gw2.kind == scientific::F_NONE && same_vals(w2, [2, 3], [14.0, 38.0, 62.0, 302.0, 390.0, 478.0]));
    let w3, gw3 = scientific::matvec(c3, v14a);
    assert(gw3.kind == scientific::F_NONE && same_vals(w3, [2, 3], [14.0, 38.0, 62.0, 86.0, 110.0, 134.0]));
    let w4, gw4 = scientific::matvec(c3, v45a);
    assert(gw4.kind == scientific::F_SHAPE);

    // The dot product of two arrays along an axis with the batch rule.
    let u1, gu1 = scientific::vecdot(c3, v4a);
    assert(gu1.kind == scientific::F_NONE && same_vals(u1, [2, 3], [14.0, 38.0, 62.0, 86.0, 110.0, 134.0]));
    let u2, gu2 = scientific::vecdot(c3, v14a);
    assert(gu2.kind == scientific::F_NONE && same_vals(u2, [2, 3], [14.0, 38.0, 62.0, 86.0, 110.0, 134.0]));
    let u3, gu3 = scientific::vecdot(c3, v214a);
    assert(gu3.kind == scientific::F_NONE && same_vals(u3, [2, 3], [14.0, 38.0, 62.0, 302.0, 390.0, 478.0]));
    let u4, gu4 = scientific::vecdot(c3, v24a);
    assert(gu4.kind == scientific::F_SHAPE);
    let u5, gu5 = scientific::vecdot(v3a, v3a);
    assert(gu5.kind == scientific::F_NONE && same_vals(u5, [], [5.0]));
    let u6, gu6 = scientific::vecdot(c3, c3, 1);
    assert(gu6.kind == scientific::F_NONE && same_vals(u6, [2, 4], [80.0, 107.0, 140.0, 179.0, 800.0, 899.0, 1004.0, 1115.0]));
    // windows
    assert(win_is("hann", 6, false, nan, [0.0, 0.34549150281252633, 0.9045084971874737, 0.9045084971874737, 0.34549150281252633, 0.0]));
    assert(win_is("hamming", 5, true, nan, [0.08000000000000007, 0.3978521825875243, 0.9121478174124759, 0.9121478174124759, 0.3978521825875243]));
    assert(win_is("triang", 6, false, nan, [0.16666666666666666, 0.5, 0.8333333333333334, 0.8333333333333334, 0.5, 0.16666666666666666]));
    assert(win_is("bartlett", 6, false, nan, [0.0, 0.4, 0.8, 0.8, 0.4, 0.0]));
    assert(win_is("blackman", 6, false, nan, [0.0, 0.2007701432625305, 0.8492298567374694, 0.8492298567374694, 0.2007701432625305, 0.0]));
    assert(win_is("blackmanharris", 6, false, nan, [6.0000000000001025e-05, 0.10301148934566379, 0.7938335106543363, 0.7938335106543363, 0.10301148934566379, 6.0000000000001025e-05]));
    assert(win_is("nuttall", 6, false, nan, [0.0003628000000000381, 0.11051525304987181, 0.7982580969501283, 0.7982580969501283, 0.11051525304987181, 0.0003628000000000381]));
    assert(win_is("flattop", 6, false, nan, [-0.0004210510000000013, -0.06771425207621193, 0.606872152576212, 0.606872152576212, -0.06771425207621193, -0.0004210510000000013]));
    assert(win_is("bohman", 6, false, nan, [0.0, 0.1791238937062839, 0.8343114522576858, 0.8343114522576855, 0.1791238937062838, 0.0]));
    assert(win_is("barthann", 6, false, nan, [0.0, 0.35857354213752, 0.8794264578624801, 0.8794264578624801, 0.3585735421375199, 0.0]));
    assert(win_is("cosine", 6, false, nan, [0.25881904510252074, 0.7071067811865475, 0.9659258262890682, 0.9659258262890683, 0.7071067811865476, 0.258819045102521]));
    assert(win_is("lanczos", 6, false, nan, [0.0, 0.5045511524271046, 0.935489283788639, 0.935489283788639, 0.5045511524271046, 0.0]));
    assert(win_is("parzen", 6, false, nan, [0.009259259259259254, 0.25, 0.8611111111111112, 0.8611111111111112, 0.25, 0.009259259259259254]));
    assert(win_is("kaiser", 6, false, 5.0, [0.036710892271286676, 0.4149036392433668, 0.9138124838692001, 0.9138124838692001, 0.4149036392433668, 0.036710892271286676]));
    assert(win_is("gaussian", 6, false, 1.5, [0.24935220877729622, 0.6065306597126334, 0.9459594689067654, 0.9459594689067654, 0.6065306597126334, 0.24935220877729622]));
    assert(win_is("tukey", 6, false, 0.4, [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]));
    assert(win_is("tukey", 7, false, nan, [0.0, 0.75, 1.0, 1.0, 1.0, 0.75, 0.0]));
    assert(win_is("exponential", 6, false, 2.0, [0.2865047968601901, 0.4723665527410147, 0.7788007830714049, 0.7788007830714049, 0.4723665527410147, 0.2865047968601901]));
    let hn, g17 = scientific::hanning(5);
    assert(g17.kind == scientific::F_NONE && all_near(hn, [0.0, 0.5, 1.0, 0.5, 0.0]));
    let hm, g18 = scientific::hamming(5);
    assert(g18.kind == scientific::F_NONE && all_near(hm, [0.08, 0.54, 1.0, 0.54, 0.08]));
    let bl, g19 = scientific::blackman(5);
    assert(g19.kind == scientific::F_NONE && all_near(bl, [0.0, 0.34, 1.0, 0.34, 0.0]));
    let ba, g20 = scientific::bartlett(5);
    assert(g20.kind == scientific::F_NONE && all_near(ba, [0.0, 0.5, 1.0, 0.5, 0.0]));
    let ka, g21 = scientific::kaiser(5, 8.6);
    assert(g21.kind == scientific::F_NONE && all_near(ka, [0.0013325139979024198, 0.34039362244018856, 1.0, 0.34039362244018856, 0.0013325139979024198]));
    let h1, g22 = scientific::hanning(1);
    assert(g22.kind == scientific::F_NONE && same_vals(h1, [1], [1.0]));
    let h0, g23 = scientific::hanning(0);
    assert(g23.kind == scientific::F_RANGE && h0.data.len == 0);
    let kb, g24 = scientific::kaiser(5, 701.0);
    assert(g24.kind == scientific::F_RANGE && kb.data.len == 0);
    let wn, g25 = scientific::get_window("nope", 5);
    assert(g25.kind == scientific::F_VALUE && wn.data.len == 0);
    // the old polynomial functions
    let pa, h0a = scientific::polyadd([1.0, 2.0], [0.0, 0.0, 3.0]);
    assert(h0a.kind == scientific::F_NONE && same_vals(pa, [3], [0.0, 1.0, 5.0]));
    let ps, h1a = scientific::polysub([1.0, 2.0, 3.0], [1.0, 2.0]);
    assert(h1a.kind == scientific::F_NONE && same_vals(ps, [3], [1.0, 1.0, 1.0]));
    let pm, h2a = scientific::polymul([1.0, 2.0], [1.0, -2.0, 0.0]);
    assert(h2a.kind == scientific::F_NONE && same_vals(pm, [4], [1.0, 0.0, -4.0, 0.0]));
    let q1, r1, h3a = scientific::polydiv([1.0, 0.0, -1.0], [1.0, 1.0]);
    assert(h3a.kind == scientific::F_NONE && same_vals(q1, [2], [1.0, -1.0]) && same_vals(r1, [1], [0.0]));
    let q2, r2, h4a = scientific::polydiv([2.0, 3.0], [1.0, 0.0, 0.0]);
    assert(h4a.kind == scientific::F_NONE && same_vals(q2, [1], [0.0]) && same_vals(r2, [2], [2.0, 3.0]));
    let q3, r3, h5a = scientific::polydiv([0.0, 0.0, 1.0], [2.0]);
    assert(h5a.kind == scientific::F_NONE && same_vals(q3, [3], [0.0, 0.0, 0.5]) && same_vals(r3, [1], [0.0]));
    let q4, r4, h6a = scientific::polydiv([1.0], [0.0, 1.0]);
    assert(h6a.kind == scientific::F_VALUE && q4.data.len == 0 && r4.data.len == 0);
    let d1, h7a = scientific::polyder([3.0, 2.0, 1.0]);
    assert(h7a.kind == scientific::F_NONE && same_vals(d1, [2], [6.0, 2.0]));
    let d2, h8a = scientific::polyder([3.0, 2.0, 1.0], 2);
    assert(h8a.kind == scientific::F_NONE && same_vals(d2, [1], [6.0]));
    let d3, h9a = scientific::polyder([3.0, 2.0, 1.0], 3);
    assert(h9a.kind == scientific::F_NONE && same_vals(d3, [1], [0.0]));
    let i1, h10a = scientific::polyint([3.0, 2.0, 1.0]);
    assert(h10a.kind == scientific::F_NONE && all_near(i1, [1.0, 1.0, 1.0, 0.0]));
    let i2, h11a = scientific::polyint([3.0, 2.0, 1.0], 2, [1.0, 2.0]);
    assert(h11a.kind == scientific::F_NONE && all_near(i2, [0.25, 1.0 / 3.0, 0.5, 1.0, 2.0]));
    let pr, h12a = scientific::poly([1.0, 2.0, 3.0]);
    assert(h12a.kind == scientific::F_NONE && same_vals(pr, [4], [1.0, -6.0, 11.0, -6.0]));
    let pre, pim, h13a = scientific::poly_cplx([1.0, 1.0, 3.0], [1.0, -1.0, 0.0]);
    assert(h13a.kind == scientific::F_NONE && all_near(pre, [1.0, -5.0, 8.0, -6.0]) && all_near(pim, [0.0, 0.0, 0.0, 0.0]));
}

fn mx(r: i64, c: i64, v: f64[:]) -> scientific::Arr {
    let a, f = scientific::from_rows(r, c, v);
    assert(f.kind == scientific::F_NONE);
    return a;
}

// The functions of scipy.linalg for factors, special matrices and the
// matrix exponential, pinned against SciPy.
fn test_scilinalg() {
    let a = mx(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
    let f, piv, g0 = scientific::lu_factor(a);
    assert(g0.kind == scientific::F_NONE);
    assert(all_near(f, [7.0, 8.0, 10.0, 0.14285714285714285, 0.8571428571428572, 1.5714285714285716, 0.5714285714285714, 0.5000000000000002, -0.5]));
    assert(same_vals(piv, [3], [2.0, 2.0, 2.0]));
    let x1, g1 = scientific::lu_solve(f, piv, av([1.0, 2.0, 3.0]));
    assert(g1.kind == scientific::F_NONE && all_near(x1, [-1.0 / 3.0, 2.0 / 3.0, 0.0]));
    let x2, g2 = scientific::lu_solve(f, piv, av([1.0, 2.0, 3.0]), true);
    assert(g2.kind == scientific::F_NONE && all_near(x2, [1.0, 0.0, 0.0]));
    let p, l, u, g3 = scientific::lu(mx(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert(g3.kind == scientific::F_NONE);
    assert(same_vals(p, [3, 3], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]));
    assert(all_near(l, [1.0, 0.0, 0.2, 1.0, 0.6, 0.5]) && all_near(u, [5.0, 6.0, 0.0, 0.8]));
    let sing, sp, g4 = scientific::lu_factor(mx(2, 2, [1.0, 2.0, 2.0, 4.0]));
    assert(g4.kind == scientific::F_NONE);
    let sx, g6 = scientific::lu_solve(sing, sp, av([1.0, 1.0]));
    assert(g6.kind == scientific::F_VALUE && sx.data.len == 0);
    // solve takes a matrix on the right.
    let sm, g7 = scientific::solve(a, mx(3, 2, [1.0, 0.0, 2.0, 1.0, 3.0, 0.0]));
    assert(g7.kind == scientific::F_NONE && sm.ndim == 2 && sm.shape[1] == 2);
    assert(near(sm.data[0], -1.0 / 3.0) && near(sm.data[2], 2.0 / 3.0) && near(sm.data[4], 0.0));
    // triangular solves
    let t = mx(3, 3, [2.0, 9.0, 9.0, 1.0, 3.0, 9.0, 4.0, 5.0, 6.0]);
    let r1, g8 = scientific::solve_triangular(t, av([2.0, 4.0, 15.0]), true);
    assert(g8.kind == scientific::F_NONE && all_near(r1, [1.0, 1.0, 1.0]));
    let r2, g9 = scientific::solve_triangular(t, av([2.0, 4.0, 15.0]), false, 1);
    assert(g9.kind == scientific::F_NONE && all_near(r2, [1.0, -1.6666666666666667, 3.5]));
    let r3, g10 = scientific::solve_triangular(t, mx(3, 1, [2.0, 4.0, 15.0]), true, 0, true);
    assert(g10.kind == scientific::F_NONE && all_near(r3, [2.0, 2.0, -3.0]));
    let r4, g11 = scientific::solve_triangular(mx(2, 2, [0.0, 1.0, 0.0, 1.0]), av([1.0, 1.0]));
    assert(g11.kind == scientific::F_VALUE && r4.data.len == 0);
    // special matrices
    let bd, g12 = scientific::block_diag(mx(2, 2, [1.0, 2.0, 3.0, 4.0]), av([5.0, 6.0, 7.0]));
    assert(g12.kind == scientific::F_NONE && same_vals(bd, [3, 5], [1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0]));
    let tp, g13 = scientific::toeplitz([1.0, 2.0, 3.0], [1.0, 5.0, 6.0, 7.0]);
    assert(g13.kind == scientific::F_NONE && same_vals(tp, [3, 4], [1.0, 5.0, 6.0, 7.0, 2.0, 1.0, 5.0, 6.0, 3.0, 2.0, 1.0, 5.0]));
    let ts, g14 = scientific::toeplitz([1.0, 2.0, 3.0]);
    assert(g14.kind == scientific::F_NONE && same_vals(ts, [3, 3], [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]));
    let hk, g15 = scientific::hankel([1.0, 2.0, 3.0], [3.0, 7.0, 8.0]);
    assert(g15.kind == scientific::F_NONE && same_vals(hk, [3, 3], [1.0, 2.0, 3.0, 2.0, 3.0, 7.0, 3.0, 7.0, 8.0]));
    let h2, g16 = scientific::hankel([1.0, 2.0, 3.0]);
    assert(g16.kind == scientific::F_NONE && same_vals(h2, [3, 3], [1.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 0.0]));
    let ci, g17 = scientific::circulant([1.0, 2.0, 3.0]);
    assert(g17.kind == scientific::F_NONE && same_vals(ci, [3, 3], [1.0, 3.0, 2.0, 2.0, 1.0, 3.0, 3.0, 2.0, 1.0]));
    // null space and range
    let nm = mx(2, 3, [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
    let ns, g18 = scientific::null_space(nm);
    assert(g18.kind == scientific::F_NONE && ns.shape[0] == 3 && ns.shape[1] == 2);
    let prod, g19 = scientific::matmul(nm, ns);
    assert(g19.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < prod.data.len { assert(mathshim_abs(prod.data[i]) < 0.000000001); i += 1; }
    let gram, g20 = scientific::matmul(scientific::transpose(ns), ns);
    assert(g20.kind == scientific::F_NONE && near(gram.data[0], 1.0) && near(gram.data[3], 1.0) && mathshim_abs(gram.data[1]) < 0.000000001);
    let nf, g21 = scientific::null_space(mx(2, 2, [1.0, 0.0, 0.0, 1.0]));
    assert(g21.kind == scientific::F_VALUE && nf.data.len == 0);
    let og, g22 = scientific::orth(mx(3, 2, [1.0, 2.0, 2.0, 4.0, 3.0, 6.0]));
    assert(g22.kind == scientific::F_NONE && og.shape[0] == 3 && og.shape[1] == 1);
    assert(near(og.data[0] * og.data[0] + og.data[1] * og.data[1] + og.data[2] * og.data[2], 1.0));
    // the matrix exponential
    let e1, g23 = scientific::expm(mx(2, 2, [0.0, 1.0, -1.0, 0.0]));
    assert(g23.kind == scientific::F_NONE && all_near(e1, [0.5403023058681397, 0.8414709848078965, -0.8414709848078966, 0.5403023058681397]));
    let e2, g24 = scientific::expm(mx(2, 2, [1.0, 2.0, 3.0, 4.0]));
    assert(g24.kind == scientific::F_NONE && all_near(e2, [51.96895619870756, 74.73656456700694, 112.10484685051044, 164.07380304921801]));
    let e3, g25 = scientific::expm(mx(2, 2, [10.0, 0.0, 0.0, -20.0]));
    assert(g25.kind == scientific::F_NONE && near(e3.data[0], 22026.465794806718) && near(e3.data[3] * 1.0e9, 2.061153622438558));
    let e4, g26 = scientific::expm(mx(2, 2, [0.0, 50.0, 0.0, 0.0]));
    assert(g26.kind == scientific::F_NONE && all_near(e4, [1.0, 50.0, 0.0, 1.0]));
    let e5, g27 = scientific::expm(mx(2, 3, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(g27.kind == scientific::F_SHAPE && e5.data.len == 0);
    // the banded solve
    let ab = mx(3, 4, [0.0, 2.0, 1.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 1.0, 1.0, 0.0]);
    let sb, g28 = scientific::solve_banded(1, 1, ab, av([1.0, 2.0, 3.0, 4.0]));
    assert(g28.kind == scientific::F_NONE && all_near(sb, [0.09577464788732393, 0.30845070422535215, 0.3619718309859155, 0.5197183098591549]));
    let sbb, g29 = scientific::solve_banded(1, 0, ab, av([1.0, 2.0, 3.0, 4.0]));
    assert(g29.kind == scientific::F_SHAPE && sbb.data.len == 0);
    // a zero on the main diagonal needs a row swap
    let abp = mx(3, 4, [0.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0]);
    let sp, g30 = scientific::solve_banded(1, 1, abp, av([1.0, 2.0, 3.0, 4.0]));
    assert(g30.kind == scientific::F_NONE && all_near(sp, [-10.0, 0.5, 4.0, 1.0]));
    // two lower and one upper diagonal, two right sides
    let ab2 = mx(4, 5, [0.0, 1.0, 2.0, 3.0, 4.0, 2.0, -1.0, 3.0, 1.0, 5.0, 1.0, 2.0, -1.0, 4.0, 0.0, 3.0, 1.0, 2.0, 0.0, 0.0]);
    let s2, g31 = scientific::solve_banded(2, 1, ab2, mx(5, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0]));
    assert(g31.kind == scientific::F_NONE && s2.shape[0] == 5 && s2.shape[1] == 2);
    assert(all_near(s2, [-0.8437499999999986, -3.6874999999999987, 2.687499999999998, 9.374999999999996, 3.2656249999999982, 8.531249999999996, -2.5468749999999982, -9.093749999999996, 2.531249999999999, 4.062499999999998]));
    // a side of 2000 solves at the cost of the band
    var big: f64[>..] = [];
    var bi: i64 = 0;
    while bi < 2000 { if bi == 0 { big.push(0.0); } else { big.push(-1.0); } bi += 1; }
    bi = 0;
    while bi < 2000 { big.push(4.0); bi += 1; }
    bi = 0;
    while bi < 2000 { if bi == 1999 { big.push(0.0); } else { big.push(-1.0); } bi += 1; }
    var ones: f64[>..] = [];
    bi = 0;
    while bi < 2000 { ones.push(1.0); bi += 1; }
    let sbig, g32 = scientific::solve_banded(1, 1, mx(3, 2000, copy(big)), av(copy(ones)));
    assert(g32.kind == scientific::F_NONE && near(sbig.data[0], 0.36602540378443865) && near(sbig.data[1000], 0.5) && near(sbig.data[1999], 0.36602540378443865));
    // a zero matrix is singular
    let sz, g33 = scientific::solve_banded(1, 1, mx(3, 2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), av([1.0, 2.0]));
    assert(g33.kind == scientific::F_VALUE && sz.data.len == 0);
}

// The cosine, sine and real transforms, pinned against SciPy.
fn test_transforms() {
    let r1, e1 = scientific::dct([1.0, 3.0, 2.0, 5.0, 4.0], 1);
    assert(e1.kind == scientific::F_NONE && all_near(r1, [25.0, -5.82842712474619, 1.0, -0.17157287525381015, -7.0]));
    let r2, e2 = scientific::dct([1.0, 3.0, 2.0, 5.0, 4.0], 1, "ortho");
    assert(e2.kind == scientific::F_NONE && all_near(r2, [6.76776695296637, -2.5000000000000004, 1.085786437626905, -0.5000000000000003, -1.2322330470336313]));
    let r3, e3 = scientific::dct([2.0, -1.0, 4.0, 0.5], 1, "forward");
    assert(e3.kind == scientific::F_NONE && all_near(r3, [1.4166666666666665, -0.5833333333333333, -0.08333333333333333, 1.9166666666666665]));
    let r4, e4 = scientific::idct([2.0, -1.0, 4.0, 0.5], 1);
    assert(e4.kind == scientific::F_NONE && all_near(r4, [1.4166666666666665, -0.5833333333333333, -0.08333333333333333, 1.9166666666666665]));
    let r5, e5 = scientific::idct([2.0, -1.0, 4.0, 0.5], 1, "ortho");
    assert(e5.kind == scientific::F_NONE && all_near(r5, [2.752671533728535, -1.1752160485348764, 0.21863080158247541, 3.4991237816439233]));
    let r6, e6 = scientific::idct([2.0, -1.0, 4.0, 0.5], 1, "forward");
    assert(e6.kind == scientific::F_NONE && all_near(r6, [8.5, -3.5, -0.5, 11.5]));
    let r7, e7 = scientific::dst([1.0, 3.0, 2.0, 5.0, 4.0], 1);
    assert(e7.kind == scientific::F_NONE && all_near(r7, [22.856406460551018, -8.660254037844386, 6.0, -1.7320508075688785, -4.8564064605510175]));
    let r8, e8 = scientific::dst([1.0, 3.0, 2.0, 5.0, 4.0], 1, "ortho");
    assert(e8.kind == scientific::F_NONE && all_near(r8, [6.598076211353315, -2.4999999999999996, 1.7320508075688772, -0.5000000000000003, -1.4019237886466838]));
    let r9, e9 = scientific::dst([2.0, -1.0, 4.0, 0.5], 1, "forward");
    assert(e9.kind == scientific::F_NONE && all_near(r9, [0.8645265359233287, -0.3024682974039271, 0.12285710677209292, 1.1273920919828955]));
    let r10, e10 = scientific::idst([2.0, -1.0, 4.0, 0.5], 1);
    assert(e10.kind == scientific::F_NONE && all_near(r10, [0.8645265359233287, -0.3024682974039271, 0.12285710677209292, 1.1273920919828955]));
    let r11, e11 = scientific::idst([2.0, -1.0, 4.0, 0.5], 1, "ortho");
    assert(e11.kind == scientific::F_NONE && all_near(r11, [2.7338729511730984, -0.956488739789604, 0.38850828413831073, 3.565126826728005]));
    let r12, e12 = scientific::dct([1.0, 3.0, 2.0, 5.0, 4.0], 2);
    assert(e12.kind == scientific::F_NONE && all_near(r12, [30.0, -8.057480106940814, -0.8541019662496844, 0.27751455142577564, -5.854101966249684]));
    let r13, e13 = scientific::dct([1.0, 3.0, 2.0, 5.0, 4.0], 2, "ortho");
    assert(e13.kind == scientific::F_NONE && all_near(r13, [6.708203932499369, -2.5479989339430063, -0.27009075673772637, 0.08775780663453792, -1.8512295868219164]));
    let r14, e14 = scientific::dct([2.0, -1.0, 4.0, 0.5], 2, "forward");
    assert(e14.kind == scientific::F_NONE && all_near(r14, [1.375, -0.13189946576462963, -0.08838834764831843, 1.2983557027760169]));
    let r15, e15 = scientific::idct([2.0, -1.0, 4.0, 0.5], 2);
    assert(e15.kind == scientific::F_NONE && all_near(r15, [0.773972327104362, -0.6682625808417307, -0.2459509815313642, 1.140241235268733]));
    let r16, e16 = scientific::idct([2.0, -1.0, 4.0, 0.5], 2, "ortho");
    assert(e16.kind == scientific::F_NONE && all_near(r16, [2.482017542598361, -1.5972387912921924, -0.4027612087078072, 3.5179824574016387]));
    let r17, e17 = scientific::idct([2.0, -1.0, 4.0, 0.5], 2, "forward");
    assert(e17.kind == scientific::F_NONE && all_near(r17, [6.191778616834896, -5.346100646733846, -1.9676078522509135, 9.121929882149864]));
    let r18, e18 = scientific::dst([1.0, 3.0, 2.0, 5.0, 4.0], 2);
    assert(e18.kind == scientific::F_NONE && all_near(r18, [20.034441853748632, -7.330937578935451, 9.034441853748632, -3.3551980886010284, -2.0]));
    let r19, e19 = scientific::dst([1.0, 3.0, 2.0, 5.0, 4.0], 2, "ortho");
    assert(e19.kind == scientific::F_NONE && all_near(r19, [6.335446790805166, -2.318246013395644, 2.85694136461995, -1.0610067961022676, -0.447213595499958]));
    let r20, e20 = scientific::dst([2.0, -1.0, 4.0, 0.5], 2, "forward");
    assert(e20.kind == scientific::F_NONE && all_near(r20, [0.9320867946116462, -0.618718433538229, 0.29041213354573686, 1.625]));
    let r21, e21 = scientific::idst([2.0, -1.0, 4.0, 0.5], 2);
    assert(e21.kind == scientific::F_NONE && all_near(r21, [1.0009445533971948, -0.16002036140608322, 0.3185330291871905, 1.2294979439904683]));
    let r22, e22 = scientific::idst([2.0, -1.0, 4.0, 0.5], 2, "ortho");
    assert(e22.kind == scientific::F_NONE && all_near(r22, [2.90432202989895, -0.5258292354160173, 0.9741707645839827, 3.40432202989895]));
    let r23, e23 = scientific::dct([1.0, 3.0, 2.0, 5.0, 4.0], 3);
    assert(e23.kind == scientific::F_NONE && all_near(r23, [18.29239555319502, -12.692057581696067, 5.000000000000002, -0.7243502833026718, -4.8759876881962825]));
    let r24, e24 = scientific::dct([1.0, 3.0, 2.0, 5.0, 4.0], 3, "ortho");
    assert(e24.kind == scientific::F_NONE && all_near(r24, [5.915549210366321, -3.882595185733698, 1.7121246595673103, -0.09807384241934769, -1.4109368642807965]));
    let r25, e25 = scientific::dct([2.0, -1.0, 4.0, 0.5], 3, "forward");
    assert(e25.kind == scientific::F_NONE && all_near(r25, [0.773972327104362, -0.6682625808417307, -0.2459509815313642, 1.140241235268733]));
    let r26, e26 = scientific::idct([2.0, -1.0, 4.0, 0.5], 3);
    assert(e26.kind == scientific::F_NONE && all_near(r26, [1.375, -0.13189946576462963, -0.08838834764831843, 1.2983557027760169]));
    let r27, e27 = scientific::idct([2.0, -1.0, 4.0, 0.5], 3, "ortho");
    assert(e27.kind == scientific::F_NONE && all_near(r27, [2.7500000000000004, -0.37306802670821004, -0.25, 3.6723044873005897]));
    let r28, e28 = scientific::idct([2.0, -1.0, 4.0, 0.5], 3, "forward");
    assert(e28.kind == scientific::F_NONE && all_near(r28, [11.0, -1.055195726117037, -0.7071067811865475, 10.386845622208135]));
    let r29, e29 = scientific::dst([1.0, 3.0, 2.0, 5.0, 4.0], 3);
    assert(e29.kind == scientific::F_NONE && all_near(r29, [20.89137864295606, -1.3174114589041253, 2.0, -0.9743846085965062, -5.183174710456689]));
    let r30, e30 = scientific::dst([1.0, 3.0, 2.0, 5.0, 4.0], 3, "ortho");
    assert(e30.kind == scientific::F_NONE && all_near(r30, [7.130377315206354, -0.9405454005067148, 1.1563988499661557, -0.8320707859501444, -1.1151204416302094]));
    let r31, e31 = scientific::dst([2.0, -1.0, 4.0, 0.5], 3, "forward");
    assert(e31.kind == scientific::F_NONE && all_near(r31, [1.0009445533971948, -0.16002036140608322, 0.3185330291871905, 1.2294979439904683]));
    let r32, e32 = scientific::idst([2.0, -1.0, 4.0, 0.5], 3);
    assert(e32.kind == scientific::F_NONE && all_near(r32, [0.9320867946116462, -0.618718433538229, 0.29041213354573686, 1.625]));
    let r33, e33 = scientific::idst([2.0, -1.0, 4.0, 0.5], 3, "ortho");
    assert(e33.kind == scientific::F_NONE && all_near(r33, [2.6363395724973113, -1.7499999999999998, 0.8214095558761754, 3.250000000000001]));
    let r34, e34 = scientific::dct([1.0, 3.0, 2.0, 5.0, 4.0], 4);
    assert(e34.kind == scientific::F_NONE && all_near(r34, [15.941223668783987, -13.61661468999582, 7.0710678118654755, -7.753429462450177, -0.5912054607332058]));
    let r35, e35 = scientific::dct([1.0, 3.0, 2.0, 5.0, 4.0], 4, "ortho");
    assert(e35.kind == scientific::F_NONE && all_near(r35, [5.041057548354302, -4.305951644129437, 2.23606797749979, -2.4518496778797525, -0.18695558210461716]));
    let r36, e36 = scientific::dct([2.0, -1.0, 4.0, 0.5], 4, "forward");
    assert(e36.kind == scientific::F_NONE && all_near(r36, [0.8624817603975973, -0.5857241728753759, 0.8220054601645551, 0.9453091715151061]));
    let r37, e37 = scientific::idct([2.0, -1.0, 4.0, 0.5], 4);
    assert(e37.kind == scientific::F_NONE && all_near(r37, [0.8624817603975973, -0.5857241728753759, 0.8220054601645551, 0.9453091715151061]));
    let r38, e38 = scientific::idct([2.0, -1.0, 4.0, 0.5], 4, "ortho");
    assert(e38.kind == scientific::F_NONE && all_near(r38, [2.4394668057074087, -1.6566781381802402, 2.3249825402189024, 2.6737381019846747]));
    let r39, e39 = scientific::idct([2.0, -1.0, 4.0, 0.5], 4, "forward");
    assert(e39.kind == scientific::F_NONE && all_near(r39, [6.899854083180778, -4.685793383003007, 6.576043681316441, 7.562473372120849]));
    let r40, e40 = scientific::dst([1.0, 3.0, 2.0, 5.0, 4.0], 4);
    assert(e40.kind == scientific::F_NONE && all_near(r40, [22.676811019908712, 0.9701413238868586, 1.4142135623730951, 4.259938541424164, -3.830664616267361]));
    let r41, e41 = scientific::dst([1.0, 3.0, 2.0, 5.0, 4.0], 4, "ortho");
    assert(e41.kind == scientific::F_NONE && all_near(r41, [7.171037289211745, 0.30678562357335903, 0.447213595499958, 1.3471108483235905, -1.2113625139619755]));
    let r42, e42 = scientific::dst([2.0, -1.0, 4.0, 0.5], 4, "forward");
    assert(e42.kind == scientific::F_NONE && all_near(r42, [0.9127203751061128, 0.12374541688730373, -0.5443767756285396, 1.2294439860448376]));
    let r43, e43 = scientific::idst([2.0, -1.0, 4.0, 0.5], 4);
    assert(e43.kind == scientific::F_NONE && all_near(r43, [0.9127203751061128, 0.12374541688730373, -0.5443767756285396, 1.2294439860448376]));
    let r44, e44 = scientific::idst([2.0, -1.0, 4.0, 0.5], 4, "ortho");
    assert(e44.kind == scientific::F_NONE && all_near(r44, [2.581563066258647, 0.3500048936870751, -1.5397300382696322, 3.4773927184852957]));
    let h45, he45 = scientific::hfft([1.0, 2.0, 3.0], [0.0, 1.0, -2.0]);
    assert(he45.kind == scientific::F_NONE && all_near(h45, [8.0, 0.0, 0.0, -4.0]));
    let h46, he46 = scientific::hfft([1.0, 2.0, 3.0], [0.0, 1.0, -2.0], 5);
    assert(he46.kind == scientific::F_NONE && all_near(h46, [11.0, -3.06706196532948, 4.597830558515455, -5.3617625810156655, -2.1690060121703096]));
    let ia47, ib47, ie47 = scientific::ihfft([1.0, 3.0, 2.0, 5.0, 4.0]);
    assert(ie47.kind == scientific::F_NONE && all_near(ia47, [3.0, -0.5, -0.5]) && all_near(ib47, [0.0, -0.5428824546345146, 0.45307685931859754]));
    let ma = mx(3, 4, [0.0, 1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4, 17.1, 20.0, 23.1]);
    let qa, qb, qe = scientific::rfft2(ma);
    assert(qe.kind == scientific::F_NONE && qa.shape[1] == 3 && all_near(qa, [116.6, -12.0, -12.600000000000003, -47.199999999999996, 1.0143593539448985, 2.4000000000000017, -47.199999999999996, 3.7856406460551013, 2.4000000000000017]) && all_near(qb, [0.0, 13.2, 0.0, 34.64101615137754, -3.7856406460551018, -1.385640646055106, -34.64101615137754, -1.014359353944899, 1.385640646055106]));
    let ba, be = scientific::irfft2(qa, qb, 4);
    assert(be.kind == scientific::F_NONE && all_near(ba, [0.0, 1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4, 17.1, 20.0, 23.1]));
    let ca, ce = scientific::reshape(av([0.0, 1.0, 2.8284271247461903, 5.196152422706632, 8.0, 11.180339887498949, 14.696938456699069, 18.520259177452136, 22.627416997969522, 27.0, 31.622776601683793, 36.4828726939094, 41.569219381653056, 46.87216658103186, 52.38320341483518, 58.09475019311125, 64.0, 70.09279563550022, 76.36753236814714, 82.8190799272728, 89.44271909999159, 96.23408959407264, 103.18914671611545, 110.30412503619254]), [2, 3, 4]);
    assert(ce.kind == scientific::F_NONE);
    let ra, rb, re2 = scientific::rfftn(ca);
    assert(re2.kind == scientific::F_NONE && ra.shape[2] == 3 && all_near(ra, [1070.5240113105895, -55.44866920261264, -57.06925098690747, -223.34612697816848, 4.241143144896084, 6.961296580030753, -223.34612697816848, 10.280292583931617, 6.961296580030753, -712.213644585258, 18.407218832293587, 17.86112098597038, 71.26468225786121, 2.0825964921200417, 2.539592526556465, 71.26468225786121, 3.466855400894173, 2.539592526556465]) && all_near(rb, [0.0, 59.03784775254108, 0.0, 148.28624010836558, -9.575469239907518, -3.1099181009967474, -148.28624010836558, -3.2061704082754985, 3.1099181009967474, 0.0, -16.999958939402646, 0.0, -35.1217847360348, -2.8360001023628683, -0.7508551677725565, 35.1217847360348, -1.24333452635276, 0.7508551677725565]));
    let sa, se = scientific::irfftn(ra, rb, 4);
    assert(se.kind == scientific::F_NONE && all_near(sa, [0.0, 1.0, 2.8284271247461903, 5.196152422706632, 8.0, 11.180339887498949, 14.696938456699069, 18.520259177452136, 22.627416997969522, 27.0, 31.622776601683793, 36.4828726939094, 41.569219381653056, 46.87216658103186, 52.38320341483518, 58.09475019311125, 64.0, 70.09279563550022, 76.36753236814714, 82.8190799272728, 89.44271909999159, 96.23408959407264, 103.18914671611545, 110.30412503619254]));
    let nf1, nfe1 = scientific::next_fast_len(1);
    let nr1, nre1 = scientific::next_fast_len(1, true);
    assert(nfe1.kind == scientific::F_NONE && nre1.kind == scientific::F_NONE && nf1 == 1 && nr1 == 1);
    let nf13, nfe13 = scientific::next_fast_len(13);
    let nr13, nre13 = scientific::next_fast_len(13, true);
    assert(nfe13.kind == scientific::F_NONE && nre13.kind == scientific::F_NONE && nf13 == 14 && nr13 == 15);
    let nf127, nfe127 = scientific::next_fast_len(127);
    let nr127, nre127 = scientific::next_fast_len(127, true);
    assert(nfe127.kind == scientific::F_NONE && nre127.kind == scientific::F_NONE && nf127 == 128 && nr127 == 128);
    let nf1031, nfe1031 = scientific::next_fast_len(1031);
    let nr1031, nre1031 = scientific::next_fast_len(1031, true);
    assert(nfe1031.kind == scientific::F_NONE && nre1031.kind == scientific::F_NONE && nf1031 == 1050 && nr1031 == 1080);
    let nf100001, nfe100001 = scientific::next_fast_len(100001);
    let nr100001, nre100001 = scientific::next_fast_len(100001, true);
    assert(nfe100001.kind == scientific::F_NONE && nre100001.kind == scientific::F_NONE && nf100001 == 100352 && nr100001 == 101250);
    // bad input
    let bk, k1 = scientific::dct([1.0, 2.0], 5);
    assert(k1.kind == scientific::F_VALUE && bk.data.len == 0);
    let bn, k2 = scientific::dct([1.0, 2.0], 2, "nope");
    assert(k2.kind == scientific::F_VALUE && bn.data.len == 0);
    let b1, k3 = scientific::dct([1.0], 1);
    assert(k3.kind == scientific::F_SHAPE && b1.data.len == 0);
    let b0, k4 = scientific::next_fast_len(0);
    assert(k4.kind == scientific::F_RANGE);
}

fn test_spatial() {
    let c_euclidean, e_euclidean = scientific::cdist(mx(4, 3, [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, -1.0, 5.0, 2.0, 0.0, 0.0, 1.0]), mx(2, 3, [2.0, 1.0, 1.0, 0.0, 3.0, -2.0]), "euclidean");
    assert(e_euclidean.kind == scientific::F_NONE && all_near(c_euclidean, [1.7320508075688772, 2.449489742783178, 3.0, 7.0710678118654755, 5.0990195135927845, 4.58257569495584, 2.23606797749979, 4.242640687119285]));
    let c_sqeuclidean, e_sqeuclidean = scientific::cdist(mx(4, 3, [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, -1.0, 5.0, 2.0, 0.0, 0.0, 1.0]), mx(2, 3, [2.0, 1.0, 1.0, 0.0, 3.0, -2.0]), "sqeuclidean");
    assert(e_sqeuclidean.kind == scientific::F_NONE && all_near(c_sqeuclidean, [3.0, 6.0, 9.0, 50.0, 26.0, 21.0, 5.0, 18.0]));
    let c_cityblock, e_cityblock = scientific::cdist(mx(4, 3, [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, -1.0, 5.0, 2.0, 0.0, 0.0, 1.0]), mx(2, 3, [2.0, 1.0, 1.0, 0.0, 3.0, -2.0]), "cityblock");
    assert(e_cityblock.kind == scientific::F_NONE && all_near(c_cityblock, [3.0, 4.0, 5.0, 12.0, 8.0, 7.0, 3.0, 6.0]));
    let c_chebyshev, e_chebyshev = scientific::cdist(mx(4, 3, [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, -1.0, 5.0, 2.0, 0.0, 0.0, 1.0]), mx(2, 3, [2.0, 1.0, 1.0, 0.0, 3.0, -2.0]), "chebyshev");
    assert(e_chebyshev.kind == scientific::F_NONE && all_near(c_chebyshev, [1.0, 2.0, 2.0, 5.0, 4.0, 4.0, 2.0, 3.0]));
    let c_cosine, e_cosine = scientific::cdist(mx(4, 3, [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, -1.0, 5.0, 2.0, 0.0, 0.0, 1.0]), mx(2, 3, [2.0, 1.0, 1.0, 0.0, 3.0, -2.0]), "cosine");
    assert(e_cosine.kind == scientific::F_NONE && all_near(c_cosine, [0.26970325665977857, 0.25579159246474925, 0.10185376097950138, 1.3328201177351375, 0.627322003750035, 0.4429933481039833, 0.5917517095361369, 1.5547001962252291]));
    let c_correlation, e_correlation = scientific::cdist(mx(4, 3, [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, -1.0, 5.0, 2.0, 0.0, 0.0, 1.0]), mx(2, 3, [2.0, 1.0, 1.0, 0.0, 3.0, -2.0]), "correlation");
    assert(e_correlation.kind == scientific::F_NONE && all_near(c_correlation, [1.0, 0.006600732201217263, 0.3066247547184636, 1.7953559537197221, 1.8660254037844388, 0.4039604393207302, 1.5, 1.8029550685469662]));
    let c_hamming, e_hamming = scientific::cdist(mx(4, 3, [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, -1.0, 5.0, 2.0, 0.0, 0.0, 1.0]), mx(2, 3, [2.0, 1.0, 1.0, 0.0, 3.0, -2.0]), "hamming");
    assert(e_hamming.kind == scientific::F_NONE && all_near(c_hamming, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6666666666666666, 0.6666666666666666]));
    let c_canberra, e_canberra = scientific::cdist(mx(4, 3, [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, -1.0, 5.0, 2.0, 0.0, 0.0, 1.0]), mx(2, 3, [2.0, 1.0, 1.0, 0.0, 3.0, -2.0]), "canberra");
    assert(e_canberra.kind == scientific::F_NONE && all_near(c_canberra, [1.6666666666666665, 2.2, 1.8333333333333333, 3.0, 1.9999999999999998, 2.25, 2.0, 2.0]));
    let c_braycurtis, e_braycurtis = scientific::cdist(mx(4, 3, [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, -1.0, 5.0, 2.0, 0.0, 0.0, 1.0]), mx(2, 3, [2.0, 1.0, 1.0, 0.0, 3.0, -2.0]), "braycurtis");
    assert(e_braycurtis.kind == scientific::F_NONE && all_near(c_braycurtis, [0.42857142857142855, 0.5, 0.45454545454545453, 1.5, 0.8, 0.7777777777777778, 0.6, 1.5]));
    let c_jaccard, e_jaccard = scientific::cdist(mx(4, 3, [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, -1.0, 5.0, 2.0, 0.0, 0.0, 1.0]), mx(2, 3, [2.0, 1.0, 1.0, 0.0, 3.0, -2.0]), "jaccard");
    assert(e_jaccard.kind == scientific::F_NONE && all_near(c_jaccard, [0.3333333333333333, 0.6666666666666666, 0.3333333333333333, 0.6666666666666666, 0.0, 0.3333333333333333, 0.6666666666666666, 0.5]));
    let a4 = mx(4, 3, [1.0, 2.0, 0.0, 4.0, 0.0, 3.0, -1.0, 5.0, 2.0, 0.0, 0.0, 1.0]);
    let pm, pe = scientific::pdist(a4, "minkowski", 3.0);
    assert(pe.kind == scientific::F_NONE && all_near(pm, [3.9578916096804053, 3.503398060386724, 2.154434690031884, 6.307993548663267, 4.160167646103808, 5.026525695313479]));
    let pc, pce = scientific::pdist(a4);
    assert(pce.kind == scientific::F_NONE && all_near(pc, [4.69041575982343, 4.123105625617661, 2.449489742783178, 7.14142842854285, 4.47213595499958, 5.196152422706632]));
    let sq, sqe = scientific::squareform(pc);
    assert(sqe.kind == scientific::F_NONE && sq.shape[0] == 4 && near(sq.data[6], 7.14142842854285) && near(sq.data[9], 7.14142842854285));
    let bk, bke = scientific::squareform(sq);
    assert(bke.kind == scientific::F_NONE && all_near(bk, pc.data));
    let bad1, bade = scientific::squareform(av([1.0, 2.0]));
    assert(bade.kind == scientific::F_SHAPE);
    let u0 = [1.0, 2.0, 0.0];
    let u1 = [4.0, 0.0, 3.0];
    let sv, sve = scientific::seuclidean(u0, u1, [1.0, 2.0, 3.0]);
    assert(sve.kind == scientific::F_NONE && near(sv, 3.7416573867739413));
    let mh, mhe = scientific::mahalanobis(u0, u1, [2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0]);
    assert(mhe.kind == scientific::F_NONE && near(mh, 4.795831523312719));
    let b0 = [1.0, 0.0, 1.0, 1.0, 0.0];
    let b1 = [1.0, 1.0, 0.0, 1.0, 0.0];
    let d1, d1e = scientific::dice(b0, b1);
    let d2, d2e = scientific::yule(b0, b1);
    let d3, d3e = scientific::russellrao(b0, b1);
    let d4, d4e = scientific::sokalsneath(b0, b1);
    let d5, d5e = scientific::rogerstanimoto(b0, b1);
    assert(d1e.kind == scientific::F_NONE && d2e.kind == scientific::F_NONE && d3e.kind == scientific::F_NONE && d4e.kind == scientific::F_NONE && d5e.kind == scientific::F_NONE);
    assert(near(d1, 0.3333333333333333) && near(d2, 0.6666666666666666) && near(d3, 0.6) && near(d4, 0.6666666666666666) && near(d5, 0.5714285714285714));
    let em, eme = scientific::cdist(a4, a4, "nope");
    assert(eme.kind == scientific::F_VALUE);
    let tr, tre = scientific::kdtree(a4);
    assert(tre.kind == scientific::F_NONE);
    let kd, ki, ke = scientific::kdtree_query(tr, [0.5, 0.5, 1.0], 3);
    assert(ke.kind == scientific::F_NONE && all_near(kd, [0.7071067811865476, 1.8708286933869707, 4.06201920231798]) && all_near(ki, [3.0, 0.0, 1.0]));
    let k1, i1, e1 = scientific::kdtree_query(tr, [0.5, 0.5, 1.0], 2, 1.0);
    assert(e1.kind == scientific::F_NONE && all_near(k1, [1.0, 3.0]) && all_near(i1, [3.0, 0.0]));
    let kb, kbe = scientific::kdtree_query_ball_point(tr, [0.5, 0.5, 1.0], 3.5);
    assert(kbe.kind == scientific::F_NONE && all_near(kb, [0.0, 3.0]));
    let hp = mx(6, 2, [0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 2.0, 1.0, 0.5]);
    let hv, ha, hper, he = scientific::convex_hull(hp);
    assert(he.kind == scientific::F_NONE && all_near(hv, [0.0, 1.0, 3.0, 4.0]) && near(ha, 4.0) && near(hper, 8.0));
    let hl, hl1, hl2, hle = scientific::convex_hull(mx(3, 2, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]));
    assert(hle.kind == scientific::F_VALUE);
}

fn test_interp() {
    let sp_notaknot, spe_notaknot = scientific::cubic_spline([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], "not-a-knot");
    assert(spe_notaknot.kind == scientific::F_NONE);
    let v_notaknot0, v_notaknot0e = scientific::spline_eval(sp_notaknot, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 0);
    assert(v_notaknot0e.kind == scientific::F_NONE && all_near(v_notaknot0, [-5.125380143112701, 2.7121447227191413, 1.3005502683363148, 2.0, 5.315605903398929, 4.108593917710197, 0.8827298747763836, -6.395102862254024]));
    let v_notaknot1, v_notaknot1e = scientific::spline_eval(sp_notaknot, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 1);
    assert(v_notaknot1e.kind == scientific::F_NONE && all_near(v_notaknot1, [17.109473762671435, 3.714668753726893, -1.963345557543232, 4.942300238521169, -2.3462435897435885, 0.9818395945140133, -8.238683661299945, -16.451975253428742]));
    let v_notaknot2, v_notaknot2e = scientific::spline_eval(sp_notaknot, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 2);
    assert(v_notaknot2e.kind == scientific::F_NONE && all_near(v_notaknot2, [-21.357245080500892, -12.129767441860464, 4.018318425760286, 13.245796064400711, -2.0901252236135957, 0.6850626118067975, -11.53273703041145, -15.844901610017887]));
    assert(near(scientific::spline_integrate(sp_notaknot, 0.5, 6.5), 22.039413759689925) && near(scientific::spline_integrate(sp_notaknot, -1.0, 8.0), 11.100130441264163));
    let sp_natural, spe_natural = scientific::cubic_spline([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], "natural");
    assert(spe_natural.kind == scientific::F_NONE);
    let v_natural0, v_natural0e = scientific::spline_eval(sp_natural, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 0);
    assert(v_natural0e.kind == scientific::F_NONE && all_near(v_natural0, [-0.47429925469863926, 1.9452898574206094, 1.6167854828256645, 2.0, 5.261066104990279, 4.363503996543531, 0.5778990602721948, -2.734466137394686]));
    let v_natural1, v_natural1e = scientific::spline_eval(sp_natural, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 1);
    assert(v_natural1e.kind == scientific::F_NONE && all_near(v_natural1, [2.316199503132427, 2.9233025491466837, -2.2127943400302437, 4.750783106502484, -2.3914074314106717, 1.2436379347591275, -5.753152408727587, -4.8229774249297925]));
    let v_natural2, v_natural2e = scientific::spline_eval(sp_natural, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 2);
    assert(v_natural2e.kind == scientific::F_NONE && all_near(v_natural2, [3.7943940375891105, -2.2766364225534677, 2.7796500324044073, 14.62929358392741, -1.8042773817239173, -0.14808813998704018, -0.7751458198314953, 3.875729099157482]));
    assert(near(scientific::spline_integrate(sp_natural, 0.5, 6.5), 22.19707617736012) && near(scientific::spline_integrate(sp_natural, -1.0, 8.0), 20.900039155325125));
    let sp_clamped, spe_clamped = scientific::cubic_spline([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], "clamped");
    assert(spe_clamped.kind == scientific::F_NONE);
    let v_clamped0, v_clamped0e = scientific::spline_eval(sp_clamped, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 0);
    assert(v_clamped0e.kind == scientific::F_NONE && all_near(v_clamped0, [2.8222440036900367, 1.4021369926199263, 1.8376728167281673, 2.0, 5.159841328413286, 4.773958726253929, 0.09228736162361573, 3.0953067343173437]));
    let v_clamped1, v_clamped1e = scientific::spline_eval(sp_clamped, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 1);
    assert(v_clamped1e.kind == scientific::F_NONE && all_near(v_clamped1, [-8.170472017220174, 2.3635747232472326, -2.3930565805658057, 4.629920049200493, -2.442749077490774, 1.6577101271012697, -1.7932057195571893, 13.694764760147603]));
    let v_clamped2, v_clamped2e = scientific::spline_eval(sp_clamped, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 2);
    assert(v_clamped2e.kind == scientific::F_NONE && all_near(v_clamped2, [21.629920049200493, 4.70519680196802, 1.9123001230012289, 15.64514145141451, -1.3098400984009881, -1.4922919229192324, 16.355811808118094, 35.27075645756457]));
    assert(near(scientific::spline_integrate(sp_clamped, 0.5, 6.5), 22.348245630893807) && near(scientific::spline_integrate(sp_clamped, -1.0, 8.0), 31.6361405801558));
    let sp_pc, spe_pc = scientific::pchip([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0]);
    let v_pc, v_pce = scientific::spline_eval(sp_pc, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5]);
    assert(v_pce.kind == scientific::F_NONE && all_near(v_pc, [-0.45000000000000023, 1.8828, 2.5499259259259257, 2.0, 4.352, 4.225037037037038, 0.647099999999998, -2.7375000000000007]));
    let d_pc, d_pce = scientific::spline_eval(sp_pc, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 1);
    assert(d_pce.kind == scientific::F_NONE && all_near(d_pc, [2.500000000000001, 2.734666666666667, -0.9955555555555555, 0.0, -0.96, 0.4977777777777778, -6.4830000000000005, -3.875]));
    let sp_ak, spe_ak = scientific::akima([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0]);
    let v_ak, v_ake = scientific::spline_eval(sp_ak, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5]);
    assert(v_ake.kind == scientific::F_NONE && v_ak.data[0] != v_ak.data[0] && v_ak.data[7] != v_ak.data[7]);
    assert(near(v_ak.data[1], 1.8439999999999999) && near(v_ak.data[4], 4.4) && near(v_ak.data[6], 0.6817928571428551));
    let d_ak, d_ake = scientific::spline_eval(sp_ak, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 1);
    assert(d_ake.kind == scientific::F_NONE && d_ak.data[0] != d_ak.data[0] && near(d_ak.data[1], 2.344761904761905) && near(d_ak.data[4], -1.0144927536231882) && near(d_ak.data[6], -6.703309523809525));
    let sp_pm, spe_pm = scientific::pchip([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [0.0, 1.0, 1.0, 3.0, 3.0, 3.0, 5.0]);
    let v_pm, v_pme = scientific::spline_eval(sp_pm, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5]);
    assert(v_pme.kind == scientific::F_NONE && all_near(v_pm, [-0.575, 0.4218, 1.0, 1.0, 3.0, 3.0, 4.717200000000001, 6.149999999999999]));
    let i_linear, i_lineare = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [0.3, 1.7, 2.5, 3.9, 5.2, 6.9], "linear");
    assert(i_lineare.kind == scientific::F_NONE && all_near(i_linear, [1.6, 2.533333333333333, 2.0, 4.4, 4.233333333333333, 0.4499999999999984]));
    let i_nearest, i_neareste = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [0.3, 1.7, 2.5, 3.9, 5.2, 6.9], "nearest");
    assert(i_neareste.kind == scientific::F_NONE && all_near(i_nearest, [1.0, 3.0, 2.0, 4.0, 4.0, 0.0]));
    let i_previous, i_previouse = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [0.3, 1.7, 2.5, 3.9, 5.2, 6.9], "previous");
    assert(i_previouse.kind == scientific::F_NONE && all_near(i_previous, [1.0, 3.0, 2.0, 5.0, 4.0, 4.5]));
    let i_next, i_nexte = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [0.3, 1.7, 2.5, 3.9, 5.2, 6.9], "next");
    assert(i_nexte.kind == scientific::F_NONE && all_near(i_next, [3.0, 2.0, 2.0, 4.0, 4.5, 0.0]));
    let i_cubic, i_cubice = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [0.3, 1.7, 2.5, 3.9, 5.2, 6.9], "cubic");
    assert(i_cubice.kind == scientific::F_NONE && all_near(i_cubic, [2.7121447227191404, 1.3005502683363146, 1.9999999999999996, 5.315605903398926, 4.108593917710197, 0.8827298747763835]));
    let i_nt, i_nte = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [1.75, 2.5, 0.5, 6.5], "nearest");
    assert(i_nte.kind == scientific::F_NONE && all_near(i_nt, [3.0, 2.0, 1.0, 4.5]));
    let i_ex, i_exe = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], "linear", true);
    assert(i_exe.kind == scientific::F_NONE && all_near(i_ex, [0.0, 1.5999999999999999, 2.533333333333333, 2.0, 4.4, 4.233333333333333, 0.4499999999999984, -2.25]));
    let sp3, spe3 = scientific::cubic_spline([0.0, 1.0, 3.0], [1.0, 2.0, 0.0]);
    let v3, v3e = scientific::spline_eval(sp3, [0.5, 2.0, 4.0]);
    assert(v3e.kind == scientific::F_NONE && all_near(v3, [1.6666666666666665, 1.6666666666666667, -3.000000000000001]));
    let b1, b1e = scientific::interp1d([0.0, 1.0], [0.0, 1.0], [2.0]);
    assert(b1e.kind == scientific::F_RANGE);
    let b2, b2e = scientific::cubic_spline([0.0, 0.0, 1.0], [1.0, 2.0, 3.0]);
    assert(b2e.kind == scientific::F_VALUE);
    let b3, b3e = scientific::akima([0.0, 1.0], [1.0, 2.0]);
    assert(b3e.kind == scientific::F_SHAPE);
    let b4, b4e = scientific::interp1d([0.0, 1.0], [0.0, 1.0], [0.5], "quad");
    assert(b4e.kind == scientific::F_VALUE);
}

fn mathshim_sin(x: f64) -> f64 { return mathshim::sine(x); }
fn mathshim_exp(x: f64) -> f64 { return mathshim::power_e(x); }
fn mathshim_sqrt(x: f64) -> f64 { return mathshim::root(x); }
fn mathshim_cos(x: f64) -> f64 { return mathshim::cosine(x); }

fn test_quad() {
    let q1, q1e, q1f = scientific::quad(0.0, 3.0, 1e-12, 1e-12, 200) { x => mathshim_sin(x) };
    assert(q1f.kind == scientific::F_NONE && near(q1, 1.9899924966004452));
    let q2, q2e, q2f = scientific::quad(-1.0, 2.5, 1e-12, 1e-12, 200) { x => x * x * x - 2.0 * x };
    assert(q2f.kind == scientific::F_NONE && near(q2, 4.265625));
    let q3, q3e, q3f = scientific::quad(-(1.0 / 0.0), (1.0 / 0.0), 1e-12, 1e-12, 200) { x => mathshim_exp(-x * x) };
    assert(q3f.kind == scientific::F_NONE && near(q3, 1.772453850905516));
    let q4, q4e, q4f = scientific::quad(0.0, (1.0 / 0.0), 1e-12, 1e-12, 200) { x => mathshim_exp(-x) };
    assert(q4f.kind == scientific::F_NONE && near(q4, 1.0000000000000002));
    let q5, q5e, q5f = scientific::quad(-(1.0 / 0.0), 1.0, 1e-12, 1e-12, 200) { x => 1.0 / (1.0 + x * x) };
    assert(q5f.kind == scientific::F_NONE && near(q5, 2.356194490192345));
    let q6, q6e, q6f = scientific::quad(0.0, 4.0, 1e-12, 1e-12, 200) { x => mathshim_sqrt(x) };
    assert(q6f.kind == scientific::F_NONE && near(q6, 5.333333333333335));
    let q7, q7e, q7f = scientific::quad(3.0, 0.0, 1e-12, 1e-12, 200) { x => mathshim_sin(x) };
    assert(q7f.kind == scientific::F_NONE && near(q7, -1.9899924966004452));
    let fq, fqe = scientific::fixed_quad(0.0, 2.0, 5) { x => mathshim_exp(x) };
    assert(fqe.kind == scientific::F_NONE && near(fq, 6.389056096688674));
    // Events and dense output.
    let et, ey, ee, ei, eyv, ef = scientific::solve_ivp_events(0.0, 10.0, [10.0, 0.0], 1, [true], [-1.0], 1.0e-8, 1.0e-10) { t, y => [y[1], -9.81, y[0]] };
    assert(ef.kind == scientific::F_NONE && ee.data.len == 1 && mathshim_abs(ee.data[0] - 1.42784312) < 1.0e-7);
    assert(mathshim_abs(et.data[et.data.len - 1] - 1.4278431229270645) < 1.0e-7 && mathshim_abs(eyv.data[1] + 14.007141035914504) < 1.0e-6 && ei.data[0] == 0.0);
    let ot, oy, oe, oi, oyv, of = scientific::solve_ivp_events(0.0, 10.0, [1.0, 0.0], 1, [false], [0.0], 1.0e-8, 1.0e-10) { t, y => [y[1], -y[0], y[0]] };
    assert(of.kind == scientific::F_NONE && oe.data.len == 3 && mathshim_abs(oe.data[0] - 1.5707963264325224) < 1.0e-7 && mathshim_abs(oe.data[1] - 4.712388979391328) < 1.0e-7 && mathshim_abs(oe.data[2] - 7.85398163235004) < 1.0e-7);
    assert(ot.data[ot.data.len - 1] == 10.0);
    let ob, obb, obe, obi, obv, obf = scientific::solve_ivp_events(0.0, 10.0, [1.0, 0.0], 2, [false], [0.0]) { t, y => [y[1], -y[0], y[0], y[1]] };
    assert(obf.kind == scientific::F_SHAPE);
    let dt, dy, dd, df = scientific::solve_ivp_dense(0.0, 6.0, [1.0, 0.0], 1.0e-8, 1.0e-10) { t, y => [y[1], -y[0]] };
    assert(df.kind == scientific::F_NONE);
    let dv, dvf = scientific::dense_eval(dd, [0.5, 1.7, 5.2]);
    assert(dvf.kind == scientific::F_NONE && all_close(dv, [0.8775825612156084, -0.47942553943018557, -0.1288444943415755, -0.9916648084454521, 0.46851667032694333, 0.883454649492061], 1.0e-7));
    let t_rk45, y_rk45, e_rk45 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "RK45") { t, y => [y[1], -y[0]] };
    assert(e_rk45.kind == scientific::F_NONE && all_near(t_rk45, [0.0, 0.000999000999000999, 0.010989010989010988, 0.11088911088911087, 0.7601505581783583, 1.6934656227081029, 2.6818921292145386, 3.662498620296199, 4.577872200228697, 5.0]) && all_near(y_rk45, [1.0, 0.0, 0.9999995009985435, -0.0009990008328333421, 0.9999396214263466, -0.010988789821184265, 0.9938580997401778, -0.11066199453652523, 0.7247097637593363, -0.6890377126475217, -0.12260764766130006, -0.992423097760466, -0.8964129826445695, -0.4431387530172439, -0.8668812844862507, 0.49842806151422436, -0.1330055785521448, 0.9910483076876293, 0.28470530202789335, 0.9585445205284323]));
    let t_rk23, y_rk23, e_rk23 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "RK23") { t, y => [y[1], -y[0]] };
    assert(e_rk23.kind == scientific::F_NONE && all_near(t_rk23, [0.0, 0.000999000999000999, 0.010989010989010988, 0.09501646775614085, 0.2631021017304901, 0.49917840208969655, 0.7929707310047646, 1.126751987501435, 1.3999644810686591, 1.6015049189722004, 1.8030453568757416, 2.028858031827757, 2.314302675177404, 2.6442877009293686, 2.927512396909703, 3.144063408764921, 3.360614420620139, 3.5818272726373714, 3.8635031091628056, 4.191437175612456, 4.478844465694581, 4.701414776242469, 4.923985086790357, 5.0]) && all_near(y_rk23, [1.0, 0.0, 0.999999500998502, -0.0009990008328333339, 0.9999396210113036, -0.010988789819940078, 0.9954872539421785, -0.09487350436657084, 0.9655528300346052, -0.2600724810893409, 0.877820072249416, -0.4786521408210731, 0.7013346669277642, -0.7121817553737404, 0.4289678535496003, -0.9022553607215416, 0.16951701028118504, -0.9843222226689229, -0.030963499382748477, -0.99826459975658, -0.23016332551647012, -0.9717924191750676, -0.44187325017090273, -0.8954838148545725, -0.6760115511939843, -0.7345848805754808, -0.8772088761745034, -0.47556509459531565, -0.9749167172987311, -0.21136544132431817, -0.9974713425177815, 0.0030596324978935663, -0.9734260739361247, 0.21730309822942753, -0.9019305259435636, 0.4255643433843891, -0.7478642959762679, 0.65937450075388, -0.49529560412761336, 0.8647741342190143, -0.22971849702806812, 0.9694494775598642, -0.010039438925150662, 0.9961437919370207, 0.21009074719856316, 0.9736865664941141, 0.28342718737507855, 0.9549188059367262]));
    let t_b, y_b, e_b = scientific::solve_ivp(1.0, -1.0, [3.0], 1e-8, 1e-10) { t, y => [-2.0 * y[0]] };
    assert(e_b.kind == scientific::F_NONE && all_near(t_b, [1.0, 0.9924163715019694, 0.9452288909139589, 0.896981996408003, 0.8487089585227905, 0.8004376140514565, 0.7521684987542352, 0.7039014265324454, 0.6556362101002151, 0.6073726788671658, 0.5591106779130687, 0.5108500665476304, 0.46259071700822413, 0.4143325132786344, 0.3660753500048519, 0.3178191315092954, 0.26956377091857076, 0.22130918934202373, 0.17305531513896355, 0.12480208325590475, 0.07654943462509203, 0.028297315609388474, -0.01995432250344239, -0.06820552394119547, -0.11645632886113524, -0.16470677373454357, -0.21295689167722867, -0.2612067127571098, -0.3094562642802536, -0.3577055710382285, -0.40595465554309856, -0.4542035382382169, -0.5024522376848576, -0.5507007707400633, -0.5989491527073207, -0.6471973974830382, -0.6954455176825663, -0.7436935247701385, -0.7919414291462445, -0.8401892402612638, -0.8884369666904695, -0.9366846162281081, -0.9849321959426126, -1.0]) && all_near(y_b, [3.0, 3.045848590731626, 3.3473015273313025, 3.6863923475153633, 4.060046278395815, 4.471558749174671, 4.924758766904875, 5.42386920767848, 5.973540915615807, 6.578895883708401, 7.245574976988688, 7.979790495716302, 8.788384063269335, 9.678890378181594, 10.659607424632133, 11.739673795515595, 12.929153848016647, 14.239131486346215, 15.681813444640502, 17.270643032602173, 19.020425403826767, 20.947465514463808, 23.069720057615154, 25.40696478827133, 27.980978799267604, 30.815747465454802, 33.93768594564593, 37.37588532627237, 41.16238370058473, 45.332464707976754, 49.92498631696867, 54.98274291465527, 60.55286407628073, 66.68725373221251, 73.44307382238468, 80.8832769474832, 89.07719297811569, 98.10117509189514, 108.03931125016106, 118.98420775588633, 131.03785218216046, 144.31256372395978, 158.9320398150422, 163.7944513686982]));
    let bq1, bq2, bqf = scientific::quad(0.0, 1.0, -1.0) { x => x };
    assert(bqf.kind == scientific::F_VALUE);
    let hq1, hq2, hqf = scientific::quad(0.0, 1.0, 1e-300, 1e-300, 3) { x => mathshim_sin(1.0 / x) };
    assert(hqf.kind == scientific::F_LIMIT);
    let bi1, bi2, bif = scientific::solve_ivp(0.0, 1.0, [1.0], 0.001, 1e-6, "LSODA") { t, y => [y[0]] };
    assert(bif.kind == scientific::F_VALUE);
    let bi3, bi4, bi5 = scientific::solve_ivp(0.0, 1.0, [1.0, 2.0]) { t, y => [y[0]] };
    assert(bi5.kind == scientific::F_SHAPE);
}

fn test_programs() {
    // Mixed integer programs.
    let kx, kv, kf = scientific::milp([-8.0, -11.0, -6.0, -4.0], [true, true, true, true], [5.0, 7.0, 4.0, 3.0], [14.0], [], [], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]);
    assert(kf.kind == scientific::F_NONE && rel_close(kx, [0.0, 1.0, 1.0, 1.0], 1.0e-9) && mathshim_abs(kv + 21.0) < 1.0e-9);
    let mx2, mv2, mf2 = scientific::milp([-1.0, -2.0], [true, false], [1.0, 1.0, 1.0, -1.0], [3.5, 1.2], [], []);
    assert(mf2.kind == scientific::F_NONE && rel_close(mx2, [0.0, 3.5], 1.0e-9) && mathshim_abs(mv2 + 7.0) < 1.0e-9);
    let mx3, mv3, mf3 = scientific::milp([1.0, 1.0], [true, true], [], [], [2.0, 2.0], [3.0]);
    assert(mf3.kind == scientific::F_VALUE);
    let mx4, mv4, mf4 = scientific::milp([1.0, 1.0], [true], [], [], [], []);
    assert(mf4.kind == scientific::F_SHAPE);
    let l1, l1v, l1f = scientific::linprog([-1.0, -2.0], [1.0, 1.0, 1.0, 3.0, 2.0, 1.0], [4.0, 6.0, 5.0], [], [], [], []);
    assert(l1f.kind == scientific::F_NONE && all_near(l1, [1.8000000000000007, 1.4]) && near(l1v, -4.6000000000000005));
    let l2, l2v, l2f = scientific::linprog([2.0, 3.0, 1.0], [-1.0, -1.0, -1.0], [-10.0], [1.0, -1.0, 0.0], [2.0], [], []);
    assert(l2f.kind == scientific::F_NONE && all_near(l2, [2.0, 0.0, 8.0]) && near(l2v, 12.0));
    let l3, l3v, l3f = scientific::linprog([1.0, 1.0], [-1.0, -2.0], [-3.0], [], [], [-5.0, -(1.0 / 0.0)], [(1.0 / 0.0), 10.0]);
    assert(l3f.kind == scientific::F_NONE && all_near(l3, [-5.0, 4.0]) && near(l3v, -1.0));
    let l4, l4v, l4f = scientific::linprog([-3.0, -5.0, -4.0], [2.0, 3.0, 0.0, 0.0, 2.0, 5.0, 3.0, 2.0, 4.0], [8.0, 10.0, 15.0], [], [], [0.0, 1.0, 0.0], [2.0, 3.0, (1.0 / 0.0)]);
    assert(l4f.kind == scientific::F_NONE && all_near(l4, [2.0, 1.3333333333333333, 1.4666666666666668]) && near(l4v, -18.53333333333333));
    let l5, l5v, l5f = scientific::linprog([1.0, -1.0], [], [], [1.0, 1.0], [2.0], [-(1.0 / 0.0), -1.0], [(1.0 / 0.0), 3.0]);
    assert(l5f.kind == scientific::F_NONE && all_near(l5, [-1.0, 3.0]) && near(l5v, -4.0));
    let nn, nnr, nnf = scientific::nnls(mx(4, 2, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0]), [2.0, 1.0, -1.0, 4.0]);
    assert(nnf.kind == scientific::F_NONE && all_near(nn, [1.8333333333333333, 0.0]) && near(nnr, 1.3540064007726602));
    let n2, n2r, n2f = scientific::nnls(mx(5, 4, [1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.5, 2.0, 1.0, 1.0]), [3.0, -2.0, 1.0, 0.5, 2.0]);
    assert(n2f.kind == scientific::F_NONE && all_near(n2, [0.6666666666666664, 0.5833333333333336, 0.0, 0.0]) && near(n2r, 2.993047499344661));
    let a1r, a1c, a1f = scientific::linear_sum_assignment(mx(3, 3, [4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0]), false);
    assert(a1f.kind == scientific::F_NONE && all_near(a1r, [0.0, 1.0, 2.0]) && all_near(a1c, [1.0, 0.0, 2.0]));
    let a2r, a2c, a2f = scientific::linear_sum_assignment(mx(3, 3, [4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0]), true);
    assert(a2f.kind == scientific::F_NONE && all_near(a2r, [0.0, 1.0, 2.0]) && all_near(a2c, [0.0, 2.0, 1.0]));
    let a3r, a3c, a3f = scientific::linear_sum_assignment(mx(2, 4, [4.0, 1.0, 3.0, 9.0, 2.0, 0.0, 5.0, 1.0]), false);
    assert(a3f.kind == scientific::F_NONE && all_near(a3r, [0.0, 1.0]) && all_near(a3c, [1.0, 3.0]));
    let a4r, a4c, a4f = scientific::linear_sum_assignment(mx(4, 2, [4.0, 1.0, 2.0, 0.0, 3.0, 2.0, 0.5, 7.0]), false);
    assert(a4f.kind == scientific::F_NONE && all_near(a4r, [1.0, 3.0]) && all_near(a4c, [1.0, 0.0]));
    let ag, agf = scientific::approx_fprime([1.0, 2.0, -0.5]) { x => x[0] * x[0] * x[1] + mathshim_sin(x[2]) };
    assert(agf.kind == scientific::F_NONE && all_near(ag, [4.000000029802322, 1.0, 0.8775825649499893]));
    let br, brf = scientific::bisect(0.0, 2.0) { x => x * x - 2.0 };
    assert(brf.kind == scientific::F_NONE && near(br, 1.4142135623730951));
    let rr, rrf = scientific::ridder(0.0, 3.0) { x => x * x * x - 2.0 * x - 5.0 + 4.0 };
    assert(rrf.kind == scientific::F_NONE && near(rr, 1.6180339887498911));
    let i1, i1v, i1f = scientific::linprog([1.0], [-1.0], [-1.0], [], [], [0.0], [0.5]);
    assert(i1f.kind == scientific::F_VALUE);
    let u1, u1v, u1f = scientific::linprog([-1.0], [], [], [], []);
    assert(u1f.kind == scientific::F_VALUE);
    let s1, s1v, s1f = scientific::linprog([1.0, 2.0], [1.0], [1.0], [], []);
    assert(s1f.kind == scientific::F_SHAPE);
    let c1, c2, cf = scientific::linear_sum_assignment(mx(2, 2, [1.0, 1.0 / 0.0, 1.0, 2.0]));
    assert(cf.kind == scientific::F_VALUE);
    let nbr, nbf = scientific::bisect(1.0, 2.0) { x => x * x - 2.0 + 5.0 };
    assert(nbf.kind == scientific::F_VALUE);
}

fn test_dsp() {
    let pg, pgf = scientific::from_rows(4, 4, [9.0, 0.0, 60.0, 0.0, 0.0, 60.0, 0.0, 708.0, 60.0, 0.0, 708.0, 0.0, 0.0, 708.0, 0.0, 9780.0]);
    let pz, pzf = scientific::solve(pg, av([0.0, 0.0, 2.0, 0.0]));
    assert(pgf.kind == scientific::F_NONE && pzf.kind == scientific::F_NONE && near(pz.data[0], -0.04329004329004329) && near(pz.data[2], 0.006493506493506493));
    let fw1, fw1f = scientific::firwin(21, [0.3]);
    assert(fw1f.kind == scientific::F_NONE && all_near(fw1, [9.34176167519117e-19, 0.0029289031790009096, 0.006342347322322931, 0.003783041971756365, -0.012387848056448455, -0.034326572827062196, -0.031859861089054115, 0.026531216606606865, 0.13786316455699657, 0.25134767914641787, 0.2995558583789267, 0.25134767914641787, 0.13786316455699657, 0.026531216606606865, -0.031859861089054115, -0.034326572827062196, -0.012387848056448455, 0.003783041971756365, 0.006342347322322931, 0.0029289031790009096, 9.34176167519117e-19]));
    let fw2, fw2f = scientific::firwin(21, [0.3], false);
    assert(fw2f.kind == scientific::F_NONE && all_near(fw2, [-4.049811415111009e-18, -0.002930143801192959, -0.006345033808135852, -0.0037846443892952065, 0.012393095290061624, 0.03434111284611499, 0.03187335626055248, -0.026542454675635645, -0.13792156051308088, -0.25145414477181155, 0.6992597359575647, -0.25145414477181155, -0.13792156051308088, -0.026542454675635645, 0.03187335626055248, 0.03434111284611499, 0.012393095290061624, -0.0037846443892952065, -0.006345033808135852, -0.002930143801192959, -4.049811415111009e-18]));
    let fw3, fw3f = scientific::firwin(31, [0.2, 0.5], false, "hann");
    assert(fw3f.kind == scientific::F_NONE && all_near(fw3, [-0.0, -0.00014473366026072868, 5.1347477247545135e-05, -0.0023878089307026035, -0.0075341704804547105, 6.7617430750278495e-18, 0.019230757412242647, 0.016793758106393993, -0.001218297966068349, 0.02022984189547462, 0.04732613340132475, -0.03869267749918786, -0.1855970250836558, -0.14354725180212657, 0.1286358403009461, 0.2973588660328248, 0.1286358403009461, -0.14354725180212657, -0.1855970250836558, -0.038692677499187866, 0.047326133401324764, 0.02022984189547462, -0.0012182979660683496, 0.016793758106394, 0.019230757412242647, 6.761743075027862e-18, -0.007534170480454718, -0.0023878089307026035, 5.13474772475452e-05, -0.00014473366026072942, -0.0]));
    let fw4, fw4f = scientific::firwin(25, [0.2, 0.5], true, "blackman");
    assert(fw4f.kind == scientific::F_NONE && all_near(fw4, [-3.5138955038598284e-19, 0.00029140643875987156, -1.7949927528714434e-18, -0.003745118805866899, -0.0049374556863222225, 0.000494354333258714, -0.01064118201960838, -0.030678215113981513, 0.029576226495676153, 0.1607246205466832, 0.13566801978202178, -0.1280391373185936, 0.7025729626959456, -0.12803913731859362, 0.1356680197820218, 0.16072462054668324, 0.029576226495676167, -0.030678215113981545, -0.01064118201960838, 0.0004943543332587145, -0.004937455686322234, -0.003745118805866899, -1.7949927528714465e-18, 0.0002914064387598722, -3.5138955038598284e-19]));
    let fw5, fw5f = scientific::firwin(15, [1000.0], true, "hamming", true, 8000.0);
    assert(fw5f.kind == scientific::F_NONE && all_near(fw5, [-0.0025775965600865667, -0.006674494920200382, -0.011421090886231026, 4.273721637740298e-18, 0.04829253250636341, 0.1318594485716218, 0.21526539121376717, 0.2505116201495313, 0.21526539121376717, 0.1318594485716218, 0.04829253250636341, 4.273721637740298e-18, -0.011421090886231026, -0.006674494920200382, -0.0025775965600865667]));
    let fw6, fw6f = scientific::firwin(21, [0.3], true, "kaiser", true, 2.0, 5.0);
    assert(fw6f.kind == scientific::F_NONE && all_near(fw6, [4.293432898844986e-19, 0.002720928098159983, 0.006780773912375082, 0.004040120705800866, -0.012938749717734762, -0.03519786593893117, -0.032286080016798197, 0.026709108004773673, 0.13832830319814807, 0.2518339613107667, 0.3000190008868794, 0.2518339613107667, 0.13832830319814807, 0.026709108004773673, -0.032286080016798197, -0.03519786593893117, -0.012938749717734762, 0.004040120705800866, 0.006780773912375082, 0.002720928098159983, 4.293432898844986e-19]));
    let zw, zr, zi, zf = scientific::freqz([0.2, 0.3, 0.2], [1.0, -0.4, 0.1], 16);
    assert(zf.kind == scientific::F_NONE && all_near(zw, [0.0, 0.19634954084936207, 0.39269908169872414, 0.5890486225480862, 0.7853981633974483, 0.9817477042468103, 1.1780972450961724, 1.3744467859455345, 1.5707963267948966, 1.7671458676442586, 1.9634954084936207, 2.1598449493429825, 2.356194490192345, 2.552544031041707, 2.748893571891069, 2.945243112740431]) && all_near(zr, [1.0, 0.9558703822250522, 0.8278830514729616, 0.6323091093921817, 0.40202601153348333, 0.18237446967487966, 0.01372138669228167, -0.0861515662909957, -0.12371134020618557, -0.11791721853266074, -0.08834405326697602, -0.04969604916721771, -0.01127527301866199, 0.02165598652001318, 0.046367130716893946, 0.06155463892161986]) && all_near(zi, [0.0, -0.24722625144999696, -0.46268125700051527, -0.6143659640622462, -0.6771730124445141, -0.6464108377705035, -0.5445465676681835, -0.4101742416852979, -0.27835051546391754, -0.1693563852684368, -0.08947825993234672, -0.03677942516941148, -0.006092234202688692, 0.008236834638692177, 0.011275793322261266, 0.007308207039902913]));
    let fc_full, fc_fullf = scientific::fftconvolve([1.0, 3.0, -2.0, 4.0, 0.5, 7.0, -1.0, 2.0, 2.5, -3.0, 1.0], [0.5, 1.0, -0.5, 2.0], "full");
    assert(fc_fullf.kind == scientific::F_NONE && all_near(fc_full, [0.5000000000000009, 2.4999999999999987, 1.5, 0.5, 11.249999999999998, -1.9999999999999976, 14.249999999999996, -2.499999999999999, 17.749999999999996, -1.9999999999999991, 0.2500000000000007, 7.499999999999999, -6.499999999999998, 2.0]));
    let fc_same, fc_samef = scientific::fftconvolve([1.0, 3.0, -2.0, 4.0, 0.5, 7.0, -1.0, 2.0, 2.5, -3.0, 1.0], [0.5, 1.0, -0.5, 2.0], "same");
    assert(fc_samef.kind == scientific::F_NONE && all_near(fc_same, [2.4999999999999987, 1.5, 0.5, 11.249999999999998, -1.9999999999999976, 14.249999999999996, -2.499999999999999, 17.749999999999996, -1.9999999999999991, 0.2500000000000007, 7.499999999999999]));
    let fc_valid, fc_validf = scientific::fftconvolve([1.0, 3.0, -2.0, 4.0, 0.5, 7.0, -1.0, 2.0, 2.5, -3.0, 1.0], [0.5, 1.0, -0.5, 2.0], "valid");
    assert(fc_validf.kind == scientific::F_NONE && all_near(fc_valid, [0.5, 11.249999999999998, -1.9999999999999976, 14.249999999999996, -2.499999999999999, 17.749999999999996, -1.9999999999999991, 0.2500000000000007]));
    let fc_v2, fc_v2f = scientific::fftconvolve([0.5, 1.0, -0.5, 2.0], [1.0, 3.0, -2.0, 4.0, 0.5, 7.0, -1.0, 2.0, 2.5, -3.0, 1.0], "valid");
    assert(fc_v2f.kind == scientific::F_NONE && all_near(fc_v2, [0.5, 11.249999999999998, -1.9999999999999976, 14.249999999999996, -2.499999999999999, 17.749999999999996, -1.9999999999999991, 0.2500000000000007]));
    let mf, mff = scientific::medfilt([1.0, 3.0, -2.0, 4.0, 0.5, 7.0, -1.0, 2.0, 2.5, -3.0, 1.0], 5);
    assert(mff.kind == scientific::F_NONE && all_near(mf, [0.0, 1.0, 1.0, 3.0, 0.5, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0]));
    let sc1, sc1f = scientific::savgol_coeffs(7, 2);
    assert(sc1f.kind == scientific::F_NONE && all_near(sc1, [-0.09523809523809523, 0.14285714285714307, 0.285714285714286, 0.33333333333333365, 0.2857142857142859, 0.142857142857143, -0.09523809523809527]));
    let sc2, sc2f = scientific::savgol_coeffs(7, 3, 1);
    assert(sc2f.kind == scientific::F_NONE && all_near(sc2, [-0.08730158730158721, 0.2658730158730158, 0.23015873015873023, 5.204170427930421e-17, -0.23015873015873023, -0.2658730158730159, 0.08730158730158737]));
    let sc3, sc3f = scientific::savgol_coeffs(9, 3, 2, 0.5);
    assert(sc3f.kind == scientific::F_NONE && all_near(sc3, [0.2424242424242424, 0.060606060606060615, -0.0692640692640693, -0.14718614718614717, -0.17316017316017315, -0.14718614718614717, -0.06926406926406925, 0.060606060606060615, 0.2424242424242424]));
    let sg_interp, sg_interpf = scientific::savgol_filter([0.3, 0.3276536078418991, 0.2223589125935188, 0.4638590687632922, 0.9820259861980096, 1.2419609070739932, 1.0136816127728492, 0.6996694662346034, 0.7410021495975131, 0.9723520984840844, 0.8707061785390153, 0.3330266611024162, -0.15713327949527736, -0.2191256136890593, -0.11187748660204291, -0.3328168277981605, -0.8679804930592206, -1.1932138175073619, -1.0259015069623527, -0.7269469096485195, -0.7648484779645464, -1.027450276190937, -0.9897716712574192, -0.491826398794253, 0.013617992050752648], 7, 3, 0, 1.0, "interp");
    assert(sg_interpf.kind == scientific::F_NONE && all_near(sg_interp, [0.36902055538705236, 0.15236218380603747, 0.2819875201614215, 0.5978477711092478, 0.8934560564931378, 1.0586588297121307, 1.0020102667692072, 0.8744411202411712, 0.8439206034798812, 0.8505574463649376, 0.7008445101194162, 0.362574007160642, 0.027661374402489003, -0.15013170302162673, -0.25951257428529645, -0.4799911615823198, -0.7977732979650343, -1.0067441907565642, -0.9942965531792799, -0.8942674395029838, -0.883679122040222, -0.9215059868654039, -0.921585005747395, -0.6474378520571571, 0.07248917111392894]));
    let sg_nearest, sg_nearestf = scientific::savgol_filter([0.3, 0.3276536078418991, 0.2223589125935188, 0.4638590687632922, 0.9820259861980096, 1.2419609070739932, 1.0136816127728492, 0.6996694662346034, 0.7410021495975131, 0.9723520984840844, 0.8707061785390153, 0.3330266611024162, -0.15713327949527736, -0.2191256136890593, -0.11187748660204291, -0.3328168277981605, -0.8679804930592206, -1.1932138175073619, -1.0259015069623527, -0.7269469096485195, -0.7648484779645464, -1.027450276190937, -0.9897716712574192, -0.491826398794253, 0.013617992050752648], 7, 3, 0, 1.0, "nearest");
    assert(sg_nearestf.kind == scientific::F_NONE && all_near(sg_nearest, [0.28120382130025573, 0.24548828401658418, 0.33655926629675365, 0.5978477711092478, 0.8934560564931378, 1.0586588297121307, 1.0020102667692072, 0.8744411202411712, 0.8439206034798812, 0.8505574463649376, 0.7008445101194162, 0.362574007160642, 0.027661374402489003, -0.15013170302162673, -0.25951257428529645, -0.4799911615823198, -0.7977732979650343, -1.0067441907565642, -0.9942965531792799, -0.8942674395029838, -0.883679122040222, -0.9215059868654039, -0.8033854938694731, -0.516130607018702, -0.17498671264025553]));
    let sg_mirror, sg_mirrorf = scientific::savgol_filter([0.3, 0.3276536078418991, 0.2223589125935188, 0.4638590687632922, 0.9820259861980096, 1.2419609070739932, 1.0136816127728492, 0.6996694662346034, 0.7410021495975131, 0.9723520984840844, 0.8707061785390153, 0.3330266611024162, -0.15713327949527736, -0.2191256136890593, -0.11187748660204291, -0.3328168277981605, -0.8679804930592206, -1.1932138175073619, -1.0259015069623527, -0.7269469096485195, -0.7648484779645464, -1.027450276190937, -0.9897716712574192, -0.491826398794253, 0.013617992050752648], 7, 3, 0, 1.0, "mirror");
    assert(sg_mirrorf.kind == scientific::F_NONE && all_near(sg_mirror, [0.2624076426005113, 0.2568331886993775, 0.3339255893594299, 0.5978477711092478, 0.8934560564931378, 1.0586588297121307, 1.0020102667692072, 0.8744411202411712, 0.8439206034798812, 0.8505574463649376, 0.7008445101194162, 0.362574007160642, 0.027661374402489003, -0.15013170302162673, -0.25951257428529645, -0.4799911615823198, -0.7977732979650343, -1.0067441907565642, -0.9942965531792799, -0.8942674395029838, -0.883679122040222, -0.9215059868654039, -0.7552479328366153, -0.49277602825292455, -0.3635914173312637]));
    let sg_constant, sg_constantf = scientific::savgol_filter([0.3, 0.3276536078418991, 0.2223589125935188, 0.4638590687632922, 0.9820259861980096, 1.2419609070739932, 1.0136816127728492, 0.6996694662346034, 0.7410021495975131, 0.9723520984840844, 0.8707061785390153, 0.3330266611024162, -0.15713327949527736, -0.2191256136890593, -0.11187748660204291, -0.3328168277981605, -0.8679804930592206, -1.1932138175073619, -1.0259015069623527, -0.7269469096485195, -0.7648484779645464, -1.027450276190937, -0.9897716712574192, -0.491826398794253, 0.013617992050752648], 7, 3, 0, 1.0, "constant");
    assert(sg_constantf.kind == scientific::F_NONE && all_near(sg_constant, [0.1812038213002557, 0.2312025697308699, 0.3651306948681822, 0.5978477711092478, 0.8934560564931378, 1.0586588297121307, 1.0020102667692072, 0.8744411202411712, 0.8439206034798812, 0.8505574463649376, 0.7008445101194162, 0.362574007160642, 0.027661374402489003, -0.15013170302162673, -0.25951257428529645, -0.4799911615823198, -0.7977732979650343, -1.0067441907565642, -0.9942965531792799, -0.8942674395029838, -0.883679122040222, -0.9215059868654039, -0.8020885422455919, -0.5167790828306427, -0.17952604332383976]));
    let sg_wrap, sg_wrapf = scientific::savgol_filter([0.3, 0.3276536078418991, 0.2223589125935188, 0.4638590687632922, 0.9820259861980096, 1.2419609070739932, 1.0136816127728492, 0.6996694662346034, 0.7410021495975131, 0.9723520984840844, 0.8707061785390153, 0.3330266611024162, -0.15713327949527736, -0.2191256136890593, -0.11187748660204291, -0.3328168277981605, -0.8679804930592206, -1.1932138175073619, -1.0259015069623527, -0.7269469096485195, -0.7648484779645464, -1.027450276190937, -0.9897716712574192, -0.491826398794253, 0.013617992050752648], 7, 3, 0, 1.0, "wrap");
    assert(sg_wrapf.kind == scientific::F_NONE && all_near(sg_wrap, [0.2090977307496174, 0.2799886065756682, 0.36383374324430107, 0.5978477711092478, 0.8934560564931378, 1.0586588297121307, 1.0020102667692072, 0.8744411202411712, 0.8439206034798812, 0.8505574463649376, 0.7008445101194162, 0.362574007160642, 0.027661374402489003, -0.15013170302162673, -0.25951257428529645, -0.4799911615823198, -0.7977732979650343, -1.0067441907565642, -0.9942965531792799, -0.8942674395029838, -0.883679122040222, -0.9215059868654039, -0.8306599708170205, -0.5051270454822521, -0.06818113864104636]));
    let sg_d, sg_df = scientific::savgol_filter([0.3, 0.3276536078418991, 0.2223589125935188, 0.4638590687632922, 0.9820259861980096, 1.2419609070739932, 1.0136816127728492, 0.6996694662346034, 0.7410021495975131, 0.9723520984840844, 0.8707061785390153, 0.3330266611024162, -0.15713327949527736, -0.2191256136890593, -0.11187748660204291, -0.3328168277981605, -0.8679804930592206, -1.1932138175073619, -1.0259015069623527, -0.7269469096485195, -0.7648484779645464, -1.027450276190937, -0.9897716712574192, -0.491826398794253, 0.013617992050752648], 7, 3, 1, 0.25);
    assert(sg_df.kind == scientific::F_NONE && all_near(sg_d, [-1.7725992933221317, -0.0673668748886247, 0.997670370169059, 1.4225124418509227, 1.4280028357047037, 0.09881225012429662, -0.9331483694408783, -0.49889148049516674, 0.4163935236218692, 0.13834433912141875, -1.2228961485560814, -1.9155610045269744, -1.0975273194309065, -0.05929477767253738, -0.32767560631265036, -1.428650524223285, -1.5868343850185134, -0.3365201981012802, 0.7815120878063037, 0.4591529979433904, -0.48315903934790594, -0.3200301581276047, 0.4337671383440782, 1.8737792224502354, 4.0000060941908675]));
    let pk1, pk1f = scientific::find_peaks([0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 3.0, 1.0, 5.0, 4.0, 4.0, 6.0, 0.5, 0.7, 0.2, 3.0, 2.0, 1.0, 4.0, 0.0]);
    assert(pk1f.kind == scientific::F_NONE && all_near(pk1, [1.0, 4.0, 7.0, 9.0, 12.0, 14.0, 16.0, 19.0]));
    let pk2, pk2f = scientific::find_peaks([0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 3.0, 1.0, 5.0, 4.0, 4.0, 6.0, 0.5, 0.7, 0.2, 3.0, 2.0, 1.0, 4.0, 0.0], 2.0);
    assert(pk2f.kind == scientific::F_NONE && all_near(pk2, [4.0, 7.0, 9.0, 12.0, 16.0, 19.0]));
    let pk3, pk3f = scientific::find_peaks([0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 3.0, 1.0, 5.0, 4.0, 4.0, 6.0, 0.5, 0.7, 0.2, 3.0, 2.0, 1.0, 4.0, 0.0], 0.0 / 0.0, 3.0);
    assert(pk3f.kind == scientific::F_NONE && all_near(pk3, [1.0, 4.0, 9.0, 12.0, 16.0, 19.0]));
    let pk4, pk4f = scientific::find_peaks([0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 3.0, 1.0, 5.0, 4.0, 4.0, 6.0, 0.5, 0.7, 0.2, 3.0, 2.0, 1.0, 4.0, 0.0], 0.0 / 0.0, 0.0, 2.0);
    assert(pk4f.kind == scientific::F_NONE && all_near(pk4, [4.0, 7.0, 12.0, 16.0, 19.0]));
    let pk5, pk5f = scientific::find_peaks([0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 3.0, 1.0, 5.0, 4.0, 4.0, 6.0, 0.5, 0.7, 0.2, 3.0, 2.0, 1.0, 4.0, 0.0], 1.0, 2.0, 0.5);
    assert(pk5f.kind == scientific::F_NONE && all_near(pk5, [1.0, 4.0, 7.0, 9.0, 12.0, 16.0, 19.0]));
    let pp, ppf = scientific::peak_prominences([0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 3.0, 1.0, 5.0, 4.0, 4.0, 6.0, 0.5, 0.7, 0.2, 3.0, 2.0, 1.0, 4.0, 0.0], [1.0, 4.0, 7.0, 9.0, 12.0, 14.0, 16.0, 19.0]);
    assert(ppf.kind == scientific::F_NONE && all_near(pp, [1.0, 2.0, 2.0, 1.0, 6.0, 0.19999999999999996, 2.0, 3.8]));
    let pd1f, pd1p, pd1e = scientific::periodogram([0.8, 0.825118635510321, 0.7667649357308379, 1.4351187926374929, 1.325938592314405, 0.006107373853762976, -0.6522695660031785, -0.2657998949072589, -0.3831431188056557, -0.7401987785254835, 0.1454915028125274, 1.3789288776568174, 1.2824517760598095, 0.7520431520810179, 0.9814759095853356, 0.7755282581475771, -0.50067489807501, -1.0732204790461601, -0.33234452142080745, 0.054488300671869105]);
    assert(pd1e.kind == scientific::F_NONE && all_near(pd1f, [0.0, 0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.30000000000000004, 0.35000000000000003, 0.4, 0.45, 0.5]) && all_near(pd1p, [1.7333369499485123e-31, 0.03622434387966118, 9.47608318658677, 0.06944862457564174, 0.1601399113760153, 1.5630346620482092, 0.5599806672167472, 0.0644731351727721, 0.020830783767041745, 0.010465452604919765, 0.004077693272727992]));
    let pd2f, pd2p, pd2e = scientific::periodogram([0.8, 0.825118635510321, 0.7667649357308379, 1.4351187926374929, 1.325938592314405, 0.006107373853762976, -0.6522695660031785, -0.2657998949072589, -0.3831431188056557, -0.7401987785254835, 0.1454915028125274, 1.3789288776568174, 1.2824517760598095, 0.7520431520810179, 0.9814759095853356, 0.7755282581475771, -0.50067489807501, -1.0732204790461601, -0.33234452142080745, 0.054488300671869105], 2.0, "hann");
    assert(pd2e.kind == scientific::F_NONE && all_near(pd2f, [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0]) && all_near(pd2p, [0.005756433759959763, 0.8318795578569342, 3.3580173408332294, 0.7924687802807604, 0.042238687955419046, 0.6763929859811776, 0.5182355825215812, 0.012165141222585373, 0.00031400805110706525, 2.7083702205284593e-05, 1.6267669950949622e-06]));
    let pd3f, pd3p, pd3e = scientific::periodogram([0.8, 0.825118635510321, 0.7667649357308379, 1.4351187926374929, 1.325938592314405, 0.006107373853762976, -0.6522695660031785, -0.2657998949072589, -0.3831431188056557, -0.7401987785254835, 0.1454915028125274, 1.3789288776568174, 1.2824517760598095, 0.7520431520810179, 0.9814759095853356, 0.7755282581475771, -0.50067489807501, -1.0732204790461601, -0.33234452142080745, 0.054488300671869105], 1.0, "hamming", false, "spectrum");
    assert(pd3e.kind == scientific::F_NONE && all_near(pd3f, [0.0, 0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.30000000000000004, 0.35000000000000003, 0.4, 0.45, 0.5]) && all_near(pd3p, [0.09245173355387341, 0.12169751412756662, 0.4989637231458907, 0.08945733703008615, 0.003089328152875481, 0.09768649855838707, 0.06863048967668127, 0.0008542758259555562, 2.204459123191585e-05, 1.372156044293214e-05, 6.432195295839863e-06]));
    let rs1, rs1f = scientific::resample([1.0, 3.0, -2.0, 4.0, 0.5, 7.0, -1.0, 2.0, 2.5, -3.0, 1.0], 12);
    assert(rs1f.kind == scientific::F_NONE && all_near(rs1, [1.0, 3.4088286995133306, -2.1529261272935996, 3.1327083160009366, 1.3953954497041607, 4.0292262293565395, 5.016374675456964, -2.9511294345248875, 5.024080474603759, -0.30978345475750846, -2.1011062906531013, 0.8719678262297724]));
    let rs2, rs2f = scientific::resample([1.0, 3.0, -2.0, 4.0, 0.5, 7.0, -1.0, 2.0, 2.5, -3.0, 1.0], 31);
    assert(rs2f.kind == scientific::F_NONE && all_near(rs2, [1.0, 2.9167215048379447, 3.8747964201741305, 2.6096378418667507, -0.12251178875432132, -2.056981850680391, -1.4717167885085531, 1.1658267679267489, 3.5792707938294632, 3.8025325364964186, 2.0544240548735346, 0.5515822921059614, 1.3639488587931572, 4.303502775738238, 6.885768985587654, 6.513226623839117, 2.932276009238599, -1.3211558288909717, -2.98946564448787, -0.8864275163358073, 3.065295021599976, 5.477854527013384, 4.359296411218501, 0.6775105498082156, -2.667347940319169, -3.4176833960133144, -1.7265095606102192, 0.3647632116675908, 1.0533081616438729, 0.40228772169277777, -0.021303482624155837]));
    let rs3, rs3f = scientific::resample([1.0, 3.0, -2.0, 4.0, 0.5, 7.0, -1.0, 2.0, 2.5, -3.0, 1.0], 20);
    assert(rs3f.kind == scientific::F_NONE && all_near(rs3, [1.0, 3.681150542217898, 2.3715255774916217, -1.6151416067282278, -1.0513089318350106, 3.132708316000936, 3.4090001001719417, 0.6614515197982412, 2.3881269015744055, 6.811666879495357, 5.016374675456964, -1.48813649707764, -2.118816070426243, 3.6027939607803043, 5.064689208662752, -0.309783454757509, -3.5054435596706464, -0.9197122568236857, 1.0622157349378516, 0.07936623345796333]));
    let rs4, rs4f = scientific::resample([1.0, 3.0, -2.0, 4.0, 0.5, 7.0, -1.0, 2.0, 2.5, -3.0, 1.0], 9);
    assert(rs4f.kind == scientific::F_NONE && all_near(rs4, [3.021124695707862, 0.20713061429099255, 1.1639008055178017, 2.0286230383838757, 4.980912930938045, 0.21478346436268422, 2.3697281902161795, -0.1309848866660851, -1.5824915800240853]));
    let rs5, rs5f = scientific::resample([0.8, 0.825118635510321, 0.7667649357308379, 1.4351187926374929, 1.325938592314405, 0.006107373853762976, -0.6522695660031785, -0.2657998949072589, -0.3831431188056557, -0.7401987785254835, 0.1454915028125274, 1.3789288776568174, 1.2824517760598095, 0.7520431520810179, 0.9814759095853356, 0.7755282581475771, -0.50067489807501, -1.0732204790461601, -0.33234452142080745, 0.054488300671869105], 40);
    assert(rs5f.kind == scientific::F_NONE && all_near(rs5, [0.8, 0.9544946819635503, 0.825118635510321, 0.6781377130355204, 0.7667649357308379, 1.0873782738859785, 1.4351187926374933, 1.5682728292638606, 1.325938592314405, 0.7264307369636667, 0.006107373853762966, -0.5126244780727316, -0.6522695660031785, -0.4897716310952664, -0.26579989490725936, -0.20754228510647021, -0.3831431188056558, -0.6492339148910417, -0.7401987785254835, -0.46502547062663524, 0.1454915028125275, 0.8583871563594059, 1.3789288776568176, 1.511780922867767, 1.2824517760598095, 0.9357983193052916, 0.752043152081018, 0.8168738049890263, 0.9814759095853358, 1.0202303971429474, 0.7755282581475772, 0.21570837428959966, -0.5006748980750099, -1.0266564961356879, -1.0732204790461601, -0.7188267844117342, -0.3323445214208076, -0.12615509835973449, 0.054488300671868786, 0.4041477989109072]));
    let rs6, rs6f = scientific::resample([0.8, 0.825118635510321, 0.7667649357308379, 1.4351187926374929, 1.325938592314405, 0.006107373853762976, -0.6522695660031785, -0.2657998949072589, -0.3831431188056557, -0.7401987785254835, 0.1454915028125274, 1.3789288776568174, 1.2824517760598095, 0.7520431520810179, 0.9814759095853356, 0.7755282581475771, -0.50067489807501, -1.0732204790461601, -0.33234452142080745, 0.054488300671869105], 10);
    assert(rs6f.kind == scientific::F_NONE && all_near(rs6, [0.7546927755672231, 0.9592169106969927, 1.1138562962583884, -0.46978458613890467, -0.5037824576150864, 0.1827246613412708, 1.3406576375909123, 0.8250797492312961, -0.2522459938374652, -0.6595125679555172]));
    let li, lif = scientific::lfiltic([0.2, 0.3, 0.2], [1.0, -0.4, 0.1], [1.0, 0.5], [2.0, -1.0]);
    assert(lif.kind == scientific::F_NONE && all_near(li, [0.75, 0.30000000000000004]));
    let e1, e1f = scientific::firwin(20, [0.3], false);
    assert(e1f.kind == scientific::F_VALUE);
    let e2, e2f = scientific::medfilt([1.0, 2.0, 3.0], 2);
    assert(e2f.kind == scientific::F_VALUE);
    let e3, e3f = scientific::savgol_filter([1.0, 2.0, 3.0], 5, 2);
    assert(e3f.kind == scientific::F_SHAPE);
    let e4, e4f = scientific::find_peaks([1.0, 1.0, 1.0]);
    assert(e4f.kind == scientific::F_VALUE);
    let e5, e5f = scientific::savgol_coeffs(6, 2);
    assert(e5f.kind == scientific::F_VALUE);
}

fn test_sparse() {
    let sd, sdf = scientific::csr_from_dense(mx(5, 4, [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 8.0]));
    let se, sef = scientific::csr_from_dense(mx(5, 4, [0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, -8.0]));
    assert(sdf.kind == scientific::F_NONE && sef.kind == scientific::F_NONE && sd.nnz == 8);
    let dd, ddf = scientific::csr_to_dense(sd);
    assert(ddf.kind == scientific::F_NONE && all_near(dd, [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 8.0]));
    let mv, mvf = scientific::csr_matvec(sd, [1.0, -2.0, 0.5, 3.0]);
    assert(mvf.kind == scientific::F_NONE && all_near(mv, [2.0, 1.5, 12.0, 0.0, 31.0]));
    let ad, adf = scientific::csr_add(sd, se);
    let add, addf = scientific::csr_to_dense(ad);
    assert(adf.kind == scientific::F_NONE && all_near(add, [1.0, 1.0, 2.0, 2.0, 3.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 11.0, 1.0, 1.0, 0.0, 0.0, 7.0, 0.0, 2.0, 0.0]));
    let sb, sbf = scientific::csr_sub(sd, se);
    let sbd, sbdf = scientific::csr_to_dense(sb);
    assert(sbf.kind == scientific::F_NONE && all_near(sbd, [1.0, -1.0, 2.0, -2.0, -3.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 7.0, 0.0, -2.0, 16.0]));
    let tr, trf = scientific::csr_transpose(sd);
    let tm, tmf = scientific::csr_matmul(sd, tr);
    let tmd, tmdf = scientific::csr_to_dense(tm);
    assert(trf.kind == scientific::F_NONE && tmf.kind == scientific::F_NONE && all_near(tmd, [5.0, 6.0, 4.0, 0.0, 7.0, 6.0, 9.0, 0.0, 0.0, 0.0, 4.0, 0.0, 77.0, 0.0, 76.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 76.0, 0.0, 113.0]));
    let r1, r1f = scientific::csr_sum(sd, 1);
    let r0, r0f = scientific::csr_sum(sd, 0);
    assert(all_near(r1, [3.0, 3.0, 15.0, 0.0, 15.0]) && all_near(r0, [12.0, 5.0, 5.0, 14.0]));
    let sc = scientific::csr_scale(sd, -2.5);
    let scd, scdf = scientific::csr_to_dense(sc);
    assert(all_near(scd, [-2.5, -0.0, -5.0, -0.0, -0.0, -0.0, -7.5, -0.0, -10.0, -12.5, -0.0, -15.0, -0.0, -0.0, -0.0, -0.0, -17.5, -0.0, -0.0, -20.0]));
    let g1, g1f = scientific::csr_get(sd, 2, 3);
    let g2, g2f = scientific::csr_get(sd, 3, 3);
    assert(near(g1, 6.0) && g2 == 0.0);
    let dg, dgf = scientific::csr_diagonal(sd);
    assert(all_near(dg, [1.0, 0.0, 0.0, 0.0]));
    let co, cof = scientific::csr_from_triplets(3, 3, [0.0, 2.0, 0.0, 1.0, 0.0], [1.0, 2.0, 1.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0, 5.0]);
    let cod, codf = scientific::csr_to_dense(co);
    assert(cof.kind == scientific::F_NONE && co.nnz == 4 && all_near(cod, [5.0, 4.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 2.0]));
    let dg1, dg1f = scientific::csr_diags([1.0, 2.0, 3.0, 4.0], 1, 4, 5);
    let dg1d, dg1df = scientific::csr_to_dense(dg1);
    assert(dg1f.kind == scientific::F_NONE && all_near(dg1d, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]));
    let dg2, dg2f = scientific::csr_diags([1.0, 2.0, 3.0, 4.0, 5.0], -1, 5, 5);
    let dg2d, dg2df = scientific::csr_to_dense(dg2);
    assert(dg2f.kind == scientific::F_NONE && all_near(dg2d, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0]));
    let dg3, dg3f = scientific::csr_diags([1.0, 2.0, 3.0, 4.0, 5.0], 0, 4, 6);
    let dg3d, dg3df = scientific::csr_to_dense(dg3);
    assert(dg3f.kind == scientific::F_NONE && all_near(dg3d, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]));
    let dg4, dg4f = scientific::csr_diags([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], -2, 6, 4);
    let dg4d, dg4df = scientific::csr_to_dense(dg4);
    assert(dg4f.kind == scientific::F_NONE && all_near(dg4d, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0]));
    let ey, eyf = scientific::csr_eye(3);
    let eyd, eydf = scientific::csr_to_dense(ey);
    assert(all_near(eyd, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]));
    let ta, taf = scientific::csr_from_dense(mx(30, 30, [2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0]));
    let sx, sxf = scientific::spsolve(ta, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]);
    assert(sxf.kind == scientific::F_NONE && all_near(sx, [159.99999999999974, 318.9999999999995, 475.99999999999926, 629.9999999999991, 779.9999999999987, 924.9999999999985, 1063.9999999999982, 1195.999999999998, 1319.9999999999977, 1434.9999999999975, 1539.9999999999973, 1633.9999999999968, 1715.9999999999964, 1784.999999999996, 1839.9999999999955, 1879.9999999999952, 1903.999999999995, 1910.9999999999948, 1899.9999999999945, 1869.9999999999945, 1819.9999999999948, 1748.999999999995, 1655.9999999999957, 1539.9999999999964, 1399.9999999999968, 1234.9999999999973, 1043.9999999999977, 825.9999999999982, 579.9999999999987, 304.9999999999994]));
    let cx, ci, cf = scientific::cg(ta, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0], 1e-12, 500);
    assert(cf.kind == scientific::F_NONE && ci == 0 && all_near(cx, [159.99999999999991, 318.9999999999999, 475.9999999999998, 629.9999999999998, 779.9999999999998, 924.9999999999998, 1063.9999999999993, 1195.9999999999995, 1319.9999999999993, 1434.999999999999, 1539.9999999999986, 1633.9999999999993, 1715.999999999999, 1784.999999999999, 1839.9999999999989, 1879.9999999999993, 1903.9999999999998, 1910.999999999999, 1899.9999999999998, 1869.9999999999993, 1819.9999999999998, 1748.999999999999, 1655.999999999999, 1539.9999999999993, 1399.9999999999995, 1234.9999999999995, 1043.9999999999998, 825.9999999999998, 580.0, 305.0]));
    let pa, paf = scientific::csr_from_dense(mx(3, 3, [0.0, 5.0, 0.0, 7.0, 0.0, 1.0, 2.0, 0.0, 0.0]));
    let px, pxf = scientific::spsolve(pa, [10.0, 3.0, 4.0]);
    assert(pxf.kind == scientific::F_NONE && all_near(px, [2.0, 2.0, -11.0]));
    let bad1, bad1f = scientific::csr_from_triplets(2, 2, [2.0], [0.0], [1.0]);
    assert(bad1f.kind == scientific::F_RANGE);
    let bad2, bad2f = scientific::csr_add(sd, tr);
    assert(bad2f.kind == scientific::F_SHAPE);
    let bad3, bad3f = scientific::csr_matvec(sd, [1.0]);
    assert(bad3f.kind == scientific::F_SHAPE);
    let bad4, bad4f = scientific::csr_from_triplets(0, 2, [], [], []);
    assert(bad4f.kind == scientific::F_RANGE);
    let nx, ni, nf = scientific::cg(pa, [1.0, 1.0, 1.0]);
    assert(nf.kind == scientific::F_VALUE);
}

fn scientific_abs(x: f64) -> f64 { return mathshim_abs(x); }

// The complex array type and its operations, pinned against NumPy.
// all_near_c tells whether the parts of a complex array agree to two lists.
fn all_near_c(a: scientific::CArr&, vr: f64[:], vi: f64[:]) -> bool {
    var i: i64 = 0;
    while i < vr.len {
        if !near(a.data[2 * i], vr[i]) { return false; }
        if !near(a.data[2 * i + 1], vi[i]) { return false; }
        i += 1;
    }
    return true;
}

fn test_carray() {
    let re = av([1.0, -2.0, 3.5, 0.0]);
    let im = av([0.5, 2.0, -1.25, 4.0]);
    let z, zf = scientific::complex_array(re, im);
    assert(zf.kind == scientific::F_NONE);
    assert(z.ndim == 1 && z.shape[0] == 4 && z.data.len == 8);
    // The parts come back as they went in.
    let zr = scientific::real(z);
    let zi = scientific::imag(z);
    assert(all_near(zr, [1.0, -2.0, 3.5, 0.0]) && all_near(zi, [0.5, 2.0, -1.25, 4.0]));
    // The size and the angle of each element.
    let za = scientific::cabs(z);
    assert(all_near(za, [1.118033988749895, 2.8284271247461903, 3.7165171868296265, 4.0]));
    let zang = scientific::angle(z);
    assert(all_near(zang, [0.4636476090008061, 2.356194490192345, -0.3430239404207034, 1.5707963267948966]));
    let zdeg = scientific::angle(z, true);
    assert(all_near(zdeg, [26.56505117707799, 135.0, -19.65382405805331, 90.0]));
    // The conjugate turns the imaginary part around.
    let zc = scientific::conj(z);
    assert(all_near_c(zc, [1.0, -2.0, 3.5, 0.0], [-0.5, -2.0, 1.25, -4.0]));
    // The tests of the parts.
    let zr0, zrf = scientific::complex_from_values([1.0, 2.0], [0.0, 0.0]);
    assert(zrf.kind == scientific::F_NONE);
    let ic0 = scientific::iscomplex(zr0);
    let ir0 = scientific::isreal(zr0);
    assert(all_near(ic0, [0.0, 0.0]) && all_near(ir0, [1.0, 1.0]));
    let ic1 = scientific::iscomplex(z);
    let ir1 = scientific::isreal(z);
    assert(all_near(ic1, [1.0, 1.0, 1.0, 1.0]) && all_near(ir1, [0.0, 0.0, 0.0, 0.0]));
    // The parts of different shapes make a fault.
    let zbad, zbadf = scientific::complex_array(re, av([1.0, 2.0]));
    assert(zbadf.kind == scientific::F_SHAPE);
    // The elementwise operations broadcast like the real ones.
    let a = mx(2, 2, [1.0, 3.0, 0.5, -2.0]);
    let ai = mx(2, 2, [2.0, -1.0, 0.5, 0.0]);
    let b = av([1.0, 2.0]);
    let bi = av([1.0, -0.5]);
    let ca, caf = scientific::complex_array(a, ai);
    let cb, cbf = scientific::complex_array(b, bi);
    assert(caf.kind == scientific::F_NONE && cbf.kind == scientific::F_NONE);
    let sum, sf = scientific::cadd(ca, cb);
    assert(sf.kind == scientific::F_NONE);
    assert(all_near_c(sum, [2.0, 5.0, 1.5, 0.0], [3.0, -1.5, 1.5, -0.5]));
    let dif, df = scientific::csub(ca, cb);
    assert(df.kind == scientific::F_NONE);
    assert(all_near_c(dif, [0.0, 1.0, -0.5, -4.0], [1.0, -0.5, -0.5, 0.5]));
    let pro, pf = scientific::cmul(ca, cb);
    assert(pf.kind == scientific::F_NONE);
    assert(all_near_c(pro, [-1.0, 5.5, 0.0, -4.0], [3.0, -3.5, 1.0, 1.0]));
    let quo, qf = scientific::cdiv(ca, cb);
    assert(qf.kind == scientific::F_NONE);
    assert(all_near_c(quo, [1.5, 1.5294117647058822, 0.5, -0.9411764705882353], [0.5, -0.11764705882352941, 0.0, -0.23529411764705882]));
    let pw, pwf = scientific::cpower(ca, cb);
    assert(pwf.kind == scientific::F_NONE);
    assert(all_near_c(pw, [-0.24720004426291722, 2.932609903259764, 0.2918503793793073, 18.09782608169716], [0.6964504870825432, -7.992978025796342, 0.13697862691506052, -6.536036508550907]));
    // The functions of one operand.
    let u, uf = scientific::complex_from_values([0.3, -1.0, 2.0, -0.5], [0.4, 2.0, -1.0, -0.5]);
    assert(uf.kind == scientific::F_NONE);
    let ue = scientific::cexp(u);
    assert(all_near_c(ue, [1.2433022950695027, -0.1530918656742263, 3.992324048441272, 0.5322807302156708], [0.5256597791969788, 0.33451182923926226, -6.217676312367968, -0.29078628821269187]));
    let ul = scientific::clog(u);
    assert(all_near_c(ul, [-0.6931471805599453, 0.8047189562170503, 0.8047189562170503, -0.34657359027997264], [0.9272952180016123, 2.0344439357957027, -0.4636476090008061, -2.356194490192345]));
    let us = scientific::csqrt(u);
    assert(all_near_c(us, [0.6324555320336759, 0.7861513777574233, 1.455346690225355, 0.3217971264527913], [0.31622776601683794, 1.272019649514069, -0.34356074972251244, -0.7768869870150187]));
    let usn = scientific::csin(u);
    assert(all_near_c(usn, [0.31947873074156474, -3.165778513216168, 1.4031192506220405, -0.5406126857131534], [0.3924066848326388, 1.9596010414216063, 0.4890562590412937, -0.4573041531842493]));
    let uc = scientific::ccos(u);
    assert(all_near_c(uc, [1.0327878842028413, 2.0327230070196656, -0.64214812471552, 0.9895848833999199], [-0.12138561220787392, 3.0518977991518, 1.0686074213827783, -0.24982639750046154]));
    let ush = scientific::csinh(u);
    assert(all_near_c(ush, [0.2804817641764754, 0.4890562590412937, 1.9596010414216063, -0.4573041531842493], [0.4070739913234487, 1.4031192506220405, -3.165778513216168, -0.5406126857131534]));
    let uch = scientific::ccosh(u);
    assert(all_near_c(uch, [0.9628205308930272, -0.64214812471552, 2.0327230070196656, 0.9895848833999199], [0.11858578787353008, -1.0686074213827783, -3.0518977991518, 0.24982639750046154]));
    // The matrices.
    let m1, m1f = scientific::complex_array(mx(2, 2, [1.0, 0.0, 2.0, 1.0]), mx(2, 2, [2.0, -1.0, 0.0, 1.0]));
    let m2, m2f = scientific::complex_array(mx(2, 2, [0.5, 1.0, -1.0, 0.5]), mx(2, 2, [0.0, 1.0, 2.0, -0.5]));
    assert(m1f.kind == scientific::F_NONE && m2f.kind == scientific::F_NONE);
    let mm, mmf = scientific::cmatmul(m1, m2);
    assert(mmf.kind == scientific::F_NONE);
    assert(all_near_c(mm, [2.5, -1.5, -2.0, 3.0], [2.0, 2.5, 1.0, 2.0]));
    let tr, trf = scientific::ctranspose(m1);
    assert(trf.kind == scientific::F_NONE);
    assert(all_near_c(tr, [1.0, 2.0, 0.0, 1.0], [2.0, 0.0, -1.0, 1.0]));
    let ad, adf = scientific::adjoint(m1);
    assert(adf.kind == scientific::F_NONE);
    assert(all_near_c(ad, [1.0, 2.0, 0.0, 1.0], [-2.0, 0.0, 1.0, -1.0]));
    // The solve of a complex system.
    let rhs, rhsf = scientific::complex_from_values([1.0, 2.0], [1.0, -1.0]);
    assert(rhsf.kind == scientific::F_NONE);
    let sol, solf = scientific::csolve(m1, rhs);
    assert(solf.kind == scientific::F_NONE);
    assert(all_near_c(sol, [0.7307692307692308, 0.11538461538461536], [-0.34615384615384615, -0.423076923076923]));
    // The solve with a matrix on the right.
    let bm, bmf = scientific::complex_array(mx(2, 2, [1.0, 2.0, 0.0, 1.0]), mx(2, 2, [0.0, 1.0, -1.0, 0.0]));
    assert(bmf.kind == scientific::F_NONE);
    let solm, solmf = scientific::csolve(m1, bm);
    assert(solmf.kind == scientific::F_NONE);
    assert(all_near_c(solm, [0.11538461538461539, 0.7307692307692308, -0.19230769230769232, 0.11538461538461534], [-0.42307692307692313, -0.34615384615384615, 0.03846153846153848, 0.5769230769230771]));
    // The inverse and the determinant.
    let iv, ivf = scientific::cinv(m1);
    assert(ivf.kind == scientific::F_NONE);
    assert(all_near_c(iv, [0.15384615384615383, 0.19230769230769235, 0.0769230769230769, 0.3461538461538462], [-0.23076923076923075, -0.03846153846153845, 0.3846153846153847, -0.2692307692307692]));
    let dr, di, dtf = scientific::cdet(m1);
    assert(dtf.kind == scientific::F_NONE);
    assert(near(dr, -1.0) && near(di, 5.0));
    // The transforms of a complex signal.
    let sig, sigf = scientific::complex_from_values([1.0, 2.0, 0.0, -1.0], [0.0, -1.0, 0.5, 2.0]);
    assert(sigf.kind == scientific::F_NONE);
    let ft, ftf = scientific::cfft(sig);
    assert(ftf.kind == scientific::F_NONE);
    assert(all_near_c(ft, [2.0, -2.0, 0.0, 4.0], [1.5, -3.5, -0.5, 2.5]));
    let itf, itff = scientific::cifft(sig);
    assert(itff.kind == scientific::F_NONE);
    assert(all_near_c(itf, [0.5, 1.0, 0.0, -0.5], [0.375, 0.625, -0.125, -0.875]));
    // The eigenvalues of a real matrix may hold complex pairs.
    let ea, eaf = scientific::eigvals_complex(mx(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]));
    assert(eaf.kind == scientific::F_NONE);
    assert(all_near_c(ea, [-0.9057401795217577, 0.1982468633970094, 16.707493316124747], [0.0, 0.0, 0.0]));
    let eb, ebf = scientific::eigvals_complex(mx(2, 2, [0.0, -1.0, 1.0, 0.0]));
    assert(ebf.kind == scientific::F_NONE);
    assert(all_near_c(eb, [0.0, 0.0], [-1.0, 1.0]));
    let ec, ecf = scientific::eigvals_complex(mx(3, 3, [2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, -4.0, 3.0]));
    assert(ecf.kind == scientific::F_NONE);
    assert(all_near_c(ec, [2.0, 3.0, 3.0], [0.0, -4.0, 4.0]));
    // A matrix that is not square makes a fault.
    let c23, c23f = scientific::complex_array(mx(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), mx(2, 3, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(c23f.kind == scientific::F_NONE);
    let bad, badf = scientific::csolve(c23, rhs);
    assert(badf.kind == scientific::F_SHAPE);
    let bad2, bad2f = scientific::cmatmul(c23, c23);
    assert(bad2f.kind == scientific::F_SHAPE);
    let bad3, bad3f = scientific::ctranspose(rhs);
    assert(bad3f.kind == scientific::F_SHAPE);
    let bad4, bad4f = scientific::eigvals_complex(mx(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert(bad4f.kind == scientific::F_SHAPE);
    return;
}

// The statistic modes of pad_modes, pinned against NumPy.
fn test_pad_stats() {
    let q = av([3.0, 1.0, 4.0, 1.0, 5.0]);
    let m = mx(2, 3, [1.0, 2.0, 3.0, 4.0, 8.0, 6.0]);
    let r1, f1 = scientific::pad_modes(q, [3], [4], "linear_ramp", 7.0);
    assert(f1.kind == scientific::F_NONE && all_near(r1, [7.0, 5.666666666666667, 4.333333333333334, 3.0, 1.0, 4.0, 1.0, 5.0, 5.5, 6.0, 6.5, 7.0]));
    let r2, f2 = scientific::pad_modes(q, [3], [4], "maximum");
    assert(f2.kind == scientific::F_NONE && all_near(r2, [5.0, 5.0, 5.0, 3.0, 1.0, 4.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0]));
    let r3, f3 = scientific::pad_modes(q, [3], [4], "minimum", 0.0, 3);
    assert(f3.kind == scientific::F_NONE && all_near(r3, [1.0, 1.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0]));
    let r4, f4 = scientific::pad_modes(q, [3], [4], "mean", 0.0, 2);
    assert(f4.kind == scientific::F_NONE && all_near(r4, [2.0, 2.0, 2.0, 3.0, 1.0, 4.0, 1.0, 5.0, 3.0, 3.0, 3.0, 3.0]));
    let r5, f5 = scientific::pad_modes(q, [3], [4], "median");
    assert(f5.kind == scientific::F_NONE && all_near(r5, [3.0, 3.0, 3.0, 3.0, 1.0, 4.0, 1.0, 5.0, 3.0, 3.0, 3.0, 3.0]));
    let r6, f6 = scientific::pad_modes(m, [1, 2], [2, 1], "mean");
    assert(f6.kind == scientific::F_NONE && all_near(r6, [4.0, 4.0, 2.5, 5.0, 4.5, 4.0, 2.0, 2.0, 1.0, 2.0, 3.0, 2.0, 6.0, 6.0, 4.0, 8.0, 6.0, 6.0, 4.0, 4.0, 2.5, 5.0, 4.5, 4.0, 4.0, 4.0, 2.5, 5.0, 4.5, 4.0]));
    let r7, f7 = scientific::pad_modes(m, [2, 1], [1, 2], "linear_ramp", -1.0);
    assert(f7.kind == scientific::F_NONE && all_near(r7, [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.5, 1.0, 0.0, -1.0, -1.0, 1.0, 2.0, 3.0, 1.0, -1.0, -1.0, 4.0, 8.0, 6.0, 2.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]));
    let r8, f8 = scientific::pad_modes(m, [1, 1], [1, 1], "median", 0.0, 2);
    assert(f8.kind == scientific::F_NONE && all_near(r8, [3.75, 2.5, 5.0, 4.5, 4.75, 1.5, 1.0, 2.0, 3.0, 2.5, 6.0, 4.0, 8.0, 6.0, 7.0, 3.75, 2.5, 5.0, 4.5, 4.75]));
    let r9, f9 = scientific::pad_modes(q, [1], [1], "mean", 0.0, -1);
    assert(f9.kind == scientific::F_VALUE);
    return;
}

// The cosine and sine transforms along many axes, pinned against SciPy.
fn test_dctn() {
    let x0 = av([1.0, 2.4622888266898326, 4.171167510947728, 6.062866266041593, 8.103282983463814, 10.270619156458231, 12.549529737646981, 14.92852786458892, 17.398638404385867, 19.952623149688797, 22.584500550247288, 25.289230792133658, 28.06249997748054, 30.90056685321995, 33.800150712639834, 36.75834735990512, 39.77256477258468, 42.840472942735936, 45.95996414826417, 49.129121044631624, 52.34619071934546, 55.60956336124427, 58.9177545493537, 62.26939041505117]);
    let x, xf = scientific::reshape(x0, [2, 3, 4]);
    assert(xf.kind == scientific::F_NONE);
    let r0, g0 = scientific::dctn(x, [0, 2]);
    assert(g0.kind == scientific::F_NONE && all_near(r0, [572.8715500276984, -57.89990479056862, 1.5543329934066363, -4.030270173898586, 894.2163306014975, -68.06365523843954, 0.885045055397859, -4.820137152500244, 1257.471567765801, -74.92385495323913, 0.6760187477795762, -5.316541228602198, -327.6032570503931, 10.78224715196562, 0.6185603137562878, 0.8208478562759156, -372.92857790109076, 7.524473343398311, 0.2208264554626511, 0.5433286805358346, -407.0613094872042, 6.043495630025117, 0.12496454556957676, 0.43278520149230637]));
    let r1, g1 = scientific::dctn(x, [0, 1, 2], 3, "ortho");
    assert(g1.kind == scientific::F_NONE && all_near(r1, [122.84341045578911, -38.2565476360308, 25.908638226000306, 1.3769744442855796, -58.528862482871645, 15.060355134850209, -11.90921190173123, -1.4272406237638187, 17.620371767234364, -5.463806689473594, 3.749543409305727, 0.2146714908807148, -73.98506732569552, 17.157327818265166, -14.838971498549116, -2.275988222671373, 17.25562393481787, -3.2084408492392753, 3.4561679976691457, 0.7541676869036333, -10.913313035669105, 2.757476220325989, -2.1802725221062556, -0.26802631295973445]));
    let r2, g2 = scientific::dstn(x, [1], 1);
    assert(g2.kind == scientific::F_NONE && all_near(r2, [42.2261699276086, 52.24071082931024, 62.937288117788604, 74.1956165976869, -32.797276808771734, -34.98066864599793, -36.82666607859912, -38.45272905218413, 9.813037993753348, 11.158234203477313, 12.739169167200679, 14.481505139331208, 193.26019046128403, 208.02474531742965, 223.04284739286334, 238.30461190173293, -48.56738148372983, -49.41799301604864, -50.23520767342774, -51.02208611029209, 34.16993137094529, 36.662853546485906, 39.20299079980667, 41.788127723206415]));
    let r3, g3 = scientific::dstn(x, [-1, 0], 4, "ortho");
    assert(g3.kind == scientific::F_NONE && all_near(r3, [59.67405338270412, 13.98516625077658, 11.025243204838889, 8.796922617710356, 85.81820710312842, 22.303437627337455, 16.726069965802818, 13.570940507386034, 114.4667659656682, 31.72803616027591, 23.146858248246943, 18.969747855527423, -16.315885128021467, -6.754749256257228, -4.115876624725164, -3.591731452681997, -10.777378516330712, -5.709364368823441, -3.310441464866356, -2.9672762697531154, -2.9757055340617526, -3.487695818561505, -1.7363024457421345, -1.6673003542772942]));
    let r4, g4 = scientific::idctn(x, [2, 1]);
    assert(g4.kind == scientific::F_NONE && all_near(r4, [4.564283502812174, -2.3142059457757935, 0.8081960802857989, -0.3554456173343866, -4.063193641890532, 1.6203939853856697, -0.6127933403239767, 0.23915326276452667, 0.5044698334815361, -0.19475575765153041, 0.08079976996579413, -0.026902131719281214, 17.309917061229775, -6.491954152635282, 2.5584297265551204, -0.9349931659607159, -8.56390431765439, 2.9778747795131775, -1.218955525877761, 0.4191616089181166, 1.3223344510769355, -0.48644757370716696, 0.19378629982824513, -0.0696241969159172]));
    let r5, g5 = scientific::idstn(x, [0, 1, 2], 3);
    assert(g5.kind == scientific::F_NONE && all_near(r5, [8.700905863107534, -0.8883483619171197, 3.6199801177941935, -0.6274213466537222, -2.8550118964446596, 0.09718761670483059, -1.176520371925791, 0.06949080920886774, 4.500730748908281, -0.42793404009117386, 1.8749892705395361, -0.30171584127455864, -7.1274225536214555, 0.21031110796336966, -2.9428261719513316, 0.149583979058181, 0.6598259841307222, 0.053959329750888484, 0.28012488550243686, 0.03918701518701173, -3.4842016501221584, 0.1225862522492332, -1.4348690767198933, 0.08782661229356736]));
    let r6, g6 = scientific::dctn(x);
    assert(g6.kind == scientific::F_NONE && all_near(r6, [5449.1188967899925, -401.77482996449453, 6.230793593167821, -28.333897110001942, -1185.762013584948, 29.486346627265767, 1.521284898437642, 2.2278868190518715, 41.91045659050428, 3.3035507330713036, 0.46026163039061474, 0.29346290249967555, -2215.186288877376, 48.70043225077816, 1.9287026295770313, 3.5939234766080475, 137.62538389102886, 8.207758400445293, 0.8549329489003654, 0.6721442346052675, 11.192589264584237, 1.7767960951941144, 0.3018719484005662, 0.16697569669655454]));
    let r7, g7 = scientific::dctn(x, []);
    assert(g7.kind == scientific::F_NONE && all_near(r7, [1.0, 2.4622888266898326, 4.171167510947728, 6.062866266041593, 8.103282983463814, 10.270619156458231, 12.549529737646981, 14.92852786458892, 17.398638404385867, 19.952623149688797, 22.584500550247288, 25.289230792133658, 28.06249997748054, 30.90056685321995, 33.800150712639834, 36.75834735990512, 39.77256477258468, 42.840472942735936, 45.95996414826417, 49.129121044631624, 52.34619071934546, 55.60956336124427, 58.9177545493537, 62.26939041505117]));
    let e1, h1 = scientific::dctn(x, [3]);
    assert(h1.kind == scientific::F_RANGE);
    let e2, h2 = scientific::dctn(x, [1, 1]);
    assert(h2.kind == scientific::F_VALUE);
    let e3, h3 = scientific::dstn(x, [0], 5);
    assert(h3.kind == scientific::F_VALUE);
    let e4, h4 = scientific::idctn(x, [0], 2, "nope");
    assert(h4.kind == scientific::F_VALUE);
    return;
}

fn dbl_lo1(x: f64) -> f64 { return 0.5 * x; }
fn dbl_hi1(x: f64) -> f64 { return 1.0 + x; }
fn dbl_zero(x: f64) -> f64 { return 0.0; }
fn dbl_sq(x: f64) -> f64 { return x * x; }
fn dbl_m1(x: f64) -> f64 { return -1.0; }
fn dbl_p1(x: f64) -> f64 { return 1.0; }

// dblquad is pinned against SciPy.
fn test_dblquad() {
    let v1, e1, f1 = scientific::dblquad(0.0, 2.0, dbl_lo1, dbl_hi1) { y, x => mathshim_exp(-x * y) * mathshim_sin(x + y) };
    assert(f1.kind == scientific::F_NONE && near(v1, 0.7360958517002549) && e1 >= 0.0);
    let v2, e2, f2 = scientific::dblquad(1.0, 3.0, dbl_zero, dbl_sq) { y, x => x * y * y };
    assert(f2.kind == scientific::F_NONE && near(v2, 273.33333333333337));
    let v3, e3, f3 = scientific::dblquad(0.0, 1.0, dbl_m1, dbl_p1) { y, x => 1.0 / (1.0 + x * x + y * y) };
    assert(f3.kind == scientific::F_NONE && near(v3, 1.2790207037406225));
    // The limits in the wrong order change the sign.
    let v4, e4, f4 = scientific::dblquad(3.0, 1.0, dbl_zero, dbl_sq) { y, x => x * y * y };
    assert(f4.kind == scientific::F_NONE && near(v4, -273.33333333333337));
    // A bad tolerance is a fault.
    let v5, e5, f5 = scientific::dblquad(0.0, 1.0, -1.0, 1.49e-8, 50, dbl_zero, dbl_sq) { y, x => x * y };
    assert(f5.kind == scientific::F_VALUE);
    // Infinite limits of x and of y.
    let v6, e6, f6 = scientific::dblquad(0.0, 1.0 / 0.0, dbl_zero, dbl_inf) { y, x => mathshim_exp(-x - y) };
    assert(f6.kind == scientific::F_NONE && mathshim_abs(v6 - 1.0) < 1.0e-7);
    let v7, e7, f7 = scientific::dblquad(-(1.0 / 0.0), 1.0 / 0.0, dbl_zero, dbl_inf) { y, x => mathshim_exp(-x * x - y) };
    assert(f7.kind == scientific::F_NONE && mathshim_abs(v7 - 1.772453850905264) < 1.0e-6);
    // A missing value of a limit is a fault.
    let v8, e8, f8 = scientific::dblquad(0.0 / 0.0, 1.0, dbl_zero, dbl_sq) { y, x => x * y };
    assert(f8.kind == scientific::F_VALUE);
    return;
}

fn dbl_inf(x: f64) -> f64 { return 1.0 / 0.0; }

// root is pinned against SciPy for four methods and the faults.
fn test_root() {
    let ms = ["hybr", "lm", "broyden1", "broyden2"];
    var k: i64 = 0;
    while k < 4 {
        let x, fv, f = scientific::root([2.0, 0.5], ms[k], 1.0e-10, 200) { v => [v[0] * v[0] + v[1] * v[1] - 4.0, v[0] * v[1] - 1.0] };
        assert(f.kind == scientific::F_NONE);
        assert(mathshim_abs(x.data[0] - 1.93185165) < 1.0e-7 && mathshim_abs(x.data[1] - 0.51763809) < 1.0e-7);
        assert(mathshim_abs(fv.data[0]) < 1.0e-6 && mathshim_abs(fv.data[1]) < 1.0e-6);
        k += 1;
    }
    // The simple iterations reach the root that SciPy finds for this
    // system: 0.8411639, 0.1588361.
    let ms2 = ["linearmixing", "diagbroyden", "excitingmixing", "anderson"];
    k = 0;
    while k < 4 {
        let xm, fm, gm = scientific::root([0.0, 0.0], ms2[k], 1.0e-9, 5000) { v => [v[0] + 0.5 * (v[0] - v[1]) * (v[0] - v[1]) * (v[0] - v[1]) - 1.0, 0.5 * (v[1] - v[0]) * (v[1] - v[0]) * (v[1] - v[0]) + v[1]] };
        assert(gm.kind == scientific::F_NONE);
        assert(mathshim_abs(xm.data[0] - 0.8411639) < 1.0e-6 && mathshim_abs(xm.data[1] - 0.1588361) < 1.0e-6);
        k += 1;
    }
    // The method df-sane takes no derivative. Its tolerance is relative
    // to the first residual.
    let xd, fd, gd = scientific::root([0.0, 0.0], "df-sane", 1.0e-10, 500) { v => [v[0] + 0.5 * (v[0] - v[1]) * (v[0] - v[1]) * (v[0] - v[1]) - 1.0, 0.5 * (v[1] - v[0]) * (v[1] - v[0]) * (v[1] - v[0]) + v[1]] };
    assert(gd.kind == scientific::F_NONE);
    assert(mathshim_abs(xd.data[0] - 0.8411639) < 1.0e-6 && mathshim_abs(xd.data[1] - 0.1588361) < 1.0e-6);
    // The method krylov uses a Krylov iteration for the linear step.
    // SciPy reaches the same root.
    let xk, fk2, gk = scientific::root([2.0, 0.5], "krylov", 1.0e-10, 200) { v => [v[0] * v[0] + v[1] * v[1] - 4.0, v[0] * v[1] - 1.0] };
    assert(gk.kind == scientific::F_NONE);
    assert(mathshim_abs(xk.data[0] - 1.93185165) < 1.0e-6 && mathshim_abs(xk.data[1] - 0.51763809) < 1.0e-6);
    let m1, m2, mg = scientific::root([1.0, 2.0], "anderson") { v => [v[0]] };
    assert(mg.kind == scientific::F_SHAPE);
    let x3, f3v, g3 = scientific::root([1.0, 1.0, 1.0]) { v => [v[0] + v[1] + v[2] - 6.0, v[0] * v[1] - 2.0, v[2] - v[0] - 1.0] };
    assert(g3.kind == scientific::F_NONE);
    assert(mathshim_abs(x3.data[0] - 2.0) < 1.0e-6 && mathshim_abs(x3.data[1] - 1.0) < 1.0e-6 && mathshim_abs(x3.data[2] - 3.0) < 1.0e-6);
    let b1, b2, g4 = scientific::root([1.0], "newton") { v => [v[0]] };
    assert(g4.kind == scientific::F_VALUE);
    let b3, b4, g5 = scientific::root([1.0], "hybr", 0.0) { v => [v[0]] };
    assert(g5.kind == scientific::F_VALUE);
    let b5, b6, g6 = scientific::root([1.0, 2.0], "broyden1") { v => [v[0]] };
    assert(g6.kind == scientific::F_SHAPE);
    let b7, b8, g7 = scientific::root([1.0], "broyden1", 1.0e-12, 3) { v => [v[0] * v[0] * v[0] - 1000.0] };
    assert(g7.kind == scientific::F_LIMIT);
    let b9, b10, g8 = scientific::root([], "hybr") { v => [v[0]] };
    assert(g8.kind == scientific::F_VALUE);
}

// dual_annealing and basinhopping find the global minimum of a
// function with many local minima. The test uses the function of
// Rastrigin, whose global minimum is zero at the origin.
fn rastrigin(v: f64[:]) -> f64 {
    return 20.0 + v[0] * v[0] - 10.0 * mathshim_cos(6.283185307179586 * v[0]) + v[1] * v[1] - 10.0 * mathshim_cos(6.283185307179586 * v[1]);
}

fn test_global_minimizers() {
    var s: i64 = 1;
    while s <= 4 {
        let d, f = scientific::dual_annealing([-5.12, -5.12], [5.12, 5.12], s) { v => rastrigin(v) };
        assert(f.kind == scientific::F_NONE);
        assert(d.data.len == 3 && d.data[2] < 1.0e-6);
        s += 1;
    }
    let d2, f2 = scientific::dual_annealing([-5.12, -5.12], [5.12, 5.12], 3) { v => rastrigin(v) };
    let d3, f3 = scientific::dual_annealing([-5.12, -5.12], [5.12, 5.12], 3) { v => rastrigin(v) };
    assert(d2.data[0] == d3.data[0] && d2.data[2] == d3.data[2]);
    let b, g = scientific::basinhopping([2.5, 2.5], 100, 1.0, 1.0, 3) { v => rastrigin(v) };
    assert(g.kind == scientific::F_NONE);
    assert(b.data[2] < 1.0e-6);
    let e1, g1 = scientific::dual_annealing([1.0], [0.0]) { v => v[0] };
    assert(g1.kind == scientific::F_VALUE);
    let e2, g2 = scientific::dual_annealing([0.0, 1.0], [1.0]) { v => v[0] };
    assert(g2.kind == scientific::F_SHAPE);
    let e3, g3 = scientific::dual_annealing([0.0], [1.0], 1, 0) { v => v[0] };
    assert(g3.kind == scientific::F_RANGE);
    let e4, g4 = scientific::dual_annealing([0.0], [1.0]) { v => v[0] / 0.0 };
    assert(g4.kind == scientific::F_VALUE);
    let e5, g5 = scientific::basinhopping([], 10) { v => 0.0 };
    assert(g5.kind == scientific::F_SHAPE);
    let e6, g6 = scientific::basinhopping([1.0], 10, 1.0, 0.0) { v => v[0] };
    assert(g6.kind == scientific::F_RANGE);
    let e7, g7 = scientific::basinhopping([1.0 / 0.0], 10) { v => v[0] };
    assert(g7.kind == scientific::F_VALUE);
}

// cubic_hermite is pinned against SciPy CubicHermiteSpline.
fn test_cubic_hermite() {
    let x = [0.0, 1.0, 2.5, 4.0];
    let y = [1.0, 3.0, 2.0, 5.0];
    let d = [0.5, -1.0, 2.0, 0.0];
    let c, f = scientific::cubic_hermite(x, y, d);
    assert(f.kind == scientific::F_NONE);
    let v, fv = scientific::spline_eval(c, [0.0, 0.5, 1.0, 1.7, 2.5, 3.3, 4.0, 5.0]);
    assert(fv.kind == scientific::F_NONE);
    let ex = [1.0, 2.1875, 3.0, 2.0023703703703704, 2.0, 3.9982222222222212, 5.0, 1.4444444444444446];
    var i: i64 = 0;
    while i < 8 { assert(mathshim_abs(v.data[i] - ex[i]) < 1.0e-12); i += 1; }
    let d1, vf1 = scientific::spline_eval(c, [0.5], 1);
    assert(mathshim_abs(d1.data[0] - 3.125) < 1.0e-12);
    let b1, g1 = scientific::cubic_hermite(x, y, [0.0, 1.0]);
    assert(g1.kind == scientific::F_SHAPE);
    let b2, g2 = scientific::cubic_hermite(x, y, [0.0, 1.0, 1.0 / 0.0, 0.0]);
    assert(g2.kind == scientific::F_VALUE);
    let b3, g3 = scientific::cubic_hermite([0.0, 0.0], [1.0, 2.0], [0.0, 0.0]);
    assert(g3.kind == scientific::F_VALUE);
}

// The periodic condition and the given end slopes of cubic_spline are
// pinned against SciPy.
// Cubic splines with a slope or a second derivative at each end.
fn test_spline_mixed_ends() {
    let kx = [0.0, 0.7, 1.5, 2.4, 4.0];
    let ky = [1.0, 2.0, 0.5, 3.0, 2.2];
    let q = [0.3, 1.0, 3.0, 5.0];
    let a, fa = scientific::cubic_spline_ends(kx, ky, 2, 1.5, 2, -0.5);
    let va, ea = scientific::spline_eval(a, q, 0);
    assert(fa.kind == scientific::F_NONE && ea.kind == scientific::F_NONE && all_near(va, [1.6730996537914262, 1.4331011255339694, 3.7385485741574564, 0.1614514258425448]));
    let b, fb = scientific::cubic_spline_ends(kx, ky, 1, 0.3, 2, 0.8);
    let vb, eb = scientific::spline_eval(b, q, 0);
    assert(fb.kind == scientific::F_NONE && all_near(vb, [1.4392747194115743, 1.5077575474634295, 3.6484652194211025, 1.551534780578896]));
    let c, fc = scientific::cubic_spline_ends(kx, ky, 2, -1.0, 1, 0.2);
    let vc, ec = scientific::spline_eval(c, q, 0);
    assert(fc.kind == scientific::F_NONE && all_near(vc, [1.7380019563098024, 1.400141276736955, 3.2582375136809136, 6.394029225950629]));
    let d, fd = scientific::cubic_spline_ends([0.0, 0.7], [1.0, 2.0], 2, 1.0, 2, 2.0);
    let vd, ed = scientific::spline_eval(d, [0.2, 0.5], 0);
    assert(fd.kind == scientific::F_NONE && all_near(vd, [1.2142857142857142, 1.635714285714286]));
    let x1, g1 = scientific::cubic_spline_ends(kx, ky, 3, 0.0, 1, 0.0);
    assert(g1.kind == scientific::F_VALUE);
    let x2, g2 = scientific::cubic_spline_ends(kx, ky, 1, 1.0 / 0.0, 1, 0.0);
    assert(g2.kind == scientific::F_VALUE);
}

// B-spline interpolation with end derivatives, pinned against SciPy.
fn test_interp_spline_bc() {
    let kx = [0.0, 0.7, 1.5, 2.4, 4.0];
    let ky = [1.0, 2.0, 0.5, 3.0, 2.2];
    let q = [0.3, 1.0, 3.0, 4.0];
    let e0: i64[0] = [];
    let f0: f64[0] = [];
    let a, fa = scientific::make_interp_spline_bc(kx, ky, 3, [1], [0.0], [1], [0.0]);
    let va, ea = scientific::bspline_eval(a, q, 0);
    assert(fa.kind == scientific::F_NONE && ea.kind == scientific::F_NONE && all_near(va, [1.4051118583571318, 1.5108523428116984, 3.312696206422242, 2.2]));
    let b, fb = scientific::make_interp_spline_bc(kx, ky, 3, [2], [0.0], [2], [0.0]);
    let vb, eb = scientific::bspline_eval(b, q, 0);
    assert(fb.kind == scientific::F_NONE && all_near(vb, [1.7098999308782359, 1.4199408943790213, 3.694483387911039, 2.2]));
    let c, fc = scientific::make_interp_spline_bc(kx, ky, 3, [1], [0.4], [2], [-0.6]);
    let vc, ec = scientific::bspline_eval(c, q, 0);
    assert(fc.kind == scientific::F_NONE && all_near(vc, [1.4506489986030193, 1.5068019484608233, 3.7633453622296695, 2.2]));
    let d, fd = scientific::make_interp_spline_bc(kx, ky, 5, [1, 2], [0.4, 0.1], [1, 3], [0.2, 0.5]);
    let vd, ed = scientific::bspline_eval(d, q, 0);
    assert(fd.kind == scientific::F_NONE && all_near(vd, [1.3077339227890068, 1.7093460382631585, 3.7899517744721862, 2.2]));
    let e, fe = scientific::make_interp_spline_bc(kx, ky, 2, [1], [0.4], e0, f0);
    let ve, ee = scientific::bspline_eval(e, q, 0);
    assert(fe.kind == scientific::F_NONE && all_near(ve, [1.2522448979591836, 2.2497767857142863, 7.298511904761907, 2.2]));
    let g, fg = scientific::make_interp_spline_bc(kx, ky, 4, [1, 2], [0.4, 1.0], [1], [0.3]);
    let vg, eg = scientific::bspline_eval(g, q, 0);
    assert(fg.kind == scientific::F_NONE && all_near(vg, [1.2461846446754583, 2.1484239716784668, 6.7424247589120405, 2.2]));
    let h1, g1 = scientific::make_interp_spline_bc(kx, ky, 3, [1], [0.4], e0, f0);
    assert(g1.kind == scientific::F_SHAPE);
    let h2, g2 = scientific::make_interp_spline_bc(kx, ky, 3, [0], [0.4], [1], [0.0]);
    assert(g2.kind == scientific::F_VALUE);
    let h3, g3 = scientific::make_interp_spline_bc(kx, ky, 3, [1], [1.0 / 0.0], [1], [0.0]);
    assert(g3.kind == scientific::F_VALUE);
}

// The design matrix of a B-spline, pinned against SciPy.
fn test_bspline_design_matrix() {
    let t = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
    let m, fm = scientific::bspline_design_matrix([-1.0, 0.0, 0.5, 2.5, 3.0, 5.0], t, 3, "yes");
    assert(fm.kind == scientific::F_NONE && m.rows == 6 && m.cols == 6);
    let d, fd = scientific::csr_to_dense(m);
    assert(fd.kind == scientific::F_NONE);
    assert(all_near(d, [8.0, -9.25, 2.4166666666666665, -0.16666666666666666, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.125, 0.59375, 0.26041666666666663, 0.020833333333333332, 0.0, 0.0, 0.0, 0.0, 0.020833333333333332, 0.26041666666666663, 0.59375, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.3333333333333333, 13.333333333333332, -38.0, 27.0]));
    let m2, f2 = scientific::bspline_design_matrix([1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);
    let d2, f3 = scientific::csr_to_dense(m2);
    assert(f2.kind == scientific::F_NONE && f3.kind == scientific::F_NONE);
    assert(all_near(d2, [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]));
    let m3, f4 = scientific::bspline_design_matrix([3.5, 4.5], t, 3, "periodic");
    let d3, f5 = scientific::csr_to_dense(m3);
    assert(f4.kind == scientific::F_NONE && f5.kind == scientific::F_NONE);
    assert(all_near(d3, [0.125, 0.59375, 0.26041666666666663, 0.020833333333333332, 0.0, 0.0, 0.0, 0.03125, 0.46875, 0.46875, 0.03125, 0.0]));
    // Faults.
    let e1, g1 = scientific::bspline_design_matrix([4.0], t, 3, "no");
    assert(g1.kind == scientific::F_RANGE);
    let e2, g2 = scientific::bspline_design_matrix([0.5], t, 3, "maybe");
    assert(g2.kind == scientific::F_VALUE);
    let e3, g3 = scientific::bspline_design_matrix([4.0], t, 3);
    assert(g3.kind == scientific::F_RANGE);
    let e4, g4 = scientific::bspline_design_matrix([0.5], [0.0, 1.0, 2.0], 3, "no");
    assert(g4.kind == scientific::F_SHAPE);
    let e5, g5 = scientific::bspline_design_matrix([0.5], t, -1, "no");
    assert(g5.kind == scientific::F_RANGE);
}

// The Bessel filter, pinned against SciPy.
fn test_bessel() {
    let b1, a1, f1 = scientific::bessel(4, [0.3], "lowpass");
    assert(f1.kind == scientific::F_NONE);
    assert(all_close(b1, [0.015961530353431626, 0.0638461214137265, 0.09576918212058975, 0.0638461214137265, 0.015961530353431626], 1.0e-9));
    assert(all_close(a1, [1.0, -1.4367543893190518, 0.9766689593856015, -0.33008120955383424, 0.04555112514219024], 1.0e-9));
    let b2, a2, f2 = scientific::bessel(3, [0.2], "lowpass", "delay");
    assert(f2.kind == scientific::F_NONE);
    assert(all_close(b2, [0.10193658742919462, 0.30580976228758383, 0.30580976228758383, 0.10193658742919462], 1.0e-9));
    assert(all_close(a2, [1.0, -0.3610204756237746, 0.20019422826451602, -0.023681053207185], 1.0e-9));
    let b3, a3, f3 = scientific::bessel(5, [0.4], "lowpass", "mag");
    assert(f3.kind == scientific::F_NONE);
    assert(all_close(b3, [0.06303420893294857, 0.3151710446647429, 0.6303420893294858, 0.6303420893294858, 0.3151710446647429, 0.06303420893294857], 1.0e-9));
    assert(all_close(a3, [1.0, 0.4856329189578801, 0.4307183479288292, 0.0777614975974916, 0.0221282543704655, 0.0008536669996880855], 1.0e-9));
    let b4, a4, f4 = scientific::bessel(4, [0.2, 0.5], "bandpass");
    assert(f4.kind == scientific::F_NONE);
    assert(all_close(a4, [1.0, -2.7701647240085996, 4.367024633703179, -4.592807605151357, 3.622387062989189, -2.103367363054253, 0.9116784049308485, -0.2610226067197879, 0.04555112514219019], 1.0e-9));
    let b5, a5, f5 = scientific::bessel(6, [0.3], "highpass", "mag");
    assert(f5.kind == scientific::F_NONE);
    assert(all_close(b5, [0.2747192665755552, -1.6483155994533312, 4.120788998633327, -5.494385331511103, 4.120788998633327, -1.6483155994533312, 0.2747192665755552], 1.0e-9));
    let b6, a6, f6 = scientific::bessel(8, [0.25], "lowpass");
    assert(f6.kind == scientific::F_NONE);
    assert(all_close(a6, [1.0, -3.718741912749568, 6.488924099995361, -6.836287042161693, 4.710807689534221, -2.159316602316135, 0.6396221620719462, -0.11148601320422366, 0.008725391960805028], 1.0e-8));
    // The sections filter like the polynomials.
    let sos, fs = scientific::bessel_sos(4, [0.3], "lowpass");
    assert(fs.kind == scientific::F_NONE && sos.shape[0] == 2 && sos.shape[1] == 6);
    let x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let ya, fya = scientific::lfilter(b1.data, a1.data, x);
    let yb, fyb = scientific::sosfilt(sos, x);
    assert(fya.kind == scientific::F_NONE && fyb.kind == scientific::F_NONE && all_close(yb, ya.data, 1.0e-9));
    // Faults.
    let e1, e2, g1 = scientific::bessel(4, [0.3], "lowpass", "other");
    assert(g1.kind == scientific::F_VALUE);
    let e3, e4, g2 = scientific::bessel(0, [0.3], "lowpass");
    assert(g2.kind == scientific::F_VALUE);
    let e5, e6, g3 = scientific::bessel(4, [1.5], "lowpass");
    assert(g3.kind == scientific::F_RANGE);
    let e7, g4 = scientific::bessel_sos(4, [0.3], "lowpass", "other");
    assert(g4.kind == scientific::F_VALUE);
}

fn test_spline_ends() {
    let x = [0.0, 1.0, 2.5, 4.0, 5.0];
    let c, f = scientific::cubic_spline(x, [1.0, 3.0, 2.0, 5.0, 1.0], "periodic");
    assert(f.kind == scientific::F_NONE);
    let v, fv = scientific::spline_eval(c, [0.0, 0.5, 1.7, 3.3, 5.0, 6.0, -1.0, 11.2]);
    let ex = [1.0, 1.6375000000000002, 2.6586074074074078, 4.297718518518518, 1.0, 3.0, 5.0, 3.1597925925925923];
    var i: i64 = 0;
    while i < 8 { assert(mathshim_abs(v.data[i] - ex[i]) < 1.0e-10); i += 1; }
    let c2, f2 = scientific::cubic_spline(x, [1.0, 3.0, 2.0, 5.0, 2.0], "clamped", 1.5, -0.5);
    assert(f2.kind == scientific::F_NONE);
    let v2, fv2 = scientific::spline_eval(c2, [0.0, 0.5, 1.7, 3.3, 5.0]);
    let ex2 = [1.0, 2.0993055555555555, 2.477471604938272, 4.180187654320987, 2.0];
    i = 0;
    while i < 5 { assert(mathshim_abs(v2.data[i] - ex2[i]) < 1.0e-10); i += 1; }
    let b1, g1 = scientific::cubic_spline(x, [1.0, 3.0, 2.0, 5.0, 2.0], "periodic");
    assert(g1.kind == scientific::F_VALUE);
    let b2, g2 = scientific::cubic_spline(x, [1.0, 3.0, 2.0, 5.0, 2.0], "clamped", 1.0 / 0.0, 0.0);
    assert(g2.kind == scientific::F_VALUE);
}

// The B-spline functions are pinned against SciPy for the degrees 1,
// 2, 3 and 5, with derivatives, extrapolation and the faults.
fn test_bspline() {
    let x = [0.0, 1.0, 2.5, 4.0, 5.0, 6.5, 8.0];
    let y = [1.0, 3.0, 2.0, 5.0, 1.0, 0.5, 2.0];
    let pts = [0.0, 0.5, 1.7, 4.4, 8.0, 9.0, -1.0];
    let ks = [1, 2, 3, 5];
    let b1, f1 = scientific::make_interp_spline(x, y, 1);
    assert(f1.kind == scientific::F_NONE);
    let v1, fv1 = scientific::splev(pts, b1);
    let e1 = [1.0, 2.0, 2.533333333333333, 3.3999999999999986, 2.0, 2.9999999999999996, -1.0];
    let d1, fd1 = scientific::splev([0.5, 4.4], b1, 1);
    let ed1 = [2.0, -4.0];
    let s1, fs1 = scientific::splev([0.5, 4.4], b1, 2);
    let es1 = [0.0, 0.0];
    var i1: i64 = 0;
    while i1 < 7 { assert(mathshim_abs(v1.data[i1] - e1[i1]) < 1.0e-8 * (1.0 + mathshim_abs(e1[i1]))); i1 += 1; }
    assert(mathshim_abs(d1.data[0] - ed1[0]) < 1.0e-8 && mathshim_abs(d1.data[1] - ed1[1]) < 1.0e-8);
    assert(mathshim_abs(s1.data[0] - es1[0]) < 1.0e-7 && mathshim_abs(s1.data[1] - es1[1]) < 1.0e-7);
    let b2, f2 = scientific::make_interp_spline(x, y, 2);
    assert(f2.kind == scientific::F_NONE);
    let v2, fv2 = scientific::splev(pts, b2);
    let e2 = [1.0, 2.4070264444072773, 2.4625541246213594, 3.707898385088548, 2.0, 3.394621917451822, -4.2562115552582185];
    let d2, fd2 = scientific::splev([0.5, 4.4], b2, 1);
    let ed2 = [2.0, -4.810139190871756];
    let s2, fs2 = scientific::splev([0.5, 4.4], b2, 2);
    let es2 = [-3.2562115552582185, -7.899425767965655];
    var i2: i64 = 0;
    while i2 < 7 { assert(mathshim_abs(v2.data[i2] - e2[i2]) < 1.0e-8 * (1.0 + mathshim_abs(e2[i2]))); i2 += 1; }
    assert(mathshim_abs(d2.data[0] - ed2[0]) < 1.0e-8 && mathshim_abs(d2.data[1] - ed2[1]) < 1.0e-8);
    assert(mathshim_abs(s2.data[0] - es2[0]) < 1.0e-7 && mathshim_abs(s2.data[1] - es2[1]) < 1.0e-7);
    let b3, f3 = scientific::make_interp_spline(x, y, 3);
    assert(f3.kind == scientific::F_NONE);
    let v3, fv3 = scientific::splev(pts, b3);
    let e3 = [1.0, 2.7759259259259257, 2.1610370370370373, 3.7092444444444412, 2.0, -1.76954732510288, -10.26296296296296];
    let d3, fd3 = scientific::splev([0.5, 4.4], b3, 1);
    let ed3 = [1.74537037037037, -4.399185185185186];
    let s3, fs3 = scientific::splev([0.5, 4.4], b3, 2);
    let es3 = [-6.207407407407406, -3.6718518518518444];
    var i3: i64 = 0;
    while i3 < 7 { assert(mathshim_abs(v3.data[i3] - e3[i3]) < 1.0e-8 * (1.0 + mathshim_abs(e3[i3]))); i3 += 1; }
    assert(mathshim_abs(d3.data[0] - ed3[0]) < 1.0e-8 && mathshim_abs(d3.data[1] - ed3[1]) < 1.0e-8);
    assert(mathshim_abs(s3.data[0] - es3[0]) < 1.0e-7 && mathshim_abs(s3.data[1] - es3[1]) < 1.0e-7);
    let b5, f5 = scientific::make_interp_spline(x, y, 5);
    assert(f5.kind == scientific::F_NONE);
    let v5, fv5 = scientific::splev(pts, b5);
    let e5 = [1.0, 4.084333544764113, 1.0722625024332686, 3.8503072596962076, 2.0, -62.79030311967932, -45.52148214932095];
    let d5, fd5 = scientific::splev([0.5, 4.4], b5, 1);
    let ed5 = [0.39162517302690636, -3.9602456578765066];
    let s5, fs5 = scientific::splev([0.5, 4.4], b5, 2);
    let es5 = [-15.832241676460812, -4.525612894168046];
    var i5: i64 = 0;
    while i5 < 7 { assert(mathshim_abs(v5.data[i5] - e5[i5]) < 1.0e-8 * (1.0 + mathshim_abs(e5[i5]))); i5 += 1; }
    assert(mathshim_abs(d5.data[0] - ed5[0]) < 1.0e-8 && mathshim_abs(d5.data[1] - ed5[1]) < 1.0e-8);
    assert(mathshim_abs(s5.data[0] - es5[0]) < 1.0e-7 && mathshim_abs(s5.data[1] - es5[1]) < 1.0e-7);
    // The knots and the coefficients of the cubic match SciPy.
    let kn = [0.0, 0.0, 0.0, 0.0, 2.5, 4.0, 5.0, 8.0, 8.0, 8.0, 8.0];
    let co = [1.0, 5.677469135802468, -2.276851851851851, 7.829629629629628, -4.521604938271605, 2.9796296296296294, 2.0];
    let bk = scientific::bs_knots_of(b3);
    let bc = scientific::bs_coefs_of(b3);
    assert(bk.len == kn.len && bc.len == co.len);
    var q: i64 = 0;
    while q < kn.len { assert(mathshim_abs(bk[q] - kn[q]) < 1.0e-12); q += 1; }
    q = 0;
    while q < co.len { assert(mathshim_abs(bc[q] - co[q]) < 1.0e-8); q += 1; }
    // A spline built from knots and coefficients gives the same values.
    let bb, fb = scientific::bspline(bk, bc, 3);
    assert(fb.kind == scientific::F_NONE);
    let vb, fvb = scientific::bspline_eval(bb, pts);
    assert(mathshim_abs(vb.data[2] - v3.data[2]) < 1.0e-12);
    // A derivative above the degree is zero.
    let z, fz = scientific::splev([1.0], b1, 2);
    assert(mathshim_abs(z.data[0]) < 1.0e-12);
    let sr, fsr = scientific::splrep(x, y, 3);
    assert(fsr.kind == scientific::F_NONE);
    let sg, fsg = scientific::splrep(x, y, 3, 0.5);
    assert(fsg.kind == scientific::F_VALUE);
    let e1, g1 = scientific::make_interp_spline([0.0, 1.0], [1.0, 2.0], 3);
    assert(g1.kind == scientific::F_SHAPE);
    let e2, g2 = scientific::make_interp_spline([0.0, 1.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0], 3);
    assert(g2.kind == scientific::F_VALUE);
    let e3, g3 = scientific::make_interp_spline(x, y, 21);
    assert(g3.kind == scientific::F_RANGE);
    let e4, g4 = scientific::bspline([0.0, 0.0, 1.0], [1.0, 2.0], 1);
    assert(g4.kind == scientific::F_SHAPE);
    let e5, g5 = scientific::bspline([0.0, 1.0, 0.5, 1.0], [1.0, 2.0], 1);
    assert(g5.kind == scientific::F_VALUE);
    let e6, g6 = scientific::splev([], b3);
    assert(g6.kind == scientific::F_SHAPE);
    let e7, g7 = scientific::splev([1.0], b3, -1);
    assert(g7.kind == scientific::F_RANGE);
}

// The windows of scipy.signal.windows are pinned against SciPy.
fn win_close(a: scientific::Arr&, e: f64[:]) -> bool {
    if a.data.len != e.len { return false; }
    var i: i64 = 0;
    while i < e.len {
        if mathshim_abs(a.data[i] - e[i]) > 1.0e-10 { return false; }
        i += 1;
    }
    return true;
}

fn test_windows_more() {
    let w0, f0 = scientific::general_cosine(9, [0.5, 0.3, 0.2]);
    assert(f0.kind == scientific::F_NONE);
    assert(win_close(w0, [0.4, 0.2878679656440357, 0.3, 0.7121320343559643, 1.0, 0.7121320343559643, 0.3, 0.2878679656440357, 0.4]));
    let w1, f1 = scientific::general_hamming(8, 0.6, false);
    assert(f1.kind == scientific::F_NONE);
    assert(win_close(w1, [0.19999999999999996, 0.31715728752538097, 0.6, 0.882842712474619, 1.0, 0.882842712474619, 0.6, 0.31715728752538097]));
    let w2, f2 = scientific::general_gaussian(9, 1.5, 2.0);
    assert(f2.kind == scientific::F_NONE);
    assert(win_close(w2, [0.01831563888873418, 0.18498139990730428, 0.6065306597126334, 0.9394130628134758, 1.0, 0.9394130628134758, 0.6065306597126334, 0.18498139990730428, 0.01831563888873418]));
    let w3, f3 = scientific::taylor(11);
    assert(f3.kind == scientific::F_NONE);
    assert(win_close(w3, [0.26248807473298397, 0.40104077462163656, 0.6089456801270016, 0.8084391241352048, 0.9490117072000545, 1.0, 0.9490117072000545, 0.8084391241352048, 0.6089456801270016, 0.40104077462163656, 0.26248807473298397]));
    let w4, f4 = scientific::taylor(10, 6, 40.0, false, false);
    assert(f4.kind == scientific::F_NONE);
    assert(win_close(w4, [0.22955793961566962, 0.5046487811660969, 0.9082773809058199, 1.3273435078317122, 1.6468903591489026, 1.7665640626635986, 1.6468903591489026, 1.3273435078317122, 0.9082773809058199, 0.5046487811660971]));
    let w5, f5 = scientific::chebwin(11, 80.0);
    assert(f5.kind == scientific::F_NONE);
    assert(win_close(w5, [0.01688772555101567, 0.09690563774467226, 0.2915289995121729, 0.5900876054755189, 0.8788139405478855, 1.0, 0.8788139405478855, 0.5900876054755189, 0.2915289995121729, 0.09690563774467226, 0.01688772555101567]));
    let w6, f6 = scientific::chebwin(10, 60.0);
    assert(f6.kind == scientific::F_NONE);
    assert(win_close(w6, [0.044313249478010214, 0.1888932622106898, 0.45729052434559303, 0.7774679588943729, 1.0, 1.0, 0.7774679588943729, 0.45729052434559303, 0.1888932622106898, 0.044313249478010214]));
    let w7, f7 = scientific::chebwin(9, 100.0, false);
    assert(f7.kind == scientific::F_NONE);
    assert(win_close(w7, [0.015138861629262326, 0.10443925148320238, 0.3446060821095421, 0.7092668575677474, 1.0, 1.0, 0.7092668575677474, 0.3446060821095421, 0.10443925148320238]));
    let w8, f8 = scientific::kaiser_bessel_derived(10, 4.0);
    assert(f8.kind == scientific::F_NONE);
    assert(win_close(w8, [0.17006094170460917, 0.4417398738085654, 0.7071067811865475, 0.897143179145666, 0.9854335472808613, 0.9854335472808613, 0.897143179145666, 0.7071067811865475, 0.4417398738085654, 0.17006094170460917]));
    let a1, g1 = scientific::kaiser_bessel_derived(9, 4.0);
    assert(g1.kind == scientific::F_VALUE);
    let a2, g2 = scientific::chebwin(0, 80.0);
    assert(g2.kind == scientific::F_RANGE);
    let a3, g3 = scientific::general_cosine(5, []);
    assert(g3.kind == scientific::F_SHAPE);
    let a4, g4 = scientific::general_gaussian(5, 0.0, 1.0);
    assert(g4.kind == scientific::F_VALUE);
    let a5, g5 = scientific::taylor(8, 0);
    assert(g5.kind == scientific::F_RANGE);
    let a6, g6 = scientific::general_hamming(1, 0.5);
    assert(g6.kind == scientific::F_NONE && a6.data[0] == 1.0);
}

// dpss is pinned against SciPy for one window, several windows, the
// periodic form and the norms.
fn test_dpss() {
    let d0, f0 = scientific::dpss(16, 2.5);
    assert(f0.kind == scientific::F_NONE);
    assert(win_close(d0, [0.016317047030418655, 0.06340126386238834, 0.15845631025268223, 0.3086652017129723, 0.5032772641781225, 0.7113142022250394, 0.8882720166419921, 0.9903288201160542, 0.9903288201160542, 0.8882720166419918, 0.711314202225039, 0.5032772641781222, 0.308665201712972, 0.15845631025268206, 0.06340126386238829, 0.01631704703041864]));
    let d1, f1 = scientific::dpss(15, 3.0, 3);
    assert(f1.kind == scientific::F_NONE);
    assert(win_close(d1, [0.0037469532703791183, 0.019738598733284245, 0.06007178916483362, 0.1335601667570441, 0.23662603744318103, 0.34848806231558105, 0.4363318253551655, 0.4697406456190516, 0.4363318253551655, 0.34848806231558105, 0.23662603744318103, 0.1335601667570441, 0.06007178916483362, 0.019738598733284245, 0.0037469532703791187, 0.018455407259290287, 0.07582785828710926, 0.18125465349655787, 0.3095306632862646, 0.3998304228518963, 0.38528012874584927, 0.2386250369880479, 5.628710300965063e-17, -0.2386250369880478, -0.3852801287458492, -0.39983042285189624, -0.30953066328626455, -0.1812546534965578, -0.07582785828710924, -0.01845540725929028, 0.06185554949593948, 0.18884289924318579, 0.33548858947046506, 0.400156093986345, 0.2997028445832034, 0.05223901255414873, -0.20925037531544208, -0.3209309213184132, -0.20925037531544213, 0.052239012554148694, 0.2997028445832034, 0.40015609398634505, 0.3354885894704652, 0.18884289924318587, 0.06185554949593951]));
    let d2, f2 = scientific::dpss(12, 2.0, 2, false, "approximate");
    assert(f2.kind == scientific::F_NONE);
    assert(win_close(d2, [0.048953888295962124, 0.1545806990403657, 0.32735499813886637, 0.5480588250591538, 0.7710682557928902, 0.9380097392736036, 1.0, 0.9380097392736036, 0.7710682557928903, 0.5480588250591539, 0.3273549981388664, 0.15458069904036575, 0.20660001936400177, 0.4677575329825821, 0.7306823089939881, 0.8721382022125558, 0.7924393885658259, 0.47356882249718596, -1.5049496818025036e-16, -0.4735688224971862, -0.7924393885658259, -0.8721382022125558, -0.7306823089939883, -0.4677575329825822]));
    let d3, f3 = scientific::dpss(20, 4.0, 4);
    assert(f3.kind == scientific::F_NONE);
    assert(win_close(d3, [0.00040168075357837387, 0.0028078249873478046, 0.011014484114096583, 0.03124001708094586, 0.07053306070242975, 0.1331186629685207, 0.21605978541162896, 0.30691464981244326, 0.38577536042617855, 0.43185430912259537, 0.43185430912259537, 0.38577536042617855, 0.30691464981244326, 0.21605978541162899, 0.13311866296852076, 0.07053306070242979, 0.03124001708094588, 0.011014484114096591, 0.002807824987347807, 0.00040168075357837425, 0.0023307242194475575, 0.013578066291007102, 0.04471378585280919, 0.10588329369261998, 0.19657350464766118, 0.2969486471173236, 0.3687184673821026, 0.3697111529304576, 0.27669497439742036, 0.10286016678522557, -0.10286016678522629, -0.27669497439742086, -0.36971115293045764, -0.3687184673821026, -0.2969486471173236, -0.19657350464766107, -0.1058832936926199, -0.04471378585280916, -0.013578066291007095, -0.0023307242194475566, 0.009331443187915733, 0.04418706065491568, 0.11916100702256513, 0.22812125781127837, 0.3314737853232691, 0.3668641255351118, 0.2859263317002585, 0.09551522418132546, -0.1298655142219386, -0.2819849996425258, -0.28198499964252577, -0.12986551422193848, 0.09551522418132562, 0.2859263317002586, 0.3668641255351118, 0.33147378532326915, 0.22812125781127843, 0.11916100702256517, 0.04418706065491569, 0.009331443187915737, 0.029639663336898242, 0.11032283569674116, 0.2351733031247714, 0.3451313944387695, 0.35394673663997367, 0.21423963250065986, -0.02531357323790679, -0.23084178071942488, -0.27089421536962716, -0.11848589453020822, 0.11848589453020833, 0.2708942153696272, 0.23084178071942482, 0.025313573237906697, -0.21423963250065992, -0.3539467366399736, -0.3451313944387694, -0.23517330312477128, -0.11032283569674113, -0.029639663336898235]));
    let d4, f4 = scientific::dpss(64, 3.5);
    assert(f4.kind == scientific::F_NONE);
    assert(win_close(d4, [0.0004363528176095625, 0.0011694279650277303, 0.002471684070279874, 0.004590970503823901, 0.007828996859395615, 0.012539975346116394, 0.01912564530369338, 0.028026299430360557, 0.03970764837226708, 0.054643619458447985, 0.07329547202426853, 0.09608790775903146, 0.12338313914535201, 0.15545413055250948, 0.19245842366382487, 0.23441408264775265, 0.28117932966896736, 0.33243737810994123, 0.38768780575300665, 0.44624554672951394, 0.50724823005927, 0.5696721716829668, 0.6323568595232404, 0.6940372852502605, 0.7533830026694702, 0.8090423621320525, 0.859690012565966, 0.9040755032813562, 0.9410706764910581, 0.9697135310354669, 0.9892463622340031, 0.9991462373460177, 0.9991462373460178, 0.9892463622340031, 0.9697135310354669, 0.9410706764910582, 0.9040755032813562, 0.8596900125659659, 0.8090423621320523, 0.7533830026694701, 0.6940372852502604, 0.6323568595232404, 0.5696721716829667, 0.50724823005927, 0.44624554672951394, 0.38768780575300665, 0.3324373781099413, 0.2811793296689674, 0.23441408264775268, 0.19245842366382487, 0.15545413055250948, 0.12338313914535198, 0.09608790775903142, 0.07329547202426848, 0.05464361945844795, 0.039707648372267056, 0.028026299430360533, 0.019125645303693366, 0.012539975346116386, 0.00782899685939561, 0.004590970503823897, 0.0024716840702798722, 0.0011694279650277294, 0.0004363528176095621]));
    // The norm subsample and the concentration ratios.
    let ds, fds = scientific::dpss(8, 2.0, 0, true, "subsample");
    assert(fds.kind == scientific::F_NONE && win_close(ds, [0.08664829928367265, 0.330962077699254, 0.686872284903535, 0.9595673129729012, 0.9595673129729012, 0.686872284903535, 0.33096207769925395, 0.08664829928367264]));
    let ds2, fds2 = scientific::dpss(10, 2.5, 0, true, "subsample");
    assert(fds2.kind == scientific::F_NONE && win_close(ds2, [0.03348635548276011, 0.1611735001228393, 0.4175208958618425, 0.7384470765647055, 0.9675042425395557, 0.9675042425395557, 0.7384470765647057, 0.41752089586184254, 0.16117350012283935, 0.03348635548276011]));
    let dw, dr, fdr = scientific::dpss_ratios(8, 2.0, 0, false);
    assert(fdr.kind == scientific::F_NONE && dr.ndim == 0 && near(dr.data[0], 0.9999780799508334));
    assert(win_close(dw, [0.0740907202847688, 0.27352337476511307, 0.5821741797391821, 0.8772657297808555, 1.0, 0.8772657297808554, 0.5821741797391818, 0.27352337476511296]));
    let dw2, dr2, fdr2 = scientific::dpss_ratios(16, 3.0, 4);
    assert(fdr2.kind == scientific::F_NONE && dr2.ndim == 1 && win_close(dr2, [0.9999999527920653, 0.9999957218401485, 0.9998315378762141, 0.9962957001645156]));
    let e1, g1 = scientific::dpss(8, 4.0);
    assert(g1.kind == scientific::F_RANGE);
    let e2, g2 = scientific::dpss(8, 0.0);
    assert(g2.kind == scientific::F_RANGE);
    let e3, g3 = scientific::dpss(8, 2.0, 9);
    assert(g3.kind == scientific::F_RANGE);
    let e4, g4 = scientific::dpss(8, 2.0, 0, true, "bogus");
    assert(g4.kind == scientific::F_VALUE);
    let e5, g5 = scientific::dpss(0, 2.0);
    assert(g5.kind == scientific::F_RANGE);
}

// cumulative_simpson takes a start value, as SciPy does.
fn test_cumulative_simpson_initial() {
    let y = [1.0, 3.0, 2.0, 5.0, 4.0];
    let x = [0.0, 1.0, 2.5, 3.0, 4.5];
    // The axis form.
    var cy: f64[>..] = [];
    var cq: i64 = 0;
    while cq < 24 { cy.push(mathshim::power(cq, 1.5)); cq += 1; }
    let cya, cyf = scientific::from_values(cy);
    let cyr, cyg = scientific::reshape(cya, [2, 3, 4]);
    assert(cyf.kind == scientific::F_NONE && cyg.kind == scientific::F_NONE);
    let ca1, caf1 = scientific::cumulative_simpson_axis(cyr, 1, [], 0.5);
    assert(caf1.kind == scientific::F_NONE && ca1.shape[1] == 2 && all_close(ca1, [1.7238576250846034, 2.810113295832983, 4.170619444818321, 5.735831784968369, 9.104569499661586, 12.120226591665965, 15.539826258871042, 19.29334363773743, 26.26680740867807, 29.119546183128683, 32.06946372096823, 35.11342772281442, 64.50198974694078, 70.57957311951756, 76.84041326725652, 83.27919915639917], 1.0e-6));
    let ca2, caf2 = scientific::cumulative_simpson_axis(cyr, 2, [0.0, 1.0, 3.0, 4.5], 1.0, 0.0);
    assert(caf2.kind == scientific::F_NONE && ca2.shape[2] == 4 && mathshim_abs(ca2.data[3] - 10.28299733046346) < 1.0e-6 && mathshim_abs(ca2.data[23] - 453.835084850994) < 1.0e-5);
    let ca3, caf3 = scientific::cumulative_simpson_axis(cyr, 2, [0.0, 1.0, 3.0]);
    assert(caf3.kind == scientific::F_SHAPE);
    let ca4, caf4 = scientific::cumulative_simpson_axis(cyr, 3);
    assert(caf4.kind == scientific::F_RANGE);
    let a, f = scientific::cumulative_simpson(y, x, 2.0);
    assert(f.kind == scientific::F_NONE && a.data.len == 5);
    assert(win_close(a, [2.0, 4.17777777777777778, 8.52777777777777778, 10.34722222222222222, 18.97222222222222222]) || (mathshim_abs(a.data[1] - 4.177777777777778) < 1.0e-12 && mathshim_abs(a.data[4] - 18.97222222222222) < 1.0e-12));
    let b, g = scientific::cumulative_simpson_dx(y, 0.5, 0.0);
    assert(g.kind == scientific::F_NONE && b.data.len == 5);
    assert(mathshim_abs(b.data[0]) < 1.0e-15 && mathshim_abs(b.data[1] - 1.125) < 1.0e-12 && mathshim_abs(b.data[3] - 4.41666666666666667) < 1.0e-12 && mathshim_abs(b.data[4] - 6.83333333333333333) < 1.0e-12);
    let c, h = scientific::cumulative_simpson(y, x);
    assert(h.kind == scientific::F_NONE && c.data.len == 4);
    let d, k = scientific::cumulative_simpson(y, x, 1.0 / 0.0);
    assert(k.kind == scientific::F_VALUE);
}

// solve_ivp takes the times t_eval and gives the values from the dense
// output of the method, as SciPy does.
// The method of Dormand and Prince of the order eight, pinned against
// SciPy 1.18.1.
fn test_dop853() {
    // The steps of the method. The times and the states match SciPy.
    let e1 = [0.0, 0.000999000999000999, 0.010989010989010988, 0.11088911088911087, 1.1098901098901095, 3.577580383341898, 5.0];
    let e2 = [1.0, 0.0, 0.9999995009985435, -0.0009990008328333421, 0.9999396214263468, -0.010988789821184265, 0.9938581000167407, -0.11066199451986314, 0.44475998750894596, -0.8956497674212576, -0.9063114415145316, 0.42208333978208246, 0.2834614005249253, 0.9587505956092199];
    let t1, y1, f1 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "DOP853") { t, v => [v[1], -v[0]] };
    assert(f1.kind == scientific::F_NONE && t1.data.len == 7 && y1.data.len == 14);
    assert(all_near(t1, e1) && all_near(y1, e2));
    // The values between the steps, from the dense output.
    let e3 = [0.8775825615538217, -0.479425538086932, -0.12884449379636104, -0.9916648039614262, 0.46851666612604054, 0.883454646570579];
    let t2, y2, d2, f2 = scientific::solve_ivp_dense(0.0, 6.0, [1.0, 0.0], 1.0e-8, 1.0e-10, "DOP853") { t, v => [v[1], -v[0]] };
    assert(f2.kind == scientific::F_NONE);
    let dv, dvf = scientific::dense_eval(d2, [0.5, 1.7, 5.2]);
    assert(dvf.kind == scientific::F_NONE && all_near(dv, e3));
    // The values at the times of t_eval.
    let e4 = [1.0, 0.0, 0.9553364760057397, -0.29552021042511084, 0.5403013851324846, -0.8414706099471434, 0.020795124339923032, -0.9997845908803505, -0.989992409386347, -0.141120163362502];
    let t3, y3, f3 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1.0e-6, 1.0e-9, "DOP853", 1.0e308, [0.0, 0.3, 1.0, 1.55, 3.0]) { t, v => [v[1], -v[0]] };
    assert(f3.kind == scientific::F_NONE && all_near(y3, e4));
    // The time and the state of an event.
    let et, ey, eet, eei, eey, ef = scientific::solve_ivp_events(0.0, 10.0, [10.0, 0.0], 1, [true], [-1.0], 1.0e-8, 1.0e-10, "DOP853") { t, v => [v[1], -9.81, v[0]] };
    assert(ef.kind == scientific::F_NONE && eet.data.len == 1);
    assert(mathshim_abs(eet.data[0] - 1.4278431229270638) < 1.0e-9);
    assert(mathshim_abs(eey.data[1] + 14.00714103591449) < 1.0e-9);
}

// The method BDF, pinned against SciPy 1.18.1. The count of steps equals
// the count of SciPy and the values agree to the tolerance of the solver.
fn test_bdf() {
    // A stiff pair, the oscillator of Van der Pol with a large damping.
    let t1, y1, f1 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 0.001, 1e-6, "BDF") { t, v => [v[1], 1000.0 * (1.0 - v[0] * v[0]) * v[1] - v[0]] };
    assert(f1.kind == scientific::F_NONE && t1.data.len == 142);
    assert(mathshim_abs(t1.data[141] - 3.0) < 1.0e-12 && mathshim_abs(t1.data[3] - 0.0014270485380032654) < 1.0e-9);
    // The system of Robertson, with three components.
    let t2, y2, f2 = scientific::solve_ivp(0.0, 40.0, [1.0, 0.0, 0.0], 1e-6, 1e-9, "BDF") { t, v => [-0.04 * v[0] + 1.0e4 * v[1] * v[2], 0.04 * v[0] - 1.0e4 * v[1] * v[2] - 3.0e7 * v[1] * v[1], 3.0e7 * v[1] * v[1]] };
    assert(f2.kind == scientific::F_NONE && t2.data.len == 130);
    assert(mathshim_abs(y2.data[3 * 129] - 0.7158277520683286) < 1.0e-8);
    // The values at the times of t_eval, from the dense output.
    let e3 = [1.0, 0.0, -2.001004843907636, 0.00665546563554308, -1.9976659519432036, 0.006679561733094654, -1.99297673523183, 0.0067058161092881, -1.9848951527017453, 0.006751652306705219];
    let t3, y3, f3 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1e-4, 1e-7, "BDF", 1.0e308, [0.0, 0.5, 1.0, 1.7, 2.9]) { t, v => [v[1], 100.0 * (1.0 - v[0] * v[0]) * v[1] - v[0]] };
    assert(f3.kind == scientific::F_NONE && all_close(y3, e3, 1.0e-8));
    // The time and the state of an event.
    let et, ey, eet, eei, eey, ef = scientific::solve_ivp_events(0.0, 3.0, [1.0, 0.0], 1, [true], [0.0], 1e-6, 1e-9, "BDF") { t, v => [v[1], 100.0 * (1.0 - v[0] * v[0]) * v[1] - v[0], v[0] - 0.5] };
    assert(ef.kind == scientific::F_NONE && et.data.len == 104);
    assert(mathshim_abs(eet.data[0] - 0.4154166021038258) < 1.0e-8 && mathshim_abs(eey.data[1] + 21.22445517558779) < 1.0e-5);
}

// The method Radau, pinned against SciPy 1.18.1. The count of steps
// equals the count of SciPy.
// head_close compares the first values of an array with a list.
fn head_close(a: scientific::Arr&, v: f64[:], tol: f64) -> bool {
    if a.data.len < v.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        if mathshim_abs(a.data[i] - v[i]) > tol * (1.0 + mathshim_abs(v[i])) { return false; }
        i += 1;
    }
    return true;
}

// The fast Hankel transform, pinned against SciPy.
fn test_fht() {
    var g: f64[>..] = [];
    var i: i64 = 0;
    while i < 16 { let x = -3.0 + 6.0 * i / 15.0; g.push(exp(-x * x)); i += 1; }
    let r1, f1 = scientific::fht(g, 0.1, 0.0, 0.0, 0.0);
    assert(f1.kind == scientific::F_NONE && head_close(r1, [0.16658286794408866, 0.2125535499243573, 0.12136785788964644, -0.055993321795547724], 1.0e-11));
    let r2, f2 = scientific::fht(g, 0.4, 0.5, 0.3, 0.7);
    assert(f2.kind == scientific::F_NONE && head_close(r2, [0.11360093664708755, 0.08674346332121119, 0.13744050051674025, 0.23965782425244547], 1.0e-11));
    let r3, f3 = scientific::fht(g, 0.4, 2.0, -0.2, -0.3);
    assert(f3.kind == scientific::F_NONE && head_close(r3, [-0.0010266310475184393, 0.00022635263065843024, 0.001499351158171856, 0.0048398554983498645], 1.0e-11));
    let r4, f4 = scientific::fht(g, 0.4, 0.5, 0.0, 1.5);
    assert(f4.kind == scientific::F_NONE && head_close(r4, [24.602791125026318, 8.050436997621295, 0.8187829868428674, 0.15983414122779394], 1.0e-11));
    let r5, f5 = scientific::fht(g, 0.4, -1.0, 0.0, 0.0);
    assert(f5.kind == scientific::F_NONE && head_close(r5, [0.5482819240477328, 0.5403825991188047, 0.5250854033959822, 0.4920255609315086], 1.0e-11));
    let back, fb = scientific::ifht(r2.data, 0.4, 0.5, 0.3, 0.7);
    assert(fb.kind == scientific::F_NONE && mathshim_abs(back.data[0] - g[0]) < 1.0e-10 && mathshim_abs(back.data[9] - g[9]) < 1.0e-10);
    var h: f64[>..] = [];
    i = 0;
    while i < 15 { let x = -3.0 + 6.0 * i / 14.0; h.push(exp(-x * x)); i += 1; }
    let r6, f6 = scientific::fht(h, 0.3, 1.0, 0.1, 0.2);
    assert(f6.kind == scientific::F_NONE && head_close(r6, [0.017451947686648563, 0.05451441806908365, 0.06457579551959093, 0.10553434465142811], 1.0e-11));
    let one, f7 = scientific::fht([2.0], 0.5, 1.0, 0.0, 0.0);
    assert(f7.kind == scientific::F_NONE && one.data.len == 1 && mathshim_abs(one.data[0] - 2.0) < 1.0e-12);
    let o1, g1 = scientific::fhtoffset(0.1, 2.0, 0.5, 0.0);
    let o2, g2 = scientific::fhtoffset(0.3, -0.5, 0.2, 0.4);
    assert(g1.kind == scientific::F_NONE && g2.kind == scientific::F_NONE);
    assert(mathshim_abs(o1 - 0.5454581477676637) < 1.0e-12 && mathshim_abs(o2 - 0.07335295675204884) < 1.0e-12);
    // Faults: a singular bias, an empty signal and bad numbers.
    let s1, h1 = scientific::fht(g, 0.4, 0.5, 0.0, -1.5);
    assert(h1.kind == scientific::F_VALUE);
    let s2, h2 = scientific::ifht(g, 0.4, 0.5, 0.0, 1.5);
    assert(h2.kind == scientific::F_VALUE);
    let e: f64[0] = [];
    let s3, h3 = scientific::fht(e, 0.4, 0.5, 0.0, 0.0);
    assert(h3.kind == scientific::F_SHAPE);
    let s4, h4 = scientific::fht(g, 0.0, 0.5, 0.0, 0.0);
    assert(h4.kind == scientific::F_VALUE);
    let s5, h5 = scientific::fht(g, 0.4, 1.0 / 0.0, 0.0, 0.0);
    assert(h5.kind == scientific::F_VALUE);
    let o3, h6 = scientific::fhtoffset(0.0, 1.0, 0.0, 0.0);
    assert(h6.kind == scientific::F_VALUE);
}

fn test_radau() {
    let t1, y1, f1 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 0.001, 1e-6, "Radau") { t, v => [v[1], 1000.0 * (1.0 - v[0] * v[0]) * v[1] - v[0]] };
    assert(f1.kind == scientific::F_NONE && t1.data.len == 68);
    assert(mathshim_abs(t1.data[67] - 3.0) < 1.0e-12 && mathshim_abs(t1.data[3] - 0.034402725378283136) < 1.0e-4);
    let t2, y2, f2 = scientific::solve_ivp(0.0, 40.0, [1.0, 0.0, 0.0], 1e-6, 1e-9, "Radau") { t, v => [-0.04 * v[0] + 1.0e4 * v[1] * v[2], 0.04 * v[0] - 1.0e4 * v[1] * v[2] - 3.0e7 * v[1] * v[1], 3.0e7 * v[1] * v[1]] };
    assert(f2.kind == scientific::F_NONE && t2.data.len == 67);
    assert(mathshim_abs(y2.data[3 * 66] - 0.715827074754968) < 1.0e-6);
    let e3 = [1.0, 0.0, -2.000925847183714, 0.006655492040691016, -1.9975905328886725, 0.006679958122998222, -1.9929053915140043, 0.006706216921244254, -1.9848305494473333, 0.006752021194119795];
    let t3, y3, f3 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1e-4, 1e-7, "Radau", 1.0e308, [0.0, 0.5, 1.0, 1.7, 2.9]) { t, v => [v[1], 100.0 * (1.0 - v[0] * v[0]) * v[1] - v[0]] };
    assert(f3.kind == scientific::F_NONE && all_close(y3, e3, 1.0e-8));
    let et, ey, eet, eei, eey, ef = scientific::solve_ivp_events(0.0, 3.0, [1.0, 0.0], 1, [true], [0.0], 1e-6, 1e-9, "Radau") { t, v => [v[1], 100.0 * (1.0 - v[0] * v[0]) * v[1] - v[0], v[0] - 0.5] };
    assert(ef.kind == scientific::F_NONE && et.data.len == 89);
    assert(mathshim_abs(eet.data[0] - 0.41542002) < 1.0e-7 && mathshim_abs(eey.data[1] + 21.22437489) < 1.0e-5);
}

// The argument first_step sets the first step, as SciPy does.
fn test_first_step() {
    let t1, y1, f1 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "RK45", 1.0e308, [], 0.01) { t, v => [v[1], -v[0]] };
    assert(f1.kind == scientific::F_NONE && t1.data.len == 9);
    assert(mathshim_abs(t1.data[1] - 0.01) < 1.0e-15 && mathshim_abs(t1.data[3] - 0.7582343760009075) < 1.0e-9);
    assert(mathshim_abs(y1.data[16] - 0.2847042338359484) < 1.0e-9);
    let t2, y2, f2 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "DOP853", 1.0e308, [], 0.05) { t, v => [v[1], -v[0]] };
    assert(f2.kind == scientific::F_NONE && t2.data.len == 5);
    assert(mathshim_abs(t2.data[3] - 3.0469539039081424) < 1.0e-8 && mathshim_abs(y2.data[8] - 0.28341575655185514) < 1.0e-8);
    // A first step below zero or beyond the span is a fault.
    let t3, y3, f3 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "RK45", 1.0e308, [], -1.0) { t, v => [v[1], -v[0]] };
    assert(f3.kind == scientific::F_VALUE);
    let t4, y4, f4 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "RK45", 1.0e308, [], 6.0) { t, v => [v[1], -v[0]] };
    assert(f4.kind == scientific::F_VALUE);
}

fn test_ivp_t_eval() {
    let e1 = [1.0, 0.0, 0.9553364803050641, -0.29552023491137414, 0.540302079728339, -0.8414711243630892, 0.020794742986437575, -0.9997835787518795, -0.9899921548382596, -0.1411198096921278];
    let t1, y1, f1 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1.0e-6, 1.0e-9, "RK45", 1.0e308, [0.0, 0.3, 1.0, 1.55, 3.0]) { t, v => [v[1], -v[0]] };
    assert(f1.kind == scientific::F_NONE && t1.data.len == 5 && y1.data.len == 10);
    var i: i64 = 0;
    while i < 10 { assert(mathshim_abs(y1.data[i] - e1[i]) < 1.0e-8); i += 1; }
    let e2 = [0.8775627709376573, -0.479416876047604, 0.07071226557404034, -0.9973594812519597, -0.8009810027734747, -0.5983202949277295];
    let t2, y2, f2 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1.0e-4, 1.0e-6, "RK23", 1.0e308, [0.5, 1.5, 2.5]) { t, v => [v[1], -v[0]] };
    assert(f2.kind == scientific::F_NONE && y2.data.len == 6);
    i = 0;
    while i < 6 { assert(mathshim_abs(y2.data[i] - e2[i]) < 1.0e-8); i += 1; }
    let e3 = [1.0, 0.0, 0.540302079728339, 0.841471124363089, -0.66627600604739, 0.7457049606986244, -0.9899921548382591, 0.14111980969212792];
    let t3, y3, f3 = scientific::solve_ivp(3.0, 0.0, [1.0, 0.0], 1.0e-6, 1.0e-9, "RK45", 1.0e308, [3.0, 2.0, 0.7, 0.0]) { t, v => [v[1], -v[0]] };
    assert(f3.kind == scientific::F_NONE && y3.data.len == 8);
    i = 0;
    while i < 8 { assert(mathshim_abs(y3.data[i] - e3[i]) < 1.0e-8); i += 1; }
    let t4, y4, f4 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1.0e-6, 1.0e-9, "RK45", 1.0e308, [4.0]) { t, v => [v[1], -v[0]] };
    assert(f4.kind == scientific::F_VALUE);
    let t5, y5, f5 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1.0e-6, 1.0e-9, "RK45", 1.0e308, [2.0, 1.0]) { t, v => [v[1], -v[0]] };
    assert(f5.kind == scientific::F_VALUE);
}

// cov_close tells whether two values of a covariance matrix agree. The
// comparison uses a scale of 1e-3, because the small values of the
// matrix lose their digits in the reference library.
fn cov_close(a: f64, b: f64) -> bool {
    var d = a - b;
    if d < 0.0 { d = -d; }
    return d <= 0.001 * 0.001;
}

// loose tells whether two numbers agree to a relative 1e-5.
fn loose(a: f64, b: f64) -> bool {
    var d = a - b;
    if d < 0.0 { d = -d; }
    var s = b;
    if s < 0.0 { s = -s; }
    return d <= 0.00001 * (s + 0.00001);
}

// curve_fit is pinned against SciPy.
fn test_curve_fit() {
    let x = [0.0, 0.36363636363636365, 0.7272727272727273, 1.0909090909090908, 1.4545454545454546, 1.8181818181818183, 2.1818181818181817, 2.5454545454545454, 2.909090909090909, 3.272727272727273, 3.6363636363636367, 4.0];
    let y = [3.02, 2.0282500972972257, 1.4812573462907253, 1.1453847417432672, 0.8573363330894807, 0.7651937511001847, 0.6365962742142248, 0.6113734634230907, 0.5169530833077661, 0.5454988590422815, 0.5521264402226302, 0.49379141105190194];
    // Bounds on the parameters.
    let bx = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let by = [1.0, 2.7, 7.4, 20.1, 54.6, 148.4];
    let pb, cb, fb = scientific::curve_fit(bx, by, [1.0, 0.5], 200, [], [0.0, 0.0], [1.5, 0.9]) { t, q => q[0] * mathshim_exp(q[1] * t) };
    assert(fb.kind == scientific::F_NONE && mathshim_abs(pb.data[0] - 1.5) < 1.0e-6 && mathshim_abs(pb.data[1] - 0.9) < 1.0e-6);
    assert(mathshim_abs(cb.data[0] - 0.47668620363771635) < 1.0e-2 && mathshim_abs(cb.data[3] - 0.009094064870893112) < 1.0e-3);
    let pc, cc, fc = scientific::curve_fit(bx, by, [1.0, 0.5], 200, [], [0.5, 0.0], [1.0e308 * 10.0, 0.95]) { t, q => q[0] * mathshim_exp(q[1] * t) };
    assert(fc.kind == scientific::F_NONE && mathshim_abs(pc.data[0] - 1.273035098317418) < 1.0e-5 && mathshim_abs(pc.data[1] - 0.95) < 1.0e-8);
    let pd, cd, fd = scientific::curve_fit(bx, by, [1.0, 0.5], 200, [], [2.0, 0.0], [3.0, 1.0]) { t, q => q[0] * mathshim_exp(q[1] * t) };
    assert(fd.kind == scientific::F_VALUE);
    let pe, ce, fe = scientific::curve_fit(bx, by, [1.0, 0.5], 200, [], [0.0], [3.0, 1.0]) { t, q => q[0] * mathshim_exp(q[1] * t) };
    assert(fe.kind == scientific::F_SHAPE);
    let p, c, f = scientific::curve_fit(x, y, [1.0, 1.0, 0.0]) { t, q => q[0] * mathshim_exp(-q[1] * t) + q[2] };
    assert(f.kind == scientific::F_NONE);
    assert(loose(p.data[0], 2.504842778143317) && loose(p.data[1], 1.2966838932208153) && loose(p.data[2], 0.501188851174762));
    assert(loose(c.data[0], 0.0008144738293533199) && loose(c.data[4], 0.001227646248841025) && loose(c.data[8], 0.00022564752813311048));
    assert(loose(c.data[1], 0.0001032144836216253) && loose(c.data[2], -0.00014630932777603877));
    let y2 = [-1.98, -0.9390909090909092, 0.19181818181818167, 1.3127272727272725, 2.3436363636363633, 3.484545454545455, 4.535454545454545, 5.656363636363636, 6.687272727272727, 7.828181818181818, 8.93909090909091, 9.98];
    let p2, c2, f2 = scientific::curve_fit(x, y2, [1.0, 0.0]) { t, q => q[0] * t + q[1] };
    assert(f2.kind == scientific::F_NONE);
    assert(loose(p2.data[0], 2.997884615327012) && loose(p2.data[1], -1.992435897275306));
    assert(loose(c2.data[0], 4.00973865517021e-05) && loose(c2.data[1], -8.019477395462219e-05) && loose(c2.data[3], 0.00022357331117456666));
    // The values of sigma scale the residuals.
    let sig = [0.05, 0.06, 0.05, 0.07, 0.05, 0.06, 0.05, 0.07, 0.05, 0.06, 0.05, 0.07];
    let ps, cs, fs = scientific::curve_fit(x, y, [1.0, 1.0, 0.0], 200, sig) { t, q => q[0] * mathshim_exp(-q[1] * t) + q[2] };
    assert(fs.kind == scientific::F_NONE);
    assert(loose(ps.data[0], 2.5105340700115826) && loose(ps.data[1], 1.3067584455977996) && loose(ps.data[2], 0.5014645542629256));
    // The covariance holds a scale. The comparison uses it, because the
    // small values of the matrix lose their digits in SciPy.
    assert(cov_close(cs.data[0], 0.0006939681062620468) && cov_close(cs.data[4], 0.001161032323389619) && cov_close(cs.data[8], 0.00022211805651708264));
    assert(cov_close(cs.data[1], 1.4921191905565632e-05) && cov_close(cs.data[2], -0.000159433318239479));
    // One value of sigma serves every point.
    let pv, cv, fv = scientific::curve_fit(x, y, [1.0, 1.0, 0.0], 200, [0.1]) { t, q => q[0] * mathshim_exp(-q[1] * t) + q[2] };
    assert(fv.kind == scientific::F_NONE);
    assert(loose(pv.data[0], 2.5048427779253335) && loose(pv.data[1], 1.2966838921487769) && loose(pv.data[2], 0.5011888506950971));
    assert(cov_close(cv.data[0], 0.0008144738275679136) && cov_close(cv.data[4], 0.0012276462627711755) && cov_close(cv.data[8], 0.00022564753169769866));
    // A matrix of sigma holds the covariance.
    let mat: f64[>..] = [];
    var mr: i64 = 0;
    while mr < 12 {
        var mc: i64 = 0;
        while mc < 12 {
            if mr == mc { mat.push(sig[mr] * sig[mr]); } else { mat.push(0.0); }
            mc += 1;
        }
        mr += 1;
    }
    let pm, cm, fm = scientific::curve_fit(x, y, [1.0, 1.0, 0.0], 200, mat) { t, q => q[0] * mathshim_exp(-q[1] * t) + q[2] };
    assert(fm.kind == scientific::F_NONE);
    assert(loose(pm.data[0], 2.510534070187927) && loose(pm.data[1], 1.3067584452752734) && loose(pm.data[2], 0.5014645540838876));
    assert(cov_close(cm.data[0], 0.0006939680930908152) && cov_close(cm.data[4], 0.00116103238092085) && cov_close(cm.data[8], 0.00022211806304844883));
    // A wrong count of sigma, a value of zero and a matrix that is not
    // positive definite are faults.
    let b7, b8, g7 = scientific::curve_fit(x, y, [1.0, 1.0, 0.0], 200, [0.1, 0.2]) { t, q => q[0] * mathshim_exp(-q[1] * t) + q[2] };
    assert(g7.kind == scientific::F_SHAPE);
    let b9, b10, g8 = scientific::curve_fit(x, y, [1.0, 1.0, 0.0], 200, [0.0]) { t, q => q[0] * mathshim_exp(-q[1] * t) + q[2] };
    assert(g8.kind == scientific::F_VALUE);
    let bad: f64[>..] = [];
    var br: i64 = 0;
    while br < 12 {
        var bc: i64 = 0;
        while bc < 12 {
            if br == bc { bad.push(-1.0); } else { bad.push(0.0); }
            bc += 1;
        }
        br += 1;
    }
    let b11, b12, g9 = scientific::curve_fit(x, y, [1.0, 1.0, 0.0], 200, bad) { t, q => q[0] * mathshim_exp(-q[1] * t) + q[2] };
    assert(g9.kind == scientific::F_VALUE);
    // Too few points and data of unequal length are faults.
    let b1, b2, g1 = scientific::curve_fit([1.0, 2.0], [1.0, 2.0], [1.0, 0.0]) { t, q => q[0] * t + q[1] };
    assert(g1.kind == scientific::F_VALUE);
    let b3, b4, g2 = scientific::curve_fit([1.0, 2.0, 3.0], [1.0, 2.0], [1.0]) { t, q => q[0] * t };
    assert(g2.kind == scientific::F_SHAPE);
    return;
}

// interpn is pinned against SciPy.
fn test_interpn() {
    let g = [0.0, 1.0, 2.5, 4.0, -1.0, 0.5, 2.0, 0.0, 2.0];
    let v0 = av([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4546487134128409, 2.454648713412841, 0.7384602626041288, 2.7384602626041286, -0.35017548837401463, 1.6498245116259853, 0.32335587945721733, 5.3233558794572176, 0.5252087174427745, 5.525208717442775, -0.24905228953044706, 4.7509477104695526, -0.4089021333016356, 7.591097866698364, -0.6641566726773585, 7.3358433273226415, 0.3149409643133779, 8.314940964313378]);
    let v, vf = scientific::reshape(v0, [4, 3, 2]);
    assert(vf.kind == scientific::F_NONE);
    let pts = mx(6, 3, [0.3, -0.2, 0.5, 2.0, 1.0, 1.9, 4.0, 2.0, 2.0, 1.0, 0.5, 0.0, 3.3, 1.7, 1.1, 0.0, -1.0, 0.0]);
    let r1, f1 = scientific::interpn(g, [4, 3, 2], v, pts);
    assert(f1.kind == scientific::F_NONE && all_near(r1, [0.33180446189445834, 4.103275036393827, 8.314940964313378, 0.7384602626041288, 3.649571391891415, 0.0]));
    let r2, f2 = scientific::interpn(g, [4, 3, 2], v, pts, "nearest");
    assert(f2.kind == scientific::F_NONE && all_near(r2, [0.0, 5.525208717442775, 8.314940964313378, 0.7384602626041288, 8.314940964313378, 0.0]));
    let q = mx(8, 3, [0.3, -0.2, 0.5, 2.0, 1.0, 1.9, 4.0, 2.0, 2.0, 1.0, 0.5, 0.0, 3.3, 1.7, 1.1, 0.0, -1.0, 0.0, 5.0, 0.0, 1.0, 1.0, 3.0, 1.0]);
    let r3, f3 = scientific::interpn(g, [4, 3, 2], v, q, "linear", false, -7.0);
    assert(f3.kind == scientific::F_NONE && all_near(r3, [0.33180446189445834, 4.103275036393827, 8.314940964313378, 0.7384602626041288, 3.649571391891415, 0.0, -7.0, -7.0]));
    let b1, h1 = scientific::interpn(g, [4, 3, 2], v, q);
    assert(h1.kind == scientific::F_RANGE);
    let b2, h2 = scientific::interpn(g, [4, 3, 2], v, pts, "cubic");
    assert(h2.kind == scientific::F_VALUE);
    let b3, h3 = scientific::interpn(g, [4, 3, 3], v, pts);
    assert(h3.kind == scientific::F_SHAPE);
    let b4, h4 = scientific::interpn([0.0, 1.0, 1.0, 2.0, 3.0, 4.0], [3, 3], mx(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]), mx(1, 2, [0.5, 1.5]));
    assert(h4.kind == scientific::F_VALUE);
    return;
}

// cheby1_sos is pinned against SciPy.
fn test_cheby1() {
    let a, fa = scientific::cheby1_sos(3, 1.0, [0.3], "lowpass");
    assert(fa.kind == scientific::F_NONE && all_near(a, [0.034384966351816024, 0.06876993270363205, 0.034384966351816024, 1.0, -0.5977088859557689, 0.0, 1.0, 1.0, 0.0, 1.0, -0.9826960523706986, 0.6664788186594006]));
    let b, fb = scientific::cheby1_sos(4, 0.5, [0.25], "lowpass");
    assert(fb.kind == scientific::F_NONE && all_near(b, [0.005619419204229541, 0.011238838408459082, 0.005619419204229541, 1.0, -1.329949932426935, 0.5031986921167156, 1.0, 2.0, 1.0, 1.0, -1.231461224536315, 0.7811808980112693]));
    let c, fc = scientific::cheby1_sos(4, 2.0, [0.4], "highpass");
    assert(fc.kind == scientific::F_NONE && all_near(c, [0.0909132735975814, -0.1818265471951628, 0.0909132735975814, 1.0, 0.5482374733732417, 0.3414066130550003, 1.0, -2.0, 1.0, 1.0, -0.4982271986803331, 0.8105470523665526]));
    let d, fd = scientific::cheby1_sos(2, 1.0, [0.2, 0.5], "bandpass");
    assert(fd.kind == scientific::F_NONE && all_near(d, [0.13822540868747046, 0.2764508173749409, 0.13822540868747046, 1.0, -0.05304216112661112, 0.557304021729915, 1.0, -2.0, 1.0, 1.0, -1.3601282270267474, 0.7067397185454787]));
    let e, fe = scientific::cheby1_sos(3, 3.0, [0.3, 0.6], "bandstop");
    assert(fe.kind == scientific::F_NONE && all_near(e, [0.24788111204717675, -0.08704122383840084, 0.24788111204717675, 1.0, -0.1297511163775456, -0.2609742924400598, 1.0, -0.3511410091698925, 1.0, 1.0, 0.6356323433677218, 0.8595451504415792, 1.0, -0.3511410091698925, 1.0, 1.0, -1.1494556109789573, 0.8803918167017183]));
    let g1, h1 = scientific::cheby1_sos(3, 0.0, [0.3], "lowpass");
    assert(h1.kind == scientific::F_VALUE);
    let g2, h2 = scientific::cheby1_sos(3, 1.0, [1.2], "lowpass");
    assert(h2.kind == scientific::F_RANGE);
    return;
}

// hfft2, ihfft2, hfftn and ihfftn are pinned against SciPy.
fn test_hfftn() {
    let zr = mx(3, 4, [1.5, -2.0, 0.5, 3.0, 0.25, -1.0, 2.0, 0.75, -0.5, 1.25, 2.5, -1.5]);
    let zi = mx(3, 4, [-1.5, 2.5, 1.25, -0.5, 0.75, 2.0, -1.0, 0.25, 3.0, 0.5, -2.0, 1.5]);
    let h, f1 = scientific::hfft2(zr, zi);
    assert(f1.kind == scientific::F_NONE && h.shape[0] == 3 && h.shape[1] == 6);
    assert(all_near(h, [10.0, -2.1208348754011492, 11.94134295108992, 12.499999999999996, -11.44134295108992, -13.379165124598849, -1.4509618943233418, 8.495190528383286, 3.705771365940053, -2.7320508075688767, 3.651923788646684, -13.611215932167726, -4.049038105676658, 1.1112159321677293, 8.848076211353316, 0.7320508075688781, 19.294228634059948, -4.49519052838329]));
    let h2, f2 = scientific::hfft2(zr, zi, 5);
    assert(f2.kind == scientific::F_NONE && h2.shape[1] == 5);
    assert(all_near(h2, [7.75, -0.46841264413391226, 16.37827975401956, -2.0348209058959785, -15.375046203989669, -3.7434301395927934, 10.422144364009288, -1.827769576130672, 2.7608714046501683, -9.229601845510924, -8.506569860407206, 7.085250501121263, 2.2685996499087713, 12.725349504696574, 4.295155997255532]));
    let a = mx(3, 4, [1.5, -2.0, 0.5, 3.0, 0.25, -1.0, 2.0, 0.75, -0.5, 1.25, 2.5, -1.5]);
    let ir, ii, f3 = scientific::ihfft2(a);
    assert(f3.kind == scientific::F_NONE && ir.shape[1] == 3);
    assert(all_near(ir, [0.5625, -0.3125, 0.47916666666666663, 0.09375, 0.6060095264191645, -0.11458333333333333, 0.09375, -0.0435095264191645, -0.11458333333333333]));
    assert(all_near(ii, [0.0, -0.33333333333333326, 0.0, 0.01804219591217581, -0.3681223537724543, 0.018042195912175787, -0.01804219591217581, -0.5485443128942123, -0.018042195912175787]));
    // The three dimensional forms.
    let z3r0 = av([1.5, -2.0, 0.5, 3.0, 0.25, -1.0, 2.0, 0.75, -0.5, 1.25, 2.5, -1.5]);
    let z3i0 = av([-1.5, 2.5, 1.25, -0.5, 0.75, 2.0, -1.0, 0.25, 3.0, 0.5, -2.0, 1.5]);
    let z3r, g1 = scientific::reshape(z3r0, [2, 3, 2]);
    let z3i, g2 = scientific::reshape(z3i0, [2, 3, 2]);
    assert(g1.kind == scientific::F_NONE && g2.kind == scientific::F_NONE);
    let h3, f4 = scientific::hfftn(z3r, z3i);
    assert(f4.kind == scientific::F_NONE);
    assert(all_near(h3, [6.75, 5.75, 1.7320508075688776, 12.044228634059948, -1.7320508075688772, -3.5442286340599476, -2.25, -1.25, -8.946152422706632, 1.4019237886466844, 1.446152422706632, 6.598076211353316]));
    let i3r, i3i, f5 = scientific::ihfftn(z3r);
    assert(f5.kind == scientific::F_NONE && i3r.shape[2] == 2);
    assert(all_near(i3i, [0.0, 0.0, 0.28867513459481287, -0.6856034446626805, -0.28867513459481287, 0.6856034446626805, 0.0, 0.0, 0.32475952641916445, 0.14433756729740643, -0.32475952641916445, -0.14433756729740643]));
    let b1, k1 = scientific::hfft2(z3r, z3i);
    assert(k1.kind == scientific::F_SHAPE);
    let b2, b3, k2 = scientific::ihfft2(z3r);
    assert(k2.kind == scientific::F_SHAPE);
    return;
}

// csr_kron, csr_hstack and csr_vstack are pinned against SciPy.
fn test_sparse_join() {
    let a, fa = scientific::csr_from_dense(mx(2, 3, [1.0, 0.0, 2.0, 0.0, 0.0, 3.0]));
    let b, fb = scientific::csr_from_dense(mx(3, 2, [0.0, 4.0, 5.0, 6.0, 0.0, 7.0]));
    let bt, fbt = scientific::csr_from_dense(mx(2, 3, [0.0, 5.0, 0.0, 4.0, 6.0, 7.0]));
    let c, fc = scientific::csr_from_dense(mx(2, 3, [2.0, 0.0, 0.0, 0.0, -1.0, 1.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE && fbt.kind == scientific::F_NONE && fc.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kron(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 6 && k.cols == 6 && k.nnz == 12);
    let kd, fkd = scientific::csr_to_dense(k);
    assert(fkd.kind == scientific::F_NONE && all_near(kd, [0.0, 4.0, 0.0, 0.0, 0.0, 8.0, 5.0, 6.0, 0.0, 0.0, 10.0, 12.0, 0.0, 7.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0, 15.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 21.0]));
    let h, fh = scientific::csr_hstack(a, bt);
    assert(fh.kind == scientific::F_NONE && h.cols == 6);
    let hd, fhd = scientific::csr_to_dense(h);
    assert(fhd.kind == scientific::F_NONE && all_near(hd, [1.0, 0.0, 2.0, 0.0, 5.0, 0.0, 0.0, 0.0, 3.0, 4.0, 6.0, 7.0]));
    let v, fv = scientific::csr_vstack(a, c);
    assert(fv.kind == scientific::F_NONE && v.rows == 4);
    let vd, fvd = scientific::csr_to_dense(v);
    assert(fvd.kind == scientific::F_NONE && all_near(vd, [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 0.0, -1.0, 1.0]));
    let e1, g1 = scientific::csr_hstack(a, b);
    assert(g1.kind == scientific::F_SHAPE);
    let e2, g2 = scientific::csr_vstack(a, b);
    assert(g2.kind == scientific::F_SHAPE);
    return;
}

// bicgstab and gmres are pinned against the dense solution of NumPy.
fn test_krylov() {
    let a, fa = scientific::csr_from_dense(mx(8, 8, [4.0, -1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -2.0, 4.1, -0.95, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -2.0, 4.2, -0.9, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -2.0, 4.3, -0.85, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -2.0, 4.4, -0.8, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -2.0, 4.5, -0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 4.6, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 4.7]));
    assert(fa.kind == scientific::F_NONE);
    let b = [1.0, -2.0, 3.0, 0.5, -1.0, 2.0, 1.5, -0.5];
    let want = [0.1387735113564285, -0.28519591834488306, 0.5824268411536323, 0.31942007245880555, 0.00031157404565690866, 0.541786991355806, 0.5832244173464183, 0.14179762440273116];
    let x1, c1, f1 = scientific::bicgstab(a, b, 1.0e-12);
    assert(f1.kind == scientific::F_NONE && c1 == 0 && all_near(x1, want));
    let x2, c2, f2 = scientific::gmres(a, b, 1.0e-12);
    assert(f2.kind == scientific::F_NONE && c2 == 0 && all_near(x2, want));
    // A restart shorter than the side still reaches the answer.
    let x3, c3, f3 = scientific::gmres(a, b, 1.0e-12, 3);
    assert(f3.kind == scientific::F_NONE && c3 == 0 && all_near(x3, want));
    // The right side must fit the matrix.
    let e1, k1, g1 = scientific::gmres(a, [1.0, 2.0]);
    assert(g1.kind == scientific::F_SHAPE);
    let e2, k2, g2 = scientific::bicgstab(a, [1.0, 2.0]);
    assert(g2.kind == scientific::F_SHAPE);
    let e3, k3, g3 = scientific::gmres(a, b, 1.0e-5, 0);
    assert(g3.kind == scientific::F_VALUE);
    return;
}

// odeint is pinned against SciPy at tight tolerances.
fn test_odeint() {
    let ts = [0.0, 0.5, 1.0, 2.0, 3.5];
    let r1, f1 = scientific::odeint([1.0, 0.0], ts, 1.0e-11, 1.0e-11) { y, t => [y[1], -y[0] - 0.2 * y[1]] };
    assert(f1.kind == scientific::F_NONE && r1.shape[0] == 5 && r1.shape[1] == 2);
    assert(all_loose(r1, [1.0, 0.0, 0.8815464028676221, -0.45623696587095147, 0.5689718911319457, -0.7627576785286944, -0.2580702634045758, -0.7516155021919233, -0.6878213068949068, 0.2367646744259087]));
    let r2, f2 = scientific::odeint([0.5], ts, 1.0e-11, 1.0e-11) { y, t => [-2.0 * y[0] + mathshim_sin(t)] };
    assert(f2.kind == scientific::F_NONE);
    assert(all_loose(r2, [0.5, 0.27376931184737185, 0.3232626309846345, 0.4597692852088899, 0.047616363747107364]));
    let r3, f3 = scientific::odeint([1.0], [2.0, 1.0, 0.0], 1.0e-11, 1.0e-11) { y, t => [-y[0]] };
    assert(f3.kind == scientific::F_NONE && all_loose(r3, [1.0, 2.718281828459045, 7.38905609893065]));
    let b1, g1 = scientific::odeint([1.0], [0.0], 1.0e-8, 1.0e-8) { y, t => [-y[0]] };
    assert(g1.kind == scientific::F_SHAPE);
    let b2, g2 = scientific::odeint([1.0], [0.0, 1.0, 0.5], 1.0e-8, 1.0e-8) { y, t => [-y[0]] };
    assert(g2.kind == scientific::F_VALUE);
    return;
}

fn all_loose(a: scientific::Arr&, v: f64[:]) -> bool {
    if a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len { if !loose(a.data[i], v[i]) { return false; } i += 1; }
    return true;
}

// root_scalar picks the bracket method by name.
fn test_root_scalar() {
    let r1, f1 = scientific::root_scalar(0.0, 2.0) { x => x * x - 2.0 };
    assert(f1.kind == scientific::F_NONE && near(r1, 1.4142135623730951));
    let r2, f2 = scientific::root_scalar(0.0, 2.0, "brenth") { x => x * x - 2.0 };
    assert(f2.kind == scientific::F_NONE && near(r2, 1.4142135623730951));
    let r3, f3 = scientific::root_scalar(0.0, 2.0, "bisect") { x => x * x - 2.0 };
    assert(f3.kind == scientific::F_NONE && near(r3, 1.4142135623730951));
    let r4, f4 = scientific::root_scalar(0.0, 2.0, "ridder") { x => x * x - 2.0 };
    assert(f4.kind == scientific::F_NONE && near(r4, 1.4142135623730951));
    let b1, g1 = scientific::root_scalar(0.0, 2.0, "secant") { x => x * x - 2.0 };
    assert(g1.kind == scientific::F_VALUE);
    let b2, g2 = scientific::root_scalar(3.0, 4.0) { x => x * x - 2.0 };
    assert(g2.kind != scientific::F_NONE);
    return;
}

// cheby2_sos is pinned against SciPy.
fn test_cheby2() {
    let r0, f0 = scientific::cheby2_sos(3, 30.0, [0.3], "lowpass");
    assert(f0.kind == scientific::F_NONE && all_near(r0, [0.0397692799675111, 0.0397692799675111, 0.0, 1.0, -0.5711000156523628, 0.0, 1.0, -0.971426219672632, 0.9999999999999998, 1.0, -1.4598918274880655, 0.6506385755873847]));
    let r1, f1 = scientific::cheby2_sos(4, 40.0, [0.25], "lowpass");
    assert(f1.kind == scientific::F_NONE && all_near(r1, [0.014946994058911952, 0.0023618813591870865, 0.01494699405891195, 1.0, -1.286192073852133, 0.4319468029476599, 1.0, -1.3305297855692517, 1.0000000000000004, 1.0, -1.6126739149245188, 0.7608292714568752]));
    let r2, f2 = scientific::cheby2_sos(4, 25.0, [0.4], "highpass");
    assert(f2.kind == scientific::F_NONE && all_near(r2, [0.20438409640140692, -0.3501043833528033, 0.20438409640140692, 1.0, -0.25303233631047123, 0.08647805683020464, 1.0, -0.7575554262491747, 0.9999999999999997, 1.0, -0.0035478386759487837, 0.558689528742404]));
    let r3, f3 = scientific::cheby2_sos(2, 30.0, [0.2, 0.5], "bandpass");
    assert(f3.kind == scientific::F_NONE && all_near(r3, [0.04021131306265627, 0.01583844789390953, 0.04021131306265626, 1.0, -0.779000714965027, 0.8294533123017738, 1.0, -1.7354143450159694, 1.0, 1.0, -1.0789880679024455, 0.8457146813671075]));
    let r4, f4 = scientific::cheby2_sos(3, 35.0, [0.3, 0.6], "bandstop");
    assert(f4.kind == scientific::F_NONE && all_near(r4, [0.146084034035982, -0.05129609513500364, 0.146084034035982, 1.0, -0.1613013579732175, -0.08127302843641923, 1.0, 0.5030145806808963, 1.0, 1.0, 1.085659506469457, 0.570049073882224, 1.0, -1.0908398786270708, 1.0, 1.0, -1.3826014621234592, 0.6551034176147763]));
    let e1, g1 = scientific::cheby2_sos(3, -1.0, [0.3], "lowpass");
    assert(g1.kind == scientific::F_VALUE);
    let e2, g2 = scientific::cheby2_sos(3, 30.0, [0.0], "lowpass");
    assert(g2.kind == scientific::F_RANGE);
    return;
}

// The rules of the count of bins and the weighted covariance are pinned
// against NumPy.
fn test_bins_and_weights() {
    let x = av([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 3.0, 4.0, 5.0, 6.0]);
    let e1, f1 = scientific::histogram_edges_rule(x, "auto");
    assert(f1.kind == scientific::F_NONE && all_near(e1, [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]));
    let e2, f2 = scientific::histogram_edges_rule(x, "fd");
    assert(f2.kind == scientific::F_NONE && all_near(e2, [1.0, 3.25, 5.5, 7.75, 10.0]));
    let e3, f3 = scientific::histogram_edges_rule(x, "sturges");
    assert(f3.kind == scientific::F_NONE && all_near(e3, [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]));
    let e4, f4 = scientific::histogram_edges_rule(x, "doane");
    assert(f4.kind == scientific::F_NONE && all_near(e4, [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]));
    let e5, f5 = scientific::histogram_edges_rule(x, "scott");
    assert(f5.kind == scientific::F_NONE && all_near(e5, [1.0, 4.0, 7.0, 10.0]));
    let e6, f6 = scientific::histogram_edges_rule(x, "rice");
    assert(f6.kind == scientific::F_NONE && all_near(e6, [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]));
    let e7, f7 = scientific::histogram_edges_rule(x, "sqrt");
    assert(f7.kind == scientific::F_NONE && all_near(e7, [1.0, 2.8, 4.6, 6.4, 8.2, 10.0]));
    // The rule of Stone chooses the count of bins by an estimate.
    let e8, f8 = scientific::histogram_edges_rule(x, "stone");
    assert(f8.kind == scientific::F_NONE && e8.data.len == 101);
    assert(near(e8.data[0], 1.0) && near(e8.data[1], 1.09) && near(e8.data[100], 10.0));
    // The counts of a histogram under a rule.
    let c1, e9, f9 = scientific::histogram_rule(x, "scott");
    assert(f9.kind == scientific::F_NONE && c1.data.len == 3 && all_near(c1, [5.0, 9.0, 6.0]));
    let b1, g1 = scientific::histogram_edges_rule(x, "unknown");
    assert(g1.kind == scientific::F_VALUE);
    // The weighted covariance.
    let m = mx(2, 5, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let fw = [1.0, 2.0, 3.0, 4.0, 5.0];
    let aw = [1.0, 2.0, 3.0, 4.0, 5.0];
    let c2, g2 = scientific::cov_weights(m, fw);
    assert(g2.kind == scientific::F_NONE && all_near(c2, [1.6666666666666667, 1.6666666666666667, 1.6666666666666667, 1.6666666666666665]));
    let c3, g3 = scientific::cov_weights(m, [], aw);
    assert(g3.kind == scientific::F_NONE && all_near(c3, [2.058823529411765, 2.058823529411765, 2.058823529411765, 2.0588235294117645]));
    let c4, g4 = scientific::cov_weights(m, fw, aw);
    assert(g4.kind == scientific::F_NONE && all_near(c4, [1.15, 1.15, 1.15, 1.15]));
    let c5, g5 = scientific::cov_weights(m, fw, aw, true, 0);
    assert(g5.kind == scientific::F_NONE && all_near(c5, [1.0644628099173554, 1.0644628099173554, 1.0644628099173554, 1.0644628099173552]));
    // A weight that does not fit the observations is a fault.
    let b2, g6 = scientific::cov_weights(m, [1.0, 2.0]);
    assert(g6.kind == scientific::F_SHAPE);
    let b3, g7 = scientific::cov_weights(m, [1.0, 2.0, 3.0, 4.0, 5.5]);
    assert(g7.kind == scientific::F_VALUE);
    return;
}

// cumulative_simpson is pinned against SciPy.
fn test_cumulative_simpson() {
    let x = [0.0, 1.0, 2.5, 4.0, 6.0, 9.5];
    let y = [1.0, 2.0, 0.5, 3.0, 2.0, 4.0];
    let r1, f1 = scientific::cumulative_simpson(y, x);
    assert(f1.kind == scientific::F_NONE && r1.data.len == 5);
    assert(all_near(r1, [1.6333333333333333, 3.958333333333333, 6.931547619047619, 12.756944444444445, 21.86489898989899]));
    let r2, f2 = scientific::cumulative_simpson_dx(y, 2.0);
    assert(f2.kind == scientific::F_NONE && all_near(r2, [3.4166666666666665, 6.333333333333333, 10.416666666666666, 16.0, 21.5]));
    // A signal of two values falls back to the trapezoid.
    let r3, f3 = scientific::cumulative_simpson([1.0, 3.0], [0.0, 2.0]);
    assert(f3.kind == scientific::F_NONE && all_near(r3, [4.0]));
    // The places must rise.
    let b1, g1 = scientific::cumulative_simpson(y, [0.0, 1.0, 1.0, 4.0, 6.0, 9.5]);
    assert(g1.kind == scientific::F_VALUE);
    let b2, g2 = scientific::cumulative_simpson(y, [0.0, 1.0]);
    assert(g2.kind == scientific::F_SHAPE);
    let b3, g3 = scientific::cumulative_simpson_dx(y, 0.0);
    assert(g3.kind == scientific::F_VALUE);
    return;
}

fn tq_zero(x: f64) -> f64 { return 0.0; }
fn tq_one(x: f64) -> f64 { return 1.0; }
fn tq_half_x(x: f64) -> f64 { return 0.5 * x; }
fn tq_one_x(x: f64) -> f64 { return 1.0 + x; }
fn tq_m1(x: f64) -> f64 { return -1.0; }
fn tq_p1(x: f64) -> f64 { return 1.0; }
fn tq_zero_xy(x: f64, y: f64) -> f64 { return 0.0; }
fn tq_one_xy(x: f64, y: f64) -> f64 { return 1.0; }
fn tq_half_y(x: f64, y: f64) -> f64 { return 0.5 * y; }
fn tq_y_half(x: f64, y: f64) -> f64 { return y + 0.5; }
fn tq_m1_xy(x: f64, y: f64) -> f64 { return -1.0; }
fn tq_p1_xy(x: f64, y: f64) -> f64 { return 1.0; }

// tplquad is pinned against SciPy.
fn test_tplquad() {
    let v1, e1, f1 = scientific::tplquad(0.0, 1.0, 1.49e-8, 1.49e-8, 50, tq_zero, tq_one, tq_zero_xy, tq_one_xy) { z, y, x => x * y * z };
    assert(f1.kind == scientific::F_NONE && near(v1, 0.12499999999999999));
    let v2, e2, f2 = scientific::tplquad(0.0, 1.0, 1.49e-8, 1.49e-8, 50, tq_half_x, tq_one_x, tq_half_y, tq_y_half) { z, y, x => mathshim_exp(-x - y - z) };
    assert(f2.kind == scientific::F_NONE && near(v2, 0.14202386703691938));
    let v3, e3, f3 = scientific::tplquad(0.0, 1.0, 1.49e-8, 1.49e-8, 50, tq_m1, tq_p1, tq_m1_xy, tq_p1_xy) { z, y, x => 1.0 / (1.0 + x * x + y * y + z * z) };
    assert(f3.kind == scientific::F_NONE && near(v3, 2.143427031150922));
    // A bad tolerance is a fault.
    let v4, e4, f4 = scientific::tplquad(0.0, 1.0, -1.0, 1.49e-8, 50, tq_zero, tq_one, tq_zero_xy, tq_one_xy) { z, y, x => x * y * z };
    assert(f4.kind == scientific::F_VALUE);
    return;
}

// close tells whether a is within tol of b, relative to one plus b.
fn close(a: f64, b: f64, tol: f64) -> bool {
    var d = a - b;
    if d < 0.0 { d = -d; }
    var m = b;
    if m < 0.0 { m = -m; }
    return d <= tol * (1.0 + m);
}

// nquad and quad_vec are pinned against SciPy.
fn test_nquad() {
    let pi = 3.141592653589793;
    let inf = 1.0 / 0.0;
    let v1, e1, f1 = scientific::nquad([0.0, 1.0, 0.0, 2.0]) { p => p[0] * p[1] * p[1] };
    assert(f1.kind == scientific::F_NONE && close(v1, 1.3333333333333335, 1.0e-12) && e1 < 1.0e-8);
    // the first value of the point belongs to the first range
    let v1b, e1b, f1b = scientific::nquad([0.0, 1.0, 0.0, 2.0]) { p => p[0] };
    assert(f1b.kind == scientific::F_NONE && close(v1b, 1.0, 1.0e-12));
    let v1c, e1c, f1c = scientific::nquad([0.0, 1.0, 0.0, 2.0]) { p => p[1] };
    assert(f1c.kind == scientific::F_NONE && close(v1c, 2.0, 1.0e-12));
    let v2, e2, f2 = scientific::nquad([-inf, inf, -inf, inf, -inf, inf]) { p => mathshim_exp(-p[0] * p[0] - p[1] * p[1] - p[2] * p[2]) };
    assert(f2.kind == scientific::F_NONE && close(v2, 5.568327996830833, 1.0e-7));
    let v3, e3, f3 = scientific::nquad([0.0, pi, 0.0, pi / 2.0]) { p => mathshim::sine(p[0] + p[1]) };
    assert(f3.kind == scientific::F_NONE && close(v3, 2.0, 1.0e-12));
    let v4, e4, f4 = scientific::nquad([0.0, 1.0, 0.0, 1.0], 1.49e-8, 1.49e-8, 200) { p => 1.0 / mathshim::root(p[0] * p[1]) };
    assert(f4.kind == scientific::F_NONE && close(v4, 4.0, 1.0e-7));
    // one range gives the value of quad
    let v5, e5, f5 = scientific::nquad([0.0, pi]) { p => mathshim::sine(p[0]) };
    let q5, qe5, qf5 = scientific::quad(0.0, pi) { x => mathshim::sine(x) };
    assert(f5.kind == scientific::F_NONE && v5 == q5 && e5 == qe5);
    // a reversed range negates, and an empty range gives zero
    let v6, e6, f6 = scientific::nquad([2.0, 0.0]) { p => p[0] };
    assert(f6.kind == scientific::F_NONE && close(v6, -2.0, 1.0e-12));
    let v7, e7, f7 = scientific::nquad([1.0, 1.0, 0.0, 5.0]) { p => p[0] };
    assert(f7.kind == scientific::F_NONE && v7 == 0.0);
    // a ray and a finite range
    let v8, e8, f8 = scientific::nquad([0.0, inf, 0.0, 1.0]) { p => mathshim_exp(-p[0]) * p[1] };
    assert(f8.kind == scientific::F_NONE && close(v8, 0.5, 1.0e-7));
    // the faults
    let z1, ze1, zf1 = scientific::nquad(empty_row()) { p => p[0] };
    assert(zf1.kind == scientific::F_SHAPE);
    let z2, ze2, zf2 = scientific::nquad([0.0, 1.0, 2.0]) { p => p[0] };
    assert(zf2.kind == scientific::F_SHAPE);
    let z3, ze3, zf3 = scientific::nquad([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]) { p => p[0] };
    assert(zf3.kind == scientific::F_LIMIT);
    let z4, ze4, zf4 = scientific::nquad([0.0, 1.0 / 0.0 - 1.0 / 0.0]) { p => p[0] };
    assert(zf4.kind == scientific::F_VALUE);
    let z5, ze5, zf5 = scientific::nquad([0.0, 1.0], -1.0) { p => p[0] };
    assert(zf5.kind == scientific::F_VALUE);
    let z6, ze6, zf6 = scientific::nquad([0.0, 1.0], 1.49e-8, 1.49e-8, 0) { p => p[0] };
    assert(zf6.kind == scientific::F_RANGE);
    let z7, ze7, zf7 = scientific::nquad([0.0, 1.0], 1.0e-30, 1.0e-30, 2) { p => mathshim::sine(100.0 * p[0]) };
    assert(zf7.kind == scientific::F_LIMIT);
    // an inner fault stops the call
    let z8, ze8, zf8 = scientific::nquad([0.0, 1.0, 0.0, 1.0], 1.0e-30, 1.0e-30, 2) { p => mathshim::sine(100.0 * p[0]) * p[1] };
    assert(zf8.kind == scientific::F_LIMIT);

    let r1, re1, rf1 = scientific::quad_vec(0.0, pi) { x, o => { o.push(mathshim::sine(x)); o.push(mathshim::cosine(x)); o.push(x * x); } };
    assert(rf1.kind == scientific::F_NONE && r1.data.len == 3);
    assert(close(r1.data[0], 2.0, 1.0e-12) && close(r1.data[1], 0.0, 1.0e-12) && close(r1.data[2], 10.335425560099939, 1.0e-12) && re1 < 1.0e-8);
    let r2, re2, rf2 = scientific::quad_vec(0.0, pi, 1.0e-200, 1.0e-8, 10000, "max") { x, o => { o.push(mathshim::sine(x)); o.push(mathshim::cosine(x)); o.push(x * x); } };
    assert(rf2.kind == scientific::F_NONE && close(r2.data[2], 10.335425560099939, 1.0e-12));
    let r3, re3, rf3 = scientific::quad_vec(0.0, 1.0) { x, o => { o.push(1.0 / mathshim::root(x)); o.push(mathshim::ln(x)); } };
    assert(rf3.kind == scientific::F_NONE && close(r3.data[0], 2.0, 1.0e-7) && close(r3.data[1], -1.0, 1.0e-7));
    let r4, re4, rf4 = scientific::quad_vec(0.0, 10.0, 1.0e-10, 1.0e-10) { x, o => { o.push(mathshim::sine(50.0 * x)); o.push(mathshim::cosine(30.0 * x)); } };
    assert(rf4.kind == scientific::F_NONE && close(r4.data[0], 0.03767699, 1.0e-6) && close(r4.data[1], -0.03332519, 1.0e-6));
    let r5, re5, rf5 = scientific::quad_vec(0.0, inf) { x, o => { o.push(mathshim_exp(-x)); o.push(mathshim_exp(-x)); } };
    assert(rf5.kind == scientific::F_NONE && close(r5.data[0], 1.0, 1.0e-8) && close(r5.data[1], 1.0, 1.0e-8));
    let r5b, re5b, rf5b = scientific::quad_vec(-inf, inf) { x, o => { o.push(mathshim_exp(-x * x)); } };
    assert(rf5b.kind == scientific::F_NONE && close(r5b.data[0], 1.7724538509055159, 1.0e-7));
    // a reversed range negates
    let r6, re6, rf6 = scientific::quad_vec(pi, 0.0) { x, o => { o.push(mathshim::sine(x)); } };
    assert(rf6.kind == scientific::F_NONE && close(r6.data[0], -2.0, 1.0e-12));
    // an empty range gives zeros of the right length
    let r7, re7, rf7 = scientific::quad_vec(1.0, 1.0) { x, o => { o.push(x); o.push(x); } };
    assert(rf7.kind == scientific::F_NONE && r7.data.len == 2 && r7.data[0] == 0.0);
    // the faults
    let g1, ge1, gf1 = scientific::quad_vec(0.0, 1.0) { x, o => { o.push(x); if x > 0.5 { o.push(x); } } };
    assert(gf1.kind == scientific::F_SHAPE && g1.data.len == 0);
    let g2, ge2, gf2 = scientific::quad_vec(0.0, 1.0) { x, o => { x; } };
    assert(gf2.kind == scientific::F_SHAPE);
    let g3, ge3, gf3 = scientific::quad_vec(0.0, 1.0 / 0.0 - 1.0 / 0.0) { x, o => { o.push(x); } };
    assert(gf3.kind == scientific::F_VALUE);
    let g4, ge4, gf4 = scientific::quad_vec(0.0, 1.0, -1.0) { x, o => { o.push(x); } };
    assert(gf4.kind == scientific::F_VALUE);
    let g5, ge5, gf5 = scientific::quad_vec(0.0, 1.0, 1.0e-200, 1.0e-8, 10000, "1") { x, o => { o.push(x); } };
    assert(gf5.kind == scientific::F_VALUE);
    let g6, ge6, gf6 = scientific::quad_vec(0.0, 1.0, 1.0e-200, 1.0e-8, 0) { x, o => { o.push(x); } };
    assert(gf6.kind == scientific::F_RANGE);
    let g7, ge7, gf7 = scientific::quad_vec(0.0, 1.0, 1.0e-200, 1.0e-12, 2) { x, o => { o.push(mathshim::sine(50.0 * x)); } };
    assert(gf7.kind == scientific::F_LIMIT);
    return;
}

// remez is pinned against SciPy for each type and for odd and even
// counts of taps.
fn test_remez() {
    let a, fa = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0]);
    assert(fa.kind == scientific::F_NONE);
    assert(all_close(a, [0.05372035912417292, -4.61214414692147e-05, -0.09157551252765038, -6.7107264123562e-05, 0.31313220746845083, 0.49991992278419434, 0.31313220746845083, -6.7107264123562e-05, -0.09157551252765038, -4.61214414692147e-05, 0.05372035912417292], 1.0e-10));
    let b, fb = scientific::remez(15, [0.0, 0.1, 0.2, 0.3, 0.35, 0.5], [0.0, 1.0, 0.0], [10.0, 1.0, 10.0]);
    assert(fb.kind == scientific::F_NONE);
    assert(all_close(b, [-0.011905703004954533, -0.0737020253398363, 0.02708217289037356, 0.17031926466831956, -0.0250884447180607, -0.26399147717667837, 0.009911974832641815, 0.30157740067295724, 0.009911974832641815, -0.26399147717667837, -0.0250884447180607, 0.17031926466831956, 0.02708217289037356, -0.0737020253398363, -0.011905703004954533], 1.0e-10));
    let c, fc = scientific::remez(12, [0.0, 0.45], [1.0], [], "differentiator");
    assert(fc.kind == scientific::F_NONE);
    assert(all_close(c, [-0.0005736103918763211, 0.001548655382839329, -0.0030141247437764315, 0.006920562112278915, -0.021291909480373357, 0.20139962561772198, -0.20139962561772198, 0.021291909480373357, -0.006920562112278915, 0.0030141247437764315, -0.001548655382839329, 0.0005736103918763211], 1.0e-10));
    let d, fd = scientific::remez(13, [0.05, 0.45], [1.0], [], "hilbert");
    assert(fd.kind == scientific::F_NONE);
    assert(all_close(d, [6.882585201938116e-05, 0.1075025400501003, -4.143283426749236e-05, 0.18302551835222908, -4.432283377647591e-05, 0.6264899668103487, 0.0, -0.6264899668103487, 4.432283377647591e-05, -0.18302551835222908, 4.143283426749236e-05, -0.1075025400501003, -6.882585201938116e-05], 1.0e-10));
    let e, fe = scientific::remez(10, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0]);
    assert(fe.kind == scientific::F_NONE);
    assert(all_close(e, [0.03446606203700606, -0.07216259080561685, -0.09029533865931524, 0.13984939677118302, 0.4471867306483239, 0.4471867306483239, 0.13984939677118302, -0.09029533865931524, -0.07216259080561685, 0.03446606203700606], 1.0e-10));
    // one pass returns the taps of that pass, as SciPy does
    let m, fm = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [], "bandpass", 1);
    assert(fm.kind == scientific::F_NONE);
    assert(all_close(m, [0.01534496766039262, 0.0279372076644553, -0.07015766044520624, -0.0487827502902359, 0.3048126927848137, 0.5564503872470827, 0.3048126927848137, -0.0487827502902359, -0.07015766044520624, 0.0279372076644553, 0.01534496766039262], 1.0e-10));
    // fs scales the edges
    let s, fs2 = scientific::remez(11, [0.0, 1600.0, 2400.0, 4000.0], [1.0, 0.0], [], "bandpass", 25, 16, 8000.0);
    assert(fs2.kind == scientific::F_NONE && all_close(s, [0.05372035912417292, -4.61214414692147e-05, -0.09157551252765038, -6.7107264123562e-05, 0.31313220746845083, 0.49991992278419434, 0.31313220746845083, -6.7107264123562e-05, -0.09157551252765038, -4.61214414692147e-05, 0.05372035912417292], 1.0e-10));
    // the faults
    let g1, gf1 = scientific::remez(11, [0.0, 0.2, 0.3], [1.0, 0.0]);
    assert(gf1.kind == scientific::F_SHAPE && g1.data.len == 0);
    let g2, gf2 = scientific::remez(11, [0.0, 0.3, 0.2, 0.5], [1.0, 0.0]);
    assert(gf2.kind == scientific::F_VALUE);
    let g3, gf3 = scientific::remez(11, [0.0, 0.2, 0.3, 0.7], [1.0, 0.0]);
    assert(gf3.kind == scientific::F_RANGE);
    let g4, gf4 = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [1.0]);
    assert(gf4.kind == scientific::F_SHAPE);
    let g5, gf5 = scientific::remez(1, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0]);
    assert(gf5.kind == scientific::F_RANGE);
    let g6, gf6 = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [], "lowpass");
    assert(gf6.kind == scientific::F_VALUE);
    let g7, gf7 = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [0.0, 1.0]);
    assert(gf7.kind == scientific::F_VALUE);
    let g8, gf8 = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 1.0 / 0.0]);
    assert(gf8.kind == scientific::F_VALUE);
    let g9, gf9 = scientific::remez(5000, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0]);
    assert(gf9.kind == scientific::F_LIMIT);
    let g9b, gf9b = scientific::remez(4096, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [], "bandpass", 25, 1024);
    assert(gf9b.kind == scientific::F_LIMIT);
    let g10, gf10 = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [], "bandpass", 25, 0);
    assert(gf10.kind == scientific::F_RANGE);
    let g11, gf11 = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [], "bandpass", 25, 16, 0.0);
    assert(gf11.kind == scientific::F_VALUE);
    // a band too narrow for the grid
    let g12, gf12 = scientific::remez(21, [0.0, 0.001, 0.4, 0.5], [1.0, 0.0], [], "bandpass", 25, 1);
    assert(gf12.kind == scientific::F_RANGE && g12.data.len == 0);
    // the exchange fails when the deviation stops rising
    let g13, gf13 = scientific::remez(73, [0.02, 0.3], [1.0], [], "differentiator");
    assert(gf13.kind == scientific::F_VALUE && g13.data.len == 0);
    return;
}

fn test_matfun() {
    let eg, egf = scientific::from_rows(3, 3, [4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0]);
    let ew, ev, evf = scientific::eigh(eg);
    assert(evf.kind == scientific::F_NONE && near(ew.data[0], 1.7078872783917407));
    var ec: i64 = 0;
    while ec < 3 {
        var er: i64 = 0;
        while er < 3 {
            var av: f64 = 0.0;
            var ek: i64 = 0;
            while ek < 3 { av += eg.data[er * 3 + ek] * ev.data[ek * 3 + ec]; ek += 1; }
            assert(mathshim_abs(av - ew.data[ec] * ev.data[er * 3 + ec]) < 1e-12);
            er += 1;
        }
        ec += 1;
    }
    let sq1, sq1f = scientific::sqrtm(mx(3, 3, [4.0, 1.0, 0.5, 2.0, 5.0, 1.0, 0.3, 1.0, 3.0]));
    assert(sq1f.kind == scientific::F_NONE && all_near(sq1, [1.9705736458839043, 0.2326189015635945, 0.12052654846139223, 0.4770456766300537, 2.197288610696068, 0.24105309692278393, 0.04870018207110656, 0.25286097042564826, 1.7126521582427425]));
    let sq2, sq2f = scientific::sqrtm(mx(3, 3, [4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0]));
    assert(sq2f.kind == scientific::F_NONE && all_near(sq2, [1.9813751385926213, 0.27199847235457564, -0.01301503760473606, 0.27199847235457564, 1.7028691474356779, 0.16202931138676027, -0.013015037604736044, 0.1620293113867601, 1.4048405998716291]));
    let lg1, lg1f = scientific::logm(mx(3, 3, [4.0, 1.0, 0.5, 2.0, 5.0, 1.0, 0.3, 1.0, 3.0]));
    assert(lg1f.kind == scientific::F_NONE && all_near(lg1, [1.3294065180596149, 0.21844838449997658, 0.11804265288679479, 0.4615884587830102, 1.5355090576680641, 0.23608530577358888, 0.02144221216596351, 0.2607769955566456, 1.058047369738802]));
    let lg2, lg2f = scientific::logm(mx(3, 3, [4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0]));
    assert(lg2f.kind == scientific::F_NONE && all_near(lg2, [1.3462209177519089, 0.2998680085203079, -0.03151209665712595, 0.2998680085203081, 1.030596860903038, 0.21295819757440607, -0.031512096657125926, 0.21295819757440604, 0.6677046590684773]));
    let fp1, fp1f = scientific::fractional_matrix_power(mx(3, 3, [4.0, 1.0, 0.5, 2.0, 5.0, 1.0, 0.3, 1.0, 3.0]), 0.3);
    assert(fp1f.kind == scientific::F_NONE && all_near(fp1, [1.4972472821626006, 0.10303621366502853, 0.05424803410513154, 0.21371622369338172, 1.5964615976459684, 0.10849606821026034, 0.01726122773643093, 0.1161398645735845, 1.377831504861876]));
    let fp2, fp2f = scientific::fractional_matrix_power(mx(3, 3, [4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0]), -1.5);
    assert(fp2f.kind == scientific::F_NONE && all_near(fp2, [0.14969850863100492, -0.08248914960039823, 0.02784907813664865, -0.0824891496003982, 0.24611219729972747, -0.09694273107349637, 0.027849078136648702, -0.09694273107349645, 0.38429950317342276]));
    let pu, pp, pf = scientific::polar(mx(3, 3, [4.0, 1.0, 0.5, 2.0, 5.0, 1.0, 0.3, 1.0, 3.0]));
    assert(pf.kind == scientific::F_NONE && all_near(pu, [0.992059084201441, -0.11589029290017523, 0.04886935097694872, 0.11660154520113564, 0.9931068338620442, -0.011953919575263015, -0.047147143180727956, 0.017557236343341512, 0.9987336433413461]) && all_near(pp, [4.187295284253818, 1.5279196670263924, 0.4711896577596735, 1.5279196670263924, 4.867201112753388, 0.9878333964419811, 0.47118965775967353, 0.9878333964419813, 3.008681685937249]));
    let cl, clf = scientific::cho_factor(mx(3, 3, [4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0]));
    let cs, csf = scientific::cho_solve(cl, av([1.0, 2.0, 3.0]));
    assert(clf.kind == scientific::F_NONE && csf.kind == scientific::F_NONE && all_near(cs, [0.15476190476190477, 0.38095238095238093, 1.4047619047619047]));
    let sy, syf = scientific::solve_sylvester(mx(2, 2, [3.0, 1.0, 0.0, 2.0]), mx(3, 3, [1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 2.0]), mx(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert(syf.kind == scientific::F_NONE && all_near(sy, [-0.04283447911158113, 0.24008461131676362, 0.2842411422527766, 0.8870967741935487, 0.6451612903225802, 1.3387096774193552]));
    let cl, clf = scientific::solve_continuous_lyapunov(mx(3, 3, [-4.0, -1.0, -0.5, -2.0, -5.0, -1.0, -0.3, -1.0, -3.0]), mx(3, 3, [1.0, 0.5, 0.0, 0.5, 2.0, 1.0, 0.0, 1.0, 3.0]));
    assert(clf.kind == scientific::F_NONE && all_near(cl, [-0.1289495773548676, -0.008766389020677931, 0.04912939688029589, -0.00876638902067795, -0.18593391332990858, -0.05279765530910223, 0.04912939688029592, -0.052797655309102264, -0.48731372125166217]));
    let dl, dlf = scientific::solve_discrete_lyapunov(mx(3, 3, [0.5, 0.1, 0.0, 0.2, 0.3, 0.1, 0.0, 0.1, 0.4]), mx(3, 3, [1.0, 0.5, 0.0, 0.5, 2.0, 1.0, 0.0, 1.0, 3.0]));
    assert(dlf.kind == scientific::F_NONE && all_near(dl, [1.486984365210881, 0.8998682629247193, 0.15999637619937518, 0.8998682629247193, 2.525144761568877, 1.4438060049393848, 0.15999637619937518, 1.443806004939385, 3.7389951523938567]));
    let bw, bv, bf = scientific::eig_banded(mx(2, 5, [0.0, 0.5, 0.7, 0.9, 0.2, 4.0, 3.0, 2.0, 1.0, 5.0]));
    assert(bf.kind == scientific::F_NONE && all_near(bw, [0.41060867538102946, 2.095896048454904, 3.2338111188766776, 4.2489152019055245, 5.010768955381866]));
    let bw2, bf2 = scientific::eigvals_banded(mx(2, 5, [0.0, 0.5, 0.7, 0.9, 0.2, 4.0, 3.0, 2.0, 1.0, 5.0]));
    assert(bf2.kind == scientific::F_NONE && all_near(bw2, [0.41060867538102863, 2.0958960484549043, 3.233811118876677, 4.2489152019055245, 5.010768955381865]));
    var bo: f64 = 0.0;
    var bi: i64 = 0;
    while bi < 25 { bo += scientific_abs(bv.data[bi]); bi += 1; }
    assert(near(bo, 7.978538605903487));
    let lw, lv, lf = scientific::eig_banded(mx(2, 5, [4.0, 3.0, 2.0, 1.0, 5.0, 0.5, 0.7, 0.9, 0.2, 0.0]), true);
    assert(lf.kind == scientific::F_NONE && all_near(lw, [0.41060867538102946, 2.095896048454904, 3.2338111188766776, 4.2489152019055245, 5.010768955381866]));
    let tw, tf = scientific::eigvals_banded(mx(3, 5, [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.5, 0.6, 0.7, 0.8, 4.0, 3.0, 2.0, 1.0, 5.0]));
    assert(tf.kind == scientific::F_NONE && all_near(tw, [-0.34285674122535553, -0.07369672660208344, 3.698430363453611, 4.441065369269095, 7.277057735104733]));
    // the select of the banded eigenvalues
    let sab = mx(2, 5, [0.0, 0.5, 0.7, 0.9, 0.2, 4.0, 3.0, 2.0, 1.0, 5.0]);
    let sv1, sf1 = scientific::eigvals_banded(sab, false, "v", 1.0, 4.0);
    assert(sf1.kind == scientific::F_NONE && all_near(sv1, [2.0958960484549043, 3.2338111188766776]));
    let si1, si1f = scientific::eigvals_banded(sab, false, "i", 0.0, 0.0, 1, 3);
    assert(si1f.kind == scientific::F_NONE && all_near(si1, [2.095896048454904, 3.233811118876678, 4.248915201905524]));
    let sw2, sv2, sf2 = scientific::eig_banded(sab, false, "i", 0.0, 0.0, 0, 1);
    assert(sf2.kind == scientific::F_NONE && all_near(sw2, [0.4106086753810293, 2.0958960484549043]) && sv2.shape[0] == 5 && sv2.shape[1] == 2);
    assert(near(scientific_abs(sv2.data[3 * 2]), 0.8301897815947307) && near(scientific_abs(sv2.data[3 * 2 + 1]), 0.5091050040794023));
    let sx1, sx1f = scientific::eigvals_banded(sab, false, "v", 4.0, 1.0);
    assert(sx1f.kind == scientific::F_RANGE && sx1.data.len == 0);
    let sx2, sx2f = scientific::eigvals_banded(sab, false, "i", 0.0, 0.0, 2, 9);
    assert(sx2f.kind == scientific::F_RANGE && sx2.data.len == 0);
    let sx3, sx3f = scientific::eigvals_banded(sab, false, "q");
    assert(sx3f.kind == scientific::F_VALUE && sx3.data.len == 0);
    let n1, n1f = scientific::sqrtm(mx(2, 2, [-1.0, 0.0, 0.0, -1.0]));
    assert(n1f.kind == scientific::F_VALUE);
    let n2, n2f = scientific::logm(mx(2, 2, [0.0, 0.0, 0.0, 0.0]));
    assert(n2f.kind == scientific::F_VALUE);
    let n3, n3f = scientific::sqrtm(mx(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert(n3f.kind == scientific::F_SHAPE);
    let n4, n4f = scientific::solve_sylvester(mx(2, 2, [1.0, 0.0, 0.0, 1.0]), mx(2, 2, [-1.0, 0.0, 0.0, -1.0]), mx(2, 2, [1.0, 1.0, 1.0, 1.0]));
    assert(n4f.kind == scientific::F_VALUE);
}

// The trigonometric and hyperbolic functions of a matrix are pinned against SciPy.
fn tm_mat(v: f64[:]) -> scientific::Arr {
    let m, f = scientific::from_rows(2, 2, v);
    return m;
}

fn test_matrix_trig() {
    let b = tm_mat([2.0, 1.0, 0.5, 3.0]);
    let s, f1 = scientific::sinm(b);
    assert(f1.kind == scientific::F_NONE && all_loose(s, [0.7400706, -0.70468966, -0.35234483, 0.03538094]));
    let c, f2 = scientific::cosm(b);
    assert(f2.kind == scientific::F_NONE && all_loose(c, [-0.25581893, -0.52641889, -0.26320944, -0.78223782]));
    let t, f3 = scientific::tanm(b);
    assert(f3.kind == scientific::F_NONE && all_loose(t, [-12.4184748, 9.25806596, 4.62903298, -3.16040885]));
    let sh, f4 = scientific::sinhm(b);
    assert(f4.kind == scientific::F_NONE && all_loose(sh, [5.00038102, 6.92808946, 3.46404473, 11.92847048]));
    let ch, f5 = scientific::coshm(b);
    assert(f5.kind == scientific::F_NONE && all_loose(ch, [5.16158917, 6.83535212, 3.41767606, 11.99694129]));
    let th, f6 = scientific::tanhm(b);
    assert(f6.kind == scientific::F_NONE && all_loose(th, [0.94162814, 0.04098791, 0.02049395, 0.98261605]));
    let sg, f7 = scientific::signm(tm_mat([2.0, 1.0, 1.0, -3.0]));
    assert(f7.kind == scientific::F_NONE && all_loose(sg, [0.92847669, 0.37139068, 0.37139068, -0.92847669]));
    let bad, f8 = scientific::signm(tm_mat([0.0, 1.0, -1.0, 0.0]));
    assert(f8.kind == scientific::F_VALUE);
    let nsq, f9 = scientific::sinm(scientific::line_arr([1.0, 2.0, 3.0]));
    assert(f9.kind == scientific::F_SHAPE);
    return;
}

// interp1d takes the names zero and slinear and the fill values of SciPy.
fn test_interp1d_kinds() {
    let x = [0.0, 1.0, 2.5, 3.0, 4.5];
    let y = [1.0, 3.0, 2.0, 5.0, 4.0];
    let xs = [0.0, 0.4, 1.0, 1.7, 2.5, 2.9, 3.6, 4.5];
    let z, f1 = scientific::interp1d(x, y, xs, "zero");
    assert(f1.kind == scientific::F_NONE && all_near(z, [1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 5.0, 4.0]));
    let sl, f2 = scientific::interp1d(x, y, xs, "slinear");
    assert(f2.kind == scientific::F_NONE && all_loose(sl, [1.0, 1.8, 3.0, 2.53333333, 2.0, 4.4, 4.6, 4.0]));
    let out = [-1.0, 5.0];
    let a, f3 = scientific::interp1d(x, y, out, "linear", false, true, -7.0, -8.0);
    assert(f3.kind == scientific::F_NONE && all_near(a, [-7.0, -8.0]));
    let b, f4 = scientific::interp1d(x, y, out, "zero", true);
    assert(f4.kind == scientific::F_NONE && all_near(b, [1.0, 4.0]));
    let qq, fq = scientific::interp1d(x, y, xs, "quadratic");
    assert(fq.kind == scientific::F_NONE && all_loose(qq, [1.0, 2.32643741, 3.0, 1.78974782, 2.0, 4.44104539, 6.7114718, 4.0]));
    let qe, fqe = scientific::interp1d(x, y, out, "quadratic", true);
    assert(fqe.kind == scientific::F_NONE && all_loose(qe, [-5.38697845, -0.2434663]));
    let c, f5 = scientific::interp1d(x, y, out, "linear");
    assert(f5.kind == scientific::F_RANGE);
    return;
}

fn sf_sos() -> scientific::Arr {
    let s, f = scientific::from_rows(2, 6, [0.01856301062689717, 0.03712602125379434, 0.01856301062689717, 1.0, -0.6727409111915273, 0.14453519983312102, 1.0, 2.0, 1.0, 1.0, -0.8976579400366445, 0.5271869046315972]);
    return s;
}

// sosfreqz is pinned against SciPy.
fn test_sosfreqz() {
    let w, hr, hi, f1 = scientific::sosfreqz(sf_sos(), 5);
    assert(f1.kind == scientific::F_NONE);
    assert(all_near(w, [0.0, 0.6283185307179586, 1.2566370614359172, 1.8849555921538759, 2.5132741228718345]));
    assert(all_loose(hr, [1.0, -0.21708827518165963, -0.09993200670106575, 0.010342348962786515, 0.0006814869992726401]));
    assert(all_loose(hi, [0.0, -0.9624215957339939, 0.2128155141360432, 0.01567225283401711, 0.00031608231753612226]));
    let r2, i2, f2 = scientific::sosfreqz_at(sf_sos(), [0.1, 1.0, 2.0]);
    assert(f2.kind == scientific::F_NONE);
    assert(all_loose(r2, [0.9671606461491014, -0.5838102178759325, 0.007400139523795283]));
    assert(all_loose(i2, [-0.25416584326812836, 0.15258273743416995, 0.008744911198204164]));
    let w3, r3, i3, f3 = scientific::sosfreqz(sf_sos(), 0);
    assert(f3.kind == scientific::F_SHAPE);
    return;
}

fn rp_signal() -> f64[>..] {
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < 20 { v.push(mathshim_sin(i * 0.3) + 0.1 * i); i += 1; }
    return copy(v);
}

// resample_poly is pinned against SciPy.
fn test_resample_poly() {
    let x = rp_signal();
    let a, f1 = scientific::resample_poly(x, 3, 2);
    assert(f1.kind == scientific::F_NONE && a.data.len == 30);
    assert(near(a.data[1], 0.2380285985652429) && near(a.data[5], 1.181436686153129) && near(a.data[15], 1.1418117248304391) && near(a.data[29], 0.9643440493764917));
    let b, f2 = scientific::resample_poly(x, 1, 2);
    assert(f2.kind == scientific::F_NONE && all_loose(b, [0.0642898273039825, 0.7568253702817194, 1.333241437659126, 1.5785041508482276, 1.4677133835228036, 1.1564010623373906, 0.7343325113364108, 0.5659863429365632, 0.5420691602674772, 1.15102267428965]));
    let c, f3 = scientific::resample_poly(x, 0, 2);
    assert(f3.kind == scientific::F_RANGE);
    return;
}

fn dc_signal() -> f64[>..] {
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < 60 { v.push(mathshim_sin(i * 0.3) + 0.1 * i); i += 1; }
    return copy(v);
}

// decimate is pinned against SciPy.
fn test_decimate() {
    let x = dc_signal();
    let a, f1 = scientific::decimate(x, 3);
    assert(f1.kind == scientific::F_NONE && a.data.len == 20);
    assert(loose(a.data[1], 1.0729684936013582) && loose(a.data[2], 1.5557205343558305) && loose(a.data[5], 0.5152653616743713));
    let b, f2 = scientific::decimate(x, 2, "fir");
    assert(f2.kind == scientific::F_NONE && b.data.len == 30);
    assert(loose(b.data[0], 0.06383926367780504) && loose(b.data[1], 0.754324189814132) && loose(b.data[5], 1.1396273776282466));
    let c, f3 = scientific::decimate(x, 4, "iir", 6);
    assert(f3.kind == scientific::F_NONE && loose(c.data[1], 1.326462621204827) && loose(c.data[3], 0.7518117215075588));
    let d, f4 = scientific::decimate(x, 0);
    assert(f4.kind == scientific::F_RANGE);
    let e, f5 = scientific::decimate(x, 2, "bessel");
    assert(f5.kind == scientific::F_VALUE);
    return;
}

// firwin2 is pinned against SciPy.
fn test_firwin2() {
    let a, f1 = scientific::firwin2(9, [0.0, 0.3, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0]);
    assert(f1.kind == scientific::F_NONE);
    assert(all_loose(a, [-0.004709708691207965, -0.01152260218143383, 0.048068435342338324, 0.25764638184042365, 0.3984375, 0.2576463818404236, 0.04806843534233829, -0.011522602181433838, -0.004709708691207965]));
    let b, f2 = scientific::firwin2(8, [0.0, 0.5, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0], 0, "none");
    assert(f2.kind == scientific::F_NONE);
    assert(all_loose(b, [-0.05385075661518586, -0.08268148379274896, 0.14568866650727913, 0.44871097697629203, 0.44871097697629203, 0.1456886665072791, -0.08268148379274898, -0.053850756615185874]));
    let c, f3 = scientific::firwin2(11, [0.0, 0.5, 1.0], [0.0, 1.0, 0.0], 64, "hamming", true);
    assert(f3.kind == scientific::F_NONE && loose(c.data[0], 0.001293533458552159) && loose(c.data[2], -0.017899195736421628) && loose(c.data[4], 0.3696412747256351) && c.data[5] == 0.0 && loose(c.data[6], -0.36964127472563496));
    let d, f4 = scientific::firwin2(11, [0.0, 500.0, 1000.0], [0.0, 1.0, 0.0], 0, "hann", false, 2000.0);
    assert(f4.kind == scientific::F_NONE && d.data.len == 11 && loose(d.data[3], -0.13434882372563778) && loose(d.data[5], 0.5) && loose(d.data[7], -0.13434882372563778));
    let e, f5 = scientific::firwin2(9, [0.1, 1.0], [1.0, 0.0]);
    assert(f5.kind == scientific::F_VALUE);
    return;
}

fn sp_signal() -> f64[>..] {
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < 40 { v.push(mathshim_sin(i * 0.7) + 0.05 * i); i += 1; }
    return copy(v);
}

// spectrogram and welch_psd are pinned against SciPy.
fn test_spectral() {
    let x = sp_signal();
    let f, t, s, f1 = scientific::spectrogram(x, 2.0, "tukey", 0.25, 16, 4);
    assert(f1.kind == scientific::F_NONE && s.shape[0] == 9 && s.shape[1] == 3);
    assert(all_near(t, [4.0, 10.0, 16.0]) && near(f.data[1], 0.125));
    assert(loose(s.data[0], 0.016398352268530342) && loose(s.data[1 * 3], 1.2403196212067757) && loose(s.data[2 * 3], 2.704032777944856) && loose(s.data[3 * 3], 0.0031842838682634464) && loose(s.data[3 * 3 + 1], 0.0028130141030115077) && loose(s.data[3 * 3 + 2], 0.05353120041768578));
    let f2, t2, s2, g2 = scientific::spectrogram(x, 1.0, "hann", 0.0 / 0.0, 8, 2, false, "spectrum");
    assert(g2.kind == scientific::F_NONE && s2.shape[0] == 5 && s2.shape[1] == 6);
    assert(loose(s2.data[0 * 6 + 1], 0.7785029913294211) && loose(s2.data[1 * 6 + 1], 1.0224606287871703) && loose(s2.data[2 * 6 + 1], 0.09495759220933907) && loose(s2.data[3 * 6 + 1], 0.00015773875399762466) && loose(s2.data[4 * 6 + 1], 3.9928095795776335e-06));
    let wf, p, f3 = scientific::welch_psd(x, 2.0, "hann", 16, 8);
    assert(f3.kind == scientific::F_NONE && p.data.len == 9);
    assert(loose(p.data[0], 0.0540555950887493) && loose(p.data[1], 1.3550356626194178) && loose(p.data[2], 2.4944916696212753) && loose(p.data[3], 0.31407830694318295));
    let wf2, p2, f4 = scientific::welch_psd(x, 1.0, "hamming", 9, -1, false, "spectrum");
    assert(f4.kind == scientific::F_NONE && p2.data.len == 5);
    assert(loose(p2.data[0], 1.2150560394826984) && loose(p2.data[1], 0.8644536996548504) && loose(p2.data[2], 0.09172796773120981));
    let a, b, c, f5 = scientific::spectrogram(x, 1.0, "hann", 0.0 / 0.0, 64);
    assert(f5.kind == scientific::F_RANGE);
    let w3, p3, f6 = scientific::welch_psd(x, 1.0, "hann", 8, 8);
    assert(f6.kind == scientific::F_RANGE);
    return;
}

// stft is pinned against SciPy.
fn test_stft() {
    let x = sp_signal();
    let f, t, zr, zi, f1 = scientific::stft(x, 2.0, "hann", 8, 4);
    assert(f1.kind == scientific::F_NONE && zr.shape[0] == 5 && zr.shape[1] == 11);
    assert(near(t.data[1], 2.0) && near(f.data[1], 0.25));
    assert(loose(zr.data[0 * 11 + 2], 0.03263936287197537) && loose(zr.data[1 * 11 + 2], 0.12002113379413716) && loose(zr.data[2 * 11 + 2], -0.13099301275623057));
    assert(loose(zi.data[1 * 11 + 2], 0.424366549395401) && loose(zi.data[2 * 11 + 2], -0.17456594372523282) && loose(zi.data[3 * 11 + 2], -0.007774043387693164));
    let f2, t2, wr, wi, g2 = scientific::stft(x, 1.0, "hann", 8, 4, false, false);
    assert(g2.kind == scientific::F_NONE && wr.shape[1] == 9);
    assert(loose(wr.data[0 * 9 + 1], 0.03263936287197537) && loose(wi.data[1 * 9 + 1], 0.424366549395401) && loose(wr.data[4 * 9 + 1], -0.0019199752318358326));
    let a, b, c, d, g3 = scientific::stft(x, 1.0, "hann", 64, 4);
    assert(g3.kind == scientific::F_NONE);
    let a2, b2, c2, d2, g4 = scientific::stft(x, 1.0, "hann", 8, 8);
    assert(g4.kind == scientific::F_RANGE);
    return;
}

// istft returns the signal of stft, and it follows SciPy.
fn test_istft() {
    let x = sp_signal();
    let f, t, zr, zi, f1 = scientific::stft(x, 2.0, "hann", 8, 4);
    assert(f1.kind == scientific::F_NONE);
    let t2, y, g = scientific::istft(zr, zi, 2.0, "hann", 8, 4);
    assert(g.kind == scientific::F_NONE && y.data.len == 40);
    assert(near(t2.data[2], 1.0));
    var i: i64 = 0;
    while i < 40 { assert(loose(y.data[i] + 1.0, x[i] + 1.0)); i += 1; }
    let f3, t3, wr, wi, f4 = scientific::stft(x, 1.0, "hann", 8, 4, false, false);
    let t4, y4, g4 = scientific::istft(wr, wi, 1.0, "hann", 8, 4, false);
    assert(g4.kind == scientific::F_VALUE);
    let t5, y5, g5 = scientific::istft(zr, zi, 2.0, "hann", 6, 4);
    assert(g5.kind == scientific::F_SHAPE);
    return;
}

// differential_evolution finds the global minimum of two test functions.
fn test_differential_evolution() {
    // The sphere: the minimum is zero at the origin.
    let a, f1 = scientific::differential_evolution([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], 3, 2000, 15, 1e-8) { v => v[0] * v[0] + v[1] * v[1] + v[2] * v[2] };
    assert(f1.kind == scientific::F_NONE && a.data.len == 4 && a.data[3] < 1e-6);
    // The Rosenbrock function: the minimum is zero at one, one.
    let b, f2 = scientific::differential_evolution([-5.0, -5.0], [5.0, 5.0], 7, 3000, 15, 1e-10) { v => 100.0 * (v[1] - v[0] * v[0]) * (v[1] - v[0] * v[0]) + (1.0 - v[0]) * (1.0 - v[0]) };
    assert(f2.kind == scientific::F_NONE && b.data[2] < 1e-5);
    // The Rastrigin function has many local minima; the global one is zero at the origin.
    let c, f3 = scientific::differential_evolution([-5.12, -5.12], [5.12, 5.12], 11, 3000, 40, 1e-10) { v => 20.0 + v[0] * v[0] - 10.0 * mathshim_cos(6.283185307179586 * v[0]) + v[1] * v[1] - 10.0 * mathshim_cos(6.283185307179586 * v[1]) };
    assert(f3.kind == scientific::F_NONE && c.data[2] < 1e-4);
    // The same seed gives the same result.
    let d, f4 = scientific::differential_evolution([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], 3, 2000, 15, 1e-8) { v => v[0] * v[0] + v[1] * v[1] + v[2] * v[2] };
    assert(d.data[0] == a.data[0] && d.data[3] == a.data[3]);
    let e, f5 = scientific::differential_evolution([1.0], [0.0]) { v => v[0] };
    assert(f5.kind == scientific::F_VALUE);
    let g, f6 = scientific::differential_evolution([0.0, 1.0], [1.0]) { v => v[0] };
    assert(f6.kind == scientific::F_SHAPE);
    return;
}

// The structure functions of sparse matrices are pinned against SciPy.
fn test_sparse_structure() {
    let a, fa = scientific::csr_from_dense(mx(3, 4, [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 0.0]));
    let b, fb = scientific::csr_from_dense(mx(2, 2, [0.0, 7.0, 8.0, 0.0]));
    let s, fs = scientific::csr_from_dense(mx(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
    let c, fc = scientific::csr_from_dense(mx(3, 4, [1.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE && fs.kind == scientific::F_NONE && fc.kind == scientific::F_NONE);
    let bd, f1 = scientific::csr_block_diag(a, b);
    let bdd, f1d = scientific::csr_to_dense(bd);
    assert(f1.kind == scientific::F_NONE && bd.rows == 5 && bd.cols == 6 && all_near(bdd, [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0]));
    let lo, f2 = scientific::csr_tril(s, -1);
    let lod, f2d = scientific::csr_to_dense(lo);
    assert(f2.kind == scientific::F_NONE && all_near(lod, [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0]));
    let up, f3 = scientific::csr_triu(s, 1);
    let upd, f3d = scientific::csr_to_dense(up);
    assert(f3.kind == scientific::F_NONE && all_near(upd, [0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]));
    let lo0, f4 = scientific::csr_tril(s);
    let lo0d, f4d = scientific::csr_to_dense(lo0);
    assert(f4.kind == scientific::F_NONE && all_near(lo0d, [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]));
    let m, f5 = scientific::csr_multiply(a, c);
    let md, f5d = scientific::csr_to_dense(m);
    assert(f5.kind == scientific::F_NONE && all_near(md, [1.0, 0.0, 6.0, 0.0, 0.0, 6.0, 0.0, 16.0, 25.0, 0.0, 0.0, 0.0]));
    let sl, f6 = scientific::csr_slice(a, 1, 3, 1, 4);
    let sld, f6d = scientific::csr_to_dense(sl);
    assert(f6.kind == scientific::F_NONE && sl.rows == 2 && sl.cols == 3 && all_near(sld, [3.0, 0.0, 4.0, 0.0, 6.0, 0.0]));
    let bad, f7 = scientific::csr_multiply(a, b);
    assert(f7.kind == scientific::F_SHAPE);
    let bad2, f8 = scientific::csr_slice(a, 2, 2, 0, 1);
    assert(f8.kind == scientific::F_RANGE);
    return;
}

// lsqr and minres are pinned against the dense solutions of NumPy.
fn test_lsqr_minres() {
    let a, fa = scientific::csr_from_dense(mx(4, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 0.0, 1.0]));
    let x1, i1, f1 = scientific::lsqr(a, [1.0, 2.0, 2.0, 3.0], 1.0e-12);
    assert(fa.kind == scientific::F_NONE && f1.kind == scientific::F_NONE && i1 == 0 && all_loose(x1, [-2.7142857142857144, 2.285714285714286]));
    let w, fw = scientific::csr_from_dense(mx(2, 4, [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0]));
    let x2, i2, f2 = scientific::lsqr(w, [1.0, 2.0], 1.0e-12);
    assert(fw.kind == scientific::F_NONE && f2.kind == scientific::F_NONE && i2 == 0 && all_loose(x2, [0.2, 0.24, 0.4, 0.32]));
    let s, fs = scientific::csr_from_dense(mx(3, 3, [2.0, 1.0, 0.0, 1.0, -3.0, 1.0, 0.0, 1.0, 4.0]));
    let x3, i3, f3 = scientific::minres(s, [1.0, 2.0, 3.0], 1.0e-12);
    assert(fs.kind == scientific::F_NONE && f3.kind == scientific::F_NONE && i3 == 0 && all_loose(x3, [0.6, -0.2, 0.8]));
    let x4, i4, f4 = scientific::lsqr(a, [1.0, 2.0, 3.0]);
    assert(f4.kind == scientific::F_SHAPE);
    let x5, i5, f5 = scientific::minres(a, [1.0, 2.0, 3.0, 4.0]);
    assert(f5.kind == scientific::F_SHAPE);
    return;
}

fn eg_matrix() -> scientific::Csr {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < 30 {
        ri.push(i); ci.push(i); v.push(2.0 + 0.1 * i);
        if i + 1 < 30 {
            ri.push(i); ci.push(i + 1); v.push(-1.0);
            ri.push(i + 1); ci.push(i); v.push(-1.0);
        }
        i += 1;
    }
    let s, f = scientific::csr_from_triplets(30, 30, copy(ri), copy(ci), copy(v));
    assert(f.kind == scientific::F_NONE);
    return s;
}

// eigsh is pinned against SciPy.
fn test_eigsh() {
    let s = eg_matrix();
    let la, v1, f1 = scientific::eigsh(s, 3, "LA");
    assert(f1.kind == scientific::F_NONE && all_loose(la, [5.835442141035716, 6.132637310688383, 6.500509781468499]));
    let sa, v2, f2 = scientific::eigsh(s, 3, "SA");
    assert(f2.kind == scientific::F_NONE && all_loose(sa, [0.39949021853149425, 0.7673626893116112, 1.0645578589642712]));
    let lm, v3, f3 = scientific::eigsh(s, 3, "LM");
    assert(f3.kind == scientific::F_NONE && all_loose(lm, [5.83544214103572, 6.132637310688378, 6.500509781468498]));
    let sm, v4, f4 = scientific::eigsh(s, 3, "SM");
    assert(f4.kind == scientific::F_NONE && all_loose(sm, [0.3994902185314938, 0.7673626893116103, 1.064557858964272]));
    // The first vector of SA holds a x = value x.
    var col: f64[>..] = [];
    var i: i64 = 0;
    while i < 30 { col.push(v2.data[i * 3]); i += 1; }
    let ax, fx = scientific::csr_matvec(s, col);
    assert(fx.kind == scientific::F_NONE);
    i = 0;
    while i < 30 { assert(near(ax.data[i] + 10.0, sa.data[0] * col[i] + 10.0)); i += 1; }
    let bad, vb, fb = scientific::eigsh(s, 30);
    assert(fb.kind == scientific::F_RANGE);
    let bad2, vb2, fb2 = scientific::eigsh(s, 3, "XX");
    assert(fb2.kind == scientific::F_VALUE);
    return;
}

// eigvalsh takes a matrix of twelve rows to full accuracy.
fn test_eigvalsh_larger() {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 144 { d.push(0.0); i += 1; }
    i = 0;
    while i < 12 {
        d[i * 12 + i] = 2.0 + 0.1 * i;
        if i + 1 < 12 { d[i * 12 + i + 1] = -1.0; d[(i + 1) * 12 + i] = -1.0; }
        i += 1;
    }
    let a, fa = scientific::from_rows(12, 12, copy(d));
    let w, f = scientific::eigvalsh(a);
    assert(fa.kind == scientific::F_NONE && f.kind == scientific::F_NONE);
    assert(all_loose(w, [0.39949588055807855, 0.7684859001589346, 1.086782275967645, 1.4388282352528388, 1.8538443683280288, 2.312938653238329, 2.7870613467616696, 3.246155631671971, 3.6611717647471615, 4.013217724032353, 4.3315140998410655, 4.7005041194419235]));
    return;
}


// maxdiff gives the largest absolute difference of the array and the
// list of the caller.
fn maxdiff(a: scientific::Arr&, v: f64[:]) -> f64 {
    var w: f64 = 0.0;
    var i: i64 = 0;
    while i < v.len {
        var d = a.data[i] - v[i];
        if d < 0.0 { d = -d; }
        if d > w { w = d; }
        i += 1;
    }
    return w;
}

// schur and funm and ldl are pinned against SciPy 1.18.1.
fn test_schur_funm_ldl() {
    let A = scientific::from_rows(5, 5, [4.0, 1.0, -2.0, 2.0, 0.0, 1.0, 3.0, 1.0, -1.0, 2.0, -2.0, 1.0, 5.0, 1.0, -1.0, 2.0, -1.0, 1.0, 3.0, 1.0, 0.0, 2.0, -1.0, 1.0, 4.0]);
    let ta, za, fa = scientific::schur(A);
    assert(fa.kind == scientific::F_NONE);
    // The change of the basis is orthogonal.
    let zta = scientific::transpose(za);
    let ga = scientific::matmul(zta, za);
    var orth: f64 = 0.0;
    var i: i64 = 0;
    while i < ga.shape[0] {
        var j: i64 = 0;
        while j < ga.shape[1] {
            let g = if j == i { 1.0 } else { 0.0 };
            var d = ga.data[i * ga.shape[1] + j] - g;
            if d < 0.0 { d = -d; }
            if d > orth { orth = d; }
            j += 1;
        }
        i += 1;
    }
    assert(orth < 1.0e-13);
    // The form rebuilds the matrix.
    let ra = scientific::matmul(za, ta);
    let ra2 = scientific::matmul(ra, zta);
    assert(maxdiff(ra2, A.data) < 1.0e-13);
    // The matrix is symmetric, so the form is diagonal and holds the
    // eigenvalues.
    var low: f64 = 0.0;
    i = 1;
    while i < ta.shape[0] {
        var d = ta.data[i * ta.shape[1] + (i - 1)];
        if d < 0.0 { d = -d; }
        if d > low { low = d; }
        i += 1;
    }
    assert(low == 0.0);
    let eva = scientific::eigvalsh(A);
    var diag_ok = true;
    i = 0;
    while i < 5 {
        var found = false;
        var j: i64 = 0;
        while j < 5 {
            if !found && loose(ta.data[i * 5 + i], eva.data[j]) { found = true; }
            j += 1;
        }
        if !found { diag_ok = false; }
        i += 1;
    }
    assert(diag_ok);
    // B holds a pair of complex values, and the form keeps a block of
    // two rows for it.
    let B = scientific::from_rows(4, 4, [1.0, 2.0, 0.0, 3.0, 4.0, 1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 2.0, 2.0, 0.0, 4.0, 1.0]);
    let tb, zb, fb = scientific::schur(B);
    assert(fb.kind == scientific::F_NONE);
    let ztb = scientific::transpose(zb);
    let rb = scientific::matmul(zb, tb);
    let rb2 = scientific::matmul(rb, ztb);
    assert(maxdiff(rb2, B.data) < 1.0e-13);
    // Below the subdiagonal the form is zero.
    var low2: f64 = 0.0;
    i = 0;
    while i < tb.shape[0] {
        var j: i64 = 0;
        while j < i - 1 {
            var d = tb.data[i * 4 + j];
            if d < 0.0 { d = -d; }
            if d > low2 { low2 = d; }
            j += 1;
        }
        i += 1;
    }
    assert(low2 == 0.0);
    // The block of two rows holds the values 1 plus and minus the
    // square root of two.
    let tr = tb.data[2 * 4 + 2] + tb.data[3 * 4 + 3];
    let det = tb.data[2 * 4 + 2] * tb.data[3 * 4 + 3] - tb.data[2 * 4 + 3] * tb.data[3 * 4 + 2];
    assert(loose(tr, 2.0) && loose(det, 3.0));
    // funm of exp and cos matches expm and cosm.
    let fa0, faf = scientific::funm(A) { x => mathshim::power_e(x) };
    assert(faf.kind == scientific::F_NONE && maxdiff(fa0, [434.58822489091307, 73.88495387687904, -407.5164308328381, 169.17283558549684, 224.6102821473738, 73.88495387687907, 117.08605144571604, -27.88445872891634, 30.18979999344893, 140.13141871107828, -407.51643083283807, -27.88445872891626, 524.0714715355521, -91.10898574982119, -202.02920946801962, 169.17283558549673, 30.189799993448926, -91.1089857498211, 115.40237007044733, 90.13466826248785, 224.61028214737374, 140.13141871107823, -202.02920946801967, 90.13466826248786, 249.96568795400466]) < 1.0e-6);
    let fc0, fcf = scientific::funm(A) { x => mathshim::cosine(x) };
    assert(fcf.kind == scientific::F_NONE && maxdiff(fc0, [0.15704861258425307, -0.31309477452687295, -0.305029467260153, 0.1971229086803145, 0.4862098577314571, -0.31309477452687307, 0.0763608808454167, -0.016327546660809586, 0.41569737623326863, 0.4941064682994968, -0.305029467260153, -0.016327546660809905, 0.5397328805261301, -0.06275755414680262, 0.14701263854926788, 0.19712290868031437, 0.4156973762332685, -0.06275755414680259, 0.03501849847974753, -0.16205918215711979, 0.48620985773145714, 0.4941064682994968, 0.1470126385492677, -0.16205918215711962, 0.1417857372232823]) < 1.0e-12);
    // funm of sin on a matrix with real values matches SciPy.
    let C = scientific::from_rows(3, 3, [2.0, 1.0, 0.5, 0.3, 5.0, 1.0, 0.1, 0.2, 7.0]);
    let fs0, fsf = scientific::funm(C) { x => mathshim::sine(x) };
    assert(fsf.kind == scientific::F_NONE && maxdiff(fs0, [0.8988026555927285, -0.5607175842086072, 0.24946869705817168, -0.14919582693367267, -0.8241060577378951, 0.8113997595198518, 0.011854842753815629, 0.18129940023287966, 0.6492549387175257]) < 1.0e-12);
    // Two equal values on the diagonal are a fault.
    let eye = scientific::eye(3);
    let b0, bf = scientific::funm(eye) { x => mathshim::power_e(x) };
    assert(bf.kind == scientific::F_VALUE);
    // A pair of complex values is a fault of funm.
    let bB0, bBf = scientific::funm(B) { x => mathshim::power_e(x) };
    assert(bBf.kind == scientific::F_VALUE);
    // ldl against SciPy on a matrix that needs no pivoting.
    let S = scientific::from_rows(5, 5, [10.0, 1.0, -2.0, 2.0, 0.0, 1.0, 9.0, 1.0, -1.0, 2.0, -2.0, 1.0, 11.0, 1.0, -1.0, 2.0, -1.0, 1.0, 9.0, 1.0, 0.0, 2.0, -1.0, 1.0, 10.0]);
    let lo, di, fl = scientific::ldl(S);
    assert(fl.kind == scientific::F_NONE);
    assert(maxdiff(lo, [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, -0.2, 0.1348314606741573, 1.0, 0.0, 0.0, 0.2, -0.1348314606741573, 0.1496232508073197, 1.0, 0.0, 0.0, 0.2247191011235955, -0.12163616792249732, 0.17790606140120704, 1.0]) < 1.0e-14);
    assert(maxdiff(di, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.9, 0.0, 0.0, 0.0, 0.0, 0.0, 10.438202247191011, 0.0, 0.0, 0.0, 0.0, 0.0, 8.204520990312163, 0.0, 0.0, 0.0, 0.0, 0.0, 9.136447126738389]) < 1.0e-14);
    let lt = scientific::transpose(lo);
    let ld = scientific::matmul(lo, di);
    let rec = scientific::matmul(ld, lt);
    assert(maxdiff(rec, S.data) < 1.0e-13);
    // A matrix that needs pivoting is a fault.
    let P0 = scientific::from_rows(2, 2, [0.0, 1.0, 1.0, 0.0]);
    let pl, pd, pf = scientific::ldl(P0);
    assert(pf.kind == scientific::F_VALUE);
    // A matrix that is not symmetric is a fault.
    let ns = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let nl, nd, nf = scientific::ldl(ns);
    assert(nf.kind == scientific::F_VALUE);
    // A matrix that is not square is a fault of schur.
    let rect = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rl, rz, rf = scientific::schur(rect);
    assert(rf.kind == scientific::F_SHAPE);
}

// rel_close tells whether the values agree to a relative tolerance.
fn rel_close(a: scientific::Arr&, v: f64[:], tol: f64) -> bool {
    if a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        var d = a.data[i] - v[i];
        if d < 0.0 { d = -d; }
        var s = v[i];
        if s < 0.0 { s = -s; }
        if d > tol * (1.0 + s) { return false; }
        i += 1;
    }
    return true;
}

// res_match tells whether each expected pole and residue has a partner in
// the result. Equal poles keep their order, and poles of one size may
// come in either order.
fn res_match(rr: scientific::Arr&, ri: scientific::Arr&, pr: scientific::Arr&, pi: scientific::Arr&, er: f64[:], ei: f64[:], ep: f64[:], eq: f64[:], tol: f64) -> bool {
    let n = er.len;
    if rr.data.len != n || ri.data.len != n || pr.data.len != n || pi.data.len != n { return false; }
    var used: i64[>..] = [];
    var i: i64 = 0;
    while i < n { used.push(0); i += 1; }
    i = 0;
    while i < n {
        var found = false;
        var j: i64 = 0;
        while j < n && !found {
            if used[j] == 0 && mathshim_abs(pr.data[j] - ep[i]) < 1.0e-6 && mathshim_abs(pi.data[j] - eq[i]) < 1.0e-6 {
                used[j] = 1;
                found = true;
                if mathshim_abs(rr.data[j] - er[i]) > tol * (1.0 + mathshim_abs(er[i])) { return false; }
                if mathshim_abs(ri.data[j] - ei[i]) > tol * (1.0 + mathshim_abs(ei[i])) { return false; }
            }
            j += 1;
        }
        if !found { return false; }
        i += 1;
    }
    return true;
}

// test_cheby_ellip_residue pins cheby1, cheby2, ellip and residue against SciPy.
fn test_cheby_ellip_residue() {
    let b1, a1, f1 = scientific::cheby1(3, 1.0, [0.3], "lowpass");
    assert(f1.kind == scientific::F_NONE && rel_close(b1, [0.034384966351816024, 0.10315489905544807, 0.10315489905544807, 0.034384966351816024], 1.0e-9) && rel_close(a1, [1.0, -1.5804049383264676, 1.2538449813550228, -0.3983603122140273], 1.0e-9));
    let b2, a2, f2 = scientific::cheby1(4, 0.5, [0.4], "highpass");
    assert(f2.kind == scientific::F_NONE && rel_close(b2, [0.12926985604483177, -0.5170794241793271, 0.7756191362689906, -0.5170794241793271, 0.12926985604483177], 1.0e-9) && rel_close(a2, [1.0, -0.35159185255713216, 0.7707500762220733, 0.06141159624217388, 0.12994289256088754], 1.0e-9));
    let b3, a3, f3 = scientific::cheby1(3, 2.0, [0.2, 0.5], "bandpass");
    assert(f3.kind == scientific::F_NONE && rel_close(b3, [0.025669272296771684, 0.0, -0.07700781689031505, 0.0, 0.07700781689031505, 0.0, -0.025669272296771684], 1.0e-9) && rel_close(a3, [1.0, -2.4301856634611205, 3.8506441743304145, -3.9367540729497987, 3.0948206588370564, -1.5205085057390646, 0.5023259688156135], 1.0e-9));
    let b4, a4, f4 = scientific::cheby1(2, 1.0, [0.3, 0.6], "bandstop");
    assert(f4.kind == scientific::F_NONE && rel_close(b4, [0.5113915629891412, -0.3591410990179514, 1.0858377099500565, -0.3591410990179514, 0.5113915629891412], 1.0e-9) && rel_close(a4, [1.0, -0.505178161726789, 0.9481005163065849, -0.3007477198660715, 0.41781097473064427], 1.0e-9));
    let b5, a5, f5 = scientific::cheby1(5, 0.1, [0.7], "lowpass");
    assert(f5.kind == scientific::F_NONE && rel_close(b5, [0.17948067861453934, 0.8974033930726967, 1.7948067861453934, 1.7948067861453934, 0.8974033930726967, 0.17948067861453934], 1.0e-9) && rel_close(a5, [1.0, 1.7414376751857006, 1.8227759057231772, 0.8819614268711546, 0.2897389232448219, 0.007467784640402062], 1.0e-9));
    let b6, a6, f6 = scientific::cheby1(1, 3.0, [0.45], "highpass");
    assert(f6.kind == scientific::F_NONE && rel_close(b6, [0.5399407391020689, -0.5399407391020689], 1.0e-9) && rel_close(a6, [1.0, -0.07988147820413756], 1.0e-9));
    let b7, a7, f7 = scientific::cheby2(3, 40.0, [0.3], "lowpass");
    assert(f7.kind == scientific::F_NONE && rel_close(b7, [0.01461536600299342, 0.00041761625757361667, 0.0004176162575736134, 0.014615366002993416], 1.0e-9) && rel_close(a7, [1.0, -2.321702622809876, 1.8559974074997827, -0.5042288201687728], 1.0e-9));
    let b8, a8, f8 = scientific::cheby2(4, 30.0, [0.4], "highpass");
    assert(f8.kind == scientific::F_NONE && rel_close(b8, [0.1505021756688369, -0.3718198623983622, 0.4963067784018436, -0.3718198623983621, 0.15050217566883686], 1.0e-9) && rel_close(a8, [1.0, 0.10364685365469449, 0.5852027782735715, -0.025502856720589447, 0.03389207319677507], 1.0e-9));
    let b9, a9, f9 = scientific::cheby2(3, 50.0, [0.2, 0.5], "bandpass");
    assert(f9.kind == scientific::F_NONE && rel_close(b9, [0.00514249360396887, -0.0077858241798205875, 0.0038201350823329743, 4.567451842491073e-18, -0.0038201350823329743, 0.007785824179820585, -0.005142493603968868], 1.0e-9) && rel_close(a9, [1.0, -2.818364416166289, 5.187010716938645, -5.623648867042587, 4.434974583401036, -2.058529439773886, 0.6245402561772314], 1.0e-9));
    let b10, a10, f10 = scientific::cheby2(2, 20.0, [0.3, 0.6], "bandstop");
    assert(f10.kind == scientific::F_NONE && rel_close(b10, [0.29524674227168485, -0.1835236263292223, 0.48702658603920024, -0.18352362632922226, 0.29524674227168474], 1.0e-9) && rel_close(a10, [1.0, -0.32378856362658914, -0.12357046778002556, -0.04325868903185556, 0.20109053836259566], 1.0e-9));
    let b11, a11, f11 = scientific::cheby2(5, 60.0, [0.7], "lowpass");
    assert(f11.kind == scientific::F_NONE && rel_close(b11, [0.049194124323051154, 0.19235279327054405, 0.34340848423181475, 0.34340848423181475, 0.19235279327054405, 0.04919412432305115], 1.0e-9) && rel_close(a11, [1.0, -0.5029273923375419, 0.7618762435242075, -0.16392355722891305, 0.07862322478958599, -0.0037377150965186034], 1.0e-9));
    let b12, a12, f12 = scientific::cheby2(1, 10.0, [0.45], "highpass");
    assert(f12.kind == scientific::F_NONE && rel_close(b12, [0.28072208013098765, -0.28072208013098765], 1.0e-9) && rel_close(a12, [1.0, 0.43855583973802476], 1.0e-9));
    let b13, a13, f13 = scientific::ellip(4, 1.0, 40.0, [0.3], "lowpass");
    assert(f13.kind == scientific::F_NONE && rel_close(b13, [0.035306770274680946, 0.023374875449924577, 0.05604962635018729, 0.02337487544992456, 0.03530677027468094], 1.0e-9) && rel_close(a13, [1.0, -2.320993290489171, 2.6771554868803458, -1.5773912625031947, 0.4158015600972941], 1.0e-9));
    let b14, a14, f14 = scientific::ellip(5, 0.5, 60.0, [0.45], "highpass");
    assert(f14.kind == scientific::F_NONE && rel_close(b14, [0.06993194407451027, -0.27751872250628495, 0.49909374860744404, -0.499093748607444, 0.27751872250628484, -0.06993194407451025], 1.0e-9) && rel_close(a14, [1.0, 0.25624027807340766, 1.172759615416179, 0.4739011876759278, 0.38282990420625396, 0.13235922349661844], 1.0e-9));
    let b15, a15, f15 = scientific::ellip(3, 2.0, 50.0, [0.2, 0.5], "bandpass");
    assert(f15.kind == scientific::F_NONE && rel_close(b15, [0.03522611320290015, -0.017269741116807104, -0.062985191660307, -2.150986307025158e-17, 0.06298519166030701, 0.017269741116807128, -0.03522611320290016], 1.0e-9) && rel_close(a15, [1.0, -2.4263240369165793, 3.8395750524417465, -3.930084225601358, 3.0959439809202056, -1.522337927306022, 0.502951502377697], 1.0e-9));
    let b16, a16, f16 = scientific::ellip(2, 1.0, 20.0, [0.3, 0.6], "bandstop");
    assert(f16.kind == scientific::F_NONE && rel_close(b16, [0.5438780275761547, -0.3725444803509182, 1.0995598931646398, -0.3725444803509183, 0.5438780275761549], 1.0e-9) && rel_close(a16, [1.0, -0.5130855072412996, 0.9957788996759761, -0.3229180567628307, 0.4584299597246405], 1.0e-9));
    let b17, a17, f17 = scientific::ellip(7, 0.5, 10.5, [0.16, 0.75], "bandstop");
    assert(f17.kind == scientific::F_NONE && rel_close(b17, [0.30990403872232686, -0.4791904455654947, 0.22707787913336458, -0.3087300561506312, 1.0552229673897062, -1.1741254780822639, 0.899437796098293, -0.7206490729032228, 0.8994377960982973, -1.1741254780822654, 1.0552229673897084, -0.30873005615062965, 0.22707787913336486, -0.4791904455654948, 0.30990403872232675], 1.0e-9) && rel_close(a17, [1.0, -1.2249844520555424, -0.9969407808024655, 0.36518626728992376, 4.043833974582704, -2.1380550654847985, -2.7006403588916865, -0.4102764005693502, 4.167742965376067, -0.6471425984530228, -1.4725906995276765, -0.5632735018442196, 1.067377858809695, -0.026195281382883892, -0.12549759685913986], 1.0e-9));
    let b18, a18, f18 = scientific::ellip(6, 3.0, 4.0, [0.14], "highpass");
    assert(f18.kind == scientific::F_NONE && rel_close(b18, [0.6889336371862705, -3.752650174992355, 8.88033987014951, -11.628946662850074, 8.880339870149509, -3.7526501749923535, 0.6889336371862702], 1.0e-9) && rel_close(a18, [1.0, -5.387728231129365, 12.631823169978034, -16.409389111480145, 12.445945381746682, -5.23035438595426, 0.9565182211777676], 1.0e-9));
    let b19, a19, f19 = scientific::ellip(1, 0.5, 10.5, [0.73], "lowpass");
    assert(f19.kind == scientific::F_NONE && rel_close(b19, [0.8637666058188674, 0.8637666058188674], 1.0e-9) && rel_close(a19, [1.0, 0.7275332116377349], 1.0e-9));
    let b20, a20, f20 = scientific::ellip(8, 2.0, 12.0, [0.6, 0.75], "bandstop");
    assert(f20.kind == scientific::F_NONE && rel_close(b20, [0.6360274762311089, 5.2115976610071755, 23.424014119327445, 72.61754726995133, 171.41841774743284, 323.0761358723434, 500.2728486810523, 646.5664566151211, 703.7227713116558, 646.5664566151211, 500.27284868105215, 323.0761358723434, 171.41841774743284, 72.6175472699513, 23.424014119327435, 5.211597661007174, 0.6360274762311089], 1.0e-9) && rel_close(a20, [1.0, 7.941844465336988, 34.5974913124384, 104.06111131674051, 238.57589056114907, 437.20814776120886, 659.0217436883831, 830.087138624647, 881.5563800952028, 791.302530630467, 598.9477343872356, 378.9245542545627, 197.2584045860479, 82.12621835265028, 26.083937483027942, 5.726031338865615, 0.6909014699724119], 1.0e-9));
    let _fb0, _fa0, ff0 = scientific::cheby1(0, 1.0, [0.3], "lowpass");
    assert(ff0.kind != scientific::F_NONE);
    let _fb1, _fa1, ff1 = scientific::cheby1(33, 1.0, [0.3], "lowpass");
    assert(ff1.kind != scientific::F_NONE);
    let _fb2, _fa2, ff2 = scientific::cheby1(3, 0.0, [0.3], "lowpass");
    assert(ff2.kind != scientific::F_NONE);
    let _fb3, _fa3, ff3 = scientific::cheby1(3, 1000000.0, [0.3], "lowpass");
    assert(ff3.kind != scientific::F_NONE);
    let _fb4, _fa4, ff4 = scientific::cheby1(3, 1.0, [0.3], "notch");
    assert(ff4.kind != scientific::F_NONE);
    let _fb5, _fa5, ff5 = scientific::cheby1(3, 1.0, [0.3, 0.5], "lowpass");
    assert(ff5.kind != scientific::F_NONE);
    let _fb6, _fa6, ff6 = scientific::cheby1(3, 1.0, [0.3], "bandpass");
    assert(ff6.kind != scientific::F_NONE);
    let _fb7, _fa7, ff7 = scientific::cheby1(3, 1.0, [0.0], "lowpass");
    assert(ff7.kind != scientific::F_NONE);
    let _fb8, _fa8, ff8 = scientific::cheby1(3, 1.0, [1.0], "lowpass");
    assert(ff8.kind != scientific::F_NONE);
    let _fb9, _fa9, ff9 = scientific::cheby1(3, 1.0, [0.5, 0.3], "bandpass");
    assert(ff9.kind != scientific::F_NONE);
    let _fb10, _fa10, ff10 = scientific::cheby2(0, 40.0, [0.3], "lowpass");
    assert(ff10.kind != scientific::F_NONE);
    let _fb11, _fa11, ff11 = scientific::cheby2(3, 0.0, [0.3], "lowpass");
    assert(ff11.kind != scientific::F_NONE);
    let _fb12, _fa12, ff12 = scientific::cheby2(3, 40.0, [1.5], "lowpass");
    assert(ff12.kind != scientific::F_NONE);
    let _fb13, _fa13, ff13 = scientific::cheby2(3, 40.0, [0.3], "notch");
    assert(ff13.kind != scientific::F_NONE);
    let _fb14, _fa14, ff14 = scientific::ellip(0, 1.0, 40.0, [0.3], "lowpass");
    assert(ff14.kind != scientific::F_NONE);
    let _fb15, _fa15, ff15 = scientific::ellip(3, 1.0, 1.0, [0.3], "lowpass");
    assert(ff15.kind != scientific::F_NONE);
    let _fb16, _fa16, ff16 = scientific::ellip(3, 2.0, 1.0, [0.3], "lowpass");
    assert(ff16.kind != scientific::F_NONE);
    let _fb17, _fa17, ff17 = scientific::ellip(3, 0.0, 40.0, [0.3], "lowpass");
    assert(ff17.kind != scientific::F_NONE);
    let _fb18, _fa18, ff18 = scientific::ellip(3, 1.0, 40.0, [0.3], "notch");
    assert(ff18.kind != scientific::F_NONE);
    let _fb19, _fa19, ff19 = scientific::ellip(3, 1.0, 40.0, [1.2], "lowpass");
    assert(ff19.kind != scientific::F_NONE);
    let rr0, ri0, pr0, pi0, k0, rf0 = scientific::residue([1.0, 2.0], [1.0, 5.0, 6.0], 0.001, "avg");
    assert(rf0.kind == scientific::F_NONE && res_match(rr0, ri0, pr0, pi0, [2.2204460492503116e-16, 0.9999999999999998], [0.0, 0.0], [-1.9999999999999998, -3.0000000000000004], [0.0, 0.0], 1.0e-8) && rel_close(k0, [], 1.0e-9));
    let rr1, ri1, pr1, pi1, k1, rf1 = scientific::residue([1.0, 0.0, 3.0], [1.0, 2.0, 5.0, 0.0], 0.001, "avg");
    assert(rf1.kind == scientific::F_NONE && res_match(rr1, ri1, pr1, pi1, [0.6000000000000001, 0.2, 0.2], [0.0, 0.4, -0.4], [0.0, -1.0, -1.0], [0.0, 2.0, -2.0], 1.0e-8) && rel_close(k1, [], 1.0e-9));
    let rr2, ri2, pr2, pi2, k2, rf2 = scientific::residue([1.0, 2.0], [1.0, 4.0, 4.0], 0.001, "avg");
    assert(rf2.kind == scientific::F_NONE && res_match(rr2, ri2, pr2, pi2, [1.0, 0.0], [0.0, 0.0], [-2.0, -2.0], [0.0, 0.0], 1.0e-8) && rel_close(k2, [], 1.0e-9));
    let rr3, ri3, pr3, pi3, k3, rf3 = scientific::residue([2.0, 1.0], [1.0, 6.0, 12.0, 8.0], 0.001, "avg");
    assert(rf3.kind == scientific::F_NONE && res_match(rr3, ri3, pr3, pi3, [0.0, 2.0, -2.9999999999999987], [0.0, 0.0, 0.0], [-1.9999999999999993, -1.9999999999999993, -1.9999999999999993], [0.0, 0.0, 0.0], 1.0e-8) && rel_close(k3, [], 1.0e-9));
    let rr4, ri4, pr4, pi4, k4, rf4 = scientific::residue([1.0, 0.0, 0.0, 1.0], [1.0, 5.0, 8.0, 4.0, 0.0], 0.001, "avg");
    assert(rf4.kind == scientific::F_NONE && res_match(rr4, ri4, pr4, pi4, [0.24999999999999978, 3.996802888650572e-15, 0.749999999999996, -3.5000000000000044], [0.0, 0.0, 0.0, 0.0], [0.0, -1.0000000000000013, -1.9999999999999996, -1.9999999999999996], [0.0, 0.0, 0.0, 0.0], 1.0e-8) && rel_close(k4, [], 1.0e-9));
    let rr5, ri5, pr5, pi5, k5, rf5 = scientific::residue([1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0], 0.001, "avg");
    assert(rf5.kind == scientific::F_NONE && res_match(rr5, ri5, pr5, pi5, [2.0, 2.0], [0.0, 0.0], [-1.0, -2.0], [0.0, 0.0], 1.0e-8) && rel_close(k5, [1.0, -1.0], 1.0e-9));
    let rr6, ri6, pr6, pi6, k6, rf6 = scientific::residue([3.0, 2.0], [1.0, 3.0, 2.0], 0.001, "avg");
    assert(rf6.kind == scientific::F_NONE && res_match(rr6, ri6, pr6, pi6, [-1.0, 4.0], [0.0, 0.0], [-1.0, -2.0], [0.0, 0.0], 1.0e-8) && rel_close(k6, [], 1.0e-9));
    let rr7, ri7, pr7, pi7, k7, rf7 = scientific::residue([1.0, 1.0], [1.0, 2.0, 3.0, 2.0, 1.0], 0.001, "avg");
    assert(rf7.kind == scientific::F_NONE && res_match(rr7, ri7, pr7, pi7, [8.539650136178704e-17, -0.16666666666666718, 8.539650136178704e-17, -0.16666666666666718], [-0.1924500897298761, -0.28867513459481325, 0.1924500897298761, 0.28867513459481325], [-0.4999999999999995, -0.4999999999999995, -0.4999999999999995, -0.4999999999999995], [0.8660254037844377, 0.8660254037844377, -0.8660254037844377, -0.8660254037844377], 1.0e-8) && rel_close(k7, [], 1.0e-9));
    let rr8, ri8, pr8, pi8, k8, rf8 = scientific::residue([3.0, 1.0], [2.0, 10.0, 12.0], 0.001, "avg");
    assert(rf8.kind == scientific::F_NONE && res_match(rr8, ri8, pr8, pi8, [-2.499999999999998, 3.9999999999999982], [0.0, 0.0], [-1.9999999999999998, -3.0000000000000004], [0.0, 0.0], 1.0e-8) && rel_close(k8, [], 1.0e-9));
    let rr9, ri9, pr9, pi9, k9, rf9 = scientific::residue([0.0, 0.0, 1.0, 2.0], [0.0, 1.0, 5.0, 6.0], 0.001, "avg");
    assert(rf9.kind == scientific::F_NONE && res_match(rr9, ri9, pr9, pi9, [2.2204460492503116e-16, 0.9999999999999998], [0.0, 0.0], [-1.9999999999999998, -3.0000000000000004], [0.0, 0.0], 1.0e-8) && rel_close(k9, [], 1.0e-9));
    let rr10, ri10, pr10, pi10, k10, rf10 = scientific::residue([0.0, 0.0], [1.0, 3.0, 2.0], 0.001, "avg");
    assert(rf10.kind == scientific::F_NONE && res_match(rr10, ri10, pr10, pi10, [0.0, 0.0], [0.0, 0.0], [-1.0, -2.0], [0.0, 0.0], 1.0e-8) && rel_close(k10, [], 1.0e-9));
    let rr11, ri11, pr11, pi11, k11, rf11 = scientific::residue([1.0, 2.0, 3.0], [2.0], 0.001, "avg");
    assert(rf11.kind == scientific::F_NONE && res_match(rr11, ri11, pr11, pi11, [], [], [], [], 1.0e-8) && rel_close(k11, [0.5, 1.0, 1.5], 1.0e-9));
    let rr12, ri12, pr12, pi12, k12, rf12 = scientific::residue([1.0, 2.0], [1.0, 2.0005, 1.0005], 0.001, "min");
    assert(rf12.kind == scientific::F_NONE && res_match(rr12, ri12, pr12, pi12, [1.0, 0.9994999999998027], [0.0, 0.0], [-1.0005000000001973, -1.0005000000001973], [0.0, 0.0], 1.0e-8) && rel_close(k12, [], 1.0e-9));
    let rr13, ri13, pr13, pi13, k13, rf13 = scientific::residue([1.0, 2.0], [1.0, 2.0005, 1.0005], 0.001, "max");
    assert(rf13.kind == scientific::F_NONE && res_match(rr13, ri13, pr13, pi13, [1.0, 1.0000000000001972], [0.0, 0.0], [-0.9999999999998027, -0.9999999999998027], [0.0, 0.0], 1.0e-8) && rel_close(k13, [], 1.0e-9));
    let rr14, ri14, pr14, pi14, k14, rf14 = scientific::residue([1.0, 2.0], [1.0, 2.0005, 1.0005], 1e-06, "avg");
    assert(rf14.kind == scientific::F_NONE && res_match(rr14, ri14, pr14, pi14, [1999.9999984218775, -1998.9999984218778], [0.0, 0.0], [-0.9999999999998027, -1.0005000000001973], [0.0, 0.0], 1.0e-8) && rel_close(k14, [], 1.0e-9));
    let _x0, _y0, _z0, _w0, _v0, rz0 = scientific::residue([1.0, 2.0], [0.0, 0.0]);
    assert(rz0.kind != scientific::F_NONE);
    let _x1, _y1, _z1, _w1, _v1, rz1 = scientific::residue([1.0, 2.0], [1.0, 3.0, 2.0], 0.001, "median");
    assert(rz1.kind != scientific::F_NONE);
    let _x2, _y2, _z2, _w2, _v2, rz2 = scientific::residue([1.0, 2.0], [1.0, 3.0, 2.0], -1.0);
    assert(rz2.kind != scientific::F_NONE);
    let _x3, _y3, _z3, _w3, _v3, rz3 = scientific::residue([1.0, 2.0], [1.0, 3.0, 2.0], 0.001 - 0.001 + 1.0 / 0.0);
    assert(rz3.kind != scientific::F_NONE);
    let _x4, _y4, _z4, _w4, _v4, rz4 = scientific::residue([1.0 / 0.0], [1.0, 3.0, 2.0]);
    assert(rz4.kind != scientific::F_NONE);
    let _x5, _y5, _z5, _w5, _v5, rz5 = scientific::residue([1.0, 2.0], [1.0, 0.0 / 0.0, 2.0]);
    assert(rz5.kind != scientific::F_NONE);
    var big: f64[>..] = [];
    var bi: i64 = 0;
    while bi < 300 { big.push(1.0); bi += 1; }
    let _x, _y, _z, _w, _v, rzl = scientific::residue([1.0], big);
    assert(rzl.kind == scientific::F_LIMIT);
}

// test_partial_fractions pins residuez, invres, invresz and unique_roots against SciPy.
fn test_partial_fractions() {
    let rr1, ri1, pr1, pi1, k1, f1 = scientific::residuez([1.0, 2.0], [1.0, -1.5, 0.5], 0.001, "avg");
    assert(f1.kind == scientific::F_NONE && res_match(rr1, ri1, pr1, pi1, [-5.0, 6.0], [0.0, 0.0], [0.5, 1.0], [0.0, 0.0], 1.0e-8) && rel_close(k1, [], 1.0e-9));
    let rr2, ri2, pr2, pi2, k2, f2 = scientific::residuez([1.0, 2.0, 3.0, 4.0], [1.0, -1.5, 0.5], 0.001, "avg");
    assert(f2.kind == scientific::F_NONE && res_match(rr2, ri2, pr2, pi2, [-49.0, 20.0], [0.0, 0.0], [0.5, 1.0], [0.0, 0.0], 1.0e-8) && rel_close(k2, [30.0, 8.0], 1.0e-9));
    let rr3, ri3, pr3, pi3, k3, f3 = scientific::residuez([1.0, 2.0], [1.0, -1.0, 0.5], 0.001, "avg");
    assert(f3.kind == scientific::F_NONE && res_match(rr3, ri3, pr3, pi3, [0.49999999999999956, 0.49999999999999956], [-2.499999999999999, 2.499999999999999], [0.4999999999999999, 0.4999999999999999], [0.49999999999999994, -0.49999999999999994], 1.0e-8) && rel_close(k3, [], 1.0e-9));
    let rr4, ri4, pr4, pi4, k4, f4 = scientific::residuez([1.0, 1.0], [1.0, -1.0, 0.25], 0.001, "avg");
    assert(f4.kind == scientific::F_NONE && res_match(rr4, ri4, pr4, pi4, [-2.0, 3.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.0], 1.0e-8) && rel_close(k4, [], 1.0e-9));
    let rr5, ri5, pr5, pi5, k5, f5 = scientific::residuez([1.0, 2.0, 0.0, 0.0], [1.0, -0.9, 0.2], 0.001, "avg");
    assert(f5.kind == scientific::F_NONE && res_match(rr5, ri5, pr5, pi5, [-24.0, 25.0], [0.0, 0.0], [0.4, 0.5], [0.0, 0.0], 1.0e-8) && rel_close(k5, [], 1.0e-9));
    let rr6, ri6, pr6, pi6, k6, f6 = scientific::residuez([0.0, 0.0], [1.0, -0.9, 0.2], 0.001, "avg");
    assert(f6.kind == scientific::F_NONE && res_match(rr6, ri6, pr6, pi6, [0.0, 0.0], [0.0, 0.0], [0.4, 0.5], [0.0, 0.0], 1.0e-8) && rel_close(k6, [], 1.0e-9));
    let rr7, ri7, pr7, pi7, k7, f7 = scientific::residuez([1.0, 2.0, 3.0], [2.0, -1.0, 0.1, 0.3], 0.001, "avg");
    assert(f7.kind == scientific::F_NONE && res_match(rr7, ri7, pr7, pi7, [1.354592544260534, -0.42729627213026716, -0.42729627213026716], [0.0, -2.4360749605750134, 2.4360749605750134], [-0.3845117293896666, 0.44225586469483313, 0.44225586469483313], [0.0, 0.4410384165535815, -0.4410384165535815], 1.0e-8) && rel_close(k7, [], 1.0e-9));
    let rr8, ri8, pr8, pi8, k8, f8 = scientific::residuez([1.0], [1.0, 0.2, -0.3, -0.1], 0.001, "avg");
    assert(f8.kind == scientific::F_NONE && res_match(rr8, ri8, pr8, pi8, [0.32267775759385237, 0.32267775759385237, 0.3546444848122946], [0.7023378652688193, -0.7023378652688193, 0.0], [-0.3960164491623792, -0.3960164491623792, 0.5920328983247589], [0.10991134771526789, -0.10991134771526789, 0.0], 1.0e-8) && rel_close(k8, [], 1.0e-9));
    let rr9, ri9, pr9, pi9, k9, f9 = scientific::residuez([1.0, 2.0], [1.0, -1.0, 0.25], 0.001, "max");
    assert(f9.kind == scientific::F_NONE && res_match(rr9, ri9, pr9, pi9, [-4.0, 5.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.0], 1.0e-8) && rel_close(k9, [], 1.0e-9));
    let ib10, ia10, if10 = scientific::invres([2.2204460492503116e-16, 0.9999999999999998], [0.0, 0.0], [-1.9999999999999998, -3.0000000000000004], [0.0, 0.0], []);
    assert(if10.kind == scientific::F_NONE && rel_close(ib10, [1.0, 2.0], 1.0e-7) && rel_close(ia10, [1.0, 5.0, 6.0], 1.0e-7));
    let ib11, ia11, if11 = scientific::invres([0.6000000000000001, 0.2, 0.2], [0.0, 0.4, -0.4], [0.0, -1.0, -1.0], [0.0, 2.0, -2.0], []);
    assert(if11.kind == scientific::F_NONE && rel_close(ib11, [1.0, 0.0, 3.0000000000000004], 1.0e-7) && rel_close(ia11, [1.0, 2.0, 5.0, 0.0], 1.0e-7));
    let ib12, ia12, if12 = scientific::invres([1.0, 0.0], [0.0, 0.0], [-2.0, -2.0], [0.0, 0.0], []);
    assert(if12.kind == scientific::F_NONE && rel_close(ib12, [1.0, 2.0], 1.0e-7) && rel_close(ia12, [1.0, 4.0, 4.0], 1.0e-7));
    let ib13, ia13, if13 = scientific::invres([0.0, 2.0, -2.9999999999999987], [0.0, 0.0, 0.0], [-1.9999999999999993, -1.9999999999999993, -1.9999999999999993], [0.0, 0.0, 0.0], []);
    assert(if13.kind == scientific::F_NONE && rel_close(ib13, [0.0, 2.0, 1.0], 1.0e-7) && rel_close(ia13, [1.0, 5.999999999999998, 11.999999999999993, 7.999999999999992], 1.0e-7));
    let ib14, ia14, if14 = scientific::invres([0.24999999999999978, 3.996802888650572e-15, 0.749999999999996, -3.5000000000000044], [0.0, 0.0, 0.0, 0.0], [0.0, -1.0000000000000013, -1.9999999999999996, -1.9999999999999996], [0.0, 0.0, 0.0, 0.0], []);
    assert(if14.kind == scientific::F_NONE && rel_close(ib14, [0.9999999999999998, -8.881784197001252e-16, -4.440892098500626e-16, 1.0], 1.0e-7) && rel_close(ia14, [1.0, 5.0, 8.000000000000002, 4.0000000000000036, 0.0], 1.0e-7));
    let ib15, ia15, if15 = scientific::invres([2.0, 2.0], [0.0, 0.0], [-1.0, -2.0], [0.0, 0.0], [1.0, -1.0]);
    assert(if15.kind == scientific::F_NONE && rel_close(ib15, [1.0, 2.0, 3.0, 4.0], 1.0e-7) && rel_close(ia15, [1.0, 3.0, 2.0], 1.0e-7));
    let ib16, ia16, if16 = scientific::invres([8.539650136178704e-17, -0.16666666666666718, 8.539650136178704e-17, -0.16666666666666718], [-0.1924500897298761, -0.28867513459481325, 0.1924500897298761, 0.28867513459481325], [-0.4999999999999995, -0.4999999999999995, -0.4999999999999995, -0.4999999999999995], [0.8660254037844377, 0.8660254037844377, -0.8660254037844377, -0.8660254037844377], []);
    assert(if16.kind == scientific::F_NONE && rel_close(ib16, [1.707930027235741e-16, 3.3306690738754696e-16, 1.0000000000000004, 1.0000000000000002], 1.0e-7) && rel_close(ia16, [1.0, 1.999999999999998, 2.999999999999994, 1.9999999999999936, 0.9999999999999958], 1.0e-7));
    let zb17, za17, zf17 = scientific::invresz([-5.0, 6.0], [0.0, 0.0], [0.5, 1.0], [0.0, 0.0], []);
    assert(zf17.kind == scientific::F_NONE && rel_close(zb17, [1.0, 2.0], 1.0e-7) && rel_close(za17, [1.0, -1.5, 0.5], 1.0e-7));
    let zb18, za18, zf18 = scientific::invresz([-49.0, 20.0], [0.0, 0.0], [0.5, 1.0], [0.0, 0.0], [30.0, 8.0]);
    assert(zf18.kind == scientific::F_NONE && rel_close(zb18, [1.0, 2.0, 3.0, 4.0], 1.0e-7) && rel_close(za18, [1.0, -1.5, 0.5], 1.0e-7));
    let zb19, za19, zf19 = scientific::invresz([0.49999999999999956, 0.49999999999999956], [-2.499999999999999, 2.499999999999999], [0.4999999999999999, 0.4999999999999999], [0.49999999999999994, -0.49999999999999994], []);
    assert(zf19.kind == scientific::F_NONE && rel_close(zb19, [0.9999999999999991, 1.9999999999999991], 1.0e-7) && rel_close(za19, [1.0, -0.9999999999999998, 0.49999999999999983], 1.0e-7));
    let zb20, za20, zf20 = scientific::invresz([-2.0, 3.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.0], []);
    assert(zf20.kind == scientific::F_NONE && rel_close(zb20, [1.0, 1.0], 1.0e-7) && rel_close(za20, [1.0, -1.0, 0.25], 1.0e-7));
    let zb21, za21, zf21 = scientific::invresz([1.354592544260534, -0.42729627213026716, -0.42729627213026716], [0.0, -2.4360749605750134, 2.4360749605750134], [-0.3845117293896666, 0.44225586469483313, 0.44225586469483313], [0.0, 0.4410384165535815, -0.4410384165535815], []);
    assert(zf21.kind == scientific::F_NONE && rel_close(zb21, [0.4999999999999997, 0.9999999999999984, 1.4999999999999976], 1.0e-7) && rel_close(za21, [1.0, -0.49999999999999967, 0.049999999999999656, 0.14999999999999972], 1.0e-7));
    let ur22, ui22, um22, uf22 = scientific::unique_roots([0.0, 1.3, 1.31, 2.8, 1.25, 2.2, 10.3], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.02, "avg");
    assert(uf22.kind == scientific::F_NONE && rel_close(ur22, [0.0, 1.3050000000000002, 2.8, 1.25, 2.2, 10.3], 1.0e-12) && rel_close(ui22, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0e-12) && rel_close(um22, [1.0, 2.0, 1.0, 1.0, 1.0, 1.0], 0.0));
    let ur23, ui23, um23, uf23 = scientific::unique_roots([0.0, 1.3, 1.31, 2.8, 1.25, 2.2, 10.3], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.02, "min");
    assert(uf23.kind == scientific::F_NONE && rel_close(ur23, [0.0, 1.3, 2.8, 1.25, 2.2, 10.3], 1.0e-12) && rel_close(ui23, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0e-12) && rel_close(um23, [1.0, 2.0, 1.0, 1.0, 1.0, 1.0], 0.0));
    let ur24, ui24, um24, uf24 = scientific::unique_roots([0.0, 1.3, 1.31, 2.8, 1.25, 2.2, 10.3], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.02, "max");
    assert(uf24.kind == scientific::F_NONE && rel_close(ur24, [0.0, 1.31, 2.8, 1.25, 2.2, 10.3], 1.0e-12) && rel_close(ui24, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0e-12) && rel_close(um24, [1.0, 2.0, 1.0, 1.0, 1.0, 1.0], 0.0));
    let ur25, ui25, um25, uf25 = scientific::unique_roots([1.0, 1.0, 2.0, 2.0, 2.0004], [1.0, 1.0005, 0.0, -1.0, -1.0], 0.001, "avg");
    assert(uf25.kind == scientific::F_NONE && rel_close(ur25, [1.0, 2.0, 2.0002], 1.0e-12) && rel_close(ui25, [1.0002499999999999, 0.0, -1.0], 1.0e-12) && rel_close(um25, [2.0, 1.0, 2.0], 0.0));
    let ur26, ui26, um26, uf26 = scientific::unique_roots([3.0, 3.0, 3.0, 1.0], [0.0, 0.0, 0.0, 0.0], 0.001, "min");
    assert(uf26.kind == scientific::F_NONE && rel_close(ur26, [3.0, 1.0], 1.0e-12) && rel_close(ui26, [0.0, 0.0], 1.0e-12) && rel_close(um26, [3.0, 1.0], 0.0));
    let _a0, _b0, _c0, _d0, _e0, zz0 = scientific::residuez([1.0, 2.0], [0.0, 0.0]);
    assert(zz0.kind != scientific::F_NONE);
    let _a1, _b1, _c1, _d1, _e1, zz1 = scientific::residuez([1.0, 2.0], [0.0, 1.0, 2.0]);
    assert(zz1.kind != scientific::F_NONE);
    let _a2, _b2, _c2, _d2, _e2, zz2 = scientific::residuez([1.0], [1.0, 2.0], 0.001, "median");
    assert(zz2.kind != scientific::F_NONE);
    let _a3, _b3, _c3, _d3, _e3, zz3 = scientific::residuez([1.0], [1.0, 2.0], -1.0);
    assert(zz3.kind != scientific::F_NONE);
    let _a4, _b4, _c4, _d4, _e4, zz4 = scientific::residuez([0.0 / 0.0], [1.0, 2.0]);
    assert(zz4.kind != scientific::F_NONE);
    let _a5, _b5, zz5 = scientific::invres([1.0], [0.0], [1.0, 2.0], [0.0, 0.0], []);
    assert(zz5.kind != scientific::F_NONE);
    let _a6, _b6, zz6 = scientific::invres([1.0], [0.0], [], [], []);
    assert(zz6.kind != scientific::F_NONE);
    let _a7, _b7, zz7 = scientific::invres([1.0], [0.0], [1.0], [0.0], [], 0.001, "median");
    assert(zz7.kind != scientific::F_NONE);
    let _a8, _b8, zz8 = scientific::invres([1.0], [0.0], [1.0], [0.0], [], -1.0);
    assert(zz8.kind != scientific::F_NONE);
    let _a9, _b9, zz9 = scientific::invres([1.0], [0.0], [1.0], [0.0 / 0.0], []);
    assert(zz9.kind != scientific::F_NONE);
    let _a10, _b10, zz10 = scientific::invres([1.0], [1.0], [1.0], [0.0], []);
    assert(zz10.kind != scientific::F_NONE);
    let _a11, _b11, zz11 = scientific::invresz([1.0], [0.0], [1.0, 2.0], [0.0, 0.0], []);
    assert(zz11.kind != scientific::F_NONE);
    let _a12, _b12, zz12 = scientific::invresz([1.0], [1.0], [1.0], [0.0], []);
    assert(zz12.kind != scientific::F_NONE);
    let _u0, _v0, _w0, uz0 = scientific::unique_roots([1.0], []);
    assert(uz0.kind != scientific::F_NONE);
    let _u1, _v1, _w1, uz1 = scientific::unique_roots([1.0], [0.0], 0.001, "median");
    assert(uz1.kind != scientific::F_NONE);
    let _u2, _v2, _w2, uz2 = scientific::unique_roots([1.0], [0.0], -1.0);
    assert(uz2.kind != scientific::F_NONE);
    let _u3, _v3, _w3, uz3 = scientific::unique_roots([0.0 / 0.0], [0.0]);
    assert(uz3.kind != scientific::F_NONE);
}

// test_minimize_grad pins the methods that use the gradient against
// SciPy. The blocks give the value and the gradient of the function of
// Rosenbrock and of a quadratic.
fn test_minimize_grad() {
    // The constrained minimization.
    let mx, mv, mf = scientific::minimize_constrained([2.0, 2.0], 1) { p => [p[0] * p[0] + p[1] * p[1], 2.0 * p[0], 2.0 * p[1], p[0] + p[1] - 1.0] };
    assert(mf.kind == scientific::F_NONE && rel_close(mx, [0.5, 0.5], 1.0e-6) && mathshim_abs(mv.data[0] - 0.5) < 1.0e-6);
    let my, myv, myf = scientific::minimize_constrained([2.0, 2.0], 0) { p => [(p[0] - 1.0) * (p[0] - 1.0) + (p[1] - 2.5) * (p[1] - 2.5), 2.0 * (p[0] - 1.0), 2.0 * (p[1] - 2.5), p[0] - 2.0 * p[1] + 2.0, -p[0] - 2.0 * p[1] + 6.0, -p[0] + 2.0 * p[1] + 2.0] };
    assert(myf.kind == scientific::F_NONE && rel_close(my, [1.4, 1.7], 1.0e-6) && mathshim_abs(myv.data[0] - 0.8) < 1.0e-6);
    let mz, mzv, mzf = scientific::minimize_constrained([1.0, 0.0], 1, [0.8, -5.0], [5.0, 5.0]) { p => [p[0] * p[0] + p[1] * p[1], 2.0 * p[0], 2.0 * p[1], p[0] + p[1] - 1.0] };
    assert(mzf.kind == scientific::F_NONE && rel_close(mz, [0.8, 0.2], 1.0e-6));
    let mb, mbv, mbf = scientific::minimize_constrained([1.0, 0.0], 3) { p => [p[0] * p[0], 2.0 * p[0], 0.0, p[1]] };
    assert(mbf.kind == scientific::F_SHAPE);
    let mc, mcv, mcf = scientific::minimize_constrained([1.0, 0.0], 0, [0.0], [1.0]) { p => [p[0] * p[0], 2.0 * p[0], 0.0] };
    assert(mcf.kind == scientific::F_SHAPE);
    // The method newton-cg.
    let nc, ncv, ncf = scientific::minimize_grad([-1.2, 1.0, 0.8], "newton-cg", false, [], [], 1.0e-8, 1.0e-8, 1.0e-8, 200) { v =>
        var s: f64 = 0.0;
        var g: f64[>..] = [0.0, 0.0, 0.0];
        var j: i64 = 0;
        while j + 1 < v.len {
            let d = v[j+1] - v[j]*v[j];
            s += 100.0 * d * d + (1.0 - v[j]) * (1.0 - v[j]);
            g[j] += -400.0 * v[j] * d - 2.0 * (1.0 - v[j]);
            g[j+1] += 200.0 * d;
            j += 1;
        }
        var o: f64[>..] = [s];
        j = 0;
        while j < g.len { o.push(g[j]); j += 1; }
        o
    };
    assert(ncf.kind == scientific::F_NONE && rel_close(nc, [1.0, 1.0, 1.0], 1.0e-6) && ncv.data[0] < 1.0e-10);
    let nd, ndv, ndf = scientific::minimize_grad([0.0, 1.0], "newton-cg", false, [], [], 1.0e-8, 1.0e-8, 1.0e-8, 200) { v =>
        let a = v[0] - 3.0;
        [a * a * a * a + (v[0] + v[1]) * (v[0] + v[1]) + v[1] * v[1] * v[1] * v[1] / 5.0, 4.0 * a * a * a + 2.0 * (v[0] + v[1]), 2.0 * (v[0] + v[1]) + 4.0 * v[1] * v[1] * v[1] / 5.0]
    };
    assert(ndf.kind == scientific::F_NONE && rel_close(nd, [2.2299597097408763, -1.3167503728453185], 1.0e-5));
    let ne, nev, nef = scientific::minimize_grad([0.0, 1.0], "newton-cg", true, [0.0, 0.0], [1.0, 1.0]) { v => [v[0] * v[0] + v[1] * v[1], 2.0 * v[0], 2.0 * v[1]] };
    assert(nef.kind == scientific::F_VALUE);
    let a, av, af = scientific::minimize_grad([-1.2, 1.0], "bfgs", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v =>
        var s: f64 = 0.0;
        var i: i64 = 0;
        while i + 1 < v.len { s += 100.0 * (v[i+1] - v[i]*v[i]) * (v[i+1] - v[i]*v[i]) + (1.0 - v[i]) * (1.0 - v[i]); i += 1; }
        var g: f64[>..] = [];
        var j: i64 = 0;
        while j < v.len { g.push(0.0); j += 1; }
        j = 0;
        while j + 1 < v.len {
            let d = v[j+1] - v[j]*v[j];
            g[j] += -400.0 * v[j] * d - 2.0 * (1.0 - v[j]);
            g[j+1] += 200.0 * d;
            j += 1;
        }
        var o: f64[>..] = [];
        o.push(s);
        j = 0;
        while j < g.len { o.push(g[j]); j += 1; }
        o
    };
    assert(af.kind == scientific::F_NONE && rel_close(a, [1.0, 1.0], 1.0e-6) && av.data[0] < 1.0e-12);
    let b, bv, bf = scientific::minimize_grad([-1.2, 1.0], "cg", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v =>
        var s: f64 = 0.0;
        var i: i64 = 0;
        while i + 1 < v.len { s += 100.0 * (v[i+1] - v[i]*v[i]) * (v[i+1] - v[i]*v[i]) + (1.0 - v[i]) * (1.0 - v[i]); i += 1; }
        var g: f64[>..] = [];
        var j: i64 = 0;
        while j < v.len { g.push(0.0); j += 1; }
        j = 0;
        while j + 1 < v.len {
            let d = v[j+1] - v[j]*v[j];
            g[j] += -400.0 * v[j] * d - 2.0 * (1.0 - v[j]);
            g[j+1] += 200.0 * d;
            j += 1;
        }
        var o: f64[>..] = [];
        o.push(s);
        j = 0;
        while j < g.len { o.push(g[j]); j += 1; }
        o
    };
    assert(bf.kind == scientific::F_NONE && rel_close(b, [1.0, 1.0], 1.0e-6) && bv.data[0] < 1.0e-12);
    let c, cv, cf = scientific::minimize_grad([-1.2, 1.0], "lbfgsb", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v =>
        var s: f64 = 0.0;
        var i: i64 = 0;
        while i + 1 < v.len { s += 100.0 * (v[i+1] - v[i]*v[i]) * (v[i+1] - v[i]*v[i]) + (1.0 - v[i]) * (1.0 - v[i]); i += 1; }
        var g: f64[>..] = [];
        var j: i64 = 0;
        while j < v.len { g.push(0.0); j += 1; }
        j = 0;
        while j + 1 < v.len {
            let d = v[j+1] - v[j]*v[j];
            g[j] += -400.0 * v[j] * d - 2.0 * (1.0 - v[j]);
            g[j+1] += 200.0 * d;
            j += 1;
        }
        var o: f64[>..] = [];
        o.push(s);
        j = 0;
        while j < g.len { o.push(g[j]); j += 1; }
        o
    };
    assert(cf.kind == scientific::F_NONE && rel_close(c, [1.0, 1.0], 1.0e-6) && cv.data[0] < 1.0e-12);
    // The quadratic has the point [4, -2] and the value -6.
    let d, dv, df = scientific::minimize_grad([0.0, 0.0], "bfgs", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v =>
        var o: f64[>..] = [];
        o.push((v[0] - 3.0) * (v[0] - 3.0) + 4.0 * (v[1] + 1.5) * (v[1] + 1.5) + v[0] * v[1]);
        o.push(2.0 * (v[0] - 3.0) + v[1]);
        o.push(8.0 * (v[1] + 1.5) + v[0]);
        o
    };
    assert(df.kind == scientific::F_NONE && rel_close(d, [4.0, -2.0], 1.0e-9) && rel_close(dv, [-6.0], 1.0e-9));
    // The bounds keep the point inside. SciPy replaces the method BFGS
    // with L-BFGS-B when bounds are given, and the package does the same.
    let e, ev, ef = scientific::minimize_grad([0.0, 0.0], "bfgs", true, [-1.0, -2.0], [2.0, 2.0], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v =>
        var o: f64[>..] = [];
        o.push((v[0] - 3.0) * (v[0] - 3.0) + 4.0 * (v[1] + 1.5) * (v[1] + 1.5) + v[0] * v[1]);
        o.push(2.0 * (v[0] - 3.0) + v[1]);
        o.push(8.0 * (v[1] + 1.5) + v[0]);
        o
    };
    assert(ef.kind == scientific::F_NONE && rel_close(e, [2.0, -1.75], 1.0e-4) && rel_close(ev, [-2.25], 1.0e-6));
    // A start outside the bounds moves to the border.
    let g2, gv2, gf2 = scientific::minimize_grad([5.0, 0.0], "lbfgsb", true, [-1.0, -2.0], [2.0, 2.0], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v =>
        var o: f64[>..] = [];
        o.push((v[0] - 3.0) * (v[0] - 3.0) + 4.0 * (v[1] + 1.5) * (v[1] + 1.5) + v[0] * v[1]);
        o.push(2.0 * (v[0] - 3.0) + v[1]);
        o.push(8.0 * (v[1] + 1.5) + v[0]);
        o
    };
    assert(gf2.kind == scientific::F_NONE && rel_close(g2, [2.0, -1.75], 1.0e-4));
    // The faults.
    let h1, h1v, h1f = scientific::minimize_grad([0.0, 0.0], "nelder-mead", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v => [v[0] * v[0]] };
    assert(h1f.kind != scientific::F_NONE);
    let h2, h2v, h2f = scientific::minimize_grad([], "bfgs", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v => [v[0] * v[0]] };
    assert(h2f.kind != scientific::F_NONE);
    let h3, h3v, h3f = scientific::minimize_grad([0.0], "bfgs", false, [], [], 0.0, 1.0e-12, 1.0e-12, 500) { v => [v[0] * v[0]] };
    assert(h3f.kind != scientific::F_NONE);
    let h4, h4v, h4f = scientific::minimize_grad([0.0], "bfgs", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 0) { v => [v[0] * v[0]] };
    assert(h4f.kind != scientific::F_NONE);
    let h5, h5v, h5f = scientific::minimize_grad([0.0, 0.0], "bfgs", true, [0.0], [1.0], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v => [v[0] * v[0]] };
    assert(h5f.kind != scientific::F_NONE);
    let h6, h6v, h6f = scientific::minimize_grad([0.0, 0.0], "bfgs", true, [1.0, 0.0], [0.0, 1.0], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v => [v[0] * v[0]] };
    assert(h6f.kind != scientific::F_NONE);
    let h7, h7v, h7f = scientific::minimize_grad([0.0, 0.0], "bfgs", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v => [v[0] * v[0]] };
    assert(h7f.kind != scientific::F_NONE);
}


// test_toms748 pins the method of Alefeld, Potra and Shi against
// SciPy.
fn test_toms748() {
    let t1, t1f = scientific::toms748(0.0, 2.0) { x => x * x - 2.0 };
    assert(t1f.kind == scientific::F_NONE && close(t1, 1.414213562373095, 1e-14));
    let t2, t2f = scientific::toms748(-3.0, 0.0) { x => x * x * x - 2.0 * x + 2.0 };
    assert(t2f.kind == scientific::F_NONE && close(t2, -1.7692923542386314, 1e-14));
    let t3, t3f = scientific::toms748(0.0, 1.0) { x => cos(x) - x };
    assert(t3f.kind == scientific::F_NONE && close(t3, 0.7390851332151607, 1e-14));
    let t4, t4f = scientific::toms748(-2.0, 2.0) { x => exp(x) - 2.0 };
    assert(t4f.kind == scientific::F_NONE && close(t4, 0.6931471805599453, 1e-14));
    let t5, t5f = scientific::toms748(0.0, 2.0) { x => x * x * x * x * x - 2.0 * x * x * x + x - 1.0 };
    assert(t5f.kind == scientific::F_NONE && close(t5, 1.3625985776649348, 1e-14));
    // The count of steps of Newton type changes the path, not the root.
    let t6, t6f = scientific::toms748(0.0, 2.0, 2.0e-12, 8.881784197001252e-16, 100, 1) { x => x * x - 2.0 };
    assert(t6f.kind == scientific::F_NONE && close(t6, 1.414213562373095, 1e-14));
    let t7, t7f = scientific::toms748(0.0, 2.0, 2.0e-12, 8.881784197001252e-16, 100, 3) { x => x * x - 2.0 };
    assert(t7f.kind == scientific::F_NONE && close(t7, 1.414213562373095, 1e-14));
    // The faults.
    let b1, b1f = scientific::toms748(0.0, 2.0, 0.0) { x => x * x - 2.0 };
    assert(b1f.kind != scientific::F_NONE);
    let b2, b2f = scientific::toms748(0.0, 2.0, 2.0e-12, -1.0) { x => x * x - 2.0 };
    assert(b2f.kind != scientific::F_NONE);
    let b3, b3f = scientific::toms748(0.0, 2.0, 2.0e-12, 8.881784197001252e-16, 0) { x => x * x - 2.0 };
    assert(b3f.kind != scientific::F_NONE);
    let b4, b4f = scientific::toms748(0.0, 2.0) { x => x * x + 1.0 };
    assert(b4f.kind != scientific::F_NONE);
    let b5, b5f = scientific::toms748(0.0, 2.0) { x => 0.0 / 0.0 };
    assert(b5f.kind != scientific::F_NONE);
}


fn main() {
    test_schur_funm_ldl();
    test_eigvalsh_larger();
    test_eigsh();
    test_lsqr_minres();
    test_sparse_structure();
    test_differential_evolution();
    test_istft();
    test_stft();
    test_spectral();
    test_firwin2();
    test_decimate();
    test_resample_poly();
    test_sosfreqz();
    test_interp1d_kinds();
    test_matrix_trig();
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
    test_products();
    test_scilinalg();
    test_transforms();
    test_spatial();
    test_interp();
    test_quad();
    test_programs();
    test_dsp();
    test_sparse();
    test_matfun();
    test_carray();
    test_pad_stats();
    test_dctn();
    test_dblquad();
    test_curve_fit();
    test_root();
    test_windows_more();
    test_dpss();
    test_ivp_t_eval();
    test_dop853();
    test_first_step();
    test_bdf();
    test_radau();
    test_fht();
    test_spline_mixed_ends();
    test_interp_spline_bc();
    test_bspline_design_matrix();
    test_bessel();
    test_cumulative_simpson_initial();
    test_bspline();
    test_spline_ends();
    test_cubic_hermite();
    test_global_minimizers();
    test_interpn();
    test_cheby1();
    test_hfftn();
    test_sparse_join();
    test_krylov();
    test_odeint();
    test_root_scalar();
    test_cheby2();
    test_bins_and_weights();
    test_cumulative_simpson();
    test_tplquad();
    test_nquad();
    test_remez();
    test_cheby_ellip_residue();
    test_partial_fractions();
    test_minimize_grad();
    test_toms748();
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
