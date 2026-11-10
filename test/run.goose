import scientific from "..";

// close tells whether two numbers agree to a small tolerance.
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

fn main() {
    test_create();
    test_shape();
    test_ops();
    test_reduce();
    test_axis_and_permute();
    print("ok\n");
    return;
}
