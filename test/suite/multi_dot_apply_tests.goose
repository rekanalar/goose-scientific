// The tests of the chain product and the two calls. Run them with
// SCIENTIFIC_SUITES=multi_dot_apply.
//
// The product of a chain of matrices, the call over a list of axes and
// the call over the values match the pins of NumPy 2.5.4.
namespace multi_dot_apply_tests;

import scientific from "../..";
import common;
import multi_dot_apply_pins;

// check_pin tells whether an array holds the shape and the values of a
// pinned case.
fn check_pin(got: scientific::Arr&, d: f64[:], lens: f64[:], i: i64, sd: f64[:], slens: f64[:]) {
    let want = common::part_of(d, lens, i);
    let wshape = common::shape_of(sd, slens, i);
    assert(common::same_shape(got, wshape));
    assert(got.data.len == want.len);
    var j: i64 = 0;
    while j < want.len {
        assert(common::close(got.data[j], want[j], 1e-9));
        j += 1;
    }
}

fn test_multi_dot_pins() {
    let cases = multi_dot_apply_pins::CH_N;
    var ci: i64 = 0;
    var ai: i64 = 0;
    while ci < cases.len {
        let count = cases[ci] as i64;
        var c = scientific::chain_new();
        var k: i64 = 0;
        while k < count {
            let shape = common::shape_of(multi_dot_apply_pins::CH_SH_D, multi_dot_apply_pins::CH_SH_L, ai);
            let values = common::part_of(multi_dot_apply_pins::CH_VAL_D, multi_dot_apply_pins::CH_VAL_L, ai);
            let next, f = scientific::chain_add(c, common::arr_of(shape, values));
            assert(f.kind == scientific::F_NONE);
            c = copy(next);
            ai += 1;
            k += 1;
        }
        let got, g = scientific::multi_dot(c);
        assert(g.kind == scientific::F_NONE);
        check_pin(got, multi_dot_apply_pins::CH_OUT_D, multi_dot_apply_pins::CH_OUT_L, ci,
                  multi_dot_apply_pins::CH_OUT_SH_D, multi_dot_apply_pins::CH_OUT_SH_L);
        ci += 1;
    }
}

fn test_multi_dot_shapes() {
    // A chain of two matrices gives a matrix.
    let v, f = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let m, f2 = scientific::reshape(v, [2, 3]);
    let w, f3 = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let n, f4 = scientific::reshape(w, [3, 2]);
    var c = scientific::chain_new();
    let c1, g1 = scientific::chain_add(c, copy(m));
    let c2, g2 = scientific::chain_add(c1, copy(n));
    let got, g = scientific::multi_dot(c2);
    assert(g.kind == scientific::F_NONE);
    assert(got.ndim == 2);
    assert(got.shape[0] == 2 && got.shape[1] == 2);
    assert(common::close(got.data[0], 22.0, 1e-9));
    assert(common::close(got.data[1], 28.0, 1e-9));
    assert(common::close(got.data[2], 49.0, 1e-9));
    assert(common::close(got.data[3], 64.0, 1e-9));
    // A chain of two vectors gives one element.
    let p, f5 = scientific::from_values([1.0, 2.0, 3.0]);
    let q, f6 = scientific::from_values([4.0, 5.0, 6.0]);
    var c2 = scientific::chain_new();
    let c3, g3 = scientific::chain_add(c2, copy(p));
    let c4, g4 = scientific::chain_add(c3, copy(q));
    let one, g5 = scientific::multi_dot(c4);
    assert(g5.kind == scientific::F_NONE);
    assert(one.ndim == 1);
    assert(one.shape[0] == 1);
    assert(common::close(one.data[0], 32.0, 1e-9));
}

fn test_multi_dot_faults() {
    let v, f = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let m, f2 = scientific::reshape(v, [2, 3]);
    // A chain of one array.
    var c0 = scientific::chain_new();
    let c1, g1 = scientific::chain_add(c0, copy(m));
    let r1, k1 = scientific::multi_dot(c1);
    assert(k1.kind == scientific::F_SHAPE);
    // An array of three dimensions.
    let t, f3 = scientific::reshape(v, [2, 3, 1]);
    let c2, g2 = scientific::chain_add(c0, copy(t));
    assert(g2.kind == scientific::F_SHAPE);
    // A vector in the middle of the chain.
    let p, f4 = scientific::from_values([1.0, 2.0, 3.0]);
    let q, f4b = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let o, f4c = scientific::reshape(q, [3, 2]);
    var c3 = scientific::chain_new();
    let c4, g3 = scientific::chain_add(c3, copy(m));
    let c5, g4 = scientific::chain_add(c4, copy(p));
    let c5b, g4b = scientific::chain_add(c5, copy(o));
    let r2, k2 = scientific::multi_dot(c5b);
    assert(k2.kind == scientific::F_SHAPE);
    // Arrays that do not align.
    let w, f5 = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    let n, f6 = scientific::reshape(w, [2, 2]);
    var c6 = scientific::chain_new();
    let c7, g5 = scientific::chain_add(c6, copy(m));
    let c8, g6 = scientific::chain_add(c7, copy(n));
    let r3, k3 = scientific::multi_dot(c8);
    assert(k3.kind == scientific::F_SHAPE);
    // A value that is not finite.
    let zero: f64 = 0.0;
    let inf: f64 = 1.0 / zero;
    let bad = common::arr_of([2, 2], [1.0, inf, 0.0, 1.0]);
    let c9, g7 = scientific::chain_add(c0, copy(bad));
    assert(g7.kind == scientific::F_VALUE);
    // A chain that does not match its own lists.
    let forged = scientific::Chain { count: 2, nd: [2], rows: [2], cols: [2], off: [0], data: [1.0] };
    let r4, k4 = scientific::multi_dot(forged);
    assert(k4.kind == scientific::F_SHAPE);
    assert(r1.data.len == 0 && r2.data.len == 0 && r3.data.len == 0 && r4.data.len == 0);
}

fn test_multi_dot_limit() {
    // The chain holds at most 256 arrays.
    let one, f = scientific::from_values([1.0]);
    var c = scientific::chain_new();
    var i: i64 = 0;
    while i < 256 {
        let next, g = scientific::chain_add(c, copy(one));
        assert(g.kind == scientific::F_NONE);
        c = copy(next);
        i += 1;
    }
    let over, g = scientific::chain_add(c, copy(one));
    assert(g.kind == scientific::F_LIMIT);
    assert(over.count == 256);
}

fn test_apply_over_axes_pins() {
    // The three calls over the axes use the same array and the same
    // lists of axes. Only the function differs.
    let lens = multi_dot_apply_pins::AO_IN_L;
    var i: i64 = 0;
    while i < lens.len {
        let shape = common::shape_of(multi_dot_apply_pins::AO_SH_D, multi_dot_apply_pins::AO_SH_L, i);
        let values = common::part_of(multi_dot_apply_pins::AO_IN_D, lens, i);
        let axes = common::shape_of(multi_dot_apply_pins::AO_AX_D, multi_dot_apply_pins::AO_AX_L, i);
        let a = common::arr_of(shape, values);
        // The sum over the axes drops the axis and the package puts it back.
        let got, g = scientific::apply_over_axes(a, axes) { x, ax =>
            let r, f = scientific::sum_axis(x, ax);
            r
        };
        assert(g.kind == scientific::F_NONE);
        check_pin(got, multi_dot_apply_pins::AO_SUM_OUT_D, multi_dot_apply_pins::AO_SUM_OUT_L, i,
                  multi_dot_apply_pins::AO_SUM_OUT_SH_D, multi_dot_apply_pins::AO_SUM_OUT_SH_L);
        // The product over the same axes.
        let gp, g2 = scientific::apply_over_axes(a, axes) { x, ax =>
            let r, f = scientific::prod_axis(x, ax);
            r
        };
        assert(g2.kind == scientific::F_NONE);
        check_pin(gp, multi_dot_apply_pins::AO_PROD_OUT_D, multi_dot_apply_pins::AO_PROD_OUT_L, i,
                  multi_dot_apply_pins::AO_PROD_OUT_SH_D, multi_dot_apply_pins::AO_PROD_OUT_SH_L);
        // The cumulative sum keeps the count of the dimensions.
        let gc, g3 = scientific::apply_over_axes(a, axes) { x, ax =>
            let r, f = scientific::cumsum(x, ax);
            r
        };
        assert(g3.kind == scientific::F_NONE);
        check_pin(gc, multi_dot_apply_pins::AO_CUMSUM_OUT_D, multi_dot_apply_pins::AO_CUMSUM_OUT_L, i,
                  multi_dot_apply_pins::AO_CUMSUM_OUT_SH_D, multi_dot_apply_pins::AO_CUMSUM_OUT_SH_L);
        i += 1;
    }
}

fn test_apply_over_axes_shape() {
    // The sum over two axes of an array of three dimensions keeps the
    // count of the dimensions and gives every axis a length of one.
    let v, f = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]);
    let a, f2 = scientific::reshape(v, [2, 3, 4]);
    let got, g = scientific::apply_over_axes(a, [0, 2]) { x, ax =>
        let r, f = scientific::sum_axis(x, ax);
        r
    };
    assert(g.kind == scientific::F_NONE);
    assert(got.ndim == 3);
    assert(got.shape[0] == 1 && got.shape[1] == 3 && got.shape[2] == 1);
    assert(common::close(got.data[0], 68.0, 1e-9));
    assert(common::close(got.data[1], 100.0, 1e-9));
    assert(common::close(got.data[2], 132.0, 1e-9));
    // An empty list of axes gives a copy of the array.
    let same, g2 = scientific::apply_over_axes(a, []) { x, ax =>
        let r, f = scientific::sum_axis(x, ax);
        r
    };
    assert(g2.kind == scientific::F_NONE);
    assert(same.ndim == 3);
    assert(same.shape[0] == 2 && same.shape[1] == 3 && same.shape[2] == 4);
    assert(common::close(same.data[5], 6.0, 1e-9));
    // A negative axis counts from the end.
    let neg, g3 = scientific::apply_over_axes(a, [-1]) { x, ax =>
        let r, f = scientific::sum_axis(x, ax);
        r
    };
    assert(g3.kind == scientific::F_NONE);
    assert(neg.shape[2] == 1);
}

fn test_apply_over_axes_faults() {
    let v, f = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let a, f2 = scientific::reshape(v, [2, 3]);
    // An axis outside the shape.
    let r1, g1 = scientific::apply_over_axes(a, [5]) { x, ax =>
        let r, f = scientific::sum_axis(x, ax);
        r
    };
    assert(g1.kind == scientific::F_RANGE);
    // A list of more than 64 axes.
    var axes: i64[>..] = [];
    var i: i64 = 0;
    while i < 65 { axes.push(0); i += 1; }
    let r2, g2 = scientific::apply_over_axes(a, axes) { x, ax =>
        let r, f = scientific::sum_axis(x, ax);
        r
    };
    assert(g2.kind == scientific::F_LIMIT);
    // A function that gives another count of dimensions.
    let r3, g3 = scientific::apply_over_axes(a, [0]) { x, ax =>
        let r, f = scientific::reshape(x, [1, 2, 3]);
        r
    };
    assert(g3.kind == scientific::F_SHAPE);
    // A function that gives an array that does not match its shape.
    let r4, g4 = scientific::apply_over_axes(a, [0]) { x, ax =>
        let r = scientific::Arr { ndim: 1, shape: [3], data: [1.0] };
        r
    };
    assert(g4.kind == scientific::F_SHAPE);
    // An array that does not agree with its own shape.
    let forged = scientific::Arr { ndim: 2, shape: [2, 3], data: [1.0] };
    let r5, g5 = scientific::apply_over_axes(forged, [0]) { x, ax =>
        let r, f = scientific::sum_axis(x, ax);
        r
    };
    assert(g5.kind == scientific::F_SHAPE);
    assert(r1.data.len == 0 && r2.data.len == 0 && r3.data.len == 0);
}

fn test_vectorize_pins() {
    // One array.
    let lens = multi_dot_apply_pins::V1_N;
    var i: i64 = 0;
    while i < lens.len {
        let shape = common::shape_of(multi_dot_apply_pins::V1_SH_D, multi_dot_apply_pins::V1_SH_L, i);
        let values = common::part_of(multi_dot_apply_pins::V1_VAL_D, multi_dot_apply_pins::V1_VAL_L, i);
        let got, g = scientific::vectorize(common::arr_of(shape, values)) { x => x * x + 1.0 };
        assert(g.kind == scientific::F_NONE);
        check_pin(got, multi_dot_apply_pins::V1_OUT_D, multi_dot_apply_pins::V1_OUT_L, i,
                  multi_dot_apply_pins::V1_OUT_SH_D, multi_dot_apply_pins::V1_OUT_SH_L);
        i += 1;
    }
    // Two arrays.
    let lens2 = multi_dot_apply_pins::V2_N;
    i = 0;
    while i < lens2.len {
        let shape = common::shape_of(multi_dot_apply_pins::V2_SH_D, multi_dot_apply_pins::V2_SH_L, 2 * i);
        let values = common::part_of(multi_dot_apply_pins::V2_VAL_D, multi_dot_apply_pins::V2_VAL_L, 2 * i);
        let shape2 = common::shape_of(multi_dot_apply_pins::V2_SH_D, multi_dot_apply_pins::V2_SH_L, 2 * i + 1);
        let values2 = common::part_of(multi_dot_apply_pins::V2_VAL_D, multi_dot_apply_pins::V2_VAL_L, 2 * i + 1);
        let got, g = scientific::vectorize2(common::arr_of(shape, values), common::arr_of(shape2, values2)) { x, y => x + y };
        assert(g.kind == scientific::F_NONE);
        check_pin(got, multi_dot_apply_pins::V2_OUT_D, multi_dot_apply_pins::V2_OUT_L, i,
                  multi_dot_apply_pins::V2_OUT_SH_D, multi_dot_apply_pins::V2_OUT_SH_L);
        i += 1;
    }
    // Three arrays.
    let lens3 = multi_dot_apply_pins::V3_N;
    i = 0;
    while i < lens3.len {
        let shape = common::shape_of(multi_dot_apply_pins::V3_SH_D, multi_dot_apply_pins::V3_SH_L, 3 * i);
        let values = common::part_of(multi_dot_apply_pins::V3_VAL_D, multi_dot_apply_pins::V3_VAL_L, 3 * i);
        let shape2 = common::shape_of(multi_dot_apply_pins::V3_SH_D, multi_dot_apply_pins::V3_SH_L, 3 * i + 1);
        let values2 = common::part_of(multi_dot_apply_pins::V3_VAL_D, multi_dot_apply_pins::V3_VAL_L, 3 * i + 1);
        let shape3 = common::shape_of(multi_dot_apply_pins::V3_SH_D, multi_dot_apply_pins::V3_SH_L, 3 * i + 2);
        let values3 = common::part_of(multi_dot_apply_pins::V3_VAL_D, multi_dot_apply_pins::V3_VAL_L, 3 * i + 2);
        let got, g = scientific::vectorize3(common::arr_of(shape, values), common::arr_of(shape2, values2), common::arr_of(shape3, values3)) { x, y, z => x + y + z };
        assert(g.kind == scientific::F_NONE);
        check_pin(got, multi_dot_apply_pins::V3_OUT_D, multi_dot_apply_pins::V3_OUT_L, i,
                  multi_dot_apply_pins::V3_OUT_SH_D, multi_dot_apply_pins::V3_OUT_SH_L);
        i += 1;
    }
}

fn test_vectorize_broadcast() {
    // A row and a column broadcast to a matrix.
    let row, f = scientific::from_values([1.0, 2.0, 3.0]);
    let col, f2 = scientific::from_values([10.0, 20.0]);
    let c, f3 = scientific::reshape(col, [2, 1]);
    let got, g = scientific::vectorize2(copy(row), copy(c)) { x, y => x + y };
    assert(g.kind == scientific::F_NONE);
    assert(got.ndim == 2);
    assert(got.shape[0] == 2 && got.shape[1] == 3);
    assert(common::close(got.data[0], 11.0, 1e-9));
    assert(common::close(got.data[5], 23.0, 1e-9));
}

fn test_vectorize_faults() {
    // An array with no element.
    let empty = scientific::Arr { ndim: 1, shape: [0], data: [] };
    let r1, g1 = scientific::vectorize(empty) { x => x };
    assert(g1.kind == scientific::F_VALUE);
    // Arrays that do not broadcast.
    let a, f = scientific::from_values([1.0, 2.0, 3.0]);
    let b, f2 = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    let r2, g2 = scientific::vectorize2(copy(a), copy(b)) { x, y => x + y };
    assert(g2.kind == scientific::F_SHAPE);
    // An array that does not agree with its own shape.
    let forged = scientific::Arr { ndim: 1, shape: [3], data: [1.0] };
    let r3, g3 = scientific::vectorize(forged) { x => x };
    assert(g3.kind == scientific::F_SHAPE);
    // A block that gives a value that is not finite. The value stays.
    let zero: f64 = 0.0;
    let inf: f64 = 1.0 / zero;
    let r4, g4 = scientific::vectorize(copy(a)) { x => inf };
    assert(g4.kind == scientific::F_NONE);
    assert(r4.data[0] - r4.data[0] != 0.0);
    assert(r1.data.len == 0 && r2.data.len == 0 && r3.data.len == 0);
}

fn test_chain_add_forged() {
    // An array that does not agree with its own shape.
    let forged = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0] };
    let c = scientific::chain_new();
    let got, g = scientific::chain_add(c, forged);
    assert(g.kind == scientific::F_SHAPE);
    assert(got.count == 0);
    // An array of three dimensions.
    let v, f = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t, f2 = scientific::reshape(v, [2, 3, 1]);
    let g2, k = scientific::chain_add(c, copy(t));
    assert(k.kind == scientific::F_SHAPE);
    // A forged chain whose lists hold more entries than its count. The
    // call gives a fault and not a stop.
    var nd: i64[..256] = [];
    var i: i64 = 0;
    while i < 256 { nd.push(1); i += 1; }
    let forgedc = scientific::Chain { count: 0, nd: copy(nd), rows: copy(nd), cols: copy(nd), off: copy(nd), data: [] };
    let one, f3 = scientific::from_values([1.0]);
    let g3, k2 = scientific::chain_add(forgedc, copy(one));
    assert(k2.kind == scientific::F_SHAPE);
    assert(g3.count == 0);
}

fn test_multi_dot_cost_saturates() {
    // The costs of the order stop at the largest value of the number
    // format. A chain with a large size gives a fault and not a
    // product of a wrong size.
    let big = scientific::Chain { count: 2, nd: [2, 2], rows: [100000, 100000], cols: [100000, 100000], off: [0, 0], data: [1.0] };
    let r, g = scientific::multi_dot(big);
    assert(g.kind == scientific::F_LIMIT);
    // A chain with more values than the limit gives F_LIMIT too.
    let over = scientific::Chain { count: 2, nd: [2, 2], rows: [268435457, 268435457], cols: [1, 1], off: [0, 0], data: [1.0] };
    let r2, g2 = scientific::multi_dot(over);
    assert(g2.kind == scientific::F_LIMIT);
    // A chain whose buffer is too short gives F_SHAPE.
    let short = scientific::Chain { count: 2, nd: [2, 2], rows: [10, 10], cols: [10, 10], off: [0, 0], data: [1.0] };
    let r3, g3 = scientific::multi_dot(short);
    assert(g3.kind == scientific::F_SHAPE);
    // A chain with a length below one gives F_SHAPE. The package holds
    // no array with an empty axis.
    let zero = scientific::Chain { count: 2, nd: [2, 2], rows: [10000000000, 0], cols: [0, 0], off: [0, 0], data: [1.0] };
    let r4, g4 = scientific::multi_dot(zero);
    assert(g4.kind == scientific::F_SHAPE);
    assert(r.data.len == 0 && r2.data.len == 0 && r3.data.len == 0 && r4.data.len == 0);
}

fn test_vectorize_limit() {
    // The array holds at most eight dimensions, so the count of the
    // elements is the limit that a forged array reaches. A shape with
    // more elements than the limit gives F_LIMIT.
    let forged = scientific::Arr { ndim: 2, shape: [268435456, 2], data: [1.0] };
    let r, g = scientific::vectorize(forged) { x => x };
    assert(g.kind == scientific::F_LIMIT);
    let r2, g2 = scientific::vectorize2(forged, forged) { x, y => x + y };
    assert(g2.kind == scientific::F_LIMIT);
    let r3, g3 = scientific::vectorize3(forged, forged, forged) { x, y, z => x + y + z };
    assert(g3.kind == scientific::F_LIMIT);
    assert(r.data.len == 0 && r2.data.len == 0 && r3.data.len == 0);
}

fn test_apply_over_axes_no_dimension() {
    // A block that gives no dimension gives F_SHAPE. The result holds
    // no axis, so the package cannot use it.
    let v, f = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    let a, f2 = scientific::reshape(v, [2, 2]);
    let r, g = scientific::apply_over_axes(a, [0]) { x, ax =>
        let z = scientific::Arr { ndim: 0, shape: [], data: [3.0] };
        z
    };
    assert(g.kind == scientific::F_SHAPE);
    assert(r.data.len == 0);
}

fn test_apply_over_axes_drop_axis() {
    // A block that drops the axis. The package inserts an axis of
    // length one at the place of the axis.
    let v, f = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let a, f2 = scientific::reshape(v, [2, 3]);
    let got, g = scientific::apply_over_axes(a, [0]) { x, ax =>
        let s, sf = scientific::sum_axis(x, ax);
        let r = scientific::Arr { ndim: 1, shape: [3], data: copy(s.data) };
        r
    };
    assert(g.kind == scientific::F_NONE);
    assert(got.ndim == 2);
    assert(got.shape[0] == 1 && got.shape[1] == 3);
    assert(common::close(got.data[0], 5.0, 1e-9));
    assert(common::close(got.data[1], 7.0, 1e-9));
    assert(common::close(got.data[2], 9.0, 1e-9));
    // A block that drops the axis of an array of three dimensions. The
    // inserted axis takes the place of the axis.
    let w, f3 = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let b, f4 = scientific::reshape(w, [2, 3, 2]);
    let got2, g2 = scientific::apply_over_axes(b, [1]) { x, ax =>
        let s, sf = scientific::sum_axis(x, ax);
        let r = scientific::Arr { ndim: 2, shape: [2, 2], data: copy(s.data) };
        r
    };
    assert(g2.kind == scientific::F_NONE);
    assert(got2.ndim == 3);
    assert(got2.shape[0] == 2 && got2.shape[1] == 1 && got2.shape[2] == 2);
    assert(common::close(got2.data[0], 9.0, 1e-9));
    assert(common::close(got2.data[1], 12.0, 1e-9));
    assert(common::close(got2.data[2], 27.0, 1e-9));
    assert(common::close(got2.data[3], 30.0, 1e-9));
    // A negative axis counts from the end. The inserted axis takes the
    // place of that axis.
    let got3, g3 = scientific::apply_over_axes(a, [-1]) { x, ax =>
        let s, sf = scientific::sum_axis(x, ax);
        let r = scientific::Arr { ndim: 1, shape: [2], data: copy(s.data) };
        r
    };
    assert(g3.kind == scientific::F_NONE);
    assert(got3.ndim == 2);
    assert(got3.shape[0] == 2 && got3.shape[1] == 1);
    assert(common::close(got3.data[0], 6.0, 1e-9));
    assert(common::close(got3.data[1], 15.0, 1e-9));
}
