// The tests of the group norms_tensors. Run them with SCIENTIFIC_SUITES=norms_tensors.
//
// The norm of a vector, the norm of a matrix, the inverse of a tensor
// and the solution of a tensor equation match the pins of NumPy 2.5.4.
namespace norms_tensors_tests;

import scientific from "../..";
import common;
import norms_tensors_pins;

// vector_order_of gives the order of the norm of a vector. The number
// of the pins stands for the order.
fn vector_order_of(code: f64) -> f64 {
    let zero: f64 = 0.0;
    if code == 1.0 { return 1.0; }
    if code == 2.0 { return 0.0; }
    if code == 3.0 { return 3.0; }
    if code == 4.0 { return 0.5; }
    if code == 5.0 { return -1.0; }
    if code == 6.0 { return 1.0 / zero; }
    if code == 7.0 { return -1.0 / zero; }
    if code == 8.0 { return -2.0; }
    return 2.0;
}

// matrix_order_of gives the name of the order of the norm of a matrix.
fn matrix_order_of(code: f64) -> const u8[:] {
    if code == 1.0 { return "nuc"; }
    if code == 2.0 { return "inf"; }
    if code == 3.0 { return "-inf"; }
    if code == 4.0 { return "1"; }
    if code == 5.0 { return "-1"; }
    if code == 6.0 { return "2"; }
    if code == 7.0 { return "-2"; }
    return "fro";
}

fn test_vector_norm_pins() {
    let lens = norms_tensors_pins::VN_IN_L;
    var i: i64 = 0;
    while i < lens.len {
        let shape = common::shape_of(norms_tensors_pins::VN_SH_D, norms_tensors_pins::VN_SH_L, i);
        let values = common::part_of(norms_tensors_pins::VN_IN_D, lens, i);
        let axes = common::shape_of(norms_tensors_pins::VN_AX_D, norms_tensors_pins::VN_AX_L, i);
        let ord = vector_order_of(norms_tensors_pins::VN_ORD[i]);
        let keep = norms_tensors_pins::VN_KEEP[i] == 1.0;
        let got, f = scientific::vector_norm(common::arr_of(shape, values), ord, axes, keep);
        assert(f.kind == scientific::F_NONE);
        let want = common::part_of(norms_tensors_pins::VN_OUT_D, norms_tensors_pins::VN_OUT_L, i);
        let wshape = common::shape_of(norms_tensors_pins::VN_OUT_SH_D, norms_tensors_pins::VN_OUT_SH_L, i);
        assert(common::same_shape(got, wshape));
        assert(got.data.len == want.len);
        var k: i64 = 0;
        while k < want.len {
            assert(common::close(got.data[k], want[k], 1e-9));
            k += 1;
        }
        i += 1;
    }
}

fn test_vector_norm_shape() {
    // The norm of a whole array gives one element. The flag that keeps
    // the dimensions gives every axis a length of one.
    let a = common::arr_of([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let got, f = scientific::vector_norm(a);
    assert(f.kind == scientific::F_NONE);
    assert(got.ndim == 1);
    assert(got.shape[0] == 1);
    assert(common::close(got.data[0], 9.539392014169456, 1e-9));
    let kept, kf = scientific::vector_norm(a, 2.0, [], true);
    assert(kf.kind == scientific::F_NONE);
    assert(kept.ndim == 2);
    assert(kept.shape[0] == 1 && kept.shape[1] == 1);
    // The axis of the norm drops from the shape.
    let one, of = scientific::vector_norm(a, 2.0, [0]);
    assert(of.kind == scientific::F_NONE);
    assert(one.ndim == 1);
    assert(one.shape[0] == 3);
    assert(common::close(one.data[0], 4.123105625617661, 1e-9));
}

fn test_vector_norm_faults() {
    let zero: f64 = 0.0;
    let inf: f64 = 1.0 / zero;
    let nan: f64 = inf - inf;
    let a = common::arr_of([2, 2], [1.0, 2.0, 3.0, 4.0]);
    // An array that does not agree with its own shape.
    let forged = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0] };
    let f1, f1k = scientific::vector_norm(forged);
    assert(f1k.kind == scientific::F_SHAPE);
    // An axis outside the shape.
    let f2, f2k = scientific::vector_norm(a, 2.0, [7]);
    assert(f2k.kind == scientific::F_RANGE);
    // An axis that repeats.
    let f3, f3k = scientific::vector_norm(a, 2.0, [0, 0]);
    assert(f3k.kind == scientific::F_VALUE);
    // An order that is not a number.
    let f4, f4k = scientific::vector_norm(a, nan);
    assert(f4k.kind == scientific::F_VALUE);
    // A value of the array that is not finite.
    let bad = common::arr_of([2], [1.0, inf]);
    let f5, f5k = scientific::vector_norm(bad);
    assert(f5k.kind == scientific::F_VALUE);
    assert(f1.data.len == 0);
}

fn test_matrix_norm_pins() {
    let lens = norms_tensors_pins::MN_IN_L;
    var i: i64 = 0;
    while i < lens.len {
        let shape = common::shape_of(norms_tensors_pins::MN_SH_D, norms_tensors_pins::MN_SH_L, i);
        let values = common::part_of(norms_tensors_pins::MN_IN_D, lens, i);
        let order = matrix_order_of(norms_tensors_pins::MN_ORD[i]);
        let keep = norms_tensors_pins::MN_KEEP[i] == 1.0;
        let got, f = scientific::matrix_norm(common::arr_of(shape, values), order, keep);
        assert(f.kind == scientific::F_NONE);
        let want = common::part_of(norms_tensors_pins::MN_OUT_D, norms_tensors_pins::MN_OUT_L, i);
        let wshape = common::shape_of(norms_tensors_pins::MN_OUT_SH_D, norms_tensors_pins::MN_OUT_SH_L, i);
        assert(common::same_shape(got, wshape));
        assert(got.data.len == want.len);
        var k: i64 = 0;
        while k < want.len {
            assert(common::close(got.data[k], want[k], 1e-9));
            k += 1;
        }
        i += 1;
    }
}

fn test_matrix_norm_shape() {
    // The last two axes drop from the shape. A stack gives one value
    // for each matrix of the stack.
    let a = common::arr_of([2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let got, f = scientific::matrix_norm(a, "fro");
    assert(f.kind == scientific::F_NONE);
    assert(got.ndim == 1);
    assert(got.shape[0] == 2);
    assert(common::close(got.data[0], 5.477225575051661, 1e-9));
    assert(common::close(got.data[1], 13.190905958272919, 1e-9));
    // The flag that keeps the dimensions leaves the last two axes in
    // their place with a length of one.
    let kept, kf = scientific::matrix_norm(a, "fro", true);
    assert(kf.kind == scientific::F_NONE);
    assert(kept.ndim == 3);
    assert(kept.shape[0] == 2 && kept.shape[1] == 1 && kept.shape[2] == 1);
}

fn test_matrix_norm_faults() {
    let a = common::arr_of([2, 2], [1.0, 2.0, 3.0, 4.0]);
    // An order that does not exist.
    let f1, f1k = scientific::matrix_norm(a, "abc");
    assert(f1k.kind == scientific::F_VALUE);
    // An array of one dimension.
    let one = common::arr_of([2], [1.0, 2.0]);
    let f2, f2k = scientific::matrix_norm(one);
    assert(f2k.kind == scientific::F_SHAPE);
    // An array that does not agree with its own shape.
    let forged = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0] };
    let f3, f3k = scientific::matrix_norm(forged);
    assert(f3k.kind == scientific::F_SHAPE);
}

fn test_tensorinv_pins() {
    let lens = norms_tensors_pins::TI_IN_L;
    var i: i64 = 0;
    while i < lens.len {
        let shape = common::shape_of(norms_tensors_pins::TI_SH_D, norms_tensors_pins::TI_SH_L, i);
        let values = common::part_of(norms_tensors_pins::TI_IN_D, lens, i);
        let ind = norms_tensors_pins::TI_IND[i] as i64;
        let got, f = scientific::tensorinv(common::arr_of(shape, values), ind);
        assert(f.kind == scientific::F_NONE);
        let want = common::part_of(norms_tensors_pins::TI_OUT_D, norms_tensors_pins::TI_OUT_L, i);
        let wshape = common::shape_of(norms_tensors_pins::TI_OUT_SH_D, norms_tensors_pins::TI_OUT_SH_L, i);
        assert(common::same_shape(got, wshape));
        assert(got.data.len == want.len);
        var k: i64 = 0;
        while k < want.len {
            assert(common::close(got.data[k], want[k], 1e-9));
            k += 1;
        }
        i += 1;
    }
}

fn test_tensorinv_product() {
    // The product of the inverse and the tensor is the identity.
    var values: f64[>..] = [];
    var i: i64 = 0;
    while i < 16 {
        if i % 5 == 0 { values.push(1.0); } else { values.push(0.0); }
        i += 1;
    }
    let a = scientific::arr_of(scientific::copy_slice(values), [2, 2, 4]);
    let got, f = scientific::tensorinv(a, 2);
    assert(f.kind == scientific::F_NONE);
    // The result holds the shape of the axes from the place to the end,
    // then the shape of the axes before it.
    assert(got.ndim == 3);
    assert(got.shape[0] == 4 && got.shape[1] == 2 && got.shape[2] == 2);
    let m, mf = scientific::reshape(got, [4, 4]);
    assert(mf.kind == scientific::F_NONE);
    let t, tf = scientific::reshape(a, [4, 4]);
    assert(tf.kind == scientific::F_NONE);
    let p, pf = scientific::matmul(m, t);
    assert(pf.kind == scientific::F_NONE);
    var k: i64 = 0;
    while k < 16 {
        var want: f64 = 0.0;
        if k % 5 == 0 { want = 1.0; }
        assert(common::close(p.data[k], want, 1e-9));
        k += 1;
    }
}

fn test_tensorinv_faults() {
    let a = common::arr_of([2, 2, 4], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    // A place of zero.
    let f1, f1k = scientific::tensorinv(a, 0);
    assert(f1k.kind == scientific::F_RANGE);
    // A place above the count of the axes.
    let f2, f2k = scientific::tensorinv(a, 3);
    assert(f2k.kind == scientific::F_RANGE);
    // A matrix that is not square.
    let f3, f3k = scientific::tensorinv(a, 1);
    assert(f3k.kind == scientific::F_SHAPE);
    // A matrix with no inverse.
    let sing = common::arr_of([2, 2], [1.0, 2.0, 2.0, 4.0]);
    let f4, f4k = scientific::tensorinv(sing, 1);
    assert(f4k.kind == scientific::F_VALUE);
}

fn test_tensorsolve_pins() {
    let lens = norms_tensors_pins::TS_A_L;
    var i: i64 = 0;
    while i < lens.len {
        let shape = common::shape_of(norms_tensors_pins::TS_A_SH_D, norms_tensors_pins::TS_A_SH_L, i);
        let values = common::part_of(norms_tensors_pins::TS_A_D, lens, i);
        let bshape = common::shape_of(norms_tensors_pins::TS_B_SH_D, norms_tensors_pins::TS_B_SH_L, i);
        let bvalues = common::part_of(norms_tensors_pins::TS_B_D, norms_tensors_pins::TS_B_L, i);
        let axes = common::shape_of(norms_tensors_pins::TS_AX_D, norms_tensors_pins::TS_AX_L, i);
        let got, f = scientific::tensorsolve(common::arr_of(shape, values), common::arr_of(bshape, bvalues), axes);
        assert(f.kind == scientific::F_NONE);
        let want = common::part_of(norms_tensors_pins::TS_OUT_D, norms_tensors_pins::TS_OUT_L, i);
        let wshape = common::shape_of(norms_tensors_pins::TS_OUT_SH_D, norms_tensors_pins::TS_OUT_SH_L, i);
        assert(common::same_shape(got, wshape));
        assert(got.data.len == want.len);
        var k: i64 = 0;
        while k < want.len {
            assert(common::close(got.data[k], want[k], 1e-9));
            k += 1;
        }
        i += 1;
    }
}

fn test_tensorsolve_axes() {
    // The axes that the caller names move to the right before the
    // inversion. The last named axis ends up rightmost.
    // The values of the tensor give the identity matrix when the axes
    // zero and one move to the right.
    var values: f64[>..] = [];
    var i: i64 = 0;
    var j: i64 = 0;
    var k: i64 = 0;
    while i < 2 {
        j = 0;
        while j < 3 {
            k = 0;
            while k < 6 {
                if j * 2 + i == k { values.push(1.0); } else { values.push(0.0); }
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    let a = common::arr_of([2, 3, 6], values);
    let b = common::arr_of([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let got, f = scientific::tensorsolve(a, b, [0, 2]);
    assert(f.kind == scientific::F_NONE);
    // The last axis of the tensor gives the shape of the result.
    assert(got.ndim == 1);
    assert(got.shape[0] == 6);
    var k2: i64 = 0;
    while k2 < 6 {
        assert(common::close(got.data[k2], k2 + 1.0, 1e-9));
        k2 += 1;
    }
    // The other order of the list moves the axes to another place, so
    // the matrix is not square.
    let other, of2 = scientific::tensorsolve(a, b, [2, 0]);
    assert(of2.kind == scientific::F_SHAPE);
    // Without the axes the matrix is a permutation, so the solution
    // differs.
    let plain, pf = scientific::tensorsolve(a, b);
    assert(pf.kind == scientific::F_NONE);
    assert(plain.ndim == 1);
    assert(plain.shape[0] == 6);
    assert(common::close(plain.data[1], 4.0, 1e-9));
}

fn test_tensorsolve_faults() {
    let a = common::arr_of([3, 3], [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]);
    // A right side that holds another count of values.
    let short = common::arr_of([2], [1.0, 2.0]);
    let f1, f1k = scientific::tensorsolve(a, short);
    assert(f1k.kind == scientific::F_SHAPE);
    // A matrix that is not square.
    let odd = common::arr_of([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let f2, f2k = scientific::tensorsolve(odd, common::arr_of([3], [1.0, 2.0, 3.0]));
    assert(f2k.kind == scientific::F_SHAPE);
    // An axis outside the shape.
    let f3, f3k = scientific::tensorsolve(a, common::arr_of([3], [1.0, 2.0, 3.0]), [5]);
    assert(f3k.kind == scientific::F_RANGE);
    // An axis that repeats.
    let f4, f4k = scientific::tensorsolve(a, common::arr_of([3], [1.0, 2.0, 3.0]), [0, 0]);
    assert(f4k.kind == scientific::F_VALUE);
    // A matrix with no inverse.
    let sing = common::arr_of([2, 2], [1.0, 2.0, 2.0, 4.0]);
    let f5, f5k = scientific::tensorsolve(sing, common::arr_of([2], [1.0, 2.0]));
    assert(f5k.kind == scientific::F_VALUE);
}

fn test_tensor_operand_faults() {
    let zero: f64 = 0.0;
    let inf: f64 = 1.0 / zero;
    let good = common::arr_of([2, 2], [1.0, 0.0, 0.0, 1.0]);
    let vec = common::arr_of([2], [1.0, 2.0]);
    // An array that does not agree with its own shape.
    let forged = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0] };
    let a1, k1 = scientific::tensorinv(forged, 1);
    assert(k1.kind == scientific::F_SHAPE);
    let a2, k2 = scientific::tensorsolve(forged, vec);
    assert(k2.kind == scientific::F_SHAPE);
    let forged_b = scientific::Arr { ndim: 1, shape: [2], data: [1.0] };
    let a3, k3 = scientific::tensorsolve(good, forged_b);
    assert(k3.kind == scientific::F_SHAPE);
    // A value that is not finite.
    let bad = common::arr_of([2, 2], [1.0, inf, 0.0, 1.0]);
    let a4, k4 = scientific::tensorinv(bad, 1);
    assert(k4.kind == scientific::F_VALUE);
    let a5, k5 = scientific::tensorsolve(bad, vec);
    assert(k5.kind == scientific::F_VALUE);
    let badb = common::arr_of([2], [1.0, inf]);
    let a6, k6 = scientific::tensorsolve(good, badb);
    assert(k6.kind == scientific::F_VALUE);
    // A right side with as many dimensions as the tensor.
    let a7, k7 = scientific::tensorsolve(good, good);
    assert(k7.kind == scientific::F_SHAPE);
    assert(a1.data.len == 0 && a2.data.len == 0 && a3.data.len == 0);
}

fn test_tensor_rank_deficient() {
    // The rows of this matrix are not independent, but the LU of the
    // package leaves a pivot that is not zero. The calls give a fault.
    let m = common::arr_of([3, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let a1, k1 = scientific::tensorinv(m, 1);
    assert(k1.kind == scientific::F_VALUE);
    let a2, k2 = scientific::tensorsolve(m, common::arr_of([3], [1.0, 2.0, 3.0]));
    assert(k2.kind == scientific::F_VALUE);
    assert(a1.data.len == 0 && a2.data.len == 0);
}
