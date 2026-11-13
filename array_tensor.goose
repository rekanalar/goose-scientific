// The inverse of a tensor and the solution of a tensor equation.
//
// A tensor is a matrix of many axes. The inverse of a tensor reads the
// buffer of the tensor as a square matrix and inverts it. The solution
// of a tensor equation reads the same matrix and solves the system.
namespace scientific;

import mathshim;

// tensor_singular tells whether a square matrix has no inverse in
// practice. The smallest singular value must exceed the largest one
// times the side times the unit roundoff, as the rank rule of NumPy
// reads. The test catches a rank-deficient matrix whose LU leaves a
// tiny pivot that is not zero.
fn tensor_singular(m, n: i64) -> bool, Fault {
    let sv, f = svdvals(m);
    if f.kind != F_NONE { return true, f; }
    let big = sv.data[0];
    let small = sv.data[sv.data.len - 1];
    return big == 0.0 || small <= (n as f64) * 2.220446049250313e-16 * big, no_fault();
}

// tensorinv gives the inverse of a tensor. The axes from the place to
// the end give the rows of a square matrix, and the axes before the
// place give the columns. The result holds the shape of the axes from
// the place to the end, then the shape of the axes before it. The
// place lies above zero and below the count of the axes.
fn tensorinv(a, ind: i64 = 2) -> Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "tensorinv");
    if f.kind != F_NONE { return empty_arr(), f; }
    if ind < 1 || ind >= a.ndim {
        return empty_arr(), fault(F_RANGE, "tensorinv", "the place stands outside the count of the axes");
    }
    if !all_finite(a.data) {
        return empty_arr(), fault(F_VALUE, "tensorinv", "a value of the array is not finite");
    }
    // The axes from the place to the end give the rows of the matrix.
    var rows: i64 = 1;
    var i: i64 = ind;
    while i < a.ndim { rows *= a.shape[i]; i += 1; }
    let n = numel(a.shape);
    if rows * rows != n {
        return empty_arr(), fault(F_SHAPE, "tensorinv", "the matrix of the tensor is not square");
    }
    let mx = mat2(copy_slice(a.data), rows, rows);
    let sing, sg = tensor_singular(mx, rows);
    if sg.kind != F_NONE { return empty_arr(), sg; }
    if sing { return empty_arr(), fault(F_VALUE, "tensorinv", "the matrix has no inverse"); }
    let inv_a, g = inv(mx);
    if g.kind != F_NONE { return empty_arr(), g; }
    if !all_finite(inv_a.data) {
        return empty_arr(), fault(F_VALUE, "tensorinv", "the matrix has no inverse");
    }
    // The result holds the axes from the place to the end, then the
    // axes before the place.
    var s: i64[>..] = [];
    i = ind;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    i = 0;
    while i < ind { s.push(a.shape[i]); i += 1; }
    return arr_of(copy(inv_a.data), s), no_fault();
}

// tensorsolve gives the solution of a tensor equation. The last axes
// of the tensor give the shape of the result. The count of those axes
// is as many as the right side has dimensions less than the tensor.
// The right side is read as a flat list of values. The axes that the
// caller names move to the right before the inversion.
fn tensorsolve(a, b, axes: i64[:] = []) -> Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "tensorsolve");
    if f.kind != F_NONE { return empty_arr(), f; }
    let g = check_stack(b.ndim, b.shape, b.data.len, 1, "tensorsolve");
    if g.kind != F_NONE { return empty_arr(), g; }
    if !all_finite(a.data) || !all_finite(b.data) {
        return empty_arr(), fault(F_VALUE, "tensorsolve", "a value of the arrays is not finite");
    }
    // The count of the last axes of the tensor.
    let naxes = a.ndim - b.ndim;
    if naxes < 1 {
        return empty_arr(), fault(F_SHAPE, "tensorsolve", "the right side holds as many dimensions as the tensor");
    }
    // The axes that the caller names move to the right. They keep the
    // order of the list, and the last one ends up rightmost. The move
    // gives a new array, so the tensor of the caller stays as it is.
    let k = axes.len;
    var dest: i64[>..] = [];
    var i: i64 = 0;
    while i < k { dest.push(a.ndim - k + i); i += 1; }
    let m, mf = moveaxis_many(a, axes, copy(dest));
    if mf.kind != F_NONE { return empty_arr(), mf; }
    // The last axes give the shape of the result and the size of the
    // matrix.
    var prod: i64 = 1;
    var s: i64[>..] = [];
    i = m.ndim - naxes;
    while i < m.ndim { prod *= m.shape[i]; s.push(m.shape[i]); i += 1; }
    if numel(m.shape) != prod * prod {
        return empty_arr(), fault(F_SHAPE, "tensorsolve", "the matrix of the tensor is not square");
    }
    if b.data.len != prod {
        return empty_arr(), fault(F_SHAPE, "tensorsolve", "the right side holds another count of values");
    }
    let mx = mat2(copy_slice(m.data), prod, prod);
    let sing, sg = tensor_singular(mx, prod);
    if sg.kind != F_NONE { return empty_arr(), sg; }
    if sing { return empty_arr(), fault(F_VALUE, "tensorsolve", "the matrix has no inverse"); }
    let x, sf = solve(mx, vec1(copy_slice(b.data)));
    if sf.kind != F_NONE { return empty_arr(), sf; }
    if !all_finite(x.data) {
        return empty_arr(), fault(F_VALUE, "tensorsolve", "the matrix has no inverse");
    }
    return arr_of(copy(x.data), s), no_fault();
}
