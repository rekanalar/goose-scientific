// The norms of a vector and of a matrix. The rules follow NumPy 2.5.4.
//
// The norm of a vector takes a real order. The norm of a matrix takes
// the name of an order. Both calls check the array against its own
// shape and against its own values before they read a value.
namespace scientific;

import mathshim;

// norm_has_axis tells whether a list of axes holds an axis.
fn norm_has_axis(axes: i64[:], ax: i64) -> bool {
    var i: i64 = 0;
    while i < axes.len {
        if axes[i] == ax { return true; }
        i += 1;
    }
    return false;
}

// norm_group_axes gives the axes that a norm weighs. An empty list of
// axes means every axis of the array. An axis below zero counts from
// the end. An axis outside the shape gives F_RANGE. An axis that
// repeats gives F_VALUE.
fn norm_group_axes(ndim: i64, axis: i64[:], out: i64[>..]&) -> Fault {
    if axis.len == 0 {
        var i: i64 = 0;
        while i < ndim { out.push(i); i += 1; }
        return no_fault();
    }
    var seen: i64 = 0;
    var i: i64 = 0;
    while i < axis.len {
        let ax = fix_axis(axis[i], ndim);
        if ax < 0 { return fault(F_RANGE, "vector_norm", "an axis stands outside the shape"); }
        if seen & (1 << ax) != 0 { return fault(F_VALUE, "vector_norm", "an axis repeats"); }
        seen |= 1 << ax;
        out.push(ax);
        i += 1;
    }
    return no_fault();
}

// norm_of_column gives the norm of the values of one column of a
// matrix. The order is a real number. The order zero counts the values
// that are not zero. The order one sums the absolute values. The order
// two takes the square root of the sum of the squares. An infinity
// takes the largest absolute value, a minus infinity the smallest one.
// Any other order takes the power of the sum of the powers.
fn norm_of_column(data: f64[:], rows: i64, cols: i64, col: i64, ord: f64) -> f64 {
    if ord == 0.0 {
        var t: f64 = 0.0;
        var g: i64 = 0;
        while g < rows {
            if data[g * cols + col] != 0.0 { t += 1.0; }
            g += 1;
        }
        return t;
    }
    if ord == 2.0 {
        var t: f64 = 0.0;
        var g: i64 = 0;
        while g < rows {
            let x = data[g * cols + col];
            t += x * x;
            g += 1;
        }
        return mathshim::root(t);
    }
    if mathshim::is_infinite_of(ord) == 1.0 {
        // An infinity takes the largest absolute value. A minus
        // infinity takes the smallest one.
        var t: f64 = 0.0;
        var g: i64 = 0;
        while g < rows {
            let x = mathshim::abs_of(data[g * cols + col]);
            if g == 0 { t = x; }
            else if ord > 0.0 { if x > t { t = x; } }
            else { if x < t { t = x; } }
            g += 1;
        }
        return t;
    }
    // Any other order takes the power of the sum of the powers.
    var s: f64 = 0.0;
    var g: i64 = 0;
    while g < rows {
        s += mathshim::power(mathshim::abs_of(data[g * cols + col]), ord);
        g += 1;
    }
    return mathshim::power(s, 1.0 / ord);
}

// vector_norm gives the norm of the values of an array along a list of
// axes. The order is a real number and the order two is the default.
// An empty list of axes gives the norm of the whole array. The axes of
// the norm drop from the shape, and a shape that loses every axis
// gives one element. The flag that keeps the dimensions leaves the
// count of the dimensions of the input and gives every axis of the
// norm a length of one.
fn vector_norm(a, ord: f64 = 2.0, axis: i64[:] = [], keepdims: bool = false) -> Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "vector_norm");
    if f.kind != F_NONE { return empty_arr(), f; }
    if ord != ord { return empty_arr(), fault(F_VALUE, "vector_norm", "the order holds no number"); }
    if !all_finite(a.data) {
        return empty_arr(), fault(F_VALUE, "vector_norm", "a value of the array is not finite");
    }
    // The axes that the norm weighs.
    var group: i64[>..] = [];
    let gf = norm_group_axes(a.ndim, axis, group);
    if gf.kind != F_NONE { return empty_arr(), gf; }
    // The axes that stay.
    var rest: i64[>..] = [];
    var i: i64 = 0;
    while i < a.ndim {
        if !norm_has_axis(group, i) { rest.push(i); }
        i += 1;
    }
    // The values of the axes of the norm stand in the first axis of a
    // matrix. The other values stand in the second axis.
    var rows: i64 = 1;
    i = 0;
    while i < group.len { rows *= a.shape[group[i]]; i += 1; }
    var cols: i64 = 1;
    i = 0;
    while i < rest.len { cols *= a.shape[rest[i]]; i += 1; }
    // The array moves the axes of the norm to the front. The move gives
    // a new array, so the array of the caller stays as it is.
    var dest: i64[>..] = [];
    i = 0;
    while i < group.len { dest.push(i); i += 1; }
    let m, mf = moveaxis_many(a, copy(group), copy(dest));
    if mf.kind != F_NONE { return empty_arr(), mf; }
    // The shape of the result.
    var s: i64[>..] = [];
    if keepdims {
        i = 0;
        while i < a.ndim {
            if norm_has_axis(group, i) { s.push(1); } else { s.push(a.shape[i]); }
            i += 1;
        }
    } else {
        i = 0;
        while i < rest.len { s.push(a.shape[rest[i]]); i += 1; }
        if s.len == 0 { s.push(1); }
    }
    // Every column of the moved array holds the values of one cell.
    var d: f64[>..] = [];
    var c: i64 = 0;
    while c < cols {
        d.push(norm_of_column(m.data, rows, cols, c, ord));
        c += 1;
    }
    var sc: i64[..8] = [];
    i = 0;
    while i < s.len { sc.push(s[i]); i += 1; }
    return Arr { ndim: sc.len, shape: sc, data: d }, no_fault();
}

// matrix_norm_code gives the number of an order of the norm of a
// matrix. A number below zero means that the order does not exist.
fn matrix_norm_code(order: const u8[:]) -> i64 {
    if order == "fro" { return 0; }
    if order == "nuc" { return 1; }
    if order == "inf" { return 2; }
    if order == "-inf" { return 3; }
    if order == "1" { return 4; }
    if order == "-1" { return 5; }
    if order == "2" { return 6; }
    if order == "-2" { return 7; }
    return -1;
}

// matrix_norm_block gives the norm of one matrix of a stack. The
// values of the matrix stand in the buffer from the place at.
fn matrix_norm_block(data: f64[:], at: i64, rows: i64, cols: i64, code: i64) -> f64, Fault {
    if code == 0 {
        // The order fro is the square root of the sum of the squares.
        var t: f64 = 0.0;
        var i: i64 = 0;
        while i < rows * cols {
            let x = data[at + i];
            t += x * x;
            i += 1;
        }
        return mathshim::root(t), no_fault();
    }
    if code == 1 || code == 6 || code == 7 {
        // The orders nuc, 2 and -2 need the singular values.
        var d: f64[>..] = [];
        var i: i64 = 0;
        while i < rows * cols { d.push(data[at + i]); i += 1; }
        let sv, sf = svdvals(mat2(copy(d), rows, cols));
        if sf.kind != F_NONE { return 0.0, sf; }
        if code == 1 {
            var t: f64 = 0.0;
            var k: i64 = 0;
            while k < sv.data.len { t += sv.data[k]; k += 1; }
            return t, no_fault();
        }
        if code == 6 { return sv.data[0], no_fault(); }
        return sv.data[sv.data.len - 1], no_fault();
    }
    // The orders inf and -inf take the sums of the rows. The orders 1
    // and -1 take the sums of the columns. The sums use the absolute
    // values. The orders inf and 1 take the largest sum.
    let by_rows = code == 2 || code == 3;
    let take_max = code == 2 || code == 4;
    var lines = cols;
    var width = rows;
    if by_rows { lines = rows; width = cols; }
    var best: f64 = 0.0;
    var l: i64 = 0;
    while l < lines {
        var t: f64 = 0.0;
        var w: i64 = 0;
        while w < width {
            if by_rows { t += mathshim::abs_of(data[at + l * cols + w]); }
            else { t += mathshim::abs_of(data[at + w * cols + l]); }
            w += 1;
        }
        if l == 0 || (take_max && t > best) || (!take_max && t < best) { best = t; }
        l += 1;
    }
    return best, no_fault();
}

// matrix_norm gives the norm of a matrix. The order is one of the
// eight names of NumPy: fro, nuc, inf, -inf, 1, -1, 2 and -2. The call
// takes no axis. The last two axes drop from the shape, and a stack of
// matrices gives one value for each matrix of the stack. The flag that
// keeps the dimensions leaves the last two axes in their place with a
// length of one.
fn matrix_norm(a, ord: const u8[:] = "fro", keepdims: bool = false) -> Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "matrix_norm");
    if f.kind != F_NONE { return empty_arr(), f; }
    if a.ndim < 2 {
        return empty_arr(), fault(F_SHAPE, "matrix_norm", "the array holds fewer than two dimensions");
    }
    let code = matrix_norm_code(ord);
    if code < 0 { return empty_arr(), fault(F_VALUE, "matrix_norm", "the order of the norm does not exist"); }
    if !all_finite(a.data) {
        return empty_arr(), fault(F_VALUE, "matrix_norm", "a value of the array is not finite");
    }
    let rows = a.shape[a.ndim - 2];
    let cols = a.shape[a.ndim - 1];
    // The leading axes give the count of the matrices.
    var count: i64 = 1;
    var i: i64 = 0;
    while i < a.ndim - 2 { count *= a.shape[i]; i += 1; }
    var d: f64[>..] = [];
    var k: i64 = 0;
    while k < count {
        let v, vf = matrix_norm_block(a.data, k * rows * cols, rows, cols, code);
        if vf.kind != F_NONE { return empty_arr(), vf; }
        d.push(v);
        k += 1;
    }
    // The shape of the result.
    var s: i64[>..] = [];
    i = 0;
    while i < a.ndim - 2 { s.push(a.shape[i]); i += 1; }
    if keepdims { s.push(1); s.push(1); }
    else if s.len == 0 { s.push(1); }
    var sc: i64[..8] = [];
    i = 0;
    while i < s.len { sc.push(s[i]); i += 1; }
    return Arr { ndim: sc.len, shape: sc, data: d }, no_fault();
}
