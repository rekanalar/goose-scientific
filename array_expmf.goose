// The Frechet derivative of the matrix exponential and the condition
// number of the matrix exponential, like expm_frechet and expm_cond
// of scipy.linalg. The derivative comes from the exponential of the
// block matrix that holds the matrix and the direction.
namespace scientific;

import mathshim;
import array_shape;
import array_scilinalg;
import array_svd;

// The largest side of the block matrix of the derivative. The block
// matrix holds twice the side of the matrix of the caller.
const FRECHET_MAX_SIDE: i64 = 128;

// The largest side of a matrix for the condition number. The form of
// the derivative holds the square of the side.
const COND_MAX_SIDE: i64 = 8;

// ef_block gives the exponential of a matrix and the Frechet
// derivative of the exponential in a direction. The method uses the
// block matrix that holds the matrix and the direction.
fn ef_block(a, e) -> Arr, Arr, Fault {
    let n = a.shape[0];
    // The block matrix holds two blocks of n rows.
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 4 * n * n { d.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            d[i * 2 * n + j] = a.data[i * n + j];
            d[i * 2 * n + n + j] = e.data[i * n + j];
            d[(n + i) * 2 * n + n + j] = a.data[i * n + j];
            j += 1;
        }
        i += 1;
    }
    let m = mat2(copy(d), 2 * n, 2 * n);
    let x, f = expm(m);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    // The upper left part holds the exponential, and the upper right
    // part holds the derivative.
    var d1: f64[>..] = [];
    var d2: f64[>..] = [];
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            d1.push(x.data[i * 2 * n + j]);
            d2.push(x.data[i * 2 * n + n + j]);
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d1), n, n), mat2(copy(d2), n, n), no_fault();
}

// expm_frechet gives the exponential of a matrix and the Frechet
// derivative of the exponential in the direction of a second matrix.
// The derivative holds the change of the exponential for a change of
// the matrix in the direction of the second matrix.
fn expm_frechet(a, e) -> Arr, Arr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] || a.shape[0] < 1 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "expm_frechet", "the operand is not a square matrix");
    }
    if e.ndim != 2 || e.shape[0] != e.shape[1] || e.shape[0] != a.shape[0] {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "expm_frechet", "the direction holds a shape that differs from the matrix");
    }
    let n = a.shape[0];
    if n > FRECHET_MAX_SIDE {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "expm_frechet", "the side is above the limit");
    }
    var i: i64 = 0;
    while i < a.data.len {
        if !(a.data[i] - a.data[i] == 0.0) {
            return empty_arr(), empty_arr(), fault(F_VALUE, "expm_frechet", "the matrix holds a value that is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < e.data.len {
        if !(e.data[i] - e.data[i] == 0.0) {
            return empty_arr(), empty_arr(), fault(F_VALUE, "expm_frechet", "the direction holds a value that is not finite");
        }
        i += 1;
    }
    return ef_block(a, e);
}

// ef_vec gives the values of a matrix in the order of the columns.
fn ef_vec(a) -> f64[>..] {
    let n = a.shape[0];
    var v: f64[>..] = [];
    var j: i64 = 0;
    while j < n {
        var i: i64 = 0;
        while i < n {
            v.push(a.data[i * n + j]);
            i += 1;
        }
        j += 1;
    }
    return v;
}

// expm_cond gives the relative condition number of the matrix
// exponential in the norm of Frobenius. The method builds the form of
// the derivative, which holds the square of the side of the matrix.
fn expm_cond(a) -> f64, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] || a.shape[0] < 1 {
        return 0.0, fault(F_SHAPE, "expm_cond", "the operand is not a square matrix");
    }
    let n = a.shape[0];
    if n > COND_MAX_SIDE {
        return 0.0, fault(F_LIMIT, "expm_cond", "the side is above the limit");
    }
    var i: i64 = 0;
    while i < a.data.len {
        if !(a.data[i] - a.data[i] == 0.0) {
            return 0.0, fault(F_VALUE, "expm_cond", "the matrix holds a value that is not finite");
        }
        i += 1;
    }
    let x, f = expm(a);
    if f.kind != F_NONE { return 0.0, f; }
    let xnorm, f2 = norm(x);
    if f2.kind != F_NONE { return 0.0, f2; }
    if xnorm == 0.0 { return 0.0, fault(F_VALUE, "expm_cond", "the exponential holds no value"); }
    // The form of the derivative holds one column for each pair of a
    // row and a column of the matrix.
    var k: f64[>..] = [];
    var r: i64 = 0;
    while r < n {
        var c: i64 = 0;
        while c < n {
            // The direction holds one value at the pair of the row and
            // the column.
            var dir: f64[>..] = [];
            var i2: i64 = 0;
            while i2 < n * n { dir.push(0.0); i2 += 1; }
            dir[r * n + c] = 1.0;
            let e = mat2(copy(dir), n, n);
            let _, der, f3 = ef_block(a, e);
            if f3.kind != F_NONE { return 0.0, f3; }
            let v = ef_vec(der);
            var t: i64 = 0;
            while t < v.len { k.push(v[t]); t += 1; }
            c += 1;
        }
        r += 1;
    }
    // The columns of the form sit beside each other.
    var form: f64[>..] = [];
    var col: i64 = 0;
    while col < n * n {
        var row: i64 = 0;
        while row < n * n {
            form.push(k[col * n * n + row]);
            row += 1;
        }
        col += 1;
    }
    let fm = mat2(copy(form), n * n, n * n);
    let vals, f5 = svdvals(fm);
    if f5.kind != F_NONE { return 0.0, f5; }
    if vals.data.len == 0 { return 0.0, fault(F_VALUE, "expm_cond", "the form of the derivative holds no value"); }
    let knorm = vals.data[0];
    let anorm, f6 = norm(a);
    if f6.kind != F_NONE { return 0.0, f6; }
    return (knorm * anorm) / xnorm, no_fault();
}
