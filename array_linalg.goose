// The products of two arrays.
//
// A matrix is an array with two dimensions. The products use the
// row-major order.
namespace scientific;

import mathshim;

// matmul multiplies two matrices. The inner dimensions must agree.
fn matmul(a, b) -> Arr, Fault {
    if a.ndim != 2 || b.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "matmul", "the operands hold a count of dimensions that is not two");
    }
    let m = a.shape[0];
    let k = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];
    if k != k2 {
        return empty_arr(), fault(F_SHAPE, "matmul", "the inner dimensions do not agree");
    }
    if m * n > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "matmul", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < m {
        var j: i64 = 0;
        while j < n {
            var t: f64 = 0.0;
            var p: i64 = 0;
            while p < k {
                t += a.data[i * k + p] * b.data[p * n + j];
                p += 1;
            }
            d.push(t);
            j += 1;
        }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(m);
    s.push(n);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// dot gives the product of two arrays. Two rows give a number. A row
// and a matrix give a row. A matrix and a row give a column.
fn dot(a, b) -> Arr, Fault {
    if a.ndim == 1 && b.ndim == 1 {
        if a.data.len != b.data.len {
            return empty_arr(), fault(F_SHAPE, "dot", "the lengths do not agree");
        }
        var t: f64 = 0.0;
        var i: i64 = 0;
        while i < a.data.len {
            t += a.data[i] * b.data[i];
            i += 1;
        }
        var d: f64[>..] = [];
        d.push(t);
        return Arr { ndim: 0, shape: [], data: d }, no_fault();
    }
    if a.ndim == 2 && b.ndim == 1 {
        if a.shape[0] != b.data.len {
            return empty_arr(), fault(F_SHAPE, "dot", "the inner dimensions do not agree");
        }
        let m = a.shape[0];
        let n = a.shape[1];
        var d: f64[>..] = [];
        var j: i64 = 0;
        while j < n {
            var t: f64 = 0.0;
            var p: i64 = 0;
            while p < m {
                t += a.data[p * n + j] * b.data[p];
                p += 1;
            }
            d.push(t);
            j += 1;
        }
        var s: i64[..8] = [];
        s.push(n);
        return Arr { ndim: 1, shape: s, data: d }, no_fault();
    }
    if a.ndim == 1 && b.ndim == 2 {
        if a.data.len != b.shape[0] {
            return empty_arr(), fault(F_SHAPE, "dot", "the inner dimensions do not agree");
        }
        let m = b.shape[0];
        let n = b.shape[1];
        var d: f64[>..] = [];
        var j: i64 = 0;
        while j < n {
            var t: f64 = 0.0;
            var p: i64 = 0;
            while p < m {
                t += a.data[p] * b.data[p * n + j];
                p += 1;
            }
            d.push(t);
            j += 1;
        }
        var s: i64[..8] = [];
        s.push(n);
        return Arr { ndim: 1, shape: s, data: d }, no_fault();
    }
    return matmul(a, b);
}

// inner gives the sum of the products of the last axis of the first
// array and the second-to-last axis of the second. A row as the second
// operand takes the last axis of the first, as NumPy does.
fn inner(a, b) -> Arr, Fault {
    if a.ndim == 0 || b.ndim == 0 {
        return empty_arr(), fault(F_SHAPE, "inner", "an operand holds no dimension");
    }
    // The length of the axis that the sum runs over.
    var contract: i64 = 0;
    if b.ndim == 1 { contract = b.data.len; } else { contract = b.shape[b.ndim - 2]; }
    let al = a.shape[a.ndim - 1];
    if al != contract {
        return empty_arr(), fault(F_SHAPE, "inner", "the inner dimensions do not agree");
    }
    // The count of the rows of the first array and of the second.
    var outer_a: i64 = 1;
    var i: i64 = 0;
    while i < a.ndim - 1 { outer_a *= a.shape[i]; i += 1; }
    var outer_b: i64 = 1;
    var last_b: i64 = 1;
    if b.ndim == 1 {
        outer_b = 1;
        last_b = 1;
    } else {
        i = 0;
        while i < b.ndim - 2 { outer_b *= b.shape[i]; i += 1; }
        last_b = b.shape[b.ndim - 1];
    }
    // The shape of the result.
    var out_shape: i64[>..] = [];
    i = 0;
    while i < a.ndim - 1 { out_shape.push(a.shape[i]); i += 1; }
    if b.ndim > 1 {
        i = 0;
        while i < b.ndim - 2 { out_shape.push(b.shape[i]); i += 1; }
        out_shape.push(last_b);
    }
    let total = outer_a * outer_b * last_b;
    if total > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "inner", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var oa: i64 = 0;
    while oa < outer_a {
        var ob: i64 = 0;
        while ob < outer_b {
            var lb: i64 = 0;
            while lb < last_b {
                var t: f64 = 0.0;
                var c: i64 = 0;
                while c < contract {
                    let av = a.data[oa * contract + c];
                    var bv: f64 = 0.0;
                    if b.ndim == 1 { bv = b.data[c]; } else { bv = b.data[ob * contract * last_b + c * last_b + lb]; }
                    t += av * bv;
                    c += 1;
                }
                d.push(t);
                lb += 1;
            }
            ob += 1;
        }
        oa += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < out_shape.len { s.push(out_shape[i]); i += 1; }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// outer gives the product of every element of a row and every element
// of a row.
fn outer(a, b) -> Arr, Fault {
    if a.ndim != 1 || b.ndim != 1 {
        return empty_arr(), fault(F_SHAPE, "outer", "the operands hold a count of dimensions that is not one");
    }
    let m = a.data.len;
    let n = b.data.len;
    if m * n > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "outer", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < m {
        var j: i64 = 0;
        while j < n {
            d.push(a.data[i] * b.data[j]);
            j += 1;
        }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(m);
    s.push(n);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// kron gives the Kronecker product of two matrices.
fn kron(a, b) -> Arr, Fault {
    if a.ndim != 2 || b.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "kron", "the operands hold a count of dimensions that is not two");
    }
    let am = a.shape[0];
    let an = a.shape[1];
    let bm = b.shape[0];
    let bn = b.shape[1];
    let m = am * bm;
    let n = an * bn;
    if m * n > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "kron", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < am {
        var j: i64 = 0;
        while j < an {
            var p: i64 = 0;
            while p < bm {
                var q: i64 = 0;
                while q < bn {
                    d.push(a.data[i * an + j] * b.data[p * bn + q]);
                    q += 1;
                }
                p += 1;
            }
            j += 1;
        }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(m);
    s.push(n);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// trace gives the sum of the elements on the diagonal.
fn trace(a) -> f64, Fault {
    if a.ndim != 2 {
        return 0.0, fault(F_SHAPE, "trace", "the operand holds a count of dimensions that is not two");
    }
    if a.shape[0] != a.shape[1] {
        return 0.0, fault(F_SHAPE, "trace", "the matrix is not square");
    }
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < a.shape[0] {
        t += a.data[i * a.shape[1] + i];
        i += 1;
    }
    return t, no_fault();
}

// norm gives the Frobenius norm of a matrix and the second norm of a
// row. Both are the square root of the sum of the squares.
fn norm(a) -> f64, Fault {
    if a.ndim == 0 {
        return 0.0, fault(F_SHAPE, "norm", "the operand holds no dimension");
    }
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < a.data.len {
        t += a.data[i] * a.data[i];
        i += 1;
    }
    return mathshim::root(t), no_fault();
}
