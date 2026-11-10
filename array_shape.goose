// The shape operations of the array: reshape, ravel, transpose,
// squeeze, expand_dims, concatenate, stack and the broadcast rule.
//
// The broadcast rule is the rule of NumPy: two shapes align at the
// last dimension. At each position the lengths are equal, or one side
// holds 1, or one side has no position left.
namespace scientific;

// broadcast_shapes gives the shape that two arrays share under the
// broadcast rule. A conflict is a fault.
fn broadcast_shapes(sa: i64[:], sb: i64[:]) -> i64[>..], Fault {
    let n = max_i(sa.len, sb.len);
    if n > MAX_DIMS {
        var none: i64[>..] = [];
        return none, fault(F_LIMIT, "broadcast", "the shape holds more dimensions than the limit");
    }
    var out: i64[>..] = [];
    var i: i64 = 0;
    while i < n {
        let a = at_or_one(sa, sa.len - n + i);
        let b = at_or_one(sb, sb.len - n + i);
        if a == b { out.push(a); }
        else if a == 1 { out.push(b); }
        else if b == 1 { out.push(a); }
        else {
            var bad: i64[>..] = [];
            return bad, fault(F_SHAPE, "broadcast", "the shapes hold two lengths that are not equal and not one");
        }
        i += 1;
    }
    if out.len == 0 { out.push(1); }
    return out, no_fault();
}

fn max_i(a: i64, b: i64) -> i64 {
    if a > b { return a; }
    return b;
}

// at_or_one gives the length at a position, or 1 when the shape has no
// position there. A position before the start is a caller error and
// gives 1.
fn at_or_one(s: i64[:], at: i64) -> i64 {
    if at < 0 { return 1; }
    return s[at];
}

// strides_of gives the element count of one step at each position of a
// shape. The last dimension holds a step of 1.
fn strides_of(shape: i64[:], out: i64[>..]&) {
    var i: i64 = 0;
    while i < shape.len { out.push(0); i += 1; }
    var acc: i64 = 1;
    i = shape.len - 1;
    while i >= 0 {
        out[i] = acc;
        acc *= shape[i];
        i -= 1;
    }
    return;
}

// reshape gives the elements a new shape. One dimension may hold -1
// and takes the length that keeps the count of elements.
fn reshape(a, shape: i64[:]) -> Arr, Fault {
    var fixed: i64 = 1;
    var free: i64 = -1;
    var i: i64 = 0;
    while i < shape.len {
        if shape[i] == -1 {
            if free >= 0 {
                return empty_arr(), fault(F_SHAPE, "reshape", "the shape holds two open dimensions");
            }
            free = i;
        } else {
            if shape[i] < 1 {
                return empty_arr(), fault(F_VALUE, "reshape", "a dimension holds a length below one");
            }
            fixed *= shape[i];
        }
        i += 1;
    }
    if free >= 0 {
        if fixed == 0 || a.data.len % fixed != 0 {
            return empty_arr(), fault(F_SHAPE, "reshape", "the open dimension takes no whole length");
        }
    } else if fixed != a.data.len {
        return empty_arr(), fault(F_SHAPE, "reshape", "the count of elements differs from the shape");
    }
    var s: i64[..8] = [];
    i = 0;
    while i < shape.len {
        if i == free { s.push(a.data.len / fixed); } else { s.push(shape[i]); }
        i += 1;
    }
    let cf = check_shape(s);
    if cf.kind != F_NONE { return empty_arr(), cf; }
    var d: f64[>..] = [];
    i = 0;
    while i < a.data.len { d.push(a.data[i]); i += 1; }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// ravel gives the elements as a one-dimensional array.
fn ravel(a) -> Arr, Fault {
    var s: i64[..8] = [];
    s.push(a.data.len);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { d.push(a.data[i]); i += 1; }
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// transpose swaps the two dimensions of a matrix. Only a matrix
// transposes in this release.
fn transpose(a) -> Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "transpose", "the array holds not two dimensions");
    }
    let rows = a.shape[0];
    let cols = a.shape[1];
    var s: i64[..8] = [];
    s.push(cols);
    s.push(rows);
    var d: f64[>..] = [];
    var r: i64 = 0;
    while r < cols {
        var c: i64 = 0;
        while c < rows {
            d.push(a.data[c * cols + r]);
            c += 1;
        }
        r += 1;
    }
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// transpose_perm permutes the dimensions of an array. The axes hold
// one number for each dimension, as in NumPy: out dim k takes in dim
// axes[k]. An axis that repeats or stays out is a fault. The default
// order is the reverse of the dimensions.
fn transpose_perm(a, axes: i64[:]) -> Arr, Fault {
    if axes.len != a.ndim {
        return empty_arr(), fault(F_SHAPE, "transpose_perm", "the permutation holds a different count than the shape");
    }
    var seen: i64 = 0;
    var i: i64 = 0;
    while i < axes.len {
        if axes[i] < 0 || axes[i] >= a.ndim {
            return empty_arr(), fault(F_RANGE, "transpose_perm", "an axis stands outside the shape");
        }
        let bit = 1 << axes[i];
        if seen & bit != 0 {
            return empty_arr(), fault(F_SHAPE, "transpose_perm", "the permutation holds one axis twice");
        }
        seen |= bit;
        i += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < a.ndim { s.push(a.shape[axes[i]]); i += 1; }
    var in_strides: i64[>..] = [];
    strides_of(a.shape, in_strides);
    let total = a.data.len;
    var d: f64[>..] = [];
    // Walk the output in row-major order and read the input through
    // the permutation and the strides of the input.
    var pos: i64[..8] = [];
    i = 0;
    while i < s.len { pos.push(0); i += 1; }
    var n: i64 = 0;
    while n < total {
        var src: i64 = 0;
        i = 0;
        while i < a.ndim { src += pos[i] * in_strides[axes[i]]; i += 1; }
        d.push(a.data[src]);
        i = a.ndim - 1;
        while i >= 0 {
            pos[i] += 1;
            if pos[i] < s[i] { break; }
            pos[i] = 0;
            i -= 1;
        }
        n += 1;
    }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// squeeze drops every dimension of length one. One dimension always
// stays, because the package holds no scalar array.
fn squeeze(a) -> Arr, Fault {
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim {
        if a.shape[i] != 1 { s.push(a.shape[i]); }
        i += 1;
    }
    if s.len == 0 { s.push(1); }
    var d: f64[>..] = [];
    i = 0;
    while i < a.data.len { d.push(a.data[i]); i += 1; }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// expand_dims puts a new dimension of length one at an axis.
fn expand_dims(a, axis: i64) -> Arr, Fault {
    if axis < 0 || axis > a.ndim {
        return empty_arr(), fault(F_RANGE, "expand_dims", "the axis stands outside the shape");
    }
    // The new dimension takes the place of one. The check runs before
    // the fill, so the buffer of the shape never holds more than the
    // limit.
    if a.ndim + 1 > MAX_DIMS {
        return empty_arr(), fault(F_LIMIT, "expand_dims", "the shape holds more dimensions than the limit");
    }
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i <= a.ndim {
        if i == axis { s.push(1); }
        if i < a.ndim { s.push(a.shape[i]); }
        i += 1;
    }
    var d: f64[>..] = [];
    i = 0;
    while i < a.data.len { d.push(a.data[i]); i += 1; }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// concatenate joins two arrays along an axis. The shapes agree at
// every other position.
fn concatenate(axis: i64, a, b) -> Arr, Fault {
    if a.ndim != b.ndim {
        return empty_arr(), fault(F_SHAPE, "concatenate", "the arrays hold different counts of dimensions");
    }
    if axis < 0 || axis >= a.ndim {
        return empty_arr(), fault(F_RANGE, "concatenate", "the axis stands outside the shape");
    }
    var i: i64 = 0;
    while i < a.ndim {
        if i != axis && a.shape[i] != b.shape[i] {
            return empty_arr(), fault(F_SHAPE, "concatenate", "the shapes differ outside the axis");
        }
        i += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < a.ndim {
        if i == axis { s.push(a.shape[i] + b.shape[i]); } else { s.push(a.shape[i]); }
        i += 1;
    }
    let cf = check_shape(s);
    if cf.kind != F_NONE { return empty_arr(), cf; }
    var d: f64[>..] = [];
    // Walk the rows before the axis. Each row takes the axis part of
    // the first source and then of the second, in inner order.
    let inner = product_after(s, axis);
    let outer = product_before(s, axis);
    let part_a = a.shape[axis];
    let part_b = b.shape[axis];
    var o: i64 = 0;
    while o < outer {
        var p: i64 = 0;
        while p < part_a {
            var q: i64 = 0;
            while q < inner {
                d.push(a.data[(o * part_a + p) * inner + q]);
                q += 1;
            }
            p += 1;
        }
        p = 0;
        while p < part_b {
            var q: i64 = 0;
            while q < inner {
                d.push(b.data[(o * part_b + p) * inner + q]);
                q += 1;
            }
            p += 1;
        }
        o += 1;
    }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

fn product_before(s: i64[..8], axis: i64) -> i64 {
    var n: i64 = 1;
    var i: i64 = 0;
    while i < axis { n *= s[i]; i += 1; }
    return n;
}

fn product_after(s: i64[..8], axis: i64) -> i64 {
    var n: i64 = 1;
    var i: i64 = axis + 1;
    while i < s.len { n *= s[i]; i += 1; }
    return n;
}

// stack joins two arrays of one shape along a new dimension. The
// axis gives the place of the new dimension, as NumPy gives it. A
// negative axis counts from the end. The default is the first
// dimension.
fn stack(a, b, axis: i64 = 0) -> Arr, Fault {
    if a.ndim != b.ndim {
        return empty_arr(), fault(F_SHAPE, "stack", "the arrays hold different counts of dimensions");
    }
    var i: i64 = 0;
    while i < a.ndim {
        if a.shape[i] != b.shape[i] {
            return empty_arr(), fault(F_SHAPE, "stack", "the shapes differ");
        }
        i += 1;
    }
    if a.ndim + 1 > MAX_DIMS {
        return empty_arr(), fault(F_LIMIT, "stack", "the shape holds more dimensions than the limit");
    }
    // A negative axis counts from the end of the new shape.
    var ax = axis;
    if ax < 0 { ax += a.ndim + 1; }
    if ax < 0 || ax > a.ndim {
        return empty_arr(), fault(F_RANGE, "stack", "the axis stands outside the shape");
    }
    // Each array gets a dimension of one at the place of the axis.
    // The join along that axis puts the first array before the
    // second one.
    let x, fx = expand_dims(a, ax);
    if fx.kind != F_NONE { return empty_arr(), fx; }
    let y, fy = expand_dims(b, ax);
    if fy.kind != F_NONE { return empty_arr(), fy; }
    let r, f = concatenate(ax, x, y);
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, "stack", f.detail); }
    return r, no_fault();
}

// index reads one element by its position in every dimension. A
// position outside a dimension is a fault.
fn index(a, pos: i64[:]) -> f64, Fault {
    if pos.len != a.ndim {
        return 0.0, fault(F_RANGE, "index", "the position holds a different count of parts than the shape");
    }
    var flat: i64 = 0;
    var acc: i64 = 1;
    var i: i64 = a.ndim - 1;
    while i >= 0 {
        if pos[i] < 0 || pos[i] >= a.shape[i] {
            return 0.0, fault(F_RANGE, "index", "a position part stands outside its dimension");
        }
        flat += pos[i] * acc;
        acc *= a.shape[i];
        i -= 1;
    }
    return a.data[flat], no_fault();
}

// pad grows an array along one axis with a count of a value at the
// start, at the end, or at both. A negative count is a fault. The
// count of growth that passes the limit is a fault.
fn pad_value(a, axis: i64, before: i64, after: i64, value: f64) -> Arr, Fault {
    if axis < 0 || axis >= a.ndim {
        return empty_arr(), fault(F_RANGE, "pad_value", "the axis stands outside the shape");
    }
    if before < 0 || after < 0 {
        return empty_arr(), fault(F_VALUE, "pad_value", "a count of growth holds a value below zero");
    }
    let width = a.shape[axis];
    let inner = product_after_of(a.shape, axis);
    let outer = product_before_of(a.shape, axis);
    let new_width = width + before + after;
    if new_width * inner * outer > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "pad_value", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var k: i64 = 0;
        while k < new_width {
            var j: i64 = 0;
            while j < inner {
                let src = k - before;
                if src < 0 || src >= width { d.push(value); }
                else { d.push(a.data[o * width * inner + src * inner + j]); }
                j += 1;
            }
            k += 1;
        }
        o += 1;
    }
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    s[axis] = new_width;
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}
