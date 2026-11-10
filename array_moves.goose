// The moves of an array: the order of the axes, the flips, the turn,
// the broadcast to a shape, the choice among arrays and the function
// along an axis.
namespace scientific;

// fix_axis turns a negative axis into the axis from the front. An axis
// outside the shape gives -1.
fn fix_axis(axis: i64, ndim: i64) -> i64 {
    var ax = axis;
    if ax < 0 { ax += ndim; }
    if ax < 0 || ax >= ndim { return -1; }
    return ax;
}

// moveaxis_many moves the axes of the source to the places of the
// destination. The other axes keep their order, as in NumPy.
fn moveaxis_many(a, source: i64[:], destination: i64[:]) -> Arr, Fault {
    if source.len != destination.len {
        return empty_arr(), fault(F_SHAPE, "moveaxis", "the source and the destination hold a different count of axes");
    }
    var src: i64[..8] = [];
    var dst: i64[..8] = [];
    var seen_s: i64 = 0;
    var seen_d: i64 = 0;
    var i: i64 = 0;
    while i < source.len {
        let s = fix_axis(source[i], a.ndim);
        let d = fix_axis(destination[i], a.ndim);
        if s < 0 || d < 0 {
            return empty_arr(), fault(F_RANGE, "moveaxis", "an axis stands outside the shape");
        }
        if seen_s & (1 << s) != 0 || seen_d & (1 << d) != 0 {
            return empty_arr(), fault(F_VALUE, "moveaxis", "an axis repeats");
        }
        seen_s |= 1 << s;
        seen_d |= 1 << d;
        src.push(s);
        dst.push(d);
        i += 1;
    }
    // The order holds the axes that stay, then the moved ones go in
    // by the place of the destination, from the smallest place up.
    var order: i64[..8] = [];
    i = 0;
    while i < a.ndim {
        if seen_s & (1 << i) == 0 { order.push(i); }
        i += 1;
    }
    var placed: i64 = 0;
    var place: i64 = 0;
    while placed < src.len {
        var j: i64 = 0;
        while j < dst.len {
            if dst[j] == place {
                // insert src[j] at the index place
                order.push(0);
                var k: i64 = order.len - 1;
                while k > place { order[k] = order[k - 1]; k -= 1; }
                order[place] = src[j];
                placed += 1;
            }
            j += 1;
        }
        place += 1;
    }
    return transpose_perm(a, order);
}

// moveaxis moves one axis to a new place. A negative axis counts from
// the end.
fn moveaxis(a, source: i64, destination: i64) -> Arr, Fault {
    var s: i64[..8] = [];
    var d: i64[..8] = [];
    s.push(source);
    d.push(destination);
    return moveaxis_many(a, s, d);
}

// flipud reverses the order of the rows.
fn flipud(a) -> Arr, Fault {
    return flip(a, 0);
}

// fliplr reverses the order of the columns. The array holds two
// dimensions or more.
fn fliplr(a) -> Arr, Fault {
    if a.ndim < 2 {
        return empty_arr(), fault(F_SHAPE, "fliplr", "the array holds fewer than two dimensions");
    }
    return flip(a, 1);
}

// rot90 turns an array by a quarter turn, k times, in the plane of two
// axes. The turn goes from the first axis toward the second one, as
// NumPy gives it. A negative k turns the other way.
fn rot90(a, k: i64 = 1, axis0: i64 = 0, axis1: i64 = 1) -> Arr, Fault {
    if a.ndim < 2 {
        return empty_arr(), fault(F_SHAPE, "rot90", "the array holds fewer than two dimensions");
    }
    let x = fix_axis(axis0, a.ndim);
    let y = fix_axis(axis1, a.ndim);
    if x < 0 || y < 0 {
        return empty_arr(), fault(F_RANGE, "rot90", "an axis stands outside the shape");
    }
    if x == y {
        return empty_arr(), fault(F_VALUE, "rot90", "the two axes are the same");
    }
    var turns = k % 4;
    if turns < 0 { turns += 4; }
    var swap: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim { swap.push(i); i += 1; }
    swap[x] = y;
    swap[y] = x;
    if turns == 0 { return clone(a), no_fault(); }
    if turns == 2 {
        let f1, e1 = flip(a, x);
        if e1.kind != F_NONE { return empty_arr(), e1; }
        return flip(f1, y);
    }
    if turns == 1 {
        let f1, e1 = flip(a, y);
        if e1.kind != F_NONE { return empty_arr(), e1; }
        return transpose_perm(f1, swap);
    }
    let t, e2 = transpose_perm(a, swap);
    if e2.kind != F_NONE { return empty_arr(), e2; }
    return flip(t, y);
}

// broadcast_to gives an array the shape that the caller names. The
// array must broadcast to that shape under the rule of NumPy.
fn broadcast_to(a, shape: i64[:]) -> Arr, Fault {
    let target, ft = broadcast_shapes(a.shape, shape);
    if ft.kind != F_NONE { return empty_arr(), ft; }
    if target.len != shape.len {
        return empty_arr(), fault(F_SHAPE, "broadcast_to", "the array holds more dimensions than the shape");
    }
    var i: i64 = 0;
    while i < shape.len {
        if target[i] != shape[i] {
            return empty_arr(), fault(F_SHAPE, "broadcast_to", "the array does not broadcast to the shape");
        }
        i += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < shape.len { s.push(shape[i]); i += 1; }
    let cf = check_shape(s);
    if cf.kind != F_NONE { return empty_arr(), cf; }
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    bcast_indices(s, a.shape, s, ia, ib);
    var d: f64[>..] = [];
    i = 0;
    while i < ia.len { d.push(a.data[ia[i]]); i += 1; }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// broadcast_arrays gives two arrays the one shape that they share.
fn broadcast_arrays(a, b) -> Arr, Arr, Fault {
    let s, fs = bshape(a, b);
    if fs.kind != F_NONE { return empty_arr(), empty_arr(), fs; }
    let x, fx = broadcast_to(a, s);
    if fx.kind != F_NONE { return empty_arr(), empty_arr(), fx; }
    let y, fy = broadcast_to(b, s);
    if fy.kind != F_NONE { return empty_arr(), empty_arr(), fy; }
    return x, y, no_fault();
}

// select picks, for each element, the choice of the first condition
// that holds, and the default where none holds. NumPy takes lists of
// arrays. Here the first axis of the conditions and the first axis
// of the choices hold the lists. Each condition and each choice take
// the shape of the remaining axes.
fn select(conds, choices, default: f64 = 0.0) -> Arr, Fault {
    if conds.ndim != choices.ndim || conds.ndim < 2 {
        return empty_arr(), fault(F_SHAPE, "select", "the conditions and the choices hold a different count of dimensions");
    }
    var i: i64 = 0;
    while i < conds.ndim {
        if conds.shape[i] != choices.shape[i] {
            return empty_arr(), fault(F_SHAPE, "select", "the conditions and the choices hold different shapes");
        }
        i += 1;
    }
    let count = conds.shape[0];
    let inner = conds.data.len / count;
    var d: f64[>..] = [];
    var e: i64 = 0;
    while e < inner {
        var v = default;
        var c: i64 = 0;
        while c < count {
            if conds.data[c * inner + e] != 0.0 {
                v = choices.data[c * inner + e];
                break;
            }
            c += 1;
        }
        d.push(v);
        e += 1;
    }
    var s: i64[..8] = [];
    i = 1;
    while i < conds.ndim { s.push(conds.shape[i]); i += 1; }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// line_at copies the line of an array along an axis. The outer index
// and the inner index pick the line.
fn line_at(a, axis: i64, outer: i64, inner: i64) -> Arr {
    let width = a.shape[axis];
    let stride = product_after_of(a.shape, axis);
    var d: f64[>..] = [];
    var k: i64 = 0;
    while k < width {
        d.push(a.data[(outer * width + k) * stride + inner]);
        k += 1;
    }
    var s: i64[..8] = [];
    s.push(width);
    return Arr { ndim: 1, shape: s, data: d };
}

// apply_along_axis calls a function on every line of an array along an
// axis. The function takes a line of one dimension and gives a line of
// one dimension. Every call gives the same length. That length replaces
// the length of the axis. The caller passes the function at the end of
// the call.
fn apply_along_axis<F: fn(Arr&) -> Arr>(a, axis: i64) -> Arr, Fault {
    let ax = fix_axis(axis, a.ndim);
    if ax < 0 {
        return empty_arr(), fault(F_RANGE, "apply_along_axis", "the axis stands outside the shape");
    }
    let stride = product_after_of(a.shape, ax);
    let outer = product_before_of(a.shape, ax);
    var parts: f64[>..] = [];
    var width: i64 = -1;
    var o: i64 = 0;
    while o < outer {
        var q: i64 = 0;
        while q < stride {
            var line = line_at(a, ax, o, q);
            let r = F(line);
            if r.ndim != 1 {
                return empty_arr(), fault(F_SHAPE, "apply_along_axis", "the function gives not one dimension");
            }
            if width < 0 { width = r.data.len; }
            if r.data.len != width {
                return empty_arr(), fault(F_SHAPE, "apply_along_axis", "the function gives lines of different lengths");
            }
            var k: i64 = 0;
            while k < width { parts.push(r.data[k]); k += 1; }
            q += 1;
        }
        o += 1;
    }
    if width < 1 {
        return empty_arr(), fault(F_VALUE, "apply_along_axis", "the function gives no element");
    }
    // The parts sit one line after another. They go to their places
    // in the new shape.
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    s[ax] = width;
    let cf = check_shape(s);
    if cf.kind != F_NONE { return empty_arr(), cf; }
    var d: f64[>..] = [];
    i = 0;
    while i < outer * width * stride { d.push(0.0); i += 1; }
    o = 0;
    while o < outer {
        var q: i64 = 0;
        while q < stride {
            var k: i64 = 0;
            while k < width {
                d[(o * width + k) * stride + q] = parts[(o * stride + q) * width + k];
                k += 1;
            }
            q += 1;
        }
        o += 1;
    }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// apply_along_axis_scalar calls a function on every line of an array
// along an axis. The function gives one number for each line. The axis
// drops from the shape. A line of one dimension gives one element.
fn apply_along_axis_scalar<F: fn(Arr&) -> f64>(a, axis: i64) -> Arr, Fault {
    let ax = fix_axis(axis, a.ndim);
    if ax < 0 {
        return empty_arr(), fault(F_RANGE, "apply_along_axis", "the axis stands outside the shape");
    }
    let stride = product_after_of(a.shape, ax);
    let outer = product_before_of(a.shape, ax);
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var q: i64 = 0;
        while q < stride {
            var line = line_at(a, ax, o, q);
            d.push(F(line));
            q += 1;
        }
        o += 1;
    }
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim {
        if i != ax { s.push(a.shape[i]); }
        i += 1;
    }
    if s.len == 0 { s.push(1); }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}
