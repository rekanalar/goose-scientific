// The slicing, the masks and the selection by position.
//
// The flat position of an element is its row-major position. The
// functions of this file use that position, as `argmin` and `argmax`
// do.
namespace scientific;

// slice_axis keeps the elements of one axis between a start and a
// stop. A negative step reads the axis backwards. The start and the
// stop count from the end when the step is negative, as in Python.
fn slice_axis(a, axis: i64, start: i64, stop: i64, step: i64) -> Arr, Fault {
    if axis < 0 || axis >= a.ndim {
        return empty_arr(), fault(F_RANGE, "slice_axis", "the axis stands outside the shape");
    }
    if step == 0 {
        return empty_arr(), fault(F_VALUE, "slice_axis", "the step holds zero");
    }
    let width = a.shape[axis];
    let inner = product_after_of(a.shape, axis);
    let outer = product_before_of(a.shape, axis);
    // The bounds count from the end when the step is negative.
    var lo = start;
    var hi = stop;
    if step > 0 {
        if lo < 0 { lo += width; }
        if hi < 0 { hi += width; }
        if lo < 0 { lo = 0; }
        if hi > width { hi = width; }
        if lo > hi { hi = lo; }
    } else {
        if lo < 0 { lo += width; }
        if hi < 0 { hi += width; }
        if lo > width - 1 { lo = width - 1; }
        if hi < -1 { hi = -1; }
        if lo < hi { hi = lo; }
    }
    // The count of the elements that the slice keeps.
    var count: i64 = 0;
    if step > 0 {
        var k = lo;
        while k < hi { count += 1; k += step; }
    } else {
        var k = lo;
        while k > hi { count += 1; k += step; }
    }
    if count < 1 {
        return empty_arr(), fault(F_SHAPE, "slice_axis", "the slice holds no element");
    }
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var k = lo;
        var c: i64 = 0;
        while c < count {
            var j: i64 = 0;
            while j < inner {
                d.push(a.data[o * width * inner + k * inner + j]);
                j += 1;
            }
            k += step;
            c += 1;
        }
        o += 1;
    }
    var s = same_shape(a);
    s[axis] = count;
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}

// take gives the elements at the flat positions of an index array.
fn take(a, index: i64[:]) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < index.len {
        if index[i] < 0 || index[i] >= a.data.len {
            return empty_arr(), fault(F_RANGE, "take", "a position stands outside the array");
        }
        d.push(a.data[index[i]]);
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(index.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// put writes the values into the elements at the flat positions of an
// index array. A position outside the array is a fault and the array
// stays as it was.
fn put(a, index: i64[:], values: f64[:]) -> Fault {
    if index.len != values.len {
        return fault(F_SHAPE, "put", "the index and the values hold a different count");
    }
    var i: i64 = 0;
    while i < index.len {
        if index[i] < 0 || index[i] >= a.data.len {
            return fault(F_RANGE, "put", "a position stands outside the array");
        }
        i += 1;
    }
    i = 0;
    while i < index.len {
        a.data[index[i]] = values[i];
        i += 1;
    }
    return no_fault();
}

// nonzero gives the flat positions of the elements that are not zero.
fn nonzero(a) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        if a.data[i] != 0.0 { d.push(i as f64); }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// flatnonzero gives the flat positions of the elements that are not
// zero, in the order of the elements.
fn flatnonzero(a) -> Arr, Fault {
    return nonzero(a);
}

// argwhere gives the position of every element that is not zero. The
// result has one row per element and one column per dimension. No
// match gives a result of zero rows, as NumPy gives it.
fn argwhere(a) -> Arr, Fault {
    var d: f64[>..] = [];
    var found: i64 = 0;
    var i: i64 = 0;
    while i < a.data.len {
        if a.data[i] != 0.0 {
            found += 1;
            var rest = i;
            var k: i64 = a.ndim - 1;
            var pos: i64[..8] = [];
            while k >= 0 { pos.push(0); k -= 1; }
            k = a.ndim - 1;
            while k >= 0 {
                pos[k] = rest % a.shape[k];
                rest /= a.shape[k];
                k -= 1;
            }
            var j: i64 = 0;
            while j < a.ndim { d.push(pos[j] as f64); j += 1; }
        }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(found);
    s.push(a.ndim);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// mask gives the elements that hold true. A zero is false and any
// other value is true.
fn mask(a, cond) -> Arr, Fault {
    let total = a.data.len;
    let cn = cond.data.len;
    if cn != total {
        return empty_arr(), fault(F_SHAPE, "mask", "the mask and the array hold a different count");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < total {
        if cond.data[i] != 0.0 { d.push(a.data[i]); }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// take_along_axis gives the elements at the positions of an index
// array along one axis. The index holds one position for each element
// of the other axes.
fn take_along_axis(a, index, axis: i64) -> Arr, Fault {
    if axis < 0 || axis >= a.ndim {
        return empty_arr(), fault(F_RANGE, "take_along_axis", "the axis stands outside the shape");
    }
    if index.ndim != a.ndim {
        return empty_arr(), fault(F_SHAPE, "take_along_axis", "the index and the array hold a different count of dimensions");
    }
    let inner = product_after_of(a.shape, axis);
    let outer = product_before_of(a.shape, axis);
    let width = a.shape[axis];
    let iw = index.shape[axis];
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var k: i64 = 0;
        while k < iw {
            var j: i64 = 0;
            while j < inner {
                let src = o * iw * inner + k * inner + j;
                let pick = index.data[src];
                if pick < 0.0 || pick >= width as f64 {
                    return empty_arr(), fault(F_RANGE, "take_along_axis", "a position stands outside the axis");
                }
                let at = o * width * inner + (pick as i64) * inner + j;
                d.push(a.data[at]);
                j += 1;
            }
            k += 1;
        }
        o += 1;
    }
    var s = same_shape(index);
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// repeat repeats every element a number of times.
fn repeat(a, count: i64) -> Arr, Fault {
    if count < 1 {
        return empty_arr(), fault(F_VALUE, "repeat", "the count holds a value below one");
    }
    let total = a.data.len * count;
    if total > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "repeat", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        var c: i64 = 0;
        while c < count { d.push(a.data[i]); c += 1; }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(total);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// tile repeats the whole array a number of times along one axis.
fn tile(a, count: i64, axis: i64) -> Arr, Fault {
    if count < 1 {
        return empty_arr(), fault(F_VALUE, "tile", "the count holds a value below one");
    }
    if axis < 0 || axis >= a.ndim {
        return empty_arr(), fault(F_RANGE, "tile", "the axis stands outside the shape");
    }
    let width = a.shape[axis];
    let inner = product_after_of(a.shape, axis);
    let outer = product_before_of(a.shape, axis);
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var c: i64 = 0;
        while c < count {
            var k: i64 = 0;
            while k < width {
                var j: i64 = 0;
                while j < inner {
                    d.push(a.data[o * width * inner + k * inner + j]);
                    j += 1;
                }
                k += 1;
            }
            c += 1;
        }
        o += 1;
    }
    var s = same_shape(a);
    s[axis] = width * count;
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}

// flatten gives a copy of the elements in one dimension.
fn flatten(a) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { d.push(a.data[i]); i += 1; }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// partition puts the elements below a pivot first and the rest after
// it. The pivot holds the first element that is not below the value.
fn partition(a, pivot: f64) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        if a.data[i] < pivot { d.push(a.data[i]); }
        i += 1;
    }
    let cut = d.len;
    i = 0;
    while i < a.data.len {
        if a.data[i] >= pivot { d.push(a.data[i]); }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(a.data.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// putmask writes a value where the mask holds true. The values come
// in a cycle when they are fewer than the places, as NumPy does. The
// array of the caller changes. A mask of a different length is a
// fault, and the check runs before the first write.
fn putmask(a, mask, values: f64[:]) -> Fault {
    if mask.data.len != a.data.len {
        return fault(F_SHAPE, "putmask", "the mask holds a different count");
    }
    if values.len == 0 {
        return fault(F_VALUE, "putmask", "the values hold no element");
    }
    var i: i64 = 0;
    while i < a.data.len {
        if mask.data[i] != 0.0 {
            a.data[i] = values[i % values.len];
        }
        i += 1;
    }
    return no_fault();
}

// place writes the values where the mask holds true, in their
// order. The values come in a cycle when they run out, as NumPy
// does. The array of the caller changes. A mask of a different
// length is a fault, and the check runs before the first write.
fn place(a, mask, values: f64[:]) -> Fault {
    if mask.data.len != a.data.len {
        return fault(F_SHAPE, "place", "the mask holds a different count");
    }
    if values.len == 0 {
        return fault(F_VALUE, "place", "the values hold no element");
    }
    var k: i64 = 0;
    var i: i64 = 0;
    while i < a.data.len {
        if mask.data[i] != 0.0 {
            a.data[i] = values[k % values.len];
            k += 1;
        }
        i += 1;
    }
    return no_fault();
}

// put_along_axis writes the values at the positions of an index
// array of the same shape, along one axis. The array of the caller
// changes. A position outside the axis is a fault, and the check of
// every position runs before the first write.
fn put_along_axis(a, index, values, axis: i64) -> Fault {
    if axis < 0 || axis >= a.ndim {
        return fault(F_RANGE, "put_along_axis", "the axis stands outside the shape");
    }
    if index.ndim != a.ndim || values.ndim != a.ndim {
        return fault(F_SHAPE, "put_along_axis", "the operands hold a different count of dimensions");
    }
    var same = true;
    var i: i64 = 0;
    while i < a.ndim {
        if index.shape[i] != a.shape[i] || values.shape[i] != a.shape[i] { same = false; }
        i += 1;
    }
    if !same {
        return fault(F_SHAPE, "put_along_axis", "the shapes of the operands do not agree");
    }
    // The check of every position runs first, so a bad index leaves
    // the array as it was.
    let total = a.data.len;
    i = 0;
    while i < total {
        let pick = index.data[i];
        let width = a.shape[axis];
        if pick < 0.0 || pick >= width as f64 {
            return fault(F_RANGE, "put_along_axis", "a position stands outside the axis");
        }
        i += 1;
    }
    i = 0;
    while i < total {
        let width = a.shape[axis];
        let inner = product_after_of(a.shape, axis);
        let outer = product_before_of(a.shape, axis);
        let flat = i;
        let j = flat % inner;
        let rest = flat / inner;
        let k = rest % width;
        let o = rest / width;
        a.data[o * width * inner + (index.data[i] as i64) * inner + j] = values.data[i];
        i += 1;
    }
    return no_fault();
}

// Parts holds the pieces of a cut. The shape holds the shape of the
// array that was cut, and the axis holds the axis of the cut. The
// buffer holds the length of every piece along the axis in two
// places each, and the count of its elements in the other place. The
// rest of the buffer holds the elements of the pieces in their order.
struct Parts { count: i64, ndim: i64, axis: i64, shape: i64[..8], buf: f64[>..] }

// part gives the piece of a number. A number outside the count of
// the pieces is a fault.
fn part(p, index: i64) -> Arr, Fault {
    if index < 0 || index >= p.count {
        return empty_arr(), fault(F_RANGE, "part", "the number stands outside the pieces");
    }
    let width = p.buf[index * 2] as i64;
    let total = p.buf[index * 2 + 1] as i64;
    var offset: i64 = 0;
    var i: i64 = 0;
    while i < index { offset += p.buf[i * 2 + 1] as i64; i += 1; }
    var d: f64[>..] = [];
    i = 0;
    while i < total { d.push(p.buf[p.count * 2 + offset + i]); i += 1; }
    // The shape of the piece holds the shape of the array with the
    // axis of the cut changed.
    var s: i64[..8] = [];
    i = 0;
    while i < p.ndim { s.push(p.shape[i]); i += 1; }
    s[p.axis] = width;
    return Arr { ndim: p.ndim, shape: s, data: d }, no_fault();
}

// split cuts an array along one axis into pieces of an equal count.
// A count that does not cut the axis is a fault, as in NumPy.
fn split(a, pieces: i64, axis: i64) -> Parts, Fault {
    if axis < 0 || axis >= a.ndim {
        return empty_parts(), fault(F_RANGE, "split", "the axis stands outside the shape");
    }
    if pieces < 1 {
        return empty_parts(), fault(F_VALUE, "split", "the count of pieces holds a value below one");
    }
    let width = a.shape[axis];
    if width % pieces != 0 {
        return empty_parts(), fault(F_VALUE, "split", "the count of pieces does not cut the axis");
    }
    return cut_axis(a, pieces, axis, false);
}

// array_split cuts an array along one axis into pieces of a count as
// equal as the axis allows. The first pieces of the cut hold one
// element more, as NumPy does.
fn array_split(a, pieces: i64, axis: i64) -> Parts, Fault {
    if axis < 0 || axis >= a.ndim {
        return empty_parts(), fault(F_RANGE, "array_split", "the axis stands outside the shape");
    }
    if pieces < 1 {
        return empty_parts(), fault(F_VALUE, "array_split", "the count of pieces holds a value below one");
    }
    return cut_axis(a, pieces, axis, true);
}

// cut_axis does the work of the two cuts. The flag tells whether the
// pieces may hold different counts.
fn cut_axis(a, pieces: i64, axis: i64, uneven: bool) -> Parts, Fault {
    let width = a.shape[axis];
    let inner = product_after_of(a.shape, axis);
    let outer = product_before_of(a.shape, axis);
    // The count of every piece. An even cut gives every piece the
    // same count. An uneven cut gives the first pieces one more.
    var sizes: i64[>..] = [];
    var base = width / pieces;
    var extra = width % pieces;
    var p: i64 = 0;
    while p < pieces {
        var size = base;
        if uneven && extra > 0 { size += 1; extra -= 1; }
        sizes.push(size);
        p += 1;
    }
    var d: f64[>..] = [];
    var start: i64 = 0;
    p = 0;
    while p < pieces {
        let size = sizes[p];
        var o: i64 = 0;
        while o < outer {
            var k: i64 = 0;
            while k < size {
                var j: i64 = 0;
                while j < inner {
                    d.push(a.data[o * width * inner + (start + k) * inner + j]);
                    j += 1;
                }
                k += 1;
            }
            o += 1;
        }
        start += size;
        p += 1;
    }
    // The buffer holds the length and the count of every piece
    // first, then the elements.
    var buf: f64[>..] = [];
    p = 0;
    while p < pieces {
        buf.push(sizes[p] as f64);
        buf.push((sizes[p] * inner * outer) as f64);
        p += 1;
    }
    p = 0;
    while p < d.len { buf.push(d[p]); p += 1; }
    var shape: i64[..8] = [];
    var q: i64 = 0;
    while q < a.ndim { shape.push(a.shape[q]); q += 1; }
    return Parts { count: pieces, ndim: a.ndim, axis: axis, shape: shape, buf: buf }, no_fault();
}

// empty_parts gives the pieces of no array.
fn empty_parts() -> Parts {
    return Parts { count: 0, ndim: 0, axis: 0, shape: [], buf: [] };
}

// hsplit cuts a matrix along its second axis.
fn hsplit(a, parts: i64) -> Parts, Fault {
    if a.ndim != 2 {
        return empty_parts(), fault(F_SHAPE, "hsplit", "the array holds a count of dimensions that is not two");
    }
    return split(a, parts, 1);
}

// vsplit cuts a matrix along its first axis.
fn vsplit(a, parts: i64) -> Parts, Fault {
    if a.ndim != 2 {
        return empty_parts(), fault(F_SHAPE, "vsplit", "the array holds a count of dimensions that is not two");
    }
    return split(a, parts, 0);
}
