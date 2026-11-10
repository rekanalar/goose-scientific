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
