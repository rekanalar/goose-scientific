// The creation functions of the array.
//
// Every function checks its shape against the limits and returns a
// fault instead of an abort. A fault result is an empty array.
namespace scientific;

import math;

// zeros builds an array where every element holds 0.0.
fn zeros(shape: i64[:]) -> Arr, Fault {
    return fill(shape, 0.0);
}

// ones builds an array where every element holds 1.0.
fn ones(shape: i64[:]) -> Arr, Fault {
    return fill(shape, 1.0);
}

// full builds an array where every element holds v.
fn full(shape: i64[:], v: f64) -> Arr, Fault {
    return fill(shape, v);
}

// arange builds the numbers from start up to stop, step apart. The
// stop value is not part of the result, as in NumPy. A step of zero is
// a fault. The count stays under MAX_ELEMENTS.
fn arange(start: f64, stop: f64, step: f64) -> Arr, Fault {
    if step == 0.0 {
        return empty_arr(), fault(F_VALUE, "arange", "the step holds zero");
    }
    var count: i64 = 0;
    if step > 0.0 && stop > start { count = ceil_of((stop - start) / step); }
    if step < 0.0 && stop < start { count = ceil_of((start - stop) / -step); }
    if count > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "arange", "the range holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var s: i64[..8] = [];
    var v: f64 = start;
    var i: i64 = 0;
    while i < count { d.push(v); v += step; i += 1; }
    s.push(count);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// ceil_of rounds a positive number up to the next whole number. The
// stdlib holds no ceil that a package can reach, so the truncation of
// the integer conversion does the job.
fn ceil_of(x: f64) -> i64 {
    var t: i64 = x as i64;
    if t < x { t += 1; }
    return t;
}

// linspace builds count numbers from start to stop. The stop value is
// part of the result, as in NumPy. The count must be at least 1.
fn linspace(start: f64, stop: f64, count: i64) -> Arr, Fault {
    if count < 1 {
        return empty_arr(), fault(F_VALUE, "linspace", "the count holds a value below one");
    }
    if count > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "linspace", "the count holds more than the limit");
    }
    var d: f64[>..] = [];
    var s: i64[..8] = [];
    let span: f64 = stop - start;
    var i: i64 = 0;
    while i < count {
        if count == 1 { d.push(start); } else { d.push(start + span * i / (count - 1)); }
        i += 1;
    }
    s.push(count);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// eye builds the n by n identity matrix.
fn eye(n: i64) -> Arr, Fault {
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "eye", "the size holds a value below one");
    }
    var shape: i64[>..] = [];
    shape.push(n);
    shape.push(n);
    var s2: i64[..8] = [];
    s2.push(n);
    s2.push(n);
    let total = n * n;
    if total > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "eye", "the matrix holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < total {
        if i % (n + 1) == 0 { d.push(1.0); } else { d.push(0.0); }
        i += 1;
    }
    return Arr { ndim: 2, shape: s2, data: d }, no_fault();
}

// diag builds the square matrix that holds values on its diagonal.
fn diag(values: f64[:]) -> Arr, Fault {
    let n = values.len;
    let total = n * n;
    if total > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "diag", "the matrix holds more elements than the limit");
    }
    var s: i64[..8] = [];
    s.push(n);
    s.push(n);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < total {
        if i % (n + 1) == 0 { d.push(values[i / (n + 1)]); } else { d.push(0.0); }
        i += 1;
    }
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// from_rows builds a rows by cols matrix from a flat row-major buffer.
fn from_rows(rows: i64, cols: i64, values: f64[:]) -> Arr, Fault {
    if rows < 1 || cols < 1 {
        return empty_arr(), fault(F_VALUE, "from_rows", "a dimension holds a length below one");
    }
    if rows * cols != values.len {
        return empty_arr(), fault(F_SHAPE, "from_rows", "the count of values differs from the shape");
    }
    var s: i64[..8] = [];
    s.push(rows);
    s.push(cols);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < values.len { d.push(values[i]); i += 1; }
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// from_values builds a one-dimensional array from a buffer.
fn from_values(values: f64[:]) -> Arr, Fault {
    if values.len < 1 {
        return empty_arr(), fault(F_VALUE, "from_values", "the buffer holds no value");
    }
    var s: i64[..8] = [];
    s.push(values.len);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < values.len { d.push(values[i]); i += 1; }
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// clone builds an independent array with the same shape and elements.
// Assignment binds a name and copies nothing, so two names of one
// array share their elements. A clone does not.
fn clone(a) -> Arr, Fault {
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    var d: f64[>..] = [];
    i = 0;
    while i < a.data.len { d.push(a.data[i]); i += 1; }
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}
