// The creation functions of the array.
//
// Every function checks its shape against the limits and returns a
// fault instead of an abort. A fault result is an empty array.
namespace scientific;

import mathshim;

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

// eye builds the identity matrix. The count of the rows and the count
// of the columns come from the caller. A count of columns of zero
// means the count of the rows. The offset moves the diagonal, as
// NumPy does.
fn eye(n: i64, m: i64 = 0, k: i64 = 0) -> Arr, Fault {
    var cols = m;
    if cols == 0 { cols = n; }
    if n < 1 || cols < 1 {
        return empty_arr(), fault(F_VALUE, "eye", "a size holds a value below one");
    }
    if n > MAX_ELEMENTS || cols > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "eye", "a size holds more than the limit");
    }
    var s2: i64[..8] = [];
    s2.push(n);
    s2.push(cols);
    let total = n * cols;
    if total > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "eye", "the matrix holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var row: i64 = 0;
    while row < n {
        var col: i64 = 0;
        while col < cols {
            if col - row == k { d.push(1.0); } else { d.push(0.0); }
            col += 1;
        }
        row += 1;
    }
    return Arr { ndim: 2, shape: s2, data: d }, no_fault();
}

// identity builds the n by n matrix of the identity. It is the name
// of NumPy for eye without an offset.
fn identity(n: i64) -> Arr, Fault { return eye(n); }

// array builds a one-dimensional array from a row of values. It is
// the name of NumPy for from_values.
fn array(values: f64[:]) -> Arr, Fault { return from_values(values); }

// diagflat builds the square matrix that holds a row of values on a
// diagonal of an offset, as NumPy does. The side of the matrix holds
// the count of the values and the size of the offset.
fn diagflat(values: f64[:], k: i64 = 0) -> Arr, Fault {
    let n = values.len;
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "diagflat", "the row holds no value");
    }
    if n > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "diagflat", "the row holds more values than the limit");
    }
    // The offset stands outside the size of the matrix, so it is a
    // fault. The check keeps the size of the matrix inside the limit.
    if k > MAX_ELEMENTS || k < -MAX_ELEMENTS {
        return empty_arr(), fault(F_RANGE, "diagflat", "the offset stands outside the matrix");
    }
    let side = n + max_i(k, -k);
    if side > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "diagflat", "the matrix holds more elements than the limit");
    }
    let total = side * side;
    if total > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "diagflat", "the matrix holds more elements than the limit");
    }
    var s: i64[..8] = [];
    s.push(side);
    s.push(side);
    var d: f64[>..] = [];
    var row: i64 = 0;
    while row < side {
        var col: i64 = 0;
        while col < side {
            var v: f64 = 0.0;
            if col - row == k {
                // The value sits at the smaller of the two places of
                // the diagonal.
                if row < col { v = values[row]; } else { v = values[col]; }
            }
            d.push(v);
            col += 1;
        }
        row += 1;
    }
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// logspace builds count numbers on a scale of a power. The numbers
// hold the power of the base, from the power of the start to the
// power of the stop, as NumPy does.
fn logspace(start: f64, stop: f64, count: i64, base: f64 = 10.0) -> Arr, Fault {
    let line, f = linspace(start, stop, count);
    if f.kind != F_NONE { return empty_arr(), f; }
    var i: i64 = 0;
    while i < line.data.len {
        line.data[i] = mathshim::power(base, line.data[i]);
        i += 1;
    }
    return line, no_fault();
}

// geomspace builds count numbers on a scale of a geometric sequence.
// The sign of the start comes off the numbers and goes back at the
// end, as NumPy does. A zero at the start or at the end is a fault.
fn geomspace(start: f64, stop: f64, count: i64) -> Arr, Fault {
    if count < 1 {
        return empty_arr(), fault(F_VALUE, "geomspace", "the count holds a value below one");
    }
    if start == 0.0 || stop == 0.0 {
        return empty_arr(), fault(F_VALUE, "geomspace", "a geometric sequence holds no zero");
    }
    var sign: f64 = 1.0;
    if start < 0.0 { sign = -1.0; }
    let line, f = logspace(mathshim::log10_of(start / sign), mathshim::log10_of(stop / sign), count, 10.0);
    if f.kind != F_NONE { return empty_arr(), f; }
    // The two ends hold the start and the stop, as NumPy does. The
    // other numbers hold the power of the ten of the scale.
    var i: i64 = 1;
    while i < line.data.len - 1 {
        line.data[i] = line.data[i] * sign;
        i += 1;
    }
    line.data[0] = start;
    if line.data.len > 1 { line.data[line.data.len - 1] = stop; }
    return line, no_fault();
}

// diag builds the square matrix that holds values on its diagonal. It
// is the same matrix that diagflat builds with no offset.
fn diag(values: f64[:]) -> Arr, Fault {
    let m, f = diagflat(values, 0);
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, "diag", f.detail); }
    return m, no_fault();
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

// empty builds an array of a shape with no element set. The package
// holds no uninitialized memory, so every element holds zero.
fn empty(shape: i64[:]) -> Arr, Fault {
    return zeros(shape);
}

// full_like builds an array of the shape of a with every element
// set to v.
fn full_like(a, v: f64) -> Arr, Fault {
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    return full(s, v);
}

// zeros_like builds an array of the shape of a with zeros.
fn zeros_like(a) -> Arr, Fault {
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    return zeros(s);
}

// ones_like builds an array of the shape of a with ones.
fn ones_like(a) -> Arr, Fault {
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    return ones(s);
}

// trim_zeros drops the zeros at the start and at the end of a row.
// A row of only zeros gives a fault, because it holds no element
// after the cut.
fn trim_zeros(a) -> Arr, Fault {
    if a.ndim != 1 {
        return empty_arr(), fault(F_SHAPE, "trim_zeros", "the array holds a count of dimensions that is not one");
    }
    var first: i64 = -1;
    var last: i64 = -1;
    var i: i64 = 0;
    while i < a.data.len {
        if a.data[i] != 0.0 {
            if first < 0 { first = i; }
            last = i;
        }
        i += 1;
    }
    if first < 0 {
        return empty_arr(), fault(F_VALUE, "trim_zeros", "every element holds zero");
    }
    var d: f64[>..] = [];
    i = first;
    while i <= last { d.push(a.data[i]); i += 1; }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// insert puts the values at the position of a row. The elements from
// the position move to the end. A position below zero or above the
// length is a fault.
fn insert(a, pos: i64, values: f64[:]) -> Arr, Fault {
    if a.ndim != 1 {
        return empty_arr(), fault(F_SHAPE, "insert", "the array holds a count of dimensions that is not one");
    }
    if pos < 0 || pos > a.data.len {
        return empty_arr(), fault(F_RANGE, "insert", "the position stands outside the array");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < pos { d.push(a.data[i]); i += 1; }
    i = 0;
    while i < values.len { d.push(values[i]); i += 1; }
    i = pos;
    while i < a.data.len { d.push(a.data[i]); i += 1; }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// tril holds the lower triangle of a matrix under the diagonal of a
// count. A count of zero holds the diagonal and below. A count above
// zero holds diagonals above it too.
fn tril(a, k: i64) -> Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "tril", "the array holds a count of dimensions that is not two");
    }
    let rows = a.shape[0];
    let cols = a.shape[1];
    var d: f64[>..] = [];
    var r: i64 = 0;
    while r < rows {
        var c: i64 = 0;
        while c < cols {
            if c <= r + k { d.push(a.data[r * cols + c]); } else { d.push(0.0); }
            c += 1;
        }
        r += 1;
    }
    var s: i64[..8] = [];
    s.push(rows);
    s.push(cols);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// triu holds the upper triangle of a matrix above the diagonal of a
// count.
fn triu(a, k: i64) -> Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "triu", "the array holds a count of dimensions that is not two");
    }
    let rows = a.shape[0];
    let cols = a.shape[1];
    var d: f64[>..] = [];
    var r: i64 = 0;
    while r < rows {
        var c: i64 = 0;
        while c < cols {
            if c >= r + k { d.push(a.data[r * cols + c]); } else { d.push(0.0); }
            c += 1;
        }
        r += 1;
    }
    var s: i64[..8] = [];
    s.push(rows);
    s.push(cols);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}
