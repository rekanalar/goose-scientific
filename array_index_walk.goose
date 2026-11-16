// The walk over the places of an array: ndindex and ndenumerate, and the
// memory questions may_share_memory and shares_memory.
//
// The language holds no iterator, so a walk gives its places as one
// array of shape (count, ndim), in row-major order.
namespace scientific;

// walk_rows appends every place of a shape to a buffer, in row-major
// order. The caller has checked the shape.
fn walk_rows(shape: i64[:], d: f64[>..]&) {
    let n = numel(shape);
    var pos: i64[..8] = [];
    var i: i64 = 0;
    while i < shape.len { pos.push(0); i += 1; }
    var cell: i64 = 0;
    while cell < n {
        i = 0;
        while i < shape.len { d.push(pos[i]); i += 1; }
        var q = shape.len - 1;
        while q >= 0 {
            pos[q] += 1;
            if pos[q] < shape[q] { break; }
            pos[q] = 0;
            q -= 1;
        }
        cell += 1;
    }
}

// ndindex gives every place of a shape, one row per place.
fn ndindex(shape: i64[:]) -> Arr, Fault {
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, "ndindex", f.detail); }
    let n = numel(shape);
    if n * shape.len > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "ndindex", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    walk_rows(shape, d);
    var s: i64[..8] = [];
    s.push(n);
    s.push(shape.len);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// ndenumerate gives the place of every element of an array, one row
// per element, and the values in the same order. NumPy gives an iterator
// of pairs.
fn ndenumerate(a) -> Arr, Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "ndenumerate");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let p, fp = ndindex(a.shape);
    if fp.kind != F_NONE { return empty_arr(), empty_arr(), fault(fp.kind, "ndenumerate", fp.detail); }
    let v, fv = ravel(a);
    if fv.kind != F_NONE { return empty_arr(), empty_arr(), fv; }
    return p, v, no_fault();
}

// may_share_memory tells whether two arrays may share memory. The package
// binds an array by reference and copies on assignment, and the language
// gives no address of a buffer. Two arrays of the package never share
// memory, so the answer is false for every pair.
fn may_share_memory(a, b) -> bool { return false; }

// shares_memory tells whether two arrays share memory. The answer is
// false for every pair, for the reason of may_share_memory.
fn shares_memory(a, b) -> bool { return false; }
