// Small helpers of NumPy: array_equiv, asarray_chkfinite, asmatrix,
// divmod, dsplit, empty_like, iterable and real_if_close.
//
// The package holds one real array type Arr and one complex array type
// CArr. A call never changes the array of the caller. Every read of an
// array checks its shape and its buffer first, so a forged array gives
// a fault and not a crash.
namespace scientific;

import mathshim;

// The machine epsilon of a float64. real_if_close counts its tolerance
// in this unit, as NumPy does.
const RIC_EPS: f64 = 2.220446049250313e-16;

// array_equiv tells whether two arrays are shape consistent and hold
// equal elements. Shape consistent means that the shapes broadcast to
// one shape. Two shapes that do not broadcast give false and no fault,
// as in NumPy. The missing value is never equal to itself. A forged
// array gives a fault of kind F_SHAPE, and a result above the limit
// of the elements gives a fault of kind F_LIMIT.
fn array_equiv(a, b) -> bool, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, "array_equiv");
    if fa.kind != F_NONE { return false, fa; }
    let fb = check_stack(b.ndim, b.shape, b.data.len, 1, "array_equiv");
    if fb.kind != F_NONE { return false, fb; }
    let s, fs = bshape(a, b);
    if fs.kind == F_SHAPE { return false, no_fault(); }
    if fs.kind != F_NONE { return false, fs; }
    let total = numel(s);
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    bcast_indices(s, a.shape, b.shape, ia, ib);
    var i: i64 = 0;
    while i < total {
        if a.data[ia[i]] != b.data[ib[i]] { return false, no_fault(); }
        i += 1;
    }
    return true, no_fault();
}

// asarray_chkfinite gives a copy of the array of the caller. An element
// that is NaN or an infinity gives a fault of kind F_VALUE, as NumPy
// raises an error. The complex form checks the real part and the
// imaginary part of every element.
fn asarray_chkfinite(a: Arr&) -> Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "asarray_chkfinite");
    if f.kind != F_NONE { return empty_arr(), f; }
    if !all_finite(a.data) {
        return empty_arr(), fault(F_VALUE, "asarray_chkfinite", "the array holds NaN or an infinity");
    }
    return clone(a);
}

fn asarray_chkfinite(c: CArr&) -> CArr, Fault {
    let f = check_stack(c.ndim, c.shape, c.data.len, 2, "asarray_chkfinite");
    if f.kind != F_NONE { return carr_empty(), f; }
    if !all_finite(c.data) {
        return carr_empty(), fault(F_VALUE, "asarray_chkfinite", "the array holds NaN or an infinity");
    }
    return carr_make(c.shape, copy(c.data)), no_fault();
}

// asmatrix gives an array of two dimensions. An array of one dimension
// becomes one row. An array of two dimensions gives a copy. An array of
// more dimensions gives a fault, as NumPy raises an error. The package
// holds no matrix class, so the result is an ordinary array.
fn asmatrix(a: Arr&) -> Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "asmatrix");
    if f.kind != F_NONE { return empty_arr(), f; }
    if a.ndim > 2 {
        return empty_arr(), fault(F_SHAPE, "asmatrix", "the array holds more than two dimensions");
    }
    return atleast_2d(a);
}

// empty_like gives an array of the shape of the array of the caller.
// The package holds no uninitialized memory, so every element holds
// zero.
fn empty_like(a) -> Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "empty_like");
    if f.kind != F_NONE { return empty_arr(), f; }
    return zeros_like(a);
}

// iterable tells whether the value of the caller can be walked. An
// array gives true. A single number gives false. The package holds no
// iterator, so no other value reaches the call.
fn iterable(x: f64) -> bool { return false; }
fn iterable(a: Arr&) -> bool { return true; }
fn iterable(a: CArr&) -> bool { return true; }

// ric_limit gives the bound of the size of an imaginary part. A
// tolerance above one counts in machine epsilons. A tolerance of one
// or less is the bound itself.
fn ric_limit(tol: f64) -> f64 {
    if tol > 1.0 { return tol * RIC_EPS; }
    return tol;
}

// real_if_close gives the real parts of a complex array when every
// imaginary part is smaller than the tolerance. A tolerance above one
// counts in machine epsilons of float64, and the default is 100. A
// tolerance of one or less is an absolute size. The call gives the real
// parts and true when it drops the imaginary parts. It gives an empty
// array and false when it keeps them, so the caller keeps its own
// complex array. A NaN or an infinity in an imaginary part keeps them.
fn real_if_close(c: CArr&, tol: f64 = 100.0) -> Arr, bool, Fault {
    let f = check_stack(c.ndim, c.shape, c.data.len, 2, "real_if_close");
    if f.kind != F_NONE { return empty_arr(), false, f; }
    let limit = ric_limit(tol);
    var i: i64 = 1;
    while i < c.data.len {
        var m = c.data[i];
        if m < 0.0 { m = -m; }
        if !(m < limit) { return empty_arr(), false, no_fault(); }
        i += 2;
    }
    return real(c), true, no_fault();
}

// The real form returns the array of the caller, as NumPy returns an
// array that is not complex. The flag is true.
fn real_if_close(a: Arr&, tol: f64 = 100.0) -> Arr, bool, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "real_if_close");
    if f.kind != F_NONE { return empty_arr(), false, f; }
    let r, fc = clone(a);
    return r, true, fc;
}

// dm_pair gives the floor quotient and the remainder of one pair, as
// NumPy does. The remainder takes the sign of the divisor. A divisor
// of zero gives the quotient of the division and a remainder of NaN. A
// NaN, or an infinity over a finite divisor, gives NaN twice.
fn dm_pair(x: f64, y: f64) -> f64, f64 {
    let zero: f64 = 0.0;
    let nan = zero / zero;
    if y == 0.0 { return x / y, nan; }
    let r = mathshim::mod_of(x, y);
    if r != r { return nan, nan; }
    return mathshim::floor_divide_of(x, y), r;
}

// divmod gives the floor quotient and the remainder of two arrays under
// the broadcast rule. The quotient rounds down. The remainder takes the
// sign of the divisor.
fn divmod(a, b) -> Arr, Arr, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, "divmod");
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), fa; }
    let fb = check_stack(b.ndim, b.shape, b.data.len, 1, "divmod");
    if fb.kind != F_NONE { return empty_arr(), empty_arr(), fb; }
    let s, fs = bshape(a, b);
    if fs.kind != F_NONE { return empty_arr(), empty_arr(), fs; }
    let total = numel(s);
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    bcast_indices(s, a.shape, b.shape, ia, ib);
    var q: f64[>..] = [];
    var r: f64[>..] = [];
    var i: i64 = 0;
    while i < total {
        let pq, pr = dm_pair(a.data[ia[i]], b.data[ib[i]]);
        q.push(pq);
        r.push(pr);
        i += 1;
    }
    var s2: i64[..8] = [];
    i = 0;
    while i < s.len { s2.push(s[i]); i += 1; }
    return Arr { ndim: s.len, shape: s, data: q }, Arr { ndim: s2.len, shape: s2, data: r }, no_fault();
}

// dsplit cuts an array of three dimensions or more along its third axis
// into pieces of an equal count. A count that does not cut the axis is
// a fault, as in NumPy. Read a piece with part.
fn dsplit(a, parts: i64) -> Parts, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "dsplit");
    if f.kind != F_NONE { return empty_parts(), f; }
    if a.ndim < 3 {
        return empty_parts(), fault(F_SHAPE, "dsplit", "the array holds fewer than three dimensions");
    }
    return split(a, parts, 2);
}

// The second form cuts at a list of places. A place outside the axis is
// a fault, where NumPy cuts at the end of the axis.
fn dsplit(a, sections: i64[:]) -> Parts, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "dsplit");
    if f.kind != F_NONE { return empty_parts(), f; }
    if a.ndim < 3 {
        return empty_parts(), fault(F_SHAPE, "dsplit", "the array holds fewer than three dimensions");
    }
    return split(a, sections, 2);
}
