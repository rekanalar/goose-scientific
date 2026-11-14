// The calls that apply a function: over a list of axes and over the
// values of arrays, as numpy.apply_over_axes and numpy.vectorize.
//
// The caller passes the function as the block at the end of the call.
// The time of a block is the time of the caller. The package cannot
// interrupt a block that does not stop.
namespace scientific;

// MAX_APPLY_AXES is the largest count of axes that apply_over_axes takes.
const MAX_APPLY_AXES: i64 = 64;

// apply_over_axes calls the block on the array for each axis of the
// list, in order. Each result feeds the next call. A result that holds
// one dimension less gets an axis of length one at the place of the
// axis. A negative axis counts from the dimensions of the original
// array. An empty list gives a copy of the array.
fn apply_over_axes<F: fn(Arr&, i64) -> Arr>(a, axes: i64[:]) -> Arr, Fault {
    let tag = "apply_over_axes";
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    if axes.len > MAX_APPLY_AXES {
        return empty_arr(), fault(F_LIMIT, tag, "the list holds more axes than the limit");
    }
    var i: i64 = 0;
    var fixed: i64[>..] = [];
    while i < axes.len {
        let ax = fix_axis(axes[i], a.ndim);
        if ax < 0 {
            return empty_arr(), fault(F_RANGE, tag, "an axis stands outside the shape");
        }
        fixed.push(ax);
        i += 1;
    }
    var r = copy(a);
    i = 0;
    while i < axes.len {
        let ax = fixed[i];
        let nxt = F(copy(r), ax);
        if nxt.ndim < 1 || (nxt.ndim != r.ndim && nxt.ndim + 1 != r.ndim) {
            return empty_arr(), fault(F_SHAPE, tag, "the function gives an array of the wrong count of dimensions");
        }
        let g = check_stack(nxt.ndim, nxt.shape, nxt.data.len, 1, tag);
        if g.kind != F_NONE { return empty_arr(), g; }
        if nxt.ndim == r.ndim {
            r = copy(nxt);
        } else {
            let e, ef = expand_dims(nxt, ax);
            if ef.kind != F_NONE { return empty_arr(), ef; }
            r = copy(e);
        }
        i += 1;
    }
    return r, no_fault();
}

// vec_shape gives the shape that two shapes share under the broadcast
// rule. The shape is checked against the limit of the elements.
fn vec_shape(sa: i64[:], sb: i64[:]) -> i64[>..], Fault {
    var none: i64[>..] = [];
    let s, f = broadcast_shapes(sa, sb);
    if f.kind != F_NONE { return none, f; }
    let c = check_shape(s);
    if c.kind != F_NONE { return none, c; }
    return s, no_fault();
}

// vec_check tests an operand of vectorize before any read.
fn vec_check(a, tag: const u8[:]) -> Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if f.kind != F_NONE { return f; }
    if a.data.len == 0 {
        return fault(F_VALUE, tag, "an array holds no element, and the result type is not known");
    }
    return no_fault();
}

// vectorize calls the block on every element of the array. The result
// holds the shape of the array.
fn vectorize<F: fn(f64) -> f64>(a) -> Arr, Fault {
    let f = vec_check(a, "vectorize");
    if f.kind != F_NONE { return empty_arr(), f; }
    return uapply(a) { x => F(x) };
}

// vectorize2 calls the block on the elements of two arrays. The arrays
// broadcast to one shape, which the result holds.
fn vectorize2<F: fn(f64, f64) -> f64>(a, b) -> Arr, Fault {
    let fa = vec_check(a, "vectorize2");
    if fa.kind != F_NONE { return empty_arr(), fa; }
    let fb = vec_check(b, "vectorize2");
    if fb.kind != F_NONE { return empty_arr(), fb; }
    let s, fs = vec_shape(a.shape, b.shape);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    let total = numel(s);
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    bcast_indices(s, a.shape, b.shape, ia, ib);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < total {
        d.push(F(a.data[ia[i]], b.data[ib[i]]));
        i += 1;
    }
    return arr_of(copy(d), s), no_fault();
}

// vectorize3 calls the block on the elements of three arrays. The
// arrays broadcast to one shape, which the result holds.
fn vectorize3<F: fn(f64, f64, f64) -> f64>(a, b, c) -> Arr, Fault {
    let fa = vec_check(a, "vectorize3");
    if fa.kind != F_NONE { return empty_arr(), fa; }
    let fb = vec_check(b, "vectorize3");
    if fb.kind != F_NONE { return empty_arr(), fb; }
    let fc = vec_check(c, "vectorize3");
    if fc.kind != F_NONE { return empty_arr(), fc; }
    let s1, f1 = vec_shape(a.shape, b.shape);
    if f1.kind != F_NONE { return empty_arr(), f1; }
    let s, f2 = vec_shape(s1, c.shape);
    if f2.kind != F_NONE { return empty_arr(), f2; }
    let total = numel(s);
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    var ic: i64[>..] = [];
    var tmp: i64[>..] = [];
    bcast_indices(s, a.shape, b.shape, ia, ib);
    bcast_indices(s, c.shape, a.shape, ic, tmp);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < total {
        d.push(F(a.data[ia[i]], b.data[ib[i]], c.data[ic[i]]));
        i += 1;
    }
    return arr_of(copy(d), s), no_fault();
}
