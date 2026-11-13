// The copy into an array, the required form of an array, the roll of an
// axis and the strided array.
//
// The copy writes into the buffer of the array of the caller. The
// source spreads over the shape of the destination. The mask holds the
// shape of the destination, and a value that is not zero means that the
// source is written.
//
// The required form gives a copy that owns its buffer and lies in
// row-major order. The package holds one type of numbers and one order
// of the memory.
//
// The roll of an axis moves an axis to a place. The strided array reads
// a buffer with a step. It gives a copy, where NumPy gives a view that
// shares the memory.
namespace scientific;

// copy_check_dst checks the destination of a copy. The buffer of the
// destination must match its shape, or the copy would write outside it.
fn copy_check_dst(dst: Arr&, tag: const u8[:]) -> Fault {
    if dst.ndim < 1 || dst.ndim > MAX_DIMS {
        return fault(F_SHAPE, tag, "the destination holds a count of dimensions that is not right");
    }
    let f = check_stack(dst.ndim, dst.shape, dst.data.len, 1, tag);
    if f.kind != F_NONE { return f; }
    return no_fault();
}

// copy_check_src checks that the source agrees with its own shape. The
// rule of the fit of the source is the rule of the broadcast, and the
// call to the broadcast gives the fault of kind F_SHAPE.
fn copy_check_src(src, tag: const u8[:]) -> Fault {
    // The source must agree with its own shape before any value is read.
    let fs = check_stack(src.ndim, src.shape, src.data.len, 1, tag);
    if fs.kind != F_NONE { return fault(fs.kind, tag, fs.detail); }
    return no_fault();
}

// copyto writes the values of the source into the array of the caller.
// The source spreads over the shape of the destination.
fn copyto(dst: Arr&, src) -> Fault {
    let f = copy_check_dst(dst, "copyto");
    if f.kind != F_NONE { return f; }
    let fc = copy_check_src(src, "copyto");
    if fc.kind != F_NONE { return fc; }
    // A source that holds the shape of the destination needs no spread.
    let s, fs = copy_spread(src, dst);
    if fs.kind != F_NONE { return fault(fs.kind, "copyto", fs.detail); }
    var i: i64 = 0;
    while i < dst.data.len {
        dst.data[i] = s.data[i];
        i += 1;
    }
    return no_fault();
}

// copy_spread gives the source in the shape of the destination. A source
// that already holds that shape is used as it is. A source that spreads
// gives a new buffer.
fn copy_spread(src, dst) -> Arr, Fault {
    let s = same_shape(dst);
    if src.ndim == dst.ndim {
        var i: i64 = 0;
        while i < src.ndim {
            if src.shape[i] != dst.shape[i] { break; }
            i += 1;
        }
        if i == src.ndim {
            let c, fc = clone(src);
            if fc.kind != F_NONE { return empty_arr(), fc; }
            return c, no_fault();
        }
    }
    return broadcast_to(src, s);
}

// copyto_where writes the values of the source into the array of the
// caller where the mask is not zero.
fn copyto_where(dst: Arr&, src, mask) -> Fault {
    let f = copy_check_dst(dst, "copyto_where");
    if f.kind != F_NONE { return f; }
    let fm = copy_check_src(mask, "copyto_where");
    if fm.kind != F_NONE { return fm; }
    let fc = copy_check_src(src, "copyto_where");
    if fc.kind != F_NONE { return fc; }
    let s, fs = copy_spread(src, dst);
    if fs.kind != F_NONE { return fault(fs.kind, "copyto_where", fs.detail); }
    // The mask spreads over the shape of the destination as well.
    let m, fm = copy_spread(mask, dst);
    if fm.kind != F_NONE { return fault(fm.kind, "copyto_where", fm.detail); }
    var i: i64 = 0;
    while i < dst.data.len {
        if m.data[i] != 0.0 { dst.data[i] = s.data[i]; }
        i += 1;
    }
    return no_fault();
}

// require gives the required form of an array. The type must be the type
// of the package, and the order must be one that the package holds. The
// result is a copy that owns its buffer.
fn require(a, dtype: const u8[:] = "", requirements: const u8[:] = "") -> Arr, Fault {
    if dtype != "" && dtype != "float64" && dtype != "f64" {
        return empty_arr(), fault(F_VALUE, "require", "the type must be float64 or f64");
    }
    if requirements != "" && requirements != "C" && requirements != "A" &&
        requirements != "E" && requirements != "O" && requirements != "W" {
        return empty_arr(), fault(F_VALUE, "require", "the order of the memory must be C, A, E, O or W");
    }
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "require");
    if f.kind != F_NONE { return empty_arr(), f; }
    let c, fc = clone(a);
    if fc.kind != F_NONE { return empty_arr(), fc; }
    return c, no_fault();
}

// rollaxis moves an axis to a place. The axis and the place are made
// positive, and the place falls by one when the axis stands before it,
// as in NumPy.
fn rollaxis(a, axis: i64, start: i64 = 0) -> Arr, Fault {
    // The array must agree with its own shape before any value is read.
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, "rollaxis");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    let n = a.ndim;
    let ax = fix_axis(axis, n);
    if ax < 0 {
        return empty_arr(), fault(F_RANGE, "rollaxis", "the axis stands outside the shape");
    }
    var place = start;
    if place < 0 { place += n; }
    if place < 0 || place > n {
        return empty_arr(), fault(F_RANGE, "rollaxis", "the place stands outside the count of the axes");
    }
    if ax < place { place -= 1; }
    return moveaxis(a, ax, place);
}

// as_strided reads a buffer with a step. The steps are in bytes, and
// every step must be a whole number of 8 bytes. The result is a copy.
fn as_strided(a, shape: i64[:], strides: i64[:]) -> Arr, Fault {
    // The array must agree with its own shape before any value is read.
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, "as_strided");
    if fa.kind != F_NONE { return empty_arr(), fa; }
    if shape.len != strides.len {
        return empty_arr(), fault(F_SHAPE, "as_strided", "the shape and the steps hold a different count of axes");
    }
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), f; }
    // Every step must be a whole number of 8 bytes, else the place of a
    // value of the result is not a whole number.
    var i: i64 = 0;
    while i < strides.len {
        let s = strides[i];
        if s != 8 * (s / 8) {
            return empty_arr(), fault(F_VALUE, "as_strided", "a step is not a whole number of 8 bytes");
        }
        i += 1;
    }
    // The steps give the places in bytes. The package holds one type of
    // 8 bytes, so the step in values is the step in bytes over eight.
    var step: i64[>..] = [];
    var i: i64 = 0;
    while i < strides.len { step.push(strides[i] / 8); i += 1; }
    let n = numel(shape);
    var d: f64[>..] = [];
    var cell: i64 = 0;
    var coord: i64[>..] = [];
    i = 0;
    while i < shape.len { coord.push(0); i += 1; }
    while cell < n {
        // The place of the value comes from the coordinates of the cell.
        batch_step(shape, coord, cell);
        var src: i64 = 0;
        var k: i64 = 0;
        while k < shape.len {
            src = src + coord[k] * step[k];
            k += 1;
        }
        // Every place must lie inside the buffer of the source.
        if src < 0 || src >= a.data.len {
            return empty_arr(), fault(F_RANGE, "as_strided", "a place of the result lies outside the buffer");
        }
        d.push(a.data[src]);
        cell += 1;
    }
    return arr_of(copy(d), shape), no_fault();
}
