// The diagonals, the triangles and the index functions: diagonal, tri, indices, unravel_index and unstack.
//
namespace scientific;

// diagonal gives the diagonal of two axes. The diagonal becomes the
// last axis of the result. The offset moves it above (positive) or
// below (negative) the main one. A diagonal with no element is a fault.
fn diagonal(a, offset: i64 = 0, axis1: i64 = 0, axis2: i64 = 1) -> Arr, Fault {
    if a.ndim < 2 {
        return empty_arr(), fault(F_SHAPE, "diagonal", "the array holds fewer than two dimensions");
    }
    let x = fix_axis(axis1, a.ndim);
    let y = fix_axis(axis2, a.ndim);
    if x < 0 || y < 0 {
        return empty_arr(), fault(F_RANGE, "diagonal", "an axis stands outside the shape");
    }
    if x == y {
        return empty_arr(), fault(F_VALUE, "diagonal", "the two axes are the same");
    }
    let n1 = a.shape[x];
    let n2 = a.shape[y];
    var len = n1;
    if offset >= 0 { len = n2 - offset; if n1 < len { len = n1; } }
    else { len = n1 + offset; if n2 < len { len = n2; } }
    if len <= 0 {
        return empty_arr(), fault(F_VALUE, "diagonal", "the diagonal holds no element");
    }
    var strides: i64[>..] = [];
    strides_of(a.shape, strides);
    var rest: i64[..8] = [];
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim {
        if i != x && i != y { rest.push(i); s.push(a.shape[i]); }
        i += 1;
    }
    s.push(len);
    var rs: i64 = 1;
    i = 0;
    while i < rest.len { rs *= a.shape[rest[i]]; i += 1; }
    var i0: i64 = 0;
    var j0: i64 = 0;
    if offset >= 0 { j0 = offset; } else { i0 = -offset; }
    var d: f64[>..] = [];
    var r: i64 = 0;
    while r < rs {
        // The place of the rest axes from the number r.
        var base: i64 = 0;
        var q = r;
        i = rest.len - 1;
        while i >= 0 {
            base += (q % a.shape[rest[i]]) * strides[rest[i]];
            q = q / a.shape[rest[i]];
            i -= 1;
        }
        var t: i64 = 0;
        while t < len {
            d.push(a.data[base + (i0 + t) * strides[x] + (j0 + t) * strides[y]]);
            t += 1;
        }
        r += 1;
    }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// fill_diagonal writes a value on the main diagonal, in place. A matrix
// that is taller than wide stops at the square unless wrap is true. An
// array of more than two dimensions must hold equal lengths.
fn fill_diagonal(a, value: f64, wrap: bool = false) -> Fault {
    if a.ndim < 2 {
        return fault(F_SHAPE, "fill_diagonal", "the array holds fewer than two dimensions");
    }
    var step: i64 = 1;
    var end = a.data.len;
    if a.ndim == 2 {
        step = a.shape[1] + 1;
        if !wrap {
            end = a.shape[1] * a.shape[1];
            if end > a.data.len { end = a.data.len; }
        }
    } else {
        var i: i64 = 1;
        var p: i64 = 1;
        while i < a.ndim {
            if a.shape[i] != a.shape[0] {
                return fault(F_SHAPE, "fill_diagonal", "an array of more than two dimensions needs equal lengths");
            }
            p *= a.shape[0];
            step += p;
            i += 1;
        }
    }
    var pos: i64 = 0;
    while pos < end {
        a.data[pos] = value;
        pos += step;
    }
    return no_fault();
}

fn two_rows(first: f64[:], second: f64[:]) -> Arr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < first.len { d.push(first[i]); i += 1; }
    i = 0;
    while i < second.len { d.push(second[i]); i += 1; }
    var s: i64[..8] = [];
    s.push(2);
    s.push(first.len);
    return Arr { ndim: 2, shape: s, data: d };
}

// tri gives a matrix of n rows and m columns. The entries on and
// below the diagonal k hold one and the others hold zero.
fn tri(n: i64, m: i64, k: i64 = 0) -> Arr, Fault {
    var shape: i64[..8] = [];
    shape.push(n);
    shape.push(m);
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), f; }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < m {
            if j <= i + k { d.push(1.0); } else { d.push(0.0); }
            j += 1;
        }
        i += 1;
    }
    return Arr { ndim: 2, shape: shape, data: d }, no_fault();
}

// tri_places gives the places of the lower (lower is true) or the upper
// triangle of a matrix of n rows and m columns, as two rows: the row
// numbers and the column numbers.
fn tri_places(n: i64, k: i64, m: i64, lower: bool, tag: const u8[:]) -> Arr, Fault {
    var cols = m;
    if cols == 0 { cols = n; }
    var shape: i64[..8] = [];
    shape.push(n);
    shape.push(cols);
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, tag, f.detail); }
    var rows: f64[>..] = [];
    var cs: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < cols {
            var hit = j >= i + k;
            if lower { hit = j <= i + k; }
            if hit { rows.push(i); cs.push(j); }
            j += 1;
        }
        i += 1;
    }
    if rows.len == 0 {
        return empty_arr(), fault(F_VALUE, tag, "the triangle holds no place");
    }
    return two_rows(rows, cs), no_fault();
}

// tril_indices gives the places of the lower triangle as two rows. The
// count m of columns of zero means a square matrix.
fn tril_indices(n: i64, k: i64 = 0, m: i64 = 0) -> Arr, Fault {
    return tri_places(n, k, m, true, "tril_indices");
}

// triu_indices gives the places of the upper triangle as two rows.
fn triu_indices(n: i64, k: i64 = 0, m: i64 = 0) -> Arr, Fault {
    return tri_places(n, k, m, false, "triu_indices");
}

// diag_indices gives the places of the main diagonal of an array of
// ndim axes of length n. The result holds one row per axis.
fn diag_indices(n: i64, ndim: i64 = 2) -> Arr, Fault {
    if n < 1 || ndim < 1 || ndim > MAX_DIMS {
        return empty_arr(), fault(F_RANGE, "diag_indices", "the length or the count of axes stands outside its range");
    }
    var d: f64[>..] = [];
    var a: i64 = 0;
    while a < ndim {
        var i: i64 = 0;
        while i < n { d.push(i); i += 1; }
        a += 1;
    }
    var s: i64[..8] = [];
    s.push(ndim);
    s.push(n);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// indices gives the grid of the places of an array of a shape. The
// result holds one more axis in front, of the length of the shape.
fn indices(shape: i64[:]) -> Arr, Fault {
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), f; }
    if shape.len >= MAX_DIMS {
        return empty_arr(), fault(F_LIMIT, "indices", "the shape holds too many dimensions");
    }
    let n = numel(shape);
    if n * shape.len > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "indices", "the result holds more elements than the limit");
    }
    var strides: i64[>..] = [];
    strides_of(shape, strides);
    var d: f64[>..] = [];
    var a: i64 = 0;
    while a < shape.len {
        var i: i64 = 0;
        while i < n { d.push((i / strides[a]) % shape[a]); i += 1; }
        a += 1;
    }
    var s: i64[..8] = [];
    s.push(shape.len);
    var i: i64 = 0;
    while i < shape.len { s.push(shape[i]); i += 1; }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// ix_ gives the open grid of two lists of places: a column of the first
// list and a row of the second. The package holds the pair of lists
// only.
fn ix_(first: i64[:], second: i64[:]) -> Arr, Arr, Fault {
    if first.len < 1 || second.len < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "ix_", "a list holds no place");
    }
    var c: f64[>..] = [];
    var r: f64[>..] = [];
    var i: i64 = 0;
    while i < first.len { c.push(first[i]); i += 1; }
    i = 0;
    while i < second.len { r.push(second[i]); i += 1; }
    var sc: i64[..8] = [];
    sc.push(first.len);
    sc.push(1);
    var sr: i64[..8] = [];
    sr.push(1);
    sr.push(second.len);
    return Arr { ndim: 2, shape: sc, data: c }, Arr { ndim: 2, shape: sr, data: r }, no_fault();
}

// unravel_index turns flat places into one row of places per axis of
// the shape. A place outside the array is a fault.
fn unravel_index(flat: i64[:], shape: i64[:]) -> Arr, Fault {
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), f; }
    if flat.len < 1 {
        return empty_arr(), fault(F_VALUE, "unravel_index", "the list holds no place");
    }
    let n = numel(shape);
    var strides: i64[>..] = [];
    strides_of(shape, strides);
    var d: f64[>..] = [];
    var a: i64 = 0;
    while a < shape.len {
        var i: i64 = 0;
        while i < flat.len {
            if flat[i] < 0 || flat[i] >= n {
                return empty_arr(), fault(F_RANGE, "unravel_index", "a place stands outside the array");
            }
            d.push((flat[i] / strides[a]) % shape[a]);
            i += 1;
        }
        a += 1;
    }
    var s: i64[..8] = [];
    s.push(shape.len);
    s.push(flat.len);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// ravel_multi_index turns one row of places per axis into flat places.
// The mode is "raise", "wrap" or "clip" for a place outside its axis.
fn ravel_multi_index(multi, shape: i64[:], mode: const u8[:] = "raise") -> Arr, Fault {
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), f; }
    if mode != "raise" && mode != "wrap" && mode != "clip" {
        return empty_arr(), fault(F_VALUE, "ravel_multi_index", "the mode holds an unknown name");
    }
    if multi.ndim != 2 || multi.shape[0] != shape.len {
        return empty_arr(), fault(F_SHAPE, "ravel_multi_index", "the table does not hold one row per axis");
    }
    var strides: i64[>..] = [];
    strides_of(shape, strides);
    let count = multi.shape[1];
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < count {
        var flat: i64 = 0;
        var a: i64 = 0;
        while a < shape.len {
            let v = multi.data[a * count + i];
            if v != v || v != mathshim::trunc_of(v) || v > 4503599627370496.0 || v < -4503599627370496.0 {
                return empty_arr(), fault(F_VALUE, "ravel_multi_index", "a place holds no whole number");
            }
            var p = v as i64;
            if mode == "wrap" {
                p = p % shape[a];
                if p < 0 { p += shape[a]; }
            } else if mode == "clip" {
                if p < 0 { p = 0; }
                if p >= shape[a] { p = shape[a] - 1; }
            } else if p < 0 || p >= shape[a] {
                return empty_arr(), fault(F_RANGE, "ravel_multi_index", "a place stands outside its axis");
            }
            flat += p * strides[a];
            a += 1;
        }
        d.push(flat);
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(count);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// unstack cuts an array into pieces along an axis, one per entry of
// the axis. Read a piece with unstack_part.
fn unstack(a, axis: i64 = 0) -> Parts, Fault {
    let ax = fix_axis(axis, a.ndim);
    if ax < 0 {
        return empty_parts(), fault(F_RANGE, "unstack", "the axis stands outside the shape");
    }
    return split(a, a.shape[ax], ax);
}

// unstack_part gives one piece of unstack with the axis of the cut
// dropped. A piece of an array of one dimension keeps its axis of
// length one, because the package holds no scalar array.
fn unstack_part(p, index: i64) -> Arr, Fault {
    let x, f = part(p, index);
    if f.kind != F_NONE { return empty_arr(), f; }
    if x.ndim == 1 { return x, no_fault(); }
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < x.ndim { if i != p.axis { s.push(x.shape[i]); } i += 1; }
    return Arr { ndim: s.len, shape: s, data: copy(x.data) }, no_fault();
}

// block_grid assembles four arrays into a grid of two rows and two columns.
// Use hstack and vstack for other grids.
fn block_grid(top_left, top_right, bottom_left, bottom_right) -> Arr, Fault {
    let t, ft = hstack(top_left, top_right);
    if ft.kind != F_NONE { return empty_arr(), ft; }
    let b, fb = hstack(bottom_left, bottom_right);
    if fb.kind != F_NONE { return empty_arr(), fb; }
    return vstack(t, b);
}
