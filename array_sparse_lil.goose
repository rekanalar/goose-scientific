// The sparse matrices of lists.
//
// A matrix of the type Lil holds, for each row, a list of the columns that
// carry a value and a list of the values, as the list of lists of SciPy
// does. The columns of a row stand in ascending order.
//
// One buffer holds both. It starts with one offset for each row and one
// more, and then it holds the entries. The offset of a row is the index of
// its first entry. An entry holds a column and a value. The entries stand
// in the order of the rows and then of the columns. A set shifts the later
// entries by one place, as a list of Python does.
namespace scientific;

import mathshim;

// Lil holds a sparse matrix of lists. The field n counts the entries.
struct Lil { rows: i64, cols: i64, n: i64, buf: f64[>..] }

fn lil_empty() -> Lil {
    return Lil { rows: 0, cols: 0, n: 0, buf: [] };
}

// lil_check_size checks the size of the buffer before it grows. The buffer
// holds one offset for each row and one more, and two values for each entry.
fn lil_check_size(rows: i64, n: i64, tag: const u8[:]) -> Fault {
    if rows + 1 + 2 * n > MAX_ELEMENTS {
        return fault(F_LIMIT, tag, "the buffer of the matrix holds more values than the limit");
    }
    return no_fault();
}

// lil_start gives the offset of the row r.
fn lil_start(l, r: i64) -> i64 {
    return l.buf[r] as i64;
}

// lil_col gives the column of the entry e.
fn lil_col(l, e: i64) -> f64 {
    return l.buf[l.rows + 1 + 2 * e];
}

// lil_val gives the value of the entry e.
fn lil_val(l, e: i64) -> f64 {
    return l.buf[l.rows + 1 + 2 * e + 1];
}

// lil_find searches the column j in the row r. It gives the index of the
// entry, or the index where the entry belongs, and whether it was found.
fn lil_find(l, r: i64, j: i64) -> i64, bool {
    var lo = lil_start(l, r);
    var hi = lil_start(l, r + 1);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let c = lil_col(l, mid) as i64;
        if c == j { return mid, true; }
        if c < j { lo = mid + 1; } else { hi = mid; }
    }
    return lo, false;
}

// lil_new gives a matrix without an entry.
fn lil_new(rows: i64, cols: i64) -> Lil, Fault {
    let c = csr_check(rows, cols, "lil_new");
    if c.kind != F_NONE { return lil_empty(), c; }
    let z = lil_check_size(rows, 0, "lil_new");
    if z.kind != F_NONE { return lil_empty(), z; }
    return Lil { rows: rows, cols: cols, n: 0, buf: nd_zeros(rows + 1) }, no_fault();
}

// lil_from_csr_tag builds a matrix from compressed rows. The compressed
// rows give the columns of a row in ascending order. A zero that the source
// stores stays. The tag names the call in a fault.
fn lil_from_csr_tag(s, tag: const u8[:]) -> Lil, Fault {
    let z = lil_check_size(s.rows, s.nnz, tag);
    if z.kind != F_NONE { return lil_empty(), z; }
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i <= s.rows { buf.push(csr_ptr(s, i)); i += 1; }
    var k: i64 = 0;
    while k < s.nnz {
        if mathshim::is_finite_of(csr_val(s, k)) == 0.0 {
            return lil_empty(), fault(F_VALUE, tag, "a value is not finite");
        }
        buf.push(csr_col(s, k)); buf.push(csr_val(s, k));
        k += 1;
    }
    return Lil { rows: s.rows, cols: s.cols, n: s.nnz, buf: buf }, no_fault();
}

// lil_from_csr builds a matrix from compressed rows.
fn lil_from_csr(s) -> Lil, Fault {
    let t, f = csr_sorted(s);
    if f.kind != F_NONE { return lil_empty(), f; }
    return lil_from_csr_tag(t, "lil_from_csr");
}

// csr_sorted gives compressed rows whose columns stand in ascending order.
fn csr_sorted(s) -> Csr, Fault {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    var i: i64 = 0;
    while i < s.rows {
        var k = csr_ptr(s, i);
        while k < csr_ptr(s, i + 1) {
            ri.push(i); ci.push(csr_col(s, k)); vv.push(csr_val(s, k));
            k += 1;
        }
        i += 1;
    }
    return csr_from_triplets(s.rows, s.cols, ri, ci, vv);
}

// lil_from_dense builds a matrix from a dense matrix.
fn lil_from_dense(a) -> Lil, Fault {
    if a.ndim != 2 {
        return lil_empty(), fault(F_SHAPE, "lil_from_dense", "the operand is not a matrix");
    }
    let s, f = csr_from_dense(a);
    if f.kind != F_NONE { return lil_empty(), f; }
    return lil_from_csr_tag(s, "lil_from_dense");
}

// lil_from_coo builds a matrix from coordinates. The duplicates add up first.
fn lil_from_coo(c) -> Lil, Fault {
    let s, f = coo_to_csr(c);
    if f.kind != F_NONE { return lil_empty(), f; }
    return lil_from_csr_tag(s, "lil_from_coo");
}

// lil_set sets the value at a place. A value of zero removes the place. An
// existing place keeps its position in the row, and a new column goes to its
// sorted place. A fault leaves the matrix as it was.
fn lil_set(l: Lil&, i: f64, j: f64, v: f64) -> Fault {
    let c = dok_place(l, i, j, "lil_set");
    if c.kind != F_NONE { return c; }
    if mathshim::is_finite_of(v) == 0.0 {
        return fault(F_VALUE, "lil_set", "the value is not finite");
    }
    let r = i as i64;
    let at, found = lil_find(l, r, j as i64);
    let base = l.rows + 1;
    if found {
        if v != 0.0 {
            l.buf[base + 2 * at + 1] = v;
            return no_fault();
        }
        // The later entries move down by one place.
        var e = at;
        while e < l.n - 1 {
            l.buf[base + 2 * e] = l.buf[base + 2 * e + 2];
            l.buf[base + 2 * e + 1] = l.buf[base + 2 * e + 3];
            e += 1;
        }
        l.buf.pop(); l.buf.pop();
        var q = r + 1;
        while q <= l.rows { l.buf[q] = l.buf[q] - 1.0; q += 1; }
        l.n -= 1;
        return no_fault();
    }
    if v == 0.0 { return no_fault(); }
    if l.n + 1 > MAX_SPARSE_NNZ {
        return fault(F_LIMIT, "lil_set", "the count of entries is above the limit");
    }
    let z = lil_check_size(l.rows, l.n + 1, "lil_set");
    if z.kind != F_NONE { return z; }
    // The later entries move up by one place.
    l.buf.push(0.0); l.buf.push(0.0);
    var e = l.n;
    while e > at {
        l.buf[base + 2 * e] = l.buf[base + 2 * e - 2];
        l.buf[base + 2 * e + 1] = l.buf[base + 2 * e - 1];
        e -= 1;
    }
    l.buf[base + 2 * at] = j;
    l.buf[base + 2 * at + 1] = v;
    var q = r + 1;
    while q <= l.rows { l.buf[q] = l.buf[q] + 1.0; q += 1; }
    l.n += 1;
    return no_fault();
}

// lil_get gives the value at a place, zero when the place holds none.
fn lil_get(l, i: f64, j: f64) -> f64, Fault {
    let c = dok_place(l, i, j, "lil_get");
    if c.kind != F_NONE { return 0.0, c; }
    let at, found = lil_find(l, i as i64, j as i64);
    if !found { return 0.0, no_fault(); }
    return lil_val(l, at), no_fault();
}

// lil_row gives the columns and the values of one row.
fn lil_row(l, i: f64) -> Arr, Arr, Fault {
    if !whole_place(i) || i < 0.0 || i >= l.rows {
        return empty_arr(), empty_arr(), fault(F_RANGE, "lil_row", "the row is not a whole row inside the matrix");
    }
    var cc: f64[>..] = [];
    var vv: f64[>..] = [];
    var e = lil_start(l, i as i64);
    let stop = lil_start(l, i as i64 + 1);
    while e < stop { cc.push(lil_col(l, e)); vv.push(lil_val(l, e)); e += 1; }
    return line_arr(copy(cc)), line_arr(copy(vv)), no_fault();
}

// lil_nnz gives the count of the entries.
fn lil_nnz(l) -> i64 {
    return l.n;
}

// lil_triplets gives the rows, the columns and the values in row-major order.
fn lil_triplets(l) -> f64[>..], f64[>..], f64[>..] {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    var r: i64 = 0;
    while r < l.rows {
        var e = lil_start(l, r);
        let stop = lil_start(l, r + 1);
        while e < stop {
            ri.push(r); ci.push(lil_col(l, e)); vv.push(lil_val(l, e));
            e += 1;
        }
        r += 1;
    }
    return ri, ci, vv;
}

// lil_to_coo gives the coordinates in row-major order.
fn lil_to_coo(l) -> Coo, Fault {
    let ri, ci, vv = lil_triplets(l);
    return coo_from_triplets(l.rows, l.cols, ri, ci, vv);
}

// lil_to_csr gives the compressed rows.
fn lil_to_csr(l) -> Csr, Fault {
    // The buffer holds the compressed rows already, so no sort is needed.
    var ptr: f64[>..] = [];
    var idx: f64[>..] = [];
    var val: f64[>..] = [];
    var r: i64 = 0;
    while r <= l.rows { ptr.push(l.buf[r]); r += 1; }
    var e: i64 = 0;
    while e < l.n { idx.push(lil_col(l, e)); val.push(lil_val(l, e)); e += 1; }
    return csr_from_parts(l.rows, l.cols, ptr, idx, val), no_fault();
}

// lil_to_dense gives the dense matrix.
fn lil_to_dense(l) -> Arr, Fault {
    if l.rows * l.cols > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "lil_to_dense", "the dense matrix holds more elements than the limit");
    }
    var out = nd_zeros(l.rows * l.cols);
    var r: i64 = 0;
    while r < l.rows {
        var e = lil_start(l, r);
        let stop = lil_start(l, r + 1);
        while e < stop {
            out[r * l.cols + (lil_col(l, e) as i64)] = lil_val(l, e);
            e += 1;
        }
        r += 1;
    }
    return mat2(copy(out), l.rows, l.cols), no_fault();
}

// lil_from_triplets builds a matrix from entries. The entries hold no place twice.
fn lil_from_triplets(rows: i64, cols: i64, ri: f64[:], ci: f64[:], vv: f64[:], tag: const u8[:]) -> Lil, Fault {
    let s, f = csr_from_triplets(rows, cols, ri, ci, vv);
    if f.kind != F_NONE { return lil_empty(), f; }
    return lil_from_csr_tag(s, tag);
}

// lil_transpose swaps the rows and the columns. The columns of each row of
// the result stand in ascending order.
fn lil_transpose(l) -> Lil, Fault {
    let ri, ci, vv = lil_triplets(l);
    return lil_from_triplets(l.cols, l.rows, ci, ri, vv, "lil_transpose");
}

// lil_scale multiplies every value by a number. A zero stays.
fn lil_scale(l, a: f64) -> Lil, Fault {
    if mathshim::is_finite_of(a) == 0.0 {
        return lil_empty(), fault(F_VALUE, "lil_scale", "the factor is not finite");
    }
    var buf = copy(l.buf);
    var e: i64 = 0;
    while e < l.n {
        let at = l.rows + 1 + 2 * e + 1;
        buf[at] = buf[at] * a;
        if mathshim::is_finite_of(buf[at]) == 0.0 {
            return lil_empty(), fault(F_VALUE, "lil_scale", "a value is not finite");
        }
        e += 1;
    }
    return Lil { rows: l.rows, cols: l.cols, n: l.n, buf: buf }, no_fault();
}

// lil_matvec gives the product of a matrix and a vector.
fn lil_matvec(l, x: f64[:]) -> Arr, Fault {
    if x.len != l.cols {
        return empty_arr(), fault(F_SHAPE, "lil_matvec", "the vector does not hold one value per column");
    }
    var out = nd_zeros(l.rows);
    var r: i64 = 0;
    while r < l.rows {
        var e = lil_start(l, r);
        let stop = lil_start(l, r + 1);
        var t: f64 = 0.0;
        while e < stop { t += lil_val(l, e) * x[lil_col(l, e) as i64]; e += 1; }
        out[r] = t;
        r += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// lil_diagonal gives the values on the main diagonal.
fn lil_diagonal(l) -> Arr, Fault {
    var n = l.rows;
    if l.cols < n { n = l.cols; }
    var out = nd_zeros(n);
    var i: i64 = 0;
    while i < n {
        let at, found = lil_find(l, i, i);
        if found { out[i] = lil_val(l, at); }
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// lil_sum gives the sum of the values. A sum that is not finite is a fault.
fn lil_sum(l) -> f64, Fault {
    var t: f64 = 0.0;
    var e: i64 = 0;
    while e < l.n { t += lil_val(l, e); e += 1; }
    if mathshim::is_finite_of(t) == 0.0 {
        return 0.0, fault(F_VALUE, "lil_sum", "the sum is not finite");
    }
    return t, no_fault();
}

// lil_assign gives the matrix l the shape and the entries of the matrix m.
fn lil_assign(l: Lil&, m: Lil&) {
    l.rows = m.rows;
    l.cols = m.cols;
    l.n = m.n;
    l.buf = copy(m.buf);
}

// lil_setdiag sets the values of one diagonal. A value of zero removes its
// place. A short list leaves the later places alone, and a long list is cut.
// A fault leaves the matrix as it was.
fn lil_setdiag(l: Lil&, k: i64, vals: f64[:]) -> Fault {
    // The offset is checked before it is added to anything. A very large
    // offset would overflow the length of the diagonal.
    if k > l.cols || k < -l.rows {
        return fault(F_RANGE, "lil_setdiag", "the offset lies outside the matrix");
    }
    let len = dia_len_of(l.rows, l.cols, k);
    if len < 1 {
        return fault(F_RANGE, "lil_setdiag", "the offset lies outside the matrix");
    }
    var m = len;
    if vals.len < m { m = vals.len; }
    var t: i64 = 0;
    while t < m {
        if mathshim::is_finite_of(vals[t]) == 0.0 {
            return fault(F_VALUE, "lil_setdiag", "a value is not finite");
        }
        t += 1;
    }
    let first = dia_first_of(k);
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    var r: i64 = 0;
    while r < l.rows {
        var e = lil_start(l, r);
        let stop = lil_start(l, r + 1);
        let dt = r - first;
        while e < stop {
            let c = lil_col(l, e);
            if (c as i64) == r + k && dt >= 0 && dt < m {
                if vals[dt] != 0.0 { ri.push(r); ci.push(c); vv.push(vals[dt]); }
            } else {
                ri.push(r); ci.push(c); vv.push(lil_val(l, e));
            }
            e += 1;
        }
        r += 1;
    }
    t = 0;
    while t < m {
        let at, found = lil_find(l, first + t, first + t + k);
        if !found && vals[t] != 0.0 {
            ri.push(first + t); ci.push(first + t + k); vv.push(vals[t]);
        }
        t += 1;
    }
    var out, f = lil_from_triplets(l.rows, l.cols, ri, ci, vv, "lil_setdiag");
    if f.kind != F_NONE { return f; }
    lil_assign(l, out);
    return no_fault();
}

// lil_resize changes the shape. The entries outside the new shape go. A
// larger shape keeps every entry and adds empty rows. One pass builds the
// new buffer, because the kept entries stay in order.
fn lil_resize(l: Lil&, rows: i64, cols: i64) -> Fault {
    let c = csr_check(rows, cols, "lil_resize");
    if c.kind != F_NONE { return c; }
    let z = lil_check_size(rows, l.n, "lil_resize");
    if z.kind != F_NONE { return z; }
    var ptr: f64[>..] = [0.0];
    var body: f64[>..] = [];
    var r: i64 = 0;
    while r < rows {
        if r < l.rows {
            var e = lil_start(l, r);
            let stop = lil_start(l, r + 1);
            while e < stop {
                if lil_col(l, e) < cols { body.push(lil_col(l, e)); body.push(lil_val(l, e)); }
                e += 1;
            }
        }
        ptr.push((body.len / 2) as f64);
        r += 1;
    }
    var i: i64 = 0;
    while i < body.len { ptr.push(body[i]); i += 1; }
    l.n = body.len / 2;
    l.rows = rows;
    l.cols = cols;
    l.buf = copy(ptr);
    return no_fault();
}
