// The sparse matrices of keys.
//
// A matrix of the type Dok holds the places that carry a value, in the
// order of their first insertion, as the dictionary of keys of SciPy does.
//
// The buffer holds, in this order, a table of cap slots and then the
// entries. A slot holds the index of an entry plus one, or zero when it is
// empty. An entry holds three values: the row, the column and the value.
// The table holds one and a half slots for each entry, so a lookup ends at
// an empty slot. The hash of a place holds a random number of the matrix,
// so a caller cannot choose places that collide.
namespace scientific;

import os;
import mathshim;

// Dok holds a sparse matrix of keys. The field n counts the entries, and
// the field cap counts the slots. The cap is a power of two.
struct Dok { rows: i64, cols: i64, n: i64, cap: i64, salt: u64, buf: f64[>..] }

// The largest count of slots.
const MAX_DOK_SLOTS: i64 = 268435456;

fn dok_empty() -> Dok {
    return Dok { rows: 0, cols: 0, n: 0, cap: 8, salt: 0, buf: [] };
}

// dok_hash gives the first slot to try for a place.
fn dok_hash(cap: i64, salt: u64, i: i64, j: i64) -> i64 {
    let c1: u64 = 11400714819323198485;
    let c2: u64 = 13787848793156543929;
    let c3: u64 = 10723151780598845931;
    var h: u64 = (salt + (i as u64) * c1 + (j as u64)) * c2;
    h = (h ^ (h >> 29)) * c3;
    h = h ^ (h >> 32);
    return ((h >> 11) as i64) & (cap - 1);
}

// dok_cap_for gives the count of slots for a count of entries. The count is
// a power of two. It holds one and a half slots for each entry, so the load
// of the table stays under two thirds.
fn dok_cap_for(n: i64) -> i64 {
    var cap: i64 = 8;
    while 2 * cap < 3 * n && cap < MAX_DOK_SLOTS { cap = cap * 2; }
    return cap;
}

// dok_find gives the index of the entry at a place, or minus one. The
// probe stops after cap steps.
fn dok_find(d, i: i64, j: i64) -> i64 {
    let mask = d.cap - 1;
    var s = dok_hash(d.cap, d.salt, i, j);
    var step: i64 = 0;
    while step < d.cap {
        let slot = d.buf[s] as i64;
        if slot == 0 { return -1; }
        let at = d.cap + 3 * (slot - 1);
        if d.buf[at] as i64 == i && d.buf[at + 1] as i64 == j { return slot - 1; }
        s = (s + 1) & mask;
        step += 1;
    }
    return -1;
}

// dok_row gives the row of the entry e.
fn dok_row(d, e: i64) -> f64 {
    return d.buf[d.cap + 3 * e];
}

// dok_col gives the column of the entry e.
fn dok_col(d, e: i64) -> f64 {
    return d.buf[d.cap + 3 * e + 1];
}

// dok_val gives the value of the entry e.
fn dok_val(d, e: i64) -> f64 {
    return d.buf[d.cap + 3 * e + 2];
}

// dok_set_val gives the value of the entry e.
fn dok_set_val(d: Dok&, e: i64, w: f64) {
    d.buf[d.cap + 3 * e + 2] = w;
}

// dok_entries gives a copy of the entries, three values each.
fn dok_entries(d) -> f64[>..] {
    var ent: f64[>..] = [];
    var i = d.cap;
    while i < d.buf.len { ent.push(d.buf[i]); i += 1; }
    return ent;
}

// dok_buffer builds the buffer of a matrix. The table comes first and the
// entries follow it. The salt of the matrix gives the hash of every place.
fn dok_buffer(cap: i64, salt: u64, ent: f64[:]) -> f64[>..] {
    let n = ent.len / 3;
    var buf = nd_zeros(cap);
    var i: i64 = 0;
    while i < n {
        var s = dok_hash(cap, salt, ent[3 * i] as i64, ent[3 * i + 1] as i64);
        while buf[s] != 0.0 { s = (s + 1) & (cap - 1); }
        buf[s] = i + 1;
        i += 1;
    }
    i = 0;
    while i < ent.len { buf.push(ent[i]); i += 1; }
    return buf;
}

// dok_make gives a matrix with the given entries. A value that is not
// finite is a fault.
fn dok_make(rows: i64, cols: i64, salt: u64, ent: f64[:], tag: const u8[:]) -> Dok, Fault {
    let n = ent.len / 3;
    var i: i64 = 0;
    while i < n {
        if mathshim::is_finite_of(ent[3 * i + 2]) == 0.0 {
            return dok_empty(), fault(F_VALUE, tag, "a value is not finite");
        }
        i += 1;
    }
    let cap = dok_cap_for(n);
    return Dok { rows: rows, cols: cols, n: n, cap: cap, salt: salt, buf: dok_buffer(cap, salt, ent) }, no_fault();
}

// dok_rebuild gives the matrix the given entries. The salt of the matrix
// stays, so the hash of every place stays.
fn dok_rebuild(d: Dok&, rows: i64, cols: i64, ent: f64[:]) {
    let cap = dok_cap_for(ent.len / 3);
    d.rows = rows;
    d.cols = cols;
    d.n = ent.len / 3;
    d.cap = cap;
    d.buf = dok_buffer(cap, d.salt, ent);
}

// dok_new gives a matrix without a key.
fn dok_new(rows: i64, cols: i64) -> Dok, Fault {
    let c = csr_check(rows, cols, "dok_new");
    if c.kind != F_NONE { return dok_empty(), c; }
    let none: f64[>..] = [];
    return dok_make(rows, cols, random_seed(), none, "dok_new"), no_fault();
}

// dok_place checks a place.
fn dok_place(d, i: f64, j: f64, tag: const u8[:]) -> Fault {
    if !whole_place(i) || !whole_place(j) || i < 0.0 || i >= d.rows || j < 0.0 || j >= d.cols {
        return fault(F_RANGE, tag, "the place is not a whole place inside the matrix");
    }
    return no_fault();
}

// dok_put stores a value at a place and keeps a zero. A fault leaves the
// matrix as it was.
fn dok_put(d: Dok&, i: i64, j: i64, v: f64, tag: const u8[:]) -> Fault {
    if mathshim::is_finite_of(v) == 0.0 {
        return fault(F_VALUE, tag, "the value is not finite");
    }
    let e = dok_find(d, i, j);
    if e >= 0 {
        dok_set_val(d, e, v);
        return no_fault();
    }
    if d.n + 1 > MAX_SPARSE_NNZ {
        return fault(F_LIMIT, tag, "the count of keys is above the limit");
    }
    // The table grows by a rebuild when it holds too many keys.
    if d.cap < dok_cap_for(d.n + 1) {
        var ent = dok_entries(d);
        ent.push(i); ent.push(j); ent.push(v);
        dok_rebuild(d, d.rows, d.cols, ent);
        return no_fault();
    }
    // The probe stops after cap steps. A matrix that a caller built by hand
    // can hold a table without an empty slot.
    var s = dok_hash(d.cap, d.salt, i, j);
    var step: i64 = 0;
    while d.buf[s] != 0.0 && step < d.cap { s = (s + 1) & (d.cap - 1); step += 1; }
    if d.buf[s] != 0.0 {
        return fault(F_VALUE, tag, "the table of the matrix holds no empty slot");
    }
    d.buf[s] = d.n + 1;
    d.buf.push(i); d.buf.push(j); d.buf.push(v);
    d.n += 1;
    return no_fault();
}

// dok_remove takes the entry e out and rebuilds the matrix.
fn dok_remove(d: Dok&, e: i64) {
    var ent: f64[>..] = [];
    var k: i64 = 0;
    while k < d.n {
        if k != e {
            ent.push(dok_row(d, k)); ent.push(dok_col(d, k)); ent.push(dok_val(d, k));
        }
        k += 1;
    }
    dok_rebuild(d, d.rows, d.cols, ent);
}

// dok_set sets the value at a place. A value of zero removes the place.
// An existing place keeps its position, and a new place goes to the end.
fn dok_set(d: Dok&, i: f64, j: f64, v: f64) -> Fault {
    let c = dok_place(d, i, j, "dok_set");
    if c.kind != F_NONE { return c; }
    if v == 0.0 {
        let e = dok_find(d, i as i64, j as i64);
        if e >= 0 { dok_remove(d, e); }
        return no_fault();
    }
    return dok_put(d, i as i64, j as i64, v, "dok_set");
}

// dok_pop removes a place and gives its value. A missing place gives zero.
fn dok_pop(d: Dok&, i: f64, j: f64) -> f64, Fault {
    let c = dok_place(d, i, j, "dok_pop");
    if c.kind != F_NONE { return 0.0, c; }
    let e = dok_find(d, i as i64, j as i64);
    if e < 0 { return 0.0, no_fault(); }
    let v = dok_val(d, e);
    dok_remove(d, e);
    return v, no_fault();
}

// dok_get gives the value at a place, zero when the place holds none.
fn dok_get(d, i: f64, j: f64) -> f64, Fault {
    let c = dok_place(d, i, j, "dok_get");
    if c.kind != F_NONE { return 0.0, c; }
    let e = dok_find(d, i as i64, j as i64);
    if e < 0 { return 0.0, no_fault(); }
    return dok_val(d, e), no_fault();
}

// dok_nnz gives the count of the keys.
fn dok_nnz(d) -> i64 {
    return d.n;
}

// dok_from_csr builds a matrix from compressed rows, row by row. A zero
// that the source stores stays.
fn dok_from_csr(s) -> Dok, Fault {
    return dok_from_csr_tag(s, "dok_from_csr");
}

// dok_from_csr_tag builds a matrix from compressed rows. The tag names the
// call in a fault.
fn dok_from_csr_tag(s, tag: const u8[:]) -> Dok, Fault {
    var ent: f64[>..] = [];
    var i: i64 = 0;
    while i < s.rows {
        var k = csr_ptr(s, i);
        while k < csr_ptr(s, i + 1) {
            ent.push(i); ent.push(csr_col(s, k)); ent.push(csr_val(s, k));
            k += 1;
        }
        i += 1;
    }
    return dok_make(s.rows, s.cols, random_seed(), ent, tag);
}

// dok_from_dense builds a matrix from a dense matrix, in the order of the rows.
fn dok_from_dense(a) -> Dok, Fault {
    if a.ndim != 2 {
        return dok_empty(), fault(F_SHAPE, "dok_from_dense", "the operand is not a matrix");
    }
    let s, f = csr_from_dense(a);
    if f.kind != F_NONE { return dok_empty(), f; }
    return dok_from_csr_tag(s, "dok_from_dense");
}

// dok_from_coo builds a matrix from coordinates. The duplicates add up
// first, and the keys then lie in the order of the rows, as SciPy gives them.
fn dok_from_coo(c) -> Dok, Fault {
    let s, f = coo_to_csr(c);
    if f.kind != F_NONE { return dok_empty(), f; }
    return dok_from_csr_tag(s, "dok_from_coo");
}

// dok_triplets gives the rows, the columns and the values in key order.
fn dok_triplets(d) -> f64[>..], f64[>..], f64[>..] {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    var e: i64 = 0;
    while e < d.n {
        ri.push(dok_row(d, e)); ci.push(dok_col(d, e)); vv.push(dok_val(d, e));
        e += 1;
    }
    return ri, ci, vv;
}

// dok_keys gives the rows, the columns and the values in the order of the keys.
fn dok_keys(d) -> Arr, Arr, Arr, Fault {
    let ri, ci, vv = dok_triplets(d);
    return line_arr(copy(ri)), line_arr(copy(ci)), line_arr(copy(vv)), no_fault();
}

// dok_to_coo gives the coordinates, in the order of the keys.
fn dok_to_coo(d) -> Coo, Fault {
    let ri, ci, vv = dok_triplets(d);
    return coo_from_triplets(d.rows, d.cols, ri, ci, vv);
}

// dok_to_csr gives the compressed rows.
fn dok_to_csr(d) -> Csr, Fault {
    let ri, ci, vv = dok_triplets(d);
    return csr_from_triplets(d.rows, d.cols, ri, ci, vv);
}

// dok_to_dense gives the dense matrix.
fn dok_to_dense(d) -> Arr, Fault {
    if d.rows * d.cols > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "dok_to_dense", "the dense matrix holds more elements than the limit");
    }
    var out = nd_zeros(d.rows * d.cols);
    var e: i64 = 0;
    while e < d.n {
        out[(dok_row(d, e) as i64) * d.cols + (dok_col(d, e) as i64)] = dok_val(d, e);
        e += 1;
    }
    return mat2(copy(out), d.rows, d.cols), no_fault();
}

// dok_transpose swaps the rows and the columns and keeps the order of the keys.
fn dok_transpose(d) -> Dok, Fault {
    var ent: f64[>..] = [];
    var e: i64 = 0;
    while e < d.n {
        ent.push(dok_col(d, e)); ent.push(dok_row(d, e)); ent.push(dok_val(d, e));
        e += 1;
    }
    return dok_make(d.cols, d.rows, d.salt, ent, "dok_transpose");
}

// dok_scale multiplies every value by a number. A zero stays.
fn dok_scale(d, a: f64) -> Dok, Fault {
    if mathshim::is_finite_of(a) == 0.0 {
        return dok_empty(), fault(F_VALUE, "dok_scale", "the factor is not finite");
    }
    var buf = copy(d.buf);
    var e: i64 = 0;
    while e < d.n {
        let at = d.cap + 3 * e + 2;
        buf[at] = buf[at] * a;
        if mathshim::is_finite_of(buf[at]) == 0.0 {
            return dok_empty(), fault(F_VALUE, "dok_scale", "a value is not finite");
        }
        e += 1;
    }
    return Dok { rows: d.rows, cols: d.cols, n: d.n, cap: d.cap, salt: d.salt, buf: buf }, no_fault();
}

// dok_matvec gives the product of a matrix and a vector.
fn dok_matvec(d, x: f64[:]) -> Arr, Fault {
    if x.len != d.cols {
        return empty_arr(), fault(F_SHAPE, "dok_matvec", "the vector does not hold one value per column");
    }
    var out = nd_zeros(d.rows);
    var e: i64 = 0;
    while e < d.n {
        let r = dok_row(d, e) as i64;
        out[r] = out[r] + dok_val(d, e) * x[dok_col(d, e) as i64];
        e += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// dok_diagonal gives the values on the main diagonal.
fn dok_diagonal(d) -> Arr, Fault {
    var n = d.rows;
    if d.cols < n { n = d.cols; }
    var out = nd_zeros(n);
    var i: i64 = 0;
    while i < n {
        let e = dok_find(d, i, i);
        if e >= 0 { out[i] = dok_val(d, e); }
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// dok_sum gives the sum of the values. A sum that is not finite is a fault.
fn dok_sum(d) -> f64, Fault {
    var t: f64 = 0.0;
    var e: i64 = 0;
    while e < d.n { t += dok_val(d, e); e += 1; }
    if mathshim::is_finite_of(t) == 0.0 {
        return 0.0, fault(F_VALUE, "dok_sum", "the sum is not finite");
    }
    return t, no_fault();
}

// dok_setdiag sets the values of one diagonal. A value of zero removes its
// place. A short list leaves the later places alone, and a long list is cut.
// The places of the diagonal that hold none join the keys at the end, in
// the order of the diagonal.
fn dok_setdiag(d: Dok&, k: i64, vals: f64[:]) -> Fault {
    // The offset is checked before it is added to anything. A very large
    // offset would overflow the length of the diagonal.
    if k > d.cols || k < -d.rows {
        return fault(F_RANGE, "dok_setdiag", "the offset lies outside the matrix");
    }
    let len = dia_len_of(d.rows, d.cols, k);
    if len < 1 {
        return fault(F_RANGE, "dok_setdiag", "the offset lies outside the matrix");
    }
    var m = len;
    if vals.len < m { m = vals.len; }
    var t: i64 = 0;
    while t < m {
        if mathshim::is_finite_of(vals[t]) == 0.0 {
            return fault(F_VALUE, "dok_setdiag", "a value is not finite");
        }
        t += 1;
    }
    let first = dia_first_of(k);
    // The entries of the diagonal that the matrix holds already. A place of
    // the diagonal holds none when it does not lie in the list.
    var found: i64[>..] = [];
    t = 0;
    while t < m {
        found.push(dok_find(d, first + t, first + t + k));
        t += 1;
    }
    // The entries that keep their place, in the order of the keys.
    var ent: f64[>..] = [];
    var e: i64 = 0;
    while e < d.n {
        var at = -1;
        t = 0;
        while t < m {
            if found[t] == e { at = t; t = m; } else { t += 1; }
        }
        if at < 0 {
            ent.push(dok_row(d, e)); ent.push(dok_col(d, e)); ent.push(dok_val(d, e));
        } else if vals[at] != 0.0 {
            ent.push(dok_row(d, e)); ent.push(dok_col(d, e)); ent.push(vals[at]);
        }
        e += 1;
    }
    // The places of the diagonal that hold none join at the end.
    t = 0;
    while t < m {
        if found[t] < 0 && vals[t] != 0.0 {
            ent.push(first + t); ent.push(first + t + k); ent.push(vals[t]);
        }
        t += 1;
    }
    if ent.len / 3 > MAX_SPARSE_NNZ {
        return fault(F_LIMIT, "dok_setdiag", "the count of keys is above the limit");
    }
    dok_rebuild(d, d.rows, d.cols, ent);
    return no_fault();
}

// dok_resize changes the shape. The keys outside the new shape go, and the
// others keep their order.
fn dok_resize(d: Dok&, rows: i64, cols: i64) -> Fault {
    let c = csr_check(rows, cols, "dok_resize");
    if c.kind != F_NONE { return c; }
    var outside = false;
    var e: i64 = 0;
    while e < d.n {
        if dok_row(d, e) >= rows || dok_col(d, e) >= cols { outside = true; }
        e += 1;
    }
    // A shape that keeps every key needs no new table. The hash of a place
    // does not depend on the shape.
    if !outside {
        d.rows = rows;
        d.cols = cols;
        return no_fault();
    }
    var kept: f64[>..] = [];
    e = 0;
    while e < d.n {
        let r = dok_row(d, e);
        let q = dok_col(d, e);
        if r < rows && q < cols { kept.push(r); kept.push(q); kept.push(dok_val(d, e)); }
        e += 1;
    }
    dok_rebuild(d, rows, cols, kept);
    return no_fault();
}
