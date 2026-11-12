// The sparse matrices of diagonals.
//
// A matrix of the type Dia holds the diagonals of the matrix. The buffer
// holds, in this order, the offset of every diagonal and the values of
// every diagonal. The offsets rise and no two of them are the same.
//
// The values of one diagonal lie in the order of the row. The first
// place of a diagonal with the offset k is the row max(0, -k). A zero
// that lies on a diagonal stays in the diagonal, so the count of the
// stored values counts the zeros. The conversion to the compressed rows
// and to the coordinates drops the zeros.
//
// The calls that have no rule of their own go through the compressed
// rows, as the other sparse formats do.
namespace scientific;

// Dia holds a sparse matrix of diagonals.
struct Dia { rows: i64, cols: i64, ndiag: i64, nnz: i64, buf: f64[>..] }

// dia_empty gives a matrix without a diagonal.
fn dia_empty() -> Dia {
    return Dia { rows: 0, cols: 0, ndiag: 0, nnz: 0, buf: [] };
}

// dia_offset gives the offset of the diagonal k.
fn dia_offset(s, k: i64) -> i64 {
    return s.buf[k] as i64;
}

// dia_first_of gives the row of the first place of a diagonal with the
// given offset.
fn dia_first_of(off: i64) -> i64 {
    if off < 0 { return -off; }
    return 0;
}

// dia_len_of gives the count of the places of a diagonal with the given
// offset in a matrix of the given shape.
fn dia_len_of(rows: i64, cols: i64, off: i64) -> i64 {
    var last = rows - 1;
    let other = cols - 1 - off;
    if other < last { last = other; }
    let first = dia_first_of(off);
    if last < first { return 0; }
    return last - first + 1;
}

// dia_len gives the count of the places of the diagonal k.
fn dia_len(s, k: i64) -> i64 {
    return dia_len_of(s.rows, s.cols, dia_offset(s, k));
}


// dia_from_entries builds a matrix from the rows, the columns and the
// values of the entries. The entries of one place add up, as the
// duplicates of SciPy add them up. A diagonal that holds no entry is
// not kept.
fn dia_from_entries(rows: i64, cols: i64, ri: f64[:], ci: f64[:], vv: f64[:]) -> Dia, Fault {
    let c = csr_check(rows, cols, "dia_from_entries");
    if c.kind != F_NONE { return dia_empty(), c; }
    let t = sparse_check_triplets(rows, cols, ri, ci, vv, "dia_from_entries");
    if t.kind != F_NONE { return dia_empty(), t; }
    let n = vv.len;
    var keys: f64[>..] = [];
    var i: i64 = 0;
    while i < n { keys.push(ri[i]); i += 1; }
    i = 0;
    while i < n { keys.push(ci[i] - ri[i]); i += 1; }
    var ord: i64[>..] = [];
    if n > 0 { ord = order_places(keys, 2, n); }
    // The offsets of the diagonals that hold an entry, and the count of
    // the places of each of them.
    var offs: f64[>..] = [];
    var lens: i64[>..] = [];
    var gbeg: i64[>..] = [];
    var gend: i64[>..] = [];
    var total: i64 = 0;
    var k: i64 = 0;
    while k < n {
        let off = (ci[ord[k]] - ri[ord[k]]) as i64;
        let len = dia_len_of(rows, cols, off);
        offs.push(off);
        lens.push(len);
        gbeg.push(k);
        total += len;
        while k < n && (ci[ord[k]] - ri[ord[k]]) as i64 == off { k += 1; }
        gend.push(k);
    }
    if total > MAX_SPARSE_NNZ {
        return dia_empty(), fault(F_LIMIT, "dia_from_entries", "the count of the stored values is above the limit");
    }
    // The buffer holds the offsets and then the values.
    var buf: f64[>..] = [];
    i = 0;
    while i < offs.len { buf.push(offs[i]); i += 1; }
    let base = offs.len;
    i = 0;
    while i < lens.len {
        var u: i64 = 0;
        while u < lens[i] { buf.push(0.0); u += 1; }
        i += 1;
    }
    // The entries of one place add up.
    var at = base;
    i = 0;
    while i < offs.len {
        let off = offs[i] as i64;
        let first = dia_first_of(off);
        var e = gbeg[i];
        while e < gend[i] {
            let f = ord[e];
            buf[at + (ri[f] as i64) - first] = buf[at + (ri[f] as i64) - first] + vv[f];
            e += 1;
        }
        at += lens[i];
        i += 1;
    }
    // A value that is not finite is a fault.
    i = base;
    while i < buf.len {
        if mathshim::is_finite_of(buf[i]) == 0.0 {
            return dia_empty(), fault(F_VALUE, "dia_from_entries", "a value is not finite");
        }
        i += 1;
    }
    return Dia { rows: rows, cols: cols, ndiag: offs.len, nnz: total, buf: buf }, no_fault();
}

// dia_from_dense builds a matrix from a dense matrix. The diagonals
// that hold no value are not kept.
fn dia_from_dense(a) -> Dia, Fault {
    let s, f = csr_from_dense(a);
    if f.kind != F_NONE { return dia_empty(), f; }
    return dia_from_csr(s);
}

// dia_to_dense gives the dense matrix of a sparse matrix.
fn dia_to_dense(s) -> Arr, Fault {
    if s.rows * s.cols > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "dia_to_dense", "the dense matrix holds more elements than the limit");
    }
    let c, f = dia_to_csr(s);
    if f.kind != F_NONE { return empty_arr(), f; }
    return csr_to_dense(c);
}

// dia_from_csr builds a matrix from a matrix of compressed rows.
fn dia_from_csr(s) -> Dia, Fault {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    var i: i64 = 0;
    while i < s.rows {
        var k = csr_ptr(s, i);
        let e = csr_ptr(s, i + 1);
        while k < e {
            ri.push(i as f64); ci.push(csr_col(s, k) as f64); vv.push(csr_val(s, k));
            k += 1;
        }
        i += 1;
    }
    return dia_from_entries(s.rows, s.cols, ri, ci, vv);
}

// dia_triplets gives the rows, the columns and the values of the
// entries of a matrix of diagonals. The zeros of the diagonals are
// dropped, as SciPy drops them.
fn dia_triplets(s) -> f64[>..], f64[>..], f64[>..] {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    var start = s.ndiag;
    var k: i64 = 0;
    while k < s.ndiag {
        let off = dia_offset(s, k);
        let first = dia_first_of(off);
        let len = dia_len(s, k);
        var t: i64 = 0;
        while t < len {
            let x = s.buf[start + t];
            if x != 0.0 {
                let r = first + t;
                ri.push(r as f64); ci.push((r + off) as f64); vv.push(x);
            }
            t += 1;
        }
        start += len;
        k += 1;
    }
    return ri, ci, vv;
}

// dia_to_csr gives the compressed rows of a matrix of diagonals. The
// zeros of the diagonals are dropped, as SciPy drops them.
fn dia_to_csr(s) -> Csr, Fault {
    let ri, ci, vv = dia_triplets(s);
    return csr_from_triplets(s.rows, s.cols, ri, ci, vv);
}

// dia_from_coo builds a matrix from a matrix of coordinates.
fn dia_from_coo(s) -> Dia, Fault {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    var k: i64 = 0;
    while k < s.nnz {
        ri.push(coo_row(s, k) as f64); ci.push(coo_col(s, k) as f64); vv.push(coo_val(s, k));
        k += 1;
    }
    return dia_from_entries(s.rows, s.cols, ri, ci, vv);
}

// dia_to_coo gives the coordinates of a matrix of diagonals. The zeros
// of the diagonals are dropped, as SciPy drops them.
fn dia_to_coo(s) -> Coo, Fault {
    let ri, ci, vv = dia_triplets(s);
    return coo_from_triplets(s.rows, s.cols, ri, ci, vv);
}

// dia_get gives the value at a place, zero when the place holds none.
fn dia_get(s, i: i64, j: i64) -> f64, Fault {
    if i < 0 || i >= s.rows || j < 0 || j >= s.cols {
        return 0.0, fault(F_RANGE, "dia_get", "the place lies outside the matrix");
    }
    let off = j - i;
    // The offsets rise, so a binary search finds the diagonal.
    var lo: i64 = 0;
    var hi = s.ndiag - 1;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let o = dia_offset(s, mid);
        if o == off {
            var start = s.ndiag;
            var k: i64 = 0;
            while k < mid { start += dia_len(s, k); k += 1; }
            return s.buf[start + (i - dia_first_of(off))], no_fault();
        }
        if o < off { lo = mid + 1; }
        else { hi = mid - 1; }
    }
    return 0.0, no_fault();
}

// dia_scale multiplies every stored value by a number.
fn dia_scale(s, a: f64) -> Dia, Fault {
    if mathshim::is_finite_of(a) == 0.0 {
        return dia_empty(), fault(F_VALUE, "dia_scale", "the factor is not finite");
    }
    var buf = copy(s.buf);
    var i: i64 = 0;
    while i < s.nnz { buf[s.ndiag + i] = buf[s.ndiag + i] * a; i += 1; }
    return Dia { rows: s.rows, cols: s.cols, ndiag: s.ndiag, nnz: s.nnz, buf: buf }, no_fault();
}

// dia_matvec gives the product of a matrix and a vector.
fn dia_matvec(s, x: f64[:]) -> Arr, Fault {
    if x.len != s.cols {
        return empty_arr(), fault(F_SHAPE, "dia_matvec", "the vector does not hold one value per column");
    }
    var out = nd_zeros(s.rows);
    var start = s.ndiag;
    var k: i64 = 0;
    while k < s.ndiag {
        let off = dia_offset(s, k);
        let first = dia_first_of(off);
        let len = dia_len(s, k);
        var t: i64 = 0;
        while t < len {
            let r = first + t;
            out[r] = out[r] + s.buf[start + t] * x[r + off];
            t += 1;
        }
        start += len;
        k += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// dia_transpose gives the transpose of a matrix. The offsets are
// negated, so the order of the diagonals turns around. The values of
// one diagonal keep their order.
fn dia_transpose(s) -> Dia, Fault {
    let c = csr_check(s.rows, s.cols, "dia_transpose");
    if c.kind != F_NONE { return dia_empty(), c; }
    // The offsets of the transpose rise, so the diagonals follow in the
    // other order. The values of one diagonal keep their order.
    var starts: i64[>..] = [];
    var lens: i64[>..] = [];
    var acc = s.ndiag;
    var q: i64 = 0;
    while q < s.ndiag {
        let len = dia_len(s, q);
        starts.push(acc); lens.push(len); acc += len; q += 1;
    }
    var buf: f64[>..] = [];
    var j: i64 = 0;
    while j < s.ndiag {
        buf.push(0.0 - (dia_offset(s, s.ndiag - 1 - j) as f64));
        j += 1;
    }
    j = 0;
    while j < s.ndiag {
        let k = s.ndiag - 1 - j;
        let start = starts[k];
        var t: i64 = 0;
        while t < lens[k] { buf.push(s.buf[start + t]); t += 1; }
        j += 1;
    }
    return Dia { rows: s.cols, cols: s.rows, ndiag: s.ndiag, nnz: s.nnz, buf: buf }, no_fault();
}

// dia_sum gives the sum of the stored values.
fn dia_sum(s) -> f64 {
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < s.nnz { t += s.buf[s.ndiag + i]; i += 1; }
    return t;
}

// dia_nnz gives the count of the stored values.
fn dia_nnz(s) -> i64 {
    return s.nnz;
}

// dia_diagonals gives the offsets and the values of the diagonals. The
// values of one diagonal lie in the order of the row.
fn dia_diagonals(s) -> Arr, Arr, Fault {
    var offs: f64[>..] = [];
    var vals: f64[>..] = [];
    var i: i64 = 0;
    while i < s.ndiag { offs.push(s.buf[i]); i += 1; }
    i = 0;
    while i < s.nnz { vals.push(s.buf[s.ndiag + i]); i += 1; }
    return line_arr(copy(offs)), line_arr(copy(vals)), no_fault();
}

// dia_check_offsets checks the offsets of the diagonals of a matrix of the
// given shape. Every offset has to be a whole number inside the matrix.
// The check comes before the cast to a whole number, because the cast of
// such a number gives no defined value.
fn dia_check_offsets(rows: i64, cols: i64, offsets: f64[:], tag: const u8[:]) -> Fault {
    var i: i64 = 0;
    while i < offsets.len {
        let o = offsets[i];
        if mathshim::is_finite_of(o) == 0.0 {
            return fault(F_RANGE, tag, "an offset is not finite");
        }
        if o > cols - 1.0 || o < 0.0 - (rows - 1.0) {
            return fault(F_RANGE, tag, "an offset lies outside the matrix");
        }
        if mathshim::trunc_of(o) != o {
            return fault(F_RANGE, tag, "an offset is not a whole number");
        }
        i += 1;
    }
    return no_fault();
}

// dia_check_dup checks that no two offsets are the same. The kind of the
// fault names the call that reports it.
fn dia_check_dup(offsets: f64[:], ord: i64[>..]&, kind: i32, tag: const u8[:]) -> Fault {
    var i: i64 = 1;
    while i < ord.len {
        if offsets[ord[i]] == offsets[ord[i - 1]] {
            return fault(kind, tag, "two offsets are the same");
        }
        i += 1;
    }
    return no_fault();
}

// dia_total_len gives the sum of the lengths of the diagonals of the
// given offsets.
fn dia_total_len(rows: i64, cols: i64, offsets: f64[:]) -> i64 {
    var total: i64 = 0;
    var i: i64 = 0;
    while i < offsets.len {
        total += dia_len_of(rows, cols, offsets[i] as i64);
        i += 1;
    }
    return total;
}

// dia_build makes the matrix from offsets that are checked and values
// that are finite. The values follow the given order of the offsets.
// The offsets come out sorted.
fn dia_build(rows: i64, cols: i64, offsets: f64[:], values: f64[:], ord: i64[>..]&, total: i64) -> Dia {
    // The place in the values of the first value of each diagonal.
    var starts: i64[>..] = [];
    var acc: i64 = 0;
    var i: i64 = 0;
    while i < offsets.len { starts.push(acc); acc += dia_len_of(rows, cols, offsets[i] as i64); i += 1; }
    var buf: f64[>..] = [];
    i = 0;
    while i < ord.len { buf.push(offsets[ord[i]]); i += 1; }
    i = 0;
    while i < ord.len {
        let k = ord[i];
        let len = dia_len_of(rows, cols, offsets[k] as i64);
        var t: i64 = 0;
        while t < len { buf.push(values[starts[k] + t]); t += 1; }
        i += 1;
    }
    return Dia { rows: rows, cols: cols, ndiag: offsets.len, nnz: total, buf: buf };
}

// dia_from_diagonals builds a matrix from the offsets of the diagonals and
// the values of the diagonals. The values of one diagonal cover its places
// inside the matrix, in the order of the row. The offsets are sorted, and
// the values follow them.
fn dia_from_diagonals(rows: i64, cols: i64, offsets: f64[:], values: f64[:]) -> Dia, Fault {
    let c = csr_check(rows, cols, "dia_from_diagonals");
    if c.kind != F_NONE { return dia_empty(), c; }
    let n = offsets.len;
    let f = dia_check_offsets(rows, cols, offsets, "dia_from_diagonals");
    if f.kind != F_NONE { return dia_empty(), f; }
    let total = dia_total_len(rows, cols, offsets);
    if total != values.len {
        return dia_empty(), fault(F_SHAPE, "dia_from_diagonals", "the count of the values does not match the diagonals");
    }
    if total > MAX_SPARSE_NNZ {
        return dia_empty(), fault(F_LIMIT, "dia_from_diagonals", "the count of the stored values is above the limit");
    }
    var i: i64 = 0;
    while i < values.len {
        if mathshim::is_finite_of(values[i]) == 0.0 {
            return dia_empty(), fault(F_VALUE, "dia_from_diagonals", "a value is not finite");
        }
        i += 1;
    }
    var ord: i64[>..] = [];
    if n > 0 { ord = order_places(offsets, 1, n); }
    let d = dia_check_dup(offsets, ord, F_VALUE, "dia_from_diagonals");
    if d.kind != F_NONE { return dia_empty(), d; }
    return dia_build(rows, cols, offsets, values, ord, total), no_fault();
}

// spdiags gives a matrix of diagonals from a dense matrix and a list of
// offsets. The data holds one row for each offset. The value of the
// diagonal d at the column j is the value data[d, j]. A column at or
// above the width of the data holds zero. The offsets come out sorted.
fn spdiags(data, offsets: f64[:], rows: i64, cols: i64) -> Dia, Fault {
    if data.ndim != 2 {
        return dia_empty(), fault(F_SHAPE, "spdiags", "the data is not a matrix");
    }
    // The shape and the buffer of the data are checked before the code
    // reads a dimension of the shape.
    let s = check_stack(data.ndim, data.shape, data.data.len, 1, "spdiags");
    if s.kind != F_NONE { return dia_empty(), s; }
    if data.shape[0] != offsets.len {
        return dia_empty(), fault(F_SHAPE, "spdiags", "the count of the offsets differs from the count of the rows of the data");
    }
    let c = csr_check(rows, cols, "spdiags");
    if c.kind != F_NONE { return dia_empty(), c; }
    let width = data.shape[1];
    // The offsets are checked before any cast. The cast of a number that
    // lies outside the range of an integer gives no defined value.
    let f = dia_check_offsets(rows, cols, offsets, "spdiags");
    if f.kind != F_NONE { return dia_empty(), f; }
    // Two offsets that are the same are a fault.
    var ord: i64[>..] = [];
    if offsets.len > 0 { ord = order_places(offsets, 1, offsets.len); }
    let d = dia_check_dup(offsets, ord, F_RANGE, "spdiags");
    if d.kind != F_NONE { return dia_empty(), d; }
    // The sum of the lengths of the diagonals is checked first. It costs
    // one pass over the offsets, and the values are not built before it.
    let total = dia_total_len(rows, cols, offsets);
    if total > MAX_SPARSE_NNZ {
        return dia_empty(), fault(F_LIMIT, "spdiags", "the count of the stored values is above the limit");
    }
    var t: i64 = 0;
    while t < data.data.len {
        if mathshim::is_finite_of(data.data[t]) == 0.0 {
            return dia_empty(), fault(F_VALUE, "spdiags", "a value of the data is not finite");
        }
        t += 1;
    }
    // The values of each diagonal lie in the order of the rows. The value
    // at the row r of the diagonal d is the value at the column r + off.
    var vals: f64[>..] = [];
    var i: i64 = 0;
    while i < offsets.len {
        let off = offsets[i] as i64;
        let len = dia_len_of(rows, cols, off);
        let first = dia_first_of(off);
        var k: i64 = 0;
        while k < len {
            let col = first + k + off;
            if col < width {
                vals.push(data.data[i * width + col]);
            } else {
                vals.push(0.0);
            }
            k += 1;
        }
        i += 1;
    }
    // The offsets and the values are checked already, so the builder does
    // not check them again.
    return dia_build(rows, cols, offsets, vals, ord, total), no_fault();
}
