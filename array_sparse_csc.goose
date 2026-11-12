// The sparse matrices of compressed columns.
//
// A matrix of the type Csc holds the stored values and their places. The
// buffer holds, in this order, the column pointers (cols + 1 values), the
// row of each stored value and the stored values. The rows of one column
// rise. The index values sit in the buffer as exact whole numbers.
//
// The type Csr holds the same values in the other order. Every call of
// this file that has no own rule converts to the compressed rows and back.
namespace scientific;

// Csc holds a sparse matrix of compressed columns.
struct Csc { rows: i64, cols: i64, nnz: i64, buf: f64[>..] }

fn csc_empty() -> Csc {
    return Csc { rows: 0, cols: 0, nnz: 0, buf: [] };
}

// csc_ptr gives the column pointer i.
fn csc_ptr(s, i: i64) -> i64 {
    return s.buf[i] as i64;
}

// csc_row gives the row of the stored value k.
fn csc_row(s, k: i64) -> i64 {
    return s.buf[s.cols + 1 + k] as i64;
}

// csc_val gives the stored value k.
fn csc_val(s, k: i64) -> f64 {
    return s.buf[s.cols + 1 + s.nnz + k];
}

// csc_check checks the sides of a matrix.
fn csc_check(rows: i64, cols: i64, tag: const u8[:]) -> Fault {
    if rows < 1 || cols < 1 || rows > MAX_SPARSE_SIDE || cols > MAX_SPARSE_SIDE {
        return fault(F_RANGE, tag, "a side stands outside one to the limit");
    }
    return no_fault();
}

// csc_from_parts builds a matrix from the column pointers, the rows and
// the values, which the caller keeps in order.
fn csc_from_parts(rows: i64, cols: i64, ptr: f64[:], idx: f64[:], val: f64[:]) -> Csc {
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < ptr.len { buf.push(ptr[i]); i += 1; }
    i = 0;
    while i < idx.len { buf.push(idx[i]); i += 1; }
    i = 0;
    while i < val.len { buf.push(val[i]); i += 1; }
    return Csc { rows: rows, cols: cols, nnz: idx.len, buf: buf };
}

// csc_from_triplets builds a matrix from the rows, the columns and the
// values of the stored entries. Entries of one place add up, as the sum
// of duplicates of SciPy adds them.
fn csc_from_triplets(rows: i64, cols: i64, ri: f64[:], ci: f64[:], v: f64[:]) -> Csc, Fault {
    let c = csc_check(rows, cols, "csc_from_triplets");
    if c.kind != F_NONE { return csc_empty(), c; }
    let t = sparse_check_triplets(rows, cols, ri, ci, v, "csc_from_triplets");
    if t.kind != F_NONE { return csc_empty(), t; }
    let n = v.len;
    var i: i64 = 0;
    // The entries of one column, with the rows in order.
    var ord: i64[>..] = [];
    i = 0;
    while i < n { ord.push(i); i += 1; }
    var s = n / 2 - 1;
    while s >= 0 { csc_sift(ord, ri, ci, s, n); s -= 1; }
    var end = n - 1;
    while end > 0 {
        let t = ord[0];
        ord[0] = ord[end];
        ord[end] = t;
        csc_sift(ord, ri, ci, 0, end);
        end -= 1;
    }
    var ptr: f64[>..] = [0.0];
    var idx: f64[>..] = [];
    var val: f64[>..] = [];
    var j: i64 = 0;
    var k: i64 = 0;
    while j < cols {
        while k < n && ci[ord[k]] == j {
            let p = ord[k];
            // The duplicates of one place add up.
            var sum = v[p];
            var q = k + 1;
            while q < n && ci[ord[q]] == j && ri[ord[q]] == ri[p] {
                sum += v[ord[q]];
                q += 1;
            }
            idx.push(ri[p]);
            val.push(sum);
            k = q;
        }
        ptr.push(idx.len);
        j += 1;
    }
    return csc_from_parts(rows, cols, ptr, idx, val), no_fault();
}

// csc_less orders two entries by the column and then by the row.
fn csc_less(ri: f64[:], ci: f64[:], x: i64, y: i64) -> bool {
    if ci[x] != ci[y] { return ci[x] < ci[y]; }
    if ri[x] != ri[y] { return ri[x] < ri[y]; }
    return x < y;
}

// csc_sift keeps the heap of the entries in order.
fn csc_sift(ord: i64[>..]&, ri: f64[:], ci: f64[:], start: i64, end: i64) {
    var root = start;
    var go = true;
    while go {
        var child = 2 * root + 1;
        if child >= end { go = false; }
        else {
            if child + 1 < end && csc_less(ri, ci, ord[child], ord[child + 1]) { child += 1; }
            if csc_less(ri, ci, ord[root], ord[child]) {
                let t = ord[root];
                ord[root] = ord[child];
                ord[child] = t;
                root = child;
            } else { go = false; }
        }
    }
}

// csc_from_csr gives the compressed columns of a matrix of compressed rows.
fn csc_from_csr(s) -> Csc, Fault {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < s.rows {
        var k = csr_ptr(s, i);
        while k < csr_ptr(s, i + 1) {
            ri.push(i as f64);
            ci.push(csr_col(s, k) as f64);
            v.push(csr_val(s, k));
            k += 1;
        }
        i += 1;
    }
    return csc_from_triplets(s.rows, s.cols, ri, ci, v);
}

// csc_to_csr gives the compressed rows of a matrix of compressed columns.
fn csc_to_csr(s) -> Csr, Fault {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var j: i64 = 0;
    while j < s.cols {
        var k = csc_ptr(s, j);
        while k < csc_ptr(s, j + 1) {
            ri.push(csc_row(s, k) as f64);
            ci.push(j as f64);
            v.push(csc_val(s, k));
            k += 1;
        }
        j += 1;
    }
    return csr_from_triplets(s.rows, s.cols, ri, ci, v);
}

// csc_from_dense gives the compressed columns of a dense matrix.
fn csc_from_dense(a) -> Csc, Fault {
    let s, f = csr_from_dense(a);
    if f.kind != F_NONE { return csc_empty(), f; }
    return csc_from_csr(s);
}

// csc_to_dense gives the dense matrix of a sparse matrix.
fn csc_to_dense(s) -> Arr, Fault {
    let c, f = csc_to_csr(s);
    if f.kind != F_NONE { return empty_arr(), f; }
    return csr_to_dense(c);
}

// csc_eye gives the sparse identity matrix of n rows.
fn csc_eye(n: i64) -> Csc, Fault {
    let c = csc_check(n, n, "csc_eye");
    if c.kind != F_NONE { return csc_empty(), c; }
    var ptr: f64[>..] = [];
    var idx: f64[>..] = [];
    var val: f64[>..] = [];
    var i: i64 = 0;
    while i < n { ptr.push(i); idx.push(i); val.push(1.0); i += 1; }
    ptr.push(n);
    return csc_from_parts(n, n, ptr, idx, val), no_fault();
}

// csc_diags gives a sparse matrix of rows by cols with the values on one
// diagonal. The offset is above zero for a diagonal above the main one.
fn csc_diags(values: f64[:], offset: i64, rows: i64, cols: i64) -> Csc, Fault {
    let c = csc_check(rows, cols, "csc_diags");
    if c.kind != F_NONE { return csc_empty(), c; }
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < rows {
        let j = i + offset;
        if j >= 0 && j < cols {
            var at = i;
            if offset < 0 { at = j; }
            if at >= values.len {
                return csc_empty(), fault(F_SHAPE, "csc_diags", "the list holds too few values for the diagonal");
            }
            ri.push(i); ci.push(j); v.push(values[at]);
        }
        i += 1;
    }
    if ri.len == 0 {
        return csc_empty(), fault(F_RANGE, "csc_diags", "the offset lies outside the matrix");
    }
    return csc_from_triplets(rows, cols, ri, ci, v);
}

// csc_transpose gives the transpose of a matrix.
fn csc_transpose(s) -> Csc, Fault {
    let c, f = csc_to_csr(s);
    if f.kind != F_NONE { return csc_empty(), f; }
    let t = csr_transpose(c);
    return csc_from_csr(t);
}

// csc_get gives the value at a place, zero when the place holds none.
fn csc_get(s, i: i64, j: i64) -> f64, Fault {
    if i < 0 || i >= s.rows || j < 0 || j >= s.cols {
        return 0.0, fault(F_RANGE, "csc_get", "the place lies outside the matrix");
    }
    var k = csc_ptr(s, j);
    while k < csc_ptr(s, j + 1) {
        if csc_row(s, k) == i { return csc_val(s, k), no_fault(); }
        k += 1;
    }
    return 0.0, no_fault();
}

// csc_scale multiplies every stored value by a number.
fn csc_scale(s, a: f64) -> Csc {
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < s.cols + 1 + s.nnz { buf.push(s.buf[i]); i += 1; }
    i = 0;
    while i < s.nnz { buf.push(s.buf[s.cols + 1 + s.nnz + i] * a); i += 1; }
    return Csc { rows: s.rows, cols: s.cols, nnz: s.nnz, buf: buf };
}

// csc_matvec gives the product of a matrix and a vector.
fn csc_matvec(s, x: f64[:]) -> Arr, Fault {
    if x.len != s.cols {
        return empty_arr(), fault(F_SHAPE, "csc_matvec", "the vector does not hold one value per column");
    }
    var out: f64[>..] = nd_zeros(s.rows);
    var j: i64 = 0;
    while j < s.cols {
        var k = csc_ptr(s, j);
        while k < csc_ptr(s, j + 1) {
            out[csc_row(s, k)] += csc_val(s, k) * x[j];
            k += 1;
        }
        j += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// csc_axpby gives a x + b y for two matrices of one shape.
fn csc_axpby(x, y, a: f64, b: f64) -> Csc, Fault {
    let cx, f = csc_to_csr(x);
    if f.kind != F_NONE { return csc_empty(), f; }
    let cy, g = csc_to_csr(y);
    if g.kind != F_NONE { return csc_empty(), g; }
    let r, h = csr_axpby(cx, cy, a, b);
    if h.kind != F_NONE { return csc_empty(), h; }
    return csc_from_csr(r);
}

// csc_add gives the sum of two matrices of one shape.
fn csc_add(x, y) -> Csc, Fault {
    return csc_axpby(x, y, 1.0, 1.0);
}

// csc_sub gives the difference of two matrices of one shape.
fn csc_sub(x, y) -> Csc, Fault {
    return csc_axpby(x, y, 1.0, -1.0);
}

// csc_matmul gives the product of two matrices.
fn csc_matmul(x, y) -> Csc, Fault {
    let cx, f = csc_to_csr(x);
    if f.kind != F_NONE { return csc_empty(), f; }
    let cy, g = csc_to_csr(y);
    if g.kind != F_NONE { return csc_empty(), g; }
    let r = csr_matmul(cx, cy);
    return csc_from_csr(r);
}

// csc_diagonal gives the diagonal of a matrix.
fn csc_diagonal(s) -> Arr, Fault {
    let c, f = csc_to_csr(s);
    if f.kind != F_NONE { return empty_arr(), f; }
    return csr_diagonal(c);
}

// csc_sum gives the sum of the values of a matrix along an axis. The axis
// 0 gives the sum of every column and the axis 1 the sum of every row.
fn csc_sum(s, axis: i64) -> Arr, Fault {
    let c, f = csc_to_csr(s);
    if f.kind != F_NONE { return empty_arr(), f; }
    return csr_sum(c, axis);
}
