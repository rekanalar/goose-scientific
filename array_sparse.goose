// The sparse matrices of compressed rows.
//
// A matrix of the type Csr holds the stored values and their places.
// The buffer holds, in this order, the row pointers (rows + 1 values),
// the column of each stored value and the stored values. The columns
// of one row rise. The index values sit in the buffer as exact whole
// numbers, which holds up to 2^53.
namespace scientific;

import mathshim;

// The largest side of a sparse matrix and the largest stored count.
const MAX_SPARSE_SIDE: i64 = 1073741824;
const MAX_SPARSE_NNZ: i64 = 134217728;

// Csr holds a sparse matrix of compressed rows.
struct Csr { rows: i64, cols: i64, nnz: i64, buf: f64[>..] }

fn csr_empty() -> Csr {
    return Csr { rows: 0, cols: 0, nnz: 0, buf: [] };
}

// csr_ptr gives the row pointer i.
fn csr_ptr(s, i: i64) -> i64 {
    return s.buf[i] as i64;
}

// csr_col gives the column of the stored value k.
fn csr_col(s, k: i64) -> i64 {
    return s.buf[s.rows + 1 + k] as i64;
}

// csr_val gives the stored value k.
fn csr_val(s, k: i64) -> f64 {
    return s.buf[s.rows + 1 + s.nnz + k];
}

// csr_check checks the sides of a matrix.
fn csr_check(rows: i64, cols: i64, tag: const u8[:]) -> Fault {
    if rows < 1 || cols < 1 || rows > MAX_SPARSE_SIDE || cols > MAX_SPARSE_SIDE {
        return fault(F_RANGE, tag, "a side stands outside one to the limit");
    }
    return no_fault();
}

// csr_from_parts builds a matrix from the row pointers, the columns and
// the values, which the caller keeps in order.
fn csr_from_parts(rows: i64, cols: i64, ptr: f64[:], idx: f64[:], val: f64[:]) -> Csr {
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < ptr.len { buf.push(ptr[i]); i += 1; }
    i = 0;
    while i < idx.len { buf.push(idx[i]); i += 1; }
    i = 0;
    while i < val.len { buf.push(val[i]); i += 1; }
    return Csr { rows: rows, cols: cols, nnz: idx.len, buf: buf };
}

// csr_from_triplets builds a matrix from the rows, the columns and the
// values of the stored entries. Entries of one place add up, as the
// sum of duplicates of SciPy adds them.
fn csr_from_triplets(rows: i64, cols: i64, ri: f64[:], ci: f64[:], v: f64[:]) -> Csr, Fault {
    let c = csr_check(rows, cols, "csr_from_triplets");
    if c.kind != F_NONE { return csr_empty(), c; }
    let n = v.len;
    if ri.len != n || ci.len != n {
        return csr_empty(), fault(F_SHAPE, "csr_from_triplets", "the rows, the columns and the values differ in count");
    }
    if n > MAX_SPARSE_NNZ {
        return csr_empty(), fault(F_LIMIT, "csr_from_triplets", "the count of entries is above the limit");
    }
    var i: i64 = 0;
    while i < n {
        if !(ri[i] >= 0.0) || ri[i] >= rows || mathshim::trunc_of(ri[i]) != ri[i] || !(ci[i] >= 0.0) || ci[i] >= cols || mathshim::trunc_of(ci[i]) != ci[i] {
            return csr_empty(), fault(F_RANGE, "csr_from_triplets", "an entry lies outside the matrix");
        }
        i += 1;
    }
    var keys: f64[>..] = [];
    i = 0;
    while i < n { keys.push(ci[i]); i += 1; }
    i = 0;
    while i < n { keys.push(ri[i]); i += 1; }
    var ord: i64[>..] = [];
    if n > 0 { ord = order_places(keys, 2, n); }
    var ptr: f64[>..] = [];
    var idx: f64[>..] = [];
    var val: f64[>..] = [];
    i = 0;
    while i <= rows { ptr.push(0.0); i += 1; }
    i = 0;
    while i < n {
        let k = ord[i];
        let last = idx.len - 1;
        if i > 0 && ri[k] == ri[ord[i - 1]] && ci[k] == ci[ord[i - 1]] {
            val[last] = val[last] + v[k];
        } else {
            idx.push(ci[k]);
            val.push(v[k]);
            ptr[(ri[k] as i64) + 1] = ptr[(ri[k] as i64) + 1] + 1.0;
        }
        i += 1;
    }
    i = 0;
    while i < rows { ptr[i + 1] = ptr[i + 1] + ptr[i]; i += 1; }
    return csr_from_parts(rows, cols, ptr, idx, val), no_fault();
}

// csr_from_dense builds a matrix from a dense matrix. The zero values
// are not stored.
fn csr_from_dense(a) -> Csr, Fault {
    if a.ndim != 2 {
        return csr_empty(), fault(F_SHAPE, "csr_from_dense", "the operand is not a matrix");
    }
    let rows = a.shape[0];
    let cols = a.shape[1];
    var ptr: f64[>..] = [0.0];
    var idx: f64[>..] = [];
    var val: f64[>..] = [];
    var i: i64 = 0;
    while i < rows {
        var j: i64 = 0;
        while j < cols {
            let x = a.data[i * cols + j];
            if x != 0.0 { idx.push(j); val.push(x); }
            j += 1;
        }
        ptr.push(idx.len);
        i += 1;
    }
    return csr_from_parts(rows, cols, ptr, idx, val), no_fault();
}

// csr_to_dense gives the dense matrix of a sparse matrix.
fn csr_to_dense(s) -> Arr, Fault {
    if s.rows * s.cols > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "csr_to_dense", "the dense matrix holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < s.rows * s.cols { d.push(0.0); i += 1; }
    i = 0;
    while i < s.rows {
        var k = csr_ptr(s, i);
        while k < csr_ptr(s, i + 1) {
            d[i * s.cols + csr_col(s, k)] = csr_val(s, k);
            k += 1;
        }
        i += 1;
    }
    return mat2(copy(d), s.rows, s.cols), no_fault();
}

// csr_eye gives the sparse identity matrix of n rows.
fn csr_eye(n: i64) -> Csr, Fault {
    let c = csr_check(n, n, "csr_eye");
    if c.kind != F_NONE { return csr_empty(), c; }
    var ptr: f64[>..] = [];
    var idx: f64[>..] = [];
    var val: f64[>..] = [];
    var i: i64 = 0;
    while i < n { ptr.push(i); idx.push(i); val.push(1.0); i += 1; }
    ptr.push(n);
    return csr_from_parts(n, n, ptr, idx, val), no_fault();
}

// csr_diags gives a sparse matrix of rows by cols with the values on
// one diagonal. The offset is above zero for a diagonal above the
// main one. The list holds one value per place of the diagonal, and
// the index follows the rule of SciPy: the row for an offset of zero
// or above, the column below.
fn csr_diags(values: f64[:], offset: i64, rows: i64, cols: i64) -> Csr, Fault {
    let c = csr_check(rows, cols, "csr_diags");
    if c.kind != F_NONE { return csr_empty(), c; }
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
                return csr_empty(), fault(F_SHAPE, "csr_diags", "the list holds too few values for the diagonal");
            }
            ri.push(i); ci.push(j); v.push(values[at]);
        }
        i += 1;
    }
    if ri.len == 0 {
        return csr_empty(), fault(F_RANGE, "csr_diags", "the offset lies outside the matrix");
    }
    return csr_from_triplets(rows, cols, ri, ci, v);
}

// csr_transpose gives the transpose of a matrix.
fn csr_transpose(s) -> Csr, Fault {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < s.rows {
        var k = csr_ptr(s, i);
        while k < csr_ptr(s, i + 1) {
            ri.push(csr_col(s, k)); ci.push(i); v.push(csr_val(s, k));
            k += 1;
        }
        i += 1;
    }
    return csr_from_triplets(s.cols, s.rows, ri, ci, v);
}

// csr_get gives the value at a place, zero when the place holds none.
fn csr_get(s, i: i64, j: i64) -> f64, Fault {
    if i < 0 || i >= s.rows || j < 0 || j >= s.cols {
        return 0.0, fault(F_RANGE, "csr_get", "the place lies outside the matrix");
    }
    var k = csr_ptr(s, i);
    while k < csr_ptr(s, i + 1) {
        if csr_col(s, k) == j { return csr_val(s, k), no_fault(); }
        k += 1;
    }
    return 0.0, no_fault();
}

// csr_scale multiplies every stored value by a number.
fn csr_scale(s, a: f64) -> Csr {
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < s.rows + 1 + s.nnz { buf.push(s.buf[i]); i += 1; }
    i = 0;
    while i < s.nnz { buf.push(s.buf[s.rows + 1 + s.nnz + i] * a); i += 1; }
    return Csr { rows: s.rows, cols: s.cols, nnz: s.nnz, buf: buf };
}

// csr_matvec gives the product of a matrix and a vector.
fn csr_matvec(s, x: f64[:]) -> Arr, Fault {
    if x.len != s.cols {
        return empty_arr(), fault(F_SHAPE, "csr_matvec", "the vector does not hold one value per column");
    }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < s.rows {
        var t: f64 = 0.0;
        var k = csr_ptr(s, i);
        while k < csr_ptr(s, i + 1) { t += csr_val(s, k) * x[csr_col(s, k)]; k += 1; }
        out.push(t);
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// csr_axpby gives a x + b y for two matrices of one shape.
fn csr_axpby(x, y, a: f64, b: f64) -> Csr, Fault {
    if x.rows != y.rows || x.cols != y.cols {
        return csr_empty(), fault(F_SHAPE, "csr_add", "the matrices differ in shape");
    }
    var ptr: f64[>..] = [0.0];
    var idx: f64[>..] = [];
    var val: f64[>..] = [];
    var i: i64 = 0;
    while i < x.rows {
        var p = csr_ptr(x, i);
        var q = csr_ptr(y, i);
        let pe = csr_ptr(x, i + 1);
        let qe = csr_ptr(y, i + 1);
        while p < pe || q < qe {
            if q >= qe || (p < pe && csr_col(x, p) < csr_col(y, q)) {
                idx.push(csr_col(x, p)); val.push(a * csr_val(x, p)); p += 1;
            } else if p >= pe || csr_col(y, q) < csr_col(x, p) {
                idx.push(csr_col(y, q)); val.push(b * csr_val(y, q)); q += 1;
            } else {
                idx.push(csr_col(x, p)); val.push(a * csr_val(x, p) + b * csr_val(y, q)); p += 1; q += 1;
            }
        }
        ptr.push(idx.len);
        i += 1;
    }
    return csr_from_parts(x.rows, x.cols, ptr, idx, val), no_fault();
}

// csr_add gives the sum of two matrices.
fn csr_add(x, y) -> Csr, Fault { return csr_axpby(x, y, 1.0, 1.0); }

// csr_sub gives the difference of two matrices.
fn csr_sub(x, y) -> Csr, Fault { return csr_axpby(x, y, 1.0, -1.0); }

// csr_matmul gives the product of two matrices.
fn csr_matmul(x, y) -> Csr, Fault {
    if x.cols != y.rows {
        return csr_empty(), fault(F_SHAPE, "csr_matmul", "the inner sides differ");
    }
    var acc: f64[>..] = [];
    var seen: bool[>..] = [];
    var i: i64 = 0;
    while i < y.cols { acc.push(0.0); seen.push(false); i += 1; }
    var ptr: f64[>..] = [0.0];
    var idx: f64[>..] = [];
    var val: f64[>..] = [];
    i = 0;
    while i < x.rows {
        var touched: f64[>..] = [];
        var p = csr_ptr(x, i);
        while p < csr_ptr(x, i + 1) {
            let a = csr_val(x, p);
            let r = csr_col(x, p);
            var q = csr_ptr(y, r);
            while q < csr_ptr(y, r + 1) {
                let c = csr_col(y, q);
                if !seen[c] { seen[c] = true; touched.push(c); }
                acc[c] = acc[c] + a * csr_val(y, q);
                q += 1;
            }
            p += 1;
        }
        if touched.len > 0 {
            let ord = order_places(touched, 1, touched.len);
            var t: i64 = 0;
            while t < touched.len {
                let c = touched[ord[t]] as i64;
                idx.push(c);
                val.push(acc[c]);
                t += 1;
            }
            t = 0;
            while t < touched.len {
                let c = touched[t] as i64;
                acc[c] = 0.0;
                seen[c] = false;
                t += 1;
            }
        }
        if idx.len > MAX_SPARSE_NNZ {
            return csr_empty(), fault(F_LIMIT, "csr_matmul", "the product holds more stored values than the limit");
        }
        ptr.push(idx.len);
        i += 1;
    }
    return csr_from_parts(x.rows, y.cols, ptr, idx, val), no_fault();
}

// csr_diagonal gives the main diagonal of a matrix.
fn csr_diagonal(s) -> Arr, Fault {
    var n = s.rows;
    if s.cols < n { n = s.cols; }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        let v, f = csr_get(s, i, i);
        out.push(v);
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// csr_sum gives the sums of the rows with axis 1, of the columns with
// axis 0.
fn csr_sum(s, axis: i64) -> Arr, Fault {
    if axis != 0 && axis != 1 {
        return empty_arr(), fault(F_RANGE, "csr_sum", "the axis is not zero or one");
    }
    var out: f64[>..] = [];
    var i: i64 = 0;
    var n = s.cols;
    if axis == 1 { n = s.rows; }
    while i < n { out.push(0.0); i += 1; }
    i = 0;
    while i < s.rows {
        var k = csr_ptr(s, i);
        while k < csr_ptr(s, i + 1) {
            if axis == 1 { out[i] = out[i] + csr_val(s, k); }
            else { out[csr_col(s, k)] = out[csr_col(s, k)] + csr_val(s, k); }
            k += 1;
        }
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// spsolve solves a x = b for a square sparse matrix. The method turns
// the matrix dense, so the side stays below 4097.
fn spsolve(s, b: f64[:]) -> Arr, Fault {
    if s.rows != s.cols || b.len != s.rows {
        return empty_arr(), fault(F_SHAPE, "spsolve", "the matrix is not square or the right side does not fit");
    }
    if s.rows > 4096 {
        return empty_arr(), fault(F_LIMIT, "spsolve", "the side is above 4096");
    }
    let d, fd = csr_to_dense(s);
    if fd.kind != F_NONE { return empty_arr(), fd; }
    let rb, fb = from_values(b);
    if fb.kind != F_NONE { return empty_arr(), fb; }
    return solve(d, rb);
}

// cg solves a x = b for a symmetric positive definite sparse matrix by
// the method of conjugate gradients. It starts from zero and stops
// when the norm of the residual is not above rtol times the norm of b.
// It gives x and the info: zero for success, or the count of turns
// when the limit of maxiter turns ran out.
fn cg(s, b: f64[:], rtol: f64 = 1.0e-5, maxiter: i64 = 0) -> Arr, i64, Fault {
    if s.rows != s.cols || b.len != s.rows || b.len < 1 {
        return empty_arr(), 0, fault(F_SHAPE, "cg", "the matrix is not square or the right side does not fit");
    }
    if !(rtol >= 0.0) {
        return empty_arr(), 0, fault(F_VALUE, "cg", "the tolerance holds a negative value");
    }
    let n = b.len;
    var limit = maxiter;
    if limit < 1 { limit = 10 * n; }
    var x: f64[>..] = [];
    var r: f64[>..] = [];
    var p: f64[>..] = [];
    var i: i64 = 0;
    var bn: f64 = 0.0;
    while i < n { x.push(0.0); r.push(b[i]); p.push(b[i]); bn += b[i] * b[i]; i += 1; }
    bn = mathshim::root(bn);
    var rs: f64 = bn * bn;
    var turn: i64 = 0;
    while turn < limit {
        if mathshim::root(rs) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
        let ap, fa = csr_matvec(s, p);
        if fa.kind != F_NONE { return empty_arr(), 0, fa; }
        var pap: f64 = 0.0;
        i = 0;
        while i < n { pap += p[i] * ap.data[i]; i += 1; }
        if !(pap > 0.0) {
            return empty_arr(), 0, fault(F_VALUE, "cg", "the matrix is not positive definite");
        }
        let alpha = rs / pap;
        var rn: f64 = 0.0;
        i = 0;
        while i < n {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * ap.data[i];
            rn += r[i] * r[i];
            i += 1;
        }
        let beta = rn / rs;
        i = 0;
        while i < n { p[i] = r[i] + beta * p[i]; i += 1; }
        rs = rn;
        turn += 1;
    }
    if mathshim::root(rs) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
    return line_arr(copy(x)), turn, no_fault();
}

// csr_kron gives the Kronecker product of two sparse matrices.
fn csr_kron(a, b) -> Csr, Fault {
    let rows = a.rows * b.rows;
    let cols = a.cols * b.cols;
    let c = csr_check(rows, cols, "csr_kron");
    if c.kind != F_NONE { return csr_empty(), c; }
    if a.nnz * b.nnz > MAX_SPARSE_NNZ {
        return csr_empty(), fault(F_LIMIT, "csr_kron", "the count of stored values is above the limit");
    }
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < a.rows {
        var p = csr_ptr(a, i);
        while p < csr_ptr(a, i + 1) {
            var k: i64 = 0;
            while k < b.rows {
                var q = csr_ptr(b, k);
                while q < csr_ptr(b, k + 1) {
                    ri.push((i * b.rows + k) as f64);
                    ci.push((csr_col(a, p) * b.cols + csr_col(b, q)) as f64);
                    v.push(csr_val(a, p) * csr_val(b, q));
                    q += 1;
                }
                k += 1;
            }
            p += 1;
        }
        i += 1;
    }
    return csr_from_triplets(rows, cols, ri, ci, v);
}

// csr_hstack joins two sparse matrices side by side. They hold the same
// count of rows.
fn csr_hstack(a, b) -> Csr, Fault {
    if a.rows != b.rows {
        return csr_empty(), fault(F_SHAPE, "csr_hstack", "the matrices differ in rows");
    }
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < a.rows {
        var p = csr_ptr(a, i);
        while p < csr_ptr(a, i + 1) {
            ri.push(i as f64);
            ci.push(csr_col(a, p) as f64);
            v.push(csr_val(a, p));
            p += 1;
        }
        p = csr_ptr(b, i);
        while p < csr_ptr(b, i + 1) {
            ri.push(i as f64);
            ci.push((a.cols + csr_col(b, p)) as f64);
            v.push(csr_val(b, p));
            p += 1;
        }
        i += 1;
    }
    return csr_from_triplets(a.rows, a.cols + b.cols, ri, ci, v);
}

// csr_vstack joins two sparse matrices one above the other. They hold
// the same count of columns.
fn csr_vstack(a, b) -> Csr, Fault {
    if a.cols != b.cols {
        return csr_empty(), fault(F_SHAPE, "csr_vstack", "the matrices differ in columns");
    }
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < a.rows + b.rows {
        var p: i64 = 0;
        var end: i64 = 0;
        if i < a.rows { p = csr_ptr(a, i); end = csr_ptr(a, i + 1); }
        else { p = csr_ptr(b, i - a.rows); end = csr_ptr(b, i - a.rows + 1); }
        while p < end {
            ri.push(i as f64);
            if i < a.rows { ci.push(csr_col(a, p) as f64); v.push(csr_val(a, p)); }
            else { ci.push(csr_col(b, p) as f64); v.push(csr_val(b, p)); }
            p += 1;
        }
        i += 1;
    }
    return csr_from_triplets(a.rows + b.rows, a.cols, ri, ci, v);
}
