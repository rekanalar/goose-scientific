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

// sparse_check_triplets checks the rows, the columns and the values of a
// list of entries. The count of the entries must be the same, the count
// must stay below the limit and every entry must lie inside the matrix.
fn sparse_check_triplets(rows: i64, cols: i64, ri: f64[:], ci: f64[:], v: f64[:], tag: const u8[:]) -> Fault {
    let n = v.len;
    if ri.len != n || ci.len != n {
        return fault(F_SHAPE, tag, "the rows, the columns and the values differ in count");
    }
    if n > MAX_SPARSE_NNZ {
        return fault(F_LIMIT, tag, "the count of entries is above the limit");
    }
    var i: i64 = 0;
    while i < n {
        if !(ri[i] >= 0.0) || ri[i] >= rows || mathshim::trunc_of(ri[i]) != ri[i] || !(ci[i] >= 0.0) || ci[i] >= cols || mathshim::trunc_of(ci[i]) != ci[i] {
            return fault(F_RANGE, tag, "an entry lies outside the matrix");
        }
        i += 1;
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
    let t = sparse_check_triplets(rows, cols, ri, ci, v, "csr_from_triplets");
    if t.kind != F_NONE { return csr_empty(), t; }
    let n = v.len;
    var i: i64 = 0;
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

// krylov_norm gives the length of a vector.
fn krylov_norm(v: f64[:]) -> f64 {
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < v.len { t += v[i] * v[i]; i += 1; }
    return mathshim::root(t);
}

// bicgstab solves a x = b for a square sparse matrix by the method of
// the stabilized biconjugate gradients, from a start of zero. It gives
// the solution and a count: zero when the residual fell below rtol times
// the size of b, and the count of turns when maxiter ran out. A
// breakdown is a fault.
fn bicgstab(s, b: f64[:], rtol: f64 = 1.0e-5, maxiter: i64 = 0) -> Arr, i64, Fault {
    if s.rows != s.cols || b.len != s.rows || b.len < 1 {
        return empty_arr(), 0, fault(F_SHAPE, "bicgstab", "the matrix is not square or the right side does not fit");
    }
    if !(rtol >= 0.0) {
        return empty_arr(), 0, fault(F_VALUE, "bicgstab", "the tolerance holds a negative value");
    }
    let n = b.len;
    var limit = maxiter;
    if limit < 1 { limit = 10 * n; }
    let bn = krylov_norm(b);
    var x: f64[>..] = [];
    var r: f64[>..] = [];
    var rh: f64[>..] = [];
    var p: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < n { x.push(0.0); r.push(b[i]); rh.push(b[i]); p.push(0.0); v.push(0.0); i += 1; }
    var rho: f64 = 1.0;
    var alpha: f64 = 1.0;
    var omega: f64 = 1.0;
    var turn: i64 = 0;
    while turn < limit {
        if krylov_norm(r) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
        var rho1: f64 = 0.0;
        i = 0;
        while i < n { rho1 += rh[i] * r[i]; i += 1; }
        if rho1 == 0.0 || omega == 0.0 {
            return empty_arr(), 0, fault(F_VALUE, "bicgstab", "the method broke down");
        }
        let beta = (rho1 / rho) * (alpha / omega);
        i = 0;
        while i < n { p[i] = r[i] + beta * (p[i] - omega * v[i]); i += 1; }
        let vv, f1 = csr_matvec(s, p);
        if f1.kind != F_NONE { return empty_arr(), 0, f1; }
        var rv: f64 = 0.0;
        i = 0;
        while i < n { v[i] = vv.data[i]; rv += rh[i] * v[i]; i += 1; }
        if rv == 0.0 { return empty_arr(), 0, fault(F_VALUE, "bicgstab", "the method broke down"); }
        alpha = rho1 / rv;
        var sv: f64[>..] = [];
        i = 0;
        while i < n { sv.push(r[i] - alpha * v[i]); i += 1; }
        if krylov_norm(sv) <= rtol * bn {
            i = 0;
            while i < n { x[i] = x[i] + alpha * p[i]; i += 1; }
            return line_arr(copy(x)), 0, no_fault();
        }
        let t, f2 = csr_matvec(s, sv);
        if f2.kind != F_NONE { return empty_arr(), 0, f2; }
        var ts: f64 = 0.0;
        var tt: f64 = 0.0;
        i = 0;
        while i < n { ts += t.data[i] * sv[i]; tt += t.data[i] * t.data[i]; i += 1; }
        if tt == 0.0 { return empty_arr(), 0, fault(F_VALUE, "bicgstab", "the method broke down"); }
        omega = ts / tt;
        i = 0;
        while i < n {
            x[i] = x[i] + alpha * p[i] + omega * sv[i];
            r[i] = sv[i] - omega * t.data[i];
            i += 1;
        }
        rho = rho1;
        turn += 1;
    }
    if krylov_norm(r) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
    return line_arr(copy(x)), turn, no_fault();
}

// gmres solves a x = b for a square sparse matrix by the method of the
// generalized minimal residual, restarted every restart turns, from a
// start of zero. It gives the solution and a count: zero when the
// residual fell below rtol times the size of b, and the count of
// restarts when maxiter ran out.
fn gmres(s, b: f64[:], rtol: f64 = 1.0e-5, restart: i64 = 20, maxiter: i64 = 0) -> Arr, i64, Fault {
    if s.rows != s.cols || b.len != s.rows || b.len < 1 {
        return empty_arr(), 0, fault(F_SHAPE, "gmres", "the matrix is not square or the right side does not fit");
    }
    if !(rtol >= 0.0) || restart < 1 {
        return empty_arr(), 0, fault(F_VALUE, "gmres", "the tolerance is negative or the restart is below one");
    }
    let n = b.len;
    var m = restart;
    if m > n { m = n; }
    var limit = maxiter;
    if limit < 1 { limit = 10 * n; }
    let bn = krylov_norm(b);
    var x: f64[>..] = [];
    var i: i64 = 0;
    while i < n { x.push(0.0); i += 1; }
    var cycle: i64 = 0;
    while cycle < limit {
        let ax, f0 = csr_matvec(s, x);
        if f0.kind != F_NONE { return empty_arr(), 0, f0; }
        var r: f64[>..] = [];
        i = 0;
        while i < n { r.push(b[i] - ax.data[i]); i += 1; }
        let beta = krylov_norm(r);
        if beta <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
        // The basis holds m + 1 vectors of n values. H holds the
        // Hessenberg matrix by columns of m + 1 values.
        var basis: f64[>..] = [];
        i = 0;
        while i < n { basis.push(r[i] / beta); i += 1; }
        var h: f64[>..] = [];
        i = 0;
        while i < (m + 1) * m { h.push(0.0); i += 1; }
        var cs: f64[>..] = [];
        var sn: f64[>..] = [];
        var g: f64[>..] = [];
        i = 0;
        while i < m + 1 { g.push(0.0); i += 1; }
        i = 0;
        while i < m { cs.push(0.0); sn.push(0.0); i += 1; }
        g[0] = beta;
        var used: i64 = 0;
        var j: i64 = 0;
        while j < m {
            var vj: f64[>..] = [];
            i = 0;
            while i < n { vj.push(basis[j * n + i]); i += 1; }
            let w0, f1 = csr_matvec(s, vj);
            if f1.kind != F_NONE { return empty_arr(), 0, f1; }
            var w: f64[>..] = [];
            i = 0;
            while i < n { w.push(w0.data[i]); i += 1; }
            var k: i64 = 0;
            while k <= j {
                var d: f64 = 0.0;
                i = 0;
                while i < n { d += w[i] * basis[k * n + i]; i += 1; }
                h[j * (m + 1) + k] = d;
                i = 0;
                while i < n { w[i] = w[i] - d * basis[k * n + i]; i += 1; }
                k += 1;
            }
            let hn = krylov_norm(w);
            h[j * (m + 1) + j + 1] = hn;
            i = 0;
            while i < n {
                if hn > 0.0 { basis.push(w[i] / hn); } else { basis.push(0.0); }
                i += 1;
            }
            // The turns of Givens keep H triangular.
            k = 0;
            while k < j {
                let t = cs[k] * h[j * (m + 1) + k] + sn[k] * h[j * (m + 1) + k + 1];
                h[j * (m + 1) + k + 1] = -sn[k] * h[j * (m + 1) + k] + cs[k] * h[j * (m + 1) + k + 1];
                h[j * (m + 1) + k] = t;
                k += 1;
            }
            let a1 = h[j * (m + 1) + j];
            let a2 = h[j * (m + 1) + j + 1];
            let den = mathshim::root(a1 * a1 + a2 * a2);
            if den == 0.0 { cs[j] = 1.0; sn[j] = 0.0; }
            else { cs[j] = a1 / den; sn[j] = a2 / den; }
            h[j * (m + 1) + j] = den;
            h[j * (m + 1) + j + 1] = 0.0;
            g[j + 1] = -sn[j] * g[j];
            g[j] = cs[j] * g[j];
            used = j + 1;
            if mathshim::abs_of(g[j + 1]) <= rtol * bn { break; }
            j += 1;
        }
        // Back substitution for the coefficients, then the update.
        var yv: f64[>..] = [];
        i = 0;
        while i < used { yv.push(0.0); i += 1; }
        i = used - 1;
        while i >= 0 {
            var t = g[i];
            var k = i + 1;
            while k < used { t -= h[k * (m + 1) + i] * yv[k]; k += 1; }
            let dd = h[i * (m + 1) + i];
            if dd == 0.0 { return empty_arr(), 0, fault(F_VALUE, "gmres", "the method broke down"); }
            yv[i] = t / dd;
            i -= 1;
        }
        i = 0;
        while i < used {
            var q: i64 = 0;
            while q < n { x[q] = x[q] + yv[i] * basis[i * n + q]; q += 1; }
            i += 1;
        }
        cycle += 1;
    }
    let ax2, f3 = csr_matvec(s, x);
    if f3.kind != F_NONE { return empty_arr(), 0, f3; }
    var rr: f64[>..] = [];
    i = 0;
    while i < n { rr.push(b[i] - ax2.data[i]); i += 1; }
    if krylov_norm(rr) <= rtol * bn { return line_arr(copy(x)), 0, no_fault(); }
    return line_arr(copy(x)), cycle, no_fault();
}

// csr_block_diag places two sparse matrices on the diagonal of a larger
// one, as block_diag of SciPy does.
fn csr_block_diag(a, b) -> Csr, Fault {
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
            else { ci.push((a.cols + csr_col(b, p)) as f64); v.push(csr_val(b, p)); }
            p += 1;
        }
        i += 1;
    }
    return csr_from_triplets(a.rows + b.rows, a.cols + b.cols, ri, ci, v);
}

// csr_part keeps the stored values of the lower part (lower true) or
// the upper part, from the diagonal k. The diagonal k holds the values
// with column minus row equal to k.
fn csr_part(a, k: i64, lower: bool) -> Csr, Fault {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < a.rows {
        var p = csr_ptr(a, i);
        while p < csr_ptr(a, i + 1) {
            let j = csr_col(a, p);
            if (lower && j - i <= k) || (!lower && j - i >= k) {
                ri.push(i as f64);
                ci.push(j as f64);
                v.push(csr_val(a, p));
            }
            p += 1;
        }
        i += 1;
    }
    return csr_from_triplets(a.rows, a.cols, ri, ci, v);
}

// csr_tril gives the lower triangle of a sparse matrix up to the
// diagonal k, as tril of SciPy does.
fn csr_tril(a, k: i64 = 0) -> Csr, Fault { return csr_part(a, k, true); }

// csr_triu gives the upper triangle of a sparse matrix from the
// diagonal k, as triu of SciPy does.
fn csr_triu(a, k: i64 = 0) -> Csr, Fault { return csr_part(a, k, false); }

// csr_multiply gives the product of two sparse matrices of one shape,
// value by value, as multiply of SciPy does.
fn csr_multiply(a, b) -> Csr, Fault {
    if a.rows != b.rows || a.cols != b.cols {
        return csr_empty(), fault(F_SHAPE, "csr_multiply", "the matrices differ in shape");
    }
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < a.rows {
        var p = csr_ptr(a, i);
        var q = csr_ptr(b, i);
        let pe = csr_ptr(a, i + 1);
        let qe = csr_ptr(b, i + 1);
        while p < pe && q < qe {
            let ja = csr_col(a, p);
            let jb = csr_col(b, q);
            if ja == jb {
                ri.push(i as f64);
                ci.push(ja as f64);
                v.push(csr_val(a, p) * csr_val(b, q));
                p += 1;
                q += 1;
            } else if ja < jb { p += 1; } else { q += 1; }
        }
        i += 1;
    }
    return csr_from_triplets(a.rows, a.cols, ri, ci, v);
}

// csr_slice gives the block of rows r0 to r1 and columns c0 to c1, with
// the end excluded, as the slice a[r0:r1, c0:c1] of SciPy does.
fn csr_slice(a, r0: i64, r1: i64, c0: i64, c1: i64) -> Csr, Fault {
    if r0 < 0 || c0 < 0 || r1 > a.rows || c1 > a.cols || r0 >= r1 || c0 >= c1 {
        return csr_empty(), fault(F_RANGE, "csr_slice", "the slice stands outside the matrix or holds no element");
    }
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = r0;
    while i < r1 {
        var p = csr_ptr(a, i);
        while p < csr_ptr(a, i + 1) {
            let j = csr_col(a, p);
            if j >= c0 && j < c1 {
                ri.push((i - r0) as f64);
                ci.push((j - c0) as f64);
                v.push(csr_val(a, p));
            }
            p += 1;
        }
        i += 1;
    }
    return csr_from_triplets(r1 - r0, c1 - c0, ri, ci, v);
}

// lsqr solves the least squares problem of a sparse matrix, of any
// shape, by the method of Paige and Saunders. It starts from zero, so
// a system with more unknowns than equations gives the solution of the
// least norm. It stops when the residual norm is not above rtol times
// the norm of b, or when the gradient of the residual is small. It
// gives x and the info: zero for success, or the count of turns when
// the limit of maxiter turns ran out.
fn lsqr(s, b: f64[:], rtol: f64 = 1.0e-8, maxiter: i64 = 0) -> Arr, i64, Fault {
    if b.len != s.rows || b.len < 1 {
        return empty_arr(), 0, fault(F_SHAPE, "lsqr", "the right side does not fit the matrix");
    }
    if !(rtol >= 0.0) {
        return empty_arr(), 0, fault(F_VALUE, "lsqr", "the tolerance holds a negative value");
    }
    let m = s.rows;
    let n = s.cols;
    let st = csr_transpose(s);
    var limit = maxiter;
    if limit < 1 { limit = 2 * n; }
    var u: f64[>..] = [];
    var i: i64 = 0;
    while i < m { u.push(b[i]); i += 1; }
    let bn = krylov_norm(b);
    var x: f64[>..] = [];
    i = 0;
    while i < n { x.push(0.0); i += 1; }
    if bn == 0.0 { return line_arr(copy(x)), 0, no_fault(); }
    i = 0;
    while i < m { u[i] = u[i] / bn; i += 1; }
    let v0, f0 = csr_matvec(st, u);
    if f0.kind != F_NONE { return empty_arr(), 0, f0; }
    var v: f64[>..] = [];
    i = 0;
    while i < n { v.push(v0.data[i]); i += 1; }
    var alpha = krylov_norm(v);
    if alpha == 0.0 { return line_arr(copy(x)), 0, no_fault(); }
    var w: f64[>..] = [];
    i = 0;
    while i < n { v[i] = v[i] / alpha; w.push(v[i]); i += 1; }
    var phibar = bn;
    var rhobar = alpha;
    var turn: i64 = 0;
    while turn < limit {
        let av, f1 = csr_matvec(s, v);
        if f1.kind != F_NONE { return empty_arr(), 0, f1; }
        i = 0;
        while i < m { u[i] = av.data[i] - alpha * u[i]; i += 1; }
        let beta = krylov_norm(u);
        if beta > 0.0 {
            i = 0;
            while i < m { u[i] = u[i] / beta; i += 1; }
            let atu, f2 = csr_matvec(st, u);
            if f2.kind != F_NONE { return empty_arr(), 0, f2; }
            i = 0;
            while i < n { v[i] = atu.data[i] - beta * v[i]; i += 1; }
            alpha = krylov_norm(v);
            if alpha > 0.0 {
                i = 0;
                while i < n { v[i] = v[i] / alpha; i += 1; }
            }
        }
        let rho = mathshim::root(rhobar * rhobar + beta * beta);
        let c = rhobar / rho;
        let sn = beta / rho;
        let theta = sn * alpha;
        rhobar = -c * alpha;
        let phi = c * phibar;
        phibar = sn * phibar;
        i = 0;
        while i < n {
            x[i] = x[i] + (phi / rho) * w[i];
            w[i] = v[i] - (theta / rho) * w[i];
            i += 1;
        }
        turn += 1;
        // The test of the residual, and the test of the gradient.
        let grad = phibar * alpha * mathshim::abs_of(c);
        if phibar <= rtol * bn || grad <= rtol * phibar * (1.0 + alpha) || beta == 0.0 || alpha == 0.0 {
            return line_arr(copy(x)), 0, no_fault();
        }
    }
    return line_arr(copy(x)), turn, no_fault();
}

// minres solves a x = b for a symmetric sparse matrix, which may be
// indefinite, by the method of Paige and Saunders. It starts from zero
// and stops when the estimate of the residual norm is not above rtol
// times the norm of b. It gives x and the info: zero for success, or
// the count of turns when the limit of maxiter turns ran out. The
// matrix must be symmetric. The method does not check that.
fn minres(s, b: f64[:], rtol: f64 = 1.0e-5, maxiter: i64 = 0) -> Arr, i64, Fault {
    if s.rows != s.cols || b.len != s.rows || b.len < 1 {
        return empty_arr(), 0, fault(F_SHAPE, "minres", "the matrix is not square or the right side does not fit");
    }
    if !(rtol >= 0.0) {
        return empty_arr(), 0, fault(F_VALUE, "minres", "the tolerance holds a negative value");
    }
    let n = b.len;
    var limit = maxiter;
    if limit < 1 { limit = 5 * n; }
    var x: f64[>..] = [];
    var y: f64[>..] = [];
    var r1: f64[>..] = [];
    var r2: f64[>..] = [];
    var w: f64[>..] = [];
    var w2: f64[>..] = [];
    var i: i64 = 0;
    while i < n { x.push(0.0); y.push(b[i]); r1.push(b[i]); r2.push(b[i]); w.push(0.0); w2.push(0.0); i += 1; }
    let beta1 = krylov_norm(b);
    if beta1 == 0.0 { return line_arr(copy(x)), 0, no_fault(); }
    var oldb: f64 = 0.0;
    var beta = beta1;
    var dbar: f64 = 0.0;
    var epsln: f64 = 0.0;
    var phibar = beta1;
    var cs: f64 = -1.0;
    var sn: f64 = 0.0;
    var turn: i64 = 0;
    while turn < limit {
        let sc = 1.0 / beta;
        var v: f64[>..] = [];
        i = 0;
        while i < n { v.push(sc * y[i]); i += 1; }
        let av, f1 = csr_matvec(s, v);
        if f1.kind != F_NONE { return empty_arr(), 0, f1; }
        i = 0;
        while i < n { y[i] = av.data[i]; i += 1; }
        if turn >= 1 {
            i = 0;
            while i < n { y[i] = y[i] - (beta / oldb) * r1[i]; i += 1; }
        }
        var alfa: f64 = 0.0;
        i = 0;
        while i < n { alfa += v[i] * y[i]; i += 1; }
        i = 0;
        while i < n { y[i] = y[i] - (alfa / beta) * r2[i]; i += 1; }
        i = 0;
        while i < n { r1[i] = r2[i]; r2[i] = y[i]; i += 1; }
        oldb = beta;
        beta = krylov_norm(y);
        let oldeps = epsln;
        let delta = cs * dbar + sn * alfa;
        let gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;
        let gamma = mathshim::root(gbar * gbar + beta * beta);
        if gamma == 0.0 {
            return empty_arr(), 0, fault(F_VALUE, "minres", "the matrix is singular");
        }
        cs = gbar / gamma;
        sn = beta / gamma;
        let phi = cs * phibar;
        phibar = sn * phibar;
        let denom = 1.0 / gamma;
        i = 0;
        while i < n {
            let w1 = w2[i];
            w2[i] = w[i];
            w[i] = (v[i] - oldeps * w1 - delta * w2[i]) * denom;
            x[i] = x[i] + phi * w[i];
            i += 1;
        }
        turn += 1;
        if mathshim::abs_of(phibar) <= rtol * beta1 || beta == 0.0 { return line_arr(copy(x)), 0, no_fault(); }
    }
    return line_arr(copy(x)), turn, no_fault();
}

// eigsh gives k eigenvalues and eigenvectors of a symmetric sparse
// matrix by the Lanczos method with a full reorthogonalization. The
// choice which is "LA" for the largest values, "SA" for the smallest
// values, "LM" for the largest sizes, and "SM" for the smallest sizes.
// The values come in rising order. The vectors are the columns of the
// second result. The matrix must be symmetric; the method does not
// check that. The method grows the Krylov space until the residuals
// are small. A space above 256 vectors is a fault.
fn eigsh(s, k: i64 = 6, which: const u8[:] = "LM", tol: f64 = 1.0e-10) -> Arr, Arr, Fault {
    if s.rows != s.cols || s.rows < 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "eigsh", "the matrix is not square or holds fewer than two rows");
    }
    let n = s.rows;
    if k < 1 || k >= n {
        return empty_arr(), empty_arr(), fault(F_RANGE, "eigsh", "k must stand from one to below the side");
    }
    if which != "LA" && which != "SA" && which != "LM" && which != "SM" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "eigsh", "which holds an unknown name");
    }
    if !(tol > 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "eigsh", "the tolerance must be above zero");
    }
    var r = default_rng(1);
    var start: f64[>..] = [];
    var i: i64 = 0;
    while i < n { start.push(random_uniform(r) - 0.5); i += 1; }
    var m = 2 * k + 20;
    if m > n { m = n; }
    while true {
        // The Lanczos steps. V holds the vectors one after another.
        var vs: f64[>..] = [];
        var al: f64[>..] = [];
        var be: f64[>..] = [];
        var nv = krylov_norm(start);
        i = 0;
        while i < n { vs.push(start[i] / nv); i += 1; }
        var steps: i64 = 0;
        var broke = false;
        while steps < m {
            var cur: f64[>..] = [];
            i = 0;
            while i < n { cur.push(vs[steps * n + i]); i += 1; }
            let av, fa = csr_matvec(s, cur);
            if fa.kind != F_NONE { return empty_arr(), empty_arr(), fa; }
            var w: f64[>..] = [];
            i = 0;
            while i < n { w.push(av.data[i]); i += 1; }
            var a: f64 = 0.0;
            i = 0;
            while i < n { a += cur[i] * w[i]; i += 1; }
            al.push(a);
            var pass: i64 = 0;
            while pass < 2 {
                var j: i64 = 0;
                while j <= steps {
                    var d: f64 = 0.0;
                    i = 0;
                    while i < n { d += vs[j * n + i] * w[i]; i += 1; }
                    i = 0;
                    while i < n { w[i] = w[i] - d * vs[j * n + i]; i += 1; }
                    j += 1;
                }
                pass += 1;
            }
            let b = krylov_norm(w);
            steps += 1;
            if steps == m { be.push(b); break; }
            if b <= 1.0e-12 * (1.0 + mathshim::abs_of(a)) { be.push(0.0); broke = true; break; }
            be.push(b);
            i = 0;
            while i < n { vs.push(w[i] / b); i += 1; }
        }
        // The tridiagonal matrix of the steps.
        var td: f64[>..] = [];
        i = 0;
        while i < steps * steps { td.push(0.0); i += 1; }
        i = 0;
        while i < steps {
            td[i * steps + i] = al[i];
            if i + 1 < steps { td[i * steps + i + 1] = be[i]; td[(i + 1) * steps + i] = be[i]; }
            i += 1;
        }
        let tv, tz, tf = eigh(mat2(copy(td), steps, steps));
        if tf.kind != F_NONE { return empty_arr(), empty_arr(), tf; }
        if steps < k {
            return empty_arr(), empty_arr(), fault(F_VALUE, "eigsh", "the Krylov space holds fewer than k vectors");
        }
        // The wanted Ritz values, by index into the rising list.
        var pick: i64[>..] = [];
        var lo: i64 = 0;
        var hi = steps - 1;
        while pick.len < k {
            var take_hi = false;
            if which == "LA" { take_hi = true; }
            else if which == "LM" { take_hi = mathshim::abs_of(tv.data[hi]) >= mathshim::abs_of(tv.data[lo]); }
            else if which == "SM" { take_hi = mathshim::abs_of(tv.data[hi]) < mathshim::abs_of(tv.data[lo]); }
            if take_hi { pick.push(hi); hi -= 1; } else { pick.push(lo); lo += 1; }
        }
        var worst: f64 = 0.0;
        var scale: f64 = 1.0e-300;
        i = 0;
        while i < k {
            let res = mathshim::abs_of(be[steps - 1] * tz.data[(steps - 1) * steps + pick[i]]);
            if res > worst { worst = res; }
            let mag = mathshim::abs_of(tv.data[pick[i]]);
            if mag > scale { scale = mag; }
            i += 1;
        }
        if broke || steps >= n || worst <= tol * scale {
            // Sort the picked indices into rising order of the value.
            var order: i64[>..] = [];
            i = 0;
            while i < k { order.push(pick[i]); i += 1; }
            var x: i64 = 1;
            while x < k {
                var y = x;
                while y > 0 && order[y - 1] > order[y] {
                    let t = order[y];
                    order[y] = order[y - 1];
                    order[y - 1] = t;
                    y -= 1;
                }
                x += 1;
            }
            var vals: f64[>..] = [];
            var vecs: f64[>..] = [];
            i = 0;
            while i < n * k { vecs.push(0.0); i += 1; }
            var c: i64 = 0;
            while c < k {
                vals.push(tv.data[order[c]]);
                var j: i64 = 0;
                while j < steps {
                    let coef = tz.data[j * steps + order[c]];
                    i = 0;
                    while i < n { vecs[i * k + c] = vecs[i * k + c] + coef * vs[j * n + i]; i += 1; }
                    j += 1;
                }
                c += 1;
            }
            return line_arr(copy(vals)), mat2(copy(vecs), n, k), no_fault();
        }
        if m >= 256 || m >= n {
            return empty_arr(), empty_arr(), fault(F_LIMIT, "eigsh", "the method did not converge within 256 vectors");
        }
        m = 2 * m;
        if m > n { m = n; }
        if m > 256 { m = 256; }
    }
    return empty_arr(), empty_arr(), no_fault();
}
