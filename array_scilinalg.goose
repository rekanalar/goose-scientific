// The LU factors, the triangular solve and the special matrices.
//
// The functions of scipy.linalg that build, factor and solve with
// matrices: the LU factors, the triangular solve, the special
// matrices, the null space, the matrix exponential and the banded
// solve. The LU factors follow LAPACK: a row swap for every column and
// the unit lower factor under the diagonal.
namespace scientific;

import mathshim;

// The largest side of a matrix that expm takes. The cost grows with the
// cube of the side.
const MAX_EXPM_SIDE: i64 = 256;

// lapack_lu factors the matrix in place. The row i swaps with the row
// piv[i], as LAPACK does. A zero pivot stays and skips the column.
fn lapack_lu(d: f64[>..]&, m: i64, n: i64, piv: i64[>..]&) {
    var k = m;
    if n < k { k = n; }
    var col: i64 = 0;
    while col < k {
        var best = col;
        var best_val = mathshim::abs_of(d[col * n + col]);
        var r: i64 = col + 1;
        while r < m {
            let v = mathshim::abs_of(d[r * n + col]);
            if v > best_val { best = r; best_val = v; }
            r += 1;
        }
        piv.push(best);
        if best_val != 0.0 {
            if best != col {
                var c: i64 = 0;
                while c < n {
                    let t = d[col * n + c];
                    d[col * n + c] = d[best * n + c];
                    d[best * n + c] = t;
                    c += 1;
                }
            }
            r = col + 1;
            while r < m {
                let f = d[r * n + col] / d[col * n + col];
                d[r * n + col] = f;
                var c: i64 = col + 1;
                while c < n {
                    d[r * n + c] = d[r * n + c] - f * d[col * n + c];
                    c += 1;
                }
                r += 1;
            }
        }
        col += 1;
    }
}

fn check_matrix(a, tag: const u8[:]) -> Fault {
    if a.ndim != 2 {
        return fault(F_SHAPE, tag, "the operand is not a matrix");
    }
    return no_fault();
}

// lu_factor gives the LU factors of a matrix in one matrix and the row
// swaps in a row. The unit lower factor sits under the diagonal. The
// row i swapped with the row piv[i]. A zero pivot gives no fault here,
// and lu_solve refuses the factors.
fn lu_factor(a) -> Arr, Arr, Fault {
    let c = check_matrix(a, "lu_factor");
    if c.kind != F_NONE { return empty_arr(), empty_arr(), c; }
    let m = a.shape[0];
    let n = a.shape[1];
    var d = copy_slice(a.data);
    var piv: i64[>..] = [];
    lapack_lu(d, m, n, piv);
    var pv: f64[>..] = [];
    var i: i64 = 0;
    while i < piv.len { pv.push(piv[i]); i += 1; }
    return mat2(copy(d), m, n), line_arr(copy(pv)), no_fault();
}

// lu gives the permutation, the unit lower factor and the upper factor
// of a matrix, so that a = p l u. The lower factor holds m rows and k
// columns. The upper factor holds k rows and n columns. The count k is
// the smaller side.
fn lu(a) -> Arr, Arr, Arr, Fault {
    let c = check_matrix(a, "lu");
    if c.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), c; }
    let m = a.shape[0];
    let n = a.shape[1];
    var k = m;
    if n < k { k = n; }
    var d = copy_slice(a.data);
    var piv: i64[>..] = [];
    lapack_lu(d, m, n, piv);
    var perm: i64[>..] = [];
    var i: i64 = 0;
    while i < m { perm.push(i); i += 1; }
    i = 0;
    while i < piv.len {
        let t = perm[i];
        perm[i] = perm[piv[i]];
        perm[piv[i]] = t;
        i += 1;
    }
    var p: f64[>..] = [];
    i = 0;
    while i < m * m { p.push(0.0); i += 1; }
    i = 0;
    while i < m { p[perm[i] * m + i] = 1.0; i += 1; }
    var l: f64[>..] = [];
    var u: f64[>..] = [];
    i = 0;
    while i < m {
        var j: i64 = 0;
        while j < k {
            if i == j { l.push(1.0); }
            else if i > j { l.push(d[i * n + j]); }
            else { l.push(0.0); }
            j += 1;
        }
        i += 1;
    }
    i = 0;
    while i < k {
        var j: i64 = 0;
        while j < n {
            if j >= i { u.push(d[i * n + j]); } else { u.push(0.0); }
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(p), m, m), mat2(copy(l), m, k), mat2(copy(u), k, n), no_fault();
}

// lu_apply solves with the LU factors on the buffer x of n rows and r
// columns, in place. The flag trans solves the system of the transpose.
fn lu_apply(lu: f64[:], n: i64, piv: f64[:], x: f64[>..]&, r: i64, trans: bool) {
    var i: i64 = 0;
    if !trans {
        i = 0;
        while i < n {
            let p = piv[i] as i64;
            if p != i { swap_rows(x, r, i, p); }
            i += 1;
        }
        i = 1;
        while i < n {
            var c: i64 = 0;
            while c < i {
                var j: i64 = 0;
                while j < r { x[i * r + j] = x[i * r + j] - lu[i * n + c] * x[c * r + j]; j += 1; }
                c += 1;
            }
            i += 1;
        }
        i = n - 1;
        while i >= 0 {
            var c: i64 = i + 1;
            while c < n {
                var j: i64 = 0;
                while j < r { x[i * r + j] = x[i * r + j] - lu[i * n + c] * x[c * r + j]; j += 1; }
                c += 1;
            }
            var j: i64 = 0;
            while j < r { x[i * r + j] = x[i * r + j] / lu[i * n + i]; j += 1; }
            i -= 1;
        }
        return;
    }
    i = 0;
    while i < n {
        var c: i64 = 0;
        while c < i {
            var j: i64 = 0;
            while j < r { x[i * r + j] = x[i * r + j] - lu[c * n + i] * x[c * r + j]; j += 1; }
            c += 1;
        }
        var j: i64 = 0;
        while j < r { x[i * r + j] = x[i * r + j] / lu[i * n + i]; j += 1; }
        i += 1;
    }
    i = n - 1;
    while i >= 0 {
        var c: i64 = i + 1;
        while c < n {
            var j: i64 = 0;
            while j < r { x[i * r + j] = x[i * r + j] - lu[c * n + i] * x[c * r + j]; j += 1; }
            c += 1;
        }
        i -= 1;
    }
    i = n - 1;
    while i >= 0 {
        let p = piv[i] as i64;
        if p != i { swap_rows(x, r, i, p); }
        i -= 1;
    }
}

fn swap_rows(x: f64[>..]&, r: i64, a: i64, b: i64) {
    var j: i64 = 0;
    while j < r {
        let t = x[a * r + j];
        x[a * r + j] = x[b * r + j];
        x[b * r + j] = t;
        j += 1;
    }
}

// lu_solve solves a system with the factors of lu_factor. The right
// side holds one column or a matrix. The flag trans solves the system
// of the transpose. A zero on the diagonal of the factor is a fault.
fn lu_solve(luf, piv, b, trans: bool = false) -> Arr, Fault {
    if luf.ndim != 2 || luf.shape[0] != luf.shape[1] {
        return empty_arr(), fault(F_SHAPE, "lu_solve", "the factors are not a square matrix");
    }
    let n = luf.shape[0];
    if piv.data.len != n {
        return empty_arr(), fault(F_SHAPE, "lu_solve", "the row of swaps does not agree with the factors");
    }
    if b.ndim < 1 || b.ndim > 2 || b.shape[0] != n {
        return empty_arr(), fault(F_SHAPE, "lu_solve", "the right side does not agree with the factors");
    }
    var r: i64 = 1;
    if b.ndim == 2 { r = b.shape[1]; }
    var i: i64 = 0;
    while i < n {
        if luf.data[i * n + i] == 0.0 {
            return empty_arr(), fault(F_VALUE, "lu_solve", "the matrix is singular");
        }
        let p = piv.data[i];
        if !(p >= 0.0 && p < n) {
            return empty_arr(), fault(F_RANGE, "lu_solve", "a swap stands outside the matrix");
        }
        i += 1;
    }
    var x = copy_slice(b.data);
    lu_apply(luf.data, n, piv.data, x, r, trans);
    var s: i64[..8] = [];
    s.push(n);
    if b.ndim == 2 { s.push(r); }
    return Arr { ndim: b.ndim, shape: s, data: copy(x) }, no_fault();
}

// solve_triangular solves a system with a triangular matrix. The flag
// lower picks the lower triangle, and the other triangle stays unread.
// The value trans of 1 or 2 solves the system of the transpose. The flag
// unit_diagonal takes the diagonal as ones. A zero on the diagonal is a
// fault.
fn solve_triangular(a, b, lower: bool = false, trans: i64 = 0, unit_diagonal: bool = false) -> Arr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return empty_arr(), fault(F_SHAPE, "solve_triangular", "the matrix is not square");
    }
    let n = a.shape[0];
    if b.ndim < 1 || b.ndim > 2 || b.shape[0] != n {
        return empty_arr(), fault(F_SHAPE, "solve_triangular", "the right side does not agree with the matrix");
    }
    if trans < 0 || trans > 2 {
        return empty_arr(), fault(F_VALUE, "solve_triangular", "trans holds a value outside zero to two");
    }
    var r: i64 = 1;
    if b.ndim == 2 { r = b.shape[1]; }
    var i: i64 = 0;
    while i < n {
        if !unit_diagonal && a.data[i * n + i] == 0.0 {
            return empty_arr(), fault(F_VALUE, "solve_triangular", "the matrix is singular");
        }
        i += 1;
    }
    let t = trans != 0;
    // The matrix of the system is the triangle or its transpose.
    var eff_lower = lower;
    if t { eff_lower = !lower; }
    var x = copy_slice(b.data);
    var step: i64 = 0;
    while step < n {
        var row = step;
        if !eff_lower { row = n - 1 - step; }
        var c: i64 = 0;
        while c < n {
            var use = false;
            if eff_lower { use = c < row; } else { use = c > row; }
            if use {
                var v = a.data[row * n + c];
                if t { v = a.data[c * n + row]; }
                var j: i64 = 0;
                while j < r { x[row * r + j] = x[row * r + j] - v * x[c * r + j]; j += 1; }
            }
            c += 1;
        }
        if !unit_diagonal {
            var j: i64 = 0;
            while j < r { x[row * r + j] = x[row * r + j] / a.data[row * n + row]; j += 1; }
        }
        step += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    if b.ndim == 2 { s.push(r); }
    return Arr { ndim: b.ndim, shape: s, data: copy(x) }, no_fault();
}

// block_diag puts two arrays on the diagonal of one matrix. A row
// counts as a matrix of one row. Call it again for more arrays.
fn block_diag(a, b) -> Arr, Fault {
    if a.ndim > 2 || b.ndim > 2 {
        return empty_arr(), fault(F_SHAPE, "block_diag", "an operand holds more than two dimensions");
    }
    var ra: i64 = 1;
    var ca = a.shape[0];
    if a.ndim == 2 { ra = a.shape[0]; ca = a.shape[1]; }
    var rb: i64 = 1;
    var cb = b.shape[0];
    if b.ndim == 2 { rb = b.shape[0]; cb = b.shape[1]; }
    if (ra + rb) * (ca + cb) > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "block_diag", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < ra + rb {
        var j: i64 = 0;
        while j < ca + cb {
            if i < ra && j < ca { d.push(a.data[i * ca + j]); }
            else if i >= ra && j >= ca { d.push(b.data[(i - ra) * cb + (j - ca)]); }
            else { d.push(0.0); }
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d), ra + rb, ca + cb), no_fault();
}

// toeplitz gives the matrix with constant diagonals. The row c holds
// the first column and the row r holds the first row. An empty r means
// r equal to c. The first element of r stays unread.
fn toeplitz(c: f64[:], r: f64[:] = []) -> Arr, Fault {
    if c.len < 1 { return empty_arr(), fault(F_VALUE, "toeplitz", "the column holds no value"); }
    var cols = r.len;
    if cols == 0 { cols = c.len; }
    if c.len * cols > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "toeplitz", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < c.len {
        var j: i64 = 0;
        while j < cols {
            if i >= j { d.push(c[i - j]); }
            else if r.len == 0 { d.push(c[j - i]); }
            else { d.push(r[j - i]); }
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d), c.len, cols), no_fault();
}

// hankel gives the matrix with constant anti-diagonals. The row c holds
// the first column and the row r holds the last row. An empty r means
// zeros. The first element of r stays unread.
fn hankel(c: f64[:], r: f64[:] = []) -> Arr, Fault {
    if c.len < 1 { return empty_arr(), fault(F_VALUE, "hankel", "the column holds no value"); }
    var cols = r.len;
    if cols == 0 { cols = c.len; }
    if c.len * cols > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "hankel", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < c.len {
        var j: i64 = 0;
        while j < cols {
            let k = i + j;
            if k < c.len { d.push(c[k]); }
            else if r.len == 0 { d.push(0.0); }
            else { d.push(r[k - c.len + 1]); }
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d), c.len, cols), no_fault();
}

// circulant gives the matrix whose columns turn the first column.
fn circulant(c: f64[:]) -> Arr, Fault {
    let n = c.len;
    if n < 1 { return empty_arr(), fault(F_VALUE, "circulant", "the column holds no value"); }
    if n * n > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "circulant", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            d.push(c[(i - j + n) % n]);
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d), n, n), no_fault();
}

