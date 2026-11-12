// The sparse matrices of coordinates.
//
// A matrix of the type Coo holds the stored values and their places. The
// buffer holds, in this order, the row of every entry, the column of every
// entry and the value of every entry. The format has no order, so a place
// may hold several entries. The count of the entries counts the
// duplicates, as it does in SciPy.
//
// The dense form and the compressed rows add the entries of one place up,
// as SciPy adds them up.
namespace scientific;

// Coo holds a sparse matrix of coordinates.
struct Coo { rows: i64, cols: i64, nnz: i64, buf: f64[>..] }

fn coo_empty() -> Coo {
    return Coo { rows: 0, cols: 0, nnz: 0, buf: [] };
}

// coo_row gives the row of the entry k.
fn coo_row(s, k: i64) -> i64 {
    return s.buf[k] as i64;
}

// coo_col gives the column of the entry k.
fn coo_col(s, k: i64) -> i64 {
    return s.buf[s.nnz + k] as i64;
}

// coo_val gives the value of the entry k.
fn coo_val(s, k: i64) -> f64 {
    return s.buf[2 * s.nnz + k];
}

// coo_from_parts builds a matrix from the rows, the columns and the values
// of the entries, which the caller keeps in order.
fn coo_from_parts(rows: i64, cols: i64, ri: f64[:], ci: f64[:], v: f64[:]) -> Coo {
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < ri.len { buf.push(ri[i]); i += 1; }
    i = 0;
    while i < ci.len { buf.push(ci[i]); i += 1; }
    i = 0;
    while i < v.len { buf.push(v[i]); i += 1; }
    return Coo { rows: rows, cols: cols, nnz: ri.len, buf: buf };
}

// coo_from_triplets builds a matrix from the rows, the columns and the
// values of the entries. The format keeps the duplicate entries, as the
// format of SciPy keeps them.
fn coo_from_triplets(rows: i64, cols: i64, ri: f64[:], ci: f64[:], v: f64[:]) -> Coo, Fault {
    let c = csr_check(rows, cols, "coo_from_triplets");
    if c.kind != F_NONE { return coo_empty(), c; }
    let t = sparse_check_triplets(rows, cols, ri, ci, v, "coo_from_triplets");
    if t.kind != F_NONE { return coo_empty(), t; }
    return coo_from_parts(rows, cols, ri, ci, v), no_fault();
}

// coo_to_csr gives the compressed rows of a matrix of coordinates. The
// entries of one place add up, as the duplicates of SciPy add them up.
fn coo_to_csr(s) -> Csr, Fault {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var k: i64 = 0;
    while k < s.nnz {
        ri.push(coo_row(s, k) as f64);
        ci.push(coo_col(s, k) as f64);
        v.push(coo_val(s, k));
        k += 1;
    }
    return csr_from_triplets(s.rows, s.cols, ri, ci, v);
}

// coo_from_csr gives the coordinates of a matrix of compressed rows.
fn coo_from_csr(s) -> Coo, Fault {
    let nnz = csr_ptr(s, s.rows);
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < s.rows {
        var k = csr_ptr(s, i);
        let e = csr_ptr(s, i + 1);
        while k < e {
            buf.push(i as f64);
            k += 1;
        }
        i += 1;
    }
    i = 0;
    while i < s.rows {
        var k = csr_ptr(s, i);
        let e = csr_ptr(s, i + 1);
        while k < e {
            buf.push(csr_col(s, k) as f64);
            k += 1;
        }
        i += 1;
    }
    i = 0;
    while i < nnz {
        buf.push(csr_val(s, i));
        i += 1;
    }
    return Coo { rows: s.rows, cols: s.cols, nnz: nnz, buf: buf }, no_fault();
}

// coo_to_dense gives the dense matrix of a sparse matrix.
fn coo_to_dense(s) -> Arr, Fault {
    let c, f = coo_to_csr(s);
    if f.kind != F_NONE { return empty_arr(), f; }
    return csr_to_dense(c);
}

// coo_from_dense gives the coordinates of a dense matrix.
fn coo_from_dense(a) -> Coo, Fault {
    let s, f = csr_from_dense(a);
    if f.kind != F_NONE { return coo_empty(), f; }
    return coo_from_csr(s);
}

// coo_transpose gives the transpose of a matrix.
fn coo_transpose(s) -> Coo, Fault {
    var buf: f64[>..] = [];
    var k: i64 = 0;
    while k < s.nnz { buf.push(coo_col(s, k) as f64); k += 1; }
    k = 0;
    while k < s.nnz { buf.push(coo_row(s, k) as f64); k += 1; }
    k = 0;
    while k < s.nnz { buf.push(coo_val(s, k)); k += 1; }
    return Coo { rows: s.cols, cols: s.rows, nnz: s.nnz, buf: buf }, no_fault();
}

// coo_get gives the value at a place, zero when the place holds none. The
// entries of one place add up.
fn coo_get(s, i: i64, j: i64) -> f64, Fault {
    if i < 0 || i >= s.rows || j < 0 || j >= s.cols {
        return 0.0, fault(F_RANGE, "coo_get", "the place lies outside the matrix");
    }
    var t: f64 = 0.0;
    var k: i64 = 0;
    while k < s.nnz {
        if coo_row(s, k) == i && coo_col(s, k) == j { t += coo_val(s, k); }
        k += 1;
    }
    return t, no_fault();
}

// coo_matvec gives the product of a matrix and a vector.
fn coo_matvec(s, x: f64[:]) -> Arr, Fault {
    if x.len != s.cols {
        return empty_arr(), fault(F_SHAPE, "coo_matvec", "the vector does not hold one value per column");
    }
    var out: f64[>..] = nd_zeros(s.rows);
    var k: i64 = 0;
    while k < s.nnz {
        out[coo_row(s, k)] += coo_val(s, k) * x[coo_col(s, k)];
        k += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// coo_scale multiplies every value by a number.
fn coo_scale(s, a: f64) -> Coo {
    var buf = copy(s.buf);
    let n = s.nnz;
    var i: i64 = 0;
    while i < n { buf[2 * n + i] = buf[2 * n + i] * a; i += 1; }
    return Coo { rows: s.rows, cols: s.cols, nnz: n, buf: buf };
}
