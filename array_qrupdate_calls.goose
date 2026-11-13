// The public calls qr_insert and qr_delete.
//
namespace scientific;

import mathshim;
import array_shape;

// qr_insert puts the rows or the columns of u before the place k of a
// QR decomposition. The factors come from the decomposition of a
// matrix with at least as many rows as columns. The word which holds
// row or col. The result holds the new orthogonal factor and the new
// triangle. The method follows scipy.linalg.qr_insert.
fn qr_insert(q, r, u, k: i64, which: const u8[:]) -> Arr, Arr, Fault {
    let f = qi_check_operands(q, r, u);
    if f.kind != F_NONE {
        return empty_arr(), empty_arr(), f;
    }
    let ff = qu_finite(q, "qr_insert");
    if ff.kind != F_NONE {
        return empty_arr(), empty_arr(), ff;
    }
    let fg = qu_finite(r, "qr_insert");
    if fg.kind != F_NONE {
        return empty_arr(), empty_arr(), fg;
    }
    let fh = qu_finite(u, "qr_insert");
    if fh.kind != F_NONE {
        return empty_arr(), empty_arr(), fh;
    }
    if which != "row" && which != "col" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "qr_insert", "the word which holds neither row nor col");
    }
    let m = q.shape[0];
    let o = q.shape[1];
    let n = r.shape[1];
    let eco = m != o && r.shape[0] == r.shape[1];
    if which == "row" {
        if k < -m || k > m {
            return empty_arr(), empty_arr(), fault(F_RANGE, "qr_insert", "the place k lies outside the rows");
        }
        var k1 = k;
        if k1 < 0 { k1 += m; }
        var p: i64 = 1;
        if u.ndim == 2 {
            p = u.shape[0];
            if u.shape[1] != n {
                return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_insert", "the vector holds a count of columns that differs from the triangle");
            }
        } else {
            if u.shape[0] != n {
                return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_insert", "the vector holds a count of columns that differs from the triangle");
            }
        }
        // The copies of the factors hold the changes.
        var qb: f64[>..] = [];
        var i: i64 = 0;
        while i < q.data.len { qb.push(q.data[i]); i += 1; }
        var rb: f64[>..] = [];
        i = 0;
        while i < r.data.len { rb.push(r.data[i]); i += 1; }
        var ub: f64[>..] = [];
        i = 0;
        while i < u.data.len { ub.push(u.data[i]); i += 1; }
        if eco {
            // The new orthogonal factor holds m plus p rows and n
            // plus p columns. The inserted rows start as an
            // identity.
            let nq = m + p;
            let ncq = n + p;
            var qn: f64[>..] = [];
            i = 0;
            while i < nq * ncq { qn.push(0.0); i += 1; }
            i = 0;
            while i < m {
                var j: i64 = 0;
                while j < n {
                    qn[i * ncq + j] = qb[i * o + j];
                    j += 1;
                }
                i += 1;
            }
            var j: i64 = 0;
            while j < p {
                qn[(m + j) * ncq + n + j] = 1.0;
                j += 1;
            }
            if p == 1 {
                qi_thin_row_insert(qn, nq, n, rb, ub, k1);
            } else {
                qi_thin_block_row_insert(qn, nq, n, rb, ub, k1, p);
            }
            // The result drops the last p columns.
            var qr: f64[>..] = [];
            i = 0;
            while i < nq {
                j = 0;
                while j < n {
                    qr.push(qn[i * ncq + j]);
                    j += 1;
                }
                i += 1;
            }
            return mat2(copy(qr), nq, n), mat2(copy(rb), n, n), no_fault();
        }
        // The full form. The new triangle holds the old triangle and
        // the inserted rows at the bottom.
        let nq = m + p;
        var qn: f64[>..] = [];
        i = 0;
        while i < nq * nq { qn.push(0.0); i += 1; }
        i = 0;
        while i < m {
            var j: i64 = 0;
            while j < m {
                qn[i * nq + j] = qb[i * o + j];
                j += 1;
            }
            i += 1;
        }
        var j: i64 = 0;
        while j < p {
            qn[(m + j) * nq + m + j] = 1.0;
            j += 1;
        }
        var rn: f64[>..] = [];
        i = 0;
        while i < nq * n { rn.push(0.0); i += 1; }
        i = 0;
        while i < m * n { rn[i] = rb[i]; i += 1; }
        i = 0;
        while i < p * n { rn[m * n + i] = ub[i]; i += 1; }
        if p == 1 {
            qi_row_insert_full1(qn, nq, n, rn, k1);
        } else {
            qi_block_row_insert(qn, nq, n, rn, k1, p);
        }
        return mat2(copy(qn), nq, nq), mat2(copy(rn), nq, n), no_fault();
    }
    // The columns.
    if k < -n || k > n {
        return empty_arr(), empty_arr(), fault(F_RANGE, "qr_insert", "the place k lies outside the columns");
    }
    var k1 = k;
    if k1 < 0 { k1 += n; }
    var p: i64 = 1;
    if u.ndim == 2 { p = u.shape[1]; }
    if u.shape[0] != m {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_insert", "the vector holds a count of rows that differs from the orthogonal factor");
    }
    // A matrix of one row takes the change in its triangle.
    if m == 1 {
        var qb: f64[>..] = [];
        var i: i64 = 0;
        while i < q.data.len { qb.push(q.data[i]); i += 1; }
        var rb: f64[>..] = [];
        i = 0;
        while i < r.data.len { rb.push(r.data[i]); i += 1; }
        var ub: f64[>..] = [];
        i = 0;
        while i < u.data.len { ub.push(u.data[i]); i += 1; }
        // The new triangle holds the old columns and the product of
        // the orthogonal factor with the vector.
        let nc = n + p;
        var rn: f64[>..] = [];
        i = 0;
        while i < nc { rn.push(0.0); i += 1; }
        i = 0;
        while i < n {
            if i < k1 { rn[i] = rb[i]; }
            if i >= k1 { rn[i + p] = rb[i]; }
            i += 1;
        }
        i = 0;
        while i < p {
            rn[k1 + i] = qb[0] * ub[i];
            i += 1;
        }
        return mat2(copy(qb), 1, 1), mat2(copy(rn), 1, nc), no_fault();
    }
    if eco {
        var p_eco: i64 = p;
        var p_full: i64 = 0;
        if n + p > m {
            p_eco = m - n;
            p_full = p - p_eco;
        }
        let ncq = n + p_eco;
        var qn: f64[>..] = [];
        var i: i64 = 0;
        while i < m * ncq { qn.push(0.0); i += 1; }
        i = 0;
        while i < m {
            var j: i64 = 0;
            while j < o {
                qn[i * ncq + j] = q.data[i * o + j];
                j += 1;
            }
            i += 1;
        }
        let nrr = n + p_eco;
        let ncr = n + p;
        var rn: f64[>..] = [];
        i = 0;
        while i < nrr * ncr { rn.push(0.0); i += 1; }
        i = 0;
        while i < n {
            var j: i64 = 0;
            while j < k1 {
                rn[i * ncr + j] = r.data[i * n + j];
                j += 1;
            }
            j = k1;
            while j < n {
                rn[i * ncr + j + p] = r.data[i * n + j];
                j += 1;
            }
            i += 1;
        }
        var ub: f64[>..] = [];
        i = 0;
        while i < u.data.len { ub.push(u.data[i]); i += 1; }
        let info = qi_thin_col_insert(qn, m, ncq, rn, ncr, ub, p, n, k1, p_eco, p_full);
        if info == 2 {
            return empty_arr(), empty_arr(), fault(F_VALUE, "qr_insert", "a column of the vector lies in the span of the orthogonal factor");
        }
        if p_full > 0 {
            // The rest of the columns joins as the product of the
            // transposed factor with the vector.
            i = 0;
            while i < m {
                var jj: i64 = 0;
                while jj < p_full {
                    var s: f64 = 0.0;
                    var kk: i64 = 0;
                    while kk < m {
                        s += qn[kk * ncq + i] * ub[kk * p + p_eco + jj];
                        kk += 1;
                    }
                    rn[i * ncr + k1 + p_eco + jj] = s;
                    jj += 1;
                }
                i += 1;
            }
            qi_block_col_insert(qn, m, ncr, rn, k1 + p_eco, p_full);
        }
        return mat2(copy(qn), m, ncq), mat2(copy(rn), nrr, ncr), no_fault();
    }
    // The full form.
    let ncr = n + p;
    var rn: f64[>..] = [];
    var i: i64 = 0;
    while i < m * ncr { rn.push(0.0); i += 1; }
    i = 0;
    while i < m {
        var j: i64 = 0;
        while j < k1 {
            rn[i * ncr + j] = r.data[i * n + j];
            j += 1;
        }
        j = k1;
        while j < n {
            rn[i * ncr + j + p] = r.data[i * n + j];
            j += 1;
        }
        i += 1;
    }
    // The new columns take the product of the transposed factor with
    // the vector.
    var ub: f64[>..] = [];
    i = 0;
    while i < u.data.len { ub.push(u.data[i]); i += 1; }
    i = 0;
    while i < m {
        var jj: i64 = 0;
        while jj < p {
            var s: f64 = 0.0;
            var kk: i64 = 0;
            while kk < m {
                s += q.data[kk * o + i] * ub[kk * p + jj];
                kk += 1;
            }
            rn[i * ncr + k1 + jj] = s;
            jj += 1;
        }
        i += 1;
    }
    var qb: f64[>..] = [];
    i = 0;
    while i < q.data.len { qb.push(q.data[i]); i += 1; }
    if p == 1 {
        qi_col_insert_full(qb, m, ncr, rn, k1);
    } else {
        qi_block_col_insert(qb, m, ncr, rn, k1, p);
    }
    return mat2(copy(qb), m, o), mat2(copy(rn), m, ncr), no_fault();
}

// qr_delete removes p rows or p columns at the place k of a QR
// decomposition. The word which holds row or col. The result holds
// the new orthogonal factor and the new triangle. The method follows
// scipy.linalg.qr_delete.
fn qr_delete(q, r, k: i64, p: i64, which: const u8[:]) -> Arr, Arr, Fault {
    if which != "row" && which != "col" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "qr_delete", "the word which holds neither row nor col");
    }
    var f = qi_check_operands(q, r, r);
    if f.kind != F_NONE {
        return empty_arr(), empty_arr(), f;
    }
    let ff = qu_finite(q, "qr_delete");
    if ff.kind != F_NONE {
        return empty_arr(), empty_arr(), ff;
    }
    let fg = qu_finite(r, "qr_delete");
    if fg.kind != F_NONE {
        return empty_arr(), empty_arr(), fg;
    }
    let m = q.shape[0];
    let o = q.shape[1];
    let n = r.shape[1];
    let eco = m != o && r.shape[0] == r.shape[1];
    if p < 1 {
        return empty_arr(), empty_arr(), fault(F_RANGE, "qr_delete", "the count p holds a value below one");
    }
    if which == "row" {
        if k < -m || k >= m {
            return empty_arr(), empty_arr(), fault(F_RANGE, "qr_delete", "the place k lies outside the rows");
        }
        if p > m {
            return empty_arr(), empty_arr(), fault(F_RANGE, "qr_delete", "the count p holds more rows than the factors hold");
        }
        var k1 = k;
        if k1 < 0 { k1 += m; }
        if k1 + p > m {
            return empty_arr(), empty_arr(), fault(F_RANGE, "qr_delete", "the place k holds too few rows for the count");
        }
        var qb: f64[>..] = [];
        var i: i64 = 0;
        while i < q.data.len { qb.push(q.data[i]); i += 1; }
        var rb: f64[>..] = [];
        i = 0;
        while i < r.data.len { rb.push(r.data[i]); i += 1; }
        if eco {
            // A triangle of one column takes a scale.
            if n == 1 {
                // The rows that stay hold the norm of the factor.
                var nq = m - p;
                var qn: f64[>..] = [];
                var row: i64 = 0;
                while row < m {
                    if row < k1 || row >= k1 + p {
                        qn.push(qb[row * o]);
                    }
                    row += 1;
                }
                let norm = qu_norm(qn, 0, nq);
                if norm == 0.0 {
                    return empty_arr(), empty_arr(), fault(F_VALUE, "qr_delete", "the rows that stay hold no length");
                }
                i = 0;
                while i < nq { qn[i] = qn[i] / norm; i += 1; }
                rb[0] = rb[0] * norm;
                return mat2(copy(qn), nq, n), mat2(copy(rb), n, n), no_fault();
            }
            var p_eco: i64 = p;
            var p_full: i64 = 0;
            if m - p < n {
                p_eco = m - n;
                p_full = p - p_eco;
            }
            let ok = qi_thin_row_delete(qb, m, n, rb, k1, p_eco, p_full);
            if !ok {
                return empty_arr(), empty_arr(), fault(F_VALUE, "qr_delete", "the reorthogonalization failed");
            }
            // The result drops the rows of the block at the front and
            // the rows of the thin part at the bottom.
            var r0: i64 = p_full;
            var r1: i64 = m - p_eco;
            var c0: i64 = p_full;
            var c1: i64 = o;
            if p_full == 0 {
                r0 = 0;
                r1 = m - p;
                c0 = 0;
                c1 = o;
            }
            var nr = r1 - r0;
            var nc = c1 - c0;
            var qn: f64[>..] = [];
            i = r0;
            while i < r1 {
                var j: i64 = c0;
                while j < c1 {
                    qn.push(qb[i * o + j]);
                    j += 1;
                }
                i += 1;
            }
            // The triangle holds the rows of the block at the front
            // and the rows of the thin part above.
            var rrows: i64 = n;
            if p_full > 0 { rrows = n - p_full; }
            var rr0: i64 = 0;
            if p_full > 0 { rr0 = p_full; }
            var rrn: f64[>..] = [];
            i = rr0;
            while i < n {
                var j: i64 = 0;
                while j < n {
                    rrn.push(rb[i * n + j]);
                    j += 1;
                }
                i += 1;
            }
            return mat2(copy(qn), nr, nc), mat2(copy(rrn), rrows, n), no_fault();
        }
        // The full form.
        let ok = qi_block_row_delete(qb, m, n, rb, k1, p);
        if !ok {
            return empty_arr(), empty_arr(), fault(F_VALUE, "qr_delete", "the reorthogonalization failed");
        }
        var nr = m - p;
        var qn: f64[>..] = [];
        i = p;
        while i < m {
            var j: i64 = p;
            while j < o {
                qn.push(qb[i * o + j]);
                j += 1;
            }
            i += 1;
        }
        var rrn: f64[>..] = [];
        i = p;
        while i < m {
            var j: i64 = 0;
            while j < n {
                rrn.push(rb[i * n + j]);
                j += 1;
            }
            i += 1;
        }
        return mat2(copy(qn), nr, o - p), mat2(copy(rrn), nr, n), no_fault();
    }
    // The columns.
    if k < -n || k >= n {
        return empty_arr(), empty_arr(), fault(F_RANGE, "qr_delete", "the place k lies outside the columns");
    }
    if p > n {
        return empty_arr(), empty_arr(), fault(F_RANGE, "qr_delete", "the count p holds more columns than the factors hold");
    }
    var k1 = k;
    if k1 < 0 { k1 += n; }
    if k1 + p > n {
        return empty_arr(), empty_arr(), fault(F_RANGE, "qr_delete", "the place k holds too few columns for the count");
    }
    var oo: i64 = n;
    if !eco { oo = m; }
    var qb: f64[>..] = [];
    var i: i64 = 0;
    while i < q.data.len { qb.push(q.data[i]); i += 1; }
    var rb: f64[>..] = [];
    i = 0;
    while i < r.data.len { rb.push(r.data[i]); i += 1; }
    if p == 1 {
        qi_col_delete(qb, m, oo, rb, n, k1);
    } else {
        // The columns to remove move to the end. The turns take the
        // subdiagonals away.
        var lim = oo;
        if n < lim { lim = n; }
        var j: i64 = k1;
        while j < n - p {
            i = 0;
            while i < lim {
                rb[i * n + j] = rb[i * n + j + p];
                i += 1;
            }
            j += 1;
        }
        qu_p_subdiag(qb, m, oo, rb, n, n - p, k1, p);
    }
    if eco {
        // The result drops the last p columns of both factors.
        var nc = o - p;
        var qn: f64[>..] = [];
        i = 0;
        while i < m {
            var j: i64 = 0;
            while j < nc {
                qn.push(qb[i * oo + j]);
                j += 1;
            }
            i += 1;
        }
        var rrn: f64[>..] = [];
        i = 0;
        while i < nc {
            var j: i64 = 0;
            while j < nc {
                rrn.push(rb[i * n + j]);
                j += 1;
            }
            i += 1;
        }
        return mat2(copy(qn), m, nc), mat2(copy(rrn), nc, nc), no_fault();
    }
    var nc = n - p;
    var rrn: f64[>..] = [];
    i = 0;
    while i < m {
        var j: i64 = 0;
        while j < nc {
            rrn.push(rb[i * n + j]);
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(qb), m, o), mat2(copy(rrn), m, nc), no_fault();
}

// qi_check_operands holds the checks of the shapes that qr_insert and
// qr_delete share.
fn qi_check_operands(q, r, u) -> Fault {
    // The count of dimensions must match the count of the shape.
    if q.ndim != q.shape.len || r.ndim != r.shape.len || u.ndim != u.shape.len {
        return fault(F_SHAPE, "qr_insert", "an operand holds a count of dimensions that does not match its shape");
    }
    if q.ndim != 2 || r.ndim != 2 {
        return fault(F_SHAPE, "qr_insert", "a factor holds a count of dimensions that is not two");
    }
    var side: i64 = 0;
    while side < q.shape.len {
        if q.shape[side] < 1 || q.shape[side] > QRU_MAX_SIDE { return fault(F_SHAPE, "qr_insert", "a factor holds a side that is out of the limits"); }
        side += 1;
    }
    side = 0;
    while side < r.shape.len {
        if r.shape[side] < 1 || r.shape[side] > QRU_MAX_SIDE { return fault(F_SHAPE, "qr_insert", "a factor holds a side that is out of the limits"); }
        side += 1;
    }
    side = 0;
    while side < u.shape.len {
        if u.shape[side] < 1 || u.shape[side] > QRU_MAX_SIDE { return fault(F_SHAPE, "qr_insert", "a vector holds a side that is out of the limits"); }
        side += 1;
    }
    if q.data.len != numel(q.shape) || r.data.len != numel(r.shape) || u.data.len != numel(u.shape) {
        return fault(F_SHAPE, "qr_insert", "an operand holds a buffer that does not match its shape");
    }
    if q.shape[1] != r.shape[0] {
        return fault(F_SHAPE, "qr_insert", "the factors hold a shape that does not fit together");
    }
    let m = q.shape[0];
    let o = q.shape[1];
    if m != o && r.shape[0] != r.shape[1] {
        return fault(F_SHAPE, "qr_insert", "the factors hold a shape that does not fit together");
    }
    return no_fault();
}
