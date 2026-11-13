// The rank one and rank p update of a QR decomposition: qr_update.
//
namespace scientific;

import mathshim;
import array_shape;

// qu_thin_rank1 updates the factors of a reduced decomposition by a
// term of rank one. The matrix q holds m rows and n columns, and the
// matrix r holds n rows and n columns.
fn qu_thin_rank1(q: f64[>..]&, m: i64, r: f64[>..]&, n: i64, u: f64[>..]&, v: f64[:], s: f64[>..]&) {
    qu_reorth(q, m, n, u, s);
    var c: f64 = 0.0;
    var sn: f64 = 0.0;
    var rr: f64 = 0.0;
    c, sn, rr = qu_lartg(s[n - 1], s[n]);
    s[n - 1] = rr;
    s[n] = 0.0;
    let t = r[(n - 1) * n + (n - 1)];
    let rlast = -t * sn;
    r[(n - 1) * n + (n - 1)] = t * c;
    qu_rot_col_vec(q, n, n - 1, u, m, c, sn);
    var j: i64 = n - 2;
    while j >= 0 {
        c, sn, rr = qu_lartg(s[j], s[j + 1]);
        s[j] = rr;
        s[j + 1] = 0.0;
        qu_rot_rows(r, n, j, j + 1, j, n - j, c, sn);
        qu_rot_cols(q, n, j, j + 1, 0, m, c, sn);
        j -= 1;
    }
    // The first row of the triangle takes the part of the change.
    var i: i64 = 0;
    while i < n {
        r[i] = r[i] + s[0] * v[i];
        i += 1;
    }
    // The triangle holds one row too many. The turns make it square.
    j = 0;
    while j < n - 1 {
        c, sn, rr = qu_lartg(r[j * n + j], r[(j + 1) * n + j]);
        r[j * n + j] = rr;
        r[(j + 1) * n + j] = 0.0;
        qu_rot_rows(r, n, j, j + 1, j + 1, n - j - 1, c, sn);
        qu_rot_cols(q, n, j, j + 1, 0, m, c, sn);
        j += 1;
    }
    c, sn, rr = qu_lartg(r[(n - 1) * n + (n - 1)], rlast);
    r[(n - 1) * n + (n - 1)] = rr;
    qu_rot_col_vec(q, n, n - 1, u, m, c, sn);
}

// qu_full_rank1 updates the factors of a full decomposition by a term
// of rank one. The matrix q holds m rows and m columns, and the matrix
// r holds m rows and n columns.
fn qu_full_rank1(q: f64[>..]&, m: i64, r: f64[>..]&, n: i64, u: f64[>..]&, v: f64[:]) {
    var j: i64 = m - 2;
    while j >= 0 {
        var c: f64 = 0.0;
        var s: f64 = 0.0;
        var rr: f64 = 0.0;
        c, s, rr = qu_lartg(u[j], u[j + 1]);
        u[j] = rr;
        u[j + 1] = 0.0;
        if n - j > 0 { qu_rot_rows(r, n, j, j + 1, j, n - j, c, s); }
        qu_rot_cols(q, m, j, j + 1, 0, m, c, s);
        j -= 1;
    }
    // The first row of the triangle takes the part of the change.
    var i: i64 = 0;
    while i < n {
        r[i] = r[i] + u[0] * v[i];
        i += 1;
    }
    // The turns make the triangle square.
    var lim = m - 1;
    if n < lim { lim = n; }
    j = 0;
    while j < lim {
        var c: f64 = 0.0;
        var s: f64 = 0.0;
        var rr: f64 = 0.0;
        c, s, rr = qu_lartg(r[j * n + j], r[(j + 1) * n + j]);
        r[j * n + j] = rr;
        r[(j + 1) * n + j] = 0.0;
        if j + 1 < m { qu_rot_rows(r, n, j, j + 1, j + 1, n - j - 1, c, s); }
        qu_rot_cols(q, m, j, j + 1, 0, m, c, s);
        j += 1;
    }
}

// qu_geqrf gives the turns of the Householder method for the part of
// a matrix that starts at the row row0 and holds p columns. The part
// changes. The factors of the turns go into the work array.
fn qu_geqrf(u: f64[>..]&, m: i64, p: i64, row0: i64, tau: f64[>..]&) {
    var lim = m - row0;
    if p < lim { lim = p; }
    var i: i64 = 0;
    while i < lim {
        // The turn removes the part of the column below the pivot.
        let alpha = u[(row0 + i) * p + i];
        var beta: f64 = 0.0;
        var t: f64 = 0.0;
        beta, t = qu_larfg(alpha, u, (row0 + i + 1) * p + i, p, m - row0 - i - 1);
        u[(row0 + i) * p + i] = beta;
        tau[i] = t;
        if t != 0.0 {
            // The turn touches the columns that come after.
            var c: i64 = i + 1;
            while c < p {
                var w: f64 = 0.0;
                var k: i64 = 0;
                while k < m - row0 - i {
                    var vk: f64 = 1.0;
                    if k > 0 { vk = u[(row0 + i + k) * p + i]; }
                    w += vk * u[(row0 + i + k) * p + c];
                    k += 1;
                }
                k = 0;
                while k < m - row0 - i {
                    var vk: f64 = 1.0;
                    if k > 0 { vk = u[(row0 + i + k) * p + i]; }
                    u[(row0 + i + k) * p + c] = u[(row0 + i + k) * p + c] - t * vk * w;
                    k += 1;
                }
                c += 1;
            }
        }
        i += 1;
    }
}

// qu_ormqr_right applies the turns of a part of a matrix to the
// columns of a matrix from the right. The part starts at the row row0
// and holds p columns.
fn qu_ormqr_right(q: f64[>..]&, m: i64, o: i64, u: f64[:], tau: f64[:], p: i64, row0: i64) {
    var i: i64 = 0;
    while i < p {
        if tau[i] != 0.0 {
            // The length of the turned vector.
            let ni = o - row0 - i;
            var k: i64 = 0;
            while k < m {
                var w: f64 = 0.0;
                var c: i64 = 0;
                while c < ni {
                    var vk: f64 = 1.0;
                    if c > 0 { vk = u[(row0 + i + c) * p + i]; }
                    w += q[k * o + (row0 + i + c)] * vk;
                    c += 1;
                }
                c = 0;
                while c < ni {
                    var vk: f64 = 1.0;
                    if c > 0 { vk = u[(row0 + i + c) * p + i]; }
                    q[k * o + (row0 + i + c)] = q[k * o + (row0 + i + c)] - tau[i] * w * vk;
                    c += 1;
                }
                k += 1;
            }
        }
        i += 1;
    }
}

// qu_p_subdiag removes the p subdiagonals of a matrix with turns. The
// matrix q holds m rows and o columns, and the matrix r holds o rows
// and n columns.
fn qu_p_subdiag(q: f64[>..]&, m: i64, o: i64, r: f64[>..]&, rn: i64, nc: i64, k: i64, p: i64) {
    var lim = m - 1;
    if nc < lim { lim = nc; }
    var j: i64 = k;
    while j < lim {
        var last = p + 1;
        if o - j < last { last = o - j; }
        let rjj = r[j * rn + j];
        // The turn removes the part of the column below the pivot.
        var beta: f64 = 0.0;
        var tau: f64 = 0.0;
        beta, tau = qu_larfg(rjj, r, (j + 1) * rn + j, rn, last - 1);
        // The turn touches the columns that come after.
        if j + 1 < nc {
            var c: i64 = j + 1;
            while c < nc {
                var w: f64 = 0.0;
                var kk: i64 = 0;
                while kk < last {
                    var vk: f64 = 1.0;
                    if kk > 0 { vk = r[(j + kk) * rn + j]; }
                    w += vk * r[(j + kk) * rn + c];
                    kk += 1;
                }
                kk = 0;
                while kk < last {
                    var vk: f64 = 1.0;
                    if kk > 0 { vk = r[(j + kk) * rn + j]; }
                    r[(j + kk) * rn + c] = r[(j + kk) * rn + c] - tau * vk * w;
                    kk += 1;
                }
                c += 1;
            }
        }
        // The turn touches the columns of the orthogonal factor.
        var kk: i64 = 0;
        while kk < m {
            var w: f64 = 0.0;
            var c: i64 = 0;
            while c < last {
                var vk: f64 = 1.0;
                if c > 0 { vk = r[(j + c) * rn + j]; }
                w += q[kk * o + (j + c)] * vk;
                c += 1;
            }
            c = 0;
            while c < last {
                var vk: f64 = 1.0;
                if c > 0 { vk = r[(j + c) * rn + j]; }
                q[kk * o + (j + c)] = q[kk * o + (j + c)] - tau * w * vk;
                c += 1;
            }
            kk += 1;
        }
        // The part below the pivot is zero.
        var z: i64 = 0;
        while z < last - 1 {
            r[(j + 1 + z) * rn + j] = 0.0;
            z += 1;
        }
        r[j * rn + j] = beta;
        j += 1;
    }
}

// qu_rank_p updates the factors of a full decomposition by a term of
// rank p. The matrix q holds m rows and m columns, and the matrix r
// holds m rows and n columns. The matrix u holds m rows and p columns.
fn qu_rank_p(q: f64[>..]&, m: i64, r: f64[>..]&, n: i64, u: f64[>..]&, v: f64[>..]&, p: i64) -> Fault {
    var tau: f64[>..] = [];
    var i: i64 = 0;
    while i < p {
        tau.push(0.0);
        i += 1;
    }
    if m > n {
        // The part of the change below the first n rows takes a turn
        // of its own.
        qu_geqrf(u, m, p, n, tau);
        qu_ormqr_right(q, m, m, u, tau, p, n);
    }
    var i2: i64 = 0;
    while i2 < p {
        // The full form with more rows than columns starts one turn
        // above, at the first row below the triangle.
        var j: i64 = m - 2;
        if m > n { j = n + i2 - 1; }
        while j >= i2 {
            var c: f64 = 0.0;
            var s: f64 = 0.0;
            var rr: f64 = 0.0;
            c, s, rr = qu_lartg(u[j * p + i2], u[(j + 1) * p + i2]);
            u[j * p + i2] = rr;
            u[(j + 1) * p + i2] = 0.0;
            if p - i2 - 1 > 0 {
                qu_rot_rows(u, p, j, j + 1, i2 + 1, p - i2 - 1, c, s);
            }
            qu_rot_rows(r, n, j, j + 1, 0, n, c, s);
            qu_rot_cols(q, m, j, j + 1, 0, m, c, s);
            j -= 1;
        }
        i2 += 1;
    }
    // The change of the triangle is the product of the triangle of
    // the change and the right part of the change.
    var vold: f64[>..] = [];
    i = 0;
    while i < v.len {
        vold.push(v[i]);
        i += 1;
    }
    var i3: i64 = 0;
    while i3 < p {
        var row: i64 = 0;
        while row < n {
            var s: f64 = 0.0;
            var j2: i64 = i3;
            while j2 < p {
                s += u[i3 * p + j2] * vold[row * p + j2];
                j2 += 1;
            }
            v[row * p + i3] = s;
            row += 1;
        }
        i3 += 1;
    }
    // The first p rows of the triangle take the change.
    var j3: i64 = 0;
    while j3 < p {
        var row: i64 = 0;
        while row < n {
            r[j3 * n + row] = r[j3 * n + row] + v[row * p + j3];
            row += 1;
        }
        j3 += 1;
    }
    // The subdiagonals go away.
    qu_p_subdiag(q, m, m, r, n, n, 0, p);
    return no_fault();
}

// qr_update updates the factors of a QR decomposition by a low rank
// term. The factors come from the decomposition of a matrix. The
// update adds the product of the two vectors to the matrix. The
// result holds the new orthogonal factor and the new triangle.
//
// The factor q holds m rows and o columns, and the triangle r holds o
// rows and n columns. One of the two factors is square. The vector u
// holds m rows, and the vector v holds n rows. Each vector holds one
// column or p columns. The result follows the factors of SciPy.
fn qr_update(q, r, u, v) -> Arr, Arr, Fault {
    // The count of dimensions must match the count of the shape. A
    // shape that holds fewer values would make a read outside the
    // shape.
    if q.ndim != q.shape.len || r.ndim != r.shape.len || u.ndim != u.shape.len || v.ndim != v.shape.len {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_update", "an operand holds a count of dimensions that does not match its shape");
    }
    if q.ndim != 2 || r.ndim != 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_update", "a factor holds a count of dimensions that is not two");
    }
    // The sides are bounded before any product. The product of the
    // sides of a shape stays free of an overflow.
    var side: i64 = 0;
    while side < q.shape.len {
        if q.shape[side] < 1 || q.shape[side] > QRU_MAX_SIDE { return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_update", "a factor holds a side that is out of the limits"); }
        side += 1;
    }
    side = 0;
    while side < r.shape.len {
        if r.shape[side] < 1 || r.shape[side] > QRU_MAX_SIDE { return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_update", "a factor holds a side that is out of the limits"); }
        side += 1;
    }
    side = 0;
    while side < u.shape.len {
        if u.shape[side] < 1 || u.shape[side] > QRU_MAX_SIDE { return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_update", "a vector holds a side that is out of the limits"); }
        side += 1;
    }
    side = 0;
    while side < v.shape.len {
        if v.shape[side] < 1 || v.shape[side] > QRU_MAX_SIDE { return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_update", "a vector holds a side that is out of the limits"); }
        side += 1;
    }
    // A shape may lie about its buffer. The length of the buffer must
    // match the shape.
    if q.data.len != numel(q.shape) || r.data.len != numel(r.shape) || u.data.len != numel(u.shape) || v.data.len != numel(v.shape) {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_update", "an operand holds a buffer that does not match its shape");
    }
    if u.ndim != v.ndim {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_update", "the two vectors hold a count of dimensions that differs");
    }
    if q.shape[1] != r.shape[0] {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_update", "the factors hold a shape that does not fit together");
    }
    let m = q.shape[0];
    let o = q.shape[1];
    let n = r.shape[1];
    // The reduced form needs a square triangle.
    if m != o && r.shape[0] != r.shape[1] {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_update", "the factors hold a shape that does not fit together");
    }
    if u.shape[0] != m {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_update", "the left vector holds a count of rows that differs");
    }
    if v.shape[0] != n {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_update", "the right vector holds a count of rows that differs");
    }
    var p: i64 = 1;
    if u.ndim == 2 {
        if u.shape[1] != v.shape[1] {
            return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_update", "the two vectors hold a count of columns that differs");
        }
        p = u.shape[1];
    }
    if p > n || p > m {
        return empty_arr(), empty_arr(), fault(F_VALUE, "qr_update", "the rank of the change is above the size of the factors");
    }
    // The part of the change below the triangle needs a turn for each
    // column. The part holds too few rows for the rank.
    if m == o && m > n && p > m - n {
        return empty_arr(), empty_arr(), fault(F_VALUE, "qr_update", "the rank of the change is above the rows below the triangle");
    }
    // The copies of the factors hold the changes.
    var qb: f64[>..] = [];
    var i: i64 = 0;
    while i < q.data.len {
        qb.push(q.data[i]);
        i += 1;
    }
    var rb: f64[>..] = [];
    i = 0;
    while i < r.data.len {
        rb.push(r.data[i]);
        i += 1;
    }
    var ub: f64[>..] = [];
    i = 0;
    while i < u.data.len {
        ub.push(u.data[i]);
        i += 1;
    }
    var vb: f64[>..] = [];
    i = 0;
    while i < v.data.len {
        vb.push(v.data[i]);
        i += 1;
    }
    // A single row has no turn. The triangle takes the change.
    if m == 1 {
        var row: i64 = 0;
        while row < n {
            var s: f64 = 0.0;
            var j: i64 = 0;
            while j < p {
                s += ub[j] * vb[row * p + j];
                j += 1;
            }
            rb[row] = rb[row] + qb[0] * s;
            row += 1;
        }
        return mat2(copy(qb), m, o), mat2(copy(rb), o, n), no_fault();
    }
    // The reduced form holds fewer columns than rows.
    if m != o {
        var j2: i64 = 0;
        while j2 < p {
            var uc: f64[>..] = [];
            var k: i64 = 0;
            while k < m {
                uc.push(ub[k * p + j2]);
                k += 1;
            }
            var vc: f64[>..] = [];
            k = 0;
            while k < n {
                vc.push(vb[k * p + j2]);
                k += 1;
            }
            var s: f64[>..] = [];
            k = 0;
            while k < 2 * n {
                s.push(0.0);
                k += 1;
            }
            qu_thin_rank1(qb, m, rb, n, uc, vc, s);
            j2 += 1;
        }
        return mat2(copy(qb), m, o), mat2(copy(rb), o, n), no_fault();
    }
    // The full form needs the change in the coordinates of the
    // orthogonal factor.
    var ut: f64[>..] = [];
    i = 0;
    while i < m * p {
        ut.push(0.0);
        i += 1;
    }
    var row2: i64 = 0;
    while row2 < m {
        var col: i64 = 0;
        while col < p {
            var s: f64 = 0.0;
            var k2: i64 = 0;
            while k2 < m {
                s += qb[k2 * o + row2] * ub[k2 * p + col];
                k2 += 1;
            }
            ut[row2 * p + col] = s;
            col += 1;
        }
        row2 += 1;
    }
    if p == 1 {
        var uc: f64[>..] = [];
        i = 0;
        while i < m {
            uc.push(ut[i]);
            i += 1;
        }
        qu_full_rank1(qb, m, rb, n, uc, vb);
        return mat2(copy(qb), m, o), mat2(copy(rb), o, n), no_fault();
    }
    let f = qu_rank_p(qb, m, rb, n, ut, vb, p);
    if f.kind != F_NONE {
        return empty_arr(), empty_arr(), f;
    }
    return mat2(copy(qb), m, o), mat2(copy(rb), o, n), no_fault();
}

