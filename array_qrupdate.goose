// The update of the QR decomposition, like qr_update of scipy.linalg.
// The method adds a low rank term to a matrix and gives the new
// orthogonal factor and the new upper triangle.
namespace scientific;

import mathshim;
import array_shape;

// The largest side of a matrix for the update. The update holds the
// factors of the matrix and the two vectors of the change.
const QRU_MAX_SIDE: i64 = 4096;

// qu_lartg gives the turn that turns the pair (f, g) into (r, 0). It
// gives the cosine, the sine and the length of the turned pair. The
// method follows LAPACK dlartg.
fn qu_lartg(f: f64, g: f64) -> f64, f64, f64 {
    let safmin = 2.2250738585072014e-308;
    let safmax = 1.7976931348623157e308;
    let rtmin = mathshim::root(safmin);
    let rtmax = mathshim::root(safmax / 2.0);
    let f1 = mathshim::abs_of(f);
    let g1 = mathshim::abs_of(g);
    if g == 0.0 { return 1.0, 0.0, f; }
    if f == 0.0 { return 0.0, mathshim::copysign_of(1.0, g), g1; }
    if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = mathshim::root(f * f + g * g);
        let c = f1 / d;
        let r = mathshim::copysign_of(d, f);
        let s = g / r;
        return c, s, r;
    }
    // The scaling keeps the sum of the squares free of overflow.
    var u = safmin;
    if f1 > u { u = f1; }
    if g1 > u { u = g1; }
    if u > safmax { u = safmax; }
    let fs = f / u;
    let gs = g / u;
    let d = mathshim::root(fs * fs + gs * gs);
    let c = mathshim::abs_of(fs) / d;
    let r = mathshim::copysign_of(d, f);
    let s = gs / r;
    return c, s, r * u;
}

// qu_norm gives the length of a part of a vector.
fn qu_norm(x: f64[:], off: i64, count: i64) -> f64 {
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < count {
        s += x[off + i] * x[off + i];
        i += 1;
    }
    return mathshim::root(s);
}

// qu_dlapy2 gives the length of the pair (x, y) without an overflow.
fn qu_dlapy2(x: f64, y: f64) -> f64 {
    let xa = mathshim::abs_of(x);
    let ya = mathshim::abs_of(y);
    var w = xa;
    if ya > w { w = ya; }
    var z = xa;
    if ya < z { z = ya; }
    if z == 0.0 { return w; }
    let r = z / w;
    return w * mathshim::root(1.0 + r * r);
}

// qu_larfg gives the turn that removes the part of a column below a
// pivot. It gives the new pivot and the factor of the turn. The part
// of the column changes. The method follows LAPACK dlarfg. The count
// holds the number of values in the part of the column.
fn qu_larfg(alpha: f64, x: f64[>..]&, off: i64, inc: i64, count: i64) -> f64, f64 {
    if count < 1 { return alpha, 0.0; }
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < count {
        let t = x[off + i * inc];
        s += t * t;
        i += 1;
    }
    let xnorm = mathshim::root(s);
    if xnorm == 0.0 { return alpha, 0.0; }
    // The sign keeps the value away from a loss of digits.
    let beta = -mathshim::copysign_of(qu_dlapy2(alpha, xnorm), alpha);
    let d = alpha - beta;
    i = 0;
    while i < count {
        x[off + i * inc] = x[off + i * inc] / d;
        i += 1;
    }
    return beta, (beta - alpha) / beta;
}

// qu_rot_rows turns two rows of a matrix over a range of columns.
fn qu_rot_rows(a: f64[>..]&, n: i64, r1: i64, r2: i64, from: i64, count: i64, c: f64, s: f64) {
    var i: i64 = 0;
    while i < count {
        let x = a[r1 * n + from + i];
        let y = a[r2 * n + from + i];
        a[r1 * n + from + i] = c * x + s * y;
        a[r2 * n + from + i] = c * y - s * x;
        i += 1;
    }
}

// qu_rot_cols turns two columns of a matrix over a range of rows.
fn qu_rot_cols(a: f64[>..]&, o: i64, c1: i64, c2: i64, from: i64, count: i64, c: f64, s: f64) {
    var i: i64 = 0;
    while i < count {
        let x = a[(from + i) * o + c1];
        let y = a[(from + i) * o + c2];
        a[(from + i) * o + c1] = c * x + s * y;
        a[(from + i) * o + c2] = c * y - s * x;
        i += 1;
    }
}

// qu_rot_col_vec turns a column of a matrix with a vector. Both
// change.
fn qu_rot_col_vec(a: f64[>..]&, o: i64, c1: i64, v: f64[>..]&, count: i64, c: f64, s: f64) {
    var i: i64 = 0;
    while i < count {
        let x = a[i * o + c1];
        let y = v[i];
        a[i * o + c1] = c * x + s * y;
        v[i] = c * y - s * x;
        i += 1;
    }
}

// qu_reorth takes the part of a vector that lies outside the columns
// of an orthogonal matrix. The work array holds the parts of the
// vector in the first n places and the length of the rest in the
// place n. The vector changes.
fn qu_reorth(q: f64[:], m: i64, n: i64, u: f64[>..]&, s: f64[>..]&) -> i64 {
    let unorm = qu_norm(u, 0, m);
    // A vector of no length has no part outside the columns. The
    // change of the factors is then nothing.
    if unorm == 0.0 {
        var z: i64 = 0;
        while z < m { u[z] = 0.0; z += 1; }
        z = 0;
        while z < n { s[z] = 0.0; z += 1; }
        s[n] = 0.0;
        return 0;
    }
    var i: i64 = 0;
    while i < m {
        u[i] = u[i] / unorm;
        i += 1;
    }
    // The first n places hold the product of the columns with the
    // vector.
    i = 0;
    while i < n {
        var t: f64 = 0.0;
        var k: i64 = 0;
        while k < m {
            t += q[k * n + i] * u[k];
            k += 1;
        }
        s[i] = t;
        i += 1;
    }
    // The vector loses the part in the span of the columns.
    i = 0;
    while i < n {
        let si = s[i];
        var k: i64 = 0;
        while k < m {
            u[k] = u[k] - q[k * n + i] * si;
            k += 1;
        }
        i += 1;
    }
    let wnorm = qu_norm(u, 0, m);
    if wnorm > 0.7071067811865476 {
        i = 0;
        while i < m {
            u[i] = u[i] / wnorm;
            i += 1;
        }
        i = 0;
        while i < n {
            s[i] = s[i] * unorm;
            i += 1;
        }
        s[n] = unorm * wnorm;
        return 0;
    }
    // One more turn makes the part outside the span exact.
    i = 0;
    while i < n {
        var t: f64 = 0.0;
        var k: i64 = 0;
        while k < m {
            t += q[k * n + i] * u[k];
            k += 1;
        }
        s[n + i] = t;
        i += 1;
    }
    i = 0;
    while i < n {
        let si = s[n + i];
        var k: i64 = 0;
        while k < m {
            u[k] = u[k] - q[k * n + i] * si;
            k += 1;
        }
        i += 1;
    }
    let wpnorm = qu_norm(u, 0, m);
    if wpnorm <= wnorm * 0.7071067811865476 {
        i = 0;
        while i < m {
            u[i] = 0.0;
            i += 1;
        }
        i = 0;
        while i < n {
            s[i] = (s[i] + s[n + i]) * unorm;
            i += 1;
        }
        s[n] = 0.0;
        return 1;
    }
    i = 0;
    while i < m {
        u[i] = u[i] / wpnorm;
        i += 1;
    }
    i = 0;
    while i < n {
        s[i] = (s[i] + s[n + i]) * unorm;
        i += 1;
    }
    s[n] = wpnorm * unorm;
    return 0;
}

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
fn qu_p_subdiag(q: f64[>..]&, m: i64, o: i64, r: f64[>..]&, n: i64, k: i64, p: i64) {
    var lim = m - 1;
    if n < lim { lim = n; }
    var j: i64 = k;
    while j < lim {
        var last = p + 1;
        if o - j < last { last = o - j; }
        let rjj = r[j * n + j];
        // The turn removes the part of the column below the pivot.
        var beta: f64 = 0.0;
        var tau: f64 = 0.0;
        beta, tau = qu_larfg(rjj, r, (j + 1) * n + j, n, last - 1);
        // The turn touches the columns that come after.
        if j + 1 < n {
            var c: i64 = j + 1;
            while c < n {
                var w: f64 = 0.0;
                var kk: i64 = 0;
                while kk < last {
                    var vk: f64 = 1.0;
                    if kk > 0 { vk = r[(j + kk) * n + j]; }
                    w += vk * r[(j + kk) * n + c];
                    kk += 1;
                }
                kk = 0;
                while kk < last {
                    var vk: f64 = 1.0;
                    if kk > 0 { vk = r[(j + kk) * n + j]; }
                    r[(j + kk) * n + c] = r[(j + kk) * n + c] - tau * vk * w;
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
                if c > 0 { vk = r[(j + c) * n + j]; }
                w += q[kk * o + (j + c)] * vk;
                c += 1;
            }
            c = 0;
            while c < last {
                var vk: f64 = 1.0;
                if c > 0 { vk = r[(j + c) * n + j]; }
                q[kk * o + (j + c)] = q[kk * o + (j + c)] - tau * w * vk;
                c += 1;
            }
            kk += 1;
        }
        // The part below the pivot is zero.
        var z: i64 = 0;
        while z < last - 1 {
            r[(j + 1 + z) * n + j] = 0.0;
            z += 1;
        }
        r[j * n + j] = beta;
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
    qu_p_subdiag(q, m, m, r, n, 0, p);
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
