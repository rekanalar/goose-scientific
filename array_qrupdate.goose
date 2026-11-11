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

// qu_finite checks that every value of an array is a finite number. A
// value that is not finite gives a fault.
fn qu_finite(a, tag: const u8[:]) -> Fault {
    var i: i64 = 0;
    while i < a.data.len {
        let v = a.data[i];
        if !(v - v == 0.0) {
            return fault(F_VALUE, tag, "an operand holds a value that is not finite");
        }
        i += 1;
    }
    return no_fault();
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

// qi_reorthx takes the part of the basis vector j that lies outside
// the columns of an orthogonal matrix. The work array holds the parts
// in the first n places and the length of the rest in the place n.
// The vector u holds m places and changes. The call gives true when
// the part outside the span keeps its length.
fn qi_reorthx(q: f64[>..]&, m: i64, n: i64, stride: i64, j: i64, u: f64[>..]&, s: f64[>..]&) -> bool {
    var z: i64 = 0;
    while z < m { u[z] = 0.0; z += 1; }
    u[j] = 1.0;
    var i: i64 = 0;
    while i < n {
        s[i] = q[j * stride + i];
        i += 1;
    }
    i = 0;
    while i < n {
        let si = s[i];
        var k: i64 = 0;
        while k < m {
            u[k] = u[k] - q[k * stride + i] * si;
            k += 1;
        }
        i += 1;
    }
    let wnorm = qu_norm(u, 0, m);
    if wnorm > 0.7071067811865476 {
        i = 0;
        while i < m { u[i] = u[i] / wnorm; i += 1; }
        s[n] = wnorm;
        return true;
    }
    i = 0;
    while i < n {
        var t: f64 = 0.0;
        var k: i64 = 0;
        while k < m {
            t += q[k * stride + i] * u[k];
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
            u[k] = u[k] - q[k * stride + i] * si;
            k += 1;
        }
        i += 1;
    }
    let wpnorm = qu_norm(u, 0, m);
    if wpnorm < wnorm * 0.7071067811865476 {
        i = 0;
        while i < m { u[i] = 0.0; i += 1; }
        i = 0;
        while i < n {
            s[i] = s[i] + s[n + i];
            i += 1;
        }
        s[n] = 0.0;
        return false;
    }
    i = 0;
    while i < m { u[i] = u[i] / wpnorm; i += 1; }
    i = 0;
    while i < n {
        s[i] = s[i] + s[n + i];
        i += 1;
    }
    s[n] = wpnorm;
    return true;
}

// qi_swap_rows swaps two rows of a flat matrix with the stride cols.
fn qi_swap_rows(a: f64[>..]&, cols: i64, r1: i64, r2: i64) {
    var i: i64 = 0;
    while i < cols {
        let t = a[r1 * cols + i];
        a[r1 * cols + i] = a[r2 * cols + i];
        a[r2 * cols + i] = t;
        i += 1;
    }
}

// qi_thin_row_insert puts the row u before the place k of a reduced
// decomposition. The matrix q holds m rows and n plus one columns,
// and the matrix r holds n rows and n columns. Both change.
fn qi_thin_row_insert(q: f64[>..]&, m: i64, n: i64, r: f64[>..]&, u: f64[>..]&, k: i64) {
    let ncol = n + 1;
    var j: i64 = 0;
    while j < n {
        var c: f64 = 0.0;
        var s: f64 = 0.0;
        var rr: f64 = 0.0;
        c, s, rr = qu_lartg(r[j * n + j], u[j]);
        r[j * n + j] = rr;
        u[j] = 0.0;
        if j + 1 < n {
            var cc: i64 = j + 1;
            while cc < n {
                let x = r[j * n + cc];
                let y = u[cc];
                r[j * n + cc] = c * x + s * y;
                u[cc] = c * y - s * x;
                cc += 1;
            }
        }
        var i: i64 = 0;
        while i < m {
            let x = q[i * ncol + j];
            let y = q[i * ncol + n];
            q[i * ncol + j] = c * x + s * y;
            q[i * ncol + n] = c * y - s * x;
            i += 1;
        }
        j += 1;
    }
    j = m - 1;
    while j > k {
        qi_swap_rows(q, ncol, j, j - 1);
        j -= 1;
    }
}

// qi_thin_block_row_insert puts the p rows of u before the place k of
// a reduced decomposition. The matrix q holds m rows and n plus p
// columns, the matrix r holds n rows and n columns, and u holds p
// rows and n columns. All change.
fn qi_thin_block_row_insert(q: f64[>..]&, m: i64, n: i64, r: f64[>..]&, u: f64[>..]&, k: i64, p: i64) {
    let ncol = n + p;
    var j: i64 = 0;
    while j < n {
        let rjj = r[j * n + j];
        var x: f64[>..] = [];
        var i2: i64 = 0;
        while i2 < p {
            x.push(u[i2 * n + j]);
            i2 += 1;
        }
        var beta: f64 = 0.0;
        var tau: f64 = 0.0;
        beta, tau = qu_larfg(rjj, x, 0, 1, p);
        i2 = 0;
        while i2 < p {
            u[i2 * n + j] = x[i2];
            i2 += 1;
        }
        // The vector of the turn holds one and the parts.
        var v: f64[>..] = [];
        v.push(1.0);
        i2 = 0;
        while i2 < p { v.push(x[i2]); i2 += 1; }
        if j + 1 < n {
            var cc: i64 = j + 1;
            while cc < n {
                var w: f64 = v[0] * r[j * n + cc];
                var kk: i64 = 1;
                while kk < p + 1 {
                    w += v[kk] * u[(kk - 1) * n + cc];
                    kk += 1;
                }
                r[j * n + cc] = r[j * n + cc] - tau * v[0] * w;
                kk = 1;
                while kk < p + 1 {
                    u[(kk - 1) * n + cc] = u[(kk - 1) * n + cc] - tau * v[kk] * w;
                    kk += 1;
                }
                cc += 1;
            }
        }
        r[j * n + j] = beta;
        var i: i64 = 0;
        while i < m {
            var w: f64 = q[i * ncol + j];
            var kk: i64 = 1;
            while kk < p + 1 {
                w += v[kk] * q[i * ncol + n + kk - 1];
                kk += 1;
            }
            q[i * ncol + j] = q[i * ncol + j] - tau * v[0] * w;
            kk = 1;
            while kk < p + 1 {
                q[i * ncol + n + kk - 1] = q[i * ncol + n + kk - 1] - tau * v[kk] * w;
                kk += 1;
            }
            i += 1;
        }
        j += 1;
    }
    if k != m - p {
        var cc: i64 = 0;
        while cc < n {
            var vals: f64[>..] = [];
            var i: i64 = 0;
            while i < m {
                vals.push(q[i * ncol + cc]);
                i += 1;
            }
            i = k;
            var t: i64 = 0;
            while i < m {
                var src: i64 = k + t - p;
                if t < p { src = m - p + t; }
                q[i * ncol + cc] = vals[src];
                t += 1;
                i += 1;
            }
            cc += 1;
        }
    }
}

// qi_row_insert_full1 puts one row before the place k of a full
// decomposition. The matrix q holds m rows and m columns, and the
// matrix r holds m rows and n columns. Both change.
fn qi_row_insert_full1(q: f64[>..]&, m: i64, n: i64, r: f64[>..]&, k: i64) {
    var lim = m - 1;
    if n < lim { lim = n; }
    var j: i64 = 0;
    while j < lim {
        var c: f64 = 0.0;
        var s: f64 = 0.0;
        var rr: f64 = 0.0;
        c, s, rr = qu_lartg(r[j * n + j], r[(m - 1) * n + j]);
        r[j * n + j] = rr;
        r[(m - 1) * n + j] = 0.0;
        if j + 1 < n {
            var cc: i64 = j + 1;
            while cc < n {
                let x = r[j * n + cc];
                let y = r[(m - 1) * n + cc];
                r[j * n + cc] = c * x + s * y;
                r[(m - 1) * n + cc] = c * y - s * x;
                cc += 1;
            }
        }
        var i: i64 = 0;
        while i < m {
            let x = q[i * m + j];
            let y = q[i * m + m - 1];
            q[i * m + j] = c * x + s * y;
            q[i * m + m - 1] = c * y - s * x;
            i += 1;
        }
        j += 1;
    }
    j = m - 1;
    while j > k {
        qi_swap_rows(q, m, j, j - 1);
        j -= 1;
    }
}

// qi_block_row_insert puts the p rows of u before the place k of a
// full decomposition. The matrix q holds m rows and m columns, and
// the matrix r holds m rows and n columns. Both change.
fn qi_block_row_insert(q: f64[>..]&, m: i64, n: i64, r: f64[>..]&, k: i64, p: i64) {
    var lim = m;
    if n < lim { lim = n; }
    var j: i64 = 0;
    while j < lim {
        let rjj = r[j * n + j];
        let hlen = m - j;
        var x: f64[>..] = [];
        var i2: i64 = 0;
        while i2 < hlen - 1 {
            x.push(r[(j + 1 + i2) * n + j]);
            i2 += 1;
        }
        var beta: f64 = 0.0;
        var tau: f64 = 0.0;
        beta, tau = qu_larfg(rjj, x, 0, 1, hlen - 1);
        i2 = 0;
        while i2 < hlen - 1 {
            r[(j + 1 + i2) * n + j] = x[i2];
            i2 += 1;
        }
        var v: f64[>..] = [];
        v.push(1.0);
        i2 = 0;
        while i2 < hlen - 1 { v.push(x[i2]); i2 += 1; }
        r[j * n + j] = 1.0;
        if j + 1 < n {
            var cc: i64 = j + 1;
            while cc < n {
                var w: f64 = 0.0;
                var kk: i64 = 0;
                while kk < hlen {
                    w += v[kk] * r[(j + kk) * n + cc];
                    kk += 1;
                }
                kk = 0;
                while kk < hlen {
                    r[(j + kk) * n + cc] = r[(j + kk) * n + cc] - tau * v[kk] * w;
                    kk += 1;
                }
                cc += 1;
            }
        }
        var i: i64 = 0;
        while i < m {
            var w: f64 = 0.0;
            var kk: i64 = 0;
            while kk < hlen {
                w += v[kk] * q[i * m + j + kk];
                kk += 1;
            }
            kk = 0;
            while kk < hlen {
                q[i * m + j + kk] = q[i * m + j + kk] - tau * v[kk] * w;
                kk += 1;
            }
            i += 1;
        }
        i2 = 0;
        while i2 < hlen - 1 {
            r[(j + 1 + i2) * n + j] = 0.0;
            i2 += 1;
        }
        r[j * n + j] = beta;
        j += 1;
    }
    if k != m - p {
        var cc: i64 = 0;
        while cc < m {
            var vals: f64[>..] = [];
            var i: i64 = 0;
            while i < m {
                vals.push(q[i * m + cc]);
                i += 1;
            }
            i = k;
            var t: i64 = 0;
            while i < m {
                var src: i64 = k + t - p;
                if t < p { src = m - p + t; }
                q[i * m + cc] = vals[src];
                t += 1;
                i += 1;
            }
            cc += 1;
        }
    }
}

// qi_thin_row_delete removes the first p_eco of p rows of a reduced
// decomposition. The matrix q holds m rows and n columns, and the
// matrix r holds n rows and n columns. Both change. The call gives
// false when the reorthogonalization fails.
fn qi_thin_row_delete(q: f64[>..]&, m: i64, n: i64, r: f64[>..]&, k: i64, p_eco: i64, p_full: i64) -> bool {
    var s: f64[>..] = [];
    var i: i64 = 0;
    while i < 3 * n { s.push(0.0); i += 1; }
    var u: f64[>..] = [];
    i = 0;
    while i < m { u.push(0.0); i += 1; }
    var mm = m;
    i = 0;
    while i < p_eco {
        var z: i64 = 0;
        while z < mm { u[z] = 0.0; z += 1; }
        if k != mm - 1 {
            var j: i64 = k;
            while j < mm - 1 {
                qi_swap_rows(q, n, j, j + 1);
                j += 1;
            }
        }
        var ok = qi_reorthx(q, mm, n, n, mm - 1, u, s);
        if !ok {
            // The basis vector lies in the span. The row of the
            // smallest length takes its place.
            var minnorm = qu_norm(q, 0, n);
            var argmin: i64 = 0;
            var j: i64 = 1;
            while j < mm {
                let rn = qu_norm(q, j * n, n);
                if rn < minnorm {
                    minnorm = rn;
                    argmin = j;
                }
                j += 1;
            }
            z = 0;
            while z < mm { u[z] = 0.0; z += 1; }
            ok = qi_reorthx(q, mm, n, n, argmin, u, s);
            if !ok {
                return false;
            }
            s[n] = 0.0;
        }
        var t: f64[>..] = [];
        z = 0;
        while z < n { t.push(0.0); z += 1; }
        var j: i64 = n - 1;
        while j >= 0 {
            var c: f64 = 0.0;
            var sn: f64 = 0.0;
            var rr: f64 = 0.0;
            c, sn, rr = qu_lartg(s[n], s[j]);
            s[n] = rr;
            s[j] = 0.0;
            var cc: i64 = j;
            while cc < n {
                let x = t[cc];
                let y = r[j * n + cc];
                t[cc] = c * x + sn * y;
                r[j * n + cc] = c * y - sn * x;
                cc += 1;
            }
            var ii: i64 = 0;
            while ii < mm - 1 {
                let x = u[ii];
                let y = q[ii * n + j];
                u[ii] = c * x + sn * y;
                q[ii * n + j] = c * y - sn * x;
                ii += 1;
            }
            j -= 1;
        }
        mm -= 1;
        i += 1;
    }
    if p_full > 0 {
        return qi_block_row_delete(q, mm, n, r, k, p_full);
    }
    return true;
}

// qi_block_row_delete removes p rows of a full decomposition. The
// matrix q holds m rows and m columns, and the matrix r holds m rows
// and n columns. Both change.
fn qi_block_row_delete(q: f64[>..]&, m: i64, n: i64, r: f64[>..]&, k: i64, p: i64) -> bool {
    if k != 0 {
        var j: i64 = k;
        while j > 0 {
            qi_swap_rows(q, m, j + p - 1, j - 1);
            j -= 1;
        }
    }
    var w: f64[>..] = [];
    var i: i64 = 0;
    while i < p * m { w.push(0.0); i += 1; }
    i = 0;
    while i < p {
        var cc: i64 = 0;
        while cc < m {
            w[i * m + cc] = q[i * m + cc];
            cc += 1;
        }
        i += 1;
    }
    i = 0;
    while i < p {
        var j: i64 = m - 2;
        while j >= i {
            var c: f64 = 0.0;
            var s: f64 = 0.0;
            var rr: f64 = 0.0;
            c, s, rr = qu_lartg(w[i * m + j], w[i * m + j + 1]);
            w[i * m + j] = rr;
            w[i * m + j + 1] = 0.0;
            if i + 1 < p {
                var ii: i64 = i + 1;
                while ii < p {
                    let x = w[ii * m + j];
                    let y = w[ii * m + j + 1];
                    w[ii * m + j] = c * x + s * y;
                    w[ii * m + j + 1] = c * y - s * x;
                    ii += 1;
                }
            }
            if j - i < n {
                var cc: i64 = j - i;
                while cc < n {
                    let x = r[j * n + cc];
                    let y = r[(j + 1) * n + cc];
                    r[j * n + cc] = c * x + s * y;
                    r[(j + 1) * n + cc] = c * y - s * x;
                    cc += 1;
                }
            }
            var ii: i64 = p;
            while ii < m {
                let x = q[ii * m + j];
                let y = q[ii * m + j + 1];
                q[ii * m + j] = c * x + s * y;
                q[ii * m + j + 1] = c * y - s * x;
                ii += 1;
            }
            j -= 1;
        }
        i += 1;
    }
    return true;
}

// qi_hessenberg turns the upper Hessenberg part of a matrix that
// starts at the row k into a triangle. The matrix q holds m rows and
// o columns, and the matrix r holds o rows and n columns. Both
// change.
fn qi_hessenberg(q: f64[>..]&, m: i64, o: i64, r: f64[>..]&, rn: i64, nc: i64, k: i64) {
    var lim = m - 1;
    if nc < lim { lim = nc; }
    var j: i64 = k;
    while j < lim {
        var c: f64 = 0.0;
        var s: f64 = 0.0;
        var rr: f64 = 0.0;
        c, s, rr = qu_lartg(r[j * rn + j], r[(j + 1) * rn + j]);
        r[j * rn + j] = rr;
        r[(j + 1) * rn + j] = 0.0;
        if j + 1 < m {
            var cc: i64 = j + 1;
            while cc < nc {
                let x = r[j * rn + cc];
                let y = r[(j + 1) * rn + cc];
                r[j * rn + cc] = c * x + s * y;
                r[(j + 1) * rn + cc] = c * y - s * x;
                cc += 1;
            }
        }
        var i: i64 = 0;
        while i < m {
            let x = q[i * o + j];
            let y = q[i * o + j + 1];
            q[i * o + j] = c * x + s * y;
            q[i * o + j + 1] = c * y - s * x;
            i += 1;
        }
        j += 1;
    }
}

// qi_col_delete removes one column at the place k. The matrix q holds
// m rows and o columns, and the matrix r holds o rows and n columns.
// Both change.
fn qi_col_delete(q: f64[>..]&, m: i64, o: i64, r: f64[>..]&, n: i64, k: i64) {
    var lim = o;
    if n < lim { lim = n; }
    var j: i64 = k;
    while j < n - 1 {
        var i: i64 = 0;
        while i < lim {
            r[i * n + j] = r[i * n + j + 1];
            i += 1;
        }
        j += 1;
    }
    qi_hessenberg(q, m, o, r, n, n - 1, k);
}

// qi_thin_col_insert puts the first p_eco columns of u at the place k
// of a reduced decomposition. The matrix q holds m rows and the
// stride qncol, and the matrix r holds the stride rncol. The matrix u
// holds m rows and the stride ustride. The call gives two when a
// column of u lies in the span of the columns of q.
fn qi_thin_col_insert(q: f64[>..]&, m: i64, qncol: i64, r: f64[>..]&, rncol: i64, u: f64[>..]&, ustride: i64, n: i64, k: i64, p_eco: i64, p_full: i64) -> i64 {
    var s: f64[>..] = [];
    var i: i64 = 0;
    while i < 2 * (n + p_eco) + 2 { s.push(0.0); i += 1; }
    var j: i64 = 0;
    while j < p_eco {
        // The part of the column outside the columns of q. This is
        // the reorthogonalization with the stride of q.
        var uc: f64[>..] = [];
        i = 0;
        while i < m {
            uc.push(u[i * ustride + j]);
            i += 1;
        }
        let unorm = qu_norm(uc, 0, m);
        i = 0;
        while i < m { uc[i] = uc[i] / unorm; i += 1; }
        var ncols = n + j;
        i = 0;
        while i < ncols {
            var t: f64 = 0.0;
            var kk: i64 = 0;
            while kk < m {
                t += q[kk * qncol + i] * uc[kk];
                kk += 1;
            }
            s[i] = t;
            i += 1;
        }
        i = 0;
        while i < ncols {
            let si = s[i];
            var kk: i64 = 0;
            while kk < m {
                uc[kk] = uc[kk] - q[kk * qncol + i] * si;
                kk += 1;
            }
            i += 1;
        }
        let wnorm = qu_norm(uc, 0, m);
        if wnorm > 0.7071067811865476 {
            i = 0;
            while i < m { uc[i] = uc[i] / wnorm; i += 1; }
            i = 0;
            while i < ncols { s[i] = s[i] * unorm; i += 1; }
            s[ncols] = unorm * wnorm;
        } else {
            i = 0;
            while i < ncols {
                var t: f64 = 0.0;
                var kk: i64 = 0;
                while kk < m {
                    t += q[kk * qncol + i] * uc[kk];
                    kk += 1;
                }
                s[ncols + i] = t;
                i += 1;
            }
            i = 0;
            while i < ncols {
                let si = s[ncols + i];
                var kk: i64 = 0;
                while kk < m {
                    uc[kk] = uc[kk] - q[kk * qncol + i] * si;
                    kk += 1;
                }
                i += 1;
            }
            let wpnorm = qu_norm(uc, 0, m);
            if wpnorm <= wnorm * 0.7071067811865476 {
                return 2;
            }
            i = 0;
            while i < m { uc[i] = uc[i] / wpnorm; i += 1; }
            i = 0;
            while i < ncols { s[i] = (s[i] + s[ncols + i]) * unorm; i += 1; }
            s[ncols] = wpnorm * unorm;
        }
        // The column joins the orthogonal factor.
        i = 0;
        while i < m {
            q[i * qncol + n + j] = uc[i];
            i += 1;
        }
        // The parts join the triangle.
        i = 0;
        while i < ncols + 1 {
            r[i * rncol + k + j] = s[i];
            i += 1;
        }
        // The turns make the triangle upper again.
        i = n - 1;
        while i >= k {
            var c: f64 = 0.0;
            var sn: f64 = 0.0;
            var rr: f64 = 0.0;
            c, sn, rr = qu_lartg(r[(i + j) * rncol + k + j], r[(i + j + 1) * rncol + k + j]);
            r[(i + j) * rncol + k + j] = rr;
            r[(i + j + 1) * rncol + k + j] = 0.0;
            let start = i + p_eco + p_full;
            var cc: i64 = start;
            while cc < n + p_eco + p_full {
                let x = r[(i + j) * rncol + cc];
                let y = r[(i + j + 1) * rncol + cc];
                r[(i + j) * rncol + cc] = c * x + sn * y;
                r[(i + j + 1) * rncol + cc] = c * y - sn * x;
                cc += 1;
            }
            var ii: i64 = 0;
            while ii < m {
                let x = q[ii * qncol + i + j];
                let y = q[ii * qncol + i + j + 1];
                q[ii * qncol + i + j] = c * x + sn * y;
                q[ii * qncol + i + j + 1] = c * y - sn * x;
                ii += 1;
            }
            i -= 1;
        }
        j += 1;
    }
    return 0;
}

// qi_col_insert_full puts one column at the place k of a full
// decomposition. The matrix q holds m rows and m columns, and the
// matrix r holds m rows and n columns. Both change.
fn qi_col_insert_full(q: f64[>..]&, m: i64, n: i64, r: f64[>..]&, k: i64) {
    var j: i64 = m - 2;
    while j >= k {
        var c: f64 = 0.0;
        var s: f64 = 0.0;
        var rr: f64 = 0.0;
        c, s, rr = qu_lartg(r[j * n + k], r[(j + 1) * n + k]);
        r[j * n + k] = rr;
        r[(j + 1) * n + k] = 0.0;
        if j + 1 < n {
            var cc: i64 = j + 1;
            while cc < n {
                let x = r[j * n + cc];
                let y = r[(j + 1) * n + cc];
                r[j * n + cc] = c * x + s * y;
                r[(j + 1) * n + cc] = c * y - s * x;
                cc += 1;
            }
        }
        var i: i64 = 0;
        while i < m {
            let x = q[i * m + j];
            let y = q[i * m + j + 1];
            q[i * m + j] = c * x + s * y;
            q[i * m + j + 1] = c * y - s * x;
            i += 1;
        }
        j -= 1;
    }
}

// qi_block_col_insert puts p columns at the place k of a full
// decomposition. The matrix q holds m rows and m columns, and the
// matrix r holds m rows and n columns. Both change.
fn qi_block_col_insert(q: f64[>..]&, m: i64, n: i64, r: f64[>..]&, k: i64, p: i64) {
    let n_orig = n - p;
    if m >= n {
        // The part of the new columns below the triangle takes a turn
        // of its own.
        var sub: f64[>..] = [];
        var i: i64 = 0;
        while i < m * p { sub.push(0.0); i += 1; }
        i = n_orig;
        while i < m {
            var jj: i64 = 0;
            while jj < p {
                sub[i * p + jj] = r[i * n + k + jj];
                jj += 1;
            }
            i += 1;
        }
        var tau: f64[>..] = [];
        i = 0;
        while i < p { tau.push(0.0); i += 1; }
        qu_geqrf(sub, m, p, n_orig, tau);
        i = n_orig;
        while i < m {
            var jj: i64 = 0;
            while jj < p {
                r[i * n + k + jj] = sub[i * p + jj];
                jj += 1;
            }
            i += 1;
        }
        qu_ormqr_right(q, m, m, sub, tau, p, n_orig);
        var jj: i64 = 0;
        while jj < p {
            let start = n_orig + 1 + jj;
            i = start;
            while i < m {
                r[i * n + k + jj] = 0.0;
                i += 1;
            }
            jj += 1;
        }
        i = 0;
        while i < p {
            var j: i64 = n_orig + i - 1;
            while j >= k + i {
                var c: f64 = 0.0;
                var s: f64 = 0.0;
                var rr: f64 = 0.0;
                c, s, rr = qu_lartg(r[j * n + k + i], r[(j + 1) * n + k + i]);
                r[j * n + k + i] = rr;
                r[(j + 1) * n + k + i] = 0.0;
                if j + 1 < n {
                    var cc: i64 = k + i + 1;
                    while cc < n {
                        let x = r[j * n + cc];
                        let y = r[(j + 1) * n + cc];
                        r[j * n + cc] = c * x + s * y;
                        r[(j + 1) * n + cc] = c * y - s * x;
                        cc += 1;
                    }
                }
                var ii: i64 = 0;
                while ii < m {
                    let x = q[ii * m + j];
                    let y = q[ii * m + j + 1];
                    q[ii * m + j] = c * x + s * y;
                    q[ii * m + j + 1] = c * y - s * x;
                    ii += 1;
                }
                j -= 1;
            }
            i += 1;
        }
    } else {
        var i: i64 = 0;
        while i < p {
            var j: i64 = m - 2;
            while j >= k + i {
                var c: f64 = 0.0;
                var s: f64 = 0.0;
                var rr: f64 = 0.0;
                c, s, rr = qu_lartg(r[j * n + k + i], r[(j + 1) * n + k + i]);
                r[j * n + k + i] = rr;
                r[(j + 1) * n + k + i] = 0.0;
                if j + 1 < n {
                    var cc: i64 = k + i + 1;
                    while cc < n {
                        let x = r[j * n + cc];
                        let y = r[(j + 1) * n + cc];
                        r[j * n + cc] = c * x + s * y;
                        r[(j + 1) * n + cc] = c * y - s * x;
                        cc += 1;
                    }
                }
                var ii: i64 = 0;
                while ii < m {
                    let x = q[ii * m + j];
                    let y = q[ii * m + j + 1];
                    q[ii * m + j] = c * x + s * y;
                    q[ii * m + j + 1] = c * y - s * x;
                    ii += 1;
                }
                j -= 1;
            }
            i += 1;
        }
    }
}

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
