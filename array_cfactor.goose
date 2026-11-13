// The factors of a complex matrix: the Householder factor and the helper of the decomposition.
//
// The factors of a complex matrix: the QR decomposition and the singular
// values.
//
// The package holds the factors of a real matrix in array_decomp.goose and in
// array_svd.goose. This file holds the two calls for a complex matrix.
namespace scientific;

import mathshim;

// The side of a complex matrix of a factor. A side above the limit gives a
// fault of kind F_LIMIT.
const C_FACTOR_SIDE: i64 = 512;

// c_cmul gives the product of two complex values. The parts of the values
// come as four numbers. The parts of the product come back as two numbers.
fn c_cmul(ar: f64, ai: f64, br: f64, bi: f64) -> f64, f64 {
    return ar * br - ai * bi, ar * bi + ai * br;
}

// c_factor_check checks an operand of a factor. It gives a fault of kind
// F_SHAPE when the operand holds a count of dimensions that is not two. It
// gives a fault of kind F_LIMIT when a side of the operand is above the
// limit. It gives a fault of kind F_VALUE when a value of the operand is not
// finite.
fn c_factor_check(a, tag: const u8[:]) -> Fault {
    if a.ndim != 2 {
        return fault(F_SHAPE, tag, "the operand holds a count of dimensions that is not two");
    }
    let f = carr_check(a, tag);
    if f.kind != F_NONE { return f; }
    if a.shape[0] > C_FACTOR_SIDE || a.shape[1] > C_FACTOR_SIDE {
        return fault(F_LIMIT, tag, "the side of the operand is above 512");
    }
    if !carr_finite(a) {
        return fault(F_VALUE, tag, "the operand holds a value that is not finite");
    }
    return no_fault();
}

// c_householder_qr gives the orthogonal matrix and the upper triangle of a
// complex matrix. The buffer a holds the parts of the values side by side.
// The turn follows the rule of LAPACK. The value beta of a turn holds a real
// value. The sign of the beta follows the real part of the diagonal element.
// The turn reflects with the conjugate product.
fn c_householder_qr(a: f64[:], m: i64, n: i64, q: f64[>..]&, r: f64[>..]&) {
    // The work buffer holds the matrix.
    var w: f64[>..] = [];
    var i: i64 = 0;
    while i < 2 * m * n {
        w.push(a[i]);
        i += 1;
    }
    var p = m;
    if n < p { p = n; }
    // The vectors of the turns. The vector of the turn k starts at the
    // index bases[k]. It holds the rows k to m-1.
    var vr: f64[>..] = [];
    var vi: f64[>..] = [];
    // The values tau of the turns.
    var tr: f64[>..] = [];
    var ti: f64[>..] = [];
    // The bases of the vectors and the sign that a turn exists.
    var bases: i64[>..] = [];
    var has: i64[>..] = [];
    var k: i64 = 0;
    while k < p {
        let e = 2 * (k * n + k);
        let ar = w[e];
        let ai = w[e + 1];
        // The length of the column and the length of the part of the
        // column below the diagonal.
        var norm: f64 = 0.0;
        var xnorm: f64 = 0.0;
        var row = k;
        while row < m {
            let f = 2 * (row * n + k);
            norm += w[f] * w[f] + w[f + 1] * w[f + 1];
            row += 1;
        }
        row = k + 1;
        while row < m {
            let f = 2 * (row * n + k);
            xnorm += w[f] * w[f] + w[f + 1] * w[f + 1];
            row += 1;
        }
        norm = mathshim::root(norm);
        xnorm = mathshim::root(xnorm);
        bases.push(vr.len);
        // A column that holds no value below the diagonal and a real
        // diagonal keeps its diagonal, as LAPACK keeps it.
        if xnorm > 0.0 || ai != 0.0 {
            has.push(1);
            var beta = -norm;
            if ar < 0.0 { beta = norm; }
            vr.push(1.0);
            vi.push(0.0);
            // The vector of the turn holds the values below the diagonal,
            // divided by the diagonal element minus the beta.
            let dr = ar - beta;
            let di = ai;
            let den = dr * dr + di * di;
            row = k + 1;
            while row < m {
                let f = 2 * (row * n + k);
                let x, y = c_cmul(w[f], w[f + 1], dr, -di);
                vr.push(x / den);
                vi.push(y / den);
                row += 1;
            }
            // The sum of the conjugate of the vector and the column.
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            row = 0;
            while row < m - k {
                let f = 2 * ((k + row) * n + k);
                let x, y = c_cmul(vr[bases[k] + row], -vi[bases[k] + row], w[f], w[f + 1]);
                sr += x;
                si += y;
                row += 1;
            }
            // The value tau holds (alpha - beta) divided by the sum.
            var t = 0.0;
            var t2 = 0.0;
            if sr != 0.0 || si != 0.0 {
                let dd = sr * sr + si * si;
                let x, y = c_cmul(dr, di, sr, -si);
                t = x / dd;
                t2 = y / dd;
            }
            tr.push(t);
            ti.push(t2);
            // The turn touches the columns that come after.
            var col = k;
            while col < n {
                var qr: f64 = 0.0;
                var qi: f64 = 0.0;
                row = 0;
                while row < m - k {
                    let f = 2 * ((k + row) * n + col);
                    let x, y = c_cmul(vr[bases[k] + row], -vi[bases[k] + row], w[f], w[f + 1]);
                    qr += x;
                    qi += y;
                    row += 1;
                }
                row = 0;
                while row < m - k {
                    let x, y = c_cmul(t, t2, qr, qi);
                    let z, zz = c_cmul(vr[bases[k] + row], vi[bases[k] + row], x, y);
                    let f = 2 * ((k + row) * n + col);
                    w[f] = w[f] - z;
                    w[f + 1] = w[f + 1] - zz;
                    row += 1;
                }
                col += 1;
            }
            // The diagonal holds the beta. The rest of the column holds
            // zero.
            w[e] = beta;
            w[e + 1] = 0.0;
            row = k + 1;
            while row < m {
                let f = 2 * (row * n + k);
                w[f] = 0.0;
                w[f + 1] = 0.0;
                row += 1;
            }
        } else {
            has.push(0);
            tr.push(0.0);
            ti.push(0.0);
        }
        k += 1;
    }
    // The upper triangle holds the first p rows of the work buffer.
    i = 0;
    while i < 2 * p * n {
        r.push(w[i]);
        i += 1;
    }
    // The orthogonal matrix starts as the unit matrix.
    var om: f64[>..] = [];
    i = 0;
    while i < 2 * m * m {
        om.push(0.0);
        i += 1;
    }
    i = 0;
    while i < m {
        om[2 * (i * m + i)] = 1.0;
        i += 1;
    }
    // The turns touch it in the reverse order. A turn of a complex matrix
    // is not a reflection, so the inverse of the turn comes with the
    // conjugate of the value tau.
    var t = p - 1;
    while t >= 0 {
        if has[t] == 1 {
            var vlen = m - t;
            var col: i64 = 0;
            while col < m {
                var sr: f64 = 0.0;
                var si: f64 = 0.0;
                var row: i64 = 0;
                while row < vlen {
                    let f = 2 * ((t + row) * m + col);
                    let x, y = c_cmul(vr[bases[t] + row], -vi[bases[t] + row], om[f], om[f + 1]);
                    sr += x;
                    si += y;
                    row += 1;
                }
                row = 0;
                while row < vlen {
                    let x, y = c_cmul(tr[t], -ti[t], sr, si);
                    let z, zz = c_cmul(vr[bases[t] + row], vi[bases[t] + row], x, y);
                    let f = 2 * ((t + row) * m + col);
                    om[f] = om[f] - z;
                    om[f + 1] = om[f + 1] - zz;
                    row += 1;
                }
                col += 1;
            }
        }
        t -= 1;
    }
    // The orthogonal matrix holds the first p columns.
    i = 0;
    while i < m * p {
        let at = 2 * ((i / p) * m + i % p);
        q.push(om[at]);
        q.push(om[at + 1]);
        i += 1;
    }
    return;
}

// c_complete_basis gives the columns that a set of columns lacks. The
// matrix holds the columns in its rows. The element of the row r and the
// column c lies at the index of the row times the count plus the column. The
// new columns come from the unit vectors that the columns do not hold.
fn c_complete_basis(cols: f64[>..]&, rows: i64, count: i64, want: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < 2 * rows * want {
        out.push(0.0);
        i += 1;
    }
    // The columns that hold.
    i = 0;
    while i < count {
        var r: i64 = 0;
        while r < rows {
            out[2 * (r * want + i)] = cols[2 * (r * rows + i)];
            out[2 * (r * want + i) + 1] = cols[2 * (r * rows + i) + 1];
            r += 1;
        }
        i += 1;
    }
    // The new columns.
    var c = count;
    while c < want {
        var r: i64 = 0;
        var found = false;
        while r < rows && !found {
            // The unit vector of the row r.
            var v: f64[>..] = [];
            var t: i64 = 0;
            while t < rows {
                v.push(0.0);
                v.push(0.0);
                t += 1;
            }
            v[2 * r] = 1.0;
            // The vector loses the part that the columns hold.
            var j: i64 = 0;
            while j < c {
                var sr: f64 = 0.0;
                var si: f64 = 0.0;
                t = 0;
                while t < rows {
                    let x, y = c_cmul(out[2 * (t * want + j)], -out[2 * (t * want + j) + 1], v[2 * t], v[2 * t + 1]);
                    sr += x;
                    si += y;
                    t += 1;
                }
                t = 0;
                while t < rows {
                    let x, y = c_cmul(sr, si, out[2 * (t * want + j)], out[2 * (t * want + j) + 1]);
                    v[2 * t] = v[2 * t] - x;
                    v[2 * t + 1] = v[2 * t + 1] - y;
                    t += 1;
                }
                j += 1;
            }
            // The length of the vector gives the scale.
            var len: f64 = 0.0;
            t = 0;
            while t < rows {
                len += v[2 * t] * v[2 * t] + v[2 * t + 1] * v[2 * t + 1];
                t += 1;
            }
            len = mathshim::root(len);
            if len > 1.0e-8 {
                t = 0;
                while t < rows {
                    out[2 * (t * want + c)] = v[2 * t] / len;
                    out[2 * (t * want + c) + 1] = v[2 * t + 1] / len;
                    t += 1;
                }
                found = true;
            } else {
                r += 1;
            }
        }
        c += 1;
    }
    return out;
}

// c_scale_of gives the factor that brings the largest part of a complex
// matrix to one when the parts are too large or too small to square. It
// gives one for a matrix of zeros and for a matrix in the safe range.
fn c_scale_of(a: f64[:]) -> f64 {
    var big: f64 = 0.0;
    var i: i64 = 0;
    while i < a.len {
        var x = a[i];
        if x < 0.0 { x = -x; }
        if x > big { big = x; }
        i += 1;
    }
    if big > 1.0e-100 && big < 1.0e100 { return 1.0; }
    if big == 0.0 { return 1.0; }
    return big;
}

// c_scaled gives a copy of the matrix divided by the factor.
fn c_scaled(a: f64[:], scale: f64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < a.len {
        out.push(a[i] / scale);
        i += 1;
    }
    return out;
}

// c_conj_transpose gives the conjugate transpose of a complex matrix of
// rows rows and cols columns.
fn c_conj_transpose(a: f64[:], rows: i64, cols: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < cols {
        var j: i64 = 0;
        while j < rows {
            out.push(a[2 * (j * cols + i)]);
            out.push(-a[2 * (j * cols + i) + 1]);
            j += 1;
        }
        i += 1;
    }
    return out;
}

// c_wide_svd gives the singular values and the vectors of a matrix with more
// columns than rows. The method works on the conjugate transpose, which has
// more rows than columns. The left vectors of the result then come from the
// right vectors of the transpose. The phase rule applies to them again.
fn c_wide_svd(a: f64[:], m: i64, n: i64, vals: f64[>..]&, left: f64[>..]&, vecs: f64[>..]&) {
    let b = c_conj_transpose(a, m, n);
    var bl: f64[>..] = [];
    var bv: f64[>..] = [];
    c_jacobi_svd(b, n, m, vals, bl, bv);
    // The left vectors are the conjugate transpose of the right vectors.
    var u = c_conj_transpose(bv, m, m);
    // The right vectors are the conjugate transpose of the left vectors.
    var v = c_conj_transpose(bl, n, n);
    var i: i64 = 0;
    // The phase rule. The column of the left vectors takes the phase that
    // makes its largest component real. The row of the right vectors takes
    // the conjugate of that phase.
    var c: i64 = 0;
    while c < m {
        var best: f64 = -1.0;
        var br: f64 = 0.0;
        var bi: f64 = 0.0;
        var r: i64 = 0;
        while r < m {
            let f = 2 * (r * m + c);
            let size = mathshim::root(u[f] * u[f] + u[f + 1] * u[f + 1]);
            if size > best {
                best = size;
                br = u[f];
                bi = u[f + 1];
            }
            r += 1;
        }
        if best > 0.0 {
            let pr = br / best;
            let pi = -bi / best;
            r = 0;
            while r < m {
                let f = 2 * (r * m + c);
                let x, y = c_cmul(pr, pi, u[f], u[f + 1]);
                u[f] = x;
                u[f + 1] = y;
                r += 1;
            }
            // The row c of the right vectors takes the conjugate phase.
            r = 0;
            while r < n {
                let f = 2 * (c * n + r);
                let x, y = c_cmul(pr, -pi, v[f], v[f + 1]);
                v[f] = x;
                v[f + 1] = y;
                r += 1;
            }
        }
        c += 1;
    }
    i = 0;
    while i < 2 * m * m {
        left.push(u[i]);
        i += 1;
    }
    i = 0;
    while i < 2 * n * n {
        vecs.push(v[i]);
        i += 1;
    }
    return;
}

