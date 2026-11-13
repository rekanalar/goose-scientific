// The Cholesky and the QR decompositions of a real matrix.
//
// The Cholesky and the QR decompositions.
//
// A matrix is an array with two dimensions. The functions of this file
// work on the elements of the matrix in the row-major order.
namespace scientific;

import mathshim;

// cholesky gives the lower triangle of the Cholesky factor of a
// matrix. The matrix must be symmetric and hold no negative value.
fn cholesky(a) -> Arr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return empty_arr(), fault(F_SHAPE, "cholesky", "the matrix is not square");
    }
    let n = a.shape[0];
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { d.push(0.0); i += 1; }
    var r: i64 = 0;
    while r < n {
        var c: i64 = 0;
        while c < r {
            var t: f64 = a.data[r * n + c];
            var k: i64 = 0;
            while k < c {
                t -= d[r * n + k] * d[c * n + k];
                k += 1;
            }
            d[r * n + c] = t / d[c * n + c];
            c += 1;
        }
        var t: f64 = a.data[r * n + r];
        var k2: i64 = 0;
        while k2 < r {
            t -= d[r * n + k2] * d[r * n + k2];
            k2 += 1;
        }
        if t <= 0.0 {
            return empty_arr(), fault(F_VALUE, "cholesky", "the matrix holds no Cholesky factor");
        }
        d[r * n + r] = mathshim::root(t);
        r += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    s.push(n);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// householder_qr splits a matrix into an orthogonal matrix and an
// upper triangle. The buffers hold the elements in the row-major
// order. The method is the Householder method.
fn householder_qr(a: f64[:], m: i64, n: i64, q: f64[>..]&, r: f64[>..]&) -> Fault {
    // The matrix of the work and the vectors of the turns.
    var w: f64[>..] = [];
    var i: i64 = 0;
    while i < m * n { w.push(a[i]); i += 1; }
    var hv: f64[>..] = [];
    i = 0;
    while i < m * m { hv.push(0.0); i += 1; }
    // The count of the turns.
    var p = m;
    if n < p { p = n; }
    var k: i64 = 0;
    while k < p {
        // The length of the part of the column that is below the turn.
        var norm_w: f64 = 0.0;
        var rr: i64 = k;
        while rr < m {
            norm_w += w[rr * n + k] * w[rr * n + k];
            rr += 1;
        }
        norm_w = mathshim::root(norm_w);
        // A column of length one stays as it is, like in LAPACK.
        if norm_w > 0.0 && k < m - 1 {
            // The sign keeps the value away from a loss of digits. The
            // sign follows LAPACK, which puts the negative sign on the
            // diagonal of the factor.
            var alpha: f64 = -norm_w;
            if w[k * n + k] < 0.0 { alpha = norm_w; }
            var v: f64[>..] = [];
            rr = k;
            while rr < m {
                v.push(w[rr * n + k]);
                rr += 1;
            }
            v[0] = v[0] - alpha;
            var norm_v: f64 = 0.0;
            rr = 0;
            while rr < v.len {
                norm_v += v[rr] * v[rr];
                rr += 1;
            }
            norm_v = mathshim::root(norm_v);
            if norm_v > 0.0 {
                rr = 0;
                while rr < v.len {
                    v[rr] = v[rr] / norm_v;
                    hv[(k + rr) * m + k] = v[rr];
                    rr += 1;
                }
                // The turn touches the columns that come after.
                var c: i64 = k;
                while c < n {
                    var s: f64 = 0.0;
                    rr = 0;
                    while rr < v.len {
                        s += v[rr] * w[(k + rr) * n + c];
                        rr += 1;
                    }
                    rr = 0;
                    while rr < v.len {
                        w[(k + rr) * n + c] = w[(k + rr) * n + c] - 2.0 * s * v[rr];
                        rr += 1;
                    }
                    c += 1;
                }
            }
        }
        k += 1;
    }
    // The upper triangle holds the result of the turns.
    i = 0;
    while i < m * n { r.push(0.0); i += 1; }
    i = 0;
    while i < m {
        var c: i64 = 0;
        while c < n {
            if i <= c { r[i * n + c] = w[i * n + c]; }
            c += 1;
        }
        i += 1;
    }
    // The orthogonal matrix starts as the unit matrix and takes the
    // turns in the reverse order.
    i = 0;
    while i < m * m { q.push(0.0); i += 1; }
    i = 0;
    while i < m { q[i * m + i] = 1.0; i += 1; }
    var t: i64 = p - 1;
    while t >= 0 {
        var c: i64 = 0;
        while c < m {
            var s: f64 = 0.0;
            var rr: i64 = t;
            while rr < m {
                s += hv[rr * m + t] * q[rr * m + c];
                rr += 1;
            }
            rr = t;
            while rr < m {
                q[rr * m + c] = q[rr * m + c] - 2.0 * s * hv[rr * m + t];
                rr += 1;
            }
            c += 1;
        }
        t -= 1;
    }
    return no_fault();
}

// qr gives the orthogonal matrix and the upper triangle of the QR
// decomposition of a matrix. The result holds the reduced form, as
// NumPy gives it by default.
fn qr(a) -> Arr, Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "qr", "the operand holds a count of dimensions that is not two");
    }
    let m = a.shape[0];
    let n = a.shape[1];
    var q: f64[>..] = [];
    var r: f64[>..] = [];
    let f = householder_qr(a.data, m, n, q, r);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    // The count of the turns.
    var p = m;
    if n < p { p = n; }
    // The orthogonal matrix holds p columns and the upper triangle
    // holds p rows.
    var qd: f64[>..] = [];
    var i: i64 = 0;
    while i < m * p { qd.push(0.0); i += 1; }
    i = 0;
    while i < m {
        var c: i64 = 0;
        while c < p {
            qd[i * p + c] = q[i * m + c];
            c += 1;
        }
        i += 1;
    }
    var rd: f64[>..] = [];
    i = 0;
    while i < p * n { rd.push(0.0); i += 1; }
    i = 0;
    while i < p {
        var c: i64 = 0;
        while c < n {
            rd[i * n + c] = r[i * n + c];
            c += 1;
        }
        i += 1;
    }
    var sq: i64[..8] = [];
    sq.push(m);
    sq.push(p);
    var sr: i64[..8] = [];
    sr.push(p);
    sr.push(n);
    return Arr { ndim: 2, shape: sq, data: qd }, Arr { ndim: 2, shape: sr, data: rd }, no_fault();
}

