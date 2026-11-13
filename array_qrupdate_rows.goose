// The insertion and deletion of rows in a QR decomposition.
//
namespace scientific;

import mathshim;
import array_shape;

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

