// The insertion and deletion of columns in a QR decomposition.
//
namespace scientific;

import mathshim;
import array_shape;

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

