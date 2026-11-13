// The Cholesky factor and the solver of a symmetric banded matrix.
//
namespace scientific;

import mathshim;
import array_decomp;

// bd_band_dense expands the banded storage into the dense symmetric
// matrix. The count of rows is the width of the band plus one. The
// upper form stores the diagonals above the main one; the lower form
// stores the ones below it.
fn bd_band_dense(ab, lower: bool, n: i64, kd: i64) -> f64[>..], Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { d.push(0.0); i += 1; }
    var j: i64 = 0;
    while j < n {
        var q: i64 = 0;
        if !lower { q = j - kd; } else { q = j; }
        var qtop = j;
        if lower { qtop = j + kd; }
        if q < 0 { q = 0; }
        if qtop > n - 1 { qtop = n - 1; }
        while q <= qtop {
            var row = q - j;
            if !lower { row += kd; }
            let v = ab.data[row * n + j];
            if !(v - v == 0.0) {
                return d, fault(F_VALUE, "cholesky_banded", "the band holds a value that is not finite");
            }
            d[q * n + j] = v;
            d[j * n + q] = v;
            q += 1;
        }
        j += 1;
    }
    return d, no_fault();
}

// cholesky_banded gives the Cholesky factor of the banded symmetric
// positive definite matrix. The storage holds the band in the rows,
// with the diagonal in the last row for the upper form and in the
// first row for the lower form. The result keeps the same form.
fn cholesky_banded(ab, lower: bool = false) -> Arr, Fault {
    if ab.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "cholesky_banded", "the operand is not a matrix");
    }
    let rows = ab.shape[0];
    let n = ab.shape[1];
    if rows < 1 || n < 1 {
        return empty_arr(), fault(F_SHAPE, "cholesky_banded", "the band holds no value");
    }
    let kd = rows - 1;
    if kd >= n {
        return empty_arr(), fault(F_SHAPE, "cholesky_banded", "the width of the band must stay below the size");
    }
    if n * n > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "cholesky_banded", "the matrix holds more elements than the limit");
    }
    let dd, fd = bd_band_dense(ab, lower, n, kd);
    if fd.kind != F_NONE { return empty_arr(), fd; }
    let dense = mat2(copy(dd), n, n);
    let lmat, fl = cholesky(dense);
    if fl.kind != F_NONE {
        return empty_arr(), fault(F_VALUE, "cholesky_banded", "the matrix holds no Cholesky factor");
    }
    // The rows of the result hold the diagonals of the band.
    var out: f64[>..] = [];
    var r: i64 = 0;
    while r <= kd {
        var j: i64 = 0;
        while j < n {
            var v: f64 = 0.0;
            if !lower {
                // The upper form holds U[j-kd+r][j], where U = L
                // transposed.
                if j - kd + r >= 0 { v = lmat.data[j * n + j - kd + r]; }
            } else {
                // The lower form holds L[j+r][j].
                if j + r <= n - 1 { v = lmat.data[(j + r) * n + j]; }
            }
            out.push(v);
            j += 1;
        }
        r += 1;
    }
    return mat2(copy(out), kd + 1, n), no_fault();
}

// bd_solve_band runs the forward and the back substitution on the
// banded Cholesky factor. The flag lower picks the order of the two
// steps. The operand b holds one or several columns.
fn bd_solve_band(cb, b, lower: bool, tag: const u8[:]) -> Arr, Fault {
    let rows = cb.shape[0];
    let n = cb.shape[1];
    let kd = rows - 1;
    if b.ndim < 1 || b.ndim > 2 || b.shape[0] != n {
        return empty_arr(), fault(F_SHAPE, tag, "the right side does not match the band");
    }
    var k: i64 = 1;
    if b.ndim == 2 { k = b.shape[1]; }
    var x: f64[>..] = [];
    var col: i64 = 0;
    while col < k {
        var y: f64[>..] = [];
        var j: i64 = 0;
        while j < n { y.push(b.data[j * k + col]); j += 1; }
        // The first step runs forward and solves the factor that
        // pairs with the forward direction. The diagonal of the upper
        // form sits in the last row and of the lower form in the
        // first row.
        j = 0;
        while j < n {
            var t = y[j];
            var q = 1;
            while q <= kd && q <= j {
                var v: f64 = 0.0;
                if !lower { v = cb.data[(kd - q) * n + j]; } else { v = cb.data[q * n + j - q]; }
                t -= v * y[j - q];
                q += 1;
            }
            var dv: f64 = 0.0;
            if !lower { dv = cb.data[kd * n + j]; } else { dv = cb.data[j]; }
            y[j] = t / dv;
            j += 1;
        }
        // The second step solves the factor itself.
        j = n - 1;
        while j >= 0 {
            var t = y[j];
            var q = 1;
            while q <= kd && q <= n - 1 - j {
                var v: f64 = 0.0;
                if !lower { v = cb.data[(kd - q) * n + j + q]; } else { v = cb.data[q * n + j]; }
                t -= v * y[j + q];
                q += 1;
            }
            var dv: f64 = 0.0;
            if !lower { dv = cb.data[kd * n + j]; } else { dv = cb.data[j]; }
            y[j] = t / dv;
            j -= 1;
        }
        j = 0;
        while j < n { x.push(y[j]); j += 1; }
        col += 1;
    }
    if b.ndim == 1 { return line_arr(copy(x)), no_fault(); }
    return mat2(copy(x), n, k), no_fault();
}

// solveh_banded solves the banded symmetric positive definite system
// with the Cholesky factor of the band. The result keeps the form of
// the right side.
fn solveh_banded(ab, b, lower: bool = false) -> Arr, Fault {
    let f, ff = cholesky_banded(ab, lower);
    if ff.kind != F_NONE { return empty_arr(), ff; }
    return bd_solve_band(f, b, lower, "solveh_banded");
}

// cho_solve_banded solves the banded system with a Cholesky factor
// that the caller already computed through cholesky_banded.
fn cho_solve_banded(cb, b, lower: bool = false) -> Arr, Fault {
    if cb.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "cho_solve_banded", "the factor is not a matrix");
    }
    return bd_solve_band(cb, b, lower, "cho_solve_banded");
}

// rq gives the RQ factorization A = R Q of a matrix. The mode r holds
// back the orthogonal matrix. The turns work on the rows from the
// bottom up. Each turn takes the values of a row segment in the
// reverse order, so that the zeros sit left of the diagonal of the
// trapezoid. The sign of the diagonal follows the leading value of
// the segment, as LAPACK does.
fn rq(a, mode: const u8[:] = "full") -> Arr, Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "rq", "the operand is not a matrix");
    }
    let m = a.shape[0];
    let n = a.shape[1];
    if m < 1 || n < 1 || m * n > MAX_ELEMENTS {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "rq", "the operand holds no value or too many");
    }
    if n * n > MAX_ELEMENTS {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "rq", "the orthogonal matrix holds more elements than the limit");
    }
    if mode != "full" && mode != "economic" && mode != "r" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "rq", "the mode is not full, economic or r");
    }
    var i: i64 = 0;
    while i < m * n {
        if !(a.data[i] - a.data[i] == 0.0) {
            return empty_arr(), empty_arr(), fault(F_VALUE, "rq", "the operand holds a value that is not finite");
        }
        i += 1;
    }
    var w: f64[>..] = [];
    var q: f64[>..] = [];
    i = 0;
    while i < m * n { w.push(a.data[i]); i += 1; }
    i = 0;
    while i < n * n { q.push(0.0); i += 1; }
    i = 0;
    while i < n { q[i * n + i] = 1.0; i += 1; }
    // The turns of the rows, from the bottom up.
    var k = m;
    if n < k { k = n; }
    var vs: f64[>..] = [];
    var ss: f64[>..] = [];
    var i2 = m - 1;
    while i2 >= m - k {
        var s = i2 + n - m;
        if s < 0 { s = 0; }
        // A segment of one value holds no turn, as in LAPACK.
        if s > 0 {
            var seg: f64[>..] = [];
            var r: i64 = 0;
            while r <= s { seg.push(w[i2 * n + s - r]); r += 1; }
            var nx: f64 = 0.0;
            r = 0;
            while r <= s { nx += seg[r] * seg[r]; r += 1; }
            nx = mathshim::root(nx);
            if nx > 0.0 {
                let x1 = seg[0];
                var beta = -nx;
                if x1 < 0.0 { beta = nx; }
                var v: f64[>..] = [];
                r = 0;
                while r <= s { v.push(seg[r]); r += 1; }
                v[0] = seg[0] - beta;
                var nv: f64 = 0.0;
                r = 0;
                while r <= s { nv += v[r] * v[r]; r += 1; }
                nv = mathshim::root(nv);
                if nv > 0.0 {
                    r = 0;
                    while r <= s { v[r] = v[r] / nv; r += 1; }
                    // The turn mixes the columns of the segment in the
                    // rows above.
                    var rr: i64 = 0;
                    while rr <= i2 {
                        var row: f64[>..] = [];
                        var j: i64 = 0;
                        while j <= s { row.push(w[rr * n + s - j]); j += 1; }
                        var t: f64 = 0.0;
                        j = 0;
                        while j <= s { t += v[j] * row[j]; j += 1; }
                        j = 0;
                        while j <= s { row[j] -= 2.0 * t * v[j]; j += 1; }
                        j = 0;
                        while j <= s { w[rr * n + s - j] = row[j]; j += 1; }
                        rr += 1;
                    }
                    // The turn keeps a place in the list of the turns.
                    r = 0;
                    while r <= s { vs.push(v[r]); r += 1; }
                    r = s + 1;
                    // Each turn holds n slots, padded with zeros.
                    while r < n { vs.push(0.0); r += 1; }
                    ss.push(s * 1.0);
                }
            }
        }
        i2 -= 1;
    }
    // The turns build the orthogonal matrix. The product takes the
    // turns of the rows from the bottom up.
    var nr: i64 = ss.len;
    var t2: i64 = 0;
    while t2 < nr {
        let s = ss[t2] as i64;
        var c: i64 = 0;
        while c < n {
            var col: f64[>..] = [];
            var r: i64 = 0;
            while r <= s { col.push(q[(s - r) * n + c]); r += 1; }
            var t: f64 = 0.0;
            r = 0;
            while r <= s { t += vs[t2 * n + r] * col[r]; r += 1; }
            r = 0;
            while r <= s { col[r] -= 2.0 * t * vs[t2 * n + r]; r += 1; }
            r = 0;
            while r <= s { q[(s - r) * n + c] = col[r]; r += 1; }
            c += 1;
        }
        t2 += 1;
    }
    // The result holds the zeros left of the diagonal of the
    // trapezoid.
    var r_out: f64[>..] = [];
    i = 0;
    while i < m {
        var j: i64 = 0;
        while j < n {
            var s = i + n - m;
            if j < s { r_out.push(0.0); } else { r_out.push(w[i * n + j]); }
            j += 1;
        }
        i += 1;
    }
    var rmat = mat2(copy(r_out), m, n);
    var qmat = mat2(copy(q), n, n);
    if mode == "economic" && n > m {
        // The economic form drops the zero columns and the zero rows.
        var re: f64[>..] = [];
        var qe: f64[>..] = [];
        i = 0;
        while i < m {
            var j: i64 = 0;
            while j < m {
                re.push(r_out[i * n + n - m + j]);
                j += 1;
            }
            i += 1;
        }
        i = n - m;
        while i < n {
            var j: i64 = 0;
            while j < n {
                qe.push(q[i * n + j]);
                j += 1;
            }
            i += 1;
        }
        rmat = mat2(copy(re), m, m);
        qmat = mat2(copy(qe), m, n);
    }
    if mode == "r" { qmat = empty_arr(); }
    return rmat, qmat, no_fault();
}
