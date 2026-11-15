// The two dimensional filters and the two dimensional transforms.
//
// medfilt2d and order_filter rank the values of a box around each
// value. sepfir2d filters the rows and then the columns of a matrix.
// firwin_2d designs a two dimensional filter. gauss_spline gives the
// values of a Gaussian spline. hilbert2 gives the analytic part of a
// matrix. wiener smooths a matrix by its local mean and its local
// variance.
//
// The formulas follow SciPy 1.18.1 (scipy.signal).
namespace scientific;

import mathshim;

// sig_interp1 gives the value of a curve at a place, as np.interp
// does. A place outside the curve takes the value of its end.
fn sig_interp1(y: f64[:], x: f64[:], at: f64) -> f64 {
    let n = x.len;
    if n < 1 { return 0.0 / 0.0; }
    if at <= x[0] { return y[0]; }
    if at >= x[n - 1] { return y[n - 1]; }
    var lo: i64 = 0;
    var hi: i64 = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x[mid] > at { hi = mid; } else { lo = mid; }
    }
    let dx = x[hi] - x[lo];
    if dx == 0.0 { return y[lo]; }
    return y[lo] + (y[hi] - y[lo]) * (at - x[lo]) / dx;
}

// medfilt2d gives the median of a box around each value of a matrix,
// as SciPy medfilt2d does. The size holds the count of rows and the
// count of columns of the box. A size of one value gives a square box
// of that side. The empty size gives a box of three by three. The
// values outside the matrix hold zero.
fn medfilt2d(a, size: i64[:] = []) -> Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "medfilt2d", "the matrix must hold two dimensions");
    }
    var sz: i64[>..] = [];
    if size.len == 1 {
        sz.push(size[0]);
        sz.push(size[0]);
    } else if size.len == 2 {
        sz.push(size[0]);
        sz.push(size[1]);
    } else if size.len == 0 {
        sz.push(3);
        sz.push(3);
    } else {
        return empty_arr(), fault(F_SHAPE, "medfilt2d", "the size holds one value or two");
    }
    let out, f = median_filter(a, sz[..], "constant", 0.0, []);
    if f.kind != F_NONE { return empty_arr(), f; }
    return out, no_fault();
}

// order_filter gives the value of a rank in a box around each value of
// a matrix, as SciPy order_filter does. The domain holds the shape of
// the box and the values of its places. A place of the domain with a
// value of no zero joins the box. The rank counts the places of the
// box from the lowest value. The values outside the matrix hold zero.
fn order_filter(a, domain, rank: i64) -> Arr, Fault {
    if a.ndim != 2 || domain.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "order_filter", "the matrix and the domain must hold two dimensions");
    }
    var d: i64 = 0;
    while d < 2 {
        if domain.shape[d] % 2 != 1 {
            return empty_arr(), fault(F_VALUE, "order_filter", "the domain must hold an odd count of places in every direction");
        }
        d += 1;
    }
    let out, f = rank_filter_fp(a, rank, domain, "constant", 0.0, []);
    if f.kind != F_NONE { return empty_arr(), f; }
    return out, no_fault();
}

// sepfir2d filters the rows and then the columns of a matrix, as SciPy
// sepfir2d does. The two filters hold the taps of the rows and of the
// columns. Each filter must hold an odd count of taps. The values
// outside the matrix come from the mirror of the edge, as SciPy does.
fn sepfir2d(a, hrow: f64[:], hcol: f64[:]) -> Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "sepfir2d", "the matrix must hold two dimensions");
    }
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, "sepfir2d");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    if hrow.len < 1 || hcol.len < 1 {
        return empty_arr(), fault(F_VALUE, "sepfir2d", "the filters must hold one tap or more");
    }
    if hrow.len % 2 != 1 || hcol.len % 2 != 1 {
        return empty_arr(), fault(F_VALUE, "sepfir2d", "each filter must hold an odd count of taps");
    }
    // SciPy convolves with the taps turned around. The values outside
    // come from the mirror of the edge.
    let rows, f1 = sepfir_axis(a, hrow, 1);
    if f1.kind != F_NONE { return empty_arr(), f1; }
    let out, f2 = sepfir_axis(rows, hcol, 0);
    if f2.kind != F_NONE { return empty_arr(), f2; }
    return out, no_fault();
}

// sepfir_axis convolves a matrix along one axis with the taps turned
// around. The values outside the matrix come from the mirror of the
// edge.
fn sepfir_axis(a, taps: f64[:], axis: i64) -> Arr, Fault {
    if a.ndim < 1 || axis < 0 || axis >= a.ndim {
        return empty_arr(), fault(F_SHAPE, "sepfir_axis", "the axis stands outside the shape");
    }
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, "sepfir_axis");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    if a.shape[axis] < 1 {
        return empty_arr(), fault(F_SHAPE, "sepfir_axis", "the side of the axis holds no element");
    }
    let n = a.shape[axis];
    let m = taps.len;
    let half = m / 2;
    var inner: i64 = 1;
    var k: i64 = axis + 1;
    while k < a.ndim { inner *= a.shape[k]; k += 1; }
    let outer = a.data.len / (n * inner);
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { out.push(0.0); i += 1; }
    var o: i64 = 0;
    while o < outer {
        var q: i64 = 0;
        while q < inner {
            let base = o * n * inner + q;
            var p: i64 = 0;
            while p < n {
                var sum: f64 = 0.0;
                var j: i64 = 0;
                while j < m {
                    let at = p + j - half;
                    sum += taps[m - 1 - j] * a.data[base + sepfir_src(at, n) * inner];
                    j += 1;
                }
                out[base + p * inner] = sum;
                p += 1;
            }
            q += 1;
        }
        o += 1;
    }
    return arr_of(copy(out), a.shape), no_fault();
}

// sepfir_src gives the index in 0 to n - 1 that stands for the index i
// under the mirror of the edge.
fn sepfir_src(i: i64, n: i64) -> i64 {
    if i >= 0 && i < n { return i; }
    if n == 1 { return 0; }
    let p = 2 * n;
    var m = i % p;
    if m < 0 { m += p; }
    if m >= n { m = p - 1 - m; }
    return m;
}

// firwin_2d designs a filter of two dimensions, as SciPy firwin_2d
// does. The counts rows and cols hold the count of taps in the two
// directions. The window is a name of get_window with its parameter
// param. The filter holds the cut-off frequency fc over the rate fs.
// The filter with circular true holds a round shape.
fn firwin_2d(rows: i64, cols: i64, fc: f64, window: const u8[:] = "hamming",
             param: f64 = 0.0 / 0.0, circular: bool = false,
             fs: f64 = 2.0) -> Arr, Fault {
    let tag = "firwin_2d";
    if rows < 1 || cols < 1 {
        return empty_arr(), fault(F_VALUE, tag, "the counts of taps must hold one value or more");
    }
    if !(fs > 0.0) || !(fc > 0.0) || fc >= fs / 2.0 {
        return empty_arr(), fault(F_VALUE, tag, "the cut-off frequency must stand above zero and below the Nyquist rate");
    }
    if !circular {
        let rw, f1 = firwin(rows, [fc], true, window, true, fs, param);
        if f1.kind != F_NONE { return empty_arr(), f1; }
        let cw, f2 = firwin(cols, [fc], true, window, true, fs, param);
        if f2.kind != F_NONE { return empty_arr(), f2; }
        var out: f64[>..] = [];
        var r: i64 = 0;
        while r < rows {
            var c: i64 = 0;
            while c < cols {
                out.push(rw.data[r] * cw.data[c]);
                c += 1;
            }
            r += 1;
        }
        var s: i64[..8] = [];
        s.push(rows);
        s.push(cols);
        return Arr { ndim: 2, shape: s, data: out }, no_fault();
    }
    // The round filter takes the values of a one dimensional filter at
    // the distance from the middle of the matrix. The rows of the
    // result follow the second count and the columns follow the first
    // count, as SciPy does.
    var n_r = rows;
    if cols > n_r { n_r = cols; }
    n_r = n_r * 8;
    if n_r > MAX_DSP_LENGTH {
        return empty_arr(), fault(F_LIMIT, tag, "the count of taps is out of range");
    }
    let wr, f1 = firwin(n_r, [fc], true, window, true, fs, param);
    if f1.kind != F_NONE { return empty_arr(), f1; }
    var grid: f64[>..] = [];
    var g: i64 = 0;
    while g < n_r { grid.push((g as f64) / ((n_r - 1) as f64)); g += 1; }
    var f1v: f64[>..] = [];
    var r: i64 = 0;
    while r < cols {
        var y: f64 = -1.0;
        if cols > 1 { y = -1.0 + 2.0 * (r as f64) / ((cols - 1) as f64); }
        var c: i64 = 0;
        while c < rows {
            var x: f64 = -1.0;
            if rows > 1 { x = -1.0 + 2.0 * (c as f64) / ((rows - 1) as f64); }
            let rad = mathshim::root(x * x + y * y);
            f1v.push(sig_interp1(wr.data, grid, rad));
            c += 1;
        }
        r += 1;
    }
    var s: i64[..8] = [];
    s.push(cols);
    s.push(rows);
    return Arr { ndim: 2, shape: s, data: f1v }, no_fault();
}

// gauss_spline gives the values of a Gaussian spline at the places x,
// as SciPy gauss_spline does. The order n sets the width of the
// spline. The count of the spline follows the count of the places.
fn gauss_spline(x: f64[:], n: i64) -> Arr, Fault {
    if x.len < 1 || !all_finite(x) {
        return empty_arr(), fault(F_VALUE, "gauss_spline", "the places must be a finite list");
    }
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "gauss_spline", "the order must hold one value or more");
    }
    let signsq = (n + 1) as f64 / 12.0;
    let factor = 1.0 / mathshim::root(6.283185307179586 * signsq);
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len {
        out.push(factor * mathshim::power_e(-x[i] * x[i] / 2.0 / signsq));
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// hilbert2 gives the analytic part of a matrix, as SciPy hilbert2 does.
// The call gives the real and the imaginary part of the transform.
fn hilbert2(a) -> Arr, Arr, Fault {
    let tag = "hilbert2";
    if a.ndim != 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the matrix must hold two dimensions");
    }
    if a.data.len < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the matrix must hold one value or more");
    }
    let rows = a.shape[0];
    let cols = a.shape[1];
    let xr, xi, f = fft2(a);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let k0 = (rows + 1) / 2;
    let k1 = (cols + 1) / 2;
    // The rows and the columns of the inner half hold twice their
    // value, as SciPy hilbert2 does.
    var r: i64 = 0;
    while r < rows {
        if r > 0 && r < k0 {
            var c: i64 = 0;
            while c < cols {
                xr.data[r * cols + c] *= 2.0;
                xi.data[r * cols + c] *= 2.0;
                c += 1;
            }
        }
        r += 1;
    }
    var c: i64 = 0;
    while c < cols {
        if c > 0 && c < k1 {
            r = 0;
            while r < rows {
                xr.data[r * cols + c] *= 2.0;
                xi.data[r * cols + c] *= 2.0;
                r += 1;
            }
        }
        c += 1;
    }
    // The outer half of the rows and of the columns holds no value.
    r = 0;
    while r < rows {
        c = 0;
        while c < cols {
            if r >= k0 || c >= k1 {
                xr.data[r * cols + c] = 0.0;
                xi.data[r * cols + c] = 0.0;
            }
            c += 1;
        }
        r += 1;
    }
    let outr, outi, f2 = ifft2(xr, xi);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
    return outr, outi, no_fault();
}

// wiener smooths a signal or a matrix by its local mean and its local
// variance, as SciPy wiener does. The size holds the count of places
// of the box in every direction. The empty size gives a box of three
// by three. The noise holds the noise power. The missing value of the
// noise takes the mean of the local variance. The values outside the
// signal hold zero.
fn wiener(a, size: i64[:] = [], noise: f64 = 0.0 / 0.0) -> Arr, Fault {
    let tag = "wiener";
    if a.ndim < 1 || a.ndim > 2 {
        return empty_arr(), fault(F_SHAPE, tag, "the signal must hold one dimension or two");
    }
    if a.data.len < 1 {
        return empty_arr(), fault(F_VALUE, tag, "the signal must hold one value or more");
    }
    var sz: i64[>..] = [];
    if size.len == 0 {
        var i: i64 = 0;
        while i < a.ndim { sz.push(3); i += 1; }
    } else if size.len == 1 && a.ndim == 2 {
        sz.push(size[0]);
        sz.push(size[0]);
    } else if size.len == a.ndim {
        var k: i64 = 0;
        while k < size.len { sz.push(size[k]); k += 1; }
    } else {
        return empty_arr(), fault(F_SHAPE, tag, "the size holds one value for each direction");
    }
    var i: i64 = 0;
    while i < sz.len {
        if sz[i] < 1 {
            return empty_arr(), fault(F_VALUE, tag, "the size holds a value below one");
        }
        i += 1;
    }
    let dims = a.shape;
    let n = a.data.len;
    var lmean: f64[>..] = [];
    var lvar: f64[>..] = [];
    var count: f64 = 1.0;
    i = 0;
    while i < sz.len { count *= sz[i] as f64; i += 1; }
    var idx: i64 = 0;
    while idx < n {
        var sum: f64 = 0.0;
        var sumsq: f64 = 0.0;
        if a.ndim == 1 {
            let centre = idx % dims[0];
            let half = sz[0] / 2;
            var k: i64 = centre - half;
            while k <= centre + sz[0] - 1 - half {
                if k >= 0 && k < dims[0] {
                    sum += a.data[k];
                    sumsq += a.data[k] * a.data[k];
                }
                k += 1;
            }
        } else {
            let r = idx / dims[1];
            let c = idx % dims[1];
            let half0 = sz[0] / 2;
            let half1 = sz[1] / 2;
            var i0: i64 = r - half0;
            while i0 <= r + sz[0] - 1 - half0 {
                var i1: i64 = c - half1;
                while i1 <= c + sz[1] - 1 - half1 {
                    if i0 >= 0 && i0 < dims[0] && i1 >= 0 && i1 < dims[1] {
                        let v = a.data[i0 * dims[1] + i1];
                        sum += v;
                        sumsq += v * v;
                    }
                    i1 += 1;
                }
                i0 += 1;
            }
        }
        let m = sum / count;
        lmean.push(m);
        lvar.push(sumsq / count - m * m);
        idx += 1;
    }
    var nz = noise;
    if nz != nz {
        var t: f64 = 0.0;
        i = 0;
        while i < lvar.len { t += lvar[i]; i += 1; }
        nz = t / (lvar.len as f64);
    }
    var out: f64[>..] = [];
    i = 0;
    while i < n {
        let m = lmean[i];
        let v = lvar[i];
        if v < nz {
            out.push(m);
        } else {
            let res = (a.data[i] - m) * (1.0 - nz / v) + m;
            out.push(res);
        }
        i += 1;
    }
    return arr_of(copy(out), dims), no_fault();
}
