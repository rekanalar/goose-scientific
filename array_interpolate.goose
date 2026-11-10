// The interpolation by a spline.
//
// The spline holds the points of a signal and the coefficients of
// the second derivative at every point. The second derivatives come
// from a system of equations that the solving of the package takes.
namespace scientific;

import mathshim;

// Spline holds the points of a signal, the values of it, and the
// second derivatives at every point. The buffer holds the points
// first, then the values, then the second derivatives.
struct Spline { count: i64, buf: f64[>..] }

// spline_point gives the point of a number of the signal.
fn spline_point(sp, i: i64) -> f64 { return sp.buf[i]; }

// spline_val gives the value of a number of the signal.
fn spline_val(sp, i: i64) -> f64 { return sp.buf[sp.count + i]; }

// spline_second gives the second derivative of a number of the
// signal.
fn spline_second(sp, i: i64) -> f64 { return sp.buf[2 * sp.count + i]; }

// spline holds the coefficients of a natural cubic spline through the
// points of a signal. The points must rise. The second derivative of
// the natural spline holds zero at the two ends.
fn spline(points: f64[:], values: f64[:]) -> Spline, Fault {
    let n = points.len;
    if n < 3 {
        return empty_spline(), fault(F_VALUE, "spline", "the signal holds fewer than three points");
    }
    if values.len != n {
        return empty_spline(), fault(F_SHAPE, "spline", "the values hold a different count");
    }
    var i: i64 = 1;
    while i < n {
        if points[i] <= points[i - 1] {
            return empty_spline(), fault(F_VALUE, "spline", "the points do not rise");
        }
        i += 1;
    }
    // The second derivatives come from a system of equations. The
    // system holds a count of the points that is two fewer than the
    // signal, and the matrix of it holds three diagonals.
    let m = n - 2;
    var mat: f64[>..] = [];
    var rhs: f64[>..] = [];
    var cell: i64 = 0;
    while cell < m * m { mat.push(0.0); cell += 1; }
    i = 0;
    while i < m {
        // The row of the system holds the three diagonals.
        mat[i * m + i] = 2.0 * (points[i + 2] - points[i]);
        if i > 0 { mat[i * m + i - 1] = points[i + 1] - points[i]; }
        if i + 1 < m { mat[i * m + i + 1] = points[i + 2] - points[i + 1]; }
        // The right side holds the difference of the slopes.
        let s1 = (values[i + 2] - values[i + 1]) / (points[i + 2] - points[i + 1]);
        let s0 = (values[i + 1] - values[i]) / (points[i + 1] - points[i]);
        rhs.push(6.0 * (s1 - s0));
        i += 1;
    }
    // The solving of the package takes the system. The second
    // derivatives of the two ends hold zero, so the values of the
    // system sit in the middle.
    var js: i64[..8] = [];
    js.push(m);
    js.push(m);
    let mat_arr = Arr { ndim: 2, shape: js, data: copy(mat) };
    var rs: i64[..8] = [];
    rs.push(m);
    let rhs_arr = Arr { ndim: 1, shape: rs, data: copy(rhs) };
    let mid, fs = solve(mat_arr, rhs_arr);
    if fs.kind != F_NONE {
        return empty_spline(), fault(F_VALUE, "spline", "the system of the spline holds no solution");
    }
    var buf: f64[>..] = [];
    i = 0;
    while i < n { buf.push(points[i]); i += 1; }
    i = 0;
    while i < n { buf.push(values[i]); i += 1; }
    buf.push(0.0);
    i = 0;
    while i < m { buf.push(mid.data[i]); i += 1; }
    buf.push(0.0);
    return Spline { count: n, buf: buf }, no_fault();
}

// empty_spline gives the spline of no signal.
fn empty_spline() -> Spline {
    return Spline { count: 0, buf: [] };
}

// spline_value gives the value of a spline at a point. A point
// outside the points of the signal is a fault.
fn spline_value(sp, x: f64) -> f64, Fault {
    if sp.count < 3 {
        return 0.0, fault(F_VALUE, "spline_value", "the spline holds no signal");
    }
    if x < spline_point(sp, 0) || x > spline_point(sp, sp.count - 1) {
        return 0.0, fault(F_RANGE, "spline_value", "the point stands outside the signal");
    }
    // The part of the signal that holds the point.
    var i: i64 = 0;
    var hi = sp.count - 1;
    while hi - i > 1 {
        let mid = (i + hi) / 2;
        if spline_point(sp, mid) <= x { i = mid; } else { hi = mid; }
    }
    // The cubic form between the two points of the part.
    let h = spline_point(sp, hi) - spline_point(sp, i);
    let a = (spline_point(sp, hi) - x) / h;
    let b = (x - spline_point(sp, i)) / h;
    return a * spline_val(sp, i) + b * spline_val(sp, hi)
        + ((a * a * a - a) * spline_second(sp, i) + (b * b * b - b) * spline_second(sp, hi)) * h * h / 6.0,
        no_fault();
}

// spline_values gives the values of a spline at a row of points.
fn spline_values(sp, xs: f64[:]) -> Arr, Fault {
    if sp.count < 3 {
        return empty_arr(), fault(F_VALUE, "spline_values", "the spline holds no signal");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < xs.len {
        let v, f = spline_value(sp, xs[i]);
        if f.kind != F_NONE { return empty_arr(), f; }
        d.push(v);
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}
