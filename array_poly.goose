// The polynomials.
//
// The coefficients of a polynomial come from the highest power to the
// lowest, as NumPy holds them.
namespace scientific;

import mathshim;

// polyval gives the value of a polynomial at a point.
fn polyval(coeffs: f64[:], x: f64) -> f64 {
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < coeffs.len {
        t = t * x + coeffs[i];
        i += 1;
    }
    return t;
}

// polyfit gives the coefficients of the polynomial that fits the
// points in the sense of the least squares. The coefficients come
// from the highest power to the lowest.
fn polyfit(x: f64[:], y: f64[:], deg: i64) -> Arr, Fault {
    if x.len != y.len {
        return empty_arr(), fault(F_SHAPE, "polyfit", "the points hold a different count");
    }
    if deg < 0 {
        return empty_arr(), fault(F_VALUE, "polyfit", "the degree holds a value below zero");
    }
    let n = x.len;
    if n < deg + 1 {
        return empty_arr(), fault(F_SHAPE, "polyfit", "the points hold fewer than the degree needs");
    }
    // The matrix of the powers of the points.
    var mat: f64[>..] = [];
    var i: i64 = 0;
    while i < n * (deg + 1) { mat.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var p: f64 = 1.0;
        var d: i64 = 0;
        while d <= deg {
            mat[i * (deg + 1) + d] = p;
            p *= x[i];
            d += 1;
        }
        i += 1;
    }
    // The solution comes from the least squares of the matrix.
    var rhs: f64[>..] = [];
    i = 0;
    while i < y.len { rhs.push(y[i]); i += 1; }
    let coeffs, f = lstsq(mat2(copy(mat), n, deg + 1), vec1(copy(rhs)));
    if f.kind != F_NONE { return empty_arr(), f; }
    // The coefficients come from the highest power to the lowest.
    var out: f64[>..] = [];
    i = deg;
    while i >= 0 {
        out.push(coeffs.data[i]);
        i -= 1;
    }
    var s: i64[..8] = [];
    s.push(deg + 1);
    return Arr { ndim: 1, shape: s, data: out }, no_fault();
}

// roots gives the roots of a polynomial. The coefficients come from
// the highest power to the lowest. A root that holds a complex number
// is a fault.
fn roots(coeffs: f64[:]) -> Arr, Fault {
    // The leading coefficient must not be zero.
    var i: i64 = 0;
    while i < coeffs.len {
        if coeffs[i] != 0.0 { break; }
        i += 1;
    }
    if i >= coeffs.len {
        return empty_arr(), fault(F_VALUE, "roots", "the polynomial holds no term");
    }
    // The polynomial without its leading zeros.
    var c: f64[>..] = [];
    var j: i64 = i;
    while j < coeffs.len {
        c.push(coeffs[j]);
        j += 1;
    }
    let n = c.len - 1;
    if n == 0 {
        // The polynomial holds no root.
        var s0: i64[..8] = [];
        s0.push(0);
        return Arr { ndim: 1, shape: s0, data: [] }, no_fault();
    }
    // The companion matrix holds the roots as its eigenvalues.
    var mat: f64[>..] = [];
    var r: i64 = 0;
    while r < n * n { mat.push(0.0); r += 1; }
    r = 0;
    while r < n - 1 {
        mat[r * n + r + 1] = 1.0;
        r += 1;
    }
    r = 0;
    while r < n {
        mat[(n - 1) * n + r] = -c[n - r] / c[0];
        r += 1;
    }
    let vals, vecs, f = eig(mat2(copy(mat), n, n));
    if f.kind != F_NONE { return empty_arr(), f; }
    // The values of the iteration hold few digits. A turn of the
    // Newton method gives the full digits.
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        out.push(newton_root(c, vals.data[i]));
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: out }, no_fault();
}

// mat2 builds a matrix from a buffer of elements.
fn mat2(d: f64[>..], rows: i64, cols: i64) -> Arr {
    var s: i64[..8] = [];
    s.push(rows);
    s.push(cols);
    return Arr { ndim: 2, shape: s, data: d };
}

// vec1 builds a row from a buffer of elements.
fn vec1(d: f64[>..]) -> Arr {
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d };
}

// newton_root gives the root of a polynomial that lies near a value.
// The coefficients come from the highest power to the lowest.
fn newton_root(c: f64[:], x: f64) -> f64 {
    var r = x;
    var pass: i64 = 0;
    while pass < 8 {
        // The value of the polynomial and its derivative.
        var t: f64 = 0.0;
        var dt: f64 = 0.0;
        var i: i64 = 0;
        while i < c.len {
            dt = dt * r + t;
            t = t * r + c[i];
            i += 1;
        }
        if dt == 0.0 { break; }
        let step = t / dt;
        r = r - step;
        if step < 0.0000000000000001 && step > -0.0000000000000001 { break; }
        pass += 1;
    }
    return r;
}
