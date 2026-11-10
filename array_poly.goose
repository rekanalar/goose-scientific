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
