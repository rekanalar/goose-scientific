// The helpers of the update of a QR decomposition: the rotations and the norms.
//
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

