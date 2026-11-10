// The interpolation of one variable: the cubic spline, the monotone
// cubic of Fritsch and Carlson, the spline of Akima and the pieces of
// scipy.interpolate.interp1d. A spline holds one cubic per interval.
// The buffer holds the n knots and then the cubics. The cubic of the
// interval i reads c[4 i] (x - x[i])^3 + c[4 i + 1]
// (x - x[i])^2 + c[4 i + 2] (x - x[i]) + c[4 i + 3].
namespace scientific;

import mathshim;

// Cubics holds the knots and the coefficients of the cubics.
struct Cubics { n: i64, extrap: bool, buf: f64[>..] }

// spline_knots checks the knots: two or more, finite, rising.
fn spline_knots(x: f64[:], y: f64[:], tag: const u8[:]) -> Fault {
    if x.len != y.len || x.len < 2 {
        return fault(F_SHAPE, tag, "the knots and the values need one count of two or more");
    }
    var i: i64 = 0;
    while i < x.len {
        if !(x[i] - x[i] == 0.0) || !(y[i] - y[i] == 0.0) {
            return fault(F_VALUE, tag, "a knot or a value is not finite");
        }
        if i > 0 && !(x[i] > x[i - 1]) {
            return fault(F_VALUE, tag, "the knots do not rise strictly");
        }
        i += 1;
    }
    return no_fault();
}

// spline_from_slopes builds the cubics from the values and the slopes
// at the knots.
fn spline_from_slopes(x: f64[:], y: f64[:], s: f64[:], extrap: bool = true) -> Cubics {
    var c: f64[>..] = [];
    var i: i64 = 0;
    while i + 1 < x.len {
        let dx = x[i + 1] - x[i];
        let sl = (y[i + 1] - y[i]) / dx;
        let t = (s[i] + s[i + 1] - 2.0 * sl) / dx;
        c.push(t / dx);
        c.push((sl - s[i]) / dx - t);
        c.push(s[i]);
        c.push(y[i]);
        i += 1;
    }
    var buf: f64[>..] = [];
    i = 0;
    while i < x.len { buf.push(x[i]); i += 1; }
    i = 0;
    while i < c.len { buf.push(c[i]); i += 1; }
    return Cubics { n: x.len, extrap: extrap, buf: buf };
}

// tridiag solves a tridiagonal system. The arrays hold the lower, main
// and upper diagonals and the right side. The arrays change.
fn tridiag(lo: f64[>..]&, di: f64[>..]&, up: f64[>..]&, b: f64[>..]&) {
    let n = di.len;
    var i: i64 = 1;
    while i < n {
        let m = lo[i] / di[i - 1];
        di[i] = di[i] - m * up[i - 1];
        b[i] = b[i] - m * b[i - 1];
        i += 1;
    }
    b[n - 1] = b[n - 1] / di[n - 1];
    i = n - 2;
    while i >= 0 {
        b[i] = (b[i] - up[i] * b[i + 1]) / di[i];
        i -= 1;
    }
}

// cubic_spline builds the interpolating cubic spline. The condition bc
// is "not-a-knot", "natural" or "clamped". A clamped spline has zero
// slope at both ends.
fn cubic_spline(x: f64[:], y: f64[:], bc: const u8[:] = "not-a-knot") -> Cubics, Fault {
    let f = spline_knots(x, y, "cubic_spline");
    if f.kind != F_NONE { return Cubics { n: 0, extrap: true, buf: [] }, f; }
    if bc != "not-a-knot" && bc != "natural" && bc != "clamped" {
        return Cubics { n: 0, extrap: true, buf: [] }, fault(F_VALUE, "cubic_spline", "the condition holds an unknown name");
    }
    let n = x.len;
    var dx: f64[>..] = [];
    var sl: f64[>..] = [];
    var i: i64 = 0;
    while i + 1 < n {
        dx.push(x[i + 1] - x[i]);
        sl.push((y[i + 1] - y[i]) / dx[i]);
        i += 1;
    }
    var s: f64[>..] = [];
    if n == 2 {
        if bc == "clamped" { s = [0.0, 0.0]; }
        else if bc == "natural" { s = [sl[0], sl[0]]; }
        else { s = [sl[0], sl[0]]; }
        return spline_from_slopes(x, y, s), no_fault();
    }
    if n == 3 && bc == "not-a-knot" {
        // One parabola through the three points.
        let d0 = dx[0];
        let d1 = dx[1];
        let a = (sl[1] - sl[0]) / (d0 + d1);
        s = [sl[0] - a * d0, sl[0] + a * d0, sl[1] + a * d1];
        return spline_from_slopes(x, y, s), no_fault();
    }
    var lo: f64[>..] = [];
    var di: f64[>..] = [];
    var up: f64[>..] = [];
    var b: f64[>..] = [];
    i = 0;
    while i < n { lo.push(0.0); di.push(0.0); up.push(0.0); b.push(0.0); i += 1; }
    i = 1;
    while i < n - 1 {
        lo[i] = dx[i];
        di[i] = 2.0 * (dx[i - 1] + dx[i]);
        up[i] = dx[i - 1];
        b[i] = 3.0 * (dx[i] * sl[i - 1] + dx[i - 1] * sl[i]);
        i += 1;
    }
    if bc == "not-a-knot" {
        let d = x[2] - x[0];
        di[0] = dx[1];
        up[0] = d;
        b[0] = ((dx[0] + 2.0 * d) * dx[1] * sl[0] + dx[0] * dx[0] * sl[1]) / d;
        let e = x[n - 1] - x[n - 3];
        lo[n - 1] = e;
        di[n - 1] = dx[n - 3];
        b[n - 1] = (dx[n - 2] * dx[n - 2] * sl[n - 3] + (2.0 * e + dx[n - 2]) * dx[n - 3] * sl[n - 2]) / e;
    } else if bc == "natural" {
        di[0] = 2.0;
        up[0] = 1.0;
        b[0] = 3.0 * sl[0];
        lo[n - 1] = 1.0;
        di[n - 1] = 2.0;
        b[n - 1] = 3.0 * sl[n - 2];
    } else {
        di[0] = 1.0;
        di[n - 1] = 1.0;
    }
    tridiag(lo, di, up, b);
    return spline_from_slopes(x, y, b), no_fault();
}

// pchip builds the monotone cubic of Fritsch and Carlson. The cubic
// keeps the monotone runs of the data.
fn pchip(x: f64[:], y: f64[:]) -> Cubics, Fault {
    let f = spline_knots(x, y, "pchip");
    if f.kind != F_NONE { return Cubics { n: 0, extrap: true, buf: [] }, f; }
    let n = x.len;
    var h: f64[>..] = [];
    var dl: f64[>..] = [];
    var i: i64 = 0;
    while i + 1 < n {
        h.push(x[i + 1] - x[i]);
        dl.push((y[i + 1] - y[i]) / h[i]);
        i += 1;
    }
    var d: f64[>..] = [];
    if n == 2 {
        d = [dl[0], dl[0]];
        return spline_from_slopes(x, y, d), no_fault();
    }
    i = 0;
    while i < n { d.push(0.0); i += 1; }
    i = 1;
    while i < n - 1 {
        if dl[i - 1] * dl[i] > 0.0 {
            let w1 = 2.0 * h[i] + h[i - 1];
            let w2 = h[i] + 2.0 * h[i - 1];
            d[i] = (w1 + w2) / (w1 / dl[i - 1] + w2 / dl[i]);
        }
        i += 1;
    }
    d[0] = pchip_edge(h[0], h[1], dl[0], dl[1]);
    d[n - 1] = pchip_edge(h[n - 2], h[n - 3], dl[n - 2], dl[n - 3]);
    return spline_from_slopes(x, y, d), no_fault();
}

fn sign_of(v: f64) -> f64 {
    if v > 0.0 { return 1.0; }
    if v < 0.0 { return -1.0; }
    return 0.0;
}

// pchip_edge gives the slope at an end by the three point formula.
fn pchip_edge(h0: f64, h1: f64, m0: f64, m1: f64) -> f64 {
    var d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if sign_of(d) != sign_of(m0) { d = 0.0; }
    else if sign_of(m0) != sign_of(m1) && mathshim::abs_of(d) > 3.0 * mathshim::abs_of(m0) { d = 3.0 * m0; }
    return d;
}

// akima builds the spline of Akima. It needs three knots at least.
fn akima(x: f64[:], y: f64[:]) -> Cubics, Fault {
    let f = spline_knots(x, y, "akima");
    if f.kind != F_NONE { return Cubics { n: 0, extrap: true, buf: [] }, f; }
    let n = x.len;
    if n < 3 {
        return Cubics { n: 0, extrap: true, buf: [] }, fault(F_SHAPE, "akima", "the spline needs three knots");
    }
    // m holds the slopes with two extra values at each end.
    var m: f64[>..] = [];
    var i: i64 = 0;
    while i < n + 3 { m.push(0.0); i += 1; }
    i = 0;
    while i + 1 < n { m[i + 2] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]); i += 1; }
    m[1] = 2.0 * m[2] - m[3];
    m[0] = 2.0 * m[1] - m[2];
    m[n + 1] = 2.0 * m[n] - m[n - 1];
    m[n + 2] = 2.0 * m[n + 1] - m[n];
    var f12: f64[>..] = [];
    var top: f64 = 0.0;
    i = 0;
    while i < n {
        let v = mathshim::abs_of(m[i + 3] - m[i + 2]) + mathshim::abs_of(m[i + 1] - m[i]);
        f12.push(v);
        if v > top { top = v; }
        i += 1;
    }
    var s: f64[>..] = [];
    i = 0;
    while i < n {
        if f12[i] > 1.0e-9 * top {
            let w1 = mathshim::abs_of(m[i + 3] - m[i + 2]);
            let w2 = mathshim::abs_of(m[i + 1] - m[i]);
            s.push((w1 * m[i + 1] + w2 * m[i + 2]) / f12[i]);
        } else {
            s.push(0.5 * (m[i + 1] + m[i + 2]));
        }
        i += 1;
    }
    return spline_from_slopes(x, y, s, false), no_fault();
}

// spline_find gives the interval that holds a point. A point outside
// the knots takes the first or the last interval.
fn spline_find(x: f64[:], n: i64, v: f64) -> i64 {
    var lo: i64 = 0;
    var hi: i64 = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if v >= x[mid] { lo = mid; } else { hi = mid; }
    }
    return lo;
}

// spline_eval gives the value of the nu-th derivative of a spline at
// the points. The derivative order runs from 0 to 3. A point outside
// the knots takes the cubic of the end interval, as SciPy does for
// cubic_spline and pchip. A spline of akima gives NaN outside.
fn spline_eval(s, xs: f64[:], nu: i64 = 0) -> Arr, Fault {
    if nu < 0 || nu > 3 {
        return empty_arr(), fault(F_RANGE, "spline_eval", "the derivative order is outside zero to three");
    }
    if xs.len < 1 {
        return empty_arr(), fault(F_SHAPE, "spline_eval", "the list of points holds no value");
    }
    var out: f64[>..] = [];
    var j: i64 = 0;
    while j < xs.len {
        let v = xs[j];
        if v != v || (!s.extrap && (v < s.buf[0] || v > s.buf[s.n - 1])) { out.push(0.0 / 0.0); j += 1; continue; }
        let i = spline_find(s.buf, s.n, v);
        let t = v - s.buf[i];
        let c3 = s.buf[s.n + 4 * i];
        let c2 = s.buf[s.n + 4 * i + 1];
        let c1 = s.buf[s.n + 4 * i + 2];
        let c0 = s.buf[s.n + 4 * i + 3];
        if nu == 0 { out.push(((c3 * t + c2) * t + c1) * t + c0); }
        else if nu == 1 { out.push((3.0 * c3 * t + 2.0 * c2) * t + c1); }
        else if nu == 2 { out.push(6.0 * c3 * t + 2.0 * c2); }
        else { out.push(6.0 * c3); }
        j += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// spline_antider gives the integral of the cubic of interval i from
// the left knot to the offset t.
fn spline_part(s, i: i64, t: f64) -> f64 {
    return (((s.buf[s.n + 4 * i] * t / 4.0 + s.buf[s.n + 4 * i + 1] / 3.0) * t + s.buf[s.n + 4 * i + 2] / 2.0) * t + s.buf[s.n + 4 * i + 3]) * t;
}

// spline_integrate gives the integral of a spline from a to b. The
// ends may lie outside the knots, and a lies above b for a negative
// result.
fn spline_integrate(s, a: f64, b: f64) -> f64 {
    if a > b { return -spline_integrate_up(s, b, a); }
    return spline_integrate_up(s, a, b);
}

// spline_integrate_up integrates from a to b with a not above b.
fn spline_integrate_up(s, a: f64, b: f64) -> f64 {
    if a == b { return 0.0; }
    let ia = spline_find(s.buf, s.n, a);
    let ib = spline_find(s.buf, s.n, b);
    if ia == ib { return spline_part(s, ia, b - s.buf[ia]) - spline_part(s, ia, a - s.buf[ia]); }
    var total = spline_part(s, ia, s.buf[ia + 1] - s.buf[ia]) - spline_part(s, ia, a - s.buf[ia]);
    var i = ia + 1;
    while i < ib { total += spline_part(s, i, s.buf[i + 1] - s.buf[i]); i += 1; }
    total += spline_part(s, ib, b - s.buf[ib]);
    return total;
}

// interp1d interpolates the values y at the knots x for the points xs.
// The kind is "linear", "slinear", "nearest", "previous", "zero", "next"
// or "cubic". With fill, a point outside the knots takes fill_low below
// them and fill_high above them, as fill_value does in SciPy. The
// kind "nearest" takes the lower knot at a tie. A point outside the
// knots is a fault unless extrapolate holds true. Then "linear" and
// "cubic" extend their end pieces, and the other kinds hold the end
// value.
fn interp1d(x: f64[:], y: f64[:], xs: f64[:], kind0: const u8[:] = "linear", extrapolate: bool = false, fill: bool = false, fill_low: f64 = 0.0, fill_high: f64 = 0.0) -> Arr, Fault {
    // SciPy names the same kinds "slinear" and "zero".
    var kind = kind0;
    if kind0 == "slinear" { kind = "linear"; }
    if kind0 == "zero" { kind = "previous"; }
    let f = spline_knots(x, y, "interp1d");
    if f.kind != F_NONE { return empty_arr(), f; }
    if kind != "linear" && kind != "nearest" && kind != "previous" && kind != "next" && kind != "cubic" {
        return empty_arr(), fault(F_VALUE, "interp1d", "the kind holds an unknown name");
    }
    if xs.len < 1 {
        return empty_arr(), fault(F_SHAPE, "interp1d", "the list of points holds no value");
    }
    var k: i64 = 0;
    while k < xs.len {
        if xs[k] != xs[k] { k += 1; continue; }
        if !extrapolate && !fill && (xs[k] < x[0] || xs[k] > x[x.len - 1]) {
            return empty_arr(), fault(F_RANGE, "interp1d", "a point lies outside the knots");
        }
        k += 1;
    }
    if kind == "cubic" {
        let sp, e = cubic_spline(x, y);
        if e.kind != F_NONE { return empty_arr(), e; }
        let r, e2 = spline_eval(sp, xs);
        if e2.kind != F_NONE || !fill || extrapolate { return r, e2; }
        var rd = copy_slice(r.data);
        var q: i64 = 0;
        while q < xs.len {
            if xs[q] < x[0] { rd[q] = fill_low; }
            else if xs[q] > x[x.len - 1] { rd[q] = fill_high; }
            q += 1;
        }
        return line_arr(copy(rd)), no_fault();
    }
    var out: f64[>..] = [];
    k = 0;
    while k < xs.len {
        let v = xs[k];
        if v != v { out.push(v); k += 1; continue; }
        if fill && !extrapolate && v < x[0] { out.push(fill_low); k += 1; continue; }
        if fill && !extrapolate && v > x[x.len - 1] { out.push(fill_high); k += 1; continue; }
        var i = spline_find(x, x.len, v);
        if kind == "linear" {
            let t = (v - x[i]) / (x[i + 1] - x[i]);
            out.push(y[i] + t * (y[i + 1] - y[i]));
        } else {
            var pick = i;
            if v <= x[0] { pick = 0; }
            else if v >= x[x.len - 1] { pick = x.len - 1; }
            else if kind == "nearest" {
                if v - x[i] > x[i + 1] - v { pick = i + 1; }
            } else if kind == "next" {
                if v > x[i] { pick = i + 1; }
            }
            out.push(y[pick]);
        }
        k += 1;
    }
    return line_arr(copy(out)), no_fault();
}
