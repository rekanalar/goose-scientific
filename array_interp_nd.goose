// The splines and the piecewise polynomials of many dimensions: the
// object of the tensor product of the B-splines of the axes and the
// object of the tensor product of the pieces of power form. The names
// follow scipy.interpolate.NdBSpline and scipy.interpolate.NdPPoly.
// The coefficients hold in row-major order: the last axis changes
// fastest.
namespace scientific;

import mathshim;

// NdSpl holds the count of the axes, the count of the knots, and one
// buffer. The buffer holds the degrees first, then the counts of the
// knots, then the knots of the axes, then the coefficients. The
// struct holds one resizable field, so the values share one buffer.
struct NdSpl { ndim: i64, kn_len: i64, buf: f64[>..] }

fn ndspl_empty() -> NdSpl {
    return NdSpl { ndim: 0, kn_len: 0, buf: [] };
}

// nd_check_input holds the rules of the data of nd_bspline and nd_poly.
// The buffer t holds the knots or the breakpoints. The flag poly
// selects the rules of the polynomial.
fn nd_check_input(nts: i64[:], ks: i64[:], t: f64[:], c: f64[:], poly: bool, tag: const u8[:]) -> Fault {
    if nts.len < 1 || ks.len != nts.len {
        if poly { return fault(F_SHAPE, tag, "the counts of the breakpoints and the degrees need one axis each"); }
        return fault(F_SHAPE, tag, "the counts of the knots and the degrees need one axis each");
    }
    if nts.len > MAX_DIMS {
        if poly { return fault(F_LIMIT, tag, "the polynomial holds more axes than the limit"); }
        return fault(F_LIMIT, tag, "the spline holds more axes than the limit");
    }
    var nc: i64 = 1;
    var npow: i64 = 1;
    var npiece: i64 = 1;
    var at: i64 = 0;
    var i: i64 = 0;
    while i < nts.len {
        if ks[i] < 0 || ks[i] > 20 { return fault(F_RANGE, tag, "a degree is outside zero to twenty"); }
        if poly && nts[i] < 2 { return fault(F_SHAPE, tag, "an axis holds too few breakpoints"); }
        if !poly && nts[i] < ks[i] + 2 { return fault(F_SHAPE, tag, "an axis holds too few knots for its degree"); }
        if at + nts[i] > t.len {
            if poly { return fault(F_SHAPE, tag, "the buffer of the breakpoints holds too few values"); }
            return fault(F_SHAPE, tag, "the buffer of the knots holds too few values");
        }
        if poly { npow *= ks[i] + 1; npiece *= nts[i] - 1; }
        if !poly { nc *= nts[i] - ks[i] - 1; }
        at += nts[i];
        i += 1;
    }
    if at != t.len {
        if poly { return fault(F_SHAPE, tag, "the buffer of the breakpoints holds other counts"); }
        return fault(F_SHAPE, tag, "the buffer of the knots holds other counts");
    }
    if poly && c.len != npow * npiece {
        return fault(F_SHAPE, tag, "the coefficients need the product of the powers and the pieces");
    }
    if !poly && c.len != nc {
        return fault(F_SHAPE, tag, "the coefficients need the product of the counts of the basis functions");
    }
    i = 0;
    while i < t.len {
        if !(t[i] - t[i] == 0.0) {
            if poly { return fault(F_VALUE, tag, "a breakpoint is not finite"); }
            return fault(F_VALUE, tag, "a knot is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < c.len {
        if !(c[i] - c[i] == 0.0) { return fault(F_VALUE, tag, "a coefficient is not finite"); }
        i += 1;
    }
    return no_fault();
}

// nd_bspline holds the spline of the caller. The knots of the axes
// come in one buffer, with the count of the knots of each axis in
// nts, and the degrees in ks.
fn nd_bspline(t: f64[:], nts: i64[:], ks: i64[:], c: f64[:]) -> NdSpl, Fault {
    let f = nd_check_input(nts, ks, t, c, false, "nd_bspline");
    if f.kind != F_NONE { return ndspl_empty(), f; }
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < ks.len { buf.push(ks[i] as f64); i += 1; }
    i = 0;
    while i < nts.len { buf.push(nts[i] as f64); i += 1; }
    i = 0;
    while i < t.len { buf.push(t[i]); i += 1; }
    i = 0;
    while i < c.len { buf.push(c[i]); i += 1; }
    return NdSpl { ndim: nts.len, kn_len: t.len, buf: buf }, no_fault();
}

// ndspl_eval gives the values of the spline at the points of xi. The
// buffer of xi holds ndim values for each point. The orders nu give
// the derivative of each axis. A point outside the knots holds the
// value of the end pieces, or NaN when extrapolate is false.
fn ndspl_eval(s, xi: f64[:], npoints: i64, nu: i64[:], extrapolate: bool) -> Arr, Fault {
    if s.ndim < 1 {
        return empty_arr(), fault(F_SHAPE, "nd_bspline_eval", "the spline holds no axis");
    }
    if nu.len != 0 && nu.len != s.ndim {
        return empty_arr(), fault(F_SHAPE, "nd_bspline_eval", "the orders need one value for each axis");
    }
    if xi.len != npoints * s.ndim {
        return empty_arr(), fault(F_SHAPE, "nd_bspline_eval", "the points need the count of the axes for each point");
    }
    var i: i64 = 0;
    while i < xi.len {
        if !(xi[i] - xi[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "nd_bspline_eval", "a point is not finite");
        }
        i += 1;
    }
    // The start of the knots of each axis and the stride of the
    // coefficients of each axis.
    var koff: i64[>..] = [];
    var stride: i64[>..] = [];
    var d: i64 = 0;
    while d < s.ndim {
        koff.push(0);
        stride.push(0);
        d += 1;
    }
    var sum: i64 = 0;
    d = 0;
    while d < s.ndim {
        koff[d] = sum;
        sum += s.buf[s.ndim + d] as i64;
        d += 1;
    }
    var acc: i64 = 1;
    d = s.ndim - 1;
    while d >= 0 {
        stride[d] = acc;
        acc *= (s.buf[s.ndim + d] as i64) - (s.buf[d] as i64) - 1;
        d -= 1;
    }
    var out: f64[>..] = [];
    var p: i64 = 0;
    while p < npoints {
        // The basis values and the first column of the basis of each
        // axis at this point.
        var basis: f64[>..] = [];
        var offs: i64[>..] = [];
        var outside = false;
        d = 0;
        while d < s.ndim {
            let base = koff[d];
            let nt = s.buf[s.ndim + d] as i64;
            var t: f64[>..] = [];
            var q: i64 = 0;
            while q < nt { t.push(s.buf[2 * s.ndim + base + q]); q += 1; }
            let v = xi[p * s.ndim + d];
            var order: i64 = 0;
            if nu.len == s.ndim { order = nu[d]; }
            if order < 0 {
                return empty_arr(), fault(F_RANGE, "nd_bspline_eval", "an order of the derivative is below zero");
            }
            let kd = s.buf[d] as i64;
            if !extrapolate && (v < t[kd] || v > t[nt - kd - 1]) {
                outside = true;
            }
            let off = biv_basis_row(t, kd, order, v, basis);
            offs.push(off);
            d += 1;
        }
        if outside {
            out.push(0.0 / 0.0);
            p += 1;
            continue;
        }
        // The walk over the products of the basis values. The counter
        // idx holds the place of each axis in its k plus one values.
        var v: f64 = 0.0;
        var idx: i64[>..] = [];
        d = 0;
        while d < s.ndim { idx.push(0); d += 1; }
        var walking = true;
        while walking {
            var ci: i64 = 0;
            var prod: f64 = 1.0;
            d = 0;
            while d < s.ndim {
                ci += (offs[d] + idx[d]) * stride[d];
                prod *= basis[d * ((s.buf[d] as i64) + 1) + idx[d]];
                d += 1;
            }
            v += s.buf[2 * s.ndim + s.kn_len + ci] * prod;
            // One step of the counter. The walk ends when the first
            // axis runs over its values.
            d = s.ndim - 1;
            while d >= 0 {
                idx[d] += 1;
                if idx[d] <= s.buf[d] as i64 { break; }
                idx[d] = 0;
                d -= 1;
            }
            if d < 0 { walking = false; }
        }
        out.push(v);
        p += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// NdPoly holds the count of the axes, the count of the breakpoints,
// and one buffer. The buffer holds the degrees first, then the counts
// of the breakpoints, then the breakpoints of the axes, then the
// coefficients of the pieces. The struct holds one resizable field,
// so the values share one buffer.
struct NdPoly { ndim: i64, xb_len: i64, buf: f64[>..] }

fn ndpoly_empty() -> NdPoly {
    return NdPoly { ndim: 0, xb_len: 0, buf: [] };
}

// nd_poly holds the polynomial of the caller. The breakpoints of the
// axes come in one buffer, with the count of the breakpoints of each
// axis in nbs, and the degrees in ks.
fn nd_poly(x: f64[:], nbs: i64[:], ks: i64[:], c: f64[:]) -> NdPoly, Fault {
    let f = nd_check_input(nbs, ks, x, c, true, "nd_poly");
    if f.kind != F_NONE { return ndpoly_empty(), f; }
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < ks.len { buf.push(ks[i] as f64); i += 1; }
    i = 0;
    while i < nbs.len { buf.push(nbs[i] as f64); i += 1; }
    i = 0;
    while i < x.len { buf.push(x[i]); i += 1; }
    i = 0;
    while i < c.len { buf.push(c[i]); i += 1; }
    return NdPoly { ndim: nbs.len, xb_len: x.len, buf: buf }, no_fault();
}

// ndpoly_piece gives the interval of the axis d that holds the point.
// The intervals are half open, and the last interval is closed.
fn ndpoly_piece(s, d: i64, base: i64, v: f64) -> i64 {
    let nb = s.buf[s.ndim + d] as i64;
    var lo: i64 = 0;
    var hi: i64 = nb - 2;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if v >= s.buf[2 * s.ndim + base + mid] { lo = mid; } else { hi = mid; }
    }
    if v >= s.buf[2 * s.ndim + base + nb - 2] { return nb - 2; }
    return lo;
}

// ndpoly_eval gives the values of the polynomial at the points of xi.
// The buffer of xi holds ndim values for each point. The orders nu
// give the derivative of each axis. A point outside the breakpoints
// holds the value of the end pieces, or NaN when extrapolate is
// false.
fn ndpoly_eval(s, xi: f64[:], npoints: i64, nu: i64[:], extrapolate: bool) -> Arr, Fault {
    if s.ndim < 1 {
        return empty_arr(), fault(F_SHAPE, "nd_poly_eval", "the polynomial holds no axis");
    }
    if nu.len != 0 && nu.len != s.ndim {
        return empty_arr(), fault(F_SHAPE, "nd_poly_eval", "the orders need one value for each axis");
    }
    if xi.len != npoints * s.ndim {
        return empty_arr(), fault(F_SHAPE, "nd_poly_eval", "the points need the count of the axes for each point");
    }
    var i: i64 = 0;
    while i < xi.len {
        if !(xi[i] - xi[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "nd_poly_eval", "a point is not finite");
        }
        i += 1;
    }
    // The strides of the powers and of the pieces, and the start of
    // the breakpoints of each axis.
    var pstride: i64[>..] = [];
    var istride: i64[>..] = [];
    var boff: i64[>..] = [];
    var d: i64 = 0;
    while d < s.ndim {
        pstride.push(0);
        istride.push(0);
        boff.push(0);
        d += 1;
    }
    var npiece: i64 = 1;
    var sum: i64 = 0;
    d = 0;
    while d < s.ndim {
        boff[d] = sum;
        sum += s.buf[s.ndim + d] as i64;
        npiece *= (s.buf[s.ndim + d] as i64) - 1;
        d += 1;
    }
    var acc: i64 = 1;
    d = s.ndim - 1;
    while d >= 0 {
        pstride[d] = acc;
        acc *= (s.buf[d] as i64) + 1;
        d -= 1;
    }
    acc = 1;
    d = s.ndim - 1;
    while d >= 0 {
        istride[d] = acc;
        acc *= (s.buf[s.ndim + d] as i64) - 1;
        d -= 1;
    }
    var out: f64[>..] = [];
    var p: i64 = 0;
    while p < npoints {
        // The piece and the offset of each axis at this point.
        var pieces: i64[>..] = [];
        var offs: f64[>..] = [];
        var outside = false;
        d = 0;
        while d < s.ndim {
            let base = boff[d];
            let v = xi[p * s.ndim + d];
            var order: i64 = 0;
            if nu.len == s.ndim { order = nu[d]; }
            if order < 0 {
                return empty_arr(), fault(F_RANGE, "nd_poly_eval", "an order of the derivative is below zero");
            }
            if !extrapolate && (v < s.buf[2 * s.ndim + base] || v > s.buf[2 * s.ndim + base + (s.buf[s.ndim + d] as i64) - 1]) {
                outside = true;
            }
            let ip = ndpoly_piece(s, d, base, v);
            pieces.push(ip);
            offs.push(v - s.buf[2 * s.ndim + base + ip]);
            d += 1;
        }
        if outside {
            out.push(0.0 / 0.0);
            p += 1;
            continue;
        }
        // The walk over the powers of the axes.
        var v: f64 = 0.0;
        var idx: i64[>..] = [];
        d = 0;
        while d < s.ndim { idx.push(0); d += 1; }
        var walking = true;
        while walking {
            // The power of each axis must stand above the order of
            // the derivative.
            var good = true;
            var fac: f64 = 1.0;
            var pw: f64 = 1.0;
            d = 0;
            while d < s.ndim {
                var order: i64 = 0;
                if nu.len == s.ndim { order = nu[d]; }
                // The row of the coefficients holds the power in
                // reverse order, as SciPy holds it.
                let m = (s.buf[d] as i64) - idx[d];
                if m < order { good = false; }
                var f: f64 = 1.0;
                var q: i64 = 0;
                while q < order {
                    f = f * ((m - q) as f64);
                    q += 1;
                }
                fac = fac * f;
                var e = m - order;
                var tp: f64 = 1.0;
                q = 0;
                while q < e { tp = tp * offs[d]; q += 1; }
                pw = pw * tp;
                d += 1;
            }
            if good {
                var ci: i64 = 0;
                var ii: i64 = 0;
                d = 0;
                while d < s.ndim {
                    ci += idx[d] * pstride[d];
                    ii += pieces[d] * istride[d];
                    d += 1;
                }
                v += s.buf[2 * s.ndim + s.xb_len + ci * npiece + ii] * fac * pw;
            }
            // One step of the counter.
            d = s.ndim - 1;
            while d >= 0 {
                idx[d] += 1;
                if idx[d] <= s.buf[d] as i64 { break; }
                idx[d] = 0;
                d -= 1;
            }
            if d < 0 { walking = false; }
        }
        out.push(v);
        p += 1;
    }
    return line_arr(copy(out)), no_fault();
}
