// The B-spline of one variable: the object BSpl with the knots, the
// coefficients and the degree, the interpolation of data by a B-spline
// and the evaluation with derivatives. The names follow
// scipy.interpolate.BSpline, make_interp_spline, splrep and splev.
namespace scientific;

import mathshim;

// BSpl holds the degree, the count of knots and one buffer with the
// knots followed by the coefficients. The count of knots is the count
// of coefficients plus the degree plus one.
struct BSpl { k: i64, nt: i64, buf: f64[>..] }

fn bspl_empty() -> BSpl {
    return BSpl { k: 0, nt: 0, buf: [] };
}

// bs_make joins the knots and the coefficients into one spline.
fn bs_make(k: i64, t: f64[:], c: f64[:]) -> BSpl {
    var buf: f64[>..] = copy_slice(t);
    var i: i64 = 0;
    while i < c.len { buf.push(c[i]); i += 1; }
    return BSpl { k: k, nt: t.len, buf: buf };
}

fn bs_knots_of(b) -> f64[>..] {
    var t: f64[>..] = [];
    var i: i64 = 0;
    while i < b.nt { t.push(b.buf[i]); i += 1; }
    return t;
}

fn bs_coefs_of(b) -> f64[>..] {
    var c: f64[>..] = [];
    var i = b.nt;
    while i < b.buf.len { c.push(b.buf[i]); i += 1; }
    return c;
}

// bspline builds a B-spline from the knots, the coefficients and the
// degree. The knots must not fall, and a knot may repeat at most
// degree plus one times.
fn bspline(t: f64[:], c: f64[:], k: i64) -> BSpl, Fault {
    if k < 0 || k > 20 {
        return bspl_empty(), fault(F_RANGE, "bspline", "the degree is outside zero to twenty");
    }
    if c.len < k + 1 || t.len != c.len + k + 1 {
        return bspl_empty(), fault(F_SHAPE, "bspline", "the knots need the count of the coefficients plus the degree plus one");
    }
    var i: i64 = 0;
    while i < t.len {
        if !(t[i] - t[i] == 0.0) {
            return bspl_empty(), fault(F_VALUE, "bspline", "a knot is not finite");
        }
        if i > 0 && t[i] < t[i - 1] {
            return bspl_empty(), fault(F_VALUE, "bspline", "the knots fall");
        }
        i += 1;
    }
    i = 0;
    while i < c.len {
        if !(c[i] - c[i] == 0.0) {
            return bspl_empty(), fault(F_VALUE, "bspline", "a coefficient is not finite");
        }
        i += 1;
    }
    if !(t[k] < t[c.len]) {
        return bspl_empty(), fault(F_VALUE, "bspline", "the base interval holds no width");
    }
    return bs_make(k, t, c), no_fault();
}

// bs_derivative gives the coefficients and the knots of the derivative
// spline of one order lower.
fn bs_derivative(t: f64[:], c: f64[:], k: i64) -> f64[>..], f64[>..] {
    var nc: f64[>..] = [];
    var i: i64 = 0;
    while i + 1 < c.len {
        let d = t[i + k + 1] - t[i + 1];
        if d == 0.0 { nc.push(0.0); } else { nc.push((k * 1.0) * (c[i + 1] - c[i]) / d); }
        i += 1;
    }
    var nt: f64[>..] = [];
    i = 1;
    while i + 1 < t.len { nt.push(t[i]); i += 1; }
    return nt, nc;
}

// bs_value evaluates the spline of the degree k at a point by the
// scheme of de Boor. A point outside the base interval takes the end
// polynomial.
fn bs_value(t: f64[:], c: f64[:], k: i64, x: f64) -> f64 {
    let n = c.len;
    var i = k;
    while i < n - 1 && x >= t[i + 1] { i += 1; }
    var d: f64[>..] = [];
    var j: i64 = 0;
    while j <= k { d.push(c[j + i - k]); j += 1; }
    var r: i64 = 1;
    while r <= k {
        j = k;
        while j >= r {
            let lo = t[j + i - k];
            let den = t[j + 1 + i - r] - lo;
            var a: f64 = 0.0;
            if den != 0.0 { a = (x - lo) / den; }
            d[j] = (1.0 - a) * d[j - 1] + a * d[j];
            j -= 1;
        }
        r += 1;
    }
    return d[k];
}

// bspline_eval gives the nu-th derivative of the spline at the points.
// A derivative above the degree is zero.
fn bspline_eval(b, xs: f64[:], nu: i64 = 0) -> Arr, Fault {
    if nu < 0 {
        return empty_arr(), fault(F_RANGE, "bspline_eval", "the derivative order is below zero");
    }
    if xs.len < 1 {
        return empty_arr(), fault(F_SHAPE, "bspline_eval", "the list of points holds no value");
    }
    var t = bs_knots_of(b);
    var c = bs_coefs_of(b);
    var k = b.k;
    var o: i64 = 0;
    while o < nu && k > 0 {
        let nt, nc = bs_derivative(t, c, k);
        t = copy_slice(nt);
        c = copy_slice(nc);
        k -= 1;
        o += 1;
    }
    var out: f64[>..] = [];
    var j: i64 = 0;
    while j < xs.len {
        if xs[j] != xs[j] { out.push(xs[j]); }
        else if o < nu { out.push(0.0); }
        else { out.push(bs_value(t, c, k, xs[j])); }
        j += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// splev follows scipy.interpolate.splev: the points first, then the
// spline and the derivative order.
fn splev(xs: f64[:], b, der: i64 = 0) -> Arr, Fault {
    return bspline_eval(b, xs, der);
}

// bs_knots gives the knot vector of the interpolation by the rule of
// the not-a-knot condition.
fn bs_knots(x: f64[:], k: i64) -> f64[>..] {
    var base: f64[>..] = [];
    var i: i64 = 0;
    if k % 2 == 1 {
        while i < x.len { base.push(x[i]); i += 1; }
    } else {
        while i + 1 < x.len { base.push((x[i + 1] + x[i]) / 2.0); i += 1; }
    }
    let k2 = (k + 1) / 2;
    var t: f64[>..] = [];
    i = 0;
    while i <= k { t.push(x[0]); i += 1; }
    i = k2;
    while i < base.len - k2 { t.push(base[i]); i += 1; }
    i = 0;
    while i <= k { t.push(x[x.len - 1]); i += 1; }
    return t;
}

// make_interp_spline builds the B-spline of degree k that passes
// through the points. The knots follow the rule of not-a-knot.
fn make_interp_spline(x: f64[:], y: f64[:], k: i64 = 3) -> BSpl, Fault {
    if k < 0 || k > 20 {
        return bspl_empty(), fault(F_RANGE, "make_interp_spline", "the degree is outside zero to twenty");
    }
    if x.len != y.len || x.len < k + 1 {
        return bspl_empty(), fault(F_SHAPE, "make_interp_spline", "the data need equal counts and at least degree plus one points");
    }
    var i: i64 = 0;
    while i < x.len {
        if !(x[i] - x[i] == 0.0) || !(y[i] - y[i] == 0.0) {
            return bspl_empty(), fault(F_VALUE, "make_interp_spline", "a point or a value is not finite");
        }
        if i > 0 && !(x[i] > x[i - 1]) {
            return bspl_empty(), fault(F_VALUE, "make_interp_spline", "the points do not rise strictly");
        }
        i += 1;
    }
    let n = x.len;
    if k == 0 {
        // Steps: the knots lie at the midpoints.
        var t0: f64[>..] = [x[0]];
        i = 0;
        while i + 1 < n { t0.push((x[i] + x[i + 1]) / 2.0); i += 1; }
        t0.push(x[n - 1]);
        t0[n] = x[n - 1];
        return bs_make(0, t0, y), no_fault();
    }
    let t = bs_knots(x, k);
    // The matrix of the collocation. A row holds the values of the
    // basis at one point.
    var a: f64[>..] = [];
    i = 0;
    while i < n * n { a.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            var unit: f64[>..] = [];
            var q: i64 = 0;
            while q < n { if q == j { unit.push(1.0); } else { unit.push(0.0); } q += 1; }
            let v = bs_value(t, unit, k, x[i]);
            a[i * n + j] = v;
            j += 1;
        }
        i += 1;
    }
    var sa: i64[..8] = [];
    sa.push(n);
    sa.push(n);
    var sb: i64[..8] = [];
    sb.push(n);
    let sol, fs = solve(Arr { ndim: 2, shape: sa, data: copy(a) }, Arr { ndim: 1, shape: sb, data: copy_slice(y) });
    if fs.kind != F_NONE {
        return bspl_empty(), fault(F_VALUE, "make_interp_spline", "the collocation matrix holds no solution");
    }
    return bs_make(k, t, sol.data), no_fault();
}

// bs_deriv_value gives the derivative of a spline at a point. A
// derivative above the degree is zero.
fn bs_deriv_value(t: f64[:], c: f64[:], k: i64, order: i64, x: f64) -> f64 {
    var tt: f64[>..] = copy_slice(t);
    var cc: f64[>..] = copy_slice(c);
    var kk = k;
    var o: i64 = 0;
    while o < order && kk > 0 {
        let nt, nc = bs_derivative(tt, cc, kk);
        tt = copy_slice(nt);
        cc = copy_slice(nc);
        kk -= 1;
        o += 1;
    }
    if o < order { return 0.0; }
    return bs_value(tt, cc, kk, x);
}

// make_interp_spline_bc builds the B-spline of degree k that passes
// through the points with derivatives at the ends. The lists hold the
// orders and the values of the left end and of the right end. The two
// counts add up to k minus one, as bc_type of SciPy needs. The knots
// repeat the first and the last point k times.
fn make_interp_spline_bc(x: f64[:], y: f64[:], k: i64, left_orders: i64[:], left_values: f64[:], right_orders: i64[:], right_values: f64[:]) -> BSpl, Fault {
    if k < 1 || k > 20 {
        return bspl_empty(), fault(F_RANGE, "make_interp_spline_bc", "the degree is outside one to twenty");
    }
    let n = x.len;
    let kl = left_orders.len;
    let kr = right_orders.len;
    if left_values.len != kl || right_values.len != kr || kl + kr != k - 1 {
        return bspl_empty(), fault(F_SHAPE, "make_interp_spline_bc", "the end derivatives must number the degree minus one, with one value each");
    }
    if x.len != y.len || n < 2 {
        return bspl_empty(), fault(F_SHAPE, "make_interp_spline_bc", "the data need equal counts and at least two points");
    }
    var i: i64 = 0;
    while i < n {
        if !(x[i] - x[i] == 0.0) || !(y[i] - y[i] == 0.0) {
            return bspl_empty(), fault(F_VALUE, "make_interp_spline_bc", "a point or a value is not finite");
        }
        if i > 0 && !(x[i] > x[i - 1]) {
            return bspl_empty(), fault(F_VALUE, "make_interp_spline_bc", "the points do not rise strictly");
        }
        i += 1;
    }
    i = 0;
    while i < kl {
        if left_orders[i] < 1 || left_orders[i] > k || !(left_values[i] - left_values[i] == 0.0) {
            return bspl_empty(), fault(F_VALUE, "make_interp_spline_bc", "an end order is outside one to the degree, or a value is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < kr {
        if right_orders[i] < 1 || right_orders[i] > k || !(right_values[i] - right_values[i] == 0.0) {
            return bspl_empty(), fault(F_VALUE, "make_interp_spline_bc", "an end order is outside one to the degree, or a value is not finite");
        }
        i += 1;
    }
    var t: f64[>..] = [];
    i = 0;
    while i < k { t.push(x[0]); i += 1; }
    i = 0;
    while i < n { t.push(x[i]); i += 1; }
    i = 0;
    while i < k { t.push(x[n - 1]); i += 1; }
    let m = n + kl + kr;
    var a: f64[>..] = [];
    var b: f64[>..] = [];
    i = 0;
    while i < m * m { a.push(0.0); i += 1; }
    var row: i64 = 0;
    var r: i64 = 0;
    while r < m {
        var order: i64 = 0;
        var at: f64 = 0.0;
        if r < kl { order = left_orders[r]; at = x[0]; b.push(left_values[r]); }
        else if r < kl + n { at = x[r - kl]; b.push(y[r - kl]); }
        else { order = right_orders[r - kl - n]; at = x[n - 1]; b.push(right_values[r - kl - n]); }
        var j: i64 = 0;
        while j < m {
            var unit: f64[>..] = [];
            var q: i64 = 0;
            while q < m { if q == j { unit.push(1.0); } else { unit.push(0.0); } q += 1; }
            a[r * m + j] = bs_deriv_value(t, unit, k, order, at);
            j += 1;
        }
        r += 1;
    }
    var sa: i64[..8] = [];
    sa.push(m);
    sa.push(m);
    var sb: i64[..8] = [];
    sb.push(m);
    let sol, fs = solve(Arr { ndim: 2, shape: sa, data: copy(a) }, Arr { ndim: 1, shape: sb, data: copy(b) });
    if fs.kind != F_NONE {
        return bspl_empty(), fault(F_VALUE, "make_interp_spline_bc", "the collocation matrix holds no solution");
    }
    return bs_make(k, t, sol.data), no_fault();
}

// bs_design_row gives the values of the k plus one basis functions
// that hold at a point, and the column of the first one. The span of
// the knot vector holds the point. The basis functions follow the
// scheme of the knot insertion.
fn bs_design_row(t: f64[:], k: i64, x: f64) -> i64, f64[>..] {
    let nt = t.len;
    var s = k;
    if x >= t[nt - k - 1] { s = nt - k - 2; }
    else {
        while s < nt - k - 2 && x >= t[s + 1] { s += 1; }
    }
    var n: f64[>..] = [];
    var left: f64[>..] = [];
    var right: f64[>..] = [];
    var i: i64 = 0;
    while i <= k { n.push(0.0); left.push(0.0); right.push(0.0); i += 1; }
    n[0] = 1.0;
    var j: i64 = 1;
    while j <= k {
        left[j] = x - t[s + 1 - j];
        right[j] = t[s + j] - x;
        var saved: f64 = 0.0;
        var r: i64 = 0;
        while r < j {
            let d = right[r + 1] + left[j - r];
            var temp: f64 = 0.0;
            if d != 0.0 { temp = n[r] / d; }
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
            r += 1;
        }
        n[j] = saved;
        j += 1;
    }
    return s - k, n;
}

// bspline_design_matrix gives the matrix of the basis functions of a
// B-spline at the points. Each row holds the k plus one values of the
// basis at one point, and the other values are zero. The matrix is
// sparse, in the type Csr. The extrapolation is "no", "yes" or
// "periodic".
fn bspline_design_matrix(x: f64[:], t: f64[:], k: i64, extrapolate: const u8[:] = "no") -> Csr, Fault {
    if k < 0 || k > 20 {
        return csr_empty(), fault(F_RANGE, "bspline_design_matrix", "the degree is outside zero to twenty");
    }
    if extrapolate != "no" && extrapolate != "yes" && extrapolate != "periodic" {
        return csr_empty(), fault(F_VALUE, "bspline_design_matrix", "the extrapolation holds an unknown name");
    }
    if t.len < 2 * k + 2 {
        return csr_empty(), fault(F_SHAPE, "bspline_design_matrix", "the knots hold too few values for the degree");
    }
    if x.len < 1 {
        return csr_empty(), fault(F_SHAPE, "bspline_design_matrix", "the list of points holds no value");
    }
    var i: i64 = 0;
    while i < t.len {
        if !(t[i] - t[i] == 0.0) {
            return csr_empty(), fault(F_VALUE, "bspline_design_matrix", "a knot is not finite");
        }
        if i > 0 && t[i] < t[i - 1] {
            return csr_empty(), fault(F_VALUE, "bspline_design_matrix", "the knots fall");
        }
        i += 1;
    }
    i = 0;
    while i < x.len {
        if !(x[i] - x[i] == 0.0) {
            return csr_empty(), fault(F_VALUE, "bspline_design_matrix", "a point is not finite");
        }
        i += 1;
    }
    let nt = t.len;
    let n = nt - k - 1;
    let lo = t[k];
    let hi = t[n];
    if extrapolate == "no" {
        i = 0;
        while i < x.len {
            if x[i] < lo || x[i] > hi {
                return csr_empty(), fault(F_RANGE, "bspline_design_matrix", "a point stands outside the knots");
            }
            i += 1;
        }
    }
    var pts: f64[>..] = [];
    i = 0;
    while i < x.len {
        if extrapolate == "periodic" {
            pts.push(lo + (x[i] - lo) - mathshim::floor_of((x[i] - lo) / (hi - lo)) * (hi - lo));
        } else {
            pts.push(x[i]);
        }
        i += 1;
    }
    var ptr: f64[>..] = [];
    var idx: f64[>..] = [];
    var val: f64[>..] = [];
    i = 0;
    while i < pts.len {
        let off, row = bs_design_row(t, k, pts[i]);
        var j: i64 = 0;
        while j <= k {
            idx.push(off + j);
            val.push(row[j]);
            j += 1;
        }
        ptr.push(i * (k + 1));
        i += 1;
    }
    ptr.push(pts.len * (k + 1));
    return csr_from_parts(pts.len, n, ptr, idx, val), no_fault();
}

// splrep builds the B-spline of the data. Only the interpolation
// with s equal to zero holds. A smoothing factor above zero is a
// fault.
fn splrep(x: f64[:], y: f64[:], k: i64 = 3, s: f64 = 0.0) -> BSpl, Fault {
    if s != 0.0 {
        return bspl_empty(), fault(F_VALUE, "splrep", "a smoothing factor above zero is not supported");
    }
    if k < 1 || k > 5 {
        return bspl_empty(), fault(F_RANGE, "splrep", "the degree is outside one to five");
    }
    return make_interp_spline(x, y, k);
}
