// The algebra of the B-spline of one variable that scipy.interpolate
// holds on the form tck: the derivative splder, the antiderivative
// splantider, the definite integral splint, every derivative spalde,
// the roots sproot and the fit with least squares make_lsq_spline.
// The spline is the type BSpl of the package, with the knots and the
// coefficients of the rule of FITPACK.
namespace scientific;

import mathshim;

// splder gives the derivative of the spline of the order n. A negative
// order gives the antiderivative of the size of the order. An order
// above the degree is a fault, and a spline with a repeated knot
// inside holds no derivative of that order.
recursive fn splder(b: BSpl&, n: i64 = 1) -> BSpl, Fault {
    if n < 0 { return splantider(b, -n); }
    if n == 0 { return copy(b), no_fault(); }
    if n > b.k {
        return bspl_empty(), fault(F_VALUE, "splder", "the order of the derivative stands above the degree of the spline");
    }
    var t = bs_knots_of(b);
    var c = bs_coefs_of(b);
    var k = b.k;
    var turn: i64 = 0;
    while turn < n {
        var i: i64 = 0;
        while i + 1 < c.len {
            if t[i + k + 1] - t[i + 1] == 0.0 {
                return bspl_empty(), fault(F_VALUE, "splder", "the spline holds a repeated knot and is not differentiable to the order");
            }
            i += 1;
        }
        let nt, nc = bs_derivative(t, c, k);
        t = copy_slice(nt);
        c = copy_slice(nc);
        k -= 1;
        turn += 1;
    }
    return bs_make(k, t, c), no_fault();
}

// splantider gives the antiderivative of the spline of the order n. A
// negative order gives the derivative of the size of the order. The
// coefficients follow the rule of FITPACK: the sum of the products of
// the coefficients and the knot steps, with a zero in front and one
// knot added on each end for every turn.
recursive fn splantider(b: BSpl&, n: i64 = 1) -> BSpl, Fault {
    if n < 0 { return splder(b, -n); }
    if n == 0 { return copy(b), no_fault(); }
    var t = bs_knots_of(b);
    var c = bs_coefs_of(b);
    var k = b.k;
    var turn: i64 = 0;
    while turn < n {
        var dt: f64[>..] = [];
        var i: i64 = 0;
        while i < c.len {
            dt.push(t[k + 1 + i] - t[i]);
            i += 1;
        }
        // The cumulative sum of the products, over the degree plus one.
        var cc: f64[>..] = [];
        var run: f64 = 0.0;
        i = 0;
        while i < c.len {
            run += c[i] * dt[i];
            cc.push(run / (k + 1));
            i += 1;
        }
        var nc: f64[>..] = [0.0];
        i = 0;
        while i < cc.len { nc.push(cc[i]); i += 1; }
        var nt: f64[>..] = [t[0]];
        i = 0;
        while i < t.len { nt.push(t[i]); i += 1; }
        nt.push(t[t.len - 1]);
        t = copy_slice(nt);
        c = copy_slice(nc);
        k += 1;
        turn += 1;
    }
    return bs_make(k, t, c), no_fault();
}

// splint gives the integral of the spline between a and b. The rule
// is the value of the antiderivative at the upper end less the value
// at the lower end, as the splint of FITPACK computes it.
fn splint(a: f64, b: f64, s) -> f64, Fault {
    let fc = bs_check(s, "splint");
    if fc.kind != F_NONE { return 0.0, fc; }
    if !(a - a == 0.0) || !(b - b == 0.0) {
        return 0.0, fault(F_VALUE, "splint", "an end of the interval is not finite");
    }
    if b < a {
        return 0.0, fault(F_VALUE, "splint", "the upper end stands below the lower end");
    }
    var st = bs_knots_of(s);
    let sc = bs_coefs_of(s);
    let lo = st[s.k];
    let hi = st[sc.len];
    if a < lo || a > hi || b < lo || b > hi {
        return 0.0, fault(F_RANGE, "splint", "an end of the interval stands outside the base interval of the knots");
    }
    let anti, f = splantider(s, 1);
    if f.kind != F_NONE { return 0.0, f; }
    var t: f64[>..] = [];
    var c: f64[>..] = [];
    var i: i64 = 0;
    while i < anti.nt { t.push(anti.buf[i]); i += 1; }
    i = anti.nt;
    while i < anti.buf.len { c.push(anti.buf[i]); i += 1; }
    let up = bs_value(t, c, anti.k, b);
    let dn = bs_value(t, c, anti.k, a);
    return up - dn, no_fault();
}

// spalde gives every derivative of the spline, of the order zero to
// the degree, at the points. The result holds one row of degree plus
// one values for each point. A point stands inside the base interval
// of the knots, as the spalde of FITPACK needs.
fn spalde(s, xs: f64[:]) -> Arr, Fault {
    let fc = bs_check(s, "spalde");
    if fc.kind != F_NONE { return empty_arr(), fc; }
    if xs.len < 1 {
        return empty_arr(), fault(F_SHAPE, "spalde", "the list of points holds no value");
    }
    let t = bs_knots_of(s);
    let c = bs_coefs_of(s);
    let k = s.k;
    let lo = t[k];
    let hi = t[c.len];
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < xs.len {
        if !(xs[i] - xs[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "spalde", "a point is not finite");
        }
        if xs[i] < lo || xs[i] > hi {
            return empty_arr(), fault(F_RANGE, "spalde", "a point stands outside the base interval of the knots");
        }
        var order: i64 = 0;
        while order <= k {
            out.push(bs_deriv_value(t, c, k, order, xs[i]));
            order += 1;
        }
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// sproot gives the roots of the spline of degree three. Every piece
// between two knots turns into its polynomial in the form of
// Bernstein, and the roots of the power form of each piece come from
// the solver of the package. A root counts when it stands inside the
// base interval, and the roots come out in rising order.
fn sproot(s) -> Arr, Fault {
    let fc = bs_check(s, "sproot");
    if fc.kind != F_NONE { return empty_arr(), fc; }
    if s.k != 3 {
        return empty_arr(), fault(F_VALUE, "sproot", "sproot works only for a spline of degree three");
    }
    let t = bs_knots_of(s);
    let c = bs_coefs_of(s);
    let nt = t.len;
    if nt < 8 {
        return empty_arr(), fault(F_VALUE, "sproot", "the knots hold fewer than eight values");
    }
    var i: i64 = 3;
    while i + 4 < nt {
        if t[i + 1] <= t[i] {
            return empty_arr(), fault(F_VALUE, "sproot", "a knot repeats inside the base interval");
        }
        i += 1;
    }
    let lo = t[3];
    let hi = t[nt - 4];
    var out: f64[>..] = [];
    i = 3;
    while i < nt - 4 {
        let a = t[i];
        let b = t[i + 1];
        if b > a {
            let h = b - a;
            let mid = (a + b) / 2.0;
            let p0 = bs_value(t, c, 3, a);
            let d1 = bs_deriv_value(t, c, 3, 1, a);
            let d2 = bs_deriv_value(t, c, 3, 2, a);
            let d3 = bs_deriv_value(t, c, 3, 3, mid);
            // The coefficients of the piece in the form of Bernstein,
            // from the Taylor form at the left end.
            let b0 = p0;
            let b1 = p0 + h / 3.0 * d1;
            let b2 = p0 + 2.0 * h / 3.0 * d1 + h * h / 6.0 * d2;
            let b3 = p0 + h * d1 + h * h / 2.0 * d2 + h * h * h / 6.0 * d3;
            // The power form of the piece on the unit interval.
            let p3 = b3 - 3.0 * b2 + 3.0 * b1 - b0;
            let p2 = 3.0 * (b2 - 2.0 * b1 + b0);
            let p1 = 3.0 * (b1 - b0);
            let p0c = b0;
            if p3 == 0.0 && p2 == 0.0 && p1 == 0.0 && p0c == 0.0 {
                i += 1;
                continue;
            }
            var co: f64[>..] = [p3, p2, p1, p0c];
            let re, im, f = roots(co);
            if f.kind != F_NONE {
                return empty_arr(), fault(F_VALUE, "sproot", "the polynomial of a piece holds no root count");
            }
            var q: i64 = 0;
            while q < re.data.len {
                let tol = 0.000000001 * (1.0 + mathshim::abs_of(p3) + mathshim::abs_of(p2) + mathshim::abs_of(p1) + mathshim::abs_of(p0c));
                if mathshim::abs_of(im.data[q]) <= tol {
                    let u = re.data[q];
                    if u >= -0.000000001 && u <= 1.000000001 {
                        let x = a + (b - a) * u;
                        if x >= lo - 0.000000001 && x <= hi + 0.000000001 {
                            out.push(x);
                        }
                    }
                }
                q += 1;
            }
        }
        i += 1;
    }
    // The roots come out in rising order, and a root that two pieces
    // share counts once.
    let ordered, sf = sort(line_arr(copy(out)));
    if sf.kind != F_NONE { return empty_arr(), sf; }
    var ded: f64[>..] = [];
    i = 0;
    while i < ordered.data.len {
        if ded.len == 0 || mathshim::abs_of(ordered.data[i] - ded[ded.len - 1]) > 0.00000001 {
            ded.push(ordered.data[i]);
        }
        i += 1;
    }
    return line_arr(copy(ded)), no_fault();
}

// make_lsq_spline follows scipy.interpolate.make_lsq_spline: the
// spline of degree k with the knots t that holds the weighted sum of
// the squared distances to the points at its smallest. The weights
// hold one when the list stays empty. The method solves the normal
// equations of the matrix of the basis, as the path of the norm
// equations of SciPy does. The points must rise strictly and stand
// inside the base interval of the knots.
fn make_lsq_spline(x: f64[:], y: f64[:], t: f64[:], k: i64 = 3, w: f64[:] = []) -> BSpl, Fault {
    if k < 0 || k > 20 {
        return bspl_empty(), fault(F_RANGE, "make_lsq_spline", "the degree is outside zero to twenty");
    }
    if t.len < 2 * k + 2 {
        return bspl_empty(), fault(F_SHAPE, "make_lsq_spline", "the knots hold too few values for the degree");
    }
    if x.len < k + 1 || x.len != y.len {
        return bspl_empty(), fault(F_SHAPE, "make_lsq_spline", "the data need equal counts of at least degree plus one points");
    }
    if w.len != 0 && w.len != x.len {
        return bspl_empty(), fault(F_SHAPE, "make_lsq_spline", "the weights need the count of the points or no value");
    }
    var i: i64 = 0;
    while i < t.len {
        if !(t[i] - t[i] == 0.0) {
            return bspl_empty(), fault(F_VALUE, "make_lsq_spline", "a knot is not finite");
        }
        if i > 0 && t[i] < t[i - 1] {
            return bspl_empty(), fault(F_VALUE, "make_lsq_spline", "the knots fall");
        }
        i += 1;
    }
    i = 0;
    while i < x.len {
        if !(x[i] - x[i] == 0.0) || !(y[i] - y[i] == 0.0) {
            return bspl_empty(), fault(F_VALUE, "make_lsq_spline", "a point or a value is not finite");
        }
        if i > 0 && !(x[i] > x[i - 1]) {
            return bspl_empty(), fault(F_VALUE, "make_lsq_spline", "the points do not rise strictly");
        }
        if k > 0 && (x[i] < t[k] || x[i] > t[t.len - k]) {
            return bspl_empty(), fault(F_VALUE, "make_lsq_spline", "a point stands outside the base interval of the knots");
        }
        i += 1;
    }
    let n = t.len - k - 1;
    // The matrix of the basis. A row holds the k plus one values of
    // the basis at one point, and the other values are zero.
    var a: f64[>..] = [];
    i = 0;
    while i < x.len * n { a.push(0.0); i += 1; }
    i = 0;
    while i < x.len {
        let off, row = bs_design_row(t, k, x[i]);
        var j: i64 = 0;
        while j <= k {
            var v = row[j];
            if w.len > 0 { v = v * w[i]; }
            a[i * n + off + j] = v;
            j += 1;
        }
        i += 1;
    }
    // The normal equations of the scaled matrix. The scale of a row is
    // the weight of the point, and the sum holds the square of the
    // weight on both sides, as the fit of SciPy does.
    var normal: f64[>..] = [];
    i = 0;
    while i < n * n { normal.push(0.0); i += 1; }
    var rhs: f64[>..] = [];
    i = 0;
    while i < n { rhs.push(0.0); i += 1; }
    var r: i64 = 0;
    while r < x.len {
        var wi: f64 = 1.0;
        if w.len > 0 { wi = w[r]; }
        var j: i64 = 0;
        while j < n {
            let av = a[r * n + j];
            if av != 0.0 {
                var q: i64 = 0;
                while q < n {
                    normal[j * n + q] += av * a[r * n + q];
                    q += 1;
                }
                rhs[j] += av * wi * y[r];
            }
            j += 1;
        }
        r += 1;
    }
    var sa: i64[..8] = [];
    sa.push(n);
    sa.push(n);
    var sb: i64[..8] = [];
    sb.push(n);
    let sol, f = solve(Arr { ndim: 2, shape: sa, data: copy(normal) }, Arr { ndim: 1, shape: sb, data: copy(rhs) });
    if f.kind != F_NONE {
        return bspl_empty(), fault(F_VALUE, "make_lsq_spline", "the system of the normal equations holds no solution");
    }
    return bs_make(k, t, sol.data), no_fault();
}
