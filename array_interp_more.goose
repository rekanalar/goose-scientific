// The polynomial and rational interpolation of scipy.interpolate that
// the other files of the package do not hold: the interpolators of
// Barycentric, Krogh and Floater and Hormann, the polynomial of
// Lagrange, the approximation of Pade, the Taylor polynomial of a
// function and the one call form of PCHIP. The names follow
// scipy.interpolate.
namespace scientific;

import mathshim;

// barycentric_weights gives the weights of the rule of Berrut and
// Trefethen. Each weight is the reciprocal of the product of the
// distances to the other nodes. Every distance takes the factor
// 4 over the width of the node range, the capacity scaling that
// SciPy uses, so the products stay small for a wide node range.
fn barycentric_weights(xi: f64[:], tag: const u8[:]) -> f64[>..], Fault {
    let n = xi.len;
    var lo: f64 = xi[0];
    var hi: f64 = xi[0];
    var i: i64 = 0;
    while i < n {
        if !(xi[i] - xi[i] == 0.0) {
            return [], fault(F_VALUE, tag, "a node is not finite");
        }
        var j: i64 = 0;
        while j < i {
            if xi[j] == xi[i] {
                return [], fault(F_VALUE, tag, "the nodes hold a repeated value");
            }
            j += 1;
        }
        if xi[i] < lo { lo = xi[i]; }
        if xi[i] > hi { hi = xi[i]; }
        i += 1;
    }
    var inv_cap = 1.0;
    if hi > lo { inv_cap = 4.0 / (hi - lo); }
    var w: f64[>..] = [];
    i = 0;
    while i < n {
        var p: f64 = 1.0;
        var j: i64 = 0;
        while j < n {
            if j != i { p = p * (inv_cap * (xi[i] - xi[j])); }
            j += 1;
        }
        if p == 0.0 {
            return [], fault(F_VALUE, tag, "a node distance holds no width");
        }
        w.push(1.0 / p);
        i += 1;
    }
    return w, no_fault();
}

// barycentric_at evaluates the rational form at one point. The value
// at a node is the data value of that node.
fn barycentric_at(w: f64[:], xi: f64[:], y: f64[:], x: f64) -> f64 {
    var i: i64 = 0;
    while i < xi.len {
        if x == xi[i] { return y[i]; }
        i += 1;
    }
    var num: f64 = 0.0;
    var den: f64 = 0.0;
    i = 0;
    while i < xi.len {
        let c = w[i] / (x - xi[i]);
        num += c * y[i];
        den += c;
        i += 1;
    }
    return num / den;
}

// barycentric_diff gives the values that the derivative interpolator
// holds at the nodes. The matrix follows the equations 9.4 and 9.5 of
// Berrut and Trefethen 2004. One application gives the first
// derivative values, and a second one gives the second derivative
// values, and so on.
fn barycentric_diff(w: f64[:], xi: f64[:], y: f64[:]) -> f64[>..] {
    let n = xi.len;
    // The row i holds the weights of the derivative. The diagonal
    // holds the negative sum of the other weights of the row, so that
    // every column of the matrix sums to zero.
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var t: f64 = 0.0;
        var j: i64 = 0;
        while j < n {
            if j != i {
                t += (w[j] / ((xi[i] - xi[j]) * w[i])) * y[j];
            }
            j += 1;
        }
        var s: f64 = 0.0;
        j = 0;
        while j < n {
            if j != i { s += w[j] / ((xi[i] - xi[j]) * w[i]); }
            j += 1;
        }
        out.push(t - s * y[i]);
        i += 1;
    }
    return out;
}

// barycentric_interpolate follows scipy.interpolate.barycentric_interpolate:
// the polynomial that passes through the nodes, at the points, with
// the derivative of the order der. The order zero is the value. The
// nodes must be distinct.
fn barycentric_interpolate(xi: f64[:], y: f64[:], xs: f64[:], der: i64 = 0) -> Arr, Fault {
    if xi.len < 1 || xi.len != y.len {
        return empty_arr(), fault(F_SHAPE, "barycentric_interpolate", "the nodes and the values need equal counts of at least one");
    }
    if xs.len < 1 {
        return empty_arr(), fault(F_SHAPE, "barycentric_interpolate", "the list of points holds no value");
    }
    if der < 0 {
        return empty_arr(), fault(F_RANGE, "barycentric_interpolate", "the derivative order is below zero");
    }
    let w, f = barycentric_weights(xi, "barycentric_interpolate");
    if f.kind != F_NONE { return empty_arr(), f; }
    var vals: f64[>..] = copy_slice(y);
    var turn: i64 = 0;
    while turn < der {
        let next = barycentric_diff(w, xi, vals);
        vals = copy(next);
        turn += 1;
    }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < xs.len {
        out.push(barycentric_at(w, xi, vals, xs[i]));
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// krogh_coef gives the coefficients of the Newton form of the
// polynomial of Krogh. A node that repeats holds a derivative: the
// values of one run of equal nodes hold the function value and the
// derivative values in order. The method is the recursion of Krogh
// 1970, in the form that SciPy uses.
fn krogh_coef(xi: f64[:], y: f64[:], tag: const u8[:]) -> f64[>..], Fault {
    let n = xi.len;
    var i: i64 = 0;
    while i < n {
        if !(xi[i] - xi[i] == 0.0) || !(y[i] - y[i] == 0.0) {
            return [], fault(F_VALUE, tag, "a node or a value is not finite");
        }
        if i > 0 && xi[i] < xi[i - 1] {
            return [], fault(F_VALUE, tag, "the nodes fall");
        }
        i += 1;
    }
    var fact: f64[>..] = [1.0];
    i = 1;
    while i <= n { fact.push(fact[i - 1] * (i as f64)); i += 1; }
    var c: f64[>..] = [];
    i = 0;
    while i <= n { c.push(0.0); i += 1; }
    var vk: f64[>..] = [];
    i = 0;
    while i < n { vk.push(0.0); i += 1; }
    c[0] = y[0];
    var k: i64 = 1;
    while k < n {
        var s: i64 = 0;
        while s <= k && xi[k - s] == xi[k] { s += 1; }
        s -= 1;
        vk[0] = y[k] / fact[s];
        i = 0;
        while i < k - s {
            if xi[i] == xi[k] {
                return [], fault(F_VALUE, tag, "a run of equal nodes is longer than the values of the derivatives");
            }
            if s == 0 {
                vk[i + 1] = (c[i] - vk[i]) / (xi[i] - xi[k]);
            } else {
                vk[i + 1] = (vk[i + 1] - vk[i]) / (xi[i] - xi[k]);
            }
            i += 1;
        }
        c[k] = vk[k - s];
        k += 1;
    }
    var out: f64[>..] = [];
    i = 0;
    while i < n { out.push(c[i]); i += 1; }
    return out, no_fault();
}

// krogh_derivs_at gives the value and the first der derivatives of the
// Newton form with the coefficients c at one point. The recursion
// walks the terms from the last one to the first one, and each step
// takes the derivative of the part that it built.
fn krogh_derivs_at(c: f64[:], xi: f64[:], x: f64, der: i64) -> f64[>..] {
    let n = c.len;
    var f: f64[>..] = [];
    var i: i64 = 0;
    while i <= der { f.push(0.0); i += 1; }
    f[0] = c[n - 1];
    i = n - 2;
    while i >= 0 {
        var g: f64[>..] = [];
        var m: i64 = 0;
        while m <= der { g.push(0.0); m += 1; }
        g[0] = c[i] + (x - xi[i]) * f[0];
        m = 1;
        while m <= der {
            g[m] = (m as f64) * f[m - 1] + (x - xi[i]) * f[m];
            m += 1;
        }
        f = copy(g);
        i -= 1;
    }
    return f;
}

// krogh_interpolate follows scipy.interpolate.krogh_interpolate: the
// polynomial of Krogh through the nodes, at the points, with every
// derivative of the order zero to der. A run of equal nodes holds the
// function value and the derivative values in order. The result holds
// der plus one value for each point, the values of one point next to
// each other.
fn krogh_interpolate(xi: f64[:], y: f64[:], xs: f64[:], der: i64 = 0) -> Arr, Fault {
    if xi.len < 1 || xi.len != y.len {
        return empty_arr(), fault(F_SHAPE, "krogh_interpolate", "the nodes and the values need equal counts of at least one");
    }
    if xs.len < 1 {
        return empty_arr(), fault(F_SHAPE, "krogh_interpolate", "the list of points holds no value");
    }
    if der < 0 || der >= xi.len {
        return empty_arr(), fault(F_RANGE, "krogh_interpolate", "the derivative order stands outside zero to the node count");
    }
    let c, f = krogh_coef(xi, y, "krogh_interpolate");
    if f.kind != F_NONE { return empty_arr(), f; }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < xs.len {
        let d = krogh_derivs_at(c, xi, xs[i], der);
        var m: i64 = 0;
        while m <= der { out.push(d[m]); m += 1; }
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// floater_hormann follows the weights of equation 18 of Floater and
// Hormann 2007, in the form that scipy.interpolate
// FloaterHormannInterpolator uses. The rational form blends the
// polynomials of degree zero to n minus d minus one, so it holds no
// pole on a real node range. The count d is the blending depth.
fn floater_hormann(xi: f64[:], y: f64[:], xs: f64[:], d: i64 = 3) -> Arr, Fault {
    let n = xi.len;
    if n < 2 || y.len != n {
        return empty_arr(), fault(F_SHAPE, "floater_hormann", "the nodes and the values need equal counts of at least two");
    }
    if xs.len < 1 {
        return empty_arr(), fault(F_SHAPE, "floater_hormann", "the list of points holds no value");
    }
    if d < 0 || d >= n {
        return empty_arr(), fault(F_RANGE, "floater_hormann", "the depth stands outside zero to the node count");
    }
    var i: i64 = 0;
    while i < n {
        if !(xi[i] - xi[i] == 0.0) || !(y[i] - y[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "floater_hormann", "a node or a value is not finite");
        }
        var j: i64 = 0;
        while j < i {
            if xi[j] == xi[i] {
                return empty_arr(), fault(F_VALUE, "floater_hormann", "the nodes hold a repeated value");
            }
            j += 1;
        }
        i += 1;
    }
    var w: f64[>..] = [];
    i = 0;
    while i < n { w.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var lo = i - d;
        if lo < 0 { lo = 0; }
        var hi = i + 1;
        if hi > n - d { hi = n - d; }
        var k: i64 = lo;
        while k < hi {
            var p: f64 = 1.0;
            var j: i64 = k;
            while j <= k + d {
                if j != i { p = p * mathshim::abs_of(xi[i] - xi[j]); }
                j += 1;
            }
            w[i] += 1.0 / p;
            k += 1;
        }
        if (i + d) % 2 != 0 { w[i] = -w[i]; }
        i += 1;
    }
    var out: f64[>..] = [];
    i = 0;
    while i < xs.len {
        out.push(barycentric_at(w, xi, y, xs[i]));
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// lagrange_poly gives the coefficients of the Lagrange polynomial,
// from the highest power to the lowest, as numpy.poly1d holds them.
// The method is the product form of SciPy: every node gives a term
// that holds the value times the product of the factors of the other
// nodes. The nodes must be distinct.
fn lagrange_poly(xi: f64[:], y: f64[:]) -> f64[>..], Fault {
    let m = xi.len;
    if m < 1 || y.len != m {
        return [], fault(F_SHAPE, "lagrange_poly", "the nodes and the values need equal counts of at least one");
    }
    var i: i64 = 0;
    while i < m {
        if !(xi[i] - xi[i] == 0.0) || !(y[i] - y[i] == 0.0) {
            return [], fault(F_VALUE, "lagrange_poly", "a node or a value is not finite");
        }
        var j: i64 = 0;
        while j < i {
            if xi[j] == xi[i] {
                return [], fault(F_VALUE, "lagrange_poly", "the nodes hold a repeated value");
            }
            j += 1;
        }
        i += 1;
    }
    // The sum of the terms holds the coefficients from the lowest
    // power to the highest one during the work.
    var acc: f64[>..] = [];
    var j: i64 = 0;
    while j < m {
        var term: f64[>..] = [y[j]];
        var k: i64 = 0;
        while k < m {
            if k != j {
                var prod: f64[>..] = [];
                var q: i64 = 0;
                while q < term.len { prod.push(0.0); q += 1; }
                prod.push(0.0);
                q = 0;
                while q < term.len {
                    prod[q] += -xi[k] * term[q];
                    prod[q + 1] += term[q];
                    q += 1;
                }
                var fac = 1.0 / (xi[j] - xi[k]);
                q = 0;
                while q < prod.len { prod[q] = prod[q] * fac; q += 1; }
                term = copy(prod);
            }
            k += 1;
        }
        if term.len > acc.len {
            var grown: f64[>..] = [];
            var q: i64 = 0;
            while q < acc.len { grown.push(acc[q]); q += 1; }
            while q < term.len { grown.push(0.0); q += 1; }
            acc = copy(grown);
        }
        var q: i64 = 0;
        while q < term.len { acc[q] += term[q]; q += 1; }
        j += 1;
    }
    var out: f64[>..] = [];
    i = acc.len - 1;
    while i >= 0 { out.push(acc[i]); i -= 1; }
    // The highest power that holds a zero is dropped, as NumPy drops
    // it in the sum of the polynomials.
    while out.len > 1 && out[0] == 0.0 {
        var less: f64[>..] = [];
        i = 1;
        while i < out.len { less.push(out[i]); i += 1; }
        out = copy(less);
    }
    return out, no_fault();
}

// pade_approx follows scipy.interpolate.pade: the ratio of two
// polynomials that holds the first m plus n plus one Taylor
// coefficients an. The orders are m for the denominator and n for the
// numerator. Both results hold the coefficients from the highest
// power to the lowest one, and the constant term of the denominator
// stays one.
fn pade_approx(an: f64[:], m: i64, n: i64) -> f64[>..], f64[>..], Fault {
    if m < 0 || n < 0 {
        return [], [], fault(F_RANGE, "pade_approx", "an order is below zero");
    }
    let ntot = m + n;
    if an.len < ntot + 1 {
        return [], [], fault(F_SHAPE, "pade_approx", "the coefficients need the count of the two orders plus one");
    }
    var size: i64 = ntot + 1;
    var a: f64[>..] = [];
    var row: i64 = 0;
    while row < size {
        var col: i64 = 0;
        while col < size {
            var v: f64 = 0.0;
            if col < n + 1 && row == col { v = 1.0; }
            if col >= n + 1 {
                let j = col - n - 1;
                if row - 1 - j >= 0 && row - 1 - j < an.len {
                    v = -an[row - 1 - j];
                }
            }
            a.push(v);
            col += 1;
        }
        row += 1;
    }
    var rhs: f64[>..] = [];
    row = 0;
    while row < size { rhs.push(an[row]); row += 1; }
    var sa: i64[..8] = [];
    sa.push(size);
    sa.push(size);
    var sb: i64[..8] = [];
    sb.push(size);
    let sol, f = solve(Arr { ndim: 2, shape: sa, data: copy(a) }, Arr { ndim: 1, shape: sb, data: copy(rhs) });
    if f.kind != F_NONE {
        return [], [], fault(F_VALUE, "pade_approx", "the system of the coefficients holds no solution");
    }
    var p: f64[>..] = [];
    row = n;
    while row >= 0 { p.push(sol.data[row]); row -= 1; }
    // The denominator holds the solved values from the highest order
    // to the lowest one, with the constant term of one.
    var q: f64[>..] = [];
    row = ntot;
    while row >= n + 1 { q.push(sol.data[row]); row -= 1; }
    q.push(1.0);
    return p, q, no_fault();
}

// taylor_approximate follows scipy.interpolate.approximate_taylor_polynomial:
// the polynomial of degree degree at the point x that the values of
// the function block on the nodes of Chebyshev reproduce. The nodes
// sit on the arc of scale around x, and one node falls on x. The
// result holds the coefficients from the highest power to the lowest
// one, and the polynomial stands at the origin, so the constant term
// is the value of the function at x. The count order is the count of
// nodes of the fit; the value below zero uses the degree.
fn taylor_approximate<F: fn(f64[:]) -> f64[:]>(x: f64, degree: i64, scale: f64, order: i64 = -1) -> f64[>..], Fault {
    var ord = order;
    if ord < 0 { ord = degree; }
    if degree < 0 {
        return [], fault(F_RANGE, "taylor_approximate", "the degree is below zero");
    }
    if ord < degree || scale <= 0.0 || !(scale - scale == 0.0) {
        return [], fault(F_VALUE, "taylor_approximate", "the order holds less than the degree, or the scale is not a positive width");
    }
    let n = ord + 1;
    // The nodes sit on the arc from the point of the scale above x to
    // the point below x, as SciPy places them. The list falls, so the
    // reversal puts it in the rising order that the recursion needs.
    var nodes: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        let angle = 3.141592653589793 * (i as f64) / (n as f64);
        nodes.push(scale * mathshim::cosine(angle) + x);
        i += 1;
    }
    let vals = F(copy(nodes));
    if vals.len < n {
        return [], fault(F_VALUE, "taylor_approximate", "the function holds fewer values than nodes");
    }
    var rnodes: f64[>..] = [];
    var rvals: f64[>..] = [];
    i = n - 1;
    while i >= 0 {
        rnodes.push(nodes[i]);
        rvals.push(vals[i]);
        i -= 1;
    }
    let c, f = krogh_coef(rnodes, rvals, "taylor_approximate");
    if f.kind != F_NONE { return [], f; }
    let d = krogh_derivs_at(c, rnodes, x, degree);
    // The coefficients of the Taylor form ascend with the factorial,
    // and the result descends.
    var asc: f64[>..] = [];
    var fact: f64 = 1.0;
    i = 0;
    while i <= degree {
        if i > 1 { fact = fact * (i as f64); }
        asc.push(d[i] / fact);
        i += 1;
    }
    var out: f64[>..] = [];
    i = degree;
    while i >= 0 { out.push(asc[i]); i -= 1; }
    return out, no_fault();
}

// pchip_interpolate follows scipy.interpolate.pchip_interpolate: the
// shape keeping spline of the package at the points. The spline
// follows the rule of PCHIP on both ends.
fn pchip_interpolate(xi: f64[:], y: f64[:], xs: f64[:]) -> Arr, Fault {
    let s, f = pchip(xi, y);
    if f.kind != F_NONE { return empty_arr(), f; }
    return spline_eval(s, xs);
}
