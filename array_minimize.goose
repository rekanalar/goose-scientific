// The minimization.
//
// A method takes the function of the caller as a parameter of a
// generic type. The caller passes the function as a block at the end
// of the call. The methods are iterative.
namespace scientific;

import mathshim;

// minimize_scalar finds the point between two points where a function
// is at its smallest. The method is the method of Brent for a
// minimum. It cuts the bracket by the golden section, and it fits a
// parabola through the three points at the end. The two points must
// hold a minimum between them, else the method ends at one of them.
fn minimize_scalar<F: fn(f64) -> f64>(a0: f64, b0: f64, xtol: f64, turns: i64) -> Arr, Fault {
    if xtol <= 0.0 {
        return empty_arr(), fault(F_VALUE, "minimize_scalar", "the tolerance holds a value of zero or below");
    }
    if turns < 1 {
        return empty_arr(), fault(F_VALUE, "minimize_scalar", "the count of turns holds a value below one");
    }
    var a = a0;
    var b = b0;
    if a > b {
        let t = a;
        a = b;
        b = t;
    }
    // The golden section. The two points inside the bracket hold the
    // two parts of it.
    let golden = 0.6180339887498949;
    var fa = F(a);
    var fb = F(b);
    var x1 = b - golden * (b - a);
    var x2 = a + golden * (b - a);
    var f1 = F(x1);
    var f2 = F(x2);
    var turn: i64 = 0;
    while turn < turns {
        if f1 <= f2 {
            // The minimum lies in the left part of the bracket.
            b = x2;
            fb = f2;
            x2 = x1;
            f2 = f1;
            x1 = b - golden * (b - a);
            f1 = F(x1);
        } else {
            // The minimum lies in the right part of the bracket.
            a = x1;
            fa = f1;
            x1 = x2;
            f1 = f2;
            x2 = a + golden * (b - a);
            f2 = F(x2);
        }
        turn += 1;
        if b - a <= xtol { break; }
    }
    // The point of the smallest value of the four points of the
    // bracket.
    var best = x1;
    var fbest = f1;
    if f2 < fbest { best = x2; fbest = f2; }
    if fa < fbest { best = a; fbest = fa; }
    if fb < fbest { best = b; fbest = fb; }
    // The parabola through the three points of the smallest values
    // gives a better point. The points come in the order of their
    // place.
    var p1 = a;
    var f1v = fa;
    var p2 = x1;
    var f2v = f1;
    var p3 = x2;
    var f3v = f2;
    // The four points in the order of their place.
    var px: f64[>..] = [];
    var pf: f64[>..] = [];
    px.push(a); pf.push(fa);
    px.push(x1); pf.push(f1);
    px.push(x2); pf.push(f2);
    px.push(b); pf.push(fb);
    // The three points of the smallest values.
    var keep: i64[>..] = [];
    var i: i64 = 0;
    while i < 4 { keep.push(i); i += 1; }
    i = 1;
    while i < 4 {
        let key = keep[i];
        var j = i - 1;
        while j >= 0 {
            let other = keep[j];
            if pf[other] > pf[key] {
                keep[j + 1] = other;
                j -= 1;
            } else { break; }
        }
        keep[j + 1] = key;
        i += 1;
    }
    // The three points in the order of their place.
    let k0 = keep[0];
    let k1 = keep[1];
    let k2 = keep[2];
    if k0 > k1 { p1 = px[k1]; f1v = pf[k1]; p2 = px[k0]; f2v = pf[k0]; p3 = px[k2]; f3v = pf[k2]; }
    else if k1 > k2 { p1 = px[k0]; f1v = pf[k0]; p2 = px[k2]; f2v = pf[k2]; p3 = px[k1]; f3v = pf[k1]; }
    else { p1 = px[k0]; f1v = pf[k0]; p2 = px[k1]; f2v = pf[k1]; p3 = px[k2]; f3v = pf[k2]; }
    let den = (p2 - p1) * (f2v - f3v) - (p3 - p2) * (f2v - f1v);
    if den != 0.0 {
        let p = p2 - 0.5 * ((p2 - p1) * (p2 - p1) * (f2v - f3v) - (p3 - p2) * (p3 - p2) * (f2v - f1v)) / den;
        if p > a && p < b && mathshim::abs_of(p - best) > xtol {
            let fp = F(p);
            if fp < fbest { best = p; }
        }
    }
    var s: i64[..8] = [];
    s.push(1);
    var out: f64[>..] = [];
    out.push(best);
    return Arr { ndim: 1, shape: s, data: out }, no_fault();
}

// minimize finds the point where a function of several variables is
// at its smallest. The function of the caller takes a vector and
// gives a vector of one value. The method is the method of Nelder and
// Mead. It holds a simplex of a count of points that is one more than
// the count of the variables.
fn minimize<F: fn(f64[:]) -> f64[:]>(x0: f64[:], xtol: f64, turns: i64) -> Arr, Fault {
    if xtol <= 0.0 {
        return empty_arr(), fault(F_VALUE, "minimize", "the tolerance holds a value of zero or below");
    }
    if turns < 1 {
        return empty_arr(), fault(F_VALUE, "minimize", "the count of turns holds a value below one");
    }
    let n = x0.len;
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "minimize", "the start holds no element");
    }
    // The simplex holds a count of points that is one more than the
    // count of the variables. The first point is the start. Every
    // other point moves away from the start in one coordinate, so the
    // simplex holds a direction for every coordinate.
    var points: f64[>..] = [];
    var values: f64[>..] = [];
    var i: i64 = 0;
    while i <= n {
        var p: f64[>..] = [];
        var j: i64 = 0;
        while j < n { p.push(x0[j]); j += 1; }
        if i > 0 {
            // The point moves in the coordinate of its number.
            var step = 0.05 * p[i - 1];
            if step == 0.0 { step = 0.00025; }
            p[i - 1] = p[i - 1] + step;
        }
        j = 0;
        while j < n { points.push(p[j]); j += 1; }
        let r = F(copy(p));
        if r.len < 1 { return empty_arr(), fault(F_VALUE, "minimize", "the function holds no value"); }
        values.push(r[0]);
        i += 1;
    }
    let count = n + 1;
    var turn: i64 = 0;
    while turn < turns {
        // The order of the points. The smallest value comes first and
        // the largest comes last.
        var order: i64[>..] = [];
        i = 0;
        while i < count { order.push(i); i += 1; }
        i = 1;
        while i < count {
            let key = order[i];
            var j = i - 1;
            while j >= 0 {
                let other = order[j];
                if values[other] > values[key] {
                    order[j + 1] = other;
                    j -= 1;
                } else { break; }
            }
            order[j + 1] = key;
            i += 1;
        }
        // The size of the simplex. It holds the distances between the
        // points.
        var size: f64 = 0.0;
        var a: i64 = 0;
        while a < count {
            var b = a + 1;
            while b < count {
                let pa = point_at(points, order[a], n);
                let pb = point_at(points, order[b], n);
                var d: f64 = 0.0;
                var k: i64 = 0;
                while k < n {
                    let t = pa[k] - pb[k];
                    d += t * t;
                    k += 1;
                }
                d = mathshim::root(d);
                if d > size { size = d; }
                b += 1;
            }
            a += 1;
        }
        if size <= xtol { break; }
        // The centroid of every point but the worst one.
        var worst = order[count - 1];
        var best = order[0];
        let second = order[count - 2];
        var centroid: f64[>..] = [];
        i = 0;
        while i < n { centroid.push(0.0); i += 1; }
        i = 0;
        while i < count {
            if order[i] != worst {
                let p = point_at(points, order[i], n);
                var k: i64 = 0;
                while k < n { centroid[k] = centroid[k] + p[k]; k += 1; }
            }
            i += 1;
        }
        i = 0;
        while i < n { centroid[i] = centroid[i] / (count - 1) as f64; i += 1; }
        // The reflection of the worst point in the centroid.
        let reflected = reflect(centroid, point_at(points, worst, n), n);
        let frr = F(copy(reflected));
        if frr.len < 1 { return empty_arr(), fault(F_VALUE, "minimize", "the function holds no value"); }
        let fr = frr[0];
        if fr < values[best] {
            // The point goes further, in the same direction.
            let expanded = expand(centroid, point_at(points, worst, n), n);
            let fee = F(copy(expanded));
            if fee.len < 1 { return empty_arr(), fault(F_VALUE, "minimize", "the function holds no value"); }
            let fe = fee[0];
            if fe < fr {
                set_point(points, worst, expanded, n);
                values[worst] = fe;
            } else {
                set_point(points, worst, reflected, n);
                values[worst] = fr;
            }
        } else if fr < values[second] {
            set_point(points, worst, reflected, n);
            values[worst] = fr;
        } else {
            // The contraction, or the shrink of the simplex toward the
            // best point.
            var contracted = contract(centroid, point_at(points, worst, n), n);
            let fcc = F(copy(contracted));
            if fcc.len < 1 { return empty_arr(), fault(F_VALUE, "minimize", "the function holds no value"); }
            let fc = fcc[0];
            if fc < values[worst] {
                set_point(points, worst, contracted, n);
                values[worst] = fc;
            } else {
                i = 1;
                while i < count {
                    let k = order[i];
                    let shrunk = shrink(point_at(points, best, n), point_at(points, k, n), n);
                    set_point(points, k, shrunk, n);
                    let fss = F(copy(shrunk));
                    if fss.len < 1 { return empty_arr(), fault(F_VALUE, "minimize", "the function holds no value"); }
                    values[k] = fss[0];
                    i += 1;
                }
            }
        }
        turn += 1;
    }
    // The point with the smallest value is the result.
    var best_index: i64 = 0;
    i = 1;
    while i < count {
        if values[i] < values[best_index] { best_index = i; }
        i += 1;
    }
    var out: f64[>..] = [];
    i = 0;
    while i < n { out.push(points[best_index * n + i]); i += 1; }
    var s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: out }, no_fault();
}

// reflect gives the point that the mirror of a point in a centre
// holds.
fn reflect(centroid, p: f64[:], n: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n { out.push(2.0 * centroid[i] - p[i]); i += 1; }
    return out;
}

// expand gives the point that the reflection of a point in a centre
// and a step further holds.
fn expand(centroid, p: f64[:], n: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n { out.push(3.0 * centroid[i] - 2.0 * p[i]); i += 1; }
    return out;
}

// contract gives the point between a centre and a point.
fn contract(centroid, p: f64[:], n: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n { out.push(centroid[i] + 0.5 * (centroid[i] - p[i])); i += 1; }
    return out;
}

// shrink gives the point between the best point of a simplex and
// another point.
fn shrink(best, p: f64[:], n: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n { out.push(best[i] + 0.5 * (p[i] - best[i])); i += 1; }
    return out;
}

// set_point puts a point in the place of another point of the
// simplex.
fn set_point(points: f64[>..]&, index: i64, p: f64[:], n: i64) {
    var i: i64 = 0;
    while i < n { points[index * n + i] = p[i]; i += 1; }
}

// point_at gives the copy of a point of the simplex.
fn point_at(points: f64[:], index: i64, n: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n { out.push(points[index * n + i]); i += 1; }
    return out;
}

// least_squares finds the point where the sum of the squares of the
// values of a function is at its smallest. The method is the method
// of Levenberg and Marquardt. The matrix of the derivatives comes
// from a small difference. The arrays lower and upper bound the
// variables. An empty array means no bound on that side, and a bound of
// plus or minus infinity leaves one variable free. A variable on its
// bound stays there while the step points outward.
fn least_squares<F: fn(f64[:]) -> f64[:]>(x0: f64[:], xtol: f64, turns: i64, lower: f64[:] = [], upper: f64[:] = []) -> Arr, Fault {
    if (lower.len != 0 && lower.len != x0.len) || (upper.len != 0 && upper.len != x0.len) {
        return empty_arr(), fault(F_SHAPE, "least_squares", "a bound holds a wrong count of values");
    }
    {
        var k: i64 = 0;
        while k < x0.len {
            var lo = -1.0e308 * 10.0;
            var hi = 1.0e308 * 10.0;
            if lower.len > 0 { lo = lower[k]; }
            if upper.len > 0 { hi = upper[k]; }
            if !(lo < hi) || !(x0[k] >= lo) || !(x0[k] <= hi) {
                return empty_arr(), fault(F_VALUE, "least_squares", "a bound is not ordered or the start stands outside the bounds");
            }
            k += 1;
        }
    }
    if xtol <= 0.0 {
        return empty_arr(), fault(F_VALUE, "least_squares", "the tolerance holds a value of zero or below");
    }
    if turns < 1 {
        return empty_arr(), fault(F_VALUE, "least_squares", "the count of turns holds a value below one");
    }
    let n = x0.len;
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "least_squares", "the start holds no element");
    }
    var x: f64[>..] = [];
    var i: i64 = 0;
    while i < n { x.push(x0[i]); i += 1; }
    var damping = 0.001;
    var turn: i64 = 0;
    while turn < turns {
        let r = F(copy(x));
        let m = r.len;
        if m < 1 {
            return empty_arr(), fault(F_VALUE, "least_squares", "the function holds no value");
        }
        // The size of the error is the sum of the squares.
        var size: f64 = 0.0;
        i = 0;
        while i < m { size += r[i] * r[i]; i += 1; }
        if mathshim::root(size) <= xtol { break; }
        // The matrix of the derivatives. A row holds a value of the
        // function, and a column holds a variable.
        var jac: f64[>..] = [];
        var cell: i64 = 0;
        while cell < m * n { jac.push(0.0); cell += 1; }
        var d: i64 = 0;
        while d < n {
            let h = 0.0000001 * (1.0 + mathshim::abs_of(x[d]));
            var probe: f64[>..] = [];
            i = 0;
            while i < n { probe.push(x[i]); i += 1; }
            probe[d] = probe[d] + h;
            let rp = F(copy(probe));
            i = 0;
            while i < m {
                if i < rp.len { jac[i * n + d] = (rp[i] - r[i]) / h; }
                i += 1;
            }
            d += 1;
        }
        // The matrix of the normal equations takes the damping on its
        // diagonal.
        var normal: f64[>..] = [];
        cell = 0;
        while cell < n * n { normal.push(0.0); cell += 1; }
        var a: i64 = 0;
        while a < n {
            var b: i64 = 0;
            while b < n {
                var t: f64 = 0.0;
                var row: i64 = 0;
                while row < m {
                    t += jac[row * n + a] * jac[row * n + b];
                    row += 1;
                }
                normal[a * n + b] = t;
                b += 1;
            }
            a += 1;
        }
        // The right side holds the products of the values and of the
        // derivatives.
        var rhs: f64[>..] = [];
        a = 0;
        while a < n {
            var t: f64 = 0.0;
            var row: i64 = 0;
            while row < m {
                t += jac[row * n + a] * r[row];
                row += 1;
            }
            rhs.push(-t);
            a += 1;
        }
        // A variable on a bound that the step would leave stays fixed.
        a = 0;
        while a < n {
            var fixed = false;
            if lower.len > 0 && x[a] <= lower[a] && rhs[a] < 0.0 { fixed = true; }
            if upper.len > 0 && x[a] >= upper[a] && rhs[a] > 0.0 { fixed = true; }
            if fixed {
                var b: i64 = 0;
                while b < n {
                    normal[a * n + b] = 0.0;
                    normal[b * n + a] = 0.0;
                    b += 1;
                }
                normal[a * n + a] = 1.0;
                rhs[a] = 0.0;
            }
            a += 1;
        }
        // The step comes from the solving of the normal equations.
        let first, fstep = lsq_step(normal, rhs, n, damping);
        if fstep.kind != F_NONE {
            return empty_arr(), fault(F_VALUE, "least_squares", "the matrix of the derivatives holds no pivot");
        }
        var step = first;
        // The damping grows while the step does not lower the size of
        // the error.
        var factor: f64 = 1.0;
        var accepted = false;
        var tries: i64 = 0;
        while tries < 30 {
            var trial: f64[>..] = [];
            i = 0;
            while i < n {
                var tv = x[i] + factor * step.data[i];
                if lower.len > 0 && tv < lower[i] { tv = lower[i]; }
                if upper.len > 0 && tv > upper[i] { tv = upper[i]; }
                trial.push(tv);
                i += 1;
            }
            let rt = F(copy(trial));
            var size2: f64 = 0.0;
            i = 0;
            while i < rt.len { size2 += rt[i] * rt[i]; i += 1; }
            if size2 < size {
                x = copy(trial);
                accepted = true;
                break;
            }
            damping = damping * 10.0;
            factor = factor / 10.0;
            let step2, fstep2 = lsq_step(normal, rhs, n, damping);
            if fstep2.kind != F_NONE { break; }
            step = copy(step2);
            tries += 1;
        }
        if !accepted {
            damping = damping / 10.0;
        }
        turn += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: x }, no_fault();
}

// lsq_step gives the step of the normal equations with a damping.
fn lsq_step(normal: f64[:], rhs: f64[:], n: i64, damping: f64) -> Arr, Fault {
    var a: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { a.push(normal[i]); i += 1; }
    i = 0;
    while i < n { a[i * n + i] = a[i * n + i] * (1.0 + damping); i += 1; }
    var s: i64[..8] = [];
    s.push(n);
    s.push(n);
    let mat = Arr { ndim: 2, shape: s, data: copy(a) };
    var rs: i64[..8] = [];
    rs.push(n);
    var rv: f64[>..] = [];
    var i: i64 = 0;
    while i < rhs.len { rv.push(rhs[i]); i += 1; }
    let vec = Arr { ndim: 1, shape: rs, data: copy(rv) };
    return solve(mat, vec);
}

// curve_fit fits a model f(x, p) to the points (xdata, ydata) by least
// squares. The model is a block { x, p => ... }. It gives the best
// parameters and their covariance. The covariance is the inverse of
// the normal matrix, scaled by the residual variance, as SciPy scales
// it with the default setting. The call needs more points than
// parameters. A singular normal matrix is a fault.
fn curve_fit<F: fn(f64, f64[:]) -> f64>(xdata: f64[:], ydata: f64[:], p0: f64[:], turns: i64 = 200, sigma: f64[:] = [], lower: f64[:] = [], upper: f64[:] = []) -> Arr, Arr, Fault {
    let m = xdata.len;
    let n = p0.len;
    if n < 1 || m != ydata.len {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "curve_fit", "the data and the start do not agree");
    }
    if m <= n {
        return empty_arr(), empty_arr(), fault(F_VALUE, "curve_fit", "the call needs more points than parameters");
    }
    var i: i64 = 0;
    while i < m {
        if !(xdata[i] - xdata[i] == 0.0) || !(ydata[i] - ydata[i] == 0.0) {
            return empty_arr(), empty_arr(), fault(F_VALUE, "curve_fit", "the data holds a value that is not finite");
        }
        i += 1;
    }
    // The form of sigma. The count 0 means no sigma. The count 1 is a
    // value for every point. The count m holds a value for each
    // point. The count m times m holds the matrix of the covariance.
    var mode: i64 = 0;
    if sigma.len == 1 { mode = 1; }
    else if sigma.len == m { mode = 2; }
    else if sigma.len == m * m && m > 1 { mode = 3; }
    else if sigma.len > 0 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "curve_fit", "sigma holds a wrong count of values");
    }
    i = 0;
    while i < sigma.len {
        if !(sigma[i] - sigma[i] == 0.0) || mathshim::is_infinite_of(sigma[i]) != 0.0 {
            return empty_arr(), empty_arr(), fault(F_VALUE, "curve_fit", "sigma holds a value that is not finite");
        }
        i += 1;
    }
    if mode == 1 || mode == 2 {
        var k: i64 = 0;
        while k < m {
            var sv = sigma[0];
            if mode == 2 { sv = sigma[k]; }
            if sv == 0.0 {
                return empty_arr(), empty_arr(), fault(F_VALUE, "curve_fit", "sigma holds a value of zero");
            }
            k += 1;
        }
    }
    // The factor of the matrix form. It is the lower triangle of the
    // Cholesky of the matrix of sigma.
    var lo = empty_arr();
    if mode == 3 {
        let sm = Arr { ndim: 2, shape: [m, m], data: copy_slice(sigma) };
        let l, lf = cholesky(sm);
        if lf.kind != F_NONE {
            return empty_arr(), empty_arr(), fault(F_VALUE, "curve_fit", "the matrix of sigma is not positive definite");
        }
        lo = copy(l);
    }
    let best, f = least_squares(p0, 1.0e-14, turns, lower, upper) { p =>
        var r: f64[>..] = [];
        var k: i64 = 0;
        while k < m {
            var v = F(xdata[k], copy(p)) - ydata[k];
            if mode == 1 { v = v / sigma[0]; }
            else if mode == 2 { v = v / sigma[k]; }
            r.push(v);
            k += 1;
        }
        if mode == 3 {
            // The factor of the matrix form scales the whole vector.
            var sh: i64[..8] = [];
            sh.push(m);
            let zz, zf = solve_triangular(lo, Arr { ndim: 1, shape: sh, data: copy(r) }, true);
            if zf.kind == F_NONE {
                r = copy(zz.data);
            } else {
                var kk: i64 = 0;
                while kk < m { r[kk] = 0.0; kk += 1; }
            }
        }
        r
    };
    if f.kind != F_NONE { return empty_arr(), empty_arr(), fault(f.kind, "curve_fit", f.detail); }
    // The Jacobian by central differences at the best point.
    var jac: f64[>..] = [];
    i = 0;
    while i < m * n { jac.push(0.0); i += 1; }
    // The residuals at the best point. They take the factor of sigma.
    var rr: f64[>..] = [];
    i = 0;
    while i < m {
        rr.push(F(xdata[i], copy(best.data)) - ydata[i]);
        i += 1;
    }
    if mode == 1 || mode == 2 {
        i = 0;
        while i < m {
            var sv = sigma[0];
            if mode == 2 { sv = sigma[i]; }
            rr[i] = rr[i] / sv;
            i += 1;
        }
    } else if mode == 3 {
        var sh: i64[..8] = [];
        sh.push(m);
        let zz, zf = solve_triangular(lo, Arr { ndim: 1, shape: sh, data: copy(rr) }, true);
        if zf.kind == F_NONE { rr = copy(zz.data); }
    }
    var res: f64 = 0.0;
    i = 0;
    while i < m {
        res += rr[i] * rr[i];
        var d: i64 = 0;
        while d < n {
            let h = 0.000001 * (1.0 + mathshim::abs_of(best.data[d]));
            var up: f64[>..] = [];
            var dn: f64[>..] = [];
            var k: i64 = 0;
            while k < n { up.push(best.data[k]); dn.push(best.data[k]); k += 1; }
            up[d] = up[d] + h;
            dn[d] = dn[d] - h;
            jac[i * n + d] = (F(xdata[i], copy(up)) - F(xdata[i], copy(dn))) / (2.0 * h);
            d += 1;
        }
        i += 1;
    }
    // The rows of the Jacobian take the factor of sigma.
    if mode == 1 || mode == 2 {
        i = 0;
        while i < m {
            var sv = sigma[0];
            if mode == 2 { sv = sigma[i]; }
            var d: i64 = 0;
            while d < n { jac[i * n + d] = jac[i * n + d] / sv; d += 1; }
            i += 1;
        }
    } else if mode == 3 {
        // The factor of the matrix form scales every column of the
        // Jacobian.
        var sh: i64[..8] = [];
        sh.push(m);
        var d: i64 = 0;
        while d < n {
            var col: f64[>..] = [];
            var row: i64 = 0;
            while row < m { col.push(jac[row * n + d]); row += 1; }
            let zz, zf = solve_triangular(lo, Arr { ndim: 1, shape: sh, data: copy(col) }, true);
            if zf.kind == F_NONE {
                row = 0;
                while row < m { jac[row * n + d] = zz.data[row]; row += 1; }
            }
            d += 1;
        }
    }
    var normal: f64[>..] = [];
    i = 0;
    while i < n * n { normal.push(0.0); i += 1; }
    var a: i64 = 0;
    while a < n {
        var b: i64 = 0;
        while b < n {
            var t: f64 = 0.0;
            var row: i64 = 0;
            while row < m { t += jac[row * n + a] * jac[row * n + b]; row += 1; }
            normal[a * n + b] = t;
            b += 1;
        }
        a += 1;
    }
    let nm = Arr { ndim: 2, shape: [n, n], data: copy(normal) };
    let covu, fi = inv(nm);
    if fi.kind != F_NONE { return empty_arr(), empty_arr(), fault(F_VALUE, "curve_fit", "the normal matrix is singular"); }
    let s2 = res / (m - n);
    var cov: f64[>..] = [];
    i = 0;
    while i < n * n { cov.push(covu.data[i] * s2); i += 1; }
    return best, Arr { ndim: 2, shape: [n, n], data: cov }, no_fault();
}

// differential_evolution finds the global minimum of a function of
// several variables inside the bounds, by the strategy best1bin of
// Storn and Price. The population holds popsize times the count of
// variables. The mutation takes a random factor between 0.5 and 1 for
// each generation. The run stops when the spread of the values of the
// population falls below tol times the mean, or after maxiter
// generations. The seed fixes the random numbers, so equal seeds give
// equal results, but the numbers differ from those of SciPy. The
// function gives the point, the value, and the fault. The result holds
// the point followed by its value as the last element. The method has
// no local polish step.
fn differential_evolution<F: fn(f64[:]) -> f64>(lower: f64[:], upper: f64[:], seed: i64 = 1, maxiter: i64 = 1000, popsize: i64 = 15, tol: f64 = 0.01, recombination: f64 = 0.7) -> Arr, Fault {
    let n = lower.len;
    if n < 1 || n != upper.len || n > 1000 {
        return empty_arr(), fault(F_SHAPE, "differential_evolution", "the bounds must hold one to 1000 variables, and the two lists must agree");
    }
    if popsize < 4 || popsize > 1000 || maxiter < 1 || maxiter > 1000000 || !(tol >= 0.0) || !(recombination >= 0.0) || !(recombination <= 1.0) {
        return empty_arr(), fault(F_RANGE, "differential_evolution", "popsize, maxiter, tol or recombination stands outside its range");
    }
    var i: i64 = 0;
    while i < n {
        if !(lower[i] - lower[i] == 0.0) || !(upper[i] - upper[i] == 0.0) || !(lower[i] <= upper[i]) {
            return empty_arr(), fault(F_VALUE, "differential_evolution", "a bound is not finite, or a lower bound lies above its upper bound");
        }
        i += 1;
    }
    let np = popsize * n;
    var r = default_rng(seed);
    var pop: f64[>..] = [];
    var val: f64[>..] = [];
    var p: i64 = 0;
    while p < np {
        var cand: f64[>..] = [];
        i = 0;
        while i < n { cand.push(lower[i] + random_uniform(r) * (upper[i] - lower[i])); i += 1; }
        let v = F(copy(cand));
        if !(v - v == 0.0) {
            return empty_arr(), fault(F_VALUE, "differential_evolution", "the function gave a value that is not finite");
        }
        i = 0;
        while i < n { pop.push(cand[i]); i += 1; }
        val.push(v);
        p += 1;
    }
    var gen: i64 = 0;
    while gen < maxiter {
        var best: i64 = 0;
        p = 1;
        while p < np { if val[p] < val[best] { best = p; } p += 1; }
        let mutation = 0.5 + 0.5 * random_uniform(r);
        p = 0;
        while p < np {
            var r1 = integers(r, 0, np);
            var r2 = integers(r, 0, np);
            while r1 == p { r1 = integers(r, 0, np); }
            while r2 == p || r2 == r1 { r2 = integers(r, 0, np); }
            let jr = integers(r, 0, n);
            var trial: f64[>..] = [];
            i = 0;
            while i < n {
                var xi = pop[p * n + i];
                if random_uniform(r) < recombination || i == jr {
                    xi = pop[best * n + i] + mutation * (pop[r1 * n + i] - pop[r2 * n + i]);
                    if xi < lower[i] || xi > upper[i] { xi = lower[i] + random_uniform(r) * (upper[i] - lower[i]); }
                }
                trial.push(xi);
                i += 1;
            }
            let tv = F(copy(trial));
            if !(tv - tv == 0.0) {
                return empty_arr(), fault(F_VALUE, "differential_evolution", "the function gave a value that is not finite");
            }
            if tv < val[p] {
                i = 0;
                while i < n { pop[p * n + i] = trial[i]; i += 1; }
                val[p] = tv;
            }
            p += 1;
        }
        gen += 1;
        var mean: f64 = 0.0;
        p = 0;
        while p < np { mean += val[p]; p += 1; }
        mean = mean / np;
        var sq: f64 = 0.0;
        p = 0;
        while p < np { sq += (val[p] - mean) * (val[p] - mean); p += 1; }
        let spread = mathshim::root(sq / np);
        if spread <= 1.0e-300 || spread <= tol * mathshim::abs_of(mean) { break; }
    }
    var best: i64 = 0;
    p = 1;
    while p < np { if val[p] < val[best] { best = p; } p += 1; }
    var out: f64[>..] = [];
    i = 0;
    while i < n { out.push(pop[best * n + i]); i += 1; }
    out.push(val[best]);
    return line_arr(copy(out)), no_fault();
}
