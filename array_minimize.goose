// The minimization of a function: the steps of the method and the line search.
//
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

