// The spatial data: the distances between points, the tables of
// distances, the search for the nearest points and the convex hull of a
// plane set. A point is a row of a matrix. A distance follows the
// definition of scipy.spatial.distance.
namespace scientific;

import mathshim;

// metric_code gives the number of a metric name, or -1.
fn metric_code(name: const u8[:]) -> i64 {
    if name == "euclidean" { return 0; }
    if name == "sqeuclidean" { return 1; }
    if name == "cityblock" { return 2; }
    if name == "chebyshev" { return 3; }
    if name == "minkowski" { return 4; }
    if name == "cosine" { return 5; }
    if name == "correlation" { return 6; }
    if name == "hamming" { return 7; }
    if name == "jaccard" { return 8; }
    if name == "canberra" { return 9; }
    if name == "braycurtis" { return 10; }
    if name == "seuclidean" { return 11; }
    if name == "mahalanobis" { return 12; }
    if name == "dice" { return 13; }
    if name == "russellrao" { return 14; }
    if name == "rogerstanimoto" { return 15; }
    if name == "sokalsneath" { return 16; }
    if name == "sokalmichener" { return 17; }
    if name == "yule" { return 18; }
    return -1;
}

// metric_check checks the name, the power and the weights for a count d
// of columns. The weights hold the variances of seuclidean, or the
// inverse covariance of mahalanobis in row order.
fn metric_check(code: i64, d: i64, p: f64, w: f64[:], tag: const u8[:]) -> Fault {
    if code < 0 {
        return fault(F_VALUE, tag, "the metric holds an unknown name");
    }
    if code == 4 && !(p > 0.0) {
        return fault(F_VALUE, tag, "the power p holds a value that is not above zero");
    }
    if code == 11 && w.len != d {
        return fault(F_SHAPE, tag, "seuclidean needs one variance per column");
    }
    if code == 12 && w.len != d * d {
        return fault(F_SHAPE, tag, "mahalanobis needs the inverse covariance of d rows and d columns");
    }
    return no_fault();
}

// dist_pair gives the distance between the row at xi of x and the row
// at yj of y. Each row holds d values.
fn dist_pair(x: f64[:], xi: i64, y: f64[:], yj: i64, d: i64, code: i64, p: f64, w: f64[:]) -> f64 {
    var s: f64 = 0.0;
    var k: i64 = 0;
    if code == 0 || code == 1 {
        while k < d { let t = x[xi + k] - y[yj + k]; s += t * t; k += 1; }
        if code == 0 { return mathshim::root(s); }
        return s;
    }
    if code == 2 {
        while k < d { s += mathshim::abs_of(x[xi + k] - y[yj + k]); k += 1; }
        return s;
    }
    if code == 3 {
        while k < d {
            let t = mathshim::abs_of(x[xi + k] - y[yj + k]);
            if t > s || t != t { s = t; }
            k += 1;
        }
        return s;
    }
    if code == 4 {
        while k < d { s += mathshim::power(mathshim::abs_of(x[xi + k] - y[yj + k]), p); k += 1; }
        return mathshim::power(s, 1.0 / p);
    }
    if code == 5 || code == 6 {
        var mu: f64 = 0.0;
        var mv: f64 = 0.0;
        if code == 6 {
            while k < d { mu += x[xi + k]; mv += y[yj + k]; k += 1; }
            mu = mu / d;
            mv = mv / d;
        }
        var uv: f64 = 0.0;
        var uu: f64 = 0.0;
        var vv: f64 = 0.0;
        k = 0;
        while k < d {
            let a = x[xi + k] - mu;
            let b = y[yj + k] - mv;
            uv += a * b;
            uu += a * a;
            vv += b * b;
            k += 1;
        }
        var r = 1.0 - uv / mathshim::root(uu * vv);
        // The distance never falls outside zero to two.
        if r < 0.0 { r = 0.0; }
        if r > 2.0 { r = 2.0; }
        return r;
    }
    if code == 7 {
        while k < d { if x[xi + k] != y[yj + k] { s += 1.0; } k += 1; }
        return s / d;
    }
    if code == 9 {
        while k < d {
            let a = mathshim::abs_of(x[xi + k] - y[yj + k]);
            let b = mathshim::abs_of(x[xi + k]) + mathshim::abs_of(y[yj + k]);
            if b != 0.0 { s += a / b; }
            k += 1;
        }
        return s;
    }
    if code == 10 {
        var num: f64 = 0.0;
        var den: f64 = 0.0;
        while k < d {
            num += mathshim::abs_of(x[xi + k] - y[yj + k]);
            den += mathshim::abs_of(x[xi + k] + y[yj + k]);
            k += 1;
        }
        return num / den;
    }
    if code == 11 {
        while k < d { let t = x[xi + k] - y[yj + k]; s += t * t / w[k]; k += 1; }
        return mathshim::root(s);
    }
    if code == 12 {
        var diff: f64[>..] = [];
        while k < d { diff.push(x[xi + k] - y[yj + k]); k += 1; }
        k = 0;
        while k < d {
            var t: f64 = 0.0;
            var j: i64 = 0;
            while j < d { t += w[k * d + j] * diff[j]; j += 1; }
            s += diff[k] * t;
            k += 1;
        }
        return mathshim::root(s);
    }
    // The metrics of the true and false counts.
    var ntt: f64 = 0.0;
    var ntf: f64 = 0.0;
    var nft: f64 = 0.0;
    var nff: f64 = 0.0;
    while k < d {
        let u = x[xi + k] != 0.0;
        let v = y[yj + k] != 0.0;
        if u && v { ntt += 1.0; }
        else if u { ntf += 1.0; }
        else if v { nft += 1.0; }
        else { nff += 1.0; }
        k += 1;
    }
    let r = ntf + nft;
    if code == 8 {
        if ntt + r == 0.0 { return 0.0; }
        return r / (ntt + r);
    }
    if code == 13 { return r / (2.0 * ntt + r); }
    if code == 14 { return (d - ntt) / d; }
    if code == 15 || code == 17 { return 2.0 * r / (ntt + nff + 2.0 * r); }
    if code == 16 { return 2.0 * r / (ntt + 2.0 * r); }
    var half = ntf * nft;
    if half == 0.0 { return 0.0; }
    return 2.0 * half / (ntt * nff + half);
}

// cdist gives the distances between every row of xa and every row of
// xb, in a matrix of one row per row of xa. The metrics are euclidean,
// sqeuclidean, cityblock, chebyshev, minkowski, cosine, correlation,
// hamming, jaccard, canberra, braycurtis, seuclidean, mahalanobis,
// dice, russellrao, rogerstanimoto, sokalsneath, sokalmichener and
// yule. The power p serves minkowski. The row w holds the variances of
// seuclidean, or the inverse covariance of mahalanobis in row order.
fn cdist(xa, xb, metric: const u8[:] = "euclidean", p: f64 = 2.0, w: f64[:] = []) -> Arr, Fault {
    if xa.ndim != 2 || xb.ndim != 2 || xa.shape[1] != xb.shape[1] {
        return empty_arr(), fault(F_SHAPE, "cdist", "the operands do not hold matrices of one count of columns");
    }
    let d = xa.shape[1];
    let code = metric_code(metric);
    let c = metric_check(code, d, p, w, "cdist");
    if c.kind != F_NONE { return empty_arr(), c; }
    let ma = xa.shape[0];
    let mb = xb.shape[0];
    if ma * mb > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "cdist", "the result holds more elements than the limit");
    }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < ma {
        var j: i64 = 0;
        while j < mb {
            out.push(dist_pair(xa.data, i * d, xb.data, j * d, d, code, p, w));
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(out), ma, mb), no_fault();
}

// pdist gives the distances between the rows of a matrix, in the
// condensed row of SciPy: the pairs (0, 1), (0, 2), and so on. The
// metrics are those of cdist. The matrix holds two rows at least.
fn pdist(x, metric: const u8[:] = "euclidean", p: f64 = 2.0, w: f64[:] = []) -> Arr, Fault {
    if x.ndim != 2 || x.shape[0] < 2 {
        return empty_arr(), fault(F_SHAPE, "pdist", "the operand is not a matrix of two rows or more");
    }
    let m = x.shape[0];
    let d = x.shape[1];
    let code = metric_code(metric);
    let c = metric_check(code, d, p, w, "pdist");
    if c.kind != F_NONE { return empty_arr(), c; }
    if m * (m - 1) / 2 > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "pdist", "the result holds more elements than the limit");
    }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < m {
        var j: i64 = i + 1;
        while j < m {
            out.push(dist_pair(x.data, i * d, x.data, j * d, d, code, p, w));
            j += 1;
        }
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// squareform turns a condensed row of distances into the square matrix,
// and a square matrix of zero diagonal into the condensed row. The
// square matrix must be symmetric.
fn squareform(x) -> Arr, Fault {
    if x.ndim == 1 {
        // The count of points m solves m (m - 1) / 2 = the length.
        var m: i64 = 2;
        while m * (m - 1) / 2 < x.data.len { m += 1; }
        if m * (m - 1) / 2 != x.data.len {
            return empty_arr(), fault(F_SHAPE, "squareform", "the length does not hold a condensed matrix");
        }
        var d: f64[>..] = [];
        var i: i64 = 0;
        while i < m * m { d.push(0.0); i += 1; }
        var at: i64 = 0;
        i = 0;
        while i < m {
            var j: i64 = i + 1;
            while j < m {
                d[i * m + j] = x.data[at];
                d[j * m + i] = x.data[at];
                at += 1;
                j += 1;
            }
            i += 1;
        }
        return mat2(copy(d), m, m), no_fault();
    }
    if x.ndim != 2 || x.shape[0] != x.shape[1] || x.shape[0] < 2 {
        return empty_arr(), fault(F_SHAPE, "squareform", "the operand is not a square matrix of two rows or more");
    }
    let m = x.shape[0];
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < m {
        if x.data[i * m + i] != 0.0 {
            return empty_arr(), fault(F_VALUE, "squareform", "the diagonal holds a value that is not zero");
        }
        var j: i64 = i + 1;
        while j < m {
            if x.data[i * m + j] != x.data[j * m + i] {
                return empty_arr(), fault(F_VALUE, "squareform", "the matrix is not symmetric");
            }
            out.push(x.data[i * m + j]);
            j += 1;
        }
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// distance gives the distance between two rows by a metric name. The
// rows hold one count of values.
fn distance(u: f64[:], v: f64[:], metric: const u8[:] = "euclidean", p: f64 = 2.0, w: f64[:] = []) -> f64, Fault {
    if u.len != v.len || u.len < 1 {
        return 0.0, fault(F_SHAPE, "distance", "the rows hold different counts or no value");
    }
    let code = metric_code(metric);
    let c = metric_check(code, u.len, p, w, "distance");
    if c.kind != F_NONE { return 0.0, c; }
    return dist_pair(u, 0, v, 0, u.len, code, p, w), no_fault();
}

// The names of scipy.spatial.distance for one pair of rows. The names
// cosine and hamming belong to other functions of the package, so the
// two distances carry the suffix _distance.
fn euclidean(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "euclidean"); }
fn sqeuclidean(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "sqeuclidean"); }
fn cityblock(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "cityblock"); }
fn chebyshev(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "chebyshev"); }
fn minkowski(u: f64[:], v: f64[:], p: f64 = 2.0) -> f64, Fault { return distance(u, v, "minkowski", p); }
fn cosine_distance(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "cosine"); }
fn correlation(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "correlation"); }
fn hamming_distance(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "hamming"); }
fn jaccard(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "jaccard"); }
fn canberra(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "canberra"); }
fn braycurtis(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "braycurtis"); }
fn seuclidean(u: f64[:], v: f64[:], variances: f64[:]) -> f64, Fault { return distance(u, v, "seuclidean", 2.0, variances); }
fn mahalanobis(u: f64[:], v: f64[:], vi: f64[:]) -> f64, Fault { return distance(u, v, "mahalanobis", 2.0, vi); }
fn dice(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "dice"); }
fn russellrao(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "russellrao"); }
fn rogerstanimoto(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "rogerstanimoto"); }
fn sokalsneath(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "sokalsneath"); }
fn sokalmichener(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "sokalmichener"); }
fn yule(u: f64[:], v: f64[:]) -> f64, Fault { return distance(u, v, "yule"); }

// KDTree holds a set of points for the search of the nearest ones. The
// search scans every point, so a query costs one pass over the set.
struct KDTree { count: i64, dim: i64, data: f64[>..] }

// kdtree builds the set from a matrix of points, one per row.
fn kdtree(points) -> KDTree, Fault {
    if points.ndim != 2 {
        return KDTree { count: 0, dim: 0, data: [] }, fault(F_SHAPE, "kdtree", "the operand is not a matrix");
    }
    return KDTree { count: points.shape[0], dim: points.shape[1], data: copy_slice(points.data) }, no_fault();
}

// kd_order gives the places of the points by their distance to a query,
// with ties in the order of the points. The query holds one row.
fn kd_order(t, x: f64[:], p: f64, dist: f64[>..]&) -> i64[>..] {
    var i: i64 = 0;
    while i < t.count {
        var s: f64 = 0.0;
        var k: i64 = 0;
        while k < t.dim {
            let a = mathshim::abs_of(t.data[i * t.dim + k] - x[k]);
            if p == 1.0 { s += a; }
            else if p == 2.0 { s += a * a; }
            else if p > 1.0e300 { if a > s { s = a; } }
            else { s += mathshim::power(a, p); }
            k += 1;
        }
        if p == 2.0 { s = mathshim::root(s); }
        else if p != 1.0 && !(p > 1.0e300) { s = mathshim::power(s, 1.0 / p); }
        dist.push(s);
        i += 1;
    }
    return order_places(dist, 1, dist.len);
}

// kdtree_query gives the k nearest points to a query row, as the
// distances and the places, nearest first. The power p picks the
// Minkowski norm, and an infinite p picks the largest difference.
fn kdtree_query(t, x: f64[:], k: i64 = 1, p: f64 = 2.0) -> Arr, Arr, Fault {
    if x.len != t.dim || t.count < 1 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "kdtree_query", "the query does not hold the dimension of the points");
    }
    if k < 1 || k > t.count {
        return empty_arr(), empty_arr(), fault(F_RANGE, "kdtree_query", "k stands outside one to the count of points");
    }
    if !(p >= 1.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "kdtree_query", "p holds a value below one");
    }
    var dist: f64[>..] = [];
    let pos = kd_order(t, x, p, dist);
    var d: f64[>..] = [];
    var ix: f64[>..] = [];
    var i: i64 = 0;
    while i < k { d.push(dist[pos[i]]); ix.push(pos[i]); i += 1; }
    return line_arr(copy(d)), line_arr(copy(ix)), no_fault();
}

// kdtree_query_ball_point gives the places of the points within the
// radius of a query row, in increasing order of place. A query with no
// point in range is a fault, because the package holds no empty array.
fn kdtree_query_ball_point(t, x: f64[:], r: f64, p: f64 = 2.0) -> Arr, Fault {
    if x.len != t.dim || t.count < 1 {
        return empty_arr(), fault(F_SHAPE, "kdtree_query_ball_point", "the query does not hold the dimension of the points");
    }
    if !(p >= 1.0) || !(r >= 0.0) {
        return empty_arr(), fault(F_VALUE, "kdtree_query_ball_point", "p or the radius holds a value out of range");
    }
    var dist: f64[>..] = [];
    let pos = kd_order(t, x, p, dist);
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < t.count { if dist[i] <= r { out.push(i); } i += 1; }
    if out.len == 0 {
        return empty_arr(), fault(F_VALUE, "kdtree_query_ball_point", "no point lies within the radius");
    }
    return line_arr(copy(out)), no_fault();
}

// convex_hull gives the hull of points of a plane. The result holds the
// places of the corner points in counterclockwise order, the area and
// the perimeter. The method is the monotone chain. A set whose points
// all lie on one line is a fault, as is a set of fewer than three
// points. SciPy names the area of the plane volume and the perimeter
// area.
fn convex_hull(points) -> Arr, f64, f64, Fault {
    if points.ndim != 2 || points.shape[1] != 2 || points.shape[0] < 3 {
        return empty_arr(), 0.0, 0.0, fault(F_SHAPE, "convex_hull", "the operand is not a matrix of three points or more and two columns");
    }
    let n = points.shape[0];
    // Sort the places by x, then by y.
    var keys: f64[>..] = [];
    var i: i64 = 0;
    while i < n { keys.push(points.data[i * 2 + 1]); i += 1; }
    i = 0;
    while i < n { keys.push(points.data[i * 2]); i += 1; }
    let ord = order_places(keys, 2, n);
    var hull: i64[>..] = [];
    var pass: i64 = 0;
    while pass < 2 {
        let start = hull.len;
        var k: i64 = 0;
        while k < n {
            var idx = ord[k];
            if pass == 1 { idx = ord[n - 1 - k]; }
            while hull.len >= start + 2 {
                let a = hull[hull.len - 2];
                let b = hull[hull.len - 1];
                let cross = (points.data[b * 2] - points.data[a * 2]) * (points.data[idx * 2 + 1] - points.data[a * 2 + 1])
                          - (points.data[b * 2 + 1] - points.data[a * 2 + 1]) * (points.data[idx * 2] - points.data[a * 2]);
                if cross <= 0.0 { hull.pop(); } else { break; }
            }
            hull.push(idx);
            k += 1;
        }
        hull.pop();
        pass += 1;
    }
    if hull.len < 3 {
        return empty_arr(), 0.0, 0.0, fault(F_VALUE, "convex_hull", "the points lie on one line");
    }
    var area: f64 = 0.0;
    var perim: f64 = 0.0;
    var v: f64[>..] = [];
    i = 0;
    while i < hull.len {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len];
        area += points.data[a * 2] * points.data[b * 2 + 1] - points.data[b * 2] * points.data[a * 2 + 1];
        let dx = points.data[b * 2] - points.data[a * 2];
        let dy = points.data[b * 2 + 1] - points.data[a * 2 + 1];
        perim += mathshim::root(dx * dx + dy * dy);
        v.push(a);
        i += 1;
    }
    return line_arr(copy(v)), area / 2.0, perim, no_fault();
}
