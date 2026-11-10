// The Delaunay triangulation of points of a plane and the scattered
// data interpolation of scipy.interpolate.griddata. The triangulation
// follows the method of Bowyer and Watson on points that the code
// scales into the unit square. It costs time of the order of the
// square of the count of points.
namespace scientific;

import mathshim;

// The count of points that delaunay takes. The time grows with the
// square of it.
const MAX_DELAUNAY: i64 = 5000;

// Delaunay holds the count of points, the count of triangles and one
// buffer. The buffer holds the points as given, then the corner places
// of each triangle. The corners of a triangle run counterclockwise.
struct Delaunay { npts: i64, nsimp: i64, buf: f64[>..] }

// dl_corner gives the place of one corner of the triangles: the index
// k counts the corners of all triangles in a row.
fn dl_corner(d, k: i64) -> i64 {
    return d.buf[d.npts * 2 + k] as i64;
}

fn delaunay_empty() -> Delaunay {
    return Delaunay { npts: 0, nsimp: 0, buf: [] };
}

// dl_incircle is true when the point d lies strictly inside the circle
// through the corners a, b and c, which run counterclockwise.
fn dl_incircle(p: f64[:], a: i64, b: i64, c: i64, d: f64, e: f64) -> bool {
    let adx = p[a * 2] - d;
    let ady = p[a * 2 + 1] - e;
    let bdx = p[b * 2] - d;
    let bdy = p[b * 2 + 1] - e;
    let cdx = p[c * 2] - d;
    let cdy = p[c * 2 + 1] - e;
    let det = (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
            - (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady)
            + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady);
    return det > 0.0;
}

// delaunay builds the triangulation of the points. The operand holds
// at least three points and two columns. A point that repeats an
// earlier point stays out of the triangles, as in SciPy. A set whose
// points all lie on one line is a fault. A set where the method loses
// part of the hull is a fault too.
fn delaunay(points) -> Delaunay, Fault {
    if points.ndim != 2 || points.shape[1] != 2 || points.shape[0] < 3 {
        return delaunay_empty(), fault(F_SHAPE, "delaunay", "the operand is not a matrix of three points or more and two columns");
    }
    let n = points.shape[0];
    if n > MAX_DELAUNAY {
        return delaunay_empty(), fault(F_LIMIT, "delaunay", "the set holds more points than the limit");
    }
    var i: i64 = 0;
    while i < n * 2 {
        if !(points.data[i] - points.data[i] == 0.0) {
            return delaunay_empty(), fault(F_VALUE, "delaunay", "a coordinate is not finite");
        }
        i += 1;
    }
    var lox = points.data[0];
    var hix = lox;
    var loy = points.data[1];
    var hiy = loy;
    i = 1;
    while i < n {
        let px = points.data[i * 2];
        let py = points.data[i * 2 + 1];
        if px < lox { lox = px; }
        if px > hix { hix = px; }
        if py < loy { loy = py; }
        if py > hiy { hiy = py; }
        i += 1;
    }
    var span = hix - lox;
    if hiy - loy > span { span = hiy - loy; }
    if !(span > 0.0) {
        return delaunay_empty(), fault(F_VALUE, "delaunay", "the points lie on one line");
    }
    // The scaled points, then the three corners of a big triangle.
    var p: f64[>..] = [];
    i = 0;
    while i < n {
        p.push((points.data[i * 2] - lox) / span);
        p.push((points.data[i * 2 + 1] - loy) / span);
        i += 1;
    }
    let big: f64 = 100000.0;
    p.push(0.5 - big);
    p.push(0.5 - big);
    p.push(0.5 + big);
    p.push(0.5 - big);
    p.push(0.5);
    p.push(0.5 + big);
    var tri: i64[>..] = [n, n + 1, n + 2];
    var alive: i64[>..] = [1];
    var head: i64[>..] = [];
    i = 0;
    while i < n + 3 { head.push(-1); i += 1; }
    i = 0;
    while i < n {
        // A repeated point keeps its first place only.
        var dup: i64 = 0;
        var j: i64 = 0;
        while j < i && dup == 0 {
            if p[j * 2] == p[i * 2] && p[j * 2 + 1] == p[i * 2 + 1] { dup = 1; }
            j += 1;
        }
        if dup == 0 {
            var bad: i64[>..] = [];
            var t: i64 = 0;
            while t < alive.len {
                if alive[t] == 1 && dl_incircle(p, tri[t * 3], tri[t * 3 + 1], tri[t * 3 + 2], p[i * 2], p[i * 2 + 1]) {
                    bad.push(t);
                }
                t += 1;
            }
            // The edges of the cavity are the edges of the bad triangles
            // whose reverse edge belongs to no bad triangle.
            var ea: i64[>..] = [];
            var eb: i64[>..] = [];
            var q: i64 = 0;
            while q < bad.len {
                let u = bad[q] * 3;
                var s: i64 = 0;
                while s < 3 {
                    ea.push(tri[u + s]);
                    eb.push(tri[u + (s + 1) % 3]);
                    s += 1;
                }
                q += 1;
            }
            q = 0;
            while q < bad.len { alive[bad[q]] = 0; q += 1; }
            // A list for each start corner finds the reverse edge fast.
            var nx: i64[>..] = [];
            var e: i64 = 0;
            while e < ea.len {
                nx.push(head[ea[e]]);
                head[ea[e]] = e;
                e += 1;
            }
            e = 0;
            while e < ea.len {
                var shared = false;
                var f = head[eb[e]];
                while f >= 0 && !shared {
                    if eb[f] == ea[e] { shared = true; }
                    f = nx[f];
                }
                if !shared {
                    tri.push(ea[e]);
                    tri.push(eb[e]);
                    tri.push(i);
                    alive.push(1);
                }
                e += 1;
            }
            e = 0;
            while e < ea.len { head[ea[e]] = -1; e += 1; }
        }
        i += 1;
    }
    var simp: i64[>..] = [];
    var t: i64 = 0;
    while t < alive.len {
        if alive[t] == 1 && tri[t * 3] < n && tri[t * 3 + 1] < n && tri[t * 3 + 2] < n {
            simp.push(tri[t * 3]);
            simp.push(tri[t * 3 + 1]);
            simp.push(tri[t * 3 + 2]);
        }
        t += 1;
    }
    if simp.len == 0 {
        return delaunay_empty(), fault(F_VALUE, "delaunay", "the points lie on one line");
    }
    // The triangles must fill the hull. A hull edge that the big triangle
    // cut away makes the area of the triangles fall short.
    let hull, harea, hperim, fh = convex_hull(points);
    if fh.kind != F_NONE { return delaunay_empty(), fault(F_VALUE, "delaunay", "the points lie on one line"); }
    var total: f64 = 0.0;
    t = 0;
    while t < simp.len / 3 {
        let a = simp[t * 3];
        let b = simp[t * 3 + 1];
        let c = simp[t * 3 + 2];
        total += ((points.data[b * 2] - points.data[a * 2]) * (points.data[c * 2 + 1] - points.data[a * 2 + 1])
                - (points.data[c * 2] - points.data[a * 2]) * (points.data[b * 2 + 1] - points.data[a * 2 + 1])) / 2.0;
        t += 1;
    }
    var gap = total - harea;
    if gap < 0.0 { gap = -gap; }
    if gap > 1.0e-9 * harea {
        return delaunay_empty(), fault(F_VALUE, "delaunay", "the points are too close to one line for the triangulation");
    }
    var buf: f64[>..] = copy_slice(points.data);
    i = 0;
    while i < simp.len { buf.push(simp[i] * 1.0); i += 1; }
    return Delaunay { npts: n, nsimp: simp.len / 3, buf: buf }, no_fault();
}

// delaunay_simplices gives the corner places of the triangles as a
// matrix with three columns.
fn delaunay_simplices(d) -> Arr {
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < d.nsimp * 3 { v.push(dl_corner(d, i) * 1.0); i += 1; }
    var sh: i64[..8] = [];
    sh.push(d.nsimp);
    sh.push(3);
    return Arr { ndim: 2, shape: sh, data: v };
}

// dl_locate gives the triangle that holds the point and its three
// barycentric coordinates in cs, or -1. A point on an edge or a corner
// takes the first triangle that holds it.
fn dl_locate(d, x: f64, y: f64, cs: f64[>..]&) -> i64 {
    let eps: f64 = 2.220446049250313e-14;
    var t: i64 = 0;
    while t < d.nsimp {
        let a = dl_corner(d, t * 3);
        let b = dl_corner(d, t * 3 + 1);
        let c = dl_corner(d, t * 3 + 2);
        let ax = d.buf[a * 2];
        let ay = d.buf[a * 2 + 1];
        let det = (d.buf[b * 2] - ax) * (d.buf[c * 2 + 1] - ay) - (d.buf[c * 2] - ax) * (d.buf[b * 2 + 1] - ay);
        if det != 0.0 {
            let l1 = ((x - ax) * (d.buf[c * 2 + 1] - ay) - (d.buf[c * 2] - ax) * (y - ay)) / det;
            let l2 = ((d.buf[b * 2] - ax) * (y - ay) - (x - ax) * (d.buf[b * 2 + 1] - ay)) / det;
            let l0 = 1.0 - l1 - l2;
            if l0 >= -eps && l1 >= -eps && l2 >= -eps && l0 <= 1.0 + eps && l1 <= 1.0 + eps && l2 <= 1.0 + eps {
                cs[0] = l0;
                cs[1] = l1;
                cs[2] = l2;
                return t;
            }
        }
        t += 1;
    }
    return -1;
}

// delaunay_find_simplex gives, for each row of xi, the place of the
// triangle that holds it, or -1 for a point outside the hull.
fn delaunay_find_simplex(d, xi) -> Arr, Fault {
    if xi.ndim != 2 || xi.shape[1] != 2 {
        return empty_arr(), fault(F_SHAPE, "delaunay_find_simplex", "the points are not a matrix with two columns");
    }
    var out: f64[>..] = [];
    var cs: f64[>..] = [0.0, 0.0, 0.0];
    var i: i64 = 0;
    while i < xi.shape[0] {
        out.push(dl_locate(d, xi.data[i * 2], xi.data[i * 2 + 1], cs) * 1.0);
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// dl_sort_order gives the places of the values from the least to the
// greatest. The sort is stable.
fn dl_sort_order(v: f64[:]) -> i64[>..] {
    var ord: i64[>..] = [];
    var i: i64 = 0;
    while i < v.len {
        ord.push(i);
        var j = ord.len - 1;
        while j > 0 && v[ord[j - 1]] > v[ord[j]] {
            let tmp = ord[j];
            ord[j] = ord[j - 1];
            ord[j - 1] = tmp;
            j -= 1;
        }
        i += 1;
    }
    return ord;
}

// griddata interpolates scattered data at the rows of xi. The points
// hold a column for each dimension, and one column of values holds one
// value for each point. The method nearest takes the value of the
// nearest point in any count of dimensions. The method linear takes the
// piecewise linear function on the Delaunay triangles of a plane, or on
// the intervals of one dimension. A point outside the hull takes
// fill_value. The flag rescale scales each axis to the unit interval
// before the search.
fn griddata(points, values, xi, method: const u8[:] = "linear", fill_value: f64 = 0.0 / 0.0, rescale: bool = false) -> Arr, Fault {
    if method != "nearest" && method != "linear" {
        return empty_arr(), fault(F_VALUE, "griddata", "the method is not nearest or linear");
    }
    if points.ndim < 1 || points.ndim > 2 || xi.ndim < 1 || xi.ndim > 2 || values.ndim != 1 {
        return empty_arr(), fault(F_SHAPE, "griddata", "the points are not a vector or a matrix");
    }
    // A vector of points is one dimension.
    let n: i64 = points.shape[0];
    var dim: i64 = 1;
    if points.ndim == 2 { dim = points.shape[1]; }
    if values.shape[0] != n {
        return empty_arr(), fault(F_SHAPE, "griddata", "the values do not hold one number for each point");
    }
    let m: i64 = xi.shape[0];
    var xdim: i64 = 1;
    if xi.ndim == 2 { xdim = xi.shape[1]; }
    if xdim != dim {
        return empty_arr(), fault(F_SHAPE, "griddata", "the points to find do not match the dimension of the data");
    }
    if n < 1 || dim < 1 {
        return empty_arr(), fault(F_SHAPE, "griddata", "the data hold no point");
    }
    var i: i64 = 0;
    while i < n * dim {
        if !(points.data[i] - points.data[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "griddata", "a point is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < n {
        if !(values.data[i] - values.data[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "griddata", "a value is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < m * dim {
        if !(xi.data[i] - xi.data[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "griddata", "a point to find is not finite");
        }
        i += 1;
    }
    // The scaled copies of the points.
    var pts: f64[>..] = copy_slice(points.data);
    var qs: f64[>..] = copy_slice(xi.data);
    if rescale {
        var c: i64 = 0;
        while c < dim {
            var lo = pts[c];
            var hi = lo;
            i = 1;
            while i < n {
                if pts[i * dim + c] < lo { lo = pts[i * dim + c]; }
                if pts[i * dim + c] > hi { hi = pts[i * dim + c]; }
                i += 1;
            }
            var sc = hi - lo;
            if sc == 0.0 { sc = 1.0; }
            i = 0;
            while i < n { pts[i * dim + c] = (pts[i * dim + c] - lo) / sc; i += 1; }
            i = 0;
            while i < m { qs[i * dim + c] = (qs[i * dim + c] - lo) / sc; i += 1; }
            c += 1;
        }
    }
    var out: f64[>..] = [];
    if method == "nearest" {
        i = 0;
        while i < m {
            var best: i64 = 0;
            var bd: f64 = -1.0;
            var j: i64 = 0;
            while j < n {
                var s: f64 = 0.0;
                var c: i64 = 0;
                while c < dim {
                    let df = pts[j * dim + c] - qs[i * dim + c];
                    s += df * df;
                    c += 1;
                }
                // A tie in one dimension goes to the lower coordinate,
                // and a tie in more dimensions goes to the lower place.
                if bd < 0.0 || s < bd || (dim == 1 && s == bd && pts[j] < pts[best]) {
                    bd = s;
                    best = j;
                }
                j += 1;
            }
            out.push(values.data[best]);
            i += 1;
        }
        return line_arr(copy(out)), no_fault();
    }
    if dim == 1 {
        let ord = dl_sort_order(pts);
        i = 0;
        while i < m {
            let q = qs[i];
            var r = fill_value;
            if q >= pts[ord[0]] && q <= pts[ord[n - 1]] {
                // The last interval whose left end lies at or below q.
                var k: i64 = 0;
                while k < n - 2 && pts[ord[k + 1]] <= q { k += 1; }
                if n == 1 {
                    r = values.data[ord[0]];
                } else {
                    let x0 = pts[ord[k]];
                    let x1 = pts[ord[k + 1]];
                    let y0 = values.data[ord[k]];
                    let y1 = values.data[ord[k + 1]];
                    if x1 == x0 { r = y0; } else { r = y0 + (y1 - y0) * (q - x0) / (x1 - x0); }
                }
            }
            out.push(r);
            i += 1;
        }
        return line_arr(copy(out)), no_fault();
    }
    if dim != 2 {
        return empty_arr(), fault(F_LIMIT, "griddata", "the method linear holds one or two dimensions");
    }
    var sh: i64[..8] = [];
    sh.push(n);
    sh.push(2);
    let d, fd = delaunay(Arr { ndim: 2, shape: sh, data: copy(pts) });
    if fd.kind != F_NONE { return empty_arr(), fd; }
    var cs: f64[>..] = [0.0, 0.0, 0.0];
    i = 0;
    while i < m {
        let t = dl_locate(d, qs[i * 2], qs[i * 2 + 1], cs);
        if t < 0 {
            out.push(fill_value);
        } else {
            out.push(cs[0] * values.data[dl_corner(d, t * 3)] + cs[1] * values.data[dl_corner(d, t * 3 + 1)] + cs[2] * values.data[dl_corner(d, t * 3 + 2)]);
        }
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}
