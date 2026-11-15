// The Clough-Tocher interpolation of scipy.interpolate in two
// dimensions, as the call clough_tocher_interpolate. The surface is a
// piecewise cubic of the Bezier form on every triangle of the
// Delaunay triangulation. The gradients at the corners come from the
// global method of Nielson and Renka, which lowers the curvature of
// the surface along every edge. The formulas follow the source of
// SciPy 1.18.1.
namespace scientific;

import mathshim;

// ct_vals_of gives the place of the values in the buffer of the
// interpolator. The gradients follow, then the neighbors of every
// triangle.
fn ct_vals_of(n: i64) -> i64 { return 2 * n; }
fn ct_grad_of(n: i64, k: i64) -> i64 { return 2 * n + n * k; }
fn ct_nb_of(n: i64, k: i64) -> i64 { return 2 * n + n * k + 2 * n * k; }

// ct_bary gives the barycentric coordinates of a point of the plane in
// the triangle of the corners a, b and c.
fn ct_bary(p: f64[:], a: i64, b: i64, c: i64, x: f64, y: f64, cs: f64[>..]&) -> bool {
    let ax = p[a * 2];
    let ay = p[a * 2 + 1];
    let det = (p[b * 2] - ax) * (p[c * 2 + 1] - ay) - (p[c * 2] - ax) * (p[b * 2 + 1] - ay);
    if det == 0.0 { return false; }
    cs[1] = ((x - ax) * (p[c * 2 + 1] - ay) - (p[c * 2] - ax) * (y - ay)) / det;
    cs[2] = ((p[b * 2] - ax) * (y - ay) - (x - ax) * (p[b * 2 + 1] - ay)) / det;
    cs[0] = 1.0 - cs[1] - cs[2];
    return true;
}

// ct_buckets fills the edges of the triangulation in buckets of one
// vertex. Every edge of a triangle holds two entries, one for each
// end. The buffer eo holds the other end of every entry and the buffer
// et holds the triangle of every entry. The buckets sort by the other
// end, so the entries of one edge stand side by side.
fn ct_buckets(simp: f64[:], nsimp: i64, n: i64, start: i64[>..]&, eo: i64[>..]&, et: i64[>..]&) {
    var ev: i64[>..] = [];
    var count: i64[>..] = [];
    var i: i64 = 0;
    while i < n + 1 { start.push(0); i += 1; }
    i = 0;
    while i < n { count.push(0); i += 1; }
    var t: i64 = 0;
    while t < nsimp {
        var k: i64 = 0;
        while k < 3 {
            let u = simp[t * 3 + (k + 1) % 3] as i64;
            let v = simp[t * 3 + (k + 2) % 3] as i64;
            ev.push(u); eo.push(v); et.push(t);
            ev.push(v); eo.push(u); et.push(t);
            count[u] += 1;
            count[v] += 1;
            k += 1;
        }
        t += 1;
    }
    // The places of the buckets.
    i = 0;
    while i < n { start[i + 1] = start[i] + count[i]; i += 1; }
    var fill: i64[>..] = [];
    i = 0;
    while i < n { fill.push(start[i]); i += 1; }
    let total = ev.len;
    var bo2: i64[>..] = [];
    var bt2: i64[>..] = [];
    i = 0;
    while i < total { bo2.push(0); bt2.push(0); i += 1; }
    i = 0;
    while i < total {
        let at = fill[ev[i]];
        bo2[at] = eo[i];
        bt2[at] = et[i];
        fill[ev[i]] += 1;
        i += 1;
    }
    i = 0;
    while i < total {
        eo[i] = bo2[i];
        et[i] = bt2[i];
        i += 1;
    }
    // The entries of one vertex sort by the other end. The insertion
    // sort holds a small count of entries per vertex.
    i = 0;
    while i < n {
        var a = start[i];
        let end = start[i + 1];
        while a < end {
            // The entry keeps its place while the others move.
            let key = eo[a];
            let keyt = et[a];
            var j = a - 1;
            while j >= start[i] && eo[j] > key {
                eo[j + 1] = eo[j];
                et[j + 1] = et[j];
                j -= 1;
            }
            eo[j + 1] = key;
            et[j + 1] = keyt;
            a += 1;
        }
        i += 1;
    }
}

// ct_neighbors gives, for every triangle, the triangle that neighbors
// it across each of its three edges, or -1. The bucket of one end of
// the edge holds the entry of both triangles that share it.
fn ct_neighbors(simp: f64[:], nsimp: i64, n: i64, start: i64[:], eo: i64[:], et: i64[:], nb: f64[>..]&) {
    var t: i64 = 0;
    while t < nsimp {
        var k: i64 = 0;
        while k < 3 {
            let u = simp[t * 3 + (k + 1) % 3] as i64;
            let v = simp[t * 3 + (k + 2) % 3] as i64;
            var found = -1;
            var j = start[u];
            while j < start[u + 1] {
                if eo[j] == v && et[j] != t {
                    found = et[j];
                    j = start[u + 1];
                }
                j += 1;
            }
            nb.push(found as f64);
            k += 1;
        }
        t += 1;
    }
}

// ct_adjacency gives the neighbors of every vertex, one entry for each
// edge that ends at the vertex. The entries of one vertex hold no
// repeat, as the set of Qhull holds none.
fn ct_adjacency(n: i64, start: i64[:], eo: i64[:], adj_start: i64[>..]&, adj: i64[>..]&) {
    adj_start.push(0);
    var i: i64 = 0;
    while i < n {
        var j = start[i];
        while j < start[i + 1] {
            if j == start[i] || eo[j] != eo[j - 1] { adj.push(eo[j]); }
            j += 1;
        }
        adj_start.push(adj.len);
        i += 1;
    }
}

// ct_gradients holds the gradients of one column of the values. The
// method of Nielson and Renka lowers the curvature along every edge
// with a Gauss-Seidel walk over the corners. The buffer holds the
// values at its start and the gradients after them.
fn ct_gradients(p: f64[:], n: i64, buf: f64[>..]&, k: i64, col: i64,
                adj_start: i64[:], adj: i64[:], tol: f64, maxiter: i64) {
    let vbase = ct_vals_of(n);
    // The gradients of the column sit at the offset of the column.
    let base = ct_grad_of(n, k) + col * 2;
    var i: i64 = 0;
    while i < n {
        buf[base + i * 2 * k] = 0.0;
        buf[base + i * 2 * k + 1] = 0.0;
        i += 1;
    }
    var turn: i64 = 0;
    while turn < maxiter {
        var err: f64 = 0.0;
        i = 0;
        while i < n {
            var q0: f64 = 0.0;
            var q1: f64 = 0.0;
            var q01: f64 = 0.0;
            var s0: f64 = 0.0;
            var s1: f64 = 0.0;
            var j = adj_start[i];
            while j < adj_start[i + 1] {
                let i2 = adj[j];
                let ex = p[i2 * 2] - p[i * 2];
                let ey = p[i2 * 2 + 1] - p[i * 2 + 1];
                let ln = mathshim::root(ex * ex + ey * ey);
                let l3 = ln * ln * ln;
                let f1 = buf[vbase + i * k + col];
                let f2 = buf[vbase + i2 * k + col];
                let df2 = -ex * buf[base + i2 * 2 * k] - ey * buf[base + i2 * 2 * k + 1];
                q0 += 4.0 * ex * ex / l3;
                q01 += 4.0 * ex * ey / l3;
                q1 += 4.0 * ey * ey / l3;
                s0 += (6.0 * (f1 - f2) - 2.0 * df2) * ex / l3;
                s1 += (6.0 * (f1 - f2) - 2.0 * df2) * ey / l3;
                j += 1;
            }
            let det = q0 * q1 - q01 * q01;
            if det == 0.0 {
                i += 1;
                continue;
            }
            let r0 = (q1 * s0 - q01 * s1) / det;
            let r1 = (-q01 * s0 + q0 * s1) / det;
            var change = buf[base + i * 2 * k] + r0;
            if change < 0.0 { change = -change; }
            var ch1 = buf[base + i * 2 * k + 1] + r1;
            if ch1 < 0.0 { ch1 = -ch1; }
            if ch1 > change { change = ch1; }
            buf[base + i * 2 * k] = -r0;
            buf[base + i * 2 * k + 1] = -r1;
            var ref0 = r0;
            if ref0 < 0.0 { ref0 = -ref0; }
            var ref1 = r1;
            if ref1 < 0.0 { ref1 = -ref1; }
            var refv = ref0;
            if ref1 > refv { refv = ref1; }
            if refv > 1.0 { change /= refv; }
            if change > err { err = change; }
            i += 1;
        }
        if err < tol { return; }
        turn += 1;
    }
}

// clough_tocher_interpolate follows
// scipy.interpolate.CloughTocher2DInterpolator: the piecewise cubic
// with a smooth first derivative on the Delaunay triangles of the
// points. The operand points holds a row of two coordinates for each
// point, the values hold one value or a row of values for each point,
// and the operand xi holds the points of the query. A point outside
// the hull takes the fill value. The flag rescale moves every axis
// into the unit width before the work.
fn clough_tocher_interpolate(points, values, xi, fill_value: f64 = 0.0 / 0.0,
                             tol: f64 = 1.0e-6, maxiter: i64 = 400,
                             rescale: bool = false) -> Arr, Fault {
    let fp = check_stack(points.ndim, points.shape, points.data.len, 1, "clough_tocher_interpolate");
    if fp.kind != F_NONE { return empty_arr(), fp; }
    let fx = check_stack(xi.ndim, xi.shape, xi.data.len, 1, "clough_tocher_interpolate");
    if fx.kind != F_NONE { return empty_arr(), fx; }
    let fv = check_stack(values.ndim, values.shape, values.data.len, 1, "clough_tocher_interpolate");
    if fv.kind != F_NONE { return empty_arr(), fv; }
    if points.ndim != 2 || points.shape[1] != 2 {
        return empty_arr(), fault(F_SHAPE, "clough_tocher_interpolate", "the points are not a matrix with two columns");
    }
    if xi.ndim != 2 || xi.shape[1] != 2 {
        return empty_arr(), fault(F_SHAPE, "clough_tocher_interpolate", "the points of the query are not a matrix with two columns");
    }
    let n = points.shape[0];
    let m = xi.shape[0];
    if values.ndim < 1 || values.ndim > 2 || values.shape[0] != n {
        return empty_arr(), fault(F_SHAPE, "clough_tocher_interpolate", "the values hold no row for each point");
    }
    var k: i64 = 1;
    if values.ndim == 2 { k = values.shape[1]; }
    if n > MAX_DELAUNAY {
        return empty_arr(), fault(F_LIMIT, "clough_tocher_interpolate", "the points hold more points than the limit");
    }
    var i: i64 = 0;
    while i < n * 2 {
        if !(points.data[i] - points.data[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "clough_tocher_interpolate", "a coordinate is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < m * 2 {
        if !(xi.data[i] - xi.data[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "clough_tocher_interpolate", "a point of the query is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < n * k {
        if !(values.data[i] - values.data[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "clough_tocher_interpolate", "a value is not finite");
        }
        i += 1;
    }
    // The rescale moves every axis into the unit width, centered at
    // the mean of the axis. The width of an axis that holds no spread
    // stays at one.
    var pts: f64[>..] = copy_slice(points.data);
    var qs: f64[>..] = copy_slice(xi.data);
    var c: i64 = 0;
    while c < 2 {
        var mean: f64 = 0.0;
        i = 0;
        while i < n { mean += pts[i * 2 + c]; i += 1; }
        mean /= n as f64;
        var lo: f64 = pts[c];
        var hi: f64 = pts[c];
        i = 1;
        while i < n {
            if pts[i * 2 + c] < lo { lo = pts[i * 2 + c]; }
            if pts[i * 2 + c] > hi { hi = pts[i * 2 + c]; }
            i += 1;
        }
        if rescale {
            var w = hi - lo;
            if !(w > 0.0) { w = 1.0; }
            i = 0;
            while i < n { pts[i * 2 + c] = (pts[i * 2 + c] - mean) / w; i += 1; }
            i = 0;
            while i < m { qs[i * 2 + c] = (qs[i * 2 + c] - mean) / w; i += 1; }
        }
        c += 1;
    }
    var sh: i64[..8] = [];
    sh.push(n);
    sh.push(2);
    let tri, f = delaunay(Arr { ndim: 2, shape: sh, data: copy(pts) });
    if f.kind != F_NONE { return empty_arr(), f; }
    // The corners of every triangle, then the edges in the buckets of
    // their ends, the neighbors of the triangles and the corners.
    var simp: f64[>..] = [];
    i = 0;
    while i < tri.nsimp * 3 { simp.push(tri.buf[tri.npts * 2 + i]); i += 1; }
    var start: i64[>..] = [];
    var eo: i64[>..] = [];
    var et: i64[>..] = [];
    ct_buckets(simp, tri.nsimp, n, start, eo, et);
    var nb: f64[>..] = [];
    ct_neighbors(simp, tri.nsimp, n, start, eo, et, nb);
    var adj_start: i64[>..] = [];
    var adj: i64[>..] = [];
    ct_adjacency(n, start, eo, adj_start, adj);
    // The values and the gradients of every column.
    var buf: f64[>..] = [];
    i = 0;
    while i < ct_nb_of(n, k) + tri.nsimp * 3 { buf.push(0.0); i += 1; }
    i = 0;
    while i < n * k { buf[ct_vals_of(n) + i] = values.data[i]; i += 1; }
    i = 0;
    while i < tri.nsimp * 3 { buf[ct_nb_of(n, k) + i] = nb[i]; i += 1; }
    i = 0;
    while i < k {
        ct_gradients(pts, n, buf, k, i, adj_start, adj, tol, maxiter);
        i += 1;
    }
    // The evaluation of every point of the query.
    var out: f64[>..] = [];
    var cs: f64[>..] = [0.0, 0.0, 0.0];
    i = 0;
    while i < m {
        let s = dl_locate(tri, qs[i * 2], qs[i * 2 + 1], cs);
        var col: i64 = 0;
        while col < k {
            if s < 0 {
                out.push(fill_value);
            } else {
                out.push(ct_single(pts, simp, nb, s, cs, buf, n, k, col));
            }
            col += 1;
        }
        i += 1;
    }
    var so: i64[..8] = [];
    so.push(m);
    if values.ndim == 2 { so.push(k); }
    return Arr { ndim: so.len, shape: so, data: out }, no_fault();
}

// ct_single evaluates the Bezier patch of the Clough-Tocher scheme on
// one triangle at the barycentric coordinates cs. The formulas follow
// the source of SciPy.
fn ct_single(p: f64[:], simp: f64[:], nb: f64[:], s: i64, cs: f64[:],
             buf: f64[:], n: i64, k: i64, col: i64) -> f64 {
    let a = simp[s * 3] as i64;
    let b = simp[s * 3 + 1] as i64;
    let d = simp[s * 3 + 2] as i64;
    let e12x = p[b * 2] - p[a * 2];
    let e12y = p[b * 2 + 1] - p[a * 2 + 1];
    let e23x = p[d * 2] - p[b * 2];
    let e23y = p[d * 2 + 1] - p[b * 2 + 1];
    let e31x = p[a * 2] - p[d * 2];
    let e31y = p[a * 2 + 1] - p[d * 2 + 1];
    let vbase = ct_vals_of(n);
    let gbase = ct_grad_of(n, k);
    let f1 = buf[vbase + a * k + col];
    let f2 = buf[vbase + b * k + col];
    let f3 = buf[vbase + d * k + col];
    let ga = gbase + col * 2;
    let df12 = buf[ga + a * 2 * k] * e12x + buf[ga + a * 2 * k + 1] * e12y;
    let df21 = -(buf[ga + b * 2 * k] * e12x + buf[ga + b * 2 * k + 1] * e12y);
    let df23 = buf[ga + b * 2 * k] * e23x + buf[ga + b * 2 * k + 1] * e23y;
    let df32 = -(buf[ga + d * 2 * k] * e23x + buf[ga + d * 2 * k + 1] * e23y);
    let df31 = buf[ga + d * 2 * k] * e31x + buf[ga + d * 2 * k + 1] * e31y;
    let df13 = -(buf[ga + a * 2 * k] * e31x + buf[ga + a * 2 * k + 1] * e31y);
    let c3000 = f1;
    let c2100 = (df12 + 3.0 * c3000) / 3.0;
    let c2010 = (df13 + 3.0 * c3000) / 3.0;
    let c0300 = f2;
    let c1200 = (df21 + 3.0 * c0300) / 3.0;
    let c0210 = (df23 + 3.0 * c0300) / 3.0;
    let c0030 = f3;
    let c1020 = (df31 + 3.0 * c0030) / 3.0;
    let c0120 = (df32 + 3.0 * c0030) / 3.0;
    let c2001 = (c2100 + c2010 + c3000) / 3.0;
    let c0201 = (c1200 + c0300 + c0210) / 3.0;
    let c0021 = (c1020 + c0120 + c0030) / 3.0;
    // The direction of the derivative across every edge follows the
    // centroid of the neighbor triangle. Two neighbors that share an
    // edge hold the same direction, which keeps the surface smooth.
    var g: f64[>..] = [];
    var y: f64[>..] = [0.0, 0.0];
    var cc: f64[>..] = [0.0, 0.0, 0.0];
    var kk: i64 = 0;
    while kk < 3 {
        let itri = nb[s * 3 + kk] as i64;
        if itri < 0 {
            g.push(-0.5);
            kk += 1;
            continue;
        }
        let u = simp[itri * 3] as i64;
        let v = simp[itri * 3 + 1] as i64;
        let w = simp[itri * 3 + 2] as i64;
        y[0] = (p[u * 2] + p[v * 2] + p[w * 2]) / 3.0;
        y[1] = (p[u * 2 + 1] + p[v * 2 + 1] + p[w * 2 + 1]) / 3.0;
        if !ct_bary(p, a, b, d, y[0], y[1], cc) {
            g.push(-0.5);
            kk += 1;
            continue;
        }
        if kk == 0 {
            g.push((2.0 * cc[2] + cc[1] - 1.0) / (2.0 - 3.0 * cc[2] - 3.0 * cc[1]));
        } else if kk == 1 {
            g.push((2.0 * cc[0] + cc[2] - 1.0) / (2.0 - 3.0 * cc[0] - 3.0 * cc[2]));
        } else {
            g.push((2.0 * cc[1] + cc[0] - 1.0) / (2.0 - 3.0 * cc[1] - 3.0 * cc[0]));
        }
        kk += 1;
    }
    let c0111 = (g[0] * (-c0300 + 3.0 * c0210 - 3.0 * c0120 + c0030)
               + (-c0300 + 2.0 * c0210 - c0120 + c0021 + c0201)) / 2.0;
    let c1011 = (g[1] * (-c0030 + 3.0 * c1020 - 3.0 * c2010 + c3000)
               + (-c0030 + 2.0 * c1020 - c2010 + c2001 + c0021)) / 2.0;
    let c1101 = (g[2] * (-c3000 + 3.0 * c2100 - 3.0 * c1200 + c0300)
               + (-c3000 + 2.0 * c2100 - c1200 + c2001 + c0201)) / 2.0;
    let c1002 = (c1101 + c1011 + c2001) / 3.0;
    let c0102 = (c1101 + c0111 + c0201) / 3.0;
    let c0012 = (c1011 + c0111 + c0021) / 3.0;
    let c0003 = (c1002 + c0102 + c0012) / 3.0;
    // The extended coordinates lift a point of the triangle into the
    // cubic of degree three over the four coordinates.
    var minval = cs[0];
    if cs[1] < minval { minval = cs[1]; }
    if cs[2] < minval { minval = cs[2]; }
    let b1 = cs[0] - minval;
    let b2 = cs[1] - minval;
    let b3 = cs[2] - minval;
    let b4 = 3.0 * minval;
    let b12 = b1 * b2;
    let b13 = b1 * b3;
    let b14 = b1 * b4;
    let b23 = b2 * b3;
    let b24 = b2 * b4;
    let b34 = b3 * b4;
    return b1 * b1 * b1 * c3000
        + 3.0 * b1 * b1 * b2 * c2100
        + 3.0 * b1 * b1 * b3 * c2010
        + 3.0 * b1 * b1 * b4 * c2001
        + 3.0 * b12 * b2 * c1200
        + 6.0 * b12 * b4 * c1101
        + 3.0 * b13 * b3 * c1020
        + 6.0 * b13 * b4 * c1011
        + 3.0 * b14 * b4 * c1002
        + b2 * b2 * b2 * c0300
        + 3.0 * b2 * b2 * b3 * c0210
        + 3.0 * b2 * b2 * b4 * c0201
        + 3.0 * b23 * b3 * c0120
        + 6.0 * b23 * b4 * c0111
        + 3.0 * b24 * b4 * c0102
        + b3 * b3 * b3 * c0030
        + 3.0 * b3 * b3 * b4 * c0021
        + 3.0 * b34 * b4 * c0012
        + b4 * b4 * b4 * c0003;
}
