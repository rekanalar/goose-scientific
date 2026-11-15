// The radial basis interpolation of scipy.interpolate: the legacy Rbf
// class as the call rbf, and the RBFInterpolator class as the call
// rbf_interpolate. The kernels and the system follow the source of
// SciPy 1.18.1. The sign of a kernel cancels between the system and
// the evaluation, so the two classes share one kernel table.
namespace scientific;

import mathshim;

// The count of points that the RBF system takes. The kernel matrix
// holds the square of it.
const MAX_RBF_POINTS: i64 = 2000;

// rbf_kernel_is tells whether the name holds a kernel of the
// interpolator of SciPy.
fn rbf_kernel_is(name: const u8[:]) -> bool {
    return name == "linear" || name == "thin_plate_spline" || name == "cubic"
        || name == "quintic" || name == "multiquadric"
        || name == "inverse_multiquadric" || name == "inverse_quadratic"
        || name == "gaussian";
}

// rbf_kernel_scale_invariant tells whether the kernel needs no shape
// parameter. The other kernels take the shape into the distance.
fn rbf_kernel_scale_invariant(name: const u8[:]) -> bool {
    return name == "linear" || name == "thin_plate_spline" || name == "cubic"
        || name == "quintic";
}

// rbf_min_degree gives the least degree of the added polynomial that
// keeps the system well posed. A kernel of no need holds -1.
fn rbf_min_degree(name: const u8[:]) -> i64 {
    if name == "multiquadric" { return 0; }
    if name == "linear" { return 0; }
    if name == "thin_plate_spline" { return 1; }
    if name == "cubic" { return 1; }
    if name == "quintic" { return 2; }
    return -1;
}

// rbf_kernel_value gives the value of the kernel at the distance r.
// The scale s multiplies the distance for the kernels that need the
// shape parameter. The thin plate spline holds zero at r of zero.
fn rbf_kernel_value(name: const u8[:], r: f64, s: f64) -> f64 {
    var t = r;
    if !rbf_kernel_scale_invariant(name) { t = r * s; }
    if name == "linear" { return -r; }
    if name == "thin_plate_spline" {
        if t == 0.0 { return 0.0; }
        return t * t * mathshim::ln(t);
    }
    if name == "cubic" { return r * r * r; }
    if name == "quintic" { return -(r * r * r * r * r); }
    if name == "multiquadric" { return -mathshim::root(t * t + 1.0); }
    if name == "inverse_multiquadric" { return 1.0 / mathshim::root(t * t + 1.0); }
    if name == "inverse_quadratic" { return 1.0 / (t * t + 1.0); }
    return mathshim::power_e(-(t * t));
}

// rbf_count_monos gives the count of the monomials of a degree in a
// count of variables. It holds comb(degree + dim, dim).
fn rbf_count_monos(dim: i64, degree: i64) -> i64 {
    if degree < 0 || dim < 1 { return 0; }
    var out: i64 = 1;
    var i: i64 = 1;
    while i <= degree {
        out = out * (dim + i) / i;
        i += 1;
    }
    return out;
}

// mono_emit walks the monomials of one degree in the order of SciPy.
// The buffer cur holds the powers of the part that the walk holds.
recursive fn mono_emit(out: f64[>..]&, cur: f64[>..]&, dim: i64, start: i64, left: i64) {
    if left == 0 {
        var k: i64 = 0;
        while k < dim { out.push(cur[k]); k += 1; }
        return;
    }
    var v: i64 = start;
    while v < dim {
        cur[v] += 1.0;
        mono_emit(out, cur, dim, v, left - 1);
        cur[v] -= 1.0;
        v += 1;
    }
}

// mono_powers gives the powers of every monomial of a degree in a
// count of variables, one row of dim values for each monomial, in the
// order of combinations with replacement that SciPy enumerates.
fn mono_powers(dim: i64, degree: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var cur: f64[>..] = [];
    var c: i64 = 0;
    while c < dim { cur.push(0.0); c += 1; }
    var deg: i64 = 0;
    while deg <= degree {
        mono_emit(out, cur, dim, 0, deg);
        deg += 1;
    }
    return out;
}

// rbf_mono_value gives the product of the powers of a row of a point.
fn rbf_mono_value(p: f64[:], at: i64, powers: f64[:], row: i64, dim: i64) -> f64 {
    var v: f64 = 1.0;
    var c: i64 = 0;
    while c < dim {
        let q = powers[row * dim + c];
        if q != 0.0 { v *= mathshim::power(p[at + c], q); }
        c += 1;
    }
    return v;
}

// rbf_dist gives the distance between two rows of a matrix.
fn rbf_dist(a: f64[:], i: i64, b: f64[:], j: i64, dim: i64) -> f64 {
    var s: f64 = 0.0;
    var c: i64 = 0;
    while c < dim {
        let df = a[i * dim + c] - b[j * dim + c];
        s += df * df;
        c += 1;
    }
    return mathshim::root(s);
}

// rbf_build_system builds and solves the system of the interpolator.
// The rows of ys hold the centers, the rows of ds hold the values of S
// columns. The result holds the coefficient of every center and every
// monomial, in the order of the rows of the system.
fn rbf_build_system(ys: f64[:], n: i64, dim: i64, ds: f64[:], s: i64,
                    smoothing: f64, kernel: const u8[:], epsilon: f64,
                    powers: f64[:], r: i64, tag: const u8[:]) -> f64[>..], f64[>..], f64[>..], Fault {
    let size = n + r;
    // The domain of the polynomial moves to the box from -1 to 1.
    var shift: f64[>..] = [];
    var scale: f64[>..] = [];
    var c: i64 = 0;
    while c < dim {
        var lo: f64 = ys[c];
        var hi: f64 = ys[c];
        var i: i64 = 1;
        while i < n {
            let v = ys[i * dim + c];
            if v < lo { lo = v; }
            if v > hi { hi = v; }
            i += 1;
        }
        shift.push((hi + lo) / 2.0);
        var w = (hi - lo) / 2.0;
        if w == 0.0 { w = 1.0; }
        scale.push(w);
        c += 1;
    }
    var lhs: f64[>..] = [];
    var i: i64 = 0;
    while i < size * size { lhs.push(0.0); i += 1; }
    // The matrix of the kernels holds the distances of the scaled
    // centers. The matrix is symmetric.
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j <= i {
            let v = rbf_kernel_value(kernel, rbf_dist(ys, i, ys, j, dim), epsilon);
            lhs[i * size + j] = v;
            lhs[j * size + i] = v;
            j += 1;
        }
        lhs[i * size + i] += smoothing;
        i += 1;
    }
    // The matrix of the monomials of the moved centers. The transpose
    // holds the same values, so the condition on the weights sits in
    // the lower rows.
    var yhat: f64[>..] = [];
    i = 0;
    while i < n * dim { yhat.push(0.0); i += 1; }
    i = 0;
    while i < n {
        c = 0;
        while c < dim {
            yhat[i * dim + c] = (ys[i * dim + c] - shift[c]) / scale[c];
            c += 1;
        }
        i += 1;
    }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < r {
            let v = rbf_mono_value(yhat, i * dim, powers, j, dim);
            lhs[i * size + n + j] = v;
            lhs[(n + j) * size + i] = v;
            j += 1;
        }
        i += 1;
    }
    var rhs: f64[>..] = [];
    i = 0;
    while i < size * s { rhs.push(0.0); i += 1; }
    i = 0;
    while i < n * s { rhs[i] = ds[i]; i += 1; }
    var sa: i64[..8] = [];
    sa.push(size);
    sa.push(size);
    var sb: i64[..8] = [];
    sb.push(size);
    sb.push(s);
    let sol, f = solve(Arr { ndim: 2, shape: sa, data: copy(lhs) }, Arr { ndim: 2, shape: sb, data: copy(rhs) });
    if f.kind != F_NONE {
        return [], [], [], fault(F_VALUE, tag, "the system of the interpolant holds no solution");
    }
    var out: f64[>..] = [];
    i = 0;
    while i < size * s { out.push(sol.data[i]); i += 1; }
    return out, shift, scale, no_fault();
}

// rbf_eval_rows evaluates the interpolant at every row of xs. The
// result lands in out, with s values for each row.
fn rbf_eval_rows(xs: f64[:], m: i64, ys: f64[:], n: i64, dim: i64,
                 kernel: const u8[:], epsilon: f64, powers: f64[:], r: i64,
                 shift: f64[:], scale: f64[:], coeffs: f64[:], s: i64,
                 out: f64[>..]&, off: i64) {
    var i: i64 = 0;
    while i < m {
        var col: i64 = 0;
        while col < s { out[off + i * s + col] = 0.0; col += 1; }
        var j: i64 = 0;
        while j < n {
            let v = rbf_kernel_value(kernel, rbf_dist(xs, i, ys, j, dim), epsilon);
            col = 0;
            while col < s {
                out[off + i * s + col] += v * coeffs[j * s + col];
                col += 1;
            }
            j += 1;
        }
        // The monomials of the moved point.
        j = 0;
        while j < r {
            var p: f64 = 1.0;
            var c: i64 = 0;
            while c < dim {
                let q = powers[j * dim + c];
                if q != 0.0 {
                    p *= mathshim::power((xs[i * dim + c] - shift[c]) / scale[c], q);
                }
                c += 1;
            }
            col = 0;
            while col < s {
                out[off + i * s + col] += p * coeffs[(n + j) * s + col];
                col += 1;
            }
            j += 1;
        }
        i += 1;
    }
}

// rbf_interpolate follows scipy.interpolate.RBFInterpolator: the
// radial basis interpolant of scattered data. The operand y holds a
// row of dim coordinates for each of n points, the operand d holds one
// value or a row of values for each point, and the operand x holds the
// points of the query. The neighbors of -1 use every point. A count of
// neighbors above zero uses the nearest points of each query. The
// kernel takes one of the eight names of the reference. The epsilon of
// -1 takes the default of the kernel, which needs no shape. The degree
// of -2 takes the least degree of the kernel, and the degree of -1
// adds no polynomial.
fn rbf_interpolate(y, d, x, neighbors: i64 = -1, smoothing: f64 = 0.0,
                   kernel: const u8[:] = "thin_plate_spline",
                   epsilon: f64 = -1.0, degree: i64 = -2) -> Arr, Fault {
    if !rbf_kernel_is(kernel) {
        return empty_arr(), fault(F_VALUE, "rbf_interpolate", "the kernel holds none of the eight names");
    }
    let f = check_stack(y.ndim, y.shape, y.data.len, 1, "rbf_interpolate");
    if f.kind != F_NONE { return empty_arr(), f; }
    let fx = check_stack(x.ndim, x.shape, x.data.len, 1, "rbf_interpolate");
    if fx.kind != F_NONE { return empty_arr(), fx; }
    let fd = check_stack(d.ndim, d.shape, d.data.len, 1, "rbf_interpolate");
    if fd.kind != F_NONE { return empty_arr(), fd; }
    if y.ndim != 2 || y.shape[1] < 1 {
        return empty_arr(), fault(F_SHAPE, "rbf_interpolate", "the centers are not a matrix with one column or more");
    }
    let n = y.shape[0];
    let dim = y.shape[1];
    if x.ndim != 2 || x.shape[1] != dim {
        return empty_arr(), fault(F_SHAPE, "rbf_interpolate", "the points of the query do not agree with the dimension");
    }
    let m = x.shape[0];
    if d.ndim < 1 || d.ndim > 2 || d.shape[0] != n {
        return empty_arr(), fault(F_SHAPE, "rbf_interpolate", "the values hold no row for each center");
    }
    var sv: i64 = 1;
    if d.ndim == 2 { sv = d.shape[1]; }
    if n > MAX_RBF_POINTS {
        return empty_arr(), fault(F_LIMIT, "rbf_interpolate", "the centers hold more points than the limit");
    }
    var i: i64 = 0;
    while i < n * dim {
        if !(y.data[i] - y.data[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "rbf_interpolate", "a center is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < m * dim {
        if !(x.data[i] - x.data[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "rbf_interpolate", "a point of the query is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < n * sv {
        if !(d.data[i] - d.data[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "rbf_interpolate", "a value is not finite");
        }
        i += 1;
    }
    // The shape parameter. The kernels of no need take one.
    var eps: f64 = epsilon;
    if eps < 0.0 {
        if rbf_kernel_scale_invariant(kernel) {
            eps = 1.0;
        } else {
            return empty_arr(), fault(F_VALUE, "rbf_interpolate", "the kernel needs a shape parameter above zero");
        }
    }
    if eps == 0.0 || !(eps - eps == 0.0) {
        return empty_arr(), fault(F_VALUE, "rbf_interpolate", "the shape parameter is not a positive width");
    }
    // The degree of the added polynomial.
    var deg = degree;
    if deg < -2 {
        return empty_arr(), fault(F_RANGE, "rbf_interpolate", "the degree stands below -2");
    }
    if deg == -2 {
        deg = rbf_min_degree(kernel);
        if deg < 0 { deg = 0; }
    }
    if deg > 512 {
        return empty_arr(), fault(F_LIMIT, "rbf_interpolate", "the degree stands above the limit");
    }
    var r: i64 = 0;
    if deg >= 0 { r = rbf_count_monos(dim, deg); }
    var nobs = n;
    if neighbors != -1 {
        if neighbors < 0 {
            return empty_arr(), fault(F_RANGE, "rbf_interpolate", "the count of the neighbors stands below -1");
        }
        nobs = neighbors;
        if nobs > n { nobs = n; }
        if nobs < 1 {
            return empty_arr(), fault(F_RANGE, "rbf_interpolate", "the count of the neighbors holds no point");
        }
    }
    if r > nobs {
        return empty_arr(), fault(F_SHAPE, "rbf_interpolate", "the centers hold fewer points than the monomials need");
    }
    let powers = mono_powers(dim, deg);
    var out: f64[>..] = [];
    i = 0;
    while i < m * sv { out.push(0.0); i += 1; }
    if neighbors == -1 {
        let coeffs, shift, scale, fb = rbf_build_system(y.data, n, dim, d.data, sv, smoothing,
                                                        kernel, eps, powers, r, "rbf_interpolate");
        if fb.kind != F_NONE { return empty_arr(), fb; }
        rbf_eval_rows(x.data, m, y.data, n, dim, kernel, eps, powers, r,
                      shift, scale, coeffs, sv, out, 0);
    } else {
        // The nearest points of each query. The system of each point
        // holds only the rows of its neighborhood.
        var idx: i64[>..] = [];
        var dist: f64[>..] = [];
        var ys: f64[>..] = [];
        var ds: f64[>..] = [];
        i = 0;
        while i < m {
            idx.clear();
            dist.clear();
            var j: i64 = 0;
            while j < n { idx.push(j); dist.push(0.0); j += 1; }
            j = 0;
            while j < n { dist[j] = rbf_dist(x.data, i, y.data, j, dim); j += 1; }
            // The selection sort of the first nobs places.
            j = 0;
            while j < nobs {
                var best = j;
                var q: i64 = j + 1;
                while q < n {
                    if dist[q] < dist[best] { best = q; }
                    q += 1;
                }
                let ti = idx[j];
                idx[j] = idx[best];
                idx[best] = ti;
                let td = dist[j];
                dist[j] = dist[best];
                dist[best] = td;
                j += 1;
            }
            ys.clear();
            ds.clear();
            j = 0;
            while j < nobs {
                var c: i64 = 0;
                while c < dim { ys.push(y.data[idx[j] * dim + c]); c += 1; }
                c = 0;
                while c < sv { ds.push(d.data[idx[j] * sv + c]); c += 1; }
                j += 1;
            }
            let coeffs, shift, scale, fb = rbf_build_system(ys, nobs, dim, ds, sv, smoothing,
                                                            kernel, eps, powers, r, "rbf_interpolate");
            if fb.kind != F_NONE { return empty_arr(), fb; }
            // The slice of the query starts at the row of the point.
            rbf_eval_rows(x.data[i * dim..], 1, ys, nobs, dim, kernel, eps, powers, r,
                          shift, scale, coeffs, sv, out, i * sv);
            i += 1;
        }
    }
    var sh: i64[..8] = [];
    sh.push(m);
    if d.ndim == 2 { sh.push(sv); }
    return Arr { ndim: sh.len, shape: sh, data: out }, no_fault();
}

// rbf_legacy_is tells whether the name holds a kernel of the legacy
// class of SciPy.
fn rbf_legacy_is(name: const u8[:]) -> bool {
    return name == "multiquadric" || name == "inverse_multiquadric"
        || name == "gaussian" || name == "linear" || name == "cubic"
        || name == "quintic" || name == "thin_plate_spline";
}

// rbf_canonical maps the alias of a name of the legacy Rbf class to
// the name of the kernel table. A name of no kernel holds itself.
fn rbf_canonical(name: const u8[:]) -> const u8[:] {
    if name == "thin_plate" { return "thin_plate_spline"; }
    if name == "thin-plate" { return "thin_plate_spline"; }
    if name == "inverse" { return "inverse_multiquadric"; }
    if name == "inverse multiquadric" { return "inverse_multiquadric"; }
    return name;
}

// rbf follows scipy.interpolate.Rbf: the radial basis interpolation of
// the legacy class. The operand points holds a row of coordinates for
// each of n points, the values hold one value for each point, and the
// operand query holds the points of the query. The function takes one
// of the names multiquadric, inverse_multiquadric, inverse, gaussian,
// linear, cubic, quintic or thin_plate, in lower case. The epsilon of
// no value takes the mean distance of the bounding box. The smooth
// count above zero relaxes the fit of the data.
fn rbf(points, values, query, function: const u8[:] = "multiquadric",
       epsilon: f64 = 0.0 / 0.0, smooth: f64 = 0.0) -> Arr, Fault {
    let name = rbf_canonical(function);
    if !rbf_legacy_is(name) {
        return empty_arr(), fault(F_VALUE, "rbf", "the function holds none of the names of the reference");
    }
    var dim: i64 = 1;
    if points.ndim == 2 { dim = points.shape[1]; }
    if points.ndim > 2 || query.ndim > 2 || dim < 1 {
        return empty_arr(), fault(F_SHAPE, "rbf", "the points or the query hold more than two dimensions");
    }
    let n = points.shape[0];
    let m = query.shape[0];
    if points.ndim < 1 || query.ndim < 1 {
        return empty_arr(), fault(F_SHAPE, "rbf", "the points or the query hold no value");
    }
    // A list holds one coordinate for each point. A matrix holds one
    // column for each coordinate.
    var qdim: i64 = 1;
    if query.ndim == 2 { qdim = query.shape[1]; }
    if qdim != dim {
        return empty_arr(), fault(F_SHAPE, "rbf", "the query holds a different count of coordinates");
    }
    let fp = check_stack(points.ndim, points.shape, points.data.len, 1, "rbf");
    if fp.kind != F_NONE { return empty_arr(), fp; }
    let fq = check_stack(query.ndim, query.shape, query.data.len, 1, "rbf");
    if fq.kind != F_NONE { return empty_arr(), fq; }
    let fv = check_stack(values.ndim, values.shape, values.data.len, 1, "rbf");
    if fv.kind != F_NONE { return empty_arr(), fv; }
    if values.ndim != 1 || values.data.len != n {
        return empty_arr(), fault(F_SHAPE, "rbf", "the values hold a different count than the points");
    }
    if n > MAX_RBF_POINTS {
        return empty_arr(), fault(F_LIMIT, "rbf", "the points hold more points than the limit");
    }
    var i: i64 = 0;
    while i < n * dim {
        if !(points.data[i] - points.data[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "rbf", "a point is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < m * dim {
        if !(query.data[i] - query.data[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "rbf", "a point of the query is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < n {
        if !(values.data[i] - values.data[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "rbf", "a value is not finite");
        }
        i += 1;
    }
    // The default epsilon is the mean distance of the edges of the
    // bounding box, over the edges that hold a width.
    var eps: f64 = epsilon;
    if !(eps - eps == 0.0) {
        var prod: f64 = 1.0;
        var count: i64 = 0;
        var c: i64 = 0;
        while c < dim {
            var lo: f64 = points.data[c];
            var hi: f64 = points.data[c];
            i = 1;
            while i < n {
                let v = points.data[i * dim + c];
                if v < lo { lo = v; }
                if v > hi { hi = v; }
                i += 1;
            }
            let edge = hi - lo;
            if edge > 0.0 { prod *= edge; count += 1; }
            c += 1;
        }
        if count == 0 {
            eps = 1.0;
        } else {
            eps = mathshim::power(prod / (n as f64), 1.0 / (count as f64));
        }
    }
    if !(eps > 0.0) || !(eps - eps == 0.0) {
        return empty_arr(), fault(F_VALUE, "rbf", "the shape parameter is not a positive width");
    }
    let inv = 1.0 / eps;
    // The system holds the kernel of every pair, with the smoothing on
    // the diagonal.
    var lhs: f64[>..] = [];
    i = 0;
    while i < n * n { lhs.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j <= i {
            let v = rbf_kernel_value(name, rbf_dist(points.data, i, points.data, j, dim), inv);
            lhs[i * n + j] = v;
            lhs[j * n + i] = v;
            j += 1;
        }
        lhs[i * n + i] -= smooth;
        i += 1;
    }
    var sa: i64[..8] = [];
    sa.push(n);
    sa.push(n);
    var sb: i64[..8] = [];
    sb.push(n);
    let sol, f = solve(Arr { ndim: 2, shape: sa, data: copy(lhs) },
                       Arr { ndim: 1, shape: sb, data: copy_slice(values.data) });
    if f.kind != F_NONE {
        return empty_arr(), fault(F_VALUE, "rbf", "the system of the interpolant holds no solution");
    }
    var out: f64[>..] = [];
    i = 0;
    while i < m {
        var t: f64 = 0.0;
        var j: i64 = 0;
        while j < n {
            t += rbf_kernel_value(name, rbf_dist(query.data, i, points.data, j, dim), inv) * sol.data[j];
            j += 1;
        }
        out.push(t);
        i += 1;
    }
    var sh: i64[..8] = [];
    sh.push(m);
    return Arr { ndim: 1, shape: sh, data: out }, no_fault();
}
