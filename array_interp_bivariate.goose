// The bivariate spline classes of SciPy: the holder of the knots and
// the coefficients of a spline of two variables, the least squares fit
// lsq_bivariate_spline, and the integral over a rectangle. The names
// follow scipy.interpolate.BivariateSpline and
// scipy.interpolate.LSQBivariateSpline. The holder is the type Grid2
// of the package, whose evaluation is rect_spline_eval and whose
// scattered evaluation and rectangle integral are bivariate_ev and
// bivariate_integral.
namespace scientific;

import mathshim;

// bivariate_spline holds the knots and the coefficients of a
// bivariate spline, as the class BivariateSpline of SciPy holds them.
// The coefficients hold the rows of the first axis first.
fn bivariate_spline(tx: f64[:], ty: f64[:], c: f64[:], kx: i64, ky: i64) -> Grid2, Fault {
    if kx < 1 || kx > 5 || ky < 1 || ky > 5 {
        return grid2_empty(), fault(F_RANGE, "bivariate_spline", "a degree is outside one to five");
    }
    let f = biv_counts(tx, ty, c, kx, ky, "bivariate_spline");
    if f.kind != F_NONE { return grid2_empty(), f; }
    return grid2_make(kx, ky, tx, ty, c), no_fault();
}

// lsq_calc_b gives the lower boundary of the fit. It follows the rule
// of the interface of SciPy: the smaller of the first knot and the
// first point, or, when the knots stand above the data, a step below
// the first knot.
fn lsq_calc_b(data: f64[:], knots: f64[:]) -> f64 {
    let val1 = min_of(data);
    let val2 = min_of(knots);
    if val2 > val1 { return val1; }
    let val_max = max_of(knots);
    return val2 - (val_max - val2) / (knots.len as f64);
}

// lsq_calc_e gives the upper boundary of the fit by the same rule.
fn lsq_calc_e(data: f64[:], knots: f64[:]) -> f64 {
    let val1 = max_of(data);
    let val2 = max_of(knots);
    if val2 < val1 { return val1; }
    let val_min = min_of(knots);
    return val2 + (val2 - val_min) / (knots.len as f64);
}

// min_of gives the smallest value of a buffer.
fn min_of(data: f64[:]) -> f64 {
    var v = data[0];
    var i: i64 = 1;
    while i < data.len {
        if data[i] < v { v = data[i]; }
        i += 1;
    }
    return v;
}

// max_of gives the largest value of a buffer.
fn max_of(data: f64[:]) -> f64 {
    var v = data[0];
    var i: i64 = 1;
    while i < data.len {
        if data[i] > v { v = data[i]; }
        i += 1;
    }
    return v;
}

// lsq_bivariate_spline follows scipy.interpolate.LSQBivariateSpline:
// the weighted least squares spline of degrees kx and ky with the
// knots tx and ty of the caller. The knots hold the interior knots
// only. The boundary of the fit takes its rule from the data and the
// knots, as the default of SciPy does. The weights hold one for an
// empty list.
fn lsq_bivariate_spline(x: f64[:], y: f64[:], z: f64[:], tx: f64[:], ty: f64[:], w: f64[:], kx: i64, ky: i64) -> Grid2, Fault {
    if kx < 1 || kx > 5 || ky < 1 || ky > 5 {
        return grid2_empty(), fault(F_RANGE, "lsq_bivariate_spline", "a degree is outside one to five");
    }
    let m = x.len;
    if y.len != m || z.len != m {
        return grid2_empty(), fault(F_SHAPE, "lsq_bivariate_spline", "the points and the values need one count");
    }
    if w.len != 0 && w.len != m {
        return grid2_empty(), fault(F_SHAPE, "lsq_bivariate_spline", "the weights need the count of the points");
    }
    if m < (kx + 1) * (ky + 1) {
        return grid2_empty(), fault(F_SHAPE, "lsq_bivariate_spline", "the data need at least the product of the degrees plus one points");
    }
    var i: i64 = 0;
    while i < m {
        if !(x[i] - x[i] == 0.0) || !(y[i] - y[i] == 0.0) || !(z[i] - z[i] == 0.0) {
            return grid2_empty(), fault(F_VALUE, "lsq_bivariate_spline", "a point or a value is not finite");
        }
        if w.len > 0 && (!(w[i] - w[i] == 0.0) || w[i] < 0.0) {
            return grid2_empty(), fault(F_VALUE, "lsq_bivariate_spline", "a weight is not a number above zero or zero");
        }
        i += 1;
    }
    i = 0;
    while i < tx.len {
        if !(tx[i] - tx[i] == 0.0) {
            return grid2_empty(), fault(F_VALUE, "lsq_bivariate_spline", "a knot of the first axis is not finite");
        }
        if i > 0 && tx[i] <= tx[i - 1] {
            return grid2_empty(), fault(F_VALUE, "lsq_bivariate_spline", "the knots of the first axis do not rise strictly");
        }
        i += 1;
    }
    i = 0;
    while i < ty.len {
        if !(ty[i] - ty[i] == 0.0) {
            return grid2_empty(), fault(F_VALUE, "lsq_bivariate_spline", "a knot of the second axis is not finite");
        }
        if i > 0 && ty[i] <= ty[i - 1] {
            return grid2_empty(), fault(F_VALUE, "lsq_bivariate_spline", "the knots of the second axis do not rise strictly");
        }
        i += 1;
    }
    // The boundary follows the rule of the interface of SciPy, which
    // reads the knots of the caller in the padded form, with the
    // zeros that fill the ends.
    let nx = 2 * kx + 2 + tx.len;
    let ny = 2 * ky + 2 + ty.len;
    let nmax = if nx > ny { nx } else { ny };
    var tpx: f64[>..] = [];
    var tpy: f64[>..] = [];
    i = 0;
    while i < nmax { tpx.push(0.0); tpy.push(0.0); i += 1; }
    i = 0;
    while i < tx.len { tpx[kx + 1 + i] = tx[i]; i += 1; }
    i = 0;
    while i < ty.len { tpy[ky + 1 + i] = ty[i]; i += 1; }
    let xb = lsq_calc_b(x, tpx);
    let xe = lsq_calc_e(x, tpx);
    let yb = lsq_calc_b(y, tpy);
    let ye = lsq_calc_e(y, tpy);
    // The full knot vectors hold the boundary kx plus one times at
    // each end.
    var t1: f64[>..] = [];
    i = 0;
    while i < kx + 1 { t1.push(xb); i += 1; }
    i = 0;
    while i < tx.len { t1.push(tx[i]); i += 1; }
    i = 0;
    while i < kx + 1 { t1.push(xe); i += 1; }
    var t2: f64[>..] = [];
    i = 0;
    while i < ky + 1 { t2.push(yb); i += 1; }
    i = 0;
    while i < ty.len { t2.push(ty[i]); i += 1; }
    i = 0;
    while i < ky + 1 { t2.push(ye); i += 1; }
    let ncx = t1.len - kx - 1;
    let ncy = t2.len - ky - 1;
    // The normal equations of the weighted least squares fit. The
    // row of a point holds the products of the basis values of the
    // two axes, with the rows of the first axis first.
    var normal: f64[>..] = [];
    var rhs: f64[>..] = [];
    i = 0;
    while i < ncx * ncy * ncx * ncy { normal.push(0.0); i += 1; }
    i = 0;
    while i < ncx * ncy { rhs.push(0.0); i += 1; }
    i = 0;
    while i < m {
        var wi: f64 = 1.0;
        if w.len > 0 { wi = w[i]; }
        let offx, rowx = bs_design_row(t1, kx, x[i]);
        let offy, rowy = bs_design_row(t2, ky, y[i]);
        var a: i64 = 0;
        while a <= kx {
            var b: i64 = 0;
            while b <= ky {
                let vab = wi * rowx[a] * rowy[b];
                rhs[(offx + a) * ncy + (offy + b)] += vab * wi * z[i];
                var c: i64 = 0;
                while c <= kx {
                    var d: i64 = 0;
                    while d <= ky {
                        let vu = wi * rowx[c] * rowy[d];
                        normal[((offx + a) * ncy + (offy + b)) * (ncx * ncy) + ((offx + c) * ncy + (offy + d))] += vab * vu;
                        d += 1;
                    }
                    c += 1;
                }
                b += 1;
            }
            a += 1;
        }
        i += 1;
    }
    var sa: i64[..8] = [];
    sa.push(ncx * ncy);
    sa.push(ncx * ncy);
    var sb: i64[..8] = [];
    sb.push(ncx * ncy);
    let sol, f2 = solve(Arr { ndim: 2, shape: sa, data: copy(normal) }, Arr { ndim: 1, shape: sb, data: copy(rhs) });
    if f2.kind != F_NONE {
        return grid2_empty(), fault(F_VALUE, "lsq_bivariate_spline", "the system of the fit holds no solution");
    }
    return grid2_make(kx, ky, t1, t2, sol.data), no_fault();
}
