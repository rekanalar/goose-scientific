// The bivariate spline of the tck form: the evaluation bisplev that
// follows scipy.interpolate.bisplev, the evaluation at single points
// that follows the grid=false path of the BivariateSpline call, and
// the integral over a rectangle that follows dblint. The spline holds
// the knots of the two axes, the coefficients and the two degrees.
// The coefficients hold the rows of the first axis first.
namespace scientific;

import mathshim;

// biv_counts checks the counts of a tck spline. The count of the
// coefficients must equal the product of the counts of the basis
// functions of the two axes.
fn biv_counts(tx: f64[:], ty: f64[:], c: f64[:], kx: i64, ky: i64, tag: const u8[:]) -> Fault {
    if kx < 0 || ky < 0 {
        return fault(F_RANGE, tag, "a degree is below zero");
    }
    if tx.len < kx + 2 || ty.len < ky + 2 {
        return fault(F_SHAPE, tag, "the knots hold too few values for the degree");
    }
    let nx = tx.len - kx - 1;
    let ny = ty.len - ky - 1;
    if c.len != nx * ny {
        return fault(F_SHAPE, tag, "the coefficients need the product of the counts of the basis functions");
    }
    var i: i64 = 0;
    while i < tx.len {
        if !(tx[i] - tx[i] == 0.0) {
            return fault(F_VALUE, tag, "a knot of the first axis is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < ty.len {
        if !(ty[i] - ty[i] == 0.0) {
            return fault(F_VALUE, tag, "a knot of the second axis is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < c.len {
        if !(c[i] - c[i] == 0.0) {
            return fault(F_VALUE, tag, "a coefficient is not finite");
        }
        i += 1;
    }
    return no_fault();
}

// biv_basis_row gives the k plus one values of the basis of one axis
// at a point, with the derivative of the order order. The values go
// into out, and the result is the column of the first value.
fn biv_basis_row(t: f64[:], k: i64, order: i64, v: f64, out: f64[>..]&) -> i64 {
    let off, row = bs_design_row(t, k, v);
    let n = t.len - k - 1;
    var j: i64 = 0;
    while j <= k {
        if order == 0 {
            out.push(row[j]);
        } else {
            var unit: f64[>..] = [];
            var q: i64 = 0;
            while q < n {
                if q == off + j { unit.push(1.0); } else { unit.push(0.0); }
                q += 1;
            }
            out.push(bs_deriv_value(t, unit, k, order, v));
        }
        j += 1;
    }
    return off;
}

// bisplev gives the values of the bivariate spline of the tck form.
// With grid true the result holds one row for each point of the first
// axis and one column for each point of the second axis, as the
// bisplev of SciPy gives. With grid false the two lists pair into
// points, as the call of BivariateSpline with grid=false gives. The
// orders dx and dy give the derivatives along the two axes.
fn bisplev(xq: f64[:], yq: f64[:], tx: f64[:], ty: f64[:], c: f64[:], kx: i64, ky: i64, grid: bool = true, dx: i64 = 0, dy: i64 = 0) -> Arr, Fault {
    let f = biv_counts(tx, ty, c, kx, ky, "bisplev");
    if f.kind != F_NONE { return empty_arr(), f; }
    if dx < 0 || dy < 0 {
        return empty_arr(), fault(F_RANGE, "bisplev", "an order of the derivative is below zero");
    }
    if xq.len < 1 || yq.len < 1 {
        return empty_arr(), fault(F_SHAPE, "bisplev", "the lists of points hold a value each");
    }
    if !grid && xq.len != yq.len {
        return empty_arr(), fault(F_SHAPE, "bisplev", "the points of the two axes need one count");
    }
    var i: i64 = 0;
    while i < xq.len {
        if !(xq[i] - xq[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "bisplev", "a point of the first axis is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < yq.len {
        if !(yq[i] - yq[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "bisplev", "a point of the second axis is not finite");
        }
        i += 1;
    }
    let nx = tx.len - kx - 1;
    let ny = ty.len - ky - 1;
    // The values of the basis of the first axis at each point of the
    // query, with the derivative of the order dx.
    var bx: f64[>..] = [];
    var ox: i64[>..] = [];
    i = 0;
    while i < xq.len {
        let off = biv_basis_row(tx, kx, dx, xq[i], bx);
        ox.push(off);
        i += 1;
    }
    var by: f64[>..] = [];
    var oy: i64[>..] = [];
    i = 0;
    while i < yq.len {
        let off = biv_basis_row(ty, ky, dy, yq[i], by);
        oy.push(off);
        i += 1;
    }
    var out: f64[>..] = [];
    if grid {
        i = 0;
        while i < xq.len {
            var j: i64 = 0;
            while j < yq.len {
                var v: f64 = 0.0;
                var a: i64 = 0;
                while a <= kx {
                    var b: i64 = 0;
                    while b <= ky {
                        v += c[(ox[i] + a) * ny + (oy[j] + b)] * bx[i * (kx + 1) + a] * by[j * (ky + 1) + b];
                        b += 1;
                    }
                    a += 1;
                }
                out.push(v);
                j += 1;
            }
            i += 1;
        }
        var sh: i64[..8] = [];
        sh.push(xq.len);
        sh.push(yq.len);
        return arr_of(copy(out), sh), no_fault();
    }
    i = 0;
    while i < xq.len {
        var v: f64 = 0.0;
        var a: i64 = 0;
        while a <= kx {
            var b: i64 = 0;
            while b <= ky {
                v += c[(ox[i] + a) * ny + (oy[i] + b)] * bx[i * (kx + 1) + a] * by[i * (ky + 1) + b];
                b += 1;
            }
            a += 1;
        }
        out.push(v);
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// biv_basis_int gives the integral of the basis function j of the
// knots t and the degree k between a and b. The integral goes through
// the coefficients of the antiderivative of the basis function, which
// the rule of the antiderivative of the package gives.
fn biv_basis_int(t: f64[:], k: i64, j: i64, a: f64, b: f64) -> f64 {
    let n = t.len - k - 1;
    if j < 0 || j >= n { return 0.0; }
    // The antiderivative holds one more basis function and one more
    // knot at each end. Only the basis j gives a step.
    var nt: f64[>..] = [];
    nt.push(t[0]);
    var i: i64 = 0;
    while i < t.len { nt.push(t[i]); i += 1; }
    nt.push(t[t.len - 1]);
    var nc: f64[>..] = [];
    i = 0;
    while i < n + 1 {
        if i > j { nc.push((t[j + k + 1] - t[j]) / ((k + 1) as f64)); } else { nc.push(0.0); }
        i += 1;
    }
    let up = bs_value(nt, nc, k + 1, b);
    let dn = bs_value(nt, nc, k + 1, a);
    return up - dn;
}

// bivariate_integral gives the integral of the spline of the tck form
// over the rectangle from xa to xb along the first axis and from ya
// to yb along the second axis. It follows dblint of SciPy: the sum of
// the products of the coefficients and the integrals of the basis
// functions of the two axes.
fn bivariate_integral(xa: f64, xb: f64, ya: f64, yb: f64, tx: f64[:], ty: f64[:], c: f64[:], kx: i64, ky: i64) -> f64, Fault {
    let f = biv_counts(tx, ty, c, kx, ky, "bivariate_integral");
    if f.kind != F_NONE { return 0.0, f; }
    if !(xa - xa == 0.0) || !(xb - xb == 0.0) || !(ya - ya == 0.0) || !(yb - yb == 0.0) {
        return 0.0, fault(F_VALUE, "bivariate_integral", "an end of the rectangle is not finite");
    }
    let nx = tx.len - kx - 1;
    let ny = ty.len - ky - 1;
    var total: f64 = 0.0;
    var i: i64 = 0;
    while i < nx {
        let vx = biv_basis_int(tx, kx, i, xa, xb);
        var j: i64 = 0;
        while j < ny {
            total += vx * biv_basis_int(ty, ky, j, ya, yb) * c[i * ny + j];
            j += 1;
        }
        i += 1;
    }
    return total, no_fault();
}
