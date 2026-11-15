// The spline of two variables on a grid of points: the object Grid2
// with the knots of the two axes and the coefficients, the fit
// rect_spline and the evaluation rect_spline_eval. The names follow
// scipy.interpolate.RectBivariateSpline, whose knots follow the rule
// of not-a-knot on both axes.
namespace scientific;

// Grid2 holds the degree of the two axes, the count of the knots of
// the two axes, the count of the coefficients of the two axes and one
// buffer. The buffer holds the knots of the first axis, the knots of
// the second axis and the coefficients. A coefficient holds the row of
// the first axis and the column of the second axis, and the rows come
// first.
struct Grid2 { kx: i64, ky: i64, nt_x: i64, nt_y: i64, nc_x: i64, nc_y: i64, buf: f64[>..] }

fn grid2_empty() -> Grid2 {
    return Grid2 { kx: 0, ky: 0, nt_x: 0, nt_y: 0, nc_x: 0, nc_y: 0, buf: [] };
}

// grid2_knots_x gives the knots of the first axis.
fn grid2_knots_x(g) -> f64[>..] {
    var t: f64[>..] = [];
    var i: i64 = 0;
    while i < g.nt_x { t.push(g.buf[i]); i += 1; }
    return t;
}

// grid2_knots_y gives the knots of the second axis.
fn grid2_knots_y(g) -> f64[>..] {
    var t: f64[>..] = [];
    var i: i64 = 0;
    while i < g.nt_y { t.push(g.buf[g.nt_x + i]); i += 1; }
    return t;
}

// grid2_coefs gives the coefficients of the grid. The list holds one
// row of the second axis for each row of the first axis.
fn grid2_coefs(g) -> f64[>..] {
    var c: f64[>..] = [];
    var i: i64 = 0;
    while i < g.nc_x * g.nc_y { c.push(g.buf[g.nt_x + g.nt_y + i]); i += 1; }
    return c;
}

// grid2_make joins the knots and the coefficients into one grid.
fn grid2_make(kx: i64, ky: i64, tx: f64[:], ty: f64[:], c: f64[:]) -> Grid2 {
    var buf: f64[>..] = copy_slice(tx);
    var i: i64 = 0;
    while i < ty.len { buf.push(ty[i]); i += 1; }
    i = 0;
    while i < c.len { buf.push(c[i]); i += 1; }
    return Grid2 { kx: kx, ky: ky, nt_x: tx.len, nt_y: ty.len,
                   nc_x: tx.len - kx - 1, nc_y: ty.len - ky - 1, buf: buf };
}

// rect_spline follows scipy.interpolate.RectBivariateSpline with the
// factor of zero: the spline of the degree kx in the first axis and
// the degree ky in the second axis that passes through the points of
// the grid. The knots follow the rule of not-a-knot on both axes. The
// values hold the rows of the second axis for each row of the first
// axis.
fn rect_spline(x: f64[:], y: f64[:], z: f64[:], kx: i64 = 3, ky: i64 = 3) -> Grid2, Fault {
    if kx < 0 || kx > 20 || ky < 0 || ky > 20 {
        return grid2_empty(), fault(F_RANGE, "rect_spline", "a degree is outside zero to twenty");
    }
    let mx = x.len;
    let my = y.len;
    if mx < kx + 1 || my < ky + 1 {
        return grid2_empty(), fault(F_SHAPE, "rect_spline", "the axes need at least degree plus one points");
    }
    if z.len != mx * my {
        return grid2_empty(), fault(F_SHAPE, "rect_spline", "the values need the product of the counts of the two axes");
    }
    var i: i64 = 0;
    while i < mx {
        if !(x[i] - x[i] == 0.0) {
            return grid2_empty(), fault(F_VALUE, "rect_spline", "a point of the first axis is not finite");
        }
        if i > 0 && !(x[i] > x[i - 1]) {
            return grid2_empty(), fault(F_VALUE, "rect_spline", "the points of the first axis do not rise strictly");
        }
        i += 1;
    }
    i = 0;
    while i < my {
        if !(y[i] - y[i] == 0.0) {
            return grid2_empty(), fault(F_VALUE, "rect_spline", "a point of the second axis is not finite");
        }
        if i > 0 && !(y[i] > y[i - 1]) {
            return grid2_empty(), fault(F_VALUE, "rect_spline", "the points of the second axis do not rise strictly");
        }
        i += 1;
    }
    i = 0;
    while i < z.len {
        if !(z[i] - z[i] == 0.0) {
            return grid2_empty(), fault(F_VALUE, "rect_spline", "a value is not finite");
        }
        i += 1;
    }
    let tx = bs_knots(x, kx);
    let ty = bs_knots(y, ky);
    let nx = tx.len - kx - 1;
    let ny = ty.len - ky - 1;
    // The matrices of the basis of the two axes. A row holds the
    // values of the basis at one point.
    var ax: f64[>..] = [];
    var ay: f64[>..] = [];
    i = 0;
    while i < mx * nx { ax.push(0.0); i += 1; }
    i = 0;
    while i < my * ny { ay.push(0.0); i += 1; }
    i = 0;
    while i < mx {
        let off, row = bs_design_row(tx, kx, x[i]);
        var j: i64 = 0;
        while j <= kx { ax[i * nx + off + j] = row[j]; j += 1; }
        i += 1;
    }
    i = 0;
    while i < my {
        let off, row = bs_design_row(ty, ky, y[i]);
        var j: i64 = 0;
        while j <= ky { ay[i * ny + off + j] = row[j]; j += 1; }
        i += 1;
    }
    // The values of the grid as a matrix with the rows of the second
    // axis.
    var zm: f64[>..] = [];
    i = 0;
    while i < mx {
        var j: i64 = 0;
        while j < my { zm.push(z[i * my + j]); j += 1; }
        i += 1;
    }
    var sax: i64[..8] = [];
    sax.push(mx);
    sax.push(mx);
    var sam: i64[..8] = [];
    sam.push(mx);
    sam.push(my);
    let m, fm = solve(Arr { ndim: 2, shape: sax, data: copy(ax) }, Arr { ndim: 2, shape: sam, data: copy(zm) });
    if fm.kind != F_NONE {
        return grid2_empty(), fault(F_VALUE, "rect_spline", "the matrix of the first axis holds no solution");
    }
    // The transpose of the values, for the second axis.
    var zt: f64[>..] = [];
    i = 0;
    while i < my {
        var j: i64 = 0;
        while j < mx { zt.push(m.data[j * my + i]); j += 1; }
        i += 1;
    }
    var say: i64[..8] = [];
    say.push(my);
    say.push(my);
    var sat: i64[..8] = [];
    sat.push(my);
    sat.push(nx);
    let mt, ft = solve(Arr { ndim: 2, shape: say, data: copy(ay) }, Arr { ndim: 2, shape: sat, data: copy(zt) });
    if ft.kind != F_NONE {
        return grid2_empty(), fault(F_VALUE, "rect_spline", "the matrix of the second axis holds no solution");
    }
    // The coefficients of the grid, with the rows of the first axis
    // first.
    var c: f64[>..] = [];
    i = 0;
    while i < nx {
        var j: i64 = 0;
        while j < ny { c.push(mt.data[j * nx + i]); j += 1; }
        i += 1;
    }
    return grid2_make(kx, ky, tx, ty, c), no_fault();
}

// rect_spline_eval gives the spline of the grid at the points of the
// two axes. The result holds one row for each point of the first axis,
// and each row holds a value for each point of the second axis. The
// orders dx and dy give the derivatives along the two axes.
fn rect_spline_eval(g, xq: f64[:], yq: f64[:], dx: i64 = 0, dy: i64 = 0) -> Arr, Fault {
    if xq.len < 1 || yq.len < 1 {
        return empty_arr(), fault(F_SHAPE, "rect_spline_eval", "the lists of points hold a value each");
    }
    if dx < 0 || dy < 0 {
        return empty_arr(), fault(F_RANGE, "rect_spline_eval", "an order of the derivative is below zero");
    }
    let tx = grid2_knots_x(g);
    let ty = grid2_knots_y(g);
    let c = grid2_coefs(g);
    var i: i64 = 0;
    while i < xq.len {
        if !(xq[i] - xq[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "rect_spline_eval", "a point of the first axis is not finite");
        }
        i += 1;
    }
    i = 0;
    while i < yq.len {
        if !(yq[i] - yq[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "rect_spline_eval", "a point of the second axis is not finite");
        }
        i += 1;
    }
    // The values of the basis of the first axis at each point, with
    // the derivative of the order dx.
    var bx: f64[>..] = [];
    var ox: i64[>..] = [];
    i = 0;
    while i < xq.len {
        let off, row = bs_design_row(tx, g.kx, xq[i]);
        var j: i64 = 0;
        while j <= g.kx {
            if dx == 0 {
                bx.push(row[j]);
            } else {
                var unit: f64[>..] = [];
                var q: i64 = 0;
                while q < g.nc_x {
                    if q == off + j { unit.push(1.0); } else { unit.push(0.0); }
                    q += 1;
                }
                bx.push(bs_deriv_value(tx, unit, g.kx, dx, xq[i]));
            }
            j += 1;
        }
        ox.push(off);
        i += 1;
    }
    var by: f64[>..] = [];
    var oy: i64[>..] = [];
    i = 0;
    while i < yq.len {
        let off, row = bs_design_row(ty, g.ky, yq[i]);
        var j: i64 = 0;
        while j <= g.ky {
            if dy == 0 {
                by.push(row[j]);
            } else {
                var unit: f64[>..] = [];
                var q: i64 = 0;
                while q < g.nc_y {
                    if q == off + j { unit.push(1.0); } else { unit.push(0.0); }
                    q += 1;
                }
                by.push(bs_deriv_value(ty, unit, g.ky, dy, yq[i]));
            }
            j += 1;
        }
        oy.push(off);
        i += 1;
    }
    var out: f64[>..] = [];
    i = 0;
    while i < xq.len {
        var j: i64 = 0;
        while j < yq.len {
            var v: f64 = 0.0;
            var a: i64 = 0;
            while a <= g.kx {
                var b: i64 = 0;
                while b <= g.ky {
                    v += c[(ox[i] + a) * g.nc_y + (oy[j] + b)] * bx[i * (g.kx + 1) + a] * by[j * (g.ky + 1) + b];
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
    return Arr { ndim: 2, shape: sh, data: copy(out) }, no_fault();
}
