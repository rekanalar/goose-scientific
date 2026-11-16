import scientific from "..";

// The bivariate splines, the smoothing splines and the splines of
// many dimensions: the least squares fit with given knots, the value
// and the integral of the fit, the smoothing spline with a factor,
// the knots of the smoothing fit, the polynomial of Bernstein and the
// spline of two axes over a grid.
fn main() {
    // The data of two axes that fill the square.
    var x: f64[>..] = [];
    var y: f64[>..] = [];
    var z: f64[>..] = [];
    var i: i64 = 0;
    while i < 10 {
        var j: i64 = 0;
        while j < 10 {
            x.push(0.05 + 0.09 * i + 0.004 * j);
            y.push(0.1 + 0.08 * j + 0.003 * i);
            z.push(sin(3.0 * x[i * 10 + j] + 2.0 * y[i * 10 + j]));
            j += 1;
        }
        i += 1;
    }
    // The least squares spline with given knots.
    var tx: f64[>..] = [];
    var ty: f64[>..] = [];
    i = 0;
    while i < 5 { tx.push(0.15 + 0.15 * i); i += 1; }
    i = 0;
    while i < 4 { ty.push(0.2 + 0.18 * i); i += 1; }
    let g, f = scientific::lsq_bivariate_spline(x, y, z, tx, ty, [], 3, 3);
    if f.kind != scientific::F_NONE {
        print("the fit holds a fault ", scientific::kind_name(f.kind), "\n");
        return;
    }
    let t1 = scientific::grid2_knots_x(g);
    let t2 = scientific::grid2_knots_y(g);
    let c = scientific::grid2_coefs(g);
    print("the knots of the first axis ", t1, "\n");
    // The values on a grid of four points and the integral.
    let grid, f2 = scientific::bisplev([0.2, 0.5, 0.8], [0.3, 0.6], t1, t2, c, 3, 3);
    print("the values on the grid ", grid.data, "\n");
    let intg, f3 = scientific::bivariate_integral(0.2, 0.8, 0.2, 0.8, t1, t2, c, 3, 3);
    print("the integral over the square ", intg, "\n");
    // The smoothing spline with a factor.
    var sx: f64[>..] = [];
    var sy: f64[>..] = [];
    i = 0;
    while i < 20 {
        sx.push(0.15 * i);
        sy.push(sin(sx[i]) + 0.1 * (sx[i] * sx[i]));
        i += 1;
    }
    let b, f4 = scientific::make_smoothing_spline(sx, sy, [], 0.5);
    print("the smoothing spline holds ", b.nt - 4, " coefficients\n");
    let v, f5 = scientific::bspline_eval(b, [0.5, 1.5, 2.5]);
    print("the values of the smoothing spline ", v.data, "\n");
    // The knots of the smoothing fit.
    let kn, f6 = scientific::generate_knots(sx, sy, [], 3, 2.0);
    print("the knots of the smoothing fit ", kn, "\n");
    // The polynomial of Bernstein and the spline of two axes.
    let bp, f7 = scientific::bpoly([1.0, 2.0, 1.0, 0.5, 2.0, 1.0, 1.5], [0.0, 1.0, 2.0]);
    let bv, f8 = scientific::bpoly_eval(bp, [0.25, 1.5]);
    print("the values of the polynomial of Bernstein ", bv.data, "\n");
    var t: f64[>..] = [];
    i = 0;
    while i < 4 { t.push(0.0); i += 1; }
    t.push(0.5);
    t.push(1.0);
    i = 0;
    while i < 4 { t.push(1.0); i += 1; }
    var nts: i64[>..] = [8, 8];
    var ks: i64[>..] = [3, 3];
    var cc: f64[>..] = [];
    i = 0;
    while i < 16 { cc.push(0.1 * i - 0.5); i += 1; }
    let nd, f9 = scientific::nd_bspline(t, nts, ks, cc);
    let nv, f10 = scientific::ndspl_eval(nd, [0.4, 0.7], 1, [], true);
    print("the value of the spline of two axes ", nv.data, "\n");
}
