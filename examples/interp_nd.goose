import scientific from "..";

// The interpolators of scattered data of the package: the radial
// basis of the legacy class, the interpolator of the radial basis of
// the modern class, the rational approximation of AAA and the
// piecewise cubic of Clough and Tocher.
fn main() {
    // Six scattered points of the plane and their values.
    let xs = [0.0, 1.0, 0.0, 1.0, 0.5, 0.25];
    let ys = [0.0, 0.0, 1.0, 1.0, 0.5, 0.75];
    let zs = [0.0, 1.0, 1.0, 0.0, 0.7, 0.3];
    var pts: f64[>..] = [];
    var vals: f64[>..] = [];
    var i: i64 = 0;
    while i < 6 {
        pts.push(xs[i]);
        pts.push(ys[i]);
        vals.push(zs[i]);
        i += 1;
    }
    var shape: i64[..8] = [];
    shape.push(6);
    shape.push(2);
    let points = scientific::arr_of(copy(pts), shape);
    var one: i64[..8] = [];
    one.push(6);
    let values = scientific::arr_of(copy(vals), one);

    // Two points of the query.
    var qp: f64[>..] = [];
    qp.push(0.4);
    qp.push(0.3);
    qp.push(0.75);
    qp.push(0.6);
    var qshape: i64[..8] = [];
    qshape.push(2);
    qshape.push(2);
    let query = scientific::arr_of(copy(qp), qshape);

    // The radial basis of the legacy class.
    let r, f1 = scientific::rbf(points, values, query);
    if f1.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f1.kind), "\n"); return; }
    print("rbf ", r.data, "\n");

    // The interpolator of the radial basis of the modern class.
    let m, f2 = scientific::rbf_interpolate(points, values, query);
    if f2.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f2.kind), "\n"); return; }
    print("rbf interpolator ", m.data, "\n");

    // The piecewise cubic of Clough and Tocher.
    let c, f3 = scientific::clough_tocher_interpolate(points, values, query);
    if f3.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f3.kind), "\n"); return; }
    print("clough tocher ", c.data, "\n");

    // The rational approximation of AAA on a line.
    let sample = [-1.0, -0.5, 0.0, 0.5, 1.0];
    var sv: f64[>..] = [];
    i = 0;
    while i < 5 { sv.push(1.0 / (1.0 + 25.0 * sample[i] * sample[i])); i += 1; }
    let a, f4 = scientific::aaa_interpolate(sample, sv, [-0.75, 0.25]);
    if f4.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f4.kind), "\n"); return; }
    print("aaa ", a.data, "\n");
    return;
}
