import scientific from "..";

// The extra interpolation calls of the package: the interpolators of
// Barycentric, Krogh and Floater and Hormann, the polynomial of
// Lagrange, the approximation of Pade, the algebra of the B-spline
// and the spline of two variables on a grid.
fn main() {
    let nodes = [0.0, 1.0, 2.0, 3.0];
    let values = [1.0, 2.0, 0.5, 3.0];
    let points = [-0.3, 0.7, 1.0, 2.4];

    // The interpolator of Barycentric and its first derivative.
    let b, f1 = scientific::barycentric_interpolate(nodes, values, points);
    if f1.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f1.kind), "\n"); return; }
    print("barycentric ", b.data, "\n");
    let d, f2 = scientific::barycentric_interpolate(nodes, values, points, 1);
    print("first derivative ", d.data, " fault ", f2.kind, "\n");

    // The interpolator of Floater and Hormann.
    let h, f3 = scientific::floater_hormann(nodes, values, points, 3);
    print("floater and hormann ", h.data, " fault ", f3.kind, "\n");

    // The interpolator of Krogh with a node that holds a derivative.
    let k, f4 = scientific::krogh_interpolate([0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.5]);
    print("krogh ", k.data, " fault ", f4.kind, "\n");

    // The polynomial of Lagrange and the approximation of Pade.
    let l, f5 = scientific::lagrange_poly([0.0, 1.0, 2.0], [1.0, 3.0, 2.0]);
    print("lagrange ", l, " fault ", f5.kind, "\n");
    let p, q, f6 = scientific::pade_approx([1.0, 1.0, 0.5, 0.16666666666666666], 2, 2);
    print("pade numerator ", p, " denominator ", q, " fault ", f6.kind, "\n");

    // The algebra of the B-spline.
    let x = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0];
    var y: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len { y.push(curve(x[i])); i += 1; }
    let s, f7 = scientific::splrep(x, y, 3);
    if f7.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f7.kind), "\n"); return; }
    let area, f8 = scientific::splint(0.5, 2.5, s);
    print("integral from 0.5 to 2.5 ", area, " fault ", f8.kind, "\n");
    let roots, f9 = scientific::sproot(s);
    print("roots ", roots.data, " fault ", f9.kind, "\n");

    // The spline of two variables on a grid.
    let gx = [0.0, 0.25, 0.5, 0.75, 1.0];
    let gy = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0];
    var gz: f64[>..] = [];
    i = 0;
    while i < gx.len {
        var j: i64 = 0;
        while j < gy.len { gz.push(curve(gx[i] + gy[j])); j += 1; }
        i += 1;
    }
    let g, f10 = scientific::rect_spline(gx, gy, gz, 3, 3);
    if f10.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f10.kind), "\n"); return; }
    let v, f11 = scientific::rect_spline_eval(g, [0.3, 0.9], [0.7, 1.9]);
    print("grid spline ", v.data, " fault ", f11.kind, "\n");
    return;
}

// curve gives a smooth test function: a cubic with a root inside
// the range.
fn curve(v: f64) -> f64 {
    return v * v * v - 3.0 * v + 1.0;
}
