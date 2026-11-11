import scientific from "..";

// An interpolator of scattered data that answers many sets of points.
fn main() {
    let p, f0 = scientific::from_rows(5, 2, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5]);
    let v, f1 = scientific::from_rows(5, 2, [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0]);
    let xi, f2 = scientific::from_rows(4, 2, [0.25, 0.25, 0.75, 0.5, 2.0, 2.0, 0.5, 0.1]);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE && f2.kind == scientific::F_NONE);
    // The triangulation is built once.
    let it, f3 = scientific::nd_interpolator(p, v, "linear", -1.0);
    assert(f3.kind == scientific::F_NONE);
    let r, f4 = scientific::nd_interpolate(it, xi);
    print("rows of two columns: ", r.data, " fault ", f4.kind, "\n");
    let n, f5 = scientific::griddata(p, v, xi, "nearest");
    print("nearest rows: ", n.data, " fault ", f5.kind, "\n");
}
