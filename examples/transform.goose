// The interpolating transforms: shift an array by half a value, double its
// size with a cubic spline, rotate it, and read it at chosen coordinates.
import scientific from "..";

fn show(name: const u8[:], a: scientific::Arr&) {
    print(name, ": ", a.shape, " ", a.data, "\n");
}

fn main() {
    let v, f0 = scientific::from_values([1.0, 4.0, 2.0, 8.0, 5.0, 3.0]);
    let m, f1 = scientific::reshape(v, [2, 3]);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);

    // A shift of half a value along the last axis, with linear interpolation.
    let s, f2 = scientific::shift(m, [0.0, 0.5], 1, "nearest");
    assert(f2.kind == scientific::F_NONE);
    show("shift", s);

    // A cubic zoom to twice the size. The corners stay in place.
    let z, f3 = scientific::zoom(m, [2.0, 2.0], 3, "mirror");
    assert(f3.kind == scientific::F_NONE);
    show("zoom", z);

    // A turn by 90 degrees without a change of shape.
    let r, f4 = scientific::rotate(m, 90.0, [1, 0], false, 1, "constant");
    assert(f4.kind == scientific::F_NONE);
    show("rotate", r);

    // The value at two chosen points. Row 0 holds the rows, row 1 the columns.
    let c, f5 = scientific::from_values([0.5, 1.0, 0.5, 1.5]);
    let pts, f6 = scientific::reshape(c, [2, 2]);
    assert(f5.kind == scientific::F_NONE && f6.kind == scientific::F_NONE);
    let p, f7 = scientific::map_coordinates(m, pts, 1, "nearest");
    assert(f7.kind == scientific::F_NONE);
    show("points", p);

    // The spline coefficients that the cubic calls compute first.
    let k, f8 = scientific::spline_filter(m, 3, "mirror");
    assert(f8.kind == scientific::F_NONE);
    show("coefficients", k);
}
