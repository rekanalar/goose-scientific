// The filters that take a block: the range in a window, the mean of the
// neighbors, and a Laplacian built from a second derivative of your own.
import scientific from "..";

fn spread(w: f64[:]) -> f64 {
    var lo = w[0];
    var hi = w[0];
    var i: i64 = 1;
    while i < w.len {
        if w[i] < lo { lo = w[i]; }
        if w[i] > hi { hi = w[i]; }
        i += 1;
    }
    return hi - lo;
}

fn second(a: scientific::Arr, axis: i64, mode: const u8[:], cval: f64) -> scientific::Arr {
    let d, g = scientific::correlate1d(a, [1.0, -2.0, 1.0], axis, mode, cval, 0);
    assert(g.kind == scientific::F_NONE);
    return d;
}

fn main() {
    let v, f0 = scientific::from_values([1.0, 2.0, 4.0, 8.0, 3.0, 5.0, 9.0, 1.0, 0.0]);
    let m, f1 = scientific::reshape(v, [3, 3]);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);

    // The range of each window of 2 by 2.
    let r, f2 = scientific::generic_filter(m, [2, 2], "nearest") { w => spread(w) };
    assert(f2.kind == scientific::F_NONE);
    print("range: ", r.data, "\n");

    // The mean of the four neighbors, with a cross-shaped footprint.
    let c, f3 = scientific::from_values([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    let cross, f4 = scientific::reshape(c, [3, 3]);
    assert(f3.kind == scientific::F_NONE && f4.kind == scientific::F_NONE);
    let n, f5 = scientific::generic_filter_fp(m, cross, "nearest") { w => (w[0] + w[1] + w[2] + w[3]) / 4.0 };
    assert(f5.kind == scientific::F_NONE);
    print("neighbors: ", n.data, "\n");

    // A Laplacian from the second difference of your own.
    let l, f6 = scientific::generic_laplace(m, "nearest") { x, ax, md, cv => second(copy(x), ax, md, cv) };
    assert(f6.kind == scientific::F_NONE);
    print("laplace: ", l.data, "\n");
}
