// The rank and derivative filters: remove a spike with the median, find an
// edge with Sobel, and take the maximum in a cross-shaped window.
import scientific from "..";

fn show(name: const u8[:], a: scientific::Arr&) {
    print(name, ": ", a.data, "\n");
}

fn main() {
    let v, f0 = scientific::from_values([1.0, 1.0, 1.0, 1.0, 90.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]);
    assert(f0.kind == scientific::F_NONE);
    let m, f1 = scientific::reshape(v, [4, 4]);
    assert(f1.kind == scientific::F_NONE);

    // A median of 3 by 3 removes the spike of 90 and keeps the step.
    let med, f2 = scientific::median_filter(m, [3, 3], "nearest");
    assert(f2.kind == scientific::F_NONE);
    show("median", med);

    // The Sobel derivative along axis 0 finds the step between rows.
    let edge, f3 = scientific::sobel(m, 0, "nearest");
    assert(f3.kind == scientific::F_NONE);
    show("sobel", edge);

    // A footprint in the shape of a cross.
    let c, f4 = scientific::from_values([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    let cross, f5 = scientific::reshape(c, [3, 3]);
    assert(f4.kind == scientific::F_NONE && f5.kind == scientific::F_NONE);
    let high, f6 = scientific::maximum_filter_fp(m, cross, "nearest");
    assert(f6.kind == scientific::F_NONE);
    show("cross maximum", high);

    // The 80th percentile of a window of three values along the last axis.
    let p, f7 = scientific::percentile_filter(m, 80.0, [1, 3], "reflect");
    assert(f7.kind == scientific::F_NONE);
    show("percentile", p);

    // A rank outside the window is a fault.
    let bad, f8 = scientific::rank_filter(m, 9, [3, 3]);
    print("rank 9 of 9 places gives a fault of kind ", scientific::kind_name(f8.kind), "\n");
}
