// The measurements of labeled regions: label a mask, then measure the size,
// the mean, the brightest place and the center of each part.
import scientific from "..";

fn show(name: const u8[:], a: scientific::Arr&) {
    print(name, ": ", a.data, "\n");
}

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

fn main() {
    let m, f0 = scientific::from_values([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    let mask, f1 = scientific::reshape(m, [3, 4]);
    let v, f2 = scientific::from_values([5.0, 7.0, 2.0, 1.0, 9.0, 3.0, 4.0, 6.0, 2.0, 8.0, 1.0, 4.0]);
    let img, f3 = scientific::reshape(v, [3, 4]);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);
    assert(f2.kind == scientific::F_NONE && f3.kind == scientific::F_NONE);

    // Number the parts. Then measure each one by its label.
    let lab, n, f4 = scientific::label(mask, 1);
    assert(f4.kind == scientific::F_NONE);
    print("parts: ", n, "\n");
    let index = [1, 2, 3];
    let sums, f5 = scientific::sum_labels(img, lab, index);
    let means, f6 = scientific::mean_labels(img, lab, index);
    let peaks, f7 = scientific::maximum_position(img, lab, index);
    let centers, f8 = scientific::center_of_mass(img, lab, index);
    assert(f5.kind == scientific::F_NONE && f6.kind == scientific::F_NONE);
    assert(f7.kind == scientific::F_NONE && f8.kind == scientific::F_NONE);
    show("sums", sums);
    show("means", means);
    show("peak places", peaks);
    show("centers", centers);

    // The bounding box of each part: start of each axis, then stop of each axis.
    let boxes, f9 = scientific::find_objects(lab);
    assert(f9.kind == scientific::F_NONE);
    show("boxes", boxes);

    // A measure of your own, as a block: the spread of each part.
    let sp, f10 = scientific::labeled_comprehension(img, lab, index) { w => spread(w) };
    assert(f10.kind == scientific::F_NONE);
    show("spread", sp);
}
