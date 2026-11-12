// The grey morphology: erode, dilate, open and close an image, take the
// gradient and the top hats, and grow a seed inside a mask.
import scientific from "..";

fn show(name: const u8[:], a: scientific::Arr&) {
    print(name, ": ", a.data, "\n");
}

fn main() {
    let v, f0 = scientific::from_values([1.0, 5.0, 2.0, 7.0, 3.0, 0.0, 9.0, 4.0, 6.0, 8.0, 1.0, 2.0]);
    let img, f1 = scientific::reshape(v, [3, 4]);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);

    // A flat box of 2 by 3 places.
    let er, f2 = scientific::grey_erosion(img, [2, 3]);
    let di, f3 = scientific::grey_dilation(img, [2, 3]);
    let gr, f4 = scientific::morphological_gradient(img, [3, 3]);
    let wt, f5 = scientific::white_tophat(img, [2, 2]);
    assert(f2.kind == scientific::F_NONE && f3.kind == scientific::F_NONE);
    assert(f4.kind == scientific::F_NONE && f5.kind == scientific::F_NONE);
    show("erosion", er);
    show("dilation", di);
    show("gradient", gr);
    show("white top hat", wt);

    // A structure with values: a cross whose center sits one step higher.
    let c, f6 = scientific::from_values([0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0]);
    let st, f7 = scientific::reshape(c, [3, 3]);
    let sd, f8 = scientific::grey_dilation_st(img, scientific::empty_arr(), st);
    assert(f6.kind == scientific::F_NONE && f7.kind == scientific::F_NONE && f8.kind == scientific::F_NONE);
    show("dilation by the structure", sd);

    // A seed grows inside a mask until nothing changes.
    let s, f9 = scientific::from_values([1.0, 0.0, 0.0, 0.0, 0.0]);
    let m, f10 = scientific::from_values([1.0, 1.0, 0.0, 1.0, 1.0]);
    let k, f11 = scientific::from_values([1.0, 1.0, 1.0]);
    let pr, f12 = scientific::binary_propagation_st(s, k, m);
    assert(f9.kind == scientific::F_NONE && f10.kind == scientific::F_NONE);
    assert(f11.kind == scientific::F_NONE && f12.kind == scientific::F_NONE);
    show("propagation", pr);
}
