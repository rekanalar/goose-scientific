// The labels, the binary morphology and the distance transform: number the
// parts of a mask, grow it, fill its holes, and measure the distance to the
// nearest background value.
import scientific from "..";

fn show(name: const u8[:], a: scientific::Arr&) {
    print(name, ": ", a.data, "\n");
}

fn main() {
    let v, f0 = scientific::from_values([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                                         0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                                         0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                                         0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                                         0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    assert(f0.kind == scientific::F_NONE);
    let m, f1 = scientific::reshape(v, [6, 6]);
    assert(f1.kind == scientific::F_NONE);

    // The two parts of the mask get a number each.
    let lab, n, f2 = scientific::label(m, 1);
    assert(f2.kind == scientific::F_NONE);
    print("parts: ", n, "\n");
    show("labels", lab);

    // The dilation grows the mask by one value on every side.
    let big, f3 = scientific::binary_dilation(m, 1);
    assert(f3.kind == scientific::F_NONE);
    show("dilated", big);

    // The hole of the ring joins no border, so the fill closes it.
    let full, f4 = scientific::binary_fill_holes(m, 1);
    assert(f4.kind == scientific::F_NONE);
    show("filled", full);

    // The distance to the nearest background value.
    let d, f5 = scientific::distance_transform_edt(m);
    assert(f5.kind == scientific::F_NONE);
    show("distance", d);
}
