// The chamfer and brute force distance transforms, and the positions of each
// value of an array of whole numbers.
import scientific from "..";

fn show(name: const u8[:], a: scientific::Arr&) {
    print(name, ": ", a.data, "\n");
}

fn main() {
    let v, f0 = scientific::from_values([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]);
    let img, f1 = scientific::reshape(v, [4, 4]);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);

    // The distance of each foreground value to the nearest zero.
    let cb, f2 = scientific::distance_transform_cdt(img, "chessboard");
    let tx, f3 = scientific::distance_transform_cdt(img, "taxicab");
    let eu, f4 = scientific::distance_transform_bf(img, "euclidean", [2.0, 1.0]);
    assert(f2.kind == scientific::F_NONE && f3.kind == scientific::F_NONE && f4.kind == scientific::F_NONE);
    show("chessboard", cb);
    show("taxicab", tx);
    show("euclidean, rows twice as far", eu);

    // The position of the nearest zero: one row of rows, then one of columns.
    let d, where, f5 = scientific::distance_transform_cdt_indices(img, "taxicab");
    assert(f5.kind == scientific::F_NONE);
    print("indices shape: ", where.shape, "\n");
    show("nearest zero, rows then columns", where);

    // The positions of each value of an array of whole numbers.
    let w, f6 = scientific::from_values([1.0, 2.0, 2.0, 1.0]);
    let lab, f7 = scientific::reshape(w, [2, 2]);
    let values, counts, rows, f8 = scientific::value_indices(lab);
    assert(f6.kind == scientific::F_NONE && f7.kind == scientific::F_NONE && f8.kind == scientific::F_NONE);
    show("values", values);
    show("counts", counts);
    show("positions, one row each", rows);
}
