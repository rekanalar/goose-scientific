// The sparse matrices of diagonals.
import scientific from "..";

fn main() {
    // A matrix with values on three diagonals.
    let a = scientific::from_rows(4, 4, [1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0, 7.0, 0.0]);
    let d, f0 = scientific::dia_from_dense(a);
    assert(f0.kind == scientific::F_NONE);
    print("diagonals: ", d.ndiag, "\n");

    // The offsets and the values of the diagonals.
    let o, v, f1 = scientific::dia_diagonals(d);
    assert(f1.kind == scientific::F_NONE);
    print("offsets: ", o.data, "\n");
    print("values: ", v.data, "\n");

    // The dense form and the product with a vector.
    let e, f2 = scientific::dia_to_dense(d);
    assert(f2.kind == scientific::F_NONE);
    print("dense: ", e.data, "\n");

    let p, f3 = scientific::dia_matvec(d, [1.0, 2.0, 3.0, 4.0]);
    assert(f3.kind == scientific::F_NONE);
    print("product: ", p.data, "\n");
    return;
}
