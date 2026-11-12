// The sparse matrices of coordinates.
import scientific from "..";

fn main() {
    // Two entries of one place stay two entries, and they add up in the dense form.
    let c, f0 = scientific::coo_from_triplets(2, 2, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]);
    assert(f0.kind == scientific::F_NONE);
    print("entries: ", c.nnz, "\n");

    let d, f1 = scientific::coo_to_dense(c);
    assert(f1.kind == scientific::F_NONE);
    print("dense: ", d.data, "\n");

    let p, f2 = scientific::coo_matvec(c, [1.0, 2.0]);
    assert(f2.kind == scientific::F_NONE);
    print("product: ", p.data, "\n");
    return;
}
