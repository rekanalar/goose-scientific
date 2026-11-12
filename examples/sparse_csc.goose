// The sparse matrices of compressed columns.
import scientific from "..";

fn main() {
    let v, f0 = scientific::from_values([1.0, 0.0, 0.0, 2.0]);
    let a, f1 = scientific::reshape(v, [2, 2]);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);

    // The compressed columns of a dense matrix.
    let c, f2 = scientific::csc_from_dense(a);
    assert(f2.kind == scientific::F_NONE);
    print("stored values: ", c.nnz, "\n");

    // The transpose of the matrix.
    let t, f3 = scientific::csc_transpose(c);
    assert(f3.kind == scientific::F_NONE);
    let d, f4 = scientific::csc_to_dense(t);
    assert(f4.kind == scientific::F_NONE);
    print("transpose: ", d.data, "\n");

    // The product of the matrix and a vector.
    let p, f5 = scientific::csc_matvec(c, [1.0, 2.0]);
    assert(f5.kind == scientific::F_NONE);
    print("product: ", p.data, "\n");
    return;
}
