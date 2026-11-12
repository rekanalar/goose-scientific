// The Kronecker sum of two square sparse matrices.
import scientific from "..";

fn main() {
    let a, f0 = scientific::csr_from_dense(scientific::from_rows(2, 2, [1.0, 2.0, 0.0, 3.0]));
    let b, f1 = scientific::csr_from_dense(scientific::from_rows(2, 2, [0.0, 5.0, 6.0, 0.0]));
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);

    // kron(I, a) + kron(b, I): a matrix of 4 rows and 4 columns.
    let k, f2 = scientific::csr_kronsum(a, b);
    assert(f2.kind == scientific::F_NONE);
    print("rows: ", k.rows, " stored values: ", k.nnz, "\n");
    let d, f3 = scientific::csr_to_dense(k);
    assert(f3.kind == scientific::F_NONE);
    print("dense: ", d.data, "\n");

    // An operand that is not square is a fault.
    let w, f4 = scientific::csr_from_dense(scientific::from_rows(2, 3, [1.0, 0.0, 2.0, 0.0, 0.0, 3.0]));
    assert(f4.kind == scientific::F_NONE);
    let e, f5 = scientific::csr_kronsum(w, b);
    print("not square: ", f5.kind == scientific::F_SHAPE, "\n");
    return;
}
