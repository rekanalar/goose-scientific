// The sparse LU factor: factor once, solve many times.
import scientific from "..";

fn main() {
    // A tridiagonal matrix of side 5.
    let a, fa = scientific::csr_from_triplets(5, 5,
        [0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0],
        [0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0],
        [2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0]);
    assert(fa.kind == scientific::F_NONE);

    // The factor keeps the matrix sparse. The count holds L with its unit
    // diagonal and U.
    let f, ff = scientific::sparse_lu(a);
    assert(ff.kind == scientific::F_NONE);
    print("stored values: ", scientific::splu_nnz(f), "\n");

    // Two right sides use the same factor.
    let x1, f1 = scientific::splu_solve(f, [1.0, 0.0, 0.0, 0.0, 1.0]);
    assert(f1.kind == scientific::F_NONE);
    print("x1: ", x1.data, "\n");
    let x2, f2 = scientific::splu_solve(f, [1.0, 2.0, 3.0, 4.0, 5.0]);
    assert(f2.kind == scientific::F_NONE);
    print("x2: ", x2.data, "\n");

    // A matrix that is singular gives a fault, not a result.
    let s, fs = scientific::csr_from_triplets(2, 2, [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 2.0, 2.0, 4.0]);
    assert(fs.kind == scientific::F_NONE);
    let sf, sff = scientific::sparse_lu(s);
    assert(sff.kind == scientific::F_VALUE);
    print("singular: ", scientific::kind_name(sff.kind), "\n");
}
