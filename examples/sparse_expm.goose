// The one-norm estimate and the matrix exponential of a sparse matrix.
import scientific from "..";

fn main() {
    // A tridiagonal matrix of side 5.
    let a, fa = scientific::csr_from_triplets(5, 5,
        [0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0],
        [0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0],
        [2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0]);
    assert(fa.kind == scientific::F_NONE);

    // The estimate is a lower bound of the one-norm. The call gives the
    // vector that reaches it and the product of the matrix with that vector
    // when the caller asks for them. A fixed generator draws the block, so
    // two runs give the same value.
    let est, v, w, fe = scientific::onenormest(a, 2, 5, true, true);
    assert(fe.kind == scientific::F_NONE);
    print("estimate: ", est, "\n");
    print("vector: ", v.data, "\n");
    print("product: ", w.data, "\n");

    // The exponential acts on a vector. The matrix stays sparse.
    let x, fx = scientific::expm_multiply(a, scientific::from_values([1.0, 0.0, 0.0, 0.0, 1.0]));
    assert(fx.kind == scientific::F_NONE);
    print("exp(A) b: ", x.data, "\n");

    // The range form gives one row for each time.
    let rows, fr = scientific::expm_multiply_range(a, scientific::from_values([1.0, 0.0, 0.0, 0.0, 1.0]), 0.0, 0.5, 3, true);
    assert(fr.kind == scientific::F_NONE);
    print("rows: ", rows.shape[0], " by ", rows.shape[1], "\n");

    // A matrix that is not square gives a fault, not a result.
    let wide, fw = scientific::csr_from_triplets(2, 3, [0.0], [2.0], [1.0]);
    assert(fw.kind == scientific::F_NONE);
    let r, fb = scientific::expm_multiply(wide, scientific::from_values([1.0, 2.0]));
    assert(fb.kind == scientific::F_SHAPE && r.data.len == 0);
    print("not square: ", scientific::kind_name(fb.kind), "\n");
}
