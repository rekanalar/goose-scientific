// The archive of the sparse formats: write a matrix, read it back,
// and solve with the incomplete factorization.
import scientific from "..";

fn main() {
    // A matrix of four sides with nine stored values.
    let a, fa = scientific::csr_from_triplets(4, 4,
        [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        [0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 3.0],
        [4.0, 1.0, 2.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]);
    assert(fa.kind == scientific::F_NONE);

    // The archive of the matrix holds the members of SciPy, so NumPy
    // reads the same file.
    let ok, f = scientific::save_npz("tmp_sparse_npz.npz", copy(a));
    assert(ok && f.kind == scientific::F_NONE);
    var bytes: u8[>..] = [];
    assert(read_file("tmp_sparse_npz.npz", bytes));
    print("archive bytes: ", bytes.len, "\n");

    // The read of the archive gives the same matrix.
    let b, fb = scientific::load_npz("tmp_sparse_npz.npz");
    assert(fb.kind == scientific::F_NONE);
    let d, fd = scientific::csr_to_dense(b);
    assert(fd.kind == scientific::F_NONE);
    print("read back: ", d.data, "\n");
    assert(delete_file("tmp_sparse_npz.npz"));

    // The incomplete factorization drops the fill that stands below the
    // threshold of the drop. A threshold of zero drops nothing, so the
    // factor is the complete factorization.
    let fct, ff = scientific::spilu(a, 0.0, 10.0, "basic");
    assert(ff.kind == scientific::F_NONE);
    print("stored values: ", scientific::splu_nnz(fct), "\n");
    let x, fx = scientific::splu_solve(fct, [1.0, 2.0, 3.0, 4.0]);
    assert(fx.kind == scientific::F_NONE);
    print("solve: ", x.data, "\n");
}
