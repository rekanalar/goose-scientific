// The sparse matrices of blocks.
import scientific from "..";

fn main() {
    // A matrix of four rows and four columns with blocks of two rows and
    // two columns.
    let a = scientific::from_rows(4, 4, [1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 6.0]);
    let s, f0 = scientific::bsr_from_dense(a, 2, 2);
    assert(f0.kind == scientific::F_NONE);
    print("blocks: ", scientific::bsr_nnz(s), "\n");

    // The row pointers, the block columns and the values of the blocks.
    let p, c, d, f1 = scientific::bsr_blocks(s);
    assert(f1.kind == scientific::F_NONE);
    print("pointers: ", p.data, "\n");
    print("columns: ", c.data, "\n");
    print("values: ", d.data, "\n");

    // The dense form and the product with a vector.
    let e, f2 = scientific::bsr_to_dense(s);
    assert(f2.kind == scientific::F_NONE);
    print("dense: ", e.data, "\n");

    let q, f3 = scientific::bsr_matvec(s, [1.0, 2.0, 3.0, 4.0]);
    assert(f3.kind == scientific::F_NONE);
    print("product: ", q.data, "\n");

    // The values on the main diagonal.
    let g, f5 = scientific::bsr_diagonal(s);
    assert(f5.kind == scientific::F_NONE);
    print("diagonal: ", g.data, "\n");

    // The transpose swaps the block size.
    let t, f4 = scientific::bsr_transpose(s);
    assert(f4.kind == scientific::F_NONE);
    print("block size: ", t.br, " ", t.bc, "\n");
    return;
}
