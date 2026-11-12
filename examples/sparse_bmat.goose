// The block matrix and the columnwise diagonals.
import scientific from "..";

fn main() {
    // A grid of two rows and two columns. The place at the bottom right
    // holds no block.
    var g, f0 = scientific::bmat_new(2, 2);
    assert(f0.kind == scientific::F_NONE);
    let a2, fa = scientific::from_rows(2, 2, [1.0, 2.0, 0.0, 3.0]);
    assert(fa.kind == scientific::F_NONE);
    let ac, fc = scientific::csr_from_dense(a2);
    assert(fc.kind == scientific::F_NONE);
    let b2, fb = scientific::from_rows(2, 1, [4.0, 5.0]);
    assert(fb.kind == scientific::F_NONE);
    let bc, fbc = scientific::csr_from_dense(b2);
    assert(fbc.kind == scientific::F_NONE);
    assert(scientific::bmat_put(g, 0.0, 0.0, ac).kind == scientific::F_NONE);
    assert(scientific::bmat_put(g, 0.0, 1.0, bc).kind == scientific::F_NONE);

    // The grid gives one matrix of three rows and three columns.
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let d, fd = scientific::csr_to_dense(m);
    assert(fd.kind == scientific::F_NONE);
    print("grid: ", d.data, "\n");

    // The diagonals are placed by column. The value of the diagonal d at
    // the column j is the value data[d, j].
    let p2, fp = scientific::from_rows(2, 4, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert(fp.kind == scientific::F_NONE);
    let s, fs = scientific::spdiags(p2, [0.0, 1.0], 3, 4);
    assert(fs.kind == scientific::F_NONE);
    let sd, fsd = scientific::dia_to_dense(s);
    assert(fsd.kind == scientific::F_NONE);
    print("diagonals: ", sd.data, "\n");
}
