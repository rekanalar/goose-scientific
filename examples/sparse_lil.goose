// The sparse matrices of lists.
import scientific from "..";

fn main() {
    // A matrix of three rows and four columns, filled place by place.
    var d, f0 = scientific::lil_new(3, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 1.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 0.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 3.0, 3.0).kind == scientific::F_NONE);
    print("entries: ", scientific::lil_nnz(d), "\n");

    // The columns of a row stand in ascending order.
    let c, v, f1 = scientific::lil_row(d, 0.0);
    assert(f1.kind == scientific::F_NONE);
    print("columns: ", c.data, "\n");
    print("values: ", v.data, "\n");

    // An overwrite keeps the place in the row. A zero removes the place.
    assert(scientific::lil_set(d, 0.0, 0.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 0.0, 0.0).kind == scientific::F_NONE);
    print("entries: ", scientific::lil_nnz(d), "\n");

    // The diagonal, the product with a vector and the sum.
    assert(scientific::lil_setdiag(d, 0, [2.0, 2.0, 2.0]).kind == scientific::F_NONE);
    let y, f2 = scientific::lil_matvec(d, [1.0, 2.0, 3.0, 4.0]);
    assert(f2.kind == scientific::F_NONE);
    print("product: ", y.data, "\n");
    let t, f3 = scientific::lil_sum(d);
    assert(f3.kind == scientific::F_NONE);
    print("sum: ", t, "\n");

    // The compressed rows.
    let s, f4 = scientific::lil_to_csr(d);
    assert(f4.kind == scientific::F_NONE);
    print("stored: ", s.nnz, "\n");
}
