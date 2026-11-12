// The sparse matrices of keys.
import scientific from "..";

fn main() {
    // A matrix of three rows and four columns, filled place by place.
    var d, f0 = scientific::dok_new(3, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 2.0, 1.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 0.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 1.0, 3.0, 3.0).kind == scientific::F_NONE);
    print("keys: ", scientific::dok_nnz(d), "\n");

    // An overwrite keeps the place in the order. A zero removes the place.
    assert(scientific::dok_set(d, 2.0, 1.0, 6.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 0.0, 0.0).kind == scientific::F_NONE);
    let r, c, v, f1 = scientific::dok_keys(d);
    assert(f1.kind == scientific::F_NONE);
    print("rows: ", r.data, "\n");
    print("columns: ", c.data, "\n");
    print("values: ", v.data, "\n");

    // The diagonal, the product with a vector and the sum.
    assert(scientific::dok_setdiag(d, 0, [2.0, 2.0, 2.0]).kind == scientific::F_NONE);
    let y, f2 = scientific::dok_matvec(d, [1.0, 2.0, 3.0, 4.0]);
    assert(f2.kind == scientific::F_NONE);
    print("product: ", y.data, "\n");
    let t, f3 = scientific::dok_sum(d);
    assert(f3.kind == scientific::F_NONE);
    print("sum: ", t, "\n");

    // The compressed rows.
    let s, f4 = scientific::dok_to_csr(d);
    assert(f4.kind == scientific::F_NONE);
    print("stored: ", s.nnz, "\n");
}
