// The partial singular value decomposition svds of a sparse matrix.
import scientific from "..";

fn main() {
    // A matrix of 6 rows and 4 columns with 8 stored values.
    let a, fa = scientific::csr_from_triplets(6, 4,
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        [0.0, 2.0, 0.0, 3.0, 1.0, 3.0, 1.0, 2.0],
        [3.0, 1.0, 2.0, -1.0, 1.0, 2.0, -2.0, 1.0]);
    assert(fa.kind == scientific::F_NONE);

    // The call gives the two largest singular values, with their vectors.
    let u, s, vt, f = scientific::svds(a, 2);
    assert(f.kind == scientific::F_NONE);
    print("largest: ", s.data, "\n");
    print("u: ", u.data, "\n");
    print("vt: ", vt.data, "\n");

    // The call gives the two smallest values when it takes the choice SM.
    let u2, s2, vt2, f2 = scientific::svds(a, 2, "SM");
    assert(f2.kind == scientific::F_NONE);
    print("smallest: ", s2.data, "\n");

    // The call gives the values alone in the mode of the values only.
    let u3, s3, vt3, f3 = scientific::svds(a, 2, "LM", 0.0, [], 0, 0, scientific::SVDS_VALUES);
    assert(f3.kind == scientific::F_NONE && u3.data.len == 0 && vt3.data.len == 0);
    print("values only: ", s3.data, "\n");

    // A count of values that does not fit the matrix gives a fault, not a result.
    let u4, s4, vt4, f4 = scientific::svds(a, 4);
    assert(f4.kind == scientific::F_RANGE && u4.data.len == 0 && s4.data.len == 0 && vt4.data.len == 0);
    print("not fit: ", scientific::kind_name(f4.kind), "\n");
}
