// The restarted solvers lgmres and gcrotmk of a sparse matrix.
import scientific from "..";

fn main() {
    // A tridiagonal matrix of side 5.
    let a, fa = scientific::csr_from_triplets(5, 5,
        [0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0],
        [0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0],
        [2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0]);
    assert(fa.kind == scientific::F_NONE);

    let b = scientific::from_values([1.0, 1.0, 1.0, 1.0, 1.0]);

    // The lgmres call keeps the last correction vectors of the earlier
    // restarts. A count of zero means that the tolerance was met.
    let x, n, fx = scientific::lgmres(a, b.data, [], 1.0e-10, 0.0, 1000, 10, 3);
    assert(fx.kind == scientific::F_NONE);
    print("lgmres count: ", n, "\n");
    print("lgmres x: ", x.data, "\n");

    // The gcrotmk call keeps pairs of vectors across the restarts.
    let y, m, fy = scientific::gcrotmk(a, b.data, [], 1.0e-10, 0.0, 1000, 10, 3);
    assert(fy.kind == scientific::F_NONE);
    print("gcrotmk count: ", m, "\n");
    print("gcrotmk x: ", y.data, "\n");

    // A matrix that is not square gives a fault, not a result.
    let wide, fw = scientific::csr_from_triplets(2, 3, [0.0], [2.0], [1.0]);
    assert(fw.kind == scientific::F_NONE);
    let z, nz, fz = scientific::lgmres(wide, scientific::from_values([1.0, 2.0]).data, []);
    assert(fz.kind == scientific::F_SHAPE && z.data.len == 0 && nz == 0);
    print("not square: ", scientific::kind_name(fz.kind), "\n");
}
