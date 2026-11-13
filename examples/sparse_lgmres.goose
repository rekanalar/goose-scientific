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

    // A trailing block sees the solution at the start of each outer turn.
    var turns: i64 = 0;
    let c, nc, fc = scientific::lgmres(a, b.data, [], 1.0e-10, 0.0, 1000, 10, 3) { v => turns += 1; };
    assert(fc.kind == scientific::F_NONE);
    print("callback turns: ", turns, "\n");

    // A preconditioner is a sparse matrix. Here it is the inverse of the diagonal.
    let m, fm = scientific::csr_from_triplets(5, 5, [0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0],
                                              [0.5, 0.5, 0.5, 0.5, 0.5]);
    assert(fm.kind == scientific::F_NONE);
    let p, np, fp = scientific::gcrotmk(a, b.data, [], 1.0e-10, 0.0, 1000, 10, 3, false, true, copy(m));
    assert(fp.kind == scientific::F_NONE);
    print("preconditioned x: ", p.data, "\n");

    // The kept pairs of one solve seed the next solve.
    let s1, n1, kp, f1 = scientific::gcrotmk_pairs(a, b.data, [], 1.0e-10);
    assert(f1.kind == scientific::F_NONE);
    let s2, n2, kp2, f2 = scientific::gcrotmk_pairs(a, scientific::from_values([2.0, 1.0, 1.0, 1.0, 2.0]).data,
                                                    [], 1.0e-10, 0.0, 1000, 20, 0, false, true,
                                                    scientific::csr_empty(), copy(kp));
    assert(f2.kind == scientific::F_NONE);
    print("second solve x: ", s2.data, "\n");

    // A matrix that is not square gives a fault, not a result.
    let wide, fw = scientific::csr_from_triplets(2, 3, [0.0], [2.0], [1.0]);
    assert(fw.kind == scientific::F_NONE);
    let z, nz, fz = scientific::lgmres(wide, scientific::from_values([1.0, 2.0]).data, []);
    assert(fz.kind == scientific::F_SHAPE && z.data.len == 0 && nz == 0);
    print("not square: ", scientific::kind_name(fz.kind), "\n");
}
