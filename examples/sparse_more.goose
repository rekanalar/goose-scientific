// The triangular solve, the bandwidth, the Laplacian of a grid and the values of a matrix.
import scientific from "..";

fn main() {
    // A lower triangular matrix and a solve.
    let low, f0 = scientific::csr_from_dense(scientific::mat2([2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 4.0, 5.0], 3, 3));
    assert(f0.kind == scientific::F_NONE);
    let x, f1 = scientific::spsolve_triangular(low, [1.0, 2.0, 3.0], true);
    assert(f1.kind == scientific::F_NONE);
    print("x: ", x.data, "\n");
    let lower, upper = scientific::is_sptriangular(low);
    print("lower: ", lower, " upper: ", upper, "\n");
    let below, above = scientific::spbandwidth(low);
    print("bandwidth: ", below, " ", above, "\n");

    // The Laplacian of a grid of three by four with the Neumann boundary.
    let lap, f2 = scientific::laplacian_nd([3, 4], "neumann");
    assert(f2.kind == scientific::F_NONE);
    let vals, f3 = scientific::laplacian_nd_values([3, 4], "neumann");
    assert(f3.kind == scientific::F_NONE);
    print("nodes: ", lap.rows, " values: ", vals.data, "\n");

    // The product of the exponential of a matrix and a vector.
    let ident, f4 = scientific::csr_eye(4);
    assert(f4.kind == scientific::F_NONE);
    let y, f5 = scientific::funm_multiply_krylov(ident, [1.0, 2.0, 3.0, 4.0]) { t => t + 0.0 };
    print("funm times v: ", y.data, " fault: ", f5.kind, "\n");

    // A matrix of chance with the density of one fourth.
    let r, f6 = scientific::random_array([8, 6], 0.25, 7);
    assert(f6.kind == scientific::F_NONE);
    print("stored: ", r.nnz, " sound: ", scientific::csr_valid(r), "\n");
}
