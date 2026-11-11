import scientific from "..";

// An LQR gain from the continuous Riccati equation: the double
// integrator with a unit cost on position and on effort.
fn main() {
    let a, fa = scientific::from_rows(2, 2, [0.0, 1.0, 0.0, 0.0]);
    let b, fb = scientific::from_rows(2, 1, [0.0, 1.0]);
    let q, fq = scientific::from_rows(2, 2, [1.0, 0.0, 0.0, 0.0]);
    let r, fr = scientific::from_rows(1, 1, [1.0]);
    let x, f = scientific::solve_continuous_are(a, b, q, r);
    assert(f.kind == scientific::F_NONE);
    // The solution is [[sqrt 2, 1], [1, sqrt 2]]. The gain is r^-1 b^T x.
    print("x = ", x.data[0], " ", x.data[1], " ", x.data[3], "\n");
}
