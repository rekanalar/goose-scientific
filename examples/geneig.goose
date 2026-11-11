import scientific from "..";

// The generalized eigenproblem of a pair of matrices.
fn pair(n: i64, re: f64[:], im: f64[:]) -> scientific::CArr {
    let r = scientific::from_rows(n, n, re);
    let i = scientific::from_rows(n, n, im);
    let c, f = scientific::complex_array(r, i);
    assert(f.kind == scientific::F_NONE);
    return c;
}

fn main() {
    let a = pair(2, [1.0, 2.0, 3.0, 4.0], [1.0, 0.0, -2.0, 1.0]);
    let b = pair(2, [2.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, -1.0]);
    let w, f1 = scientific::ceigvals_pair(a, b);
    print("eigenvalues: ", w.data, " fault ", f1.kind, "\n");
    // A singular B gives an eigenvalue at infinity.
    let a2 = pair(2, [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]);
    let b2 = pair(2, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]);
    let w2, f2 = scientific::ceigvals_pair(a2, b2);
    print("singular b: ", w2.data, " fault ", f2.kind, "\n");
    // A Hermitian pair with a positive definite B has real eigenvalues.
    let h = pair(2, [2.0, 1.0, 1.0, 3.0], [0.0, -1.0, 1.0, 0.0]);
    let d = pair(2, [4.0, 0.0, 0.0, 2.0], [0.0, 1.0, -1.0, 0.0]);
    let e, f3 = scientific::ceigvalsh_pair(h, d);
    print("hermitian pair: ", e.data, " fault ", f3.kind, "\n");
}
