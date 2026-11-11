import scientific from "..";

// The eigenvalues of a complex matrix and of a Hermitian matrix.
fn main() {
    let re = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let im = scientific::from_rows(2, 2, [1.0, 0.0, -2.0, 1.0]);
    let a, f0 = scientific::complex_array(re, im);
    assert(f0.kind == scientific::F_NONE);
    let w, v, f1 = scientific::ceig(a);
    print("eigenvalues: ", w.data, " fault ", f1.kind, "\n");
    // A Hermitian matrix has real eigenvalues in rising order.
    let hr = scientific::from_rows(2, 2, [2.0, 0.0, 0.0, 3.0]);
    let hi = scientific::from_rows(2, 2, [0.0, -1.0, 1.0, 0.0]);
    let h, f2 = scientific::complex_array(hr, hi);
    assert(f2.kind == scientific::F_NONE);
    let e, f3 = scientific::ceigvalsh(h, "L");
    print("hermitian eigenvalues: ", e.data, " fault ", f3.kind, "\n");
}
