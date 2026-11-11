import scientific from "..";

// The QR decomposition and the singular values of a complex matrix.
fn main() {
    let re = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let im = scientific::from_rows(2, 2, [1.0, 0.0, -2.0, 1.0]);
    let a, f0 = scientific::complex_array(re, im);
    assert(f0.kind == scientific::F_NONE);
    // The diagonal of the upper triangle holds real values.
    let q, r, f1 = scientific::cqr(a);
    print("diagonal of r: ", r.data[0], " ", r.data[1], " ", r.data[6], " ", r.data[7], " fault ", f1.kind, "\n");
    // The singular values come from the largest to the smallest.
    let u, s, vh, f2 = scientific::csvd(a);
    print("singular values: ", s.data[0], " ", s.data[1], " fault ", f2.kind, "\n");
}
