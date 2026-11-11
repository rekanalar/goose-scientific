import scientific from "..";

// The products of complex arrays, the factor of a complex matrix and
// the transform of a complex array.
fn main() {
    let a, f0 = scientific::complex_from_values([1.0, 3.0], [2.0, -1.0]);
    let b, f1 = scientific::complex_from_values([2.0, -1.0], [-1.0, 0.5]);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);
    // The product of two vectors is a scalar.
    let p, fp = scientific::cdot(a, b);
    print("product: ", p.data[0], " ", p.data[1], " fault ", fp.kind, "\n");
    // The scalar product conjugates its first operand.
    let cr, ci, cf = scientific::cvdot(a, b);
    print("scalar product: ", cr, " ", ci, " fault ", cf.kind, "\n");
    // The factor of a Hermitian matrix holds its lower triangle.
    let re = scientific::from_rows(2, 2, [2.0, 1.0, 1.0, 3.0]);
    let im = scientific::from_rows(2, 2, [0.0, 1.0, -1.0, 0.0]);
    let m, fm = scientific::complex_array(re, im);
    assert(fm.kind == scientific::F_NONE);
    let l, fl = scientific::ccholesky(m);
    print("factor: ", l.data[0], " ", l.data[4], " ", l.data[5], " fault ", fl.kind, "\n");
    // The transform runs along every axis.
    let f, ff = scientific::cfftn(m);
    print("transform: ", f.data[0], " ", f.data[1], " fault ", ff.kind, "\n");
    let g, gf = scientific::cifftn(m);
    print("inverse transform: ", g.data[0], " ", g.data[1], " fault ", gf.kind, "\n");
}
