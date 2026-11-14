// The random laws of a vector. The program draws a point of the
// Dirichlet law and a point of the normal law of a vector.
import std;
import scientific from "..";

fn main() {
    var r = scientific::default_rng(7);
    let alpha, fa = scientific::from_values([1.0, 2.0, 3.0]);
    assert(fa.kind == scientific::F_NONE);
    let p, fp = scientific::dirichlet(r, alpha);
    assert(fp.kind == scientific::F_NONE);
    print("dirichlet ");
    var i: i64 = 0;
    while i < p.data.len { print(p.data[i], " "); i += 1; }
    print("\n");
    let mean, fm = scientific::from_values([0.0, 0.0]);
    assert(fm.kind == scientific::F_NONE);
    let cov, fc = scientific::from_rows(2, 2, [2.0, 0.5, 0.5, 1.0]);
    assert(fc.kind == scientific::F_NONE);
    let x, fx = scientific::multivariate_normal(r, mean, cov);
    assert(fx.kind == scientific::F_NONE);
    print("normal of a vector ");
    i = 0;
    while i < x.data.len { print(x.data[i], " "); i += 1; }
    print("\n");
    let bad, fbad = scientific::from_rows(2, 2, [1.0, 2.0, 2.0, 1.0]);
    assert(fbad.kind == scientific::F_NONE);
    let y, fy = scientific::multivariate_normal(r, mean, bad);
    assert(fy.kind == scientific::F_VALUE);
    print("the call refuses a covariance that is not positive semidefinite\n");
}
