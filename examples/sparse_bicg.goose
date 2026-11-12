// The biconjugate gradient solvers.
import scientific from "..";

fn main() {
    let a, f1 = scientific::from_rows(2, 2, [4.0, -1.0, 2.0, 3.0]);
    assert(f1.kind == scientific::F_NONE);

    let s, f2 = scientific::csr_from_dense(a);
    assert(f2.kind == scientific::F_NONE);

    // The four methods solve the same system.
    let x1, c1, f3 = scientific::bicg(s, [1.0, 2.0], 1.0e-12);
    assert(f3.kind == scientific::F_NONE && c1 == 0);
    print("bicg: ", x1.data, "\n");

    let x2, c2, f4 = scientific::cgs(s, [1.0, 2.0], 1.0e-12);
    assert(f4.kind == scientific::F_NONE && c2 == 0);
    print("cgs: ", x2.data, "\n");

    let x3, c3, f5 = scientific::qmr(s, [1.0, 2.0], 1.0e-12);
    assert(f5.kind == scientific::F_NONE && c3 == 0);
    print("qmr: ", x3.data, "\n");

    let x4, c4, f6 = scientific::tfqmr(s, [1.0, 2.0], 1.0e-12);
    assert(f6.kind == scientific::F_NONE && c4 == 0);
    print("tfqmr: ", x4.data, "\n");
    return;
}
