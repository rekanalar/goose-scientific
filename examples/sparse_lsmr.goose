// The least squares solver LSMR.
import scientific from "..";

fn main() {
    // Three equations and two unknowns.
    let a, f0 = scientific::from_rows(3, 2, [1.0, 0.0, 0.0, 2.0, 1.0, 1.0]);
    assert(f0.kind == scientific::F_NONE);
    let s, f1 = scientific::csr_from_dense(a);
    assert(f1.kind == scientific::F_NONE);

    let r, f2 = scientific::lsmr(s, [1.0, 2.0, 3.0]);
    assert(f2.kind == scientific::F_NONE);
    print("solution: ", r.x.data, "\n");
    print("reason to stop: ", r.istop, " after ", r.itn, " turns\n");

    // The damping term pulls the solution toward zero.
    let q, f3 = scientific::lsmr(s, [1.0, 2.0, 3.0], 1.0);
    assert(f3.kind == scientific::F_NONE);
    print("damped solution: ", q.x.data, "\n");
    return;
}
