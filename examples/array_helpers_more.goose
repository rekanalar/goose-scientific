// The small helpers of NumPy.
//
// The example builds an array from a block, takes the floor quotient and
// the remainder, and tests two arrays for equivalence.
import std;
import scientific from "..";

fn main() {
    let a, f1 = scientific::fromfunction([2, 3]) { p => p[0] * 10.0 + p[1] };
    assert(f1.kind == scientific::F_NONE);
    print("fromfunction:");
    var i: i64 = 0;
    while i < a.data.len { print(" ", a.data[i]); i += 1; }
    print("\n");

    let x, f2 = scientific::from_values([7.0, -7.0]);
    assert(f2.kind == scientific::F_NONE);
    let y, f3 = scientific::from_values([2.0, 2.0]);
    assert(f3.kind == scientific::F_NONE);
    let q, r, f4 = scientific::divmod(x, y);
    assert(f4.kind == scientific::F_NONE);
    print("divmod: q ", q.data[0], " ", q.data[1], " r ", r.data[0], " ", r.data[1], "\n");

    let eq, f5 = scientific::array_equiv(x, y);
    assert(f5.kind == scientific::F_NONE);
    print("array_equiv: ", eq, "\n");
}
