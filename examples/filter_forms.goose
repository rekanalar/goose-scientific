// A filter through its forms: the zeros, the poles and the gain, the
// numerator and the denominator, and the state space form.
import std;
import scientific from "..";

fn show(label: const u8[:], a: scientific::Arr&) {
    print(label);
    var i: i64 = 0;
    while i < a.data.len { print(" ", a.data[i]); i += 1; }
    print("\n");
}

fn main() {
    // The filter H(s) = 5 (s - 2)(s - 6) / ((s - 1)(s - 8)).
    let b, a, f1 = scientific::zpk2tf([2.0, 6.0], [0.0, 0.0], [1.0, 8.0], [0.0, 0.0], 5.0);
    assert(f1.kind == scientific::F_NONE);
    show("numerator:", b);
    show("denominator:", a);

    // The way back gives the zeros, the poles and the gain.
    let zr, zi, pr, pim, k, f2 = scientific::tf2zpk(b.data, a.data);
    assert(f2.kind == scientific::F_NONE);
    show("zeros:", zr);
    show("poles:", pr);
    print("gain: ", k, "\n");

    // The state space form of the controller canonical type.
    let am, bm, cm, d, f3 = scientific::tf2ss(b.data, a.data);
    assert(f3.kind == scientific::F_NONE);
    show("A:", am);
    show("B:", bm);
    show("C:", cm);
    print("D: ", d, "\n");

    // The state space form gives the coefficients again.
    let nb, na, f4 = scientific::ss2tf(am, bm.data, cm.data, d);
    assert(f4.kind == scientific::F_NONE);
    show("numerator again:", nb);
    show("denominator again:", na);

    // A zero without a partner is not a real filter. The call refuses it.
    let b2, a2, f5 = scientific::zpk2tf([-0.5], [0.3], [0.9], [0.0], 2.5);
    assert(f5.kind == scientific::F_SHAPE);
    print("a zero without a partner: fault ", scientific::kind_name(f5.kind), "\n");
}
