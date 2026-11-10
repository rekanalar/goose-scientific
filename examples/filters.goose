import scientific from "..";

// An elliptic lowpass filter of order 4: 1 dB of ripple in the pass
// band, 40 dB of attenuation in the stop band, cut at 0.3 of the
// Nyquist rate. Then the partial fraction expansion of a rational
// function with a double pole.
fn main() {
    let b, a, f = scientific::ellip(4, 1.0, 40.0, [0.3], "lowpass");
    if f.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f.kind), "\n"); return; }
    print("numerator ", b.data, "\n");
    print("denominator ", a.data, "\n");

    // (s + 2) / (s + 2)^2 has the pole -2 twice, with residues 1 and 0.
    let rr, ri, pr, pi, k, rf = scientific::residue([1.0, 2.0], [1.0, 4.0, 4.0]);
    if rf.kind != scientific::F_NONE { print("fault ", scientific::kind_name(rf.kind), "\n"); return; }
    print("residues ", rr.data, " poles ", pr.data, " direct terms ", k.data.len, "\n");

    // A ripple of zero is a fault, not a filter.
    let _b, _a, bad = scientific::cheby1(3, 0.0, [0.3], "lowpass");
    print("zero ripple ", scientific::kind_name(bad.kind), "\n");
    return;
}
