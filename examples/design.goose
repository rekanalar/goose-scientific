import scientific from "..";

// The design of an optimal lowpass filter by the exchange of Parks and
// McClellan: a pass band to 0.2 and a stop band from 0.3, in units of
// the sampling rate.
fn main() {
    let h, f = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0]);
    if f.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f.kind), "\n"); return; }
    print("taps ", h.data.len, " centre ", h.data[5], "\n");

    // A differentiator with too many taps for its band does not converge.
    let g, gf = scientific::remez(73, [0.02, 0.3], [1.0], [], "differentiator");
    print("narrow design fault ", scientific::kind_name(gf.kind), "\n");
    return;
}
