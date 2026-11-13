// The tests of the group remez. Run them with SCIENTIFIC_SUITES=remez.
namespace remez_tests;

import scientific from "../..";
import os;
import common;

// remez is pinned against SciPy for each type and for odd and even
// counts of taps.
fn test_remez() {
    let a, fa = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0]);
    assert(fa.kind == scientific::F_NONE);
    assert(common::all_close(a, [0.05372035912417292, -4.61214414692147e-05, -0.09157551252765038, -6.7107264123562e-05, 0.31313220746845083, 0.49991992278419434, 0.31313220746845083, -6.7107264123562e-05, -0.09157551252765038, -4.61214414692147e-05, 0.05372035912417292], 1.0e-10));
    let b, fb = scientific::remez(15, [0.0, 0.1, 0.2, 0.3, 0.35, 0.5], [0.0, 1.0, 0.0], [10.0, 1.0, 10.0]);
    assert(fb.kind == scientific::F_NONE);
    assert(common::all_close(b, [-0.011905703004954533, -0.0737020253398363, 0.02708217289037356, 0.17031926466831956, -0.0250884447180607, -0.26399147717667837, 0.009911974832641815, 0.30157740067295724, 0.009911974832641815, -0.26399147717667837, -0.0250884447180607, 0.17031926466831956, 0.02708217289037356, -0.0737020253398363, -0.011905703004954533], 1.0e-10));
    let c, fc = scientific::remez(12, [0.0, 0.45], [1.0], [], "differentiator");
    assert(fc.kind == scientific::F_NONE);
    assert(common::all_close(c, [-0.0005736103918763211, 0.001548655382839329, -0.0030141247437764315, 0.006920562112278915, -0.021291909480373357, 0.20139962561772198, -0.20139962561772198, 0.021291909480373357, -0.006920562112278915, 0.0030141247437764315, -0.001548655382839329, 0.0005736103918763211], 1.0e-10));
    let d, fd = scientific::remez(13, [0.05, 0.45], [1.0], [], "hilbert");
    assert(fd.kind == scientific::F_NONE);
    assert(common::all_close(d, [6.882585201938116e-05, 0.1075025400501003, -4.143283426749236e-05, 0.18302551835222908, -4.432283377647591e-05, 0.6264899668103487, 0.0, -0.6264899668103487, 4.432283377647591e-05, -0.18302551835222908, 4.143283426749236e-05, -0.1075025400501003, -6.882585201938116e-05], 1.0e-10));
    let e, fe = scientific::remez(10, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0]);
    assert(fe.kind == scientific::F_NONE);
    assert(common::all_close(e, [0.03446606203700606, -0.07216259080561685, -0.09029533865931524, 0.13984939677118302, 0.4471867306483239, 0.4471867306483239, 0.13984939677118302, -0.09029533865931524, -0.07216259080561685, 0.03446606203700606], 1.0e-10));
    // one pass returns the taps of that pass, as SciPy does
    let m, fm = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [], "bandpass", 1);
    assert(fm.kind == scientific::F_NONE);
    assert(common::all_close(m, [0.01534496766039262, 0.0279372076644553, -0.07015766044520624, -0.0487827502902359, 0.3048126927848137, 0.5564503872470827, 0.3048126927848137, -0.0487827502902359, -0.07015766044520624, 0.0279372076644553, 0.01534496766039262], 1.0e-10));
    // fs scales the edges
    let s, fs2 = scientific::remez(11, [0.0, 1600.0, 2400.0, 4000.0], [1.0, 0.0], [], "bandpass", 25, 16, 8000.0);
    assert(fs2.kind == scientific::F_NONE && common::all_close(s, [0.05372035912417292, -4.61214414692147e-05, -0.09157551252765038, -6.7107264123562e-05, 0.31313220746845083, 0.49991992278419434, 0.31313220746845083, -6.7107264123562e-05, -0.09157551252765038, -4.61214414692147e-05, 0.05372035912417292], 1.0e-10));
    // the faults
    let g1, gf1 = scientific::remez(11, [0.0, 0.2, 0.3], [1.0, 0.0]);
    assert(gf1.kind == scientific::F_SHAPE && g1.data.len == 0);
    let g2, gf2 = scientific::remez(11, [0.0, 0.3, 0.2, 0.5], [1.0, 0.0]);
    assert(gf2.kind == scientific::F_VALUE);
    let g3, gf3 = scientific::remez(11, [0.0, 0.2, 0.3, 0.7], [1.0, 0.0]);
    assert(gf3.kind == scientific::F_RANGE);
    let g4, gf4 = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [1.0]);
    assert(gf4.kind == scientific::F_SHAPE);
    let g5, gf5 = scientific::remez(1, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0]);
    assert(gf5.kind == scientific::F_RANGE);
    let g6, gf6 = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [], "lowpass");
    assert(gf6.kind == scientific::F_VALUE);
    let g7, gf7 = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [0.0, 1.0]);
    assert(gf7.kind == scientific::F_VALUE);
    let g8, gf8 = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 1.0 / 0.0]);
    assert(gf8.kind == scientific::F_VALUE);
    let g9, gf9 = scientific::remez(5000, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0]);
    assert(gf9.kind == scientific::F_LIMIT);
    let g9b, gf9b = scientific::remez(4096, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [], "bandpass", 25, 1024);
    assert(gf9b.kind == scientific::F_LIMIT);
    let g10, gf10 = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [], "bandpass", 25, 0);
    assert(gf10.kind == scientific::F_RANGE);
    let g11, gf11 = scientific::remez(11, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], [], "bandpass", 25, 16, 0.0);
    assert(gf11.kind == scientific::F_VALUE);
    // a band too narrow for the grid
    let g12, gf12 = scientific::remez(21, [0.0, 0.001, 0.4, 0.5], [1.0, 0.0], [], "bandpass", 25, 1);
    assert(gf12.kind == scientific::F_RANGE && g12.data.len == 0);
    // the exchange fails when the deviation stops rising
    let g13, gf13 = scientific::remez(73, [0.02, 0.3], [1.0], [], "differentiator");
    assert(gf13.kind == scientific::F_VALUE && g13.data.len == 0);
    return;
}
