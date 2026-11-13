// The tests of the group istft. Run them with SCIENTIFIC_SUITES=istft.
namespace istft_tests;

import scientific from "../..";
import os;
import common;

// istft returns the signal of stft, and it follows SciPy.
fn test_istft() {
    let x = common::sp_signal();
    let f, t, zr, zi, f1 = scientific::stft(x, 2.0, "hann", 8, 4);
    assert(f1.kind == scientific::F_NONE);
    let t2, y, g = scientific::istft(zr, zi, 2.0, "hann", 8, 4);
    assert(g.kind == scientific::F_NONE && y.data.len == 40);
    assert(common::near(t2.data[2], 1.0));
    var i: i64 = 0;
    while i < 40 { assert(common::loose(y.data[i] + 1.0, x[i] + 1.0)); i += 1; }
    let f3, t3, wr, wi, f4 = scientific::stft(x, 1.0, "hann", 8, 4, false, false);
    let t4, y4, g4 = scientific::istft(wr, wi, 1.0, "hann", 8, 4, false);
    assert(g4.kind == scientific::F_VALUE);
    let t5, y5, g5 = scientific::istft(zr, zi, 2.0, "hann", 6, 4);
    assert(g5.kind == scientific::F_SHAPE);
    return;
}
