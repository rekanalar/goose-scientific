// The tests of the group stft. Run them with SCIENTIFIC_SUITES=stft.
namespace stft_tests;

import scientific from "../..";
import os;
import common;

// stft is pinned against SciPy.
fn test_stft() {
    let x = common::sp_signal();
    let f, t, zr, zi, f1 = scientific::stft(x, 2.0, "hann", 8, 4);
    assert(f1.kind == scientific::F_NONE && zr.shape[0] == 5 && zr.shape[1] == 11);
    assert(common::near(t.data[1], 2.0) && common::near(f.data[1], 0.25));
    assert(common::loose(zr.data[0 * 11 + 2], 0.03263936287197537) && common::loose(zr.data[1 * 11 + 2], 0.12002113379413716) && common::loose(zr.data[2 * 11 + 2], -0.13099301275623057));
    assert(common::loose(zi.data[1 * 11 + 2], 0.424366549395401) && common::loose(zi.data[2 * 11 + 2], -0.17456594372523282) && common::loose(zi.data[3 * 11 + 2], -0.007774043387693164));
    let f2, t2, wr, wi, g2 = scientific::stft(x, 1.0, "hann", 8, 4, false, false);
    assert(g2.kind == scientific::F_NONE && wr.shape[1] == 9);
    assert(common::loose(wr.data[0 * 9 + 1], 0.03263936287197537) && common::loose(wi.data[1 * 9 + 1], 0.424366549395401) && common::loose(wr.data[4 * 9 + 1], -0.0019199752318358326));
    let a, b, c, d, g3 = scientific::stft(x, 1.0, "hann", 64, 4);
    assert(g3.kind == scientific::F_NONE);
    let a2, b2, c2, d2, g4 = scientific::stft(x, 1.0, "hann", 8, 8);
    assert(g4.kind == scientific::F_RANGE);
    return;
}
