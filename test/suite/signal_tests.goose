// The tests of the group signal. Run them with SCIENTIFIC_SUITES=signal.
namespace signal_tests;

import scientific from "../..";
import os;
import common;

fn test_signal() {
    // The overlap of two signals.
    let c1, f1 = scientific::convolve([1.0, 2.0, 3.0], [0.0, 1.0, 0.5]);
    assert(f1.kind == scientific::F_NONE && c1.data.len == 5);
    assert(c1.data[0] == 0.0 && c1.data[1] == 1.0 && c1.data[2] == 2.5);
    assert(c1.data[3] == 4.0 && c1.data[4] == 1.5);
    // The match of a pattern in a signal.
    let k1, f2 = scientific::correlate([1.0, 2.0, 3.0], [0.0, 1.0, 0.5]);
    assert(f2.kind == scientific::F_NONE && k1.data.len == 5);
    assert(k1.data[0] == 0.5 && k1.data[1] == 2.0 && k1.data[2] == 3.5);
    assert(k1.data[3] == 3.0 && k1.data[4] == 0.0);
    // The line under a straight signal comes out, so the signal holds
    // zeros.
    let d1, f3 = scientific::detrend([0.0, 1.0, 2.0, 3.0]);
    assert(f3.kind == scientific::F_NONE && d1.data.len == 4);
    assert(common::mathshim_abs(d1.data[0]) < 0.0000001 && common::mathshim_abs(d1.data[3]) < 0.0000001);
    // The size of the parts of a sine holds its frequency.
    var y: f64[>..] = [];
    var i: i64 = 0;
    while i < 64 { y.push(mathshim::sine(6.283185307179586 * 8.0 * i / 64.0)); i += 1; }
    let w, f4 = scientific::welch(y, 64);
    assert(f4.kind == scientific::F_NONE && w.data.len == 33);
    var best: i64 = 0;
    i = 1;
    while i < w.data.len {
        if w.data[i] > w.data[best] { best = i; }
        i += 1;
    }
    assert(best == 8 && common::close(w.data[best], 21.333333333333343));
    // A signal that holds no value is a fault.
    let bad1, fb1 = scientific::convolve([], [1.0]);
    assert(fb1.kind == scientific::F_VALUE);
    let bad2, fb2 = scientific::correlate([1.0], []);
    assert(fb2.kind == scientific::F_VALUE);
    // A signal of too few values is a fault.
    let bad3, fb3 = scientific::detrend([1.0]);
    assert(fb3.kind == scientific::F_VALUE);
    // A part of too few values is a fault.
    let bad4, fb4 = scientific::welch(y, 1);
    assert(fb4.kind == scientific::F_VALUE);
    // A signal that holds fewer values than a part is a fault.
    let bad5, fb5 = scientific::welch([1.0, 2.0], 8);
    assert(fb5.kind == scientific::F_VALUE);
}

// spectrogram and welch_psd are pinned against SciPy.
fn test_spectral() {
    let x = common::sp_signal();
    let f, t, s, f1 = scientific::spectrogram(x, 2.0, "tukey", 0.25, 16, 4);
    assert(f1.kind == scientific::F_NONE && s.shape[0] == 9 && s.shape[1] == 3);
    assert(common::all_near(t, [4.0, 10.0, 16.0]) && common::near(f.data[1], 0.125));
    assert(common::loose(s.data[0], 0.016398352268530342) && common::loose(s.data[1 * 3], 1.2403196212067757) && common::loose(s.data[2 * 3], 2.704032777944856) && common::loose(s.data[3 * 3], 0.0031842838682634464) && common::loose(s.data[3 * 3 + 1], 0.0028130141030115077) && common::loose(s.data[3 * 3 + 2], 0.05353120041768578));
    let f2, t2, s2, g2 = scientific::spectrogram(x, 1.0, "hann", 0.0 / 0.0, 8, 2, false, "spectrum");
    assert(g2.kind == scientific::F_NONE && s2.shape[0] == 5 && s2.shape[1] == 6);
    assert(common::loose(s2.data[0 * 6 + 1], 0.7785029913294211) && common::loose(s2.data[1 * 6 + 1], 1.0224606287871703) && common::loose(s2.data[2 * 6 + 1], 0.09495759220933907) && common::loose(s2.data[3 * 6 + 1], 0.00015773875399762466) && common::loose(s2.data[4 * 6 + 1], 3.9928095795776335e-06));
    let wf, p, f3 = scientific::welch_psd(x, 2.0, "hann", 16, 8);
    assert(f3.kind == scientific::F_NONE && p.data.len == 9);
    assert(common::loose(p.data[0], 0.0540555950887493) && common::loose(p.data[1], 1.3550356626194178) && common::loose(p.data[2], 2.4944916696212753) && common::loose(p.data[3], 0.31407830694318295));
    let wf2, p2, f4 = scientific::welch_psd(x, 1.0, "hamming", 9, -1, false, "spectrum");
    assert(f4.kind == scientific::F_NONE && p2.data.len == 5);
    assert(common::loose(p2.data[0], 1.2150560394826984) && common::loose(p2.data[1], 0.8644536996548504) && common::loose(p2.data[2], 0.09172796773120981));
    let a, b, c, f5 = scientific::spectrogram(x, 1.0, "hann", 0.0 / 0.0, 64);
    assert(f5.kind == scientific::F_RANGE);
    let w3, p3, f6 = scientific::welch_psd(x, 1.0, "hann", 8, 8);
    assert(f6.kind == scientific::F_RANGE);
    return;
}
