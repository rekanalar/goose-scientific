// The tests of the group bessel. Run them with SCIENTIFIC_SUITES=bessel.
namespace bessel_tests;

import scientific from "../..";
import os;
import common;

// The Bessel filter, pinned against SciPy.
fn test_bessel() {
    let b1, a1, f1 = scientific::bessel(4, [0.3], "lowpass");
    assert(f1.kind == scientific::F_NONE);
    assert(common::all_close(b1, [0.015961530353431626, 0.0638461214137265, 0.09576918212058975, 0.0638461214137265, 0.015961530353431626], 1.0e-9));
    assert(common::all_close(a1, [1.0, -1.4367543893190518, 0.9766689593856015, -0.33008120955383424, 0.04555112514219024], 1.0e-9));
    let b2, a2, f2 = scientific::bessel(3, [0.2], "lowpass", "delay");
    assert(f2.kind == scientific::F_NONE);
    assert(common::all_close(b2, [0.10193658742919462, 0.30580976228758383, 0.30580976228758383, 0.10193658742919462], 1.0e-9));
    assert(common::all_close(a2, [1.0, -0.3610204756237746, 0.20019422826451602, -0.023681053207185], 1.0e-9));
    let b3, a3, f3 = scientific::bessel(5, [0.4], "lowpass", "mag");
    assert(f3.kind == scientific::F_NONE);
    assert(common::all_close(b3, [0.06303420893294857, 0.3151710446647429, 0.6303420893294858, 0.6303420893294858, 0.3151710446647429, 0.06303420893294857], 1.0e-9));
    assert(common::all_close(a3, [1.0, 0.4856329189578801, 0.4307183479288292, 0.0777614975974916, 0.0221282543704655, 0.0008536669996880855], 1.0e-9));
    let b4, a4, f4 = scientific::bessel(4, [0.2, 0.5], "bandpass");
    assert(f4.kind == scientific::F_NONE);
    assert(common::all_close(a4, [1.0, -2.7701647240085996, 4.367024633703179, -4.592807605151357, 3.622387062989189, -2.103367363054253, 0.9116784049308485, -0.2610226067197879, 0.04555112514219019], 1.0e-9));
    let b5, a5, f5 = scientific::bessel(6, [0.3], "highpass", "mag");
    assert(f5.kind == scientific::F_NONE);
    assert(common::all_close(b5, [0.2747192665755552, -1.6483155994533312, 4.120788998633327, -5.494385331511103, 4.120788998633327, -1.6483155994533312, 0.2747192665755552], 1.0e-9));
    let b6, a6, f6 = scientific::bessel(8, [0.25], "lowpass");
    assert(f6.kind == scientific::F_NONE);
    assert(common::all_close(a6, [1.0, -3.718741912749568, 6.488924099995361, -6.836287042161693, 4.710807689534221, -2.159316602316135, 0.6396221620719462, -0.11148601320422366, 0.008725391960805028], 1.0e-8));
    // The sections filter like the polynomials.
    let sos, fs = scientific::bessel_sos(4, [0.3], "lowpass");
    assert(fs.kind == scientific::F_NONE && sos.shape[0] == 2 && sos.shape[1] == 6);
    let x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let ya, fya = scientific::lfilter(b1.data, a1.data, x);
    let yb, fyb = scientific::sosfilt(sos, x);
    assert(fya.kind == scientific::F_NONE && fyb.kind == scientific::F_NONE && common::all_close(yb, ya.data, 1.0e-9));
    // Faults.
    let e1, e2, g1 = scientific::bessel(4, [0.3], "lowpass", "other");
    assert(g1.kind == scientific::F_VALUE);
    let e3, e4, g2 = scientific::bessel(0, [0.3], "lowpass");
    assert(g2.kind == scientific::F_VALUE);
    let e5, e6, g3 = scientific::bessel(4, [1.5], "lowpass");
    assert(g3.kind == scientific::F_RANGE);
    let e7, g4 = scientific::bessel_sos(4, [0.3], "lowpass", "other");
    assert(g4.kind == scientific::F_VALUE);
}
