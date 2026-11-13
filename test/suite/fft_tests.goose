// The tests of the group fft. Run them with SCIENTIFIC_SUITES=fft.
namespace fft_tests;

import scientific from "../..";
import os;
import common;

// The discrete Fourier transform.
fn test_fft() {
    // The transform of a signal that holds one value.
    let re, im, f = scientific::fft([1.0, 0.0, 0.0, 0.0]);
    assert(f.kind == scientific::F_NONE && re.data.len == 4);
    assert(re.data[0] == 1.0 && re.data[1] == 1.0 && im.data[0] == 0.0);
    // The transform of a signal of four values.
    let re2, im2, f2 = scientific::fft([1.0, 2.0, 3.0, 4.0]);
    assert(f2.kind == scientific::F_NONE);
    assert(common::close(re2.data[0], 10.0) && common::close(re2.data[1], -2.0) && common::close(im2.data[1], 2.0));
    assert(common::close(re2.data[2], -2.0) && common::close(re2.data[3], -2.0) && common::close(im2.data[3], -2.0));
    // A length that holds a factor that is not a power of two.
    let re6, im6, f3 = scientific::fft([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert(f3.kind == scientific::F_NONE && re6.data.len == 6);
    assert(common::close(re6.data[0], 21.0) && common::close(re6.data[1], -3.0) && common::close(im6.data[1], 5.19615242));
    assert(common::close(re6.data[2], -3.0) && common::close(im6.data[2], 1.73205081));
    assert(common::close(re6.data[3], -3.0) && common::close(im6.data[5], -5.19615242));
    // The signal comes back from its transform.
    let back, back_im, f4 = scientific::ifft(re2.data, im2.data);
    assert(f4.kind == scientific::F_NONE);
    assert(common::close(back.data[0], 1.0) && common::close(back.data[3], 4.0));
    let back6, back6_im, f5 = scientific::ifft(re6.data, im6.data);
    assert(f5.kind == scientific::F_NONE);
    assert(common::close(back6.data[0], 1.0) && common::close(back6.data[5], 6.0));
    // A signal that holds no element is a fault.
    let bad, bad_im, f6 = scientific::fft([]);
    assert(f6.kind == scientific::F_VALUE);
    // The frequencies of a transform.
    let fr, f7 = scientific::fftfreq(4, 1.0);
    assert(f7.kind == scientific::F_NONE);
    assert(common::close(fr.data[0], 0.0) && common::close(fr.data[1], 0.25) && common::close(fr.data[2], -0.5) && common::close(fr.data[3], -0.25));
    let fr2, f8 = scientific::fftfreq(4, 0.5);
    assert(common::close(fr2.data[1], 0.5));
    // The shift moves the zero frequency to the middle.
    let sh, f9 = scientific::fftshift(fr);
    assert(f9.kind == scientific::F_NONE);
    assert(common::close(sh.data[0], -0.5) && common::close(sh.data[1], -0.25) && common::close(sh.data[2], 0.0));
    // The transform of a matrix.
    let m = common::make23();
    let mre, mim, f10 = scientific::fft2(m);
    assert(f10.kind == scientific::F_NONE && mre.shape[0] == 2 && mre.shape[1] == 3);
    assert(common::close(mre.data[0], 21.0) && common::close(mre.data[3], -9.0) && common::close(mim.data[1], 1.73205081));
    // The transform with only the positive frequencies.
    let rre, rim, f11 = scientific::rfft([1.0, 2.0, 3.0, 4.0]);
    assert(f11.kind == scientific::F_NONE && rre.data.len == 3);
    assert(common::close(rre.data[0], 10.0) && common::close(rre.data[1], -2.0) && common::close(rim.data[1], 2.0));
    let rback, f12 = scientific::irfft(rre.data, rim.data, 4);
    assert(f12.kind == scientific::F_NONE);
    assert(common::close(rback.data[0], 1.0) && common::close(rback.data[3], 4.0));
    // A sine of three turns over eight places peaks at the place of
    // its frequency, with the sign of NumPy.
    var sine: f64[>..] = [];
    var i9: i64 = 0;
    while i9 < 8 {
        sine.push(mathshim::sine(6.283185307179586 * 3.0 * i9 / 8.0));
        i9 += 1;
    }
    let sre, sim, f13 = scientific::fft(sine);
    assert(f13.kind == scientific::F_NONE);
    assert(common::close(sre.data[3], 0.0) && common::close(sim.data[3], -4.0));
    // The peak stands in the positive half. The negative half holds
    // the mirror of it.
    var peak: i64 = 0;
    var best: f64 = 0.0;
    i9 = 1;
    while i9 < 4 {
        let mag = sre.data[i9] * sre.data[i9] + sim.data[i9] * sim.data[i9];
        if mag > best { best = mag; peak = i9; }
        i9 += 1;
    }
    assert(peak == 3 && best > 15.9 && common::close(sre.data[0], 0.0));
    // The transform of a count of dimensions that is not two is a
    // fault.
    let bad2, bad2_im, f14 = scientific::fft2(fr);
    assert(f14.kind == scientific::F_SHAPE);
    // Parts that hold a different count are a fault.
    let bad3, f15 = scientific::irfft(rre.data, [1.0], 4);
    assert(f15.kind == scientific::F_SHAPE);
}
