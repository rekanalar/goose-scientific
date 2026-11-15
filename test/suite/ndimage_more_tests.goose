// The tests of the group ndimage_more. Run them with SCIENTIFIC_SUITES=ndimage_more.
//
// The pins come from the generated file of the group.
namespace ndimage_more_tests;

import scientific from "../..";
import os;
import common;
import ndimage_more_pins;

// The tolerance of every value of a pin.
const TOL: f64 = 1.0e-8;

// nm_close tells whether two values agree to the tolerance.
fn nm_close(a: f64, b: f64) -> bool {
    let d = common::mathshim_abs(a - b);
    return d <= TOL * (1.0 + common::mathshim_abs(b));
}

// nm_img gives the image of the pins as a real array.
fn nm_img() -> scientific::Arr {
    var sh: i64[..8] = [];
    sh.push(6);
    sh.push(8);
    return scientific::Arr { ndim: 2, shape: sh, data: scientific::copy_slice(ndimage_more_pins::IMG) };
}

// The four filters of the transform. The input goes through the
// complex array, and the result comes back to the real part.
fn test_ndimage_more_fourier() {
    let cimg = scientific::complex_real(nm_img());
    let g, f1 = scientific::fourier_gaussian(cimg, [1.2, 0.7]);
    assert(f1.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < 48 {
        assert(nm_close(scientific::real(g).data[i], ndimage_more_pins::GAUSS[i]));
        i += 1;
    }
    let u, f2 = scientific::fourier_uniform(cimg, [3.0, 2.0]);
    assert(f2.kind == scientific::F_NONE);
    i = 0;
    while i < 48 {
        assert(nm_close(scientific::real(u).data[i], ndimage_more_pins::UNIFORM[i]));
        i += 1;
    }
    let e, f3 = scientific::fourier_ellipsoid(cimg, [3.0, 4.0]);
    assert(f3.kind == scientific::F_NONE);
    i = 0;
    while i < 48 {
        assert(nm_close(scientific::real(e).data[i], ndimage_more_pins::ELLIPSOID[i]));
        i += 1;
    }
    let s, f4 = scientific::fourier_shift(cimg, [1.5, -2.25]);
    assert(f4.kind == scientific::F_NONE);
    i = 0;
    while i < 48 {
        assert(nm_close(scientific::real(s).data[i], ndimage_more_pins::SHIFT[i]));
        i += 1;
    }
}

// The faults of the filters.
fn test_ndimage_more_fourier_faults() {
    let cimg = scientific::complex_real(nm_img());
    // A count of the parameters that does not agree with the shape is
    // a fault.
    let g, f1 = scientific::fourier_gaussian(cimg, [1.0]);
    assert(f1.kind == scientific::F_SHAPE);
    let u, f2 = scientific::fourier_uniform(cimg, [1.0, 2.0, 3.0]);
    assert(f2.kind == scientific::F_SHAPE);
    // A parameter that holds the missing value is a fault.
    let e, f3 = scientific::fourier_ellipsoid(cimg, [1.0, 0.0 / 0.0]);
    assert(f3.kind == scientific::F_VALUE);
    // An operand that is not finite is a fault.
    var bad: f64[>..] = [];
    var i: i64 = 0;
    while i < 96 {
        if i == 5 { bad.push(0.0 / 0.0); } else { bad.push(cimg.data[i]); }
        i += 1;
    }
    let badimg = scientific::CArr { ndim: 2, shape: cimg.shape, data: copy(bad) };
    let s, f4 = scientific::fourier_shift(badimg, [1.0, 1.0]);
    assert(f4.kind == scientific::F_VALUE);
}

// The function of Bessel of the order one.
fn test_ndimage_more_bessel() {
    // The root of the function of Bessel stands at about 3.8317.
    let lo = scientific::bessel_j1(3.8317);
    assert(lo > 0.0 && common::mathshim_abs(lo) < 1.0e-4);
    // The value at zero is zero, and the value at one stands at about
    // 0.44005.
    assert(scientific::bessel_j1(0.0) == 0.0);
    let v = scientific::bessel_j1(1.0);
    assert(common::mathshim_abs(v - 0.4400505857) < 1.0e-6);
    // The value stands odd in the sign: j1 of minus x is minus j1 of x.
    assert(scientific::bessel_j1(-1.0) == -v);
}

// The watershed of the distance transform.
fn test_ndimage_more_watershed() {
    var sh: i64[..8] = [];
    sh.push(5);
    sh.push(7);
    let img = scientific::Arr { ndim: 2, shape: sh, data: scientific::copy_slice(ndimage_more_pins::WS_IN) };
    let mk = scientific::Arr { ndim: 2, shape: sh, data: scientific::copy_slice(ndimage_more_pins::WS_MARK) };
    let out, f = scientific::watershed_ift(img, mk);
    assert(f.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < 35 {
        assert(out.data[i] == ndimage_more_pins::WS_OUT[i]);
        i += 1;
    }
    // A shape of the markers that does not agree is a fault.
    var sh2: i64[..8] = [];
    sh2.push(5);
    sh2.push(6);
    var short: f64[>..] = [];
    i = 0;
    while i < 30 { short.push(0.0); i += 1; }
    let mk2 = scientific::Arr { ndim: 2, shape: sh2, data: copy(short) };
    let out2, f2 = scientific::watershed_ift(img, mk2);
    assert(f2.kind == scientific::F_SHAPE);
    // A value of the input below zero is a fault.
    var neg: f64[>..] = [];
    i = 0;
    while i < 35 {
        if i == 3 { neg.push(-1.0); } else { neg.push(img.data[i]); }
        i += 1;
    }
    let negimg = scientific::Arr { ndim: 2, shape: sh, data: copy(neg) };
    let out3, f3 = scientific::watershed_ift(negimg, mk);
    assert(f3.kind == scientific::F_VALUE);
}

// The standard deviation of the labels of a region.
fn test_ndimage_more_std() {
    var sh: i64[..8] = [];
    sh.push(2);
    sh.push(3);
    let a = scientific::Arr { ndim: 2, shape: sh, data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
    let lab = scientific::Arr { ndim: 2, shape: sh, data: [1.0, 1.0, 1.0, 2.0, 2.0, 2.0] };
    // With no index list SciPy pools every label above zero into one
    // group: the standard deviation of one to six is 1.7078 (SciPy 1.18.1).
    let sd, f = scientific::standard_deviation_labels(a, lab);
    assert(f.kind == scientific::F_NONE);
    assert(sd.data.len == 1);
    assert(common::mathshim_abs(sd.data[0] - 1.707825127659933) < 1.0e-12);
    // With the index list one and two every group stands alone. The
    // standard deviation of a group is the root of the mean of the
    // square of the distance to the mean of the group: root of 2/3.
    let sg, f2 = scientific::standard_deviation_labels(a, lab, [1, 2]);
    assert(f2.kind == scientific::F_NONE);
    assert(sg.data.len == 2);
    assert(common::mathshim_abs(sg.data[0] - mathshim::root(2.0 / 3.0)) < 1.0e-12);
    assert(common::mathshim_abs(sg.data[1] - mathshim::root(2.0 / 3.0)) < 1.0e-12);
}

// An array that a caller built by hand and broke does not agree with
// its shape. The call that reads the buffer of an array gives a fault.
fn test_watershed_broken_array() {
    // The input and the markers hold the same count of values, but the
    // shape asks for more values than the buffer holds.
    var sh: i64[..8] = [];
    sh.push(5);
    sh.push(7);
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < 30 { buf.push(1.0); i += 1; }
    let img = scientific::Arr { ndim: 2, shape: sh, data: copy(buf) };
    let mk = scientific::Arr { ndim: 2, shape: sh, data: copy(buf) };
    let out, f = scientific::watershed_ift(img, mk);
    assert(f.kind == scientific::F_SHAPE);
}
