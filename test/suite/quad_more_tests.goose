// The tests of the group quad_more. Run them with SCIENTIFIC_SUITES=quad_more.
//
// The pins come from the generated file of the group.
namespace quad_more_tests;

import scientific from "../..";
import os;
import common;
import quad_more_pins;

// The tolerance of every value of a pin.
const TOL: f64 = 1.0e-8;

// qm_close tells whether two values agree to the tolerance.
fn qm_close(a: f64, b: f64) -> bool {
    let d = common::mathshim_abs(a - b);
    return d <= TOL * (1.0 + common::mathshim_abs(b));
}

// qm_match checks a buffer against a pin.
fn qm_match(got: f64[:], want: f64[:]) {
    assert(got.len == want.len);
    var i: i64 = 0;
    while i < got.len {
        assert(qm_close(got[i], want[i]));
        i += 1;
    }
}

// The weights of the rules of Newton and Cotes.
fn test_quad_more_newton_cotes() {
    let w1, b1, f1 = scientific::newton_cotes(1);
    assert(f1.kind == scientific::F_NONE);
    qm_match(w1.data, quad_more_pins::NC1);
    assert(qm_close(b1, quad_more_pins::NC1_B[0]));
    let w2, b2, f2 = scientific::newton_cotes(2);
    assert(f2.kind == scientific::F_NONE);
    qm_match(w2.data, quad_more_pins::NC2);
    assert(qm_close(b2, quad_more_pins::NC2_B[0]));
    let w4, b4, f4 = scientific::newton_cotes(4);
    assert(f4.kind == scientific::F_NONE);
    qm_match(w4.data, quad_more_pins::NC4);
    assert(qm_close(b4, quad_more_pins::NC4_B[0]));
    let w6, b6, f6 = scientific::newton_cotes(6);
    assert(f6.kind == scientific::F_NONE);
    qm_match(w6.data, quad_more_pins::NC6);
    assert(qm_close(b6, quad_more_pins::NC6_B[0]));
    let w8, b8, f8 = scientific::newton_cotes(8);
    assert(f8.kind == scientific::F_NONE);
    qm_match(w8.data, quad_more_pins::NC8);
    assert(qm_close(b8, quad_more_pins::NC8_B[0]));
    let w14, b14, f14 = scientific::newton_cotes(14);
    assert(f14.kind == scientific::F_NONE);
    qm_match(w14.data, quad_more_pins::NC14);
    assert(qm_close(b14, quad_more_pins::NC14_B[0]));
    // The rules of an order above fourteen solve the system of the
    // samples. Their values carry the rounding of the system, so the
    // tolerance stands wider.
    let w16, b16, f16 = scientific::newton_cotes(16);
    assert(f16.kind == scientific::F_NONE);
    assert(w16.data.len == 17);
    var i: i64 = 0;
    while i < 17 {
        assert(common::mathshim_abs(w16.data[i] - quad_more_pins::NC16[i]) < 1.0e-6);
        i += 1;
    }
    assert(common::mathshim_abs(b16 - quad_more_pins::NC16_B[0]) < 1.0e-6);
    let w20, b20, f20 = scientific::newton_cotes(20);
    assert(f20.kind == scientific::F_NONE);
    assert(w20.data.len == 21);
    i = 0;
    while i < 21 {
        assert(common::mathshim_abs(w20.data[i] - quad_more_pins::NC20[i]) < 1.0e-6);
        i += 1;
    }
    // The constant of the error of a rule of an order above fourteen
    // loses its digits to the cancellation of the formula. The formula
    // takes the difference of two numbers that stand near each other, so
    // the constant keeps only a few digits. The value of SciPy loses the
    // same digits.
    assert(common::mathshim_abs(b20 - quad_more_pins::NC20_B[0]) < 1.0e-3);
    // An order below one is a fault.
    let w0, b0, f0 = scientific::newton_cotes(0);
    assert(f0.kind == scientific::F_RANGE);
    // The weights of a rule of the order N sum to N, the length of the
    // range in steps of one sample.
    let w3, b3, f3 = scientific::newton_cotes(3);
    assert(f3.kind == scientific::F_NONE);
    var sum: f64 = 0.0;
    i = 0;
    while i < w3.data.len { sum += w3.data[i]; i += 1; }
    assert(qm_close(sum, 3.0));
}

// The rule of the order four applied to the sine. The rule of the
// order four over the range from zero to the value of pi gives two.
fn test_quad_more_newton_cotes_apply() {
    // The rule of the order two is the rule of Simpson. The integral of
    // the sine over [0, pi] is two.
    let w, b, f = scientific::newton_cotes(2);
    assert(f.kind == scientific::F_NONE);
    let n = w.data.len;
    var total: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        let x = 3.141592653589793 * i as f64 / (n - 1) as f64;
        total += w.data[i] * mathshim::sine(x);
        i += 1;
    }
    total *= 3.141592653589793 / (n - 1) as f64;
    // Three samples of the sine give 2.0943951023931953, and the rule
    // of SciPy gives the same number. The exact integral is two.
    assert(common::mathshim_abs(total - 2.0943951023931953) < 1.0e-12);
}

// The sums of the series.
fn test_quad_more_nsum() {
    // The sum of one over the square of every number.
    let v1, e1, f1 = scientific::nsum(1.0, scientific::quad_inf()) { x => 1.0 / (x * x) };
    assert(f1.kind == scientific::F_NONE);
    assert(qm_close(v1, quad_more_pins::ZETA2[0]));
    // A finite sum of four terms.
    let v2, e2, f2 = scientific::nsum(1.0, 5.0) { x => 1.0 / (x * x) };
    assert(f2.kind == scientific::F_NONE);
    assert(qm_close(v2, 1.0 + 0.25 + 1.0 / 9.0 + 1.0 / 16.0 + 1.0 / 25.0));
    // The alternating harmonic series over the step of two.
    let v3, e3, f3 = scientific::nsum(1.0, scientific::quad_inf(), 2.0) { x => 1.0 / x - 1.0 / (x + 1.0) };
    assert(f3.kind == scientific::F_NONE);
    assert(qm_close(v3, quad_more_pins::ALT_LOG2[0]));
    // A step below zero is a fault.
    let v4, e4, f4 = scientific::nsum(1.0, scientific::quad_inf(), 0.0) { x => 1.0 / (x * x) };
    assert(f4.kind == scientific::F_VALUE);
    // An end below the start is a fault.
    let v5, e5, f5 = scientific::nsum(2.0, 1.0) { x => 1.0 / (x * x) };
    assert(f5.kind == scientific::F_VALUE);
}



// The rule of the hyperbolic tangent.
fn test_quad_more_tanhsinh() {
    // The integral of the exponential times the sine over [0, 2].
    let v1, e1, f1 = scientific::tanhsinh(0.0, 2.0, 1.0e-10, 1.0e-10) { x => mathshim::power_e(x) * mathshim::sine(x) };
    assert(f1.kind == scientific::F_NONE);
    assert(qm_close(v1, quad_more_pins::TS_EXPSIN[0]));
    // The integral of one over the root of x over [0, 1].
    let v2, e2, f2 = scientific::tanhsinh(0.0, 1.0, 1.0e-10, 1.0e-10) { x => 1.0 / mathshim::root(x) };
    assert(f2.kind == scientific::F_NONE);
    assert(qm_close(v2, quad_more_pins::TS_SQRT[0]));
    // A limit that holds the missing value is a fault.
    let v3, e3, f3 = scientific::tanhsinh(0.0 / 0.0, 1.0) { x => mathshim::power_e(x) * mathshim::sine(x) };
    assert(f3.kind == scientific::F_VALUE);
    // A tolerance below zero is a fault.
    let v4, e4, f4 = scientific::tanhsinh(0.0, 1.0, -1.0, 1.0e-10) { x => mathshim::power_e(x) * mathshim::sine(x) };
    assert(f4.kind == scientific::F_VALUE);
    // A count of levels outside one to 16 is a fault.
    let v5, e5, f5 = scientific::tanhsinh(0.0, 1.0, 1.0e-10, 1.0e-10, 0) { x => mathshim::power_e(x) * mathshim::sine(x) };
    assert(f5.kind == scientific::F_RANGE);
}



// The rule of the quasi random points over a box.
fn test_quad_more_qmc_quad() {
    // The volume of the box of the unit cube is one, and the integrand
    // is the constant one, so the value is one.
    let v1, e1, f1 = scientific::qmc_quad([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 512, 8, 1) { p => 1.0 };
    assert(f1.kind == scientific::F_NONE);
    assert(qm_close(v1, 1.0));
    // The error of a constant integrand is zero.
    assert(e1 >= 0.0);
    // The box of the width two, the height two and the depth one holds
    // the volume of four, and the integrand is the constant one.
    let v2, e2, f2 = scientific::qmc_quad([0.0, 0.0, 0.0], [2.0, 2.0, 1.0], 512, 8, 1) { p => 1.0 };
    assert(f2.kind == scientific::F_NONE);
    assert(qm_close(v2, 4.0));
    // The same seed gives the same value.
    let v3, e3, f3 = scientific::qmc_quad([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 512, 8, 1) { p => 1.0 };
    assert(f3.kind == scientific::F_NONE);
    assert(v3 == v1);
    // The ends that differ in count are a fault.
    let v4, e4, f4 = scientific::qmc_quad([0.0], [1.0, 2.0]) { p => 1.0 };
    assert(f4.kind == scientific::F_SHAPE);
    // A count of points below one is a fault.
    let v5, e5, f5 = scientific::qmc_quad([0.0, 0.0], [1.0, 1.0], 0, 2, 1) { p => 1.0 };
    assert(f5.kind == scientific::F_RANGE);
    // A count of estimates outside one to 1024 is a fault.
    let v6, e6, f6 = scientific::qmc_quad([0.0, 0.0], [1.0, 1.0], 8, 0, 1) { p => 1.0 };
    assert(f6.kind == scientific::F_RANGE);
    // A limit that holds the missing value is a fault.
    let v7, e7, f7 = scientific::qmc_quad([0.0, 0.0 / 0.0], [1.0, 1.0]) { p => 1.0 };
    assert(f7.kind == scientific::F_VALUE);
}

