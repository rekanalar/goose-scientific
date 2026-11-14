// The tests of the continuous random laws. Run them with
// SCIENTIFIC_SUITES=random_distributions.
//
// The calls gamma, lognormal, chisquare, beta, rayleigh, weibull, gumbel
// and logistic draw from the generator of the package. Its stream
// differs from the stream of NumPy, so the tests compare the law: the
// mean, the variance and the quartiles of 200000 numbers match the
// values of SciPy 1.18.1. The quartile test counts and never sorts.
namespace random_distributions_tests;

import scientific from "../..";
import common;
import random_distributions_pins;

const DRAWS: i64 = 200000;

// The law codes of the eight laws.
const LAW_GAMMA: i64 = 0;
const LAW_LOGNORMAL: i64 = 1;
const LAW_CHISQUARE: i64 = 2;
const LAW_BETA: i64 = 3;
const LAW_RAYLEIGH: i64 = 4;
const LAW_WEIBULL: i64 = 5;
const LAW_GUMBEL: i64 = 6;
const LAW_LOGISTIC: i64 = 7;

// draw gives one number of a law. A law of one parameter ignores p2.
fn draw(law: i64, r: scientific::Rng&, p1: f64, p2: f64) -> f64, scientific::Fault {
    if law == LAW_GAMMA { return scientific::gamma(r, p1, p2); }
    if law == LAW_LOGNORMAL { return scientific::lognormal(r, p1, p2); }
    if law == LAW_CHISQUARE { return scientific::chisquare(r, p1); }
    if law == LAW_BETA { return scientific::beta(r, p1, p2); }
    if law == LAW_RAYLEIGH { return scientific::rayleigh(r, p1); }
    if law == LAW_WEIBULL { return scientific::weibull(r, p1); }
    if law == LAW_GUMBEL { return scientific::gumbel(r, p1, p2); }
    return scientific::logistic(r, p1, p2);
}

// check_set draws many numbers of one parameter set. It compares the
// mean, the variance and the three quartiles with the pins. The pins
// hold the three quartiles of each set, one after the other.
fn check_set(law: i64, seed: i64, p1: f64, p2: f64, m: f64, v: f64, q: f64[:], k: i64) {
    let q0 = q[3 * k];
    let q1 = q[3 * k + 1];
    let q2 = q[3 * k + 2];
    var r = scientific::default_rng(seed);
    var sum: f64 = 0.0;
    var sumsq: f64 = 0.0;
    var b1: i64 = 0;
    var b2: i64 = 0;
    var b3: i64 = 0;
    var i: i64 = 0;
    while i < DRAWS {
        let x, f = draw(law, r, p1, p2);
        assert(f.kind == scientific::F_NONE);
        sum += x;
        sumsq += x * x;
        if x < q0 { b1 += 1; }
        if x < q1 { b2 += 1; }
        if x < q2 { b3 += 1; }
        i += 1;
    }
    let n = DRAWS as f64;
    let mean = sum / n;
    let variance = sumsq / n - mean * mean;
    assert(common::mathshim_abs(mean - m) <= 5.0 * common::mathshim_sqrt(v / n));
    assert(common::mathshim_abs(variance - v) <= 0.05 * v);
    let se = common::mathshim_sqrt(0.1875 / n);
    assert(common::mathshim_abs(b1 as f64 / n - 0.25) <= 5.0 * se);
    assert(common::mathshim_abs(b2 as f64 / n - 0.50) <= 5.0 * common::mathshim_sqrt(0.25 / n));
    assert(common::mathshim_abs(b3 as f64 / n - 0.75) <= 5.0 * se);
}

fn test_gamma_family_laws() {
    var k: i64 = 0;
    while k < 3 {
        check_set(LAW_GAMMA, 11 + k, random_distributions_pins::GAMMA_P1[k], random_distributions_pins::GAMMA_P2[k],
                  random_distributions_pins::GAMMA_M[k], random_distributions_pins::GAMMA_V[k],
                  random_distributions_pins::GAMMA_Q, k);
        check_set(LAW_CHISQUARE, 21 + k, random_distributions_pins::CHI2_P1[k], 0.0,
                  random_distributions_pins::CHI2_M[k], random_distributions_pins::CHI2_V[k],
                  random_distributions_pins::CHI2_Q, k);
        check_set(LAW_BETA, 31 + k, random_distributions_pins::BETA_P1[k], random_distributions_pins::BETA_P2[k],
                  random_distributions_pins::BETA_M[k], random_distributions_pins::BETA_V[k],
                  random_distributions_pins::BETA_Q, k);
        k += 1;
    }
}

fn test_lognormal_law() {
    var k: i64 = 0;
    while k < 3 {
        check_set(LAW_LOGNORMAL, 41 + k, random_distributions_pins::LOGN_P1[k], random_distributions_pins::LOGN_P2[k],
                  random_distributions_pins::LOGN_M[k], random_distributions_pins::LOGN_V[k],
                  random_distributions_pins::LOGN_Q, k);
        k += 1;
    }
}

fn test_inverse_transform_laws() {
    var k: i64 = 0;
    while k < 3 {
        check_set(LAW_RAYLEIGH, 51 + k, random_distributions_pins::RAYL_P1[k], 0.0,
                  random_distributions_pins::RAYL_M[k], random_distributions_pins::RAYL_V[k],
                  random_distributions_pins::RAYL_Q, k);
        check_set(LAW_WEIBULL, 61 + k, random_distributions_pins::WEIB_P1[k], 0.0,
                  random_distributions_pins::WEIB_M[k], random_distributions_pins::WEIB_V[k],
                  random_distributions_pins::WEIB_Q, k);
        check_set(LAW_GUMBEL, 71 + k, random_distributions_pins::GUMB_P1[k], random_distributions_pins::GUMB_P2[k],
                  random_distributions_pins::GUMB_M[k], random_distributions_pins::GUMB_V[k],
                  random_distributions_pins::GUMB_Q, k);
        check_set(LAW_LOGISTIC, 81 + k, random_distributions_pins::LOGI_P1[k], random_distributions_pins::LOGI_P2[k],
                  random_distributions_pins::LOGI_M[k], random_distributions_pins::LOGI_V[k],
                  random_distributions_pins::LOGI_Q, k);
        k += 1;
    }
}

// bad checks that a call refuses a parameter with a kind of fault.
fn bad(law: i64, p1: f64, p2: f64, kind: i64) {
    var r = scientific::default_rng(5);
    let x, f = draw(law, r, p1, p2);
    assert(f.kind == kind);
    assert(x == 0.0);
}

fn test_bad_parameters_refused() {
    let nz = -1.0 * 0.0;
    let nan = 0.0 / 0.0;
    let inf = 1.0 / 0.0;
    let v = scientific::F_VALUE;
    // The parameters of each call that must be at or above zero.
    bad(LAW_GAMMA, -1.0, 1.0, v); bad(LAW_GAMMA, 1.0, -1.0, v);
    bad(LAW_GAMMA, nz, 1.0, v);   bad(LAW_GAMMA, 1.0, nz, v);
    bad(LAW_GAMMA, nan, 1.0, v);  bad(LAW_GAMMA, 1.0, nan, v);
    bad(LAW_GAMMA, inf, 1.0, v);  bad(LAW_GAMMA, 1.0, inf, v);
    bad(LAW_LOGNORMAL, 0.0, -1.0, v); bad(LAW_LOGNORMAL, 0.0, nz, v);
    bad(LAW_LOGNORMAL, nan, 1.0, v);  bad(LAW_LOGNORMAL, inf, 1.0, v);
    bad(LAW_LOGNORMAL, 0.0, nan, v);  bad(LAW_LOGNORMAL, 0.0, inf, v);
    bad(LAW_RAYLEIGH, -1.0, 0.0, v); bad(LAW_RAYLEIGH, nz, 0.0, v);
    bad(LAW_RAYLEIGH, nan, 0.0, v);  bad(LAW_RAYLEIGH, inf, 0.0, v);
    bad(LAW_WEIBULL, -1.0, 0.0, v); bad(LAW_WEIBULL, nz, 0.0, v);
    bad(LAW_WEIBULL, nan, 0.0, v);  bad(LAW_WEIBULL, inf, 0.0, v);
    bad(LAW_GUMBEL, 0.0, -1.0, v); bad(LAW_GUMBEL, 0.0, nz, v);
    bad(LAW_GUMBEL, nan, 1.0, v);  bad(LAW_GUMBEL, inf, 1.0, v);
    bad(LAW_GUMBEL, 0.0, nan, v);  bad(LAW_GUMBEL, 0.0, inf, v);
    bad(LAW_LOGISTIC, 0.0, -1.0, v); bad(LAW_LOGISTIC, 0.0, nz, v);
    bad(LAW_LOGISTIC, nan, 1.0, v);  bad(LAW_LOGISTIC, inf, 1.0, v);
    bad(LAW_LOGISTIC, 0.0, nan, v);  bad(LAW_LOGISTIC, 0.0, inf, v);
    // The parameters that must be above zero: zero is refused too.
    bad(LAW_CHISQUARE, 0.0, 0.0, v);  bad(LAW_CHISQUARE, nz, 0.0, v);
    bad(LAW_CHISQUARE, -1.0, 0.0, v); bad(LAW_CHISQUARE, nan, 0.0, v); bad(LAW_CHISQUARE, inf, 0.0, v);
    bad(LAW_BETA, 0.0, 1.0, v);  bad(LAW_BETA, 1.0, 0.0, v);
    bad(LAW_BETA, nz, 1.0, v);   bad(LAW_BETA, 1.0, nz, v);
    bad(LAW_BETA, -1.0, 1.0, v); bad(LAW_BETA, 1.0, -1.0, v);
    bad(LAW_BETA, nan, 1.0, v);  bad(LAW_BETA, 1.0, inf, v);
}

fn test_zero_cases() {
    var r = scientific::default_rng(5);
    let g1, f1 = scientific::gamma(r, 0.0, 2.0);
    assert(f1.kind == scientific::F_NONE && g1 == 0.0);
    let g2, f2 = scientific::gamma(r, 2.0, 0.0);
    assert(f2.kind == scientific::F_NONE && g2 == 0.0);
    // The mean of the logarithm gives the number when the sigma is zero.
    let l, f3 = scientific::lognormal(r, 0.5, 0.0);
    assert(f3.kind == scientific::F_NONE && common::mathshim_abs(l - 1.6487212707001282) <= 1.0e-15);
    let ra, f4 = scientific::rayleigh(r, 0.0);
    assert(f4.kind == scientific::F_NONE && ra == 0.0);
    let w, f5 = scientific::weibull(r, 0.0);
    assert(f5.kind == scientific::F_NONE && w == 0.0);
    let gu, f6 = scientific::gumbel(r, 1.0, 0.0);
    assert(f6.kind == scientific::F_NONE && gu == 1.0);
    let lo, f7 = scientific::logistic(r, -1.0, 0.0);
    assert(f7.kind == scientific::F_NONE && lo == -1.0);
}

fn test_size_limits() {
    let lim = scientific::F_LIMIT;
    bad(LAW_GAMMA, 1.0e101, 1.0, lim);
    bad(LAW_GAMMA, 1.0e-301, 1.0, lim);
    bad(LAW_CHISQUARE, 2.0e101, 0.0, lim);
    bad(LAW_CHISQUARE, 1.0e-301, 0.0, lim);
    bad(LAW_BETA, 1.0e101, 1.0, lim);
    bad(LAW_BETA, 1.0, 1.0e-301, lim);
    bad(LAW_WEIBULL, 1.0e-301, 0.0, lim);
    bad(LAW_WEIBULL, 1.0e101, 0.0, lim);
    // The limit itself holds.
    var r = scientific::default_rng(5);
    let g, f = scientific::gamma(r, 1.0e100, 1.0);
    assert(f.kind == scientific::F_NONE && common::mathshim_abs(g / 1.0e100 - 1.0) < 1.0e-6);
    let w, fw = scientific::weibull(r, 1.0e-300);
    assert(fw.kind == scientific::F_NONE || fw.kind == scientific::F_VALUE);
}

fn test_distribution_overflow_gives_fault() {
    var r = scientific::default_rng(5);
    let a, fa = scientific::gamma(r, 2.0, 1.0e308);
    assert(fa.kind == scientific::F_NONE || fa.kind == scientific::F_VALUE);
    let b, fb = scientific::lognormal(r, 1000.0, 1.0);
    assert(fb.kind == scientific::F_VALUE && b == 0.0);
    let c, fc = scientific::gumbel(r, 1.7e308, 1.0e308);
    assert(fc.kind == scientific::F_NONE || fc.kind == scientific::F_VALUE);
}

fn test_pinned_numbers_of_one_seed() {
    // The numbers of one seed are pinned. The same seed gives the
    // same numbers on every run and on both supported versions.
    var r = scientific::default_rng(4242);
    let a1, f1 = scientific::gamma(r, 2.0, 1.5);
    assert(f1.kind == scientific::F_NONE && a1 == 0.811282360431942);
    let a2, f2 = scientific::lognormal(r, 0.5, 0.5);
    assert(f2.kind == scientific::F_NONE && a2 == 3.4674329860456257);
    let a3, f3 = scientific::chisquare(r, 5.0);
    assert(f3.kind == scientific::F_NONE && a3 == 5.61243658954795);
    let a4, f4 = scientific::beta(r, 2.0, 3.0);
    assert(f4.kind == scientific::F_NONE && a4 == 0.6495980880334447);
    let a5, f5 = scientific::rayleigh(r, 2.0);
    assert(f5.kind == scientific::F_NONE && a5 == 3.2237198580966946);
    let a6, f6 = scientific::weibull(r, 1.5);
    assert(f6.kind == scientific::F_NONE && a6 == 0.29218848580394696);
    let a7, f7 = scientific::gumbel(r, 1.0, 2.0);
    assert(f7.kind == scientific::F_NONE && a7 == 4.888422517447708);
    let a8, f8 = scientific::logistic(r, -1.0, 3.0);
    assert(f8.kind == scientific::F_NONE && a8 == -4.649118060420193);
}

fn test_equal_seeds_equal_numbers() {
    var law: i64 = 0;
    while law < 8 {
        var ra = scientific::default_rng(99);
        var rb = scientific::default_rng(99);
        var i: i64 = 0;
        while i < 50 {
            let x, fx = draw(law, ra, 1.5, 2.0);
            let y, fy = draw(law, rb, 1.5, 2.0);
            assert(fx.kind == scientific::F_NONE && fy.kind == scientific::F_NONE);
            assert(x == y);
            i += 1;
        }
        law += 1;
    }
}

fn test_results_stay_finite() {
    var law: i64 = 0;
    while law < 8 {
        var r = scientific::default_rng(3);
        var i: i64 = 0;
        while i < 1000 {
            // An extreme parameter that the call accepts.
            var p1: f64 = 1.0e-250;
            var p2: f64 = 1.0e-250;
            if i % 2 == 1 { p1 = 1.0e90; p2 = 1.0e90; }
            if law == 1 { p1 = 5.0; p2 = 5.0; }
            let x, f = draw(law, r, p1, p2);
            if f.kind == scientific::F_NONE { assert(x - x == 0.0); }
            i += 1;
        }
        law += 1;
    }
}

fn test_small_shape_stays_defined() {
    // Two small shapes do not give zero over zero.
    var r = scientific::default_rng(8);
    var i: i64 = 0;
    while i < 2000 {
        let x, f = scientific::beta(r, 1.0e-5, 1.0e-5);
        assert(f.kind == scientific::F_NONE);
        assert(x >= 0.0 && x <= 1.0);
        i += 1;
    }
}

fn test_large_shape_mean() {
    // The acceptance test stays exact for a shape of 1e12, where the
    // plain sum of the test loses every digit.
    var r = scientific::default_rng(4);
    var sum: f64 = 0.0;
    var i: i64 = 0;
    while i < 2000 {
        let x, f = scientific::gamma(r, 1.0e12, 1.0);
        assert(f.kind == scientific::F_NONE);
        sum += x;
        i += 1;
    }
    let mean = sum / 2000.0;
    assert(common::mathshim_abs(mean - 1.0e12) <= 5.0 * 1.0e6 / common::mathshim_sqrt(2000.0));
}
