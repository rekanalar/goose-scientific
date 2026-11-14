// The tests of the laws of the F family, of the circle, of the Pareto,
// of the triangle and of the Wald form. Run them with
// SCIENTIFIC_SUITES=random_laws.
//
// The calls f, noncentral_chisquare, noncentral_f, pareto, triangular,
// vonmises and wald draw from the generator of the package. Its stream
// differs from the stream of NumPy, so the tests compare the law: the
// mean, the variance and the quartiles of 200000 numbers match the
// values of SciPy 1.18.1. The quartile test counts and never sorts.
//
// The F law, the noncentral F law and the Pareto law have a heavy
// tail. The sample variance of such a law settles slowly, so the test
// compares their mean and their quartiles only.
//
// The law of the von Mises lies on the circle. Its test takes the mean
// of the cosines and of the sines of the numbers.
namespace random_laws_tests;

import scientific from "../..";
import common;
import common_13;
import random_laws_pins;

const DRAWS: i64 = 200000;



// check_mean_and_quartiles draws many numbers of one parameter set. It
// compares the mean with the pin and the count of the numbers below
// each of the three quartiles. The count never sorts the numbers.
fn check_mean_and_quartiles(law: i64, seed: i64, p1: f64, p2: f64, p3: f64,
                             m: f64, v: f64, q: f64[:], k: i64, with_var: bool) {
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
        let x, f = common_13::law_draw(law, r, p1, p2, p3);
        assert(f.kind == scientific::F_NONE);
        sum += x;
        sumsq += x * x;
        if x < q0 { b1 += 1; }
        if x < q1 { b2 += 1; }
        if x < q2 { b3 += 1; }
        i += 1;
    }
    let n = DRAWS as f64;
    assert(common_13::law_mean_ok(sum, n, m, v));
    // The variance of a heavy tail settles too slowly to compare.
    if with_var {
        assert(common_13::law_var_ok(sum, sumsq, n, v));
    }
    assert(common_13::law_quartiles_ok(n, b1, b2, b3));
}

// check_circle draws many numbers of the von Mises law and compares
// them on the circle. The mean of the cosines of the numbers matches
// the ratio of the modified Bessel functions. The mean of the sines
// stands at zero. Every number lies between minus pi and pi.
fn check_circle(seed: i64, kappa: f64, a: f64) {
    var r = scientific::default_rng(seed);
    var csum: f64 = 0.0;
    var ssum: f64 = 0.0;
    var i: i64 = 0;
    while i < DRAWS {
        let x, f = scientific::vonmises(r, 0.0, kappa);
        assert(f.kind == scientific::F_NONE);
        assert(x >= -constants::pi && x <= constants::pi);
        csum += common::mathshim_cos(x);
        ssum += common::mathshim_sin(x);
        i += 1;
    }
    let n = DRAWS as f64;
    assert(common::mathshim_abs(csum / n - a) <= 0.05 * a);
    assert(common::mathshim_abs(ssum / n) <= 0.05 * a);
}

// check_range draws many numbers of one law and checks that every
// number lies inside the bounds. A law that leaves its range gives a
// fault, and the fault gives a failed test.
fn check_range(law: i64, seed: i64, p1: f64, p2: f64, p3: f64, lo: f64, hi: f64) {
    var r = scientific::default_rng(seed);
    var i: i64 = 0;
    while i < 5000 {
        let x, f = common_13::law_draw(law, r, p1, p2, p3);
        assert(f.kind == scientific::F_NONE);
        assert(x >= lo && x <= hi);
        i += 1;
    }
}

fn test_f_family_laws() {
    var k: i64 = 0;
    while k < 3 {
        check_mean_and_quartiles(common_13::LAW_F, 11 + k, random_laws_pins::F_P1[k], random_laws_pins::F_P2[k], 0.0,
                                 random_laws_pins::F_M[k], random_laws_pins::F_V[k],
                                 random_laws_pins::F_Q, k, false);
        check_mean_and_quartiles(common_13::LAW_NCX2, 21 + k, random_laws_pins::NCX2_P1[k], random_laws_pins::NCX2_P2[k], 0.0,
                                 random_laws_pins::NCX2_M[k], random_laws_pins::NCX2_V[k],
                                 random_laws_pins::NCX2_Q, k, true);
        check_mean_and_quartiles(common_13::LAW_NCF, 31 + k, random_laws_pins::NCF_P1[k], random_laws_pins::NCF_P2[k],
                                 random_laws_pins::NCF_P3[k],
                                 random_laws_pins::NCF_M[k], random_laws_pins::NCF_V[k],
                                 random_laws_pins::NCF_Q, k, false);
        k += 1;
    }
}

fn test_pareto_and_triangular_laws() {
    var k: i64 = 0;
    while k < 3 {
        check_mean_and_quartiles(common_13::LAW_PARETO, 61 + k, random_laws_pins::PARETO_P1[k], 0.0, 0.0,
                                 random_laws_pins::PARETO_M[k], random_laws_pins::PARETO_V[k],
                                 random_laws_pins::PARETO_Q, k, false);
        check_mean_and_quartiles(common_13::LAW_TRI, 41 + k, random_laws_pins::TRI_P1[k], random_laws_pins::TRI_P2[k],
                                 random_laws_pins::TRI_P3[k],
                                 random_laws_pins::TRI_M[k], random_laws_pins::TRI_V[k],
                                 random_laws_pins::TRI_Q, k, true);
        k += 1;
    }
}

fn test_wald_law() {
    var k: i64 = 0;
    while k < 3 {
        check_mean_and_quartiles(common_13::LAW_WALD, 51 + k, random_laws_pins::WALD_P1[k], random_laws_pins::WALD_P2[k], 0.0,
                                 random_laws_pins::WALD_M[k], random_laws_pins::WALD_V[k],
                                 random_laws_pins::WALD_Q, k, true);
        k += 1;
    }
}

fn test_von_mises_law() {
    var k: i64 = 0;
    while k < 3 {
        check_circle(71 + k, random_laws_pins::VM_P2[k], random_laws_pins::VM_A[k]);
        check_mean_and_quartiles(common_13::LAW_VM, 71 + k, random_laws_pins::VM_P1[k], random_laws_pins::VM_P2[k], 0.0,
                                 random_laws_pins::VM_M[k], random_laws_pins::VM_V[k],
                                 random_laws_pins::VM_Q, k, true);
        k += 1;
    }
}

// The Pareto law has no mean below a shape of one. The mean of many
// numbers of a shape of one stands above the mean of a shape of two.
fn test_pareto_shape_of_one() {
    var r = scientific::default_rng(91);
    var sum: f64 = 0.0;
    var i: i64 = 0;
    while i < 20000 {
        let x, f = scientific::pareto(r, 1.0);
        assert(f.kind == scientific::F_NONE && x >= 0.0);
        sum += x;
        i += 1;
    }
    let mean_one = sum / 20000.0;
    var r2 = scientific::default_rng(92);
    sum = 0.0;
    i = 0;
    while i < 20000 {
        let x, f = scientific::pareto(r2, 2.0);
        assert(f.kind == scientific::F_NONE);
        sum += x;
        i += 1;
    }
    let mean_two = sum / 20000.0;
    assert(mean_one > mean_two);
}

// bad checks that a call refuses a parameter with a kind of fault.
fn bad(law: i64, p1: f64, p2: f64, p3: f64, kind: i64) {
    var r = scientific::default_rng(5);
    let x, f = common_13::law_draw(law, r, p1, p2, p3);
    assert(f.kind == kind);
    assert(x == 0.0);
}

fn test_laws_bad_parameters_refused() {
    let nz = -1.0 * 0.0;
    let nan = 0.0 / 0.0;
    let inf = 1.0 / 0.0;
    let v = scientific::F_VALUE;
    // The counts of the degrees of freedom are above zero.
    bad(common_13::LAW_F, 0.0, 5.0, 0.0, v); bad(common_13::LAW_F, 5.0, 0.0, 0.0, v);
    bad(common_13::LAW_F, nz, 5.0, 0.0, v);  bad(common_13::LAW_F, 5.0, nz, 0.0, v);
    bad(common_13::LAW_F, -1.0, 5.0, 0.0, v); bad(common_13::LAW_F, 5.0, -1.0, 0.0, v);
    bad(common_13::LAW_F, nan, 5.0, 0.0, v); bad(common_13::LAW_F, 5.0, nan, 0.0, v);
    bad(common_13::LAW_F, inf, 5.0, 0.0, v); bad(common_13::LAW_F, 5.0, inf, 0.0, v);
    bad(common_13::LAW_NCX2, 0.0, 2.0, 0.0, v); bad(common_13::LAW_NCX2, nz, 2.0, 0.0, v);
    bad(common_13::LAW_NCX2, -1.0, 2.0, 0.0, v); bad(common_13::LAW_NCX2, nan, 2.0, 0.0, v);
    bad(common_13::LAW_NCX2, 5.0, nan, 0.0, v); bad(common_13::LAW_NCX2, 5.0, inf, 0.0, v);
    // The noncentrality is at or above zero. A negative zero is refused.
    bad(common_13::LAW_NCX2, 5.0, -1.0, 0.0, v); bad(common_13::LAW_NCX2, 5.0, nz, 0.0, v);
    bad(common_13::LAW_NCF, 0.0, 5.0, 2.0, v); bad(common_13::LAW_NCF, 5.0, 0.0, 2.0, v);
    bad(common_13::LAW_NCF, nz, 5.0, 2.0, v);  bad(common_13::LAW_NCF, 5.0, nz, 2.0, v);
    bad(common_13::LAW_NCF, 5.0, 5.0, -1.0, v); bad(common_13::LAW_NCF, 5.0, 5.0, nz, v);
    bad(common_13::LAW_NCF, nan, 5.0, 2.0, v); bad(common_13::LAW_NCF, 5.0, nan, 2.0, v);
    bad(common_13::LAW_NCF, 5.0, 5.0, nan, v); bad(common_13::LAW_NCF, 5.0, 5.0, inf, v);
    // The shape of the Pareto law is above zero.
    bad(common_13::LAW_PARETO, 0.0, 0.0, 0.0, v); bad(common_13::LAW_PARETO, nz, 0.0, 0.0, v);
    bad(common_13::LAW_PARETO, -1.0, 0.0, 0.0, v); bad(common_13::LAW_PARETO, nan, 0.0, 0.0, v);
    bad(common_13::LAW_PARETO, inf, 0.0, 0.0, v);
    // The ends of the triangular law stand in order and differ.
    bad(common_13::LAW_TRI, 1.0, 0.0, 3.0, v); bad(common_13::LAW_TRI, 0.0, 4.0, 3.0, v);
    bad(common_13::LAW_TRI, 1.0, 1.0, 1.0, v);
    bad(common_13::LAW_TRI, nan, 1.0, 3.0, v); bad(common_13::LAW_TRI, 0.0, nan, 3.0, v);
    bad(common_13::LAW_TRI, 0.0, 1.0, nan, v); bad(common_13::LAW_TRI, inf, 1.0, 3.0, v);
    // The concentration of the von Mises law is at or above zero.
    bad(common_13::LAW_VM, 0.0, -1.0, 0.0, v); bad(common_13::LAW_VM, 0.0, nz, 0.0, v);
    bad(common_13::LAW_VM, 0.0, nan, 0.0, v); bad(common_13::LAW_VM, 0.0, inf, 0.0, v);
    bad(common_13::LAW_VM, nan, 1.0, 0.0, v); bad(common_13::LAW_VM, inf, 1.0, 0.0, v);
    // The mean and the scale of the Wald law are above zero.
    bad(common_13::LAW_WALD, 0.0, 1.0, 0.0, v); bad(common_13::LAW_WALD, nz, 1.0, 0.0, v);
    bad(common_13::LAW_WALD, -1.0, 1.0, 0.0, v); bad(common_13::LAW_WALD, 1.0, 0.0, 0.0, v);
    bad(common_13::LAW_WALD, 1.0, nz, 0.0, v);
    bad(common_13::LAW_WALD, 1.0, -1.0, 0.0, v);
    bad(common_13::LAW_WALD, nan, 1.0, 0.0, v); bad(common_13::LAW_WALD, 1.0, nan, 0.0, v);
    bad(common_13::LAW_WALD, inf, 1.0, 0.0, v); bad(common_13::LAW_WALD, 1.0, inf, 0.0, v);
}

fn test_laws_size_limits() {
    let lim = scientific::F_LIMIT;
    bad(common_13::LAW_F, 1.0e201, 5.0, 0.0, lim);
    bad(common_13::LAW_F, 5.0, 1.0e201, 0.0, lim);
    bad(common_13::LAW_F, 1.0e-301, 5.0, 0.0, lim);
    bad(common_13::LAW_NCX2, 1.0e201, 2.0, 0.0, lim);
    bad(common_13::LAW_NCX2, 1.0e-301, 2.0, 0.0, lim);
    bad(common_13::LAW_NCF, 5.0, 1.0e201, 2.0, lim);
    bad(common_13::LAW_NCF, 5.0, 1.0e-301, 2.0, lim);
    bad(common_13::LAW_PARETO, 1.0e101, 0.0, 0.0, lim);
    bad(common_13::LAW_PARETO, 1.0e-301, 0.0, 0.0, lim);
    // A count of degrees of freedom below one with a large
    // noncentrality stops, because the rate of the Poisson law stops
    // above 30.
    bad(common_13::LAW_NCX2, 0.5, 100.0, 0.0, lim);
    bad(common_13::LAW_NCF, 0.5, 5.0, 100.0, lim);
}

// A concentration of the von Mises law above 1e6 gives the normal
// form of NumPy. The number lies between minus pi and pi.
fn test_von_mises_large_concentration() {
    check_range(common_13::LAW_VM, 101, 0.0, 1.0e7, 0.0, -constants::pi, constants::pi);
}

// A concentration of zero gives a uniform number of the circle.
fn test_von_mises_zero_concentration() {
    check_range(common_13::LAW_VM, 102, 0.0, 0.0, 0.0, -constants::pi, constants::pi);
}

// A location that stands away from zero gives a number that still lies
// between minus pi and pi. The circle wraps.
fn test_von_mises_location_away_from_zero() {
    check_range(common_13::LAW_VM, 103, 2.5, 1.5, 0.0, -constants::pi, constants::pi);
}

// The number of the triangular law lies between the left and the
// right.
fn test_triangular_number_stays_in_the_range() {
    check_range(common_13::LAW_TRI, 104, -2.0, 0.0, 3.0, -2.0, 3.0);
}

// The number of the Wald law stands above zero.
fn test_wald_number_stays_above_zero() {
    var r = scientific::default_rng(105);
    var i: i64 = 0;
    while i < 5000 {
        let x, f = scientific::wald(r, 2.0, 3.0);
        assert(f.kind == scientific::F_NONE);
        assert(x > 0.0);
        i += 1;
    }
}

fn test_laws_results_stay_finite() {
    var law: i64 = common_13::LAW_F;
    while law <= common_13::LAW_WALD {
        var r = scientific::default_rng(3);
        var i: i64 = 0;
        while i < 1000 {
            var p1: f64 = 1.0e-250;
            var p2: f64 = 1.0e-250;
            var p3: f64 = 1.0e-250;
            if i % 2 == 1 { p1 = 1.0e90; p2 = 1.0e90; p3 = 1.0e90; }
            if law == common_13::LAW_TRI { p1 = -1.0e90; p2 = 0.0; p3 = 1.0e90; }
            if law == common_13::LAW_VM { p1 = 0.0; p2 = 1.0e-5; }
            let x, f = common_13::law_draw(law, r, p1, p2, p3);
            if f.kind == scientific::F_NONE { assert(x - x == 0.0); }
            i += 1;
        }
        law += 1;
    }
}

fn test_laws_equal_seeds_equal_numbers() {
    var law: i64 = common_13::LAW_F;
    while law <= common_13::LAW_WALD {
        assert(common_13::law_same_seed(law, 1.5, 2.0, 3.0));
        law += 1;
    }
}

fn test_laws_pinned_numbers_of_one_seed() {
    // The numbers of one seed are pinned. The same seed gives the
    // same numbers on every run and on both supported versions.
    var r = scientific::default_rng(4242);
    let a1, f1 = scientific::f(r, 5.0, 8.0);
    assert(f1.kind == scientific::F_NONE && a1 == 0.18110241660775597);
    let a2, f2 = scientific::noncentral_chisquare(r, 5.0, 2.0);
    assert(f2.kind == scientific::F_NONE && a2 == 7.514777973958559);
    let a3, f3 = scientific::noncentral_f(r, 5.0, 8.0, 3.0);
    assert(f3.kind == scientific::F_NONE && a3 == 0.42218338191757426);
    let a4, f4 = scientific::pareto(r, 6.0);
    assert(f4.kind == scientific::F_NONE && a4 == 0.04420176084536842);
    let a5, f5 = scientific::triangular(r, -2.0, 0.0, 3.0);
    assert(f5.kind == scientific::F_NONE && a5 == 0.6944645641768181);
    let a6, f6 = scientific::vonmises(r, 0.0, 1.5);
    assert(f6.kind == scientific::F_NONE && a6 == 0.39715315604708623);
    let a7, f7 = scientific::wald(r, 2.0, 3.0);
    assert(f7.kind == scientific::F_NONE && a7 == 0.7336328923767664);
}

// The security of the new laws. The sampler of the Wald law loses no
// number to a cancellation, so a small scale gives no fault. The
// triangular law takes the roots apart, so a wide range gives a number
// and not a fault. A range wider than the largest number gives a fault
// of the kind of the limit.
fn test_laws_survive_a_hostile_parameter() {
    var r = scientific::default_rng(77);
    // A small scale of the Wald law gives no fault.
    var i: i64 = 0;
    while i < 2000 {
        let x, f = scientific::wald(r, 2.0, 1.0e-9);
        assert(f.kind == scientific::F_NONE);
        assert(x > 0.0);
        i += 1;
    }
    // A wide range of the triangular law gives a number.
    var j: i64 = 0;
    while j < 200 {
        let x, f = scientific::triangular(r, -1.0e200, 0.0, 1.0e200);
        assert(f.kind == scientific::F_NONE);
        assert(x >= -1.0e200 && x <= 1.0e200);
        j += 1;
    }
    // A range wider than the largest number gives a fault of the kind
    // of the limit.
    let w, fw = scientific::triangular(r, -1.0e308, 0.0, 1.0e308);
    assert(fw.kind == scientific::F_LIMIT && w == 0.0);
    // A mean of the Wald law above the limit of the shape gives a fault
    // of the kind of the limit.
    let m, fmm = scientific::wald(r, 1.0e200, 1.0);
    assert(fmm.kind == scientific::F_LIMIT && m == 0.0);
    // A concentration of the von Mises law below zero gives a fault of
    // the kind of the value.
    let c, fc = scientific::vonmises(r, 0.0, -5.0);
    assert(fc.kind == scientific::F_VALUE && c == 0.0);
}
