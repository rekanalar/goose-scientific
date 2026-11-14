// The tests of the discrete random laws. Run them with
// SCIENTIFIC_SUITES=random_discrete.
//
// The calls binomial, geometric, hypergeometric, logseries,
// negative_binomial, zipf and multinomial draw from the generator of
// the package. Its stream differs from the stream of NumPy, so the
// tests compare the law: the mean, the variance and the value of the
// distribution function of 200000 numbers match SciPy 1.18.1. The
// count at or below a threshold never sorts the numbers.
//
// The sample variance of the zipf law settles slowly, so the test
// compares its mean and its distribution function only.
namespace random_discrete_tests;

import scientific from "../..";
import common;
import random_discrete_pins;
import common_13;

const DRAWS: i64 = 200000;

const D_BIN: i64 = 0;
const D_GEO: i64 = 1;
const D_HYP: i64 = 2;
const D_LOG: i64 = 3;
const D_NBIN: i64 = 4;
const D_ZIPF: i64 = 5;

// disc_draw gives one number of a law as a real number. A law of fewer
// parameters ignores the ones it does not take.
fn disc_draw(law: i64, r: scientific::Rng&, p1: f64, p2: f64, p3: f64) -> f64, scientific::Fault {
    if law == D_BIN {
        let x, f = scientific::binomial(r, p1 as i64, p2);
        return x as f64, f;
    }
    if law == D_GEO {
        let x, f = scientific::geometric(r, p1);
        return x as f64, f;
    }
    if law == D_HYP {
        let x, f = scientific::hypergeometric(r, p1 as i64, p2 as i64, p3 as i64);
        return x as f64, f;
    }
    if law == D_LOG {
        let x, f = scientific::logseries(r, p1);
        return x as f64, f;
    }
    if law == D_NBIN {
        let x, f = scientific::negative_binomial(r, p1, p2);
        return x as f64, f;
    }
    let x, f = scientific::zipf(r, p1);
    return x as f64, f;
}

// check_set draws many numbers of one parameter set. It compares the
// mean with the pin, the variance when the flag is set, and the share
// of the numbers at or below each of the three thresholds.
fn check_set(law: i64, seed: i64, p1: f64, p2: f64, p3: f64,
             m: f64, v: f64, t: f64[:], c: f64[:], k: i64, with_var: bool) {
    var r = scientific::default_rng(seed);
    var sum: f64 = 0.0;
    var sumsq: f64 = 0.0;
    var b0: i64 = 0;
    var b1: i64 = 0;
    var b2: i64 = 0;
    var i: i64 = 0;
    while i < DRAWS {
        let x, f = disc_draw(law, r, p1, p2, p3);
        assert(f.kind == scientific::F_NONE);
        sum += x;
        sumsq += x * x;
        if x <= t[3 * k] { b0 += 1; }
        if x <= t[3 * k + 1] { b1 += 1; }
        if x <= t[3 * k + 2] { b2 += 1; }
        i += 1;
    }
    let n = DRAWS as f64;
    assert(common_13::law_mean_ok(sum, n, m, v));
    if with_var {
        assert(common_13::law_var_ok(sum, sumsq, n, v));
    }
    assert(share_ok(b0, n, c[3 * k]));
    assert(share_ok(b1, n, c[3 * k + 1]));
    assert(share_ok(b2, n, c[3 * k + 2]));
}

// share_ok tells whether the share of a count stands near the value of
// the distribution function. The tolerance is five standard errors.
fn share_ok(count: i64, n: f64, c: f64) -> bool {
    let se = common::mathshim_sqrt(c * (1.0 - c) / n);
    return common::mathshim_abs(count as f64 / n - c) <= 5.0 * se + 1.0e-9;
}

fn test_binomial_law() {
    var k: i64 = 0;
    while k < 3 {
        check_set(D_BIN, 11 + k, random_discrete_pins::BIN_P1[k], random_discrete_pins::BIN_P2[k], 0.0,
                  random_discrete_pins::BIN_M[k], random_discrete_pins::BIN_V[k],
                  random_discrete_pins::BIN_T, random_discrete_pins::BIN_C, k, true);
        k += 1;
    }
}

fn test_geometric_law() {
    var k: i64 = 0;
    while k < 3 {
        check_set(D_GEO, 21 + k, random_discrete_pins::GEO_P1[k], 0.0, 0.0,
                  random_discrete_pins::GEO_M[k], random_discrete_pins::GEO_V[k],
                  random_discrete_pins::GEO_T, random_discrete_pins::GEO_C, k, true);
        k += 1;
    }
}

fn test_hypergeometric_law() {
    var k: i64 = 0;
    while k < 3 {
        check_set(D_HYP, 31 + k, random_discrete_pins::HYP_P1[k], random_discrete_pins::HYP_P2[k],
                  random_discrete_pins::HYP_P3[k],
                  random_discrete_pins::HYP_M[k], random_discrete_pins::HYP_V[k],
                  random_discrete_pins::HYP_T, random_discrete_pins::HYP_C, k, true);
        k += 1;
    }
}

fn test_logseries_law() {
    var k: i64 = 0;
    while k < 3 {
        check_set(D_LOG, 41 + k, random_discrete_pins::LOG_P1[k], 0.0, 0.0,
                  random_discrete_pins::LOG_M[k], random_discrete_pins::LOG_V[k],
                  random_discrete_pins::LOG_T, random_discrete_pins::LOG_C, k, true);
        k += 1;
    }
}

fn test_negative_binomial_law() {
    var k: i64 = 0;
    while k < 3 {
        check_set(D_NBIN, 51 + k, random_discrete_pins::NBIN_P1[k], random_discrete_pins::NBIN_P2[k], 0.0,
                  random_discrete_pins::NBIN_M[k], random_discrete_pins::NBIN_V[k],
                  random_discrete_pins::NBIN_T, random_discrete_pins::NBIN_C, k, true);
        k += 1;
    }
}

fn test_zipf_law() {
    var k: i64 = 0;
    while k < 3 {
        check_set(D_ZIPF, 61 + k, random_discrete_pins::ZIPF_P1[k], 0.0, 0.0,
                  random_discrete_pins::ZIPF_M[k], random_discrete_pins::ZIPF_V[k],
                  random_discrete_pins::ZIPF_T, random_discrete_pins::ZIPF_C, k, false);
        k += 1;
    }
}

// The counts of the multinomial law sum to the count of trials. Each
// class has the mean n*p and the variance n*p*(1-p).
fn test_multinomial_law() {
    var r = scientific::default_rng(71);
    let pv, f_pv = scientific::from_values([0.2, 0.3, 0.5]);
    var s0: f64 = 0.0;
    var s0sq: f64 = 0.0;
    var s1: f64 = 0.0;
    var s2: f64 = 0.0;
    var i: i64 = 0;
    let draws = 20000;
    while i < draws {
        let c, f = scientific::multinomial(r, 100, pv);
        assert(f.kind == scientific::F_NONE);
        assert(c.data.len == 3);
        assert(c.data[0] + c.data[1] + c.data[2] == 100.0);
        s0 += c.data[0];
        s0sq += c.data[0] * c.data[0];
        s1 += c.data[1];
        s2 += c.data[2];
        i += 1;
    }
    let n = draws as f64;
    assert(common_13::law_mean_ok(s0, n, 20.0, 16.0));
    assert(common_13::law_mean_ok(s1, n, 30.0, 21.0));
    assert(common_13::law_mean_ok(s2, n, 50.0, 25.0));
    assert(common_13::law_var_ok(s0, s0sq, n, 16.0));
}

// bad tells whether a call refuses a parameter with a kind of fault.
fn bad(law: i64, p1: f64, p2: f64, p3: f64, kind: i64) {
    var r = scientific::default_rng(5);
    let x, f = disc_draw(law, r, p1, p2, p3);
    assert(common_13::law_bad(x, f, kind));
}

fn test_discrete_bad_parameters_refused() {
    let nan = 0.0 / 0.0;
    let inf = 1.0 / 0.0;
    let v = scientific::F_VALUE;
    bad(D_BIN, -1.0, 0.5, 0.0, v); bad(D_BIN, 10.0, -0.1, 0.0, v);
    bad(D_BIN, 10.0, 1.1, 0.0, v); bad(D_BIN, 10.0, nan, 0.0, v);
    bad(D_BIN, 10.0, inf, 0.0, v);
    bad(D_GEO, 0.0, 0.0, 0.0, v); bad(D_GEO, -0.5, 0.0, 0.0, v);
    bad(D_GEO, 1.5, 0.0, 0.0, v); bad(D_GEO, nan, 0.0, 0.0, v);
    bad(D_HYP, -1.0, 5.0, 2.0, v); bad(D_HYP, 5.0, -1.0, 2.0, v);
    bad(D_HYP, 5.0, 5.0, -1.0, v); bad(D_HYP, 5.0, 5.0, 11.0, v);
    bad(D_LOG, -0.1, 0.0, 0.0, v); bad(D_LOG, 1.0, 0.0, 0.0, v);
    bad(D_LOG, nan, 0.0, 0.0, v); bad(D_LOG, inf, 0.0, 0.0, v);
    bad(D_NBIN, 0.0, 0.5, 0.0, v); bad(D_NBIN, -1.0, 0.5, 0.0, v);
    bad(D_NBIN, 5.0, 0.0, 0.0, v); bad(D_NBIN, 5.0, 1.5, 0.0, v);
    bad(D_NBIN, nan, 0.5, 0.0, v); bad(D_NBIN, 5.0, nan, 0.0, v);
    bad(D_ZIPF, 1.0, 0.0, 0.0, v); bad(D_ZIPF, 0.5, 0.0, 0.0, v);
    bad(D_ZIPF, nan, 0.0, 0.0, v); bad(D_ZIPF, inf, 0.0, 0.0, v);
}

fn test_discrete_size_limits() {
    let lim = scientific::F_LIMIT;
    // The cost of the binomial call grows with the mean of the smaller
    // side. A mean above 1.0e6 gives a fault.
    bad(D_BIN, 1.0e10, 0.5, 0.0, lim);
    // The urn and the sample have limits.
    bad(D_HYP, 2.0e15, 5.0, 2.0, lim);
    bad(D_HYP, 1.0e12, 1.0e12, 1.0e12, lim);
    // The Poisson sum of the negative binomial law stops at 1.0e6.
    bad(D_NBIN, 1.0, 1.0e-9, 0.0, lim);
    // A shape so near one that the method rejects 1000 rounds.
    bad(D_ZIPF, 1.0000001, 0.0, 0.0, lim);
    // The count of trials of the multinomial law has a limit too.
    var r = scientific::default_rng(5);
    let pv, f_pv = scientific::from_values([0.5, 0.5]);
    let c, f = scientific::multinomial(r, 1000000000000000000, pv);
    assert(f.kind == lim && c.data.len == 0);
}

// The edges of each law hold exact values.
fn test_discrete_edges() {
    var r = scientific::default_rng(6);
    let a, fa = scientific::binomial(r, 50, 1.0);
    assert(fa.kind == scientific::F_NONE && a == 50);
    let b, fb = scientific::binomial(r, 50, 0.0);
    assert(fb.kind == scientific::F_NONE && b == 0);
    let b2, fb2 = scientific::binomial(r, 0, 0.5);
    assert(fb2.kind == scientific::F_NONE && b2 == 0);
    let g, fg = scientific::geometric(r, 1.0);
    assert(fg.kind == scientific::F_NONE && g == 1);
    let h, fh = scientific::hypergeometric(r, 5, 7, 0);
    assert(fh.kind == scientific::F_NONE && h == 0);
    let h2, fh2 = scientific::hypergeometric(r, 5, 7, 12);
    assert(fh2.kind == scientific::F_NONE && h2 == 5);
    let l, fl = scientific::logseries(r, 0.0);
    assert(fl.kind == scientific::F_NONE && l == 1);
    let nb, fnb = scientific::negative_binomial(r, 3.0, 1.0);
    assert(fnb.kind == scientific::F_NONE && nb == 0);
    let z, fz = scientific::zipf(r, 2000.0);
    assert(fz.kind == scientific::F_NONE && z == 1);
}

// A binomial number lies between zero and the count of trials. A
// number of the hypergeometric law lies between the bounds of the urn.
// A number of the geometric, logseries and zipf laws stands at or above
// one. A very small probability of the geometric law gives a number or
// a fault of the kind of the limit, never a number below one.
fn test_discrete_numbers_stay_in_range() {
    var r = scientific::default_rng(81);
    var i: i64 = 0;
    while i < 5000 {
        let a, fa = scientific::binomial(r, 30, 0.7);
        assert(fa.kind == scientific::F_NONE && a >= 0 && a <= 30);
        let h, fh = scientific::hypergeometric(r, 8, 6, 10);
        assert(fh.kind == scientific::F_NONE && h >= 4 && h <= 8);
        let g, fg = scientific::geometric(r, 0.001);
        assert(fg.kind == scientific::F_NONE && g >= 1);
        let l, fl = scientific::logseries(r, 0.99);
        assert(fl.kind == scientific::F_NONE && l >= 1);
        let z, fz = scientific::zipf(r, 3.0);
        assert(fz.kind == scientific::F_NONE && z >= 1);
        i += 1;
    }
    let t, ft = scientific::geometric(r, 1.0e-300);
    if ft.kind == scientific::F_NONE { assert(t >= 1); } else { assert(ft.kind == scientific::F_LIMIT && t == 0); }
}

fn test_multinomial_refuses_bad_input() {
    var r = scientific::default_rng(7);
    let v = scientific::F_VALUE;
    let ok, f_ok = scientific::from_values([0.5, 0.5]);
    let neg, f1 = scientific::multinomial(r, -1, ok);
    assert(f1.kind == v && neg.data.len == 0);
    let big, f_big = scientific::from_values([0.8, 0.8, 0.1]);
    let m2, f2 = scientific::multinomial(r, 10, big);
    assert(f2.kind == v && m2.data.len == 0);
    let outside, f_outside = scientific::from_values([1.5, -0.5]);
    let m3, f3 = scientific::multinomial(r, 10, outside);
    assert(f3.kind == v && m3.data.len == 0);
    let nanrow, f_nanrow = scientific::from_values([0.0 / 0.0, 0.5]);
    let m4, f4 = scientific::multinomial(r, 10, nanrow);
    assert(f4.kind == v && m4.data.len == 0);
    let empty, f_empty = scientific::from_values([]);
    let m5, f5 = scientific::multinomial(r, 10, empty);
    assert(f5.kind == scientific::F_SHAPE && m5.data.len == 0);
    let m, f_m = scientific::from_rows(2, 2, [0.25, 0.25, 0.25, 0.25]);
    let m6, f6 = scientific::multinomial(r, 10, m);
    assert(f6.kind == scientific::F_SHAPE && m6.data.len == 0);
    // Zero trials give a row of zeros.
    let z, f7 = scientific::multinomial(r, 0, ok);
    assert(f7.kind == scientific::F_NONE && z.data.len == 2 && z.data[0] == 0.0 && z.data[1] == 0.0);
    // A class of probability zero never gets a count.
    let zero, f_zero = scientific::from_values([0.0, 1.0]);
    let m8, f8 = scientific::multinomial(r, 25, zero);
    assert(f8.kind == scientific::F_NONE && m8.data[0] == 0.0 && m8.data[1] == 25.0);
}

fn test_discrete_equal_seeds_equal_numbers() {
    var law: i64 = D_BIN;
    while law <= D_ZIPF {
        var ra = scientific::default_rng(99);
        var rb = scientific::default_rng(99);
        var i: i64 = 0;
        while i < 50 {
            let x, fx = disc_draw(law, ra, 20.0, 0.4, 5.0);
            let y, fy = disc_draw(law, rb, 20.0, 0.4, 5.0);
            assert(fx.kind == fy.kind);
            assert(x == y);
            i += 1;
        }
        law += 1;
    }
}

fn test_discrete_pinned_numbers_of_one_seed() {
    // The numbers of one seed are pinned. The same seed gives the
    // same numbers on every run and on both supported versions.
    var r = scientific::default_rng(4242);
    let a, fa = scientific::binomial(r, 20, 0.3);
    assert(fa.kind == scientific::F_NONE && a == 5);
    let b, fb = scientific::geometric(r, 0.2);
    assert(fb.kind == scientific::F_NONE && b == 7);
    let c, fc = scientific::hypergeometric(r, 10, 20, 12);
    assert(fc.kind == scientific::F_NONE && c == 3);
    let d, fd = scientific::logseries(r, 0.6);
    assert(fd.kind == scientific::F_NONE && d == 5);
    let e, fe = scientific::negative_binomial(r, 5.0, 0.4);
    assert(fe.kind == scientific::F_NONE && e == 6);
    let z, fz = scientific::zipf(r, 4.0);
    assert(fz.kind == scientific::F_NONE && z == 1);
    let pv, fpv = scientific::from_values([0.2, 0.3, 0.5]);
    let m, fm = scientific::multinomial(r, 30, pv);
    assert(fm.kind == scientific::F_NONE);
    assert(m.data[0] == 7.0 && m.data[1] == 11.0 && m.data[2] == 12.0);
}

// The cost cap of the binomial call holds for each class. The whole
// call has one budget too. A long row of classes that each pass the
// cap gives a fault of the kind of the limit and not a long run.
fn test_multinomial_budget_covers_all_classes() {
    var r = scientific::default_rng(8);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 1000 { d.push(0.001); i += 1; }
    let row, frow = scientific::from_values(d);
    let m, f = scientific::multinomial(r, 900000000, row);
    assert(f.kind == scientific::F_LIMIT && m.data.len == 0);
    // A small count over the same row stays inside the budget.
    let m2, f2 = scientific::multinomial(r, 1000, row);
    assert(f2.kind == scientific::F_NONE && m2.data.len == 1000);
    var total: f64 = 0.0;
    i = 0;
    while i < 1000 { total += m2.data[i]; i += 1; }
    assert(total == 1000.0);
}
