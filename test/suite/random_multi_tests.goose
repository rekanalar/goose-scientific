// The tests of the random laws of a vector. Run them with
// SCIENTIFIC_SUITES=random_multi.
//
// The calls dirichlet and multivariate_normal draw from the generator
// of the package. Its stream differs from the stream of NumPy, so the
// tests compare the law: the mean of the Dirichlet law and the mean
// and the covariance of the normal law of a vector match the values of
// NumPy 2.5.4.
//
// The covariance of a sample comes from the sum of the products of the
// deviations of the two values. The test never sorts the numbers.
namespace random_multi_tests;

import scientific from "../..";
import common;
import random_multi_pins;

const M_DRAWS: i64 = 200000;

// dirichlet_mean draws many points of the law of a row of
// concentrations. Each point sums to one, and the mean of each value
// stands near the value that the row of want holds.
fn dirichlet_mean(seed: i64, alpha_values: f64[:], want: f64[:]) {
    var r = scientific::default_rng(seed);
    let alpha, fa = scientific::from_values(alpha_values);
    assert(fa.kind == scientific::F_NONE);
    let n = want.len;
    var sums: f64[>..] = [];
    var i: i64 = 0;
    while i < n { sums.push(0.0); i += 1; }
    var d: i64 = 0;
    while d < M_DRAWS {
        let p, fp = scientific::dirichlet(r, alpha);
        assert(fp.kind == scientific::F_NONE);
        assert(p.data.len == n);
        var total: f64 = 0.0;
        var k: i64 = 0;
        while k < n { sums[k] += p.data[k]; total += p.data[k]; k += 1; }
        assert(common::close(total, 1.0, 1.0e-9));
        d += 1;
    }
    var k2: i64 = 0;
    while k2 < n {
        assert(common::close(sums[k2] / M_DRAWS, want[k2], 0.01));
        k2 += 1;
    }
}

fn test_dirichlet_gives_the_mean_of_the_law() {
    dirichlet_mean(21, random_multi_pins::DIR_A1, random_multi_pins::DIR_M1);
}

fn test_dirichlet_gives_the_mean_of_the_second_law() {
    dirichlet_mean(22, random_multi_pins::DIR_A2, random_multi_pins::DIR_M2);
}

fn test_dirichlet_refuses_a_concentration_of_zero() {
    var r = scientific::default_rng(23);
    let alpha, fa = scientific::from_values([0.0, 1.0, 2.0]);
    assert(fa.kind == scientific::F_NONE);
    let p, fp = scientific::dirichlet(r, alpha);
    assert(fp.kind == scientific::F_VALUE && p.data.len == 0);
    let neg, fa2 = scientific::from_values([-1.0, 2.0]);
    assert(fa2.kind == scientific::F_NONE);
    let p2, fp2 = scientific::dirichlet(r, neg);
    assert(fp2.kind == scientific::F_VALUE && p2.data.len == 0);
}

// mvn_moments draws many points of the normal law of a vector of n
// values. The mean of each coordinate stands near the mean of the
// law. The covariance of the sample stands near the covariance of the
// law.
fn mvn_moments(seed: i64, n: i64, mean_values: f64[:], cov_values: f64[:], tol_mean: f64, tol_cov: f64) {
    var r = scientific::default_rng(seed);
    let mean, fm = scientific::from_values(mean_values);
    assert(fm.kind == scientific::F_NONE);
    let cov, fc = scientific::from_rows(n, n, cov_values);
    assert(fc.kind == scientific::F_NONE);
    var sums: f64[>..] = [];
    var cross: f64[>..] = [];
    var i: i64 = 0;
    while i < n { sums.push(0.0); i += 1; }
    i = 0;
    while i < n * n { cross.push(0.0); i += 1; }
    var d: i64 = 0;
    while d < M_DRAWS {
        let x, fx = scientific::multivariate_normal(r, mean, cov);
        assert(fx.kind == scientific::F_NONE);
        assert(x.data.len == n);
        var a: i64 = 0;
        while a < n {
            sums[a] += x.data[a];
            var b: i64 = 0;
            while b < n {
                cross[a * n + b] += (x.data[a] - mean_values[a]) * (x.data[b] - mean_values[b]);
                b += 1;
            }
            a += 1;
        }
        d += 1;
    }
    var k: i64 = 0;
    while k < n {
        assert(common::close(sums[k] / M_DRAWS, mean_values[k], tol_mean));
        k += 1;
    }
    var c: i64 = 0;
    while c < n * n {
        assert(common::close(cross[c] / M_DRAWS, cov_values[c], tol_cov));
        c += 1;
    }
}

fn test_multivariate_normal_gives_the_mean_and_the_covariance() {
    mvn_moments(31, 2, random_multi_pins::MVN_M1, random_multi_pins::MVN_C1, 0.02, 0.05);
}

fn test_multivariate_normal_gives_the_moments_of_a_three_by_three() {
    mvn_moments(32, 3, random_multi_pins::MVN_M2, random_multi_pins::MVN_C2, 0.03, 0.06);
}

fn test_multivariate_normal_refuses_a_bad_covariance() {
    var r = scientific::default_rng(33);
    let mean, fm = scientific::from_values([0.0, 0.0]);
    assert(fm.kind == scientific::F_NONE);
    // A covariance that is not symmetric.
    let asym, fa = scientific::from_rows(2, 2, [1.0, 0.5, 0.2, 1.0]);
    assert(fa.kind == scientific::F_NONE);
    let x, fx = scientific::multivariate_normal(r, mean, asym);
    assert(fx.kind == scientific::F_VALUE && x.data.len == 0);
    // A covariance that is not positive semidefinite.
    let neg, fa2 = scientific::from_rows(2, 2, [1.0, 2.0, 2.0, 1.0]);
    assert(fa2.kind == scientific::F_NONE);
    let x2, fx2 = scientific::multivariate_normal(r, mean, neg);
    assert(fx2.kind == scientific::F_VALUE && x2.data.len == 0);
    // A mean and a covariance of a different length.
    let short, fa3 = scientific::from_values([0.0]);
    assert(fa3.kind == scientific::F_NONE);
    let x3, fx3 = scientific::multivariate_normal(r, short, asym);
    assert(fx3.kind == scientific::F_SHAPE && x3.data.len == 0);
}

fn test_the_same_seed_gives_the_same_points() {
    var ra = scientific::default_rng(41);
    var rb = scientific::default_rng(41);
    let alpha, fa = scientific::from_values(random_multi_pins::DIR_A1);
    assert(fa.kind == scientific::F_NONE);
    let mean, fm = scientific::from_values(random_multi_pins::MVN_M3);
    assert(fm.kind == scientific::F_NONE);
    let cov, fc = scientific::from_rows(1, 1, random_multi_pins::MVN_C3);
    assert(fc.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < 20 {
        let p, fp = scientific::dirichlet(ra, alpha);
        let q, fq = scientific::dirichlet(rb, alpha);
        assert(fp.kind == scientific::F_NONE && fq.kind == scientific::F_NONE);
        var k: i64 = 0;
        while k < 3 {
            assert(p.data[k] == q.data[k]);
            k += 1;
        }
        let x, fx = scientific::multivariate_normal(ra, mean, cov);
        let y, fy = scientific::multivariate_normal(rb, mean, cov);
        assert(fx.kind == scientific::F_NONE && fy.kind == scientific::F_NONE);
        assert(x.data[0] == y.data[0]);
        i += 1;
    }
}
