// The helpers that the tests of the random laws share.
namespace common_13;

import scientific from "../..";
import common;

// law_mean_ok tells whether the mean of a sample stands near the mean
// of the law. The tolerance is five standard errors.
fn law_mean_ok(sum: f64, n: f64, m: f64, v: f64) -> bool {
    return common::mathshim_abs(sum / n - m) <= 5.0 * common::mathshim_sqrt(v / n);
}

// law_var_ok tells whether the variance of a sample stands near the
// variance of the law. Only the laws whose variance settles fast use
// it.
fn law_var_ok(sum: f64, sumsq: f64, n: f64, v: f64) -> bool {
    let mean = sum / n;
    return common::mathshim_abs(sumsq / n - mean * mean - v) <= 0.05 * v;
}

// law_quartiles_ok tells whether the counts of the numbers below the
// three quartiles stand near a quarter, a half and three quarters.
// The count never sorts the numbers.
fn law_quartiles_ok(n: f64, b1: i64, b2: i64, b3: i64) -> bool {
    let se = common::mathshim_sqrt(0.1875 / n);
    if common::mathshim_abs(b1 as f64 / n - 0.25) > 5.0 * se { return false; }
    if common::mathshim_abs(b2 as f64 / n - 0.50) > 5.0 * common::mathshim_sqrt(0.25 / n) { return false; }
    if common::mathshim_abs(b3 as f64 / n - 0.75) > 5.0 * se { return false; }
    return true;
}

// law_bad tells whether a call refuses a parameter with a kind of
// fault and gives zero.
fn law_bad(x: f64, f: scientific::Fault, kind: i64) -> bool {
    return f.kind == kind && x == 0.0;
}

// law_same_seed tells whether two generators of one seed give the same
// numbers of a law. It draws fifty pairs.
fn law_same_seed(law: i64, p1: f64, p2: f64, p3: f64) -> bool {
    var ra = scientific::default_rng(99);
    var rb = scientific::default_rng(99);
    var i: i64 = 0;
    while i < 50 {
        let x, fx = law_draw(law, ra, p1, p2, p3);
        let y, fy = law_draw(law, rb, p1, p2, p3);
        if fx.kind != scientific::F_NONE || fy.kind != scientific::F_NONE { return false; }
        if x != y { return false; }
        i += 1;
    }
    return true;
}

// The codes of the fifteen laws of the package. The codes of the two
// groups of random laws sit here, so that one table names every law
// and the dispatcher below uses it.
const LAW_GAMMA: i64 = 0;
const LAW_LOGNORMAL: i64 = 1;
const LAW_CHISQUARE: i64 = 2;
const LAW_BETA: i64 = 3;
const LAW_RAYLEIGH: i64 = 4;
const LAW_WEIBULL: i64 = 5;
const LAW_GUMBEL: i64 = 6;
const LAW_LOGISTIC: i64 = 7;
const LAW_F: i64 = 8;
const LAW_NCX2: i64 = 9;
const LAW_NCF: i64 = 10;
const LAW_PARETO: i64 = 11;
const LAW_TRI: i64 = 12;
const LAW_VM: i64 = 13;
const LAW_WALD: i64 = 14;

// law_draw gives one number of a law. A law of fewer parameters
// ignores the ones it does not take.
fn law_draw(law: i64, r: scientific::Rng&, p1: f64, p2: f64, p3: f64) -> f64, scientific::Fault {
    if law == LAW_GAMMA { return scientific::gamma(r, p1, p2); }
    if law == LAW_LOGNORMAL { return scientific::lognormal(r, p1, p2); }
    if law == LAW_CHISQUARE { return scientific::chisquare(r, p1); }
    if law == LAW_BETA { return scientific::beta(r, p1, p2); }
    if law == LAW_RAYLEIGH { return scientific::rayleigh(r, p1); }
    if law == LAW_WEIBULL { return scientific::weibull(r, p1); }
    if law == LAW_GUMBEL { return scientific::gumbel(r, p1, p2); }
    if law == LAW_LOGISTIC { return scientific::logistic(r, p1, p2); }
    if law == LAW_F { return scientific::f(r, p1, p2); }
    if law == LAW_NCX2 { return scientific::noncentral_chisquare(r, p1, p2); }
    if law == LAW_NCF { return scientific::noncentral_f(r, p1, p2, p3); }
    if law == LAW_PARETO { return scientific::pareto(r, p1); }
    if law == LAW_TRI { return scientific::triangular(r, p1, p2, p3); }
    if law == LAW_VM { return scientific::vonmises(r, p1, p2); }
    return scientific::wald(r, p1, p2);
}
