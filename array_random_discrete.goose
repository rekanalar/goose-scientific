// The discrete random laws.
//
// The calls binomial, geometric, hypergeometric, logseries,
// negative_binomial, zipf and multinomial follow numpy.random.Generator.
// Each call takes the generator by reference and gives a whole number
// and a fault. The multinomial call gives an array of counts. The stream
// of the generator differs from the stream of NumPy. The law is the
// same: the mean, the variance and the distribution function agree.
//
// The package holds no function for the logarithm of the gamma
// function that a sampler may use here. So the samplers use methods
// that need none: the gaps of a geometric law, the urn, the method of
// Kemp and the method of Devroye. Each method has a cost limit, and a
// call over the limit gives F_LIMIT.
namespace scientific;

import std;
import mathshim;

// The most draws that one call of the binomial law or of the
// hypergeometric law makes. The cost of the call grows with the mean
// of the smaller side.
const DISC_COST_MAX: f64 = 1.0e6;
// The largest whole number that a call gives back. A larger value
// does not fit in the type.
const DISC_INT_MAX: f64 = 9.0e18;
// The largest count of trials of the multinomial law. Every count
// stays exact as a real number below it.
const DISC_TRIALS_MAX: f64 = 9.0e15;
// The largest size of one side of the urn of the hypergeometric law.
const DISC_URN_MAX: f64 = 1.0e15;
// A shape of the zipf law above this size gives one. The chance of
// any other number stands below 2 to the minus 1000.
const DISC_ZIPF_ONE: f64 = 1000.0;
// The slack of the sum of the probabilities of the multinomial law.
// NumPy uses the same value.
const DISC_SUM_SLACK: f64 = 1.0e-12;

// disc_check_prob checks a probability. The low end and the high end
// are in the range or out of it, as the flags say. A value that is not
// a number fails.
fn disc_check_prob(p: f64, low_open: bool, high_open: bool, tag: const u8[:]) -> Fault {
    let low_ok = if low_open { p > 0.0 } else { p >= 0.0 };
    let high_ok = if high_open { p < 1.0 } else { p <= 1.0 };
    if !(low_ok && high_ok) {
        return fault(F_VALUE, tag, "the probability stands outside its range");
    }
    return no_fault();
}

// disc_gap gives the count of trials up to and with the next success
// of a geometric law. The argument lq is the logarithm of one less
// the probability.
fn disc_gap(r: Rng&, lq: f64) -> f64 {
    let u = random_uniform(r);
    return mathshim::floor_of(mathshim::ln(1.0 - u) / lq) + 1.0;
}

// binomial gives the count of successes in n trials of a probability
// p. The method counts the gaps of the geometric law. The smaller of p
// and one less p sets the cost, so the call takes the complement of a
// large p. The count of trials is at or above zero.
fn binomial(r: Rng&, n: i64, p: f64) -> i64, Fault {
    if n < 0 {
        return 0, fault(F_VALUE, "binomial", "the count of trials is below zero");
    }
    let fp = disc_check_prob(p, false, false, "binomial");
    if fp.kind != F_NONE { return 0, fp; }
    if n == 0 || p == 0.0 { return 0, no_fault(); }
    if p == 1.0 { return n, no_fault(); }
    let flip = p > 0.5;
    let q = if flip { 1.0 - p } else { p };
    let nf = n as f64;
    if nf * q > DISC_COST_MAX {
        return 0, fault(F_LIMIT, "binomial", "the mean of the smaller side stands above 1.0e6");
    }
    let lq = mathshim::log1p_of(-q);
    var pos: f64 = disc_gap(r, lq);
    var count: i64 = 0;
    while pos <= nf {
        count += 1;
        pos += disc_gap(r, lq);
    }
    let x = if flip { n - count } else { count };
    return x, no_fault();
}

// geometric gives the count of trials up to and with the first
// success. The count is at or above one. The probability stands above
// zero and at or below one. The method takes the inverse of the law.
fn geometric(r: Rng&, p: f64) -> i64, Fault {
    let fp = disc_check_prob(p, true, false, "geometric");
    if fp.kind != F_NONE { return 0, fp; }
    if p == 1.0 { return 1, no_fault(); }
    let x = disc_gap(r, mathshim::log1p_of(-p));
    if x > DISC_INT_MAX {
        return 0, fault(F_LIMIT, "geometric", "the count stands above the largest whole number");
    }
    return x as i64, no_fault();
}

// hypergeometric gives the count of good items in a sample of
// nsample items drawn without a return from an urn. The urn holds
// ngood good items and nbad bad items. The method draws the items one
// by one. It draws the smaller of the sample and its complement.
fn hypergeometric(r: Rng&, ngood: i64, nbad: i64, nsample: i64) -> i64, Fault {
    if ngood < 0 || nbad < 0 {
        return 0, fault(F_VALUE, "hypergeometric", "a count of items is below zero");
    }
    if (ngood as f64) > DISC_URN_MAX || (nbad as f64) > DISC_URN_MAX {
        return 0, fault(F_LIMIT, "hypergeometric", "a side of the urn stands above 1.0e15");
    }
    let total = ngood + nbad;
    if nsample < 0 || nsample > total {
        return 0, fault(F_VALUE, "hypergeometric", "the sample stands outside zero to the size of the urn");
    }
    let flip = nsample > total - nsample;
    let k = if flip { total - nsample } else { nsample };
    if (k as f64) > DISC_COST_MAX {
        return 0, fault(F_LIMIT, "hypergeometric", "the smaller side of the sample stands above 1.0e6");
    }
    var good = ngood;
    var left = total;
    var hit: i64 = 0;
    var i: i64 = 0;
    while i < k {
        let u = random_uniform(r);
        if u * (left as f64) < (good as f64) {
            hit += 1;
            good -= 1;
        }
        left -= 1;
        i += 1;
    }
    let x = if flip { ngood - hit } else { hit };
    return x, no_fault();
}

// logseries gives a number of the logarithmic series law. The method
// is the one of Kemp. The probability is at or above zero and below
// one.
fn logseries(r: Rng&, p: f64) -> i64, Fault {
    let fp = disc_check_prob(p, false, true, "logseries");
    if fp.kind != F_NONE { return 0, fp; }
    let lr = mathshim::log1p_of(-p);
    var round: i64 = 0;
    while round < DIST_ROUNDS {
        let v = random_uniform(r);
        if v >= p { return 1, no_fault(); }
        let u = random_uniform(r);
        let q = -mathshim::expm1_of(lr * u);
        if v <= q * q {
            let x = mathshim::floor_of(1.0 + mathshim::ln(v) / mathshim::ln(q));
            if x >= 1.0 && v > 0.0 {
                if x > DISC_INT_MAX {
                    return 0, fault(F_LIMIT, "logseries", "the number stands above the largest whole number");
                }
                return x as i64, no_fault();
            }
        } else {
            if v >= q { return 1, no_fault(); }
            return 2, no_fault();
        }
        round += 1;
    }
    return 0, fault(F_LIMIT, "logseries", "the sampler did not accept a number in 1000 rounds");
}

// negative_binomial gives the count of failures before n successes of
// a probability p. The count n is a real number above zero. The method
// is the gamma mixture of Poisson numbers. The mean of the mixture
// stands at or below 1.0e6.
fn negative_binomial(r: Rng&, n: f64, p: f64) -> i64, Fault {
    let fn_ = dist_check_pos(n, "negative_binomial", "the count of successes is not above zero");
    if fn_.kind != F_NONE { return 0, fn_; }
    let fp = disc_check_prob(p, true, false, "negative_binomial");
    if fp.kind != F_NONE { return 0, fp; }
    if p == 1.0 { return 0, no_fault(); }
    let y, fy = gamma(r, n, (1.0 - p) / p);
    if fy.kind != F_NONE { return 0, fy; }
    if y > POISSON_RATE_SUM_MAX {
        return 0, fault(F_LIMIT, "negative_binomial", "the rate of the Poisson mixture stands above 1.0e6");
    }
    return poisson(r, y);
}

// zipf gives a number of the zipf law. The method is the rejection
// method of Devroye. The shape stands above one. A shape above 1000
// gives one. A shape so near one that the method rejects 1000 rounds
// gives F_LIMIT.
fn zipf(r: Rng&, a: f64) -> i64, Fault {
    if !(a > 1.0) || mathshim::is_finite_of(a) == 0.0 {
        return 0, fault(F_VALUE, "zipf", "the shape is not a finite number above one");
    }
    if a > DISC_ZIPF_ONE { return 1, no_fault(); }
    let am1 = a - 1.0;
    let b = mathshim::power(2.0, am1);
    var round: i64 = 0;
    while round < DIST_ROUNDS {
        let u = 1.0 - random_uniform(r);
        let v = random_uniform(r);
        let x = mathshim::floor_of(mathshim::power(u, -1.0 / am1));
        if x >= 1.0 && x <= DISC_INT_MAX {
            let t = mathshim::power(1.0 + 1.0 / x, am1);
            if v * x * (t - 1.0) / (b - 1.0) <= t / b { return x as i64, no_fault(); }
        }
        round += 1;
    }
    return 0, fault(F_LIMIT, "zipf", "the sampler did not accept a number in 1000 rounds");
}

// multinomial gives the counts of n trials over the classes of a row
// of probabilities. Each count comes from a binomial number of the
// trials that remain. The row holds one dimension, and each value
// stands in the range zero to one. The sum of all but the last value
// stands at or below one.
fn multinomial(r: Rng&, n: i64, pvals) -> Arr, Fault {
    if pvals.ndim != 1 || pvals.data.len < 1 {
        return empty_arr(), fault(F_SHAPE, "multinomial", "the probabilities hold a count of dimensions that is not one, or no value");
    }
    if n < 0 {
        return empty_arr(), fault(F_VALUE, "multinomial", "the count of trials is below zero");
    }
    if (n as f64) > DISC_TRIALS_MAX {
        return empty_arr(), fault(F_LIMIT, "multinomial", "the count of trials stands above 9.0e15");
    }
    let k = pvals.data.len;
    var sum: f64 = 0.0;
    var i: i64 = 0;
    while i < k {
        let fp = disc_check_prob(pvals.data[i], false, false, "multinomial");
        if fp.kind != F_NONE { return empty_arr(), fp; }
        if i < k - 1 { sum += pvals.data[i]; }
        i += 1;
    }
    if sum > 1.0 + DISC_SUM_SLACK {
        return empty_arr(), fault(F_VALUE, "multinomial", "the sum of the probabilities stands above one");
    }
    // The cost cap of the binomial call holds for each class. The cost
    // of the whole call is the sum over the classes, so a long row
    // would pass the cap class by class. One budget covers them all.
    var budget: f64 = DISC_COST_MAX;
    var d: f64[>..] = [];
    var left = n;
    var mass: f64 = 1.0;
    i = 0;
    while i < k - 1 {
        var x: i64 = 0;
        if left > 0 && mass > 0.0 {
            var c = pvals.data[i] / mass;
            if c > 1.0 { c = 1.0; }
            // The cost of the call is the mean of its smaller side.
            budget -= (left as f64) * (if c > 0.5 { 1.0 - c } else { c });
            if budget < 0.0 {
                return empty_arr(), fault(F_LIMIT, "multinomial", "the cost of all classes stands above 1.0e6 draws");
            }
            let bx, fb = binomial(r, left, c);
            if fb.kind != F_NONE { return empty_arr(), fb; }
            x = bx;
        }
        d.push(x as f64);
        left -= x;
        mass -= pvals.data[i];
        i += 1;
    }
    d.push(left as f64);
    var s: i64[..8] = [];
    s.push(k);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}
