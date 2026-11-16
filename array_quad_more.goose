// The quadrature: the rules of Newton and Cotes, the sums, the rule of
// the hyperbolic tangent and the rule of the quasi random points.
//
// The weights of Newton and Cotes.
//
// newton_cotes gives the weights of a rule of Newton and Cotes of an
// order. The rule of an order up to twenty holds the weights as
// fractions. The rule of a higher order solves the system of the
// samples. The second result is the factor of the term of the error.
//
// The sums.
//
// nsum sums the terms of a series. The series may hold an infinite
// count of terms. The method sums the terms up to a point and carries
// the rest of the sum with the test of the integral.
//
// The rule of the hyperbolic tangent.
//
// tanhsinh integrates a function from a to b by the rule of the double
// exponential. The node t maps to x = m + h tanh(pi/2 sinh(t)), where m
// is the middle and h is the half of the width. Each level halves the
// step of t and sums again. The call stops when two levels agree to the
// tolerance. A range that reaches to infinity takes the call quad.
//
// The rule of the quasi random points.
//
// qmc_quad integrates a function over a box with points of a low
// discrepancy sequence. The sequence of this package is the sequence
// of Halton, and its points do not equal the points of SciPy, because
// the sources of the points differ. The error of the call follows the
// rule of the estimates, so the error does not depend on the points.
namespace scientific;

import mathshim;

// The weights of the rules of Newton and Cotes of the order one to
// fourteen. The count of the weights of a rule of the order N is N + 1.
// The weights of the rule of an order stand over one denominator. The
// tables hold the numerators and the denominators of the weights and of
// the factor of the error.
const NC_MAX: i64 = 14;

const NC_NA: i64[14] = [1, 1, 3, 2, 5, 1, 7, 4, 9, 5, 11, 1, 13, 7];
const NC_DA: i64[14] = [2, 3, 8, 45, 288, 140, 17280, 14175, 89600, 299376, 87091200, 5255250, 402361344000, 2501928000];
const NC_NB: i64[14] = [-1, -1, -3, -8, -275, -9, -8183, -2368, -4671, -673175, -2224234463, -3012, -2639651053, -3740727473];
const NC_DB: i64[14] = [12, 90, 80, 945, 12096, 1400, 518400, 467775, 394240, 163459296, 237758976000, 875875, 344881152000, 1275983280000];

// The values of the numerators of the weights. The weights of the rule
// of the order N stand in the table at the places of the row N.
const NC_ROW_AT: i64[15] = [0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 104, 119];
const NC_VI: f64[119] = [
    1.0, 1.0,
    1.0, 4.0, 1.0,
    1.0, 3.0, 3.0, 1.0,
    7.0, 32.0, 12.0, 32.0, 7.0,
    19.0, 75.0, 50.0, 50.0, 75.0, 19.0,
    41.0, 216.0, 27.0, 272.0, 27.0, 216.0, 41.0,
    751.0, 3577.0, 1323.0, 2989.0, 2989.0, 1323.0, 3577.0, 751.0,
    989.0, 5888.0, -928.0, 10496.0, -4540.0, 10496.0, -928.0, 5888.0, 989.0,
    2857.0, 15741.0, 1080.0, 19344.0, 5778.0, 5778.0, 19344.0, 1080.0, 15741.0, 2857.0,
    16067.0, 106300.0, -48525.0, 272400.0, -260550.0, 427368.0, -260550.0, 272400.0, -48525.0, 106300.0, 16067.0,
    2171465.0, 13486539.0, -3237113.0, 25226685.0, -9595542.0, 15493566.0, 15493566.0, -9595542.0, 25226685.0, -3237113.0, 13486539.0, 2171465.0,
    1364651.0, 9903168.0, -7587864.0, 35725120.0, -51491295.0, 87516288.0, -87797136.0, 87516288.0, -51491295.0, 35725120.0, -7587864.0, 9903168.0, 1364651.0,
    8181904909.0, 56280729661.0, -31268252574.0, 156074417954.0, -151659573325.0, 206683437987.0, -43111992612.0, -43111992612.0, 206683437987.0, -151659573325.0, 156074417954.0, -31268252574.0, 56280729661.0, 8181904909.0,
    90241897.0, 710986864.0, -770720657.0, 3501442784.0, -6625093363.0, 12630121616.0, -16802270373.0, 19534438464.0, -16802270373.0, 12630121616.0, -6625093363.0, 3501442784.0, -770720657.0, 710986864.0, 90241897.0
];

// nc_ln_gamma gives the logarithm of the factorial of a whole number.
fn nc_ln_gamma(n: i64) -> f64 {
    var t: f64 = 0.0;
    var k: i64 = 2;
    while k <= n {
        t += mathshim::ln(k);
        k += 1;
    }
    return t;
}

// newton_cotes gives the weights of the rule of Newton and Cotes of the
// order n. The rule of an order up to twenty holds the weights as
// fractions, and the weights of a higher order come from the system of
// the samples. The second result is the factor of the term of the
// error.
fn newton_cotes(n: i64) -> Arr, f64, Fault {
    if n < 1 || n > 1024 {
        return empty_arr(), 0.0, fault(F_RANGE, "newton_cotes", "the order stands outside one to 1024");
    }
    if n <= NC_MAX {
        let at = NC_ROW_AT[n - 1];
        var w: f64[>..] = [];
        var i: i64 = 0;
        let count = NC_ROW_AT[n] - at;
        while i < count {
            w.push(NC_NA[n - 1] * NC_VI[at + i] / NC_DA[n - 1]);
            i += 1;
        }
        var s: i64[..8] = [];
        s.push(count);
        return Arr { ndim: 1, shape: s, data: w }, NC_NB[n - 1] as f64 / NC_DB[n - 1], no_fault();
    }
    // The rule of an order from fifteen to twenty holds the exact
    // fractions of its weights, because the system of the samples loses
    // the digits of the fractions at these orders.
    let ew, eb, ok = nc_exact(n);
    if ok {
        var s: i64[..8] = [];
        s.push(ew.len);
        return Arr { ndim: 1, shape: s, data: ew }, eb, no_fault();
    }
    // The system of the samples. The matrix holds the powers of the
    // places of the samples from minus one to one. A row stands for a
    // power and a column stands for a sample, as SciPy builds it with
    // ti ** nvec[:, newaxis]. The weights then read the even columns of
    // the inverse, where a row of the inverse stands for a sample.
    var cd: f64[>..] = [];
    var pw: i64 = 0;
    while pw <= n {
        var i: i64 = 0;
        while i <= n {
            let ti = 2.0 * i as f64 / n - 1.0;
            var p: f64 = 1.0;
            var j: i64 = 0;
            while j < pw { p *= ti; j += 1; }
            cd.push(p);
            i += 1;
        }
        pw += 1;
    }
    var i: i64 = 0;
    let c = mat2(copy(cd), n + 1, n + 1);
    let ci, f0 = inv(c);
    if f0.kind != F_NONE { return empty_arr(), 0.0, f0; }
    // Two refinements lift the values of the inverse.
    var turn: i64 = 0;
    while turn < 2 {
        var prod: f64[>..] = [];
        i = 0;
        while i <= n {
            var row: f64[>..] = [];
            var j: i64 = 0;
            while j <= n {
                var sum: f64 = 0.0;
                var l: i64 = 0;
                while l <= n { sum += ci.data[i * (n + 1) + l] * c.data[l * (n + 1) + j]; l += 1; }
                row.push(sum);
                j += 1;
            }
            var k: i64 = 0;
            while k <= n {
                var sum: f64 = 0.0;
                var l: i64 = 0;
                while l <= n { sum += row[l] * ci.data[l * (n + 1) + k]; l += 1; }
                prod.push(2.0 * ci.data[i * (n + 1) + k] - sum);
                k += 1;
            }
            i += 1;
        }
        ci.data = copy(prod);
        turn += 1;
    }
    // The weights hold the even columns of the inverse times the factors
    // of the odd powers.
    var w: f64[>..] = [];
    var fac: f64[>..] = [];
    var odd: i64 = 0;
    while odd <= n {
        fac.push(2.0 / (odd + 1.0));
        odd += 2;
    }
    i = 0;
    while i <= n {
        var sum: f64 = 0.0;
        var j: i64 = 0;
        var col: i64 = 0;
        while col <= n {
            if col % 2 == 0 { sum += ci.data[i * (n + 1) + col] * fac[j]; j += 1; }
            col += 1;
        }
        w.push(sum * n / 2.0);
        i += 1;
    }
    // The factor of the error. A rule of an even order carries the
    // power of one more degree.
    var bn: f64 = 0.0;
    var power: i64 = 0;
    if n % 2 == 0 {
        bn = n as f64 / (n + 3);
        power = n + 2;
    } else {
        bn = n as f64 / (n + 2);
        power = n + 1;
    }
    var dot: f64 = 0.0;
    i = 0;
    while i <= n {
        var yi = i as f64 / n;
        dot += mathshim::power(yi, power) * w[i];
        i += 1;
    }
    bn -= dot;
    // The factor of the error holds the factorial of the power, as the
    // formula of SciPy holds it. The helper gives the logarithm of the
    // factorial of its argument, and the argument is the power.
    let lnfac = power * mathshim::ln(n) - nc_ln_gamma(power);
    return line_arr(copy(w)), bn * mathshim::power_e(lnfac), no_fault();
}

// quad_inf gives the infinite value that the range of quad takes for an
// end at infinity. quad_is_inf names a value that is infinite but not
// the missing value.
fn quad_inf() -> f64 {
    var v: f64 = 1.0e308;
    while v - v == 0.0 { v = v * 2.0; }
    return v;
}

// NSUM_MAX_TERMS holds the largest count of the terms of the direct sum.
const NSUM_MAX_TERMS: i64 = 1048576;

// nsum sums the terms of a series from a to b. The series may hold an
// infinite count of terms, with an infinite value for the end. The
// method sums the terms up to a point c and carries the rest of the
// sum with the test of the integral. The point c is the first point
// whose term falls below the tolerance. The result holds the sum and
// the estimate of the error.
fn nsum<F: fn(f64) -> f64>(a: f64, b: f64, step: f64 = 1.0) -> f64, f64, Fault {
    if !(step > 0.0) || step != step {
        return 0.0, 0.0, fault(F_VALUE, "nsum", "the step holds a value below zero or the missing value");
    }
    if !(b >= a) {
        return 0.0, 0.0, fault(F_VALUE, "nsum", "the end stands below the start");
    }
    if a != a || b != b {
        return 0.0, 0.0, fault(F_VALUE, "nsum", "a limit holds the missing value");
    }
    let inf_end = quad_is_inf(b);
    var nterms: i64 = NSUM_MAX_TERMS;
    if !inf_end {
        nterms = mathshim::floor_of((b - a) / step) as i64 + 1;
        if nterms < 0 { nterms = 0; }
    }
    // The sum of the terms from a to the end b, when the count of the
    // terms fits.
    if !inf_end && nterms <= NSUM_MAX_TERMS {
        var total: f64 = 0.0;
        var k: i64 = 0;
        while k < nterms {
            total += F(a + k as f64 * step);
            k += 1;
        }
        return total, 0.0, no_fault();
    }
    // The integral test. The sum runs to the point c and the rest of
    // the sum comes from the integral from c.
    let tol = 1.0e-8;
    var c: f64 = a + NSUM_MAX_TERMS as f64 * step;
    var k: i64 = 0;
    var found = false;
    while k < NSUM_MAX_TERMS {
        let t = F(a + k as f64 * step);
        if !(t > 0.0) || t != t {
            return 0.0, 0.0, fault(F_VALUE, "nsum", "the terms of the series must be finite and above zero");
        }
        if t < tol {
            c = a + k as f64 * step;
            found = true;
            break;
        }
        k += 1;
    }
    if !found {
        return 0.0, 0.0, fault(F_LIMIT, "nsum", "the terms did not fall below the tolerance within the count of the terms");
    }
    var total: f64 = 0.0;
    var t: f64 = a;
    while t < c {
        total += F(t);
        t += step;
    }
    // The integral from c and the half of the term at c. A sum over a
    // step holds one term for each step of the range, so the integral
    // divides by the step.
    let iv, ie, fq = quad(c, quad_inf()) { x => F(x) };
    if fq.kind != F_NONE { return 0.0, 0.0, fq; }
    total += iv / step + 0.5 * F(c);
    return total, 0.5 * F(c) + ie / step, no_fault();
}

// tanhsinh integrates a function from a to b by the rule of the double
// exponential. The node t maps to x = m + h tanh(pi/2 sinh(t)), where m
// is the middle and h is the half of the width. Each level halves the
// step of t and sums again. The call stops when two levels agree to the
// tolerance. A range that reaches to infinity takes the call quad.
fn tanhsinh<F: fn(f64) -> f64>(a: f64, b: f64, epsabs: f64 = 1.0e-10, epsrel: f64 = 1.0e-10, maxlevel: i64 = 10) -> f64, f64, Fault {
    if a != a || b != b {
        return 0.0, 0.0, fault(F_VALUE, "tanhsinh", "a limit holds the missing value");
    }
    if !(epsabs >= 0.0) || !(epsrel >= 0.0) {
        return 0.0, 0.0, fault(F_VALUE, "tanhsinh", "the tolerance holds a value below zero");
    }
    if maxlevel < 1 || maxlevel > 16 {
        return 0.0, 0.0, fault(F_RANGE, "tanhsinh", "the level stands outside one to 16");
    }
    if quad_is_inf(a) || quad_is_inf(b) {
        let v, e, f = quad(a, b, epsabs, epsrel) { x => F(x) };
        return v, e, f;
    }
    if a == b { return 0.0, 0.0, no_fault(); }
    var lo = a;
    var hi = b;
    var sgn: f64 = 1.0;
    if a > b { lo = b; hi = a; sgn = -1.0; }
    let mid = 0.5 * (lo + hi);
    let half = 0.5 * (hi - lo);
    let hp = 1.5707963267948966;
    var prev: f64 = 0.0;
    var err: f64 = 1.0e300;
    var level: i64 = 1;
    while level <= maxlevel {
        let h = mathshim::power(2.0, -(level as f64));
        var sum: f64 = 0.0;
        var k: i64 = 0;
        while true {
            let t = h * k as f64;
            let sh = mathshim::sinh_of(t);
            let u = hp * sh;
            if u > 20.0 { break; }
            let ch = mathshim::cosh_of(u);
            let w = hp * mathshim::cosh_of(t) / (ch * ch);
            // The distance of the node from its end. The difference
            // half - half * tanh(u) loses every digit near an end, so
            // the distance comes from 2 half / (1 + exp(2 u)). A
            // function that is singular at an end sees the true place.
            let dist = 2.0 * half / (1.0 + mathshim::power_e(2.0 * u));
            let xr = hi - dist;
            let xl = lo + dist;
            // A node that rounds onto an end adds no weight: the weight
            // there is below the rounding of the sum.
            if !(dist > 0.0) || xr <= xl && k > 0 { break; }
            var term = F(xr) + F(xl);
            if k == 0 { term = F(mid); }
            sum += w * term;
            k += 1;
        }
        let cur = sgn * half * h * sum;
        err = mathshim::abs_of(cur - prev);
        if level > 2 && (err <= epsabs || err <= epsrel * mathshim::abs_of(cur)) {
            return cur, err, no_fault();
        }
        prev = cur;
        level += 1;
    }
    return prev, err, fault(F_LIMIT, "tanhsinh", "the levels did not agree to the tolerance");
}

// halton_place gives the place of the index i of the sequence of Halton
// at the base.
fn halton_place(i: i64, base: i64) -> f64 {
    var f: f64 = 1.0;
    var r: f64 = 0.0;
    var k = i;
    while k > 0 {
        f /= base;
        r += f * (k % base);
        k = k / base;
    }
    return r;
}

// QMC_MAX_POINTS holds the largest count of the points of one estimate.
const QMC_MAX_POINTS: i64 = 4194304;

// qmc_quad integrates a function over a box with points of the sequence
// of Halton. The method draws n_estimates sets of the points and gives
// the mean of the estimates and the error of the estimates. The points
// of the sequence of this package do not equal the points of SciPy,
// because the sources of the points differ.
fn qmc_quad<F: fn(f64[:]) -> f64>(a: f64[:], b: f64[:], n_points: i64 = 1024, n_estimates: i64 = 8, seed: i64 = 0) -> f64, f64, Fault {
    if a.len != b.len || a.len < 1 {
        return 0.0, 0.0, fault(F_SHAPE, "qmc_quad", "the ends of the box differ in count or hold no dimension");
    }
    if n_points < 1 || n_points > QMC_MAX_POINTS {
        return 0.0, 0.0, fault(F_RANGE, "qmc_quad", "the count of the points stands outside one to the limit");
    }
    if n_estimates < 1 || n_estimates > 1024 {
        return 0.0, 0.0, fault(F_RANGE, "qmc_quad", "the count of the estimates stands outside one to 1024");
    }
    var d: i64 = 0;
    while d < a.len {
        if a[d] != a[d] || b[d] != b[d] || !(b[d] > a[d]) {
            return 0.0, 0.0, fault(F_VALUE, "qmc_quad", "a limit holds the missing value or an end below the other end");
        }
        d += 1;
    }
    // The first primes up to the count of the dimensions.
    var primes: i64[>..] = [];
    var cand: i64 = 2;
    while primes.len < a.len {
        var is_p = true;
        var p: i64 = 0;
        while p < primes.len {
            if primes[p] * primes[p] > cand { break; }
            if cand % primes[p] == 0 { is_p = false; break; }
            p += 1;
        }
        if is_p { primes.push(cand); }
        cand += 1;
    }
    // The estimates. The shift of the digital net gives each estimate
    // its own set of the points.
    var r = default_rng(seed);
    var means: f64[>..] = [];
    var e: i64 = 0;
    while e < n_estimates {
        let shift = uniform(r, 0.0, 1.0);
        var sum: f64 = 0.0;
        var p: i64 = 0;
        while p < n_points {
            var x: f64[>..] = [];
            d = 0;
            while d < a.len {
                var u = halton_place(p, primes[d]);
                u = u + shift;
                if u >= 1.0 { u -= 1.0; }
                x.push(a[d] + (b[d] - a[d]) * u);
                d += 1;
            }
            sum += F(copy(x));
            p += 1;
        }
        var vol: f64 = 1.0;
        d = 0;
        while d < a.len { vol *= b[d] - a[d]; d += 1; }
        means.push(sum * vol / n_points as f64);
        e += 1;
    }
    var mean: f64 = 0.0;
    var i: i64 = 0;
    while i < means.len { mean += means[i]; i += 1; }
    mean /= means.len as f64;
    var varsum: f64 = 0.0;
    i = 0;
    while i < means.len {
        let d2 = means[i] - mean;
        varsum += d2 * d2;
        i += 1;
    }
    let err = mathshim::root(varsum / (means.len - 1) as f64) / mathshim::root(means.len as f64);
    return mean, err, no_fault();
}
