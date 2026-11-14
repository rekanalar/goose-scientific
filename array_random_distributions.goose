// The continuous random laws.
//
// The calls gamma, lognormal, chisquare, beta, rayleigh, weibull, gumbel
// and logistic follow numpy.random.Generator. Each call takes the
// generator by reference and gives one number and a fault. The stream
// of the generator differs from the stream of NumPy. The law is the
// same: the mean, the variance and the quartiles agree.
//
// The rules of the parameters are the rules of NumPy. A negative zero
// is refused where NumPy refuses it. A value that is not finite is
// refused too. The package never gives a number that is not finite.
namespace scientific;

import std;
import mathshim;

// A shape or a count above this size gives F_LIMIT.
const DIST_SHAPE_MAX: f64 = 1.0e100;
// A shape above zero and below this size gives F_LIMIT.
const DIST_SHAPE_MIN: f64 = 1.0e-300;
// The rejection loop of the gamma sampler stops after this many rounds.
const DIST_ROUNDS: i64 = 1000;
// The smallest nonzero value that the complement of a uniform number
// gives. It stands in for an exact zero before a logarithm.
const DIST_TINY: f64 = 1.1102230246251565e-16;

// dist_check_nonneg checks a parameter that must be at or above zero.
// A negative zero fails, as in NumPy.
fn dist_check_nonneg(x: f64, tag: const u8[:], name: const u8[:]) -> Fault {
    if mathshim::is_finite_of(x) == 0.0 {
        return fault(F_VALUE, tag, name);
    }
    if mathshim::signbit_of(x) == 1.0 {
        return fault(F_VALUE, tag, name);
    }
    return no_fault();
}

// dist_check_pos checks a parameter that must be above zero.
fn dist_check_pos(x: f64, tag: const u8[:], name: const u8[:]) -> Fault {
    let f = dist_check_nonneg(x, tag, name);
    if f.kind != F_NONE { return f; }
    if x == 0.0 { return fault(F_VALUE, tag, name); }
    return no_fault();
}

// dist_check_shape checks the size of a shape that is above zero.
fn dist_check_shape(x: f64, tag: const u8[:]) -> Fault {
    if x > DIST_SHAPE_MAX || x < DIST_SHAPE_MIN {
        return fault(F_LIMIT, tag, "the shape stands outside 1.0e-300 to 1.0e100");
    }
    return no_fault();
}

// dist_ok checks that a result is finite and gives it with no fault.
fn dist_ok(x: f64, tag: const u8[:]) -> f64, Fault {
    if mathshim::is_finite_of(x) == 0.0 {
        return 0.0, fault(F_VALUE, tag, "the result is not finite");
    }
    return x, no_fault();
}

// dist_accept gives d times (1 - v + ln v) for the cube v of one plus
// w. A large d makes the plain sum lose every digit, so a small w uses
// the series of the sum. The series is -4.5 w^2 plus 3 times the sum of
// (-1)^(k+1) w^k / k for k from 4 up. It holds to a relative size of
// 1e-12 for |w| below 0.1.
fn dist_accept(d: f64, w: f64, v: f64) -> f64 {
    if mathshim::abs_of(w) >= 0.1 {
        return d * (1.0 - v + mathshim::ln(v));
    }
    var sum: f64 = 0.0;
    var p: f64 = w * w * w * w;
    var k: i64 = 4;
    var sign: f64 = -1.0;
    while k <= 16 {
        sum += sign * p / k;
        p *= w;
        sign = -sign;
        k += 1;
    }
    return d * (-4.5 * w * w + 3.0 * sum);
}

// gamma_log gives the logarithm of a gamma number of scale one. A shape
// at or above one uses the method of Marsaglia and Tsang. A shape below
// one uses the law of the shape plus one, times a power of a uniform
// number. The caller has checked the shape.
fn gamma_log(r: Rng&, shape: f64, tag: const u8[:]) -> f64, Fault {
    var s = shape;
    if shape < 1.0 { s = shape + 1.0; }
    let d = s - 1.0 / 3.0;
    let c = 1.0 / mathshim::root(9.0 * d);
    var round: i64 = 0;
    while round < DIST_ROUNDS {
        let x, _ = normal(r, 0.0, 1.0);
        let t = 1.0 + c * x;
        if t > 0.0 {
            let v = t * t * t;
            let u = 1.0 - rand_flt(r.g);
            if mathshim::ln(u) < 0.5 * x * x + dist_accept(d, c * x, v) {
                var lg = mathshim::ln(d * v);
                if shape < 1.0 { lg += mathshim::ln(1.0 - rand_flt(r.g)) / shape; }
                return lg, no_fault();
            }
        }
        round += 1;
    }
    return 0.0, fault(F_LIMIT, tag, "the sampler did not accept a number in 1000 rounds");
}

// gamma gives a number of the gamma law. The shape and the scale are at
// or above zero. A shape of zero or a scale of zero gives zero.
fn gamma(r: Rng&, shape: f64, scale: f64) -> f64, Fault {
    let fs = dist_check_nonneg(shape, "gamma", "the shape is below zero or not finite");
    if fs.kind != F_NONE { return 0.0, fs; }
    let fc = dist_check_nonneg(scale, "gamma", "the scale is below zero or not finite");
    if fc.kind != F_NONE { return 0.0, fc; }
    if shape == 0.0 || scale == 0.0 { return 0.0, no_fault(); }
    let fl = dist_check_shape(shape, "gamma");
    if fl.kind != F_NONE { return 0.0, fl; }
    let lg, fg = gamma_log(r, shape, "gamma");
    if fg.kind != F_NONE { return 0.0, fg; }
    let x = mathshim::power_e(lg) * scale;
    return dist_ok(x, "gamma");
}

// chisquare gives a number of the chi-square law. It is the gamma law
// of half the count of the degrees of freedom, with a scale of two.
fn chisquare(r: Rng&, df: f64) -> f64, Fault {
    let fd = dist_check_pos(df, "chisquare", "the count of the degrees of freedom is not above zero");
    if fd.kind != F_NONE { return 0.0, fd; }
    let fl = dist_check_shape(df / 2.0, "chisquare");
    if fl.kind != F_NONE { return 0.0, fl; }
    let lg, fg = gamma_log(r, df / 2.0, "chisquare");
    if fg.kind != F_NONE { return 0.0, fg; }
    let x = 2.0 * mathshim::power_e(lg);
    return dist_ok(x, "chisquare");
}

// beta gives a number of the beta law, in the range zero to one. It is
// the ratio of two gamma numbers. The call forms the ratio from the
// logarithms, so a small shape cannot give zero over zero.
fn beta(r: Rng&, a: f64, b: f64) -> f64, Fault {
    let fa = dist_check_pos(a, "beta", "the first shape is not above zero");
    if fa.kind != F_NONE { return 0.0, fa; }
    let fb = dist_check_pos(b, "beta", "the second shape is not above zero");
    if fb.kind != F_NONE { return 0.0, fb; }
    let fla = dist_check_shape(a, "beta");
    if fla.kind != F_NONE { return 0.0, fla; }
    let flb = dist_check_shape(b, "beta");
    if flb.kind != F_NONE { return 0.0, flb; }
    let lx, fx = gamma_log(r, a, "beta");
    if fx.kind != F_NONE { return 0.0, fx; }
    let ly, fy = gamma_log(r, b, "beta");
    if fy.kind != F_NONE { return 0.0, fy; }
    let x = 1.0 / (1.0 + mathshim::power_e(ly - lx));
    if x < 0.0 || x > 1.0 {
        return 0.0, fault(F_VALUE, "beta", "the result stands outside the range zero to one");
    }
    return dist_ok(x, "beta");
}

// lognormal gives a number whose logarithm is normal. A sigma of zero
// gives the exponential function of the mean.
fn lognormal(r: Rng&, mean: f64, sigma: f64) -> f64, Fault {
    if mathshim::is_finite_of(mean) == 0.0 {
        return 0.0, fault(F_VALUE, "lognormal", "the mean is not finite");
    }
    let fs = dist_check_nonneg(sigma, "lognormal", "the sigma is below zero or not finite");
    if fs.kind != F_NONE { return 0.0, fs; }
    var z: f64 = 0.0;
    if sigma > 0.0 {
        let n, fn_ = normal(r, mean, sigma);
        if fn_.kind != F_NONE { return 0.0, fn_; }
        z = n;
    } else {
        z = mean;
    }
    let x = mathshim::power_e(z);
    return dist_ok(x, "lognormal");
}

// rayleigh gives a number of the Rayleigh law. A scale of zero gives
// zero.
fn rayleigh(r: Rng&, scale: f64) -> f64, Fault {
    let fs = dist_check_nonneg(scale, "rayleigh", "the scale is below zero or not finite");
    if fs.kind != F_NONE { return 0.0, fs; }
    if scale == 0.0 { return 0.0, no_fault(); }
    let u = 1.0 - rand_flt(r.g);
    let x = scale * mathshim::root(-2.0 * mathshim::ln(u));
    return dist_ok(x, "rayleigh");
}

// weibull gives a number of the Weibull law of the shape a. The scale
// of the law is one, as in NumPy. A shape of zero gives zero.
fn weibull(r: Rng&, a: f64) -> f64, Fault {
    let fa = dist_check_nonneg(a, "weibull", "the shape is below zero or not finite");
    if fa.kind != F_NONE { return 0.0, fa; }
    if a == 0.0 { return 0.0, no_fault(); }
    let fl = dist_check_shape(a, "weibull");
    if fl.kind != F_NONE { return 0.0, fl; }
    let w = -mathshim::ln(1.0 - rand_flt(r.g));
    return dist_ok(mathshim::power(w, 1.0 / a), "weibull");
}

// gumbel gives a number of the Gumbel law of the right hand. Its mean
// lies above the location. A scale of zero gives the location.
fn gumbel(r: Rng&, loc: f64, scale: f64) -> f64, Fault {
    if mathshim::is_finite_of(loc) == 0.0 {
        return 0.0, fault(F_VALUE, "gumbel", "the location is not finite");
    }
    let fs = dist_check_nonneg(scale, "gumbel", "the scale is below zero or not finite");
    if fs.kind != F_NONE { return 0.0, fs; }
    if scale == 0.0 { return loc, no_fault(); }
    var w = -mathshim::ln(1.0 - rand_flt(r.g));
    if w == 0.0 { w = DIST_TINY; }
    let x = loc - scale * mathshim::ln(w);
    return dist_ok(x, "gumbel");
}

// logistic gives a number of the logistic law. A scale of zero gives
// the location.
fn logistic(r: Rng&, loc: f64, scale: f64) -> f64, Fault {
    if mathshim::is_finite_of(loc) == 0.0 {
        return 0.0, fault(F_VALUE, "logistic", "the location is not finite");
    }
    let fs = dist_check_nonneg(scale, "logistic", "the scale is below zero or not finite");
    if fs.kind != F_NONE { return 0.0, fs; }
    if scale == 0.0 { return loc, no_fault(); }
    var u = rand_flt(r.g);
    if u < DIST_TINY { u = DIST_TINY; }
    let x = loc + scale * mathshim::ln(u / (1.0 - u));
    return dist_ok(x, "logistic");
}
