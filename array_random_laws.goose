// The laws of the F family, of the circle, of the Pareto, of the
// triangle and of the Wald form.
//
// The calls f, noncentral_chisquare, noncentral_f, pareto, triangular,
// vonmises and wald follow numpy.random.Generator. Each call takes the
// generator by reference and gives one number and a fault. The stream
// of the generator differs from the stream of NumPy. The law is the
// same: the mean, the variance and the quartiles agree.
//
// The rules of the parameters are the rules of NumPy. A negative zero
// is refused where NumPy refuses it. A value that is not finite is
// refused too. The package never gives a number that is not finite.
//
// The law of the von Mises lies on the circle. NumPy returns the
// number in the range from minus pi to pi. SciPy centres the range on
// the location. The package follows NumPy.
namespace scientific;

import std;
import mathshim;

// The count of the degrees of freedom below which the noncentral
// chi-square law uses the mixture of Poisson numbers.
const LAW_NCX_MIX: f64 = 1.0;
// The count of the degrees of freedom above which the noncentral
// chi-square law adds a chi-square number of the count less one.
const LAW_NCX_ONE: f64 = 1.0;
// A concentration of the von Mises law below this size gives a
// uniform number of the circle.
const LAW_VM_FLAT: f64 = 1.0e-8;
// A concentration below this size gives the short form of the
// constant of the sampler.
const LAW_VM_SHORT: f64 = 1.0e-5;
// A concentration above this size gives the normal form of NumPy.
const LAW_VM_NORMAL: f64 = 1.0e6;

// law_wrap gives a number of the circle in the range from minus pi to
// pi. A number of the range from minus pi to pi stays as it is.
fn law_wrap(x: f64) -> f64 {
    let two_pi = 2.0 * PI_VALUE;
    return mathshim::mod_of(x + PI_VALUE, two_pi) - PI_VALUE;
}

// law_ratio gives the ratio of two numbers, each divided by its count
// of degrees of freedom. The F law and the noncentral F law use it.
fn law_ratio(a: f64, b: f64, dfnum: f64, dfden: f64, tag: const u8[:]) -> f64, Fault {
    let x = (a / dfnum) / (b / dfden);
    return dist_ok(x, tag);
}

// f gives a number of the F law. The law is the ratio of two
// chi-square numbers, each divided by its count of degrees of
// freedom. Both counts are above zero.
fn f(r: Rng&, dfnum: f64, dfden: f64) -> f64, Fault {
    let fn_ = dist_check_count(dfnum, "f", "the count of the numerator is not above zero");
    if fn_.kind != F_NONE { return 0.0, fn_; }
    let fd = dist_check_count(dfden, "f", "the count of the denominator is not above zero");
    if fd.kind != F_NONE { return 0.0, fd; }
    let a, fa = chisquare(r, dfnum);
    if fa.kind != F_NONE { return 0.0, fa; }
    let b, fb = chisquare(r, dfden);
    if fb.kind != F_NONE { return 0.0, fb; }
    return law_ratio(a, b, dfnum, dfden, "f");
}

// noncentral_chisquare gives a number of the noncentral chi-square
// law. A count of one degree of freedom or more gives the square of
// a normal number shifted by the root of the noncentrality, plus a
// chi-square number of the count less one. A count below one gives a
// chi-square number of the count plus twice a Poisson number of rate
// half the noncentrality.
fn noncentral_chisquare(r: Rng&, df: f64, nonc: f64) -> f64, Fault {
    let fd = dist_check_count(df, "noncentral_chisquare", "the count of the degrees of freedom is not above zero");
    if fd.kind != F_NONE { return 0.0, fd; }
    let fc = dist_check_nonneg(nonc, "noncentral_chisquare", "the noncentrality is below zero or not finite");
    if fc.kind != F_NONE { return 0.0, fc; }
    if df >= LAW_NCX_MIX {
        let z, fz = normal(r, 0.0, 1.0);
        if fz.kind != F_NONE { return 0.0, fz; }
        let w = z + mathshim::root(nonc);
        var x = w * w;
        if df > LAW_NCX_ONE {
            let c, fcs = chisquare(r, df - 1.0);
            if fcs.kind != F_NONE { return 0.0, fcs; }
            x += c;
        }
        return dist_ok(x, "noncentral_chisquare");
    }
    // A count below one uses the mixture of Poisson numbers. The
    // rate of the Poisson law stands at or below 1.0e6, so the
    // noncentrality stands at or below 2.0e6.
    if nonc > 2.0 * POISSON_RATE_SUM_MAX {
        return 0.0, fault(F_LIMIT, "noncentral_chisquare", "the noncentrality stands above 2.0e6");
    }
    let k, fk = poisson(r, nonc / 2.0);
    if fk.kind != F_NONE { return 0.0, fk; }
    let c, fcs = chisquare(r, df + 2.0 * k);
    if fcs.kind != F_NONE { return 0.0, fcs; }
    return dist_ok(c, "noncentral_chisquare");
}

// noncentral_f gives a number of the noncentral F law. The law is the
// ratio of a noncentral chi-square number and a chi-square number,
// each divided by its count of degrees of freedom.
fn noncentral_f(r: Rng&, dfnum: f64, dfden: f64, nonc: f64) -> f64, Fault {
    let fn_ = dist_check_count(dfnum, "noncentral_f", "the count of the numerator is not above zero");
    if fn_.kind != F_NONE { return 0.0, fn_; }
    let fd = dist_check_count(dfden, "noncentral_f", "the count of the denominator is not above zero");
    if fd.kind != F_NONE { return 0.0, fd; }
    let fc = dist_check_nonneg(nonc, "noncentral_f", "the noncentrality is below zero or not finite");
    if fc.kind != F_NONE { return 0.0, fc; }
    let a, fa = noncentral_chisquare(r, dfnum, nonc);
    if fa.kind != F_NONE { return 0.0, fa; }
    let b, fb = chisquare(r, dfden);
    if fb.kind != F_NONE { return 0.0, fb; }
    return law_ratio(a, b, dfnum, dfden, "noncentral_f");
}

// pareto gives a number of the Pareto law of NumPy. That law is the
// Lomax form of SciPy, pareto with a shape a and a location of minus
// one. Its support starts at zero and its mean is one over the shape
// less one. The shape is above zero.
fn pareto(r: Rng&, a: f64) -> f64, Fault {
    let fa = dist_check_pos(a, "pareto", "the shape is not above zero");
    if fa.kind != F_NONE { return 0.0, fa; }
    let fl = dist_check_shape(a, "pareto");
    if fl.kind != F_NONE { return 0.0, fl; }
    let u = 1.0 - random_uniform(r);
    let x = mathshim::power(u, -1.0 / a) - 1.0;
    return dist_ok(x, "pareto");
}

// triangular gives a number of the triangular law. The method takes
// the inverse of the law. The left is at or below the mode and the
// mode is at or below the right. The left and the right differ.
fn triangular(r: Rng&, left: f64, mode: f64, right: f64) -> f64, Fault {
    let fl = dist_check_finite(left, "triangular", "an end of the range is not finite");
    if fl.kind != F_NONE { return 0.0, fl; }
    let fm = dist_check_finite(mode, "triangular", "an end of the range is not finite");
    if fm.kind != F_NONE { return 0.0, fm; }
    let fr = dist_check_finite(right, "triangular", "an end of the range is not finite");
    if fr.kind != F_NONE { return 0.0, fr; }
    if left > mode {
        return 0.0, fault(F_VALUE, "triangular", "the left stands above the mode");
    }
    if mode > right {
        return 0.0, fault(F_VALUE, "triangular", "the mode stands above the right");
    }
    if left == right {
        return 0.0, fault(F_VALUE, "triangular", "the left and the right are the same");
    }
    let u = random_uniform(r);
    let width = right - left;
    if mathshim::is_finite_of(width) == 0.0 {
        return 0.0, fault(F_LIMIT, "triangular", "the range stands wider than the largest number");
    }
    // The root of a product of three wide numbers overflows before the
    // root of each factor does, so the call takes the roots apart.
    let x = if u < (mode - left) / width {
        left + mathshim::root(u * width) * mathshim::root(mode - left)
    } else {
        right - mathshim::root((1.0 - u) * width) * mathshim::root(right - mode)
    };
    return dist_ok(x, "triangular");
}

// vonmises_angle gives a number of the von Mises law with a location
// of zero. The method is the one of Devroye. It draws about two
// numbers for one number of the law. The rejection stops after 1000
// rounds.
fn vonmises_angle(r: Rng&, kappa: f64) -> f64, Fault {
    let fc = dist_check_nonneg(kappa, "vonmises", "the concentration is below zero or not finite");
    if fc.kind != F_NONE { return 0.0, fc; }
    if kappa < LAW_VM_FLAT {
        let u = random_uniform(r);
        return PI_VALUE * (2.0 * u - 1.0), no_fault();
    }
    // The constant of the method. A small concentration gives the
    // short form of it.
    var s: f64 = 0.0;
    if kappa < LAW_VM_SHORT {
        s = 1.0 / kappa + kappa;
    } else {
        let q = 1.0 + mathshim::root(1.0 + 4.0 * kappa * kappa);
        let rho = (q - mathshim::root(2.0 * q)) / (2.0 * kappa);
        s = (1.0 + rho * rho) / (2.0 * rho);
    }
    var round: i64 = 0;
    while round < DIST_ROUNDS {
        // The first uniform gives the point on the circle and the sign.
        // The cosine is even, so the sign of the number is free.
        let a = 2.0 * random_uniform(r) - 1.0;
        let z = mathshim::cosine(PI_VALUE * a);
        let w = (1.0 + s * z) / (s + z);
        let y = kappa * (s - w);
        let v = random_uniform(r);
        var ok: bool = y * (2.0 - y) - v >= 0.0;
        if !ok && y > 0.0 && v > 0.0 {
            ok = mathshim::ln(y / v) + 1.0 - y >= 0.0;
        }
        if ok {
            let x = if a < 0.0 { -mathshim::arccosine_of(w) } else { mathshim::arccosine_of(w) };
            return x, no_fault();
        }
        round += 1;
    }
    return 0.0, fault(F_LIMIT, "vonmises", "the sampler did not accept a number in 1000 rounds");
}

// vonmises gives a number of the von Mises law. The number lies
// between minus pi and pi, as in NumPy. The location is a place on
// the circle, so the mean of the numbers is not the location when the
// location stands away from zero. The concentration is at or above
// zero. A concentration above 1e6 gives the normal form of NumPy.
fn vonmises(r: Rng&, mu: f64, kappa: f64) -> f64, Fault {
    let fl = dist_check_finite(mu, "vonmises", "the location is not finite");
    if fl.kind != F_NONE { return 0.0, fl; }
    let fc = dist_check_nonneg(kappa, "vonmises", "the concentration is below zero or not finite");
    if fc.kind != F_NONE { return 0.0, fc; }
    if kappa > LAW_VM_NORMAL {
        let z, fz = normal(r, 0.0, 1.0);
        if fz.kind != F_NONE { return 0.0, fz; }
        let x = mu + mathshim::root(1.0 / kappa) * z;
        return dist_ok(law_wrap(x), "vonmises");
    }
    let a, fa = vonmises_angle(r, kappa);
    if fa.kind != F_NONE { return 0.0, fa; }
    return dist_ok(law_wrap(mu + a), "vonmises");
}

// wald gives a number of the Wald law. The law is the inverse
// Gaussian law, with a mean of mean and a variance of the mean cubed
// over the scale. The method is the one of Michael, Schucany and
// Haas. The mean and the scale are above zero.
fn wald(r: Rng&, mean: f64, scale: f64) -> f64, Fault {
    let fm = dist_check_pos(mean, "wald", "the mean is not above zero");
    if fm.kind != F_NONE { return 0.0, fm; }
    let fs = dist_check_pos(scale, "wald", "the scale is not above zero");
    if fs.kind != F_NONE { return 0.0, fs; }
    let fml = dist_check_shape(mean, "wald");
    if fml.kind != F_NONE { return 0.0, fml; }
    let fsl = dist_check_shape(scale, "wald");
    if fsl.kind != F_NONE { return 0.0, fsl; }
    let m2 = mean * mean;
    let z, fd = normal(r, 0.0, 1.0);
    if fd.kind != F_NONE { return 0.0, fd; }
    // The two roots of the law are the mean plus the term and the mean
    // divided by the term. The larger root has no cancellation, so the
    // sampler computes it and takes the smaller one from it. The
    // larger root stands with probability mean over mean plus the root.
    let big = mean + m2 * z * z / (2.0 * scale)
            + (mean * mathshim::abs_of(z) / (2.0 * scale)) * mathshim::root(m2 * z * z + 4.0 * mean * scale);
    if big <= 0.0 {
        return 0.0, fault(F_VALUE, "wald", "the result stands at or below zero");
    }
    let u = random_uniform(r);
    let x = if u <= mean / (mean + big) { big } else { m2 / big };
    return dist_ok(x, "wald");
}
