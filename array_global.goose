// The global minimizers that use random steps: simulated annealing
// and basin hopping. Equal seeds give equal results, but the numbers
// differ from those of SciPy.
namespace scientific;

import mathshim;

// gl_clip moves a value into the bounds.
fn gl_clip(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo { return lo; }
    if v > hi { return hi; }
    return v;
}

// gl_finite says whether a number is finite.
fn gl_finite(v: f64) -> bool {
    return v - v == 0.0;
}

// gl_polish runs the local search from a point and keeps the result
// inside the bounds. The result holds the point and the value.
fn gl_polish<F: fn(f64[:]) -> f64>(x: f64[:], lower: f64[:], upper: f64[:], bounded: bool) -> f64[>..], f64 {
    let r, f = minimize(x, 1.0e-10, 400) { v => [F(copy(v))] };
    var best: f64[>..] = copy_slice(x);
    if f.kind == F_NONE && r.data.len == x.len {
        var cand: f64[>..] = [];
        var i: i64 = 0;
        while i < x.len {
            var c = r.data[i];
            if bounded { c = gl_clip(c, lower[i], upper[i]); }
            cand.push(c);
            i += 1;
        }
        let fc = F(copy(cand));
        let f0 = F(copy_slice(x));
        if gl_finite(fc) && fc <= f0 { return copy_slice(cand), fc; }
        return best, f0;
    }
    return best, F(copy_slice(x));
}

// dual_annealing finds the global minimum of a function inside the
// bounds. It runs a simulated annealing with a visiting distribution
// of heavy tails and the cooling schedule of the generalized
// annealing, and it ends with a local search from the best point. The
// result holds the point followed by its value as the last element.
// The visiting distribution is a Cauchy form, not the exact
// distribution of SciPy.
fn dual_annealing<F: fn(f64[:]) -> f64>(lower: f64[:], upper: f64[:], seed: i64 = 1, maxiter: i64 = 1000, initial_temp: f64 = 5230.0) -> Arr, Fault {
    let n = lower.len;
    if n < 1 || n != upper.len || n > 1000 {
        return empty_arr(), fault(F_SHAPE, "dual_annealing", "the bounds must hold one to 1000 variables, and the two lists must agree");
    }
    if maxiter < 1 || maxiter > 1000000 || !(initial_temp > 0.0) {
        return empty_arr(), fault(F_RANGE, "dual_annealing", "maxiter or the initial temperature stands outside its range");
    }
    var i: i64 = 0;
    while i < n {
        if !gl_finite(lower[i]) || !gl_finite(upper[i]) || !(lower[i] <= upper[i]) {
            return empty_arr(), fault(F_VALUE, "dual_annealing", "a bound is not finite, or a lower bound lies above its upper bound");
        }
        i += 1;
    }
    var r = default_rng(seed);
    var x: f64[>..] = [];
    i = 0;
    while i < n { x.push(lower[i] + random_uniform(r) * (upper[i] - lower[i])); i += 1; }
    var fx = F(copy(x));
    if !gl_finite(fx) {
        return empty_arr(), fault(F_VALUE, "dual_annealing", "the function gave a value that is not finite");
    }
    var best: f64[>..] = copy_slice(x);
    var fbest = fx;
    let qv: f64 = 2.62;
    let pi: f64 = 3.141592653589793;
    var t: i64 = 0;
    while t < maxiter * n {
        let temp = initial_temp * (mathshim::power(2.0, qv - 1.0) - 1.0) / (mathshim::power(2.0 + t, qv - 1.0) - 1.0);
        var cand: f64[>..] = [];
        i = 0;
        while i < n {
            let span = upper[i] - lower[i];
            let jump = mathshim::tan_of(pi * (random_uniform(r) - 0.5)) * span * (0.02 + 0.5 * mathshim::root(temp / initial_temp));
            var c = x[i] + jump;
            if c < lower[i] || c > upper[i] { c = lower[i] + random_uniform(r) * span; }
            cand.push(c);
            i += 1;
        }
        let fc = F(copy(cand));
        if !gl_finite(fc) {
            return empty_arr(), fault(F_VALUE, "dual_annealing", "the function gave a value that is not finite");
        }
        var accept = fc < fx;
        if !accept {
            let p = mathshim::power_e(-(fc - fx) / (temp + 1.0e-300));
            accept = random_uniform(r) < p;
        }
        if accept {
            x = copy_slice(cand);
            fx = fc;
            if fc < fbest {
                // A new best point goes to the local search at once.
                let lp, lf = gl_polish(cand, lower, upper, true) { v => F(copy(v)) };
                best = copy_slice(lp);
                fbest = lf;
            }
        }
        t += 1;
    }
    let pb, pf = gl_polish(best, lower, upper, true) { v => F(copy(v)) };
    var out: f64[>..] = copy_slice(pb);
    out.push(pf);
    return line_arr(copy(out)), no_fault();
}

// basinhopping finds the global minimum of a function from a start
// point. Each turn moves the point by a random step, runs a local
// search, and accepts the new minimum by the rule of Metropolis at the
// temperature T. The result holds the best point followed by its
// value as the last element. The local search is the method of Nelder
// and Mead.
fn basinhopping<F: fn(f64[:]) -> f64>(x0: f64[:], niter: i64 = 100, temperature: f64 = 1.0, stepsize: f64 = 0.5, seed: i64 = 1) -> Arr, Fault {
    let n = x0.len;
    if n < 1 || n > 1000 {
        return empty_arr(), fault(F_SHAPE, "basinhopping", "the start must hold one to 1000 variables");
    }
    if niter < 0 || niter > 1000000 || !(temperature >= 0.0) || !(stepsize > 0.0) {
        return empty_arr(), fault(F_RANGE, "basinhopping", "niter, T or the step size stands outside its range");
    }
    var i: i64 = 0;
    while i < n {
        if !gl_finite(x0[i]) {
            return empty_arr(), fault(F_VALUE, "basinhopping", "the start holds a value that is not finite");
        }
        i += 1;
    }
    let f0 = F(copy_slice(x0));
    if !gl_finite(f0) {
        return empty_arr(), fault(F_VALUE, "basinhopping", "the function gave a value that is not finite");
    }
    var r = default_rng(seed);
    let none: f64[>..] = [];
    let s0, sf0 = gl_polish(x0, none, none, false) { v => F(copy(v)) };
    var cur: f64[>..] = copy_slice(s0);
    var fcur = sf0;
    var best: f64[>..] = copy_slice(s0);
    var fbest = sf0;
    var k: i64 = 0;
    while k < niter {
        var trial: f64[>..] = [];
        i = 0;
        while i < n { trial.push(cur[i] + stepsize * (2.0 * random_uniform(r) - 1.0)); i += 1; }
        let tp, tf = gl_polish(trial, none, none, false) { v => F(copy(v)) };
        if !gl_finite(tf) {
            return empty_arr(), fault(F_VALUE, "basinhopping", "the function gave a value that is not finite");
        }
        var accept = tf < fcur;
        if !accept && temperature > 0.0 {
            accept = random_uniform(r) < mathshim::power_e(-(tf - fcur) / temperature);
        }
        if accept {
            cur = copy_slice(tp);
            fcur = tf;
            if tf < fbest { best = copy_slice(tp); fbest = tf; }
        }
        k += 1;
    }
    var out: f64[>..] = copy_slice(best);
    out.push(fbest);
    return line_arr(copy(out)), no_fault();
}
