// The rules that choose the count of bins of a histogram, and the
// weighted covariance.
//
// NumPy names the rules of the count of bins: auto, fd, sturges,
// doane, scott, rice, sqrt and stone. Each rule gives a width. The
// count of bins is the span of the data over the width, rounded up.
namespace scientific;

import mathshim;

// hist_width_fd gives the width of the rule of Freedman and Diaconis.
fn hist_width_fd(a) -> f64, Fault {
    let n = a.data.len;
    let q75, f1 = percentile(a, 75.0);
    if f1.kind != F_NONE { return 0.0, f1; }
    let q25, f2 = percentile(a, 25.0);
    if f2.kind != F_NONE { return 0.0, f2; }
    return 2.0 * (q75 - q25) * mathshim::power(n as f64, -1.0 / 3.0), no_fault();
}

// hist_width_sturges gives the width of the rule of Sturges.
fn hist_width_sturges(a, span: f64) -> f64 {
    return span / (mathshim::log2_of(a.data.len as f64) + 1.0);
}

// hist_width_sqrt gives the width of the rule of the square root.
fn hist_width_sqrt(a, span: f64) -> f64 {
    return span / mathshim::root(a.data.len as f64);
}

// hist_width_given gives the width of a rule by its number. The number
// is one for Sturges, two for Rice, three for the square root, four for
// Scott, five for Freedman and Diaconis and six for Doane.
fn hist_width_given(a, rule: i64, span: f64) -> f64, Fault {
    let n = a.data.len;
    if rule == 1 { return hist_width_sturges(a, span), no_fault(); }
    if rule == 2 { return span / (2.0 * mathshim::power(n as f64, 1.0 / 3.0)), no_fault(); }
    if rule == 3 { return hist_width_sqrt(a, span), no_fault(); }
    if rule == 4 {
        let sd, f = std(a, 0);
        if f.kind != F_NONE { return 0.0, f; }
        return mathshim::power(24.0 * mathshim::root(PI_VALUE) / n as f64, 1.0 / 3.0) * sd, no_fault();
    }
    if rule == 5 { return hist_width_fd(a); }
    if rule == 6 {
        if n <= 2 { return 0.0, no_fault(); }
        let sd, f = std(a, 0);
        if f.kind != F_NONE { return 0.0, f; }
        if !(sd > 0.0) { return 0.0, no_fault(); }
        let mu, f2 = mean(a);
        if f2.kind != F_NONE { return 0.0, f2; }
        var g1: f64 = 0.0;
        var i: i64 = 0;
        while i < n {
            let t = (a.data[i] - mu) / sd;
            g1 += t * t * t;
            i += 1;
        }
        g1 = g1 / n as f64;
        let sg1 = mathshim::root(6.0 * (n - 2) as f64 / ((n + 1) as f64 * (n + 3) as f64));
        return span / (1.0 + mathshim::log2_of(n as f64) + mathshim::log2_of(1.0 + mathshim::abs_of(g1) / sg1)), no_fault();
    }
    return 0.0, fault(F_VALUE, "histogram_rule", "the rule holds an unknown name");
}

// hist_width gives the width of a bin under a rule. The data lies
// between lo and hi. The rule takes the count of values.
fn hist_width(a, rule: const u8[:], lo: f64, hi: f64) -> f64, Fault {
    let span = hi - lo;
    if rule == "sturges" { return hist_width_sturges(a, span), no_fault(); }
    if rule == "rice" { return span / (2.0 * mathshim::power(a.data.len as f64, 1.0 / 3.0)), no_fault(); }
    if rule == "sqrt" { return hist_width_sqrt(a, span), no_fault(); }
    if rule == "scott" { return hist_width_given(a, 4, span); }
    if rule == "fd" { return hist_width_fd(a); }
    if rule == "doane" { return hist_width_given(a, 6, span); }
    if rule == "auto" {
        let fd = hist_width_fd(a);
        let sq = hist_width_sqrt(a, span);
        let st = hist_width_sturges(a, span);
        var corrected = fd;
        if sq / 2.0 > corrected { corrected = sq / 2.0; }
        if st < corrected { corrected = st; }
        return corrected, no_fault();
    }
    if rule == "stone" {
        if a.data.len <= 1 || span == 0.0 { return 0.0, no_fault(); }
        let bins = hist_stone_bins(a, lo, hi);
        return span / bins as f64, no_fault();
    }
    return 0.0, fault(F_VALUE, "histogram_rule", "the rule holds an unknown name");
}

// hist_stone_bins gives the count of bins that minimises the estimated
// squared error of the rule of Stone. The count lies between one and
// the larger of one hundred and the root of the count of values.
fn hist_stone_bins(a, lo: f64, hi: f64) -> i64 {
    let n = a.data.len;
    let span = hi - lo;
    var top: i64 = 100;
    let r = mathshim::root(n as f64) as i64;
    if r > top { top = r; }
    var best: i64 = 1;
    var bestv: f64 = 0.0;
    var b: i64 = 1;
    while b <= top {
        let v = hist_stone_jhat(a, lo, hi, b, n, span);
        if b == 1 || v < bestv { bestv = v; best = b; }
        b += 1;
    }
    return best;
}

// hist_stone_jhat gives the estimated squared error of a count of bins.
fn hist_stone_jhat(a, lo: f64, hi: f64, bins: i64, n: i64, span: f64) -> f64 {
    let hh = span / bins as f64;
    var dotsum: f64 = 0.0;
    var b: i64 = 0;
    while b < bins {
        let c = hist_count_in_bin(a, lo, hi, b, bins);
        let p = c / n as f64;
        dotsum += p * p;
        b += 1;
    }
    return (2.0 - (n + 1) as f64 * dotsum) / hh;
}

// hist_count_in_bin counts the values of one bin of a range.
fn hist_count_in_bin(a, lo: f64, hi: f64, bin: i64, bins: i64) -> f64 {
    let w = (hi - lo) / bins as f64;
    var c: f64 = 0.0;
    var i: i64 = 0;
    while i < a.data.len {
        let v = a.data[i];
        if v >= lo && v <= hi {
            var b = ((v - lo) / w) as i64;
            if b >= bins { b = bins - 1; }
            if b == bin { c += 1.0; }
        }
        i += 1;
    }
    return c;
}

// histogram_rule counts the values of an array in bins of equal width.
// The rule chooses the count of bins: "auto", "fd", "sturges", "doane",
// "scott", "rice", "sqrt" or "stone". The count is the span of the data
// over the width of the rule, rounded up. A width of zero gives one bin.
fn histogram_rule(a, rule: const u8[:]) -> Arr, Arr, Fault {
    if a.data.len < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "histogram_rule", "the array holds no value");
    }
    let lo, f = min(a);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let hi, f2 = max(a);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
    if hi == lo {
        return empty_arr(), empty_arr(), fault(F_VALUE, "histogram_rule", "every value holds the same number");
    }
    let w, f3 = hist_width(a, rule, lo, hi);
    if f3.kind != F_NONE { return empty_arr(), empty_arr(), f3; }
    var bins: i64 = 1;
    if w > 0.0 {
        bins = mathshim::ceil_of((hi - lo) / w) as i64;
        if bins < 1 { bins = 1; }
    }
    if bins > 1000000 {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "histogram_rule", "the count of bins is above the limit");
    }
    let counts, _, f4 = histogram(a, bins);
    if f4.kind != F_NONE { return empty_arr(), empty_arr(), f4; }
    // The edges run from the low value to the high value, as the rule
    // of NumPy sets them.
    var edges: f64[>..] = [];
    var i: i64 = 0;
    while i <= bins {
        if i == bins { edges.push(hi); }
        else { edges.push(lo + (hi - lo) * i as f64 / bins as f64); }
        i += 1;
    }
    var se: i64[..8] = [];
    se.push(bins + 1);
    return counts, Arr { ndim: 1, shape: se, data: copy(edges) }, no_fault();
}

// histogram_edges_rule gives the edges of the bins of a histogram under
// a rule.
fn histogram_edges_rule(a, rule: const u8[:]) -> Arr, Fault {
    let c, e, f = histogram_rule(a, rule);
    if f.kind != F_NONE { return empty_arr(), f; }
    return e, no_fault();
}

// cov_weights gives the covariance matrix of variables with weights. The
// frequency weights count the times each observation stands in, and the
// analysis weights weigh each observation. The weights take one value
// per observation. An empty list means no weight of that kind. The
// count of observations comes from the shape of the matrix.
fn cov_weights(m, fweights: f64[:] = [], aweights: f64[:] = [], rowvar: bool = true, ddof: i64 = 1) -> Arr, Fault {
    var x: f64[>..] = [];
    var nvar: i64 = 0;
    var nobs: i64 = 0;
    let f = vars_of(m, rowvar, "cov_weights", x, nvar, nobs);
    if f.kind != F_NONE { return empty_arr(), f; }
    if fweights.len > 0 && fweights.len != nobs {
        return empty_arr(), fault(F_SHAPE, "cov_weights", "the frequency weights do not fit the observations");
    }
    if aweights.len > 0 && aweights.len != nobs {
        return empty_arr(), fault(F_SHAPE, "cov_weights", "the analysis weights do not fit the observations");
    }
    var w: f64[>..] = [];
    var i: i64 = 0;
    while i < nobs {
        var v: f64 = 1.0;
        if fweights.len > 0 { v = fweights[i]; }
        if aweights.len > 0 { v = v * aweights[i]; }
        if !(v >= 0.0) {
            return empty_arr(), fault(F_VALUE, "cov_weights", "a weight holds a negative value");
        }
        w.push(v);
        i += 1;
    }
    if fweights.len > 0 {
        i = 0;
        while i < nobs {
            if fweights[i] != mathshim::trunc_of(fweights[i]) {
                return empty_arr(), fault(F_VALUE, "cov_weights", "a frequency weight holds no whole number");
            }
            i += 1;
        }
    }
    let weighted = fweights.len > 0 || aweights.len > 0;
    var wsum: f64 = 0.0;
    if weighted {
        i = 0;
        while i < nobs { wsum += w[i]; i += 1; }
    }
    // The mean of each variable.
    var mu: f64[>..] = [];
    var v: i64 = 0;
    while v < nvar {
        var t: f64 = 0.0;
        i = 0;
        while i < nobs {
            if weighted { t += x[v * nobs + i] * w[i]; }
            else { t += x[v * nobs + i]; }
            i += 1;
        }
        if weighted {
            if wsum == 0.0 { return empty_arr(), fault(F_VALUE, "cov_weights", "the weights sum to zero"); }
            mu.push(t / wsum);
        } else {
            mu.push(t / nobs as f64);
        }
        v += 1;
    }
    // The divisor of the result.
    var fact: f64 = 0.0;
    if !weighted { fact = nobs as f64 - ddof as f64; }
    else if ddof == 0 { fact = wsum; }
    else if aweights.len == 0 { fact = wsum - ddof as f64; }
    else {
        var wa: f64 = 0.0;
        i = 0;
        while i < nobs { wa += w[i] * aweights[i]; i += 1; }
        fact = wsum - ddof as f64 * wa / wsum;
    }
    if fact <= 0.0 { fact = 0.0; }
    // The result is the sum of the products of the centred values.
    var d: f64[>..] = [];
    var r: i64 = 0;
    while r < nvar {
        var c: i64 = 0;
        while c < nvar {
            var t: f64 = 0.0;
            i = 0;
            while i < nobs {
                let a1 = x[r * nobs + i] - mu[r];
                let b1 = x[c * nobs + i] - mu[c];
                if weighted { t += a1 * b1 * w[i]; }
                else { t += a1 * b1; }
                i += 1;
            }
            d.push(t / fact);
            c += 1;
        }
        r += 1;
    }
    var s: i64[..8] = [];
    s.push(nvar);
    s.push(nvar);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}
