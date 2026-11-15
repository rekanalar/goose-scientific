// The global optimizer shgo of SciPy shgo.
//
// The method samples the box, holds a complex of the samples, and it
// refines the regions that hold a point near the best value with a
// local search inside the bounds. The sampling follows the sequence
// of Halton, as SciPy holds it with the setting sampling_method of
// halton. Each round adds samples and refines the regions again. The
// local search walks the directions of the coordinates, and it halts
// the step when no direction lowers the value.
namespace scientific;

import mathshim;

// The largest count of the dimensions of the box.
const SHG_MAX_DIMS: i64 = 16;

// The largest count of the samples of one round.
const SHG_MAX_SAMPLES: i64 = 100000;

// The largest count of the rounds.
const SHG_MAX_ITERS: i64 = 100;

// The largest count of the calls of the function.
const SHG_MAX_FEV: i64 = 1000000;

// The largest count of the regions that one round refines.
const SHG_MAX_LOCAL: i64 = 8;

// shg_base gives the base of the dimension k of the sequence of
// Halton. The bases are the first primes.
fn shg_base(k: i64) -> i64 {
    if k == 0 { return 2; }
    if k == 1 { return 3; }
    if k == 2 { return 5; }
    if k == 3 { return 7; }
    if k == 4 { return 11; }
    if k == 5 { return 13; }
    if k == 6 { return 17; }
    if k == 7 { return 19; }
    if k == 8 { return 23; }
    if k == 9 { return 29; }
    if k == 10 { return 31; }
    if k == 11 { return 37; }
    if k == 12 { return 41; }
    if k == 13 { return 43; }
    if k == 14 { return 47; }
    return 53;
}

// shg_halton gives the point of the sequence of Halton of the index i
// in the unit box of d dimensions. The point comes from the reversal
// of the digits of the index in the base of each dimension.
fn shg_halton(i: i64, d: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var k: i64 = 0;
    while k < d {
        out.push(halton_place(i, shg_base(k)));
        k += 1;
    }
    return out;
}

// shg_local runs the local search from a start point inside the
// bounds. The step walks the directions of the coordinates, and it
// halves when no direction lowers the value. The answer holds the
// point and the value.
fn shg_local<F: fn(f64[:]) -> f64>(x0: f64[:], lower: f64[:], upper: f64[:], fev: i64[>..]&) -> f64[>..], f64 {
    let d = x0.len;
    var x: f64[>..] = [];
    var i: i64 = 0;
    while i < d { x.push(x0[i]); i += 1; }
    var fx = F(copy(x));
    fev[0] += 1;
    var step: f64 = 0.125;
    var span: f64 = 0.0;
    i = 0;
    while i < d {
        let w = upper[i] - lower[i];
        if w > span { span = w; }
        i += 1;
    }
    if span <= 0.0 { span = 1.0; }
    while step > 1.0e-11 * span && fev[0] < SHG_MAX_FEV {
        var moved = false;
        var k: i64 = 0;
        while k < d {
            var way: i64 = -1;
            while way <= 1 {
                if way != 0 {
                    let old = x[k];
                    var t = old + (way as f64) * step;
                    if t < lower[k] { t = lower[k]; }
                    if t > upper[k] { t = upper[k]; }
                    if t != old {
                        x[k] = t;
                        let ft = F(copy(x));
                        fev[0] += 1;
                        if ft < fx {
                            fx = ft;
                            moved = true;
                        } else {
                            x[k] = old;
                        }
                    }
                }
                way += 2;
            }
            k += 1;
        }
        if !moved { step = step * 0.5; }
    }
    return x, fx;
}

// shgo finds the global minimum of a function in a box. The box holds
// the lower and the upper bounds of every variable. The count n holds
// the samples of one round, and the count iters holds the rounds. The
// answer holds the point of the least value. The sampling follows the
// sequence of Halton, and the regions near the best samples are
// refined with a local search that stays inside the bounds.
fn shgo<F: fn(f64[:]) -> f64>(lower: f64[:], upper: f64[:], n: i64 = 100, iters: i64 = 1) -> Arr, Fault {
    let d = lower.len;
    if d < 1 {
        return empty_arr(), fault(F_VALUE, "shgo", "the bounds hold no element");
    }
    if upper.len != d {
        return empty_arr(), fault(F_SHAPE, "shgo", "the bounds do not hold the same count");
    }
    if d > SHG_MAX_DIMS {
        return empty_arr(), fault(F_LIMIT, "shgo", "the box holds more dimensions than the limit");
    }
    if n < d + 1 {
        return empty_arr(), fault(F_VALUE, "shgo", "the count of the samples holds a value below the count of the variables plus one");
    }
    if n > SHG_MAX_SAMPLES {
        return empty_arr(), fault(F_LIMIT, "shgo", "the count of the samples holds a value above the limit");
    }
    if iters < 1 {
        return empty_arr(), fault(F_VALUE, "shgo", "the count of the rounds holds a value below one");
    }
    if iters > SHG_MAX_ITERS {
        return empty_arr(), fault(F_LIMIT, "shgo", "the count of the rounds holds a value above the limit");
    }
    var i: i64 = 0;
    while i < d {
        if !(lower[i] - lower[i] == 0.0) || !(upper[i] - upper[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "shgo", "a bound holds a value that is not finite");
        }
        if lower[i] > upper[i] {
            return empty_arr(), fault(F_VALUE, "shgo", "a lower bound lies above its upper bound");
        }
        i += 1;
    }
    var fev: i64[>..] = [];
    fev.push(0);
    var bestx: f64[>..] = [];
    var bestv: f64 = 0.0;
    var have = false;
    var round: i64 = 0;
    var at: i64 = 1;
    while round < iters {
        // The samples of the round, their values, and the places of
        // the best values.
        var pts: f64[>..] = [];
        var vals: f64[>..] = [];
        var count: i64 = 0;
        while count < n {
            let h = shg_halton(at, d);
            var p: f64[>..] = [];
            i = 0;
            while i < d { p.push(lower[i] + h[i] * (upper[i] - lower[i])); i += 1; }
            let v = F(copy(p));
            fev[0] += 1;
            i = 0;
            while i < d { pts.push(p[i]); i += 1; }
            vals.push(v);
            at += 1;
            count += 1;
        }
        // The places of the samples in the order of their values. A
        // simple walk finds the smallest values, because the count of
        // the samples that the search needs is small.
        var order: i64[>..] = [];
        count = 0;
        while count < n { order.push(count); count += 1; }
        i = 1;
        while i < n {
            let key = order[i];
            let kv = vals[key];
            var j = i - 1;
            while j >= 0 && vals[order[j]] > kv {
                order[j + 1] = order[j];
                j -= 1;
            }
            order[j + 1] = key;
            i += 1;
        }
        // The regions to refine: the best samples that stand apart
        // from each other. A sample near a better one holds no region
        // of its own.
        var span: f64 = 0.0;
        i = 0;
        while i < d {
            let w = upper[i] - lower[i];
            if w > span { span = w; }
            i += 1;
        }
        let cut = 0.03125 * span;
        var picks: i64[>..] = [];
        count = 0;
        while count < n && picks.len < SHG_MAX_LOCAL {
            let k = order[count];
            var near = false;
            var q: i64 = 0;
            while q < picks.len && !near {
                let pk = picks[q];
                var dist2: f64 = 0.0;
                i = 0;
                while i < d {
                    let t = pts[pk * d + i] - pts[k * d + i];
                    dist2 += t * t;
                    i += 1;
                }
                if dist2 < cut * cut { near = true; }
                q += 1;
            }
            if !near { picks.push(k); }
            count += 1;
        }
        // The best point of the samples holds a region of its own, so
        // the search never loses the first basin.
        count = 0;
        while count < n {
            if !have || vals[count] < bestv {
                bestv = vals[count];
                var p: f64[>..] = [];
                i = 0;
                while i < d { p.push(pts[count * d + i]); i += 1; }
                bestx = copy(p);
                have = true;
            }
            count += 1;
        }
        // The local search from every region.
        count = 0;
        while count < picks.len {
            var start: f64[>..] = [];
            i = 0;
            while i < d { start.push(pts[picks[count] * d + i]); i += 1; }
            let lx, lv = shg_local(start, lower, upper, fev, F);
            if !have || lv < bestv {
                bestv = lv;
                bestx = copy(lx);
                have = true;
            }
            count += 1;
        }
        round += 1;
    }
    if !have || fev[0] >= SHG_MAX_FEV {
        return empty_arr(), fault(F_LIMIT, "shgo", "the count of the calls ran out before the search ended");
    }
    return vec1(copy(bestx)), no_fault();
}
