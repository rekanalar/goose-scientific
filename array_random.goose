// The sampling.
//
// The generator comes from the standard library. It holds one word of
// state. A seed gives the same numbers on every run and on both
// supported versions of Goose.
namespace scientific;

import std;
import mathshim;

// Rng holds the state of the generator and the second of a pair of
// normal numbers, when one waits.
struct Rng { g: rng, cache: f64, holds_cache: bool }

// default_rng gives a generator from a seed. The same seed gives the
// same numbers.
fn default_rng(seed: i64) -> Rng {
    return Rng { g: rng { s: seed as u64 }, cache: 0.0, holds_cache: false };
}

// uniform gives a real number of the half-open range from a to b.
fn uniform(r: Rng&, a: f64, b: f64) -> f64 {
    return a + rand_flt(r.g) * (b - a);
}

// random_uniform gives a real number of the range zero to one. The
// name holds the word uniform, because the name random of NumPy is
// the name of the module of the standard library.
fn random_uniform(r: Rng&) -> f64 {
    return rand_flt(r.g);
}

// integers gives a whole number of the half-open range from low to
// high. A range that holds no number is a fault.
fn integers(r: Rng&, low: i64, high: i64) -> i64, Fault {
    let width = high - low;
    if width < 1 {
        return 0, fault(F_VALUE, "integers", "the range holds no number");
    }
    return low + rand_int(r.g, width), no_fault();
}

// normal gives a real number of the normal law. The mean and the
// spread come from the caller. The method is the pair of Box and
// Muller. The pair gives two numbers, and one waits in the cache for
// the next call. A spread below zero is a fault.
fn normal(r: Rng&, mean: f64, spread: f64) -> f64, Fault {
    if spread < 0.0 {
        return 0.0, fault(F_VALUE, "normal", "the spread holds a value below zero");
    }
    if r.holds_cache {
        r.holds_cache = false;
        return mean + spread * r.cache, no_fault();
    }
    let u1 = rand_flt(r.g);
    let u2 = rand_flt(r.g);
    // The length of the vector of the pair comes from the first
    // uniform. The angle comes from the second.
    let mag = mathshim::root(-2.0 * mathshim::ln(1.0 - u1));
    let angle = 6.283185307179586 * u2;
    r.cache = mag * mathshim::sine(angle);
    r.holds_cache = true;
    return mean + spread * mag * mathshim::cosine(angle), no_fault();
}

// exponential gives a real number of the exponential law. The scale
// comes from the caller. The method takes the inverse of the law. A
// scale below zero is a fault.
fn exponential(r: Rng&, scale: f64) -> f64, Fault {
    if scale < 0.0 {
        return 0.0, fault(F_VALUE, "exponential", "the scale holds a value below zero");
    }
    let u = rand_flt(r.g);
    return -mathshim::ln(1.0 - u) * scale, no_fault();
}

// poisson gives a count of events of a rate. The method of Knuth
// takes the product of the uniforms until it falls below the law of
// the rate. The method holds a small rate well. A rate above 30 is a
// fault.
fn poisson(r: Rng&, rate: f64) -> i64, Fault {
    if rate < 0.0 {
        return 0, fault(F_VALUE, "poisson", "the rate holds a value below zero");
    }
    if rate > 30.0 {
        return 0, fault(F_VALUE, "poisson", "the rate holds a value above 30");
    }
    if rate == 0.0 { return 0, no_fault(); }
    let limit = mathshim::power_e(-rate);
    var product: f64 = 1.0;
    var count: i64 = 0;
    while product > limit {
        product *= rand_flt(r.g);
        count += 1;
    }
    return count - 1, no_fault();
}

// choice picks elements of a row. A count below zero is a fault. A
// pick without a repeat that asks for more elements than the row
// holds is a fault.
fn choice(r: Rng&, a, count: i64, replace: bool) -> Arr, Fault {
    if a.ndim != 1 {
        return empty_arr(), fault(F_SHAPE, "choice", "the array holds a count of dimensions that is not one");
    }
    if count < 0 {
        return empty_arr(), fault(F_VALUE, "choice", "the count holds a value below zero");
    }
    if a.data.len == 0 {
        return empty_arr(), fault(F_VALUE, "choice", "the array holds no element");
    }
    var d: f64[>..] = [];
    if replace {
        var i: i64 = 0;
        while i < count {
            d.push(a.data[rand_int(r.g, a.data.len)]);
            i += 1;
        }
    } else {
        if count > a.data.len {
            return empty_arr(), fault(F_VALUE, "choice", "the count passes the count of the array");
        }
        // The positions come shuffled, and the picks take the first
        // ones.
        var pos: f64[>..] = [];
        var i: i64 = 0;
        while i < a.data.len { pos.push(i as f64); i += 1; }
        shuffle(pos, r.g);
        i = 0;
        while i < count {
            d.push(a.data[pos[i] as i64]);
            i += 1;
        }
    }
    var s: i64[..8] = [];
    s.push(count);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// permutation gives the order of a count. The order holds the whole
// numbers from zero to the count, shuffled.
fn permutation(r: Rng&, n: i64) -> Arr, Fault {
    if n < 0 {
        return empty_arr(), fault(F_VALUE, "permutation", "the count holds a value below zero");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n { d.push(i); i += 1; }
    shuffle(d, r.g);
    var s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// shuffled gives a copy of a row, shuffled. The array of the caller
// stays as it was.
fn shuffled(r: Rng&, a) -> Arr, Fault {
    if a.ndim != 1 {
        return empty_arr(), fault(F_SHAPE, "shuffled", "the array holds a count of dimensions that is not one");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { d.push(a.data[i]); i += 1; }
    shuffle(d, r.g);
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// shuffle_rows shuffles the rows of a matrix along its first axis in
// place. The array of the caller changes. A row moves as a whole.
fn shuffle_rows(r: Rng&, a) -> Fault {
    if a.ndim != 2 {
        return fault(F_SHAPE, "shuffle_rows", "the array holds a count of dimensions that is not two");
    }
    let rows = a.shape[0];
    let cols = a.shape[1];
    var i = rows - 1;
    while i > 0 {
        let j = rand_int(r.g, i + 1);
        if j != i {
            var c: i64 = 0;
            while c < cols {
                let t = a.data[i * cols + c];
                a.data[i * cols + c] = a.data[j * cols + c];
                a.data[j * cols + c] = t;
                c += 1;
            }
        }
        i -= 1;
    }
    return no_fault();
}
