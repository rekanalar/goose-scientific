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

// The size of one part of the rate of the Poisson law. The method of
// the law holds a small rate well.
const POISSON_RATE_MAX: f64 = 30.0;
// The largest rate of the Poisson law. A larger rate gives F_VALUE.
const POISSON_RATE_SUM_MAX: f64 = 1.0e6;

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
// the next call. A spread below zero is a fault. A mean or a spread
// that is not finite is a fault.
fn normal(r: Rng&, mean: f64, spread: f64) -> f64, Fault {
    if spread < 0.0 {
        return 0.0, fault(F_VALUE, "normal", "the spread holds a value below zero");
    }
    if mathshim::is_finite_of(mean) == 0.0 || mathshim::is_finite_of(spread) == 0.0 {
        return 0.0, fault(F_VALUE, "normal", "the mean or the spread is not finite");
    }
    if r.holds_cache {
        r.holds_cache = false;
        let c = mean + spread * r.cache;
        if mathshim::is_finite_of(c) == 0.0 {
            return 0.0, fault(F_VALUE, "normal", "the result is not finite");
        }
        return c, no_fault();
    }
    let u1 = rand_flt(r.g);
    let u2 = rand_flt(r.g);
    // The length of the vector of the pair comes from the first
    // uniform. The angle comes from the second.
    let mag = mathshim::root(-2.0 * mathshim::ln(1.0 - u1));
    let angle = 6.283185307179586 * u2;
    r.cache = mag * mathshim::sine(angle);
    r.holds_cache = true;
    let x = mean + spread * mag * mathshim::cosine(angle);
    if mathshim::is_finite_of(x) == 0.0 {
        return 0.0, fault(F_VALUE, "normal", "the result is not finite");
    }
    return x, no_fault();
}

// exponential gives a real number of the exponential law. The scale
// comes from the caller. The method takes the inverse of the law. A
// scale below zero is a fault. A scale that is not finite is a fault.
fn exponential(r: Rng&, scale: f64) -> f64, Fault {
    if scale < 0.0 {
        return 0.0, fault(F_VALUE, "exponential", "the scale holds a value below zero");
    }
    if mathshim::is_finite_of(scale) == 0.0 {
        return 0.0, fault(F_VALUE, "exponential", "the scale is not finite");
    }
    let u = rand_flt(r.g);
    let x = -mathshim::ln(1.0 - u) * scale;
    if mathshim::is_finite_of(x) == 0.0 {
        return 0.0, fault(F_VALUE, "exponential", "the result is not finite");
    }
    return x, no_fault();
}

// poisson gives a count of events of a rate. The method of Knuth
// takes the product of the uniforms until it falls below the law of
// the rate. The method holds a rate of 30 and below well, so a larger
// rate splits into parts of at most 30. The sum of Poisson numbers of
// rates that add up is a Poisson number of the sum. A rate above
// 1.0e6 is a fault. A rate that is not finite is a fault.
fn poisson(r: Rng&, rate: f64) -> i64, Fault {
    if rate < 0.0 {
        return 0, fault(F_VALUE, "poisson", "the rate holds a value below zero");
    }
    if mathshim::is_finite_of(rate) == 0.0 {
        return 0, fault(F_VALUE, "poisson", "the rate is not finite");
    }
    if rate > POISSON_RATE_SUM_MAX {
        return 0, fault(F_VALUE, "poisson", "the rate holds a value above 1.0e6");
    }
    if rate == 0.0 { return 0, no_fault(); }
    let parts = mathshim::floor_of((rate - 1.0e-9) / POISSON_RATE_MAX) + 1.0;
    let each = rate / parts;
    var total: i64 = 0;
    var part: i64 = 0;
    let count = parts as i64;
    while part < count {
        let limit = mathshim::power_e(-each);
        var product: f64 = 1.0;
        var k: i64 = 0;
        // A rate so small that the limit rounds to one never enters
        // the loop. The count of that part is zero.
        while product > limit {
            product *= rand_flt(r.g);
            k += 1;
        }
        if k > 0 { total += k - 1; }
        part += 1;
    }
    return total, no_fault();
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
        mix(pos, r.g);
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
    mix(d, r.g);
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
    mix(d, r.g);
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

// The names of NumPy for the generator. The package holds no global
// state, so seed returns a generator and the calls take the
// generator as the first argument.

// mix shuffles a buffer of numbers with the Fisher Yates loop of the
// standard library, so a seed gives the same order as the standard
// shuffle.
fn mix(d: f64[>..]&, g: rng&) {
    var i = d.len - 1;
    while i > 0 {
        let j = rand_int(g, i + 1);
        let t = d[i];
        d[i] = d[j];
        d[j] = t;
        i -= 1;
    }
}

// seed gives a generator from a seed. The same seed gives the same
// numbers. It is the name of NumPy for default_rng.
fn seed(s: i64) -> Rng {
    return default_rng(s);
}

// shuffle shuffles the elements of a one dimensional array in place.
// The array of the caller changes. The method is the Fisher Yates
// loop of the standard library, so a seed gives the same order.
fn shuffle(r: Rng&, a) -> Fault {
    if a.ndim != 1 {
        return fault(F_SHAPE, "shuffle", "the array holds a count of dimensions that is not one");
    }
    var i = a.data.len - 1;
    while i > 0 {
        let j = rand_int(r.g, i + 1);
        let t = a.data[i];
        a.data[i] = a.data[j];
        a.data[j] = t;
        i -= 1;
    }
    return no_fault();
}

// random_sample gives one real number of the range zero to one.
fn random_sample(r: Rng&) -> f64 {
    return random_uniform(r);
}

// rand_shape builds an array of a shape from a seed value of zero to
// one. The shape holds the length of every dimension.
fn rand_shape(r: Rng&, shape: i64[:], seed_kind: bool) -> Arr, Fault {
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), f; }
    let n = numel(shape);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        if seed_kind {
            let v, g = normal(r, 0.0, 1.0);
            if g.kind != F_NONE { return empty_arr(), g; }
            d.push(v);
        } else {
            d.push(uniform(r, 0.0, 1.0));
        }
        i += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < shape.len { s.push(shape[i]); i += 1; }
    return Arr { ndim: shape.len, shape: s, data: d }, no_fault();
}

// rand gives an array of uniform numbers of the range zero to one.
// The shape holds the length of every dimension. An empty shape is a
// fault.
fn rand(r: Rng&, shape: i64[:]) -> Arr, Fault {
    if shape.len < 1 {
        return empty_arr(), fault(F_VALUE, "rand", "the shape holds no dimension");
    }
    return rand_shape(r, shape, false);
}

// randn gives an array of numbers of the standard normal law.
fn randn(r: Rng&, shape: i64[:]) -> Arr, Fault {
    if shape.len < 1 {
        return empty_arr(), fault(F_VALUE, "randn", "the shape holds no dimension");
    }
    return rand_shape(r, shape, true);
}

// randint gives one whole number of the half-open range from low to
// high. A range that holds no number is a fault.
fn randint(r: Rng&, low: i64, high: i64) -> i64, Fault {
    return integers(r, low, high);
}

// rand_bytes gives count random bytes. The caller passes an empty
// buffer and the buffer of the caller holds the bytes. A count below
// one is a fault.
fn bytes(r: Rng&, count: i64, out: u8[>..]&) -> Fault {
    if count < 1 || count > MAX_ELEMENTS {
        return fault(F_RANGE, "bytes", "the count holds a value outside one to the limit");
    }
    var i: i64 = 0;
    while i < count {
        out.push(rand_int(r.g, 256) as u8);
        i += 1;
    }
    return no_fault();
}

// get_state gives the word of the state of the generator. The state
// holds one word, so the state of NumPy, which holds a key and a
// place, has no name here.
fn get_state(r: Rng&) -> i64 {
    return r.g.s as i64;
}

// set_state gives a generator from a word of state. A generator that
// set_state builds holds the same numbers as the generator that
// get_state read.
fn set_state(s: i64) -> Rng {
    return Rng { g: rng { s: s as u64 }, cache: 0.0, holds_cache: false };
}
