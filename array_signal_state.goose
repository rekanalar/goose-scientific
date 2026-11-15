// The state space form of a system and the forms beside it.
//
// A system holds four parts: the square matrix A of the states, the
// input column b, the output row c and the direct part d. The package
// holds one input and one output, so b and c are vectors and d is one
// number. The calls of this file check the shapes of the four parts
// and give the zeros, the poles and the gain, as the calls of
// scipy.signal do. The forms of the coefficients live in the file of
// the forms of a filter.
namespace scientific;

import mathshim;

// The largest side of a matrix of states that the calls of the state
// space take. The exponential of a matrix holds the same limit.
const STATE_SIDE_MAX: i64 = 64;

// state_check holds the rules of a state space form: A is square of a
// side of one to the limit, b and c hold one value per state, and
// every value is finite.
fn state_check(a, b: f64[:], c: f64[:], d: f64, tag: const u8[:]) -> Fault {
    if a.ndim != 2 || a.shape.len != 2 || a.shape[0] != a.shape[1] || a.shape[0] < 1 {
        return fault(F_SHAPE, tag, "the matrix of the states is not square");
    }
    let n = a.shape[0];
    if n > STATE_SIDE_MAX {
        return fault(F_LIMIT, tag, "the matrix of the states holds a side above the limit");
    }
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fs.kind != F_NONE { return fs; }
    if b.len != n || c.len != n {
        return fault(F_SHAPE, tag, "the input or the output does not match the matrix of the states");
    }
    if !all_finite(a.data) || !all_finite(b) || !all_finite(c) || !all_finite([d]) {
        return fault(F_VALUE, tag, "a value is not finite");
    }
    return no_fault();
}

// abcd_normalize checks the shapes of the four parts of a state space
// form and gives every part back as a matrix of two dimensions. A
// holds the shape of n by n, b of n by one, c of one by n and d of one
// by one. Shapes that do not agree give a fault, as SciPy raises.
fn abcd_normalize(a, b: f64[:], c: f64[:], d: f64) -> Arr, Arr, Arr, Arr, Fault {
    let f = state_check(a, b, c, d, "abcd_normalize");
    if f.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), f;
    }
    let n = a.shape[0];
    var i: i64 = 0;
    var am: f64[>..] = [];
    while i < n * n { am.push(a.data[i]); i += 1; }
    var bm: f64[>..] = [];
    i = 0;
    while i < n { bm.push(b[i]); i += 1; }
    var cm: f64[>..] = [];
    i = 0;
    while i < n { cm.push(c[i]); i += 1; }
    var dm: f64[>..] = [];
    dm.push(d);
    return mat2(copy(am), n, n), mat2(copy(bm), n, 1), mat2(copy(cm), 1, n), mat2(copy(dm), 1, 1), no_fault();
}

// ss2zpk gives the zeros, the poles and the gain of a system from its
// state space form. It is tf2zpk of the numerator and the denominator
// of ss2tf, as SciPy holds it. The order of the roots is the order of
// the package, not the order of SciPy.
fn ss2zpk(a, b: f64[:], c: f64[:], d: f64) -> Arr, Arr, Arr, Arr, f64, Fault {
    let num, den, f = ss2tf(a, b, c, d);
    if f.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f;
    }
    return tf2zpk(num.data, den.data);
}
