// The Clarkson-Woodruff transform of scipy.linalg, a sketch of a
// matrix by the sparse embedding. Each row of the input goes to one row
// of the result, with a sign of plus or minus one.
namespace scientific;

import std;

// clarkson_woodruff_transform gives the sketch S A of a matrix A of n
// rows. The sketch holds sketch_size rows and the columns of A. The
// matrix S holds one entry of plus or minus one in each column. The
// generator default_rng(seed) draws all n hashes first, each by
// integers(g, 0, sketch_size). It then draws all n signs, each by
// integers(g, 0, 2): the value 1 gives plus one. The same seed gives
// the same sketch. The stream differs from the stream of SciPy, so a
// seed gives another sketch there. A size below one is a fault, and so
// is a size whose result passes the element limit. A value that is not
// finite is a fault.
fn clarkson_woodruff_transform(a, sketch_size: i64, seed: i64 = 1) -> Arr, Fault {
    let tag = "clarkson_woodruff_transform";
    if a.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, tag, "the operand holds a count of dimensions that is not two");
    }
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    if sketch_size < 1 {
        return empty_arr(), fault(F_VALUE, tag, "the sketch size holds a value below one");
    }
    let n = a.shape[0];
    let d = a.shape[1];
    // The quotient holds the limit without a product that can overflow.
    if sketch_size > MAX_ELEMENTS / d {
        return empty_arr(), fault(F_LIMIT, tag, "the sketch holds more elements than the limit");
    }
    var i: i64 = 0;
    while i < a.data.len {
        let v = a.data[i];
        if !(v - v == 0.0) {
            return empty_arr(), fault(F_VALUE, tag, "the operand holds a value that is not finite");
        }
        i += 1;
    }
    var g = default_rng(seed);
    var h: i64[>..] = [];
    i = 0;
    while i < n {
        let v, f = integers(g, 0, sketch_size);
        h.push(v);
        i += 1;
    }
    var out: f64[>..] = [];
    i = 0;
    while i < sketch_size * d { out.push(0.0); i += 1; }
    i = 0;
    while i < n {
        let b, f = integers(g, 0, 2);
        var sg: f64 = -1.0;
        if b == 1 { sg = 1.0; }
        var j: i64 = 0;
        while j < d {
            out[h[i] * d + j] += sg * a.data[i * d + j];
            j += 1;
        }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(sketch_size);
    s.push(d);
    return Arr { ndim: 2, shape: s, data: out }, no_fault();
}
