// The cubature of a function of a vector over a box.
//
// The call follows cubature of SciPy with the rule of Gauss and
// Kronrod. The package nests the adaptive rule of quad over the box,
// as nquad does, so the estimate of the error differs from the
// estimate of SciPy. The value agrees to the tolerance.
//
namespace scientific;

// cubature integrates the function of the caller over the box with
// the corners a and b. The block takes the point as a list of the
// values of the variables and gives one number. The first value of
// the point belongs to the first pair of the corners. An end may be
// infinite. The count n bounds the intervals of every range. The call
// gives the value and the estimate of the error.
fn cubature<F: fn(f64[:]) -> f64>(a: f64[:], b: f64[:], n: i64 = 1000) -> f64, f64, Fault {
    if a.len != b.len || a.len < 1 {
        return 0.0, 0.0, fault(F_SHAPE, "cubature", "the corners hold a different count of values or no value");
    }
    if a.len > MAX_NQUAD_RANGES {
        return 0.0, 0.0, fault(F_LIMIT, "cubature", "the box holds more than eight dimensions");
    }
    var ranges: f64[>..] = [];
    var i: i64 = 0;
    while i < a.len {
        if !(a[i] - a[i] == 0.0) || !(b[i] - b[i] == 0.0) {
            return 0.0, 0.0, fault(F_VALUE, "cubature", "a corner holds a value that is not finite");
        }
        ranges.push(a[i]);
        ranges.push(b[i]);
        i += 1;
    }
    // The tolerances hold the defaults of SciPy: the relative error of
    // eight digits and no absolute tolerance.
    return nquad(ranges, 0.0, 1.0e-8, n) { point => F(point) };
}
