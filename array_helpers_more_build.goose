// The build of an array from a block: fromfunction and fromiter.
//
// The caller passes the function as the block at the end of the call.
// The package holds no iterator and no function value, so a block takes
// the place of the function of NumPy. The time of a block is the time
// of the caller. The package cannot interrupt a block that does not
// stop. A value that the block gives passes into the array, NaN and
// the infinities too, as in NumPy.
namespace scientific;

// fromiter builds an array of one dimension from a block that gives the
// element at each place. The block takes the place as a whole number,
// from 0 to count minus 1. The count is required: NumPy reads an
// iterator to its end when no count is given, and the package holds no
// iterator to end. A count below one is a fault of kind F_VALUE, and a
// count above the limit of the elements is a fault of kind F_LIMIT.
fn fromiter<F: fn(i64) -> f64>(count: i64) -> Arr, Fault {
    if count < 1 {
        return empty_arr(), fault(F_VALUE, "fromiter", "the count holds a value below one");
    }
    if count > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "fromiter", "the count holds more than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < count { d.push(F(i)); i += 1; }
    var s: i64[..8] = [];
    s.push(count);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// fromfunction_call gives the value of the block at a point. The point
// is a view, so the block takes a slice on every version of the
// compiler.
fn fromfunction_call<F: fn(f64[:]) -> f64>(point: f64[:]) -> f64 { return F(point); }

// fromfunction builds an array of a shape from a block of the
// coordinates. The block takes the point as a row of floats, one value
// for each axis, as NumPy passes the coordinates as floats. NumPy calls
// its function once with whole arrays of coordinates. The package calls
// the block once for each element, so a block that reads more than its
// own point gives a different result. The shape follows the rules of
// every shape: at most eight dimensions, every length at least one,
// and at most the limit of the elements.
fn fromfunction<F: fn(f64[:]) -> f64>(shape: i64[:]) -> Arr, Fault {
    let fs = check_shape(shape);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    let total = numel(shape);
    var point: f64[>..] = [];
    var pos: i64[>..] = [];
    var k: i64 = 0;
    while k < shape.len { point.push(0.0); pos.push(0); k += 1; }
    var d: f64[>..] = [];
    var n: i64 = 0;
    while n < total {
        k = 0;
        while k < shape.len { point[k] = pos[k] as f64; k += 1; }
        d.push(fromfunction_call(point) { p => F(p) });
        // Count one up in row-major order.
        k = shape.len - 1;
        while k >= 0 {
            pos[k] += 1;
            if pos[k] < shape[k] { break; }
            pos[k] = 0;
            k -= 1;
        }
        n += 1;
    }
    var s: i64[..8] = [];
    k = 0;
    while k < shape.len { s.push(shape[k]); k += 1; }
    return Arr { ndim: shape.len, shape: s, data: d }, no_fault();
}
