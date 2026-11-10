// The integration of the values of a signal.
//
// The functions take the values of a signal and a spacing. A caller
// that needs the integral of a function samples the function first
// and hands the values over.
namespace scientific;

// trapezoid takes the integral of the values of a signal by the rule
// of the trapezoid. The spacing comes from the caller. A spacing of
// zero is a fault.
fn trapezoid(y: f64[:], dx: f64) -> f64, Fault {
    if dx == 0.0 {
        return 0.0, fault(F_VALUE, "trapezoid", "the spacing holds zero");
    }
    let n = y.len;
    if n < 2 {
        return 0.0, fault(F_VALUE, "trapezoid", "the signal holds fewer than two values");
    }
    var t: f64 = 0.0;
    var i: i64 = 1;
    while i < n {
        t += (y[i - 1] + y[i]) / 2.0;
        i += 1;
    }
    return t * dx, no_fault();
}

// simpson takes the integral of the values of a signal by the rule of
// Simpson. A signal of an even count of parts holds no middle point
// of the last part, and the rule of the trapezoid takes that part, as
// NumPy does.
fn simpson(y: f64[:], dx: f64) -> f64, Fault {
    if dx == 0.0 {
        return 0.0, fault(F_VALUE, "simpson", "the spacing holds zero");
    }
    let n = y.len;
    if n < 3 {
        return 0.0, fault(F_VALUE, "simpson", "the signal holds fewer than three values");
    }
    // The parts of the signal. A part is the space between two
    // values.
    let parts = n - 1;
    let pairs = parts / 2;
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < pairs {
        // The area of the parabola over the pair of parts.
        t += y[2 * i] + 4.0 * y[2 * i + 1] + y[2 * i + 2];
        i += 1;
    }
    t = t * dx / 3.0;
    // The last part of an even count takes the rule of the trapezoid.
    if parts % 2 == 1 {
        t += (y[n - 2] + y[n - 1]) / 2.0 * dx;
    }
    return t, no_fault();
}

// cumulative_trapezoid takes the integral of the values up to every
// point. The result holds one value fewer than the signal, as NumPy
// does.
fn cumulative_trapezoid(y: f64[:], dx: f64) -> Arr, Fault {
    if dx == 0.0 {
        return empty_arr(), fault(F_VALUE, "cumulative_trapezoid", "the spacing holds zero");
    }
    let n = y.len;
    if n < 2 {
        return empty_arr(), fault(F_VALUE, "cumulative_trapezoid", "the signal holds fewer than two values");
    }
    var d: f64[>..] = [];
    var t: f64 = 0.0;
    var i: i64 = 1;
    while i < n {
        t += (y[i - 1] + y[i]) / 2.0 * dx;
        d.push(t);
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// romberg takes the integral of the values of a signal on a grid of
// equal spacing by the extrapolation of Romberg. The count of the
// parts of the grid must be a power of two, as the method of the
// references holds. A grid that holds fewer than three points, or a
// count of the parts that is not a power of two, is a fault.
fn romberg(y: f64[:], dx: f64) -> f64, Fault {
    if dx == 0.0 {
        return 0.0, fault(F_VALUE, "romberg", "the spacing holds zero");
    }
    let n = y.len;
    if n < 3 {
        return 0.0, fault(F_VALUE, "romberg", "the grid holds fewer than three points");
    }
    let parts = n - 1;
    if !is_power_of_two_int(parts) {
        return 0.0, fault(F_VALUE, "romberg", "the count of the parts is not a power of two");
    }
    // The count of the levels of the grid.
    var levels: i64 = 0;
    var size = parts;
    while size > 1 {
        size /= 2;
        levels += 1;
    }
    // The rule of the trapezoid at every level of the grid. The
    // coarsest level holds one part.
    var t: f64[>..] = [];
    var i: i64 = 0;
    while i <= levels { t.push(0.0); i += 1; }
    t[0] = (y[0] + y[n - 1]) / 2.0 * dx * parts;
    var h = dx * parts;
    var step = parts;
    i = 1;
    while i <= levels {
        h = h / 2.0;
        step = step / 2;
        // The new points of this level stand at the odd multiples of
        // the step.
        var sum: f64 = 0.0;
        var k = step;
        while k < n - 1 {
            sum += y[k];
            k += step * 2;
        }
        t[i] = t[i - 1] / 2.0 + h * sum;
        i += 1;
    }
    // The table of the extrapolation. The row of a level holds the
    // values of that level, and the columns take the errors out.
    let width = levels + 1;
    var table: f64[>..] = [];
    i = 0;
    while i < width * width { table.push(0.0); i += 1; }
    i = 0;
    while i <= levels {
        table[i * width + 0] = t[i];
        i += 1;
    }
    i = 1;
    while i <= levels {
        var j: i64 = 1;
        while j <= i {
            let power = four_power(j);
            table[i * width + j] = (power * table[i * width + j - 1] - table[(i - 1) * width + j - 1]) / (power - 1.0);
            j += 1;
        }
        i += 1;
    }
    return table[levels * width + levels], no_fault();
}

// is_power_of_two_int tells whether a count holds a power of two.
fn is_power_of_two_int(n: i64) -> bool {
    if n < 1 { return false; }
    var v = n;
    while v % 2 == 0 { v /= 2; }
    return v == 1;
}

// four_power gives four to the power of a count.
fn four_power(k: i64) -> f64 {
    var p: f64 = 1.0;
    var i: i64 = 0;
    while i < k { p *= 4.0; i += 1; }
    return p;
}
