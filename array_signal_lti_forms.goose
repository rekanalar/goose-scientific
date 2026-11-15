// The containers of a linear system and the discrete simulation.
//
// SciPy holds a linear system in a class. The package holds the three
// forms in three structs. A transfer function holds the numerator and
// the denominator, from the highest power to the lowest. A state space
// holds the four matrices A, B, C and D. A zeros poles gain system
// holds the roots and the scale; the roots sit in two complex arrays,
// the real part and the imaginary part interleaved. The field dt holds
// the step of the time. Zero names a continuous system.
//
// The calls follow SciPy 1.18.1: lti and dlti build the transfer
// function of a system, dlsim runs an input through a discrete state
// space system.
namespace scientific;

import mathshim;

// StateSpace holds the four matrices of a system of one input and one
// output and the step of the time. It replaces the class StateSpace of
// SciPy. The language holds one growable field in a struct, so the
// three matrices of the states, of the input and of the output sit in
// one buffer: the states take n times n values, the input takes n
// values and the output takes n values. The direct part is one number.
struct StateSpace {
    n: i64,
    dt: f64,
    d: f64,
    buf: f64[>..],
}

// TransferFunction holds the two polynomials of a system and the step
// of the time. It replaces the class TransferFunction of SciPy. The two
// polynomials sit in one buffer: the numerator takes num_len values and
// the denominator takes den_len values.
struct TransferFunction {
    num_len: i64,
    den_len: i64,
    dt: f64,
    buf: f64[>..],
}

// ZpkSystem holds the zeros and the poles of a system and the gain and
// the step of the time. The zeros and the poles sit in one complex
// array with the real and the imaginary parts interleaved: the zeros
// take zeros_len values and the poles take poles_len values. It
// replaces the class ZerosPolesGain of SciPy.
struct ZpkSystem {
    zeros_len: i64,
    poles_len: i64,
    gain: f64,
    dt: f64,
    buf: CArr,
}

// ltif_check checks the two lists of coefficients and puts them in the
// form that SciPy holds: it drops the leading zeros of the
// denominator, divides both lists by the first coefficient of the
// denominator, and drops the leading coefficients of the numerator
// whose size is at or below 1e-14. One coefficient of each list stays.
fn ltif_check(num: f64[:], den: f64[:], tag: const u8[:], nb: f64[>..]&, na: f64[>..]&) -> Fault {
    if num.len < 1 || den.len < 1 {
        return fault(F_VALUE, tag, "a list of coefficients holds no element");
    }
    if !all_finite(num) || !all_finite(den) {
        return fault(F_VALUE, tag, "a coefficient is not finite");
    }
    var i: i64 = 0;
    while i < den.len && den[i] == 0.0 { i += 1; }
    if i >= den.len {
        return fault(F_VALUE, tag, "the denominator holds only zeros");
    }
    let lead = den[i];
    var j = i;
    while j < den.len { na.push(den[j] / lead); j += 1; }
    var t: i64 = 0;
    while t < num.len - 1 && mathshim::abs_of(num[t] / lead) <= 1.0e-14 { t += 1; }
    j = t;
    while j < num.len { nb.push(num[j] / lead); j += 1; }
    if !all_finite(na) || !all_finite(nb) {
        return fault(F_VALUE, tag, "the division by the first coefficient overflows");
    }
    return no_fault();
}

// lti builds the transfer function of a continuous system from its
// numerator and its denominator. The coefficients come from the
// highest power to the lowest. The call stores them in the form that
// SciPy holds after normalize. The step of the time holds zero.
fn lti(num: f64[:], den: f64[:]) -> TransferFunction, Fault {
    var nb: f64[>..] = [];
    var na: f64[>..] = [];
    let f = ltif_check(num, den, "lti", nb, na);
    if f.kind != F_NONE {
        return TransferFunction { num_len: 0, den_len: 0, dt: 0.0, buf: [] }, f;
    }
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < nb.len { buf.push(nb[i]); i += 1; }
    i = 0;
    while i < na.len { buf.push(na[i]); i += 1; }
    return TransferFunction { num_len: nb.len, den_len: na.len, dt: 0.0, buf: buf }, no_fault();
}

// dlti builds the transfer function of a discrete system from its
// numerator and its denominator. The coefficients come from the
// highest power of z to the lowest. The step of the time holds one,
// as SciPy holds an unnamed step.
fn dlti(num: f64[:], den: f64[:]) -> TransferFunction, Fault {
    var nb: f64[>..] = [];
    var na: f64[>..] = [];
    let f = ltif_check(num, den, "dlti", nb, na);
    if f.kind != F_NONE {
        return TransferFunction { num_len: 0, den_len: 0, dt: 0.0, buf: [] }, f;
    }
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < nb.len { buf.push(nb[i]); i += 1; }
    i = 0;
    while i < na.len { buf.push(na[i]); i += 1; }
    return TransferFunction { num_len: nb.len, den_len: na.len, dt: 1.0, buf: buf }, no_fault();
}

// state_space builds a system of one input and one output. The matrix
// of the states must be square, the matrix of the input must hold one
// column and the matrix of the output must hold one row. The three
// matrices go into one buffer.
fn state_space(a, b, c, d, dt: f64) -> StateSpace, Fault {
    let tag = "state_space";
    if a.ndim != 2 || a.shape.len != 2 || a.shape[0] != a.shape[1] {
        return StateSpace { n: 0, dt: 0.0, d: 0.0, buf: [] },
            fault(F_SHAPE, tag, "the matrix of the states is not square");
    }
    let n = a.shape[0];
    if b.ndim != 2 || b.shape[0] != n || b.shape[1] != 1 {
        return StateSpace { n: 0, dt: 0.0, d: 0.0, buf: [] },
            fault(F_SHAPE, tag, "the matrix of the input does not match the states");
    }
    if c.ndim != 2 || c.shape[0] != 1 || c.shape[1] != n {
        return StateSpace { n: 0, dt: 0.0, d: 0.0, buf: [] },
            fault(F_SHAPE, tag, "the matrix of the output does not match the states");
    }
    if d.ndim != 2 || d.shape[0] != 1 || d.shape[1] != 1 {
        return StateSpace { n: 0, dt: 0.0, d: 0.0, buf: [] },
            fault(F_SHAPE, tag, "the direct part is not one value");
    }
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fa.kind != F_NONE { return StateSpace { n: 0, dt: 0.0, d: 0.0, buf: [] }, fa; }
    let fb = check_stack(b.ndim, b.shape, b.data.len, 1, tag);
    if fb.kind != F_NONE { return StateSpace { n: 0, dt: 0.0, d: 0.0, buf: [] }, fb; }
    let fc = check_stack(c.ndim, c.shape, c.data.len, 1, tag);
    if fc.kind != F_NONE { return StateSpace { n: 0, dt: 0.0, d: 0.0, buf: [] }, fc; }
    let fd = check_stack(d.ndim, d.shape, d.data.len, 1, tag);
    if fd.kind != F_NONE { return StateSpace { n: 0, dt: 0.0, d: 0.0, buf: [] }, fd; }
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { buf.push(a.data[i]); i += 1; }
    i = 0;
    while i < b.data.len { buf.push(b.data[i]); i += 1; }
    i = 0;
    while i < c.data.len { buf.push(c.data[i]); i += 1; }
    return StateSpace { n: n, dt: dt, d: d.data[0], buf: buf }, no_fault();
}

// dlsim runs an input through a discrete system of one input and one
// output and gives the times, the outputs and the states. The system
// holds the four matrices and the step of the time. With no times the
// input holds one value per step. With times the call reads the input
// at every time of the output by a linear fit between the times, as
// SciPy does. The start holds the state at the time zero; an empty
// start holds zero in every state. The states sit in one row per time.
fn dlsim(system, u: f64[:], t: f64[:] = [], x0: f64[:] = []) -> Arr, Arr, Arr, Fault {
    let n = system.n;
    if n < 1 {
        return empty_arr(), empty_arr(), empty_arr(),
            fault(F_SHAPE, "dlsim", "the system holds no state");
    }
    // The buffer holds the states, then the input, then the output.
    if system.buf.len != n * n + 2 * n {
        return empty_arr(), empty_arr(), empty_arr(),
            fault(F_SHAPE, "dlsim", "the buffer of the system does not hold the matrices");
    }
    if !all_finite(system.buf) || system.d - system.d != 0.0 {
        return empty_arr(), empty_arr(), empty_arr(),
            fault(F_VALUE, "dlsim", "a value of the system is not finite");
    }
    let dt = system.dt;
    if !(dt > 0.0) || dt - dt != 0.0 {
        return empty_arr(), empty_arr(), empty_arr(),
            fault(F_VALUE, "dlsim", "the step of the time holds zero, a value below zero or one that is not finite");
    }
    if u.len < 1 || !all_finite(u) {
        return empty_arr(), empty_arr(), empty_arr(),
            fault(F_VALUE, "dlsim", "the input holds no value or a value that is not finite");
    }
    if x0.len > 0 && x0.len != n {
        return empty_arr(), empty_arr(), empty_arr(),
            fault(F_SHAPE, "dlsim", "the start does not match the states");
    }
    if x0.len == n && !all_finite(x0) {
        return empty_arr(), empty_arr(), empty_arr(),
            fault(F_VALUE, "dlsim", "the start holds a value that is not finite");
    }
    var tout: f64[>..] = [];
    var udt: f64[>..] = [];
    var out: i64 = 0;
    if t.len == 0 {
        out = u.len;
        let stop = (out - 1) as f64 * dt;
        let g, f = linspace(0.0, stop, out);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
        tout = copy_slice(g.data);
        var i: i64 = 0;
        while i < u.len { udt.push(u[i]); i += 1; }
    } else {
        if t.len != u.len {
            return empty_arr(), empty_arr(), empty_arr(),
                fault(F_SHAPE, "dlsim", "the input does not hold one value per time");
        }
        if !all_finite(t) {
            return empty_arr(), empty_arr(), empty_arr(),
                fault(F_VALUE, "dlsim", "a time is not finite");
        }
        var i: i64 = 1;
        while i < t.len {
            if !(t[i] > t[i - 1]) {
                return empty_arr(), empty_arr(), empty_arr(),
                    fault(F_VALUE, "dlsim", "the times do not rise");
            }
            i += 1;
        }
        let stop = t[t.len - 1];
        if stop < 0.0 {
            return empty_arr(), empty_arr(), empty_arr(),
                fault(F_VALUE, "dlsim", "the last time holds a value below zero");
        }
        out = (stop / dt) as i64 + 1;
        if out > LTI_GRID_MAX {
            return empty_arr(), empty_arr(), empty_arr(),
                fault(F_LIMIT, "dlsim", "the count of the steps is above the limit");
        }
        let g, f = linspace(0.0, stop, out);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
        tout = copy_slice(g.data);
        // The input follows a linear fit between the times. Outside the
        // times the fit runs on with the slope of the nearest part, as
        // the spline of SciPy does.
        var k: i64 = 0;
        var seg: i64 = 0;
        while k < out {
            let x = tout[k];
            while seg < t.len - 2 && t[seg + 1] < x { seg += 1; }
            let dt2 = t[seg + 1] - t[seg];
            let r = (x - t[seg]) / dt2;
            udt.push(u[seg] + (u[seg + 1] - u[seg]) * r);
            k += 1;
        }
    }
    // The states walk the steps. The output reads one state per time.
    // The read of the row before the write stays behind the end of the
    // buffer, so one buffer holds the whole walk.
    var xs: f64[>..] = [];
    var i: i64 = 0;
    if x0.len == n {
        while i < n { xs.push(x0[i]); i += 1; }
    } else {
        while i < n { xs.push(0.0); i += 1; }
    }
    var s: i64 = 1;
    while s < out {
        i = 0;
        while i < n {
            var v = system.buf[n * n + i] * udt[s - 1];
            var j: i64 = 0;
            while j < n { v += system.buf[i * n + j] * xs[(s - 1) * n + j]; j += 1; }
            xs.push(v);
            i += 1;
        }
        s += 1;
    }
    var ys: f64[>..] = [];
    s = 0;
    while s < out {
        var v = system.d * udt[s];
        var j: i64 = 0;
        while j < n { v += system.buf[n * n + n + j] * xs[s * n + j]; j += 1; }
        ys.push(v);
        s += 1;
    }
    var sh: i64[..8] = [];
    sh.push(out);
    sh.push(n);
    return flt_arr(copy(tout)), flt_arr(copy(ys)),
        Arr { ndim: 2, shape: sh, data: xs }, no_fault();
}
