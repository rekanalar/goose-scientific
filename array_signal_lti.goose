// The responses of the systems of one input and one output.
//
// A continuous system holds the four parts of the state space: the
// matrix A of the states, the input b, the output c and the direct
// part d. The impulse is the state moved by the input b over the
// exponential of the matrix. The step is the integral of the impulse.
// The simulation of lsim walks the input over the steps of the time.
//
// A discrete system holds two lists of coefficients in the powers of
// z, from the highest power to the lowest. The calls of the response
// turn them into the powers of one over z and read the response off
// the unit circle, as scipy.signal does. The magnitude of bode is in
// decibel and the phase is in degrees, smooth across the turns.
namespace scientific;

import mathshim;

// The largest count of the points of a response.
const LTI_GRID_MAX: i64 = 1000000;

// state_default_times gives the times of a response when the caller
// gives none. The span is seven times the slowest decay of the states,
// as SciPy holds it.
fn state_default_times(a, n: i64, tag: const u8[:]) -> Arr, Fault {
    let ev, f = eigvals_complex(a);
    if f.kind != F_NONE { return empty_arr(), f; }
    var r: f64 = 0.0;
    var first = true;
    var i: i64 = 0;
    while i < ev.data.len / 2 {
        let v = mathshim::abs_of(ev.data[2 * i]);
        if first || v < r { r = v; first = false; }
        i += 1;
    }
    if r == 0.0 { r = 1.0; }
    let tc = 1.0 / r;
    let t, f2 = linspace(0.0, 7.0 * tc, n);
    if f2.kind != F_NONE { return empty_arr(), f2; }
    return t, no_fault();
}

// state_free walks the state of a system with no input. It gives the
// state at every time of the list. The times may hold any spacing.
fn state_free(a, times: f64[:], x0: f64[:], out: f64[>..]&) -> Fault {
    let n = a.shape[0];
    // The walk reads the matrix and the state by their count, so the
    // buffer of the matrix must agree with its shape.
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, "state_free");
    if fa.kind != F_NONE { return fa; }
    if x0.len < n {
        return fault(F_SHAPE, "state_free", "the state holds fewer values than the matrix has rows");
    }
    var x = copy_slice(x0);
    var i: i64 = 0;
    while i < times.len {
        if i > 0 {
            // The state moves by the exponential of the step.
            var m: f64[>..] = [];
            var j: i64 = 0;
            while j < n * n { m.push(a.data[j] * (times[i] - times[i - 1])); j += 1; }
            let e, f = expm(mat2(copy(m), n, n));
            if f.kind != F_NONE { return f; }
            var nx: f64[>..] = [];
            j = 0;
            while j < n {
                var t: f64 = 0.0;
                var k: i64 = 0;
                while k < n { t += e.data[j * n + k] * x[k]; k += 1; }
                nx.push(t);
                j += 1;
            }
            x = copy(nx);
        }
        var j: i64 = 0;
        while j < n { out.push(x[j]); j += 1; }
        i += 1;
    }
    return no_fault();
}

// state_out reads the output of a state. The row c and the direct part
// d turn a state into one value.
fn state_out(c: f64[:], x: f64[:], at: i64, d: f64, u: f64) -> f64 {
    var y = d * u;
    var j: i64 = 0;
    while j < c.len { y += c[j] * x[at + j]; j += 1; }
    return y;
}

// state_hold gives the two parts of the exponential of one step of the
// hold: the matrix that moves the state and the part that the held
// input adds. The step sits in dt.
fn state_hold(a, b: f64[:], dt: f64, tag: const u8[:]) -> Arr, Arr, Fault {
    let n = a.shape[0];
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), fa; }
    if b.len < n {
        return empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the input holds fewer values than the matrix has rows");
    }
    // The block matrix holds A dt, b dt and a row and a column of zero.
    var m: f64[>..] = [];
    var i: i64 = 0;
    let w = n + 1;
    while i < w * w { m.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n { m[i * w + j] = a.data[i * n + j] * dt; j += 1; }
        m[i * w + n] = b[i] * dt;
        i += 1;
    }
    let e, f = expm(mat2(copy(m), w, w));
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    var ad: f64[>..] = [];
    var bd: f64[>..] = [];
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n { ad.push(e.data[i * w + j]); j += 1; }
        bd.push(e.data[i * w + n]);
        i += 1;
    }
    return mat2(copy(ad), n, n), flt_arr(copy(bd)), no_fault();
}

// lsim runs an input through a continuous system. The times hold the
// points of the output, the input holds one value per time. With
// interp true the input runs linear between the times, as SciPy does
// by default; with interp false the input holds its value over each
// step. The start holds the state at the first time. The results are
// the times, the outputs and the states, one state per time.
fn lsim(a, b: f64[:], c: f64[:], d: f64, u: f64[:], t: f64[:], x0: f64[:] = [], interp: bool = true) -> Arr, Arr, Arr, Fault {
    let f = state_check(a, b, c, d, "lsim");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
    let n = a.shape[0];
    if t.len < 1 || t[0] < 0.0 {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "lsim", "the times hold no value or start below zero");
    }
    if u.len != t.len {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "lsim", "the input does not hold one value per time");
    }
    if x0.len > 0 && x0.len != n {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "lsim", "the start does not match the states");
    }
    if !all_finite(u) || !all_finite(t) {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "lsim", "a value of the input or the times is not finite");
    }
    var start: f64[>..] = [];
    var i: i64 = 0;
    if x0.len == n {
        while i < n { start.push(x0[i]); i += 1; }
    } else {
        while i < n { start.push(0.0); i += 1; }
    }
    // The start moves to the first time with no input.
    var x: f64[>..] = [];
    if t[0] > 0.0 {
        var m: f64[>..] = [];
        i = 0;
        while i < n * n { m.push(a.data[i] * t[0]); i += 1; }
        let e, f2 = expm(mat2(copy(m), n, n));
        if f2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f2; }
        i = 0;
        while i < n {
            var s: f64 = 0.0;
            var j: i64 = 0;
            while j < n { s += e.data[i * n + j] * start[j]; j += 1; }
            x.push(s);
            i += 1;
        }
    } else {
        x = copy(start);
    }
    var xs: f64[>..] = [];
    var ys: f64[>..] = [];
    i = 0;
    while i < n { xs.push(x[i]); i += 1; }
    ys.push(state_out(c, x, 0, d, u[0]));
    if t.len == 1 {
        return flt_arr(copy_slice(t)), flt_arr(copy(ys)),
            Arr { ndim: 2, shape: [t.len, n], data: copy(xs) }, no_fault();
    }
    let dt = t[1] - t[0];
    if !(dt > 0.0) {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "lsim", "the times do not rise");
    }
    // The steps must hold one spacing, as SciPy asks.
    var i2: i64 = 2;
    while i2 < t.len {
        if mathshim::abs_of(t[i2] - t[i2 - 1] - dt) > 1.0e-9 * (1.0 + mathshim::abs_of(dt)) {
            return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "lsim", "the steps of the time do not hold one spacing");
        }
        i2 += 1;
    }
    // The parts of one step.
    var e11: f64[>..] = [];
    var p0: f64[>..] = [];
    var p1: f64[>..] = [];
    if interp {
        // The block matrix holds the step, the held value and the rise.
        let w = n + 2;
        var m: f64[>..] = [];
        i = 0;
        while i < w * w { m.push(0.0); i += 1; }
        i = 0;
        while i < n {
            var j: i64 = 0;
            while j < n { m[i * w + j] = a.data[i * n + j] * dt; j += 1; }
            m[i * w + n] = b[i] * dt;
            i += 1;
        }
        m[n * w + n + 1] = dt;
        let e, f2 = expm(mat2(copy(m), w, w));
        if f2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f2; }
        i = 0;
        while i < n {
            var j: i64 = 0;
            while j < n { e11.push(e.data[i * w + j]); j += 1; }
            p0.push(e.data[i * w + n] - e.data[i * w + n + 1] / dt);
            p1.push(e.data[i * w + n + 1] / dt);
            i += 1;
        }
    } else {
        let ad, bd, f2 = state_hold(a, b, dt, "lsim");
        if f2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f2; }
        i = 0;
        while i < n * n { e11.push(ad.data[i]); i += 1; }
        i = 0;
        while i < n { p0.push(bd.data[i]); p1.push(0.0); i += 1; }
    }
    var nx: f64[>..] = [];
    i = 1;
    while i < t.len {
        nx = [];
        var j: i64 = 0;
        while j < n {
            var s: f64 = 0.0;
            var k: i64 = 0;
            while k < n { s += e11[j * n + k] * x[k]; k += 1; }
            s += p0[j] * u[i - 1] + p1[j] * u[i];
            nx.push(s);
            j += 1;
        }
        x = copy(nx);
        var j2: i64 = 0;
        while j2 < n { xs.push(x[j2]); j2 += 1; }
        ys.push(state_out(c, x, 0, d, u[i]));
        i += 1;
    }
    return flt_arr(copy_slice(t)), flt_arr(copy(ys)),
        Arr { ndim: 2, shape: [t.len, n], data: xs }, no_fault();
}

// impulse gives the answer of a continuous system to one hit. The
// result holds the times and the values of the answer. When the
// caller gives no times, the span holds one hundred points over seven
// times the slowest decay of the states, as SciPy holds it.
fn impulse(a, b: f64[:], c: f64[:], d: f64, t: f64[:] = [], n: i64 = 100) -> Arr, Arr, Fault {
    let f = state_check(a, b, c, d, "impulse");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    if n < 1 || n > LTI_GRID_MAX {
        return empty_arr(), empty_arr(), fault(F_RANGE, "impulse", "the count of the points stands outside one to the limit");
    }
    var times = copy_slice(t);
    if times.len == 0 {
        let dt2, f2 = state_default_times(a, n, "impulse");
        if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
        times = copy_slice(dt2.data);
    }
    // The input is zero, so the state starts at b and runs free.
    var start: f64[>..] = [];
    var i: i64 = 0;
    while i < b.len { start.push(b[i]); i += 1; }
    var xs: f64[>..] = [];
    let fx = state_free(a, times, start, xs);
    if fx.kind != F_NONE { return empty_arr(), empty_arr(), fx; }
    var ys: f64[>..] = [];
    i = 0;
    while i < times.len { ys.push(state_out(c, xs, i * b.len, 0.0, 0.0)); i += 1; }
    return flt_arr(copy_slice(times)), flt_arr(copy(ys)), no_fault();
}

// step gives the answer of a continuous system to a held input of one.
// The result holds the times and the values of the answer. When the
// caller gives no times, the span holds one hundred points over seven
// times the slowest decay of the states, as SciPy holds it.
fn step(a, b: f64[:], c: f64[:], d: f64, t: f64[:] = [], n: i64 = 100) -> Arr, Arr, Fault {
    let f = state_check(a, b, c, d, "step");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    if n < 1 || n > LTI_GRID_MAX {
        return empty_arr(), empty_arr(), fault(F_RANGE, "step", "the count of the points stands outside one to the limit");
    }
    var times = copy_slice(t);
    if times.len == 0 {
        let dt2, f2 = state_default_times(a, n, "step");
        if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
        times = copy_slice(dt2.data);
    }
    // The state starts at zero and the input adds the integral of the
    // exponential over each span. The integral sits in the last column
    // of the exponential of the block matrix.
    let n2 = a.shape[0];
    var ys: f64[>..] = [];
    var i: i64 = 0;
    while i < times.len {
        var m: f64[>..] = [];
        let w = n2 + 1;
        var j: i64 = 0;
        while j < w * w { m.push(0.0); j += 1; }
        j = 0;
        while j < n2 {
            var k: i64 = 0;
            while k < n2 { m[j * w + k] = a.data[j * n2 + k] * times[i]; k += 1; }
            m[j * w + n2] = b[j] * times[i];
            j += 1;
        }
        let e, f2 = expm(mat2(copy(m), w, w));
        if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
        var x: f64[>..] = [];
        j = 0;
        while j < n2 { x.push(e.data[j * w + n2]); j += 1; }
        ys.push(state_out(c, x, 0, d, 1.0));
        i += 1;
    }
    return flt_arr(copy_slice(times)), flt_arr(copy(ys)), no_fault();
}

// sig_poly_asc gives the value of a polynomial with real coefficients
// at a complex point. The coefficients come from the lowest power to
// the highest.
fn sig_poly_asc(c: f64[:], z: Cplx) -> Cplx {
    var t = Cplx { re: 0.0, im: 0.0 };
    var i: i64 = c.len - 1;
    while i >= 0 {
        t = c_mul(t, z);
        t = c_add(t, Cplx { re: c[i], im: 0.0 });
        i -= 1;
    }
    return t;
}

// sig_phase_deg gives the phase of a response in degrees, smooth
// across the turns of the circle.
fn sig_phase_deg(hr: f64[:], hi: f64[:]) -> Arr, Fault {
    var ang: f64[>..] = [];
    var i: i64 = 0;
    while i < hr.len { ang.push(mathshim::arctangent2_of(hi[i], hr[i])); i += 1; }
    let u, f = unwrap(flt_arr(copy(ang)));
    if f.kind != F_NONE { return empty_arr(), f; }
    var i2: i64 = 0;
    while i2 < u.data.len { u.data[i2] = u.data[i2] * 180.0 / 3.141592653589793; i2 += 1; }
    return u, no_fault();
}

// freqresp gives the answer of a continuous system of two lists of
// coefficients over a grid of frequencies. The coefficients come from
// the highest power to the lowest. When the caller gives no grid, the
// grid holds two hundred points over the decades of the roots, as
// SciPy holds it. The results are the frequencies, the real parts and
// the imaginary parts of the answer.
fn freqresp(num: f64[:], den: f64[:], w: f64[:] = [], n: i64 = 200) -> Arr, Arr, Arr, Fault {
    // The coefficients go into the scale of the filter first, as the
    // form of a system does.
    var nb: f64[>..] = [];
    var na: f64[>..] = [];
    let f0 = form_normalize(num, den, "freqresp", nb, na);
    if f0.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f0; }
    var grid = copy_slice(w);
    if grid.len == 0 {
        let g, f = findfreqs(nb, na, n);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
        grid = copy_slice(g.data);
    }
    if grid.len > LTI_GRID_MAX {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_RANGE, "freqresp", "the grid holds more points than the limit");
    }
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    var i: i64 = 0;
    while i < grid.len {
        let h = c_div(sig_poly_at(nb, grid[i]), sig_poly_at(na, grid[i]));
        if !all_finite([h.re, h.im]) {
            return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "freqresp", "the answer is not finite at a point");
        }
        hr.push(h.re);
        hi.push(h.im);
        i += 1;
    }
    return flt_arr(copy(grid)), flt_arr(copy(hr)), flt_arr(copy(hi)), no_fault();
}

// bode gives the size and the phase of the answer of a continuous
// system of two lists of coefficients. The size is in decibel and the
// phase is in degrees, smooth across the turns. When the caller gives
// no grid, the grid holds one hundred points over the decades of the
// roots, as SciPy holds it.
fn bode(num: f64[:], den: f64[:], w: f64[:] = [], n: i64 = 100) -> Arr, Arr, Arr, Fault {
    let grid, hr, hi, f = freqresp(num, den, w, n);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
    var mag: f64[>..] = [];
    var i: i64 = 0;
    while i < hr.data.len {
        mag.push(20.0 * mathshim::log10_of(mathshim::hypot_of(hr.data[i], hi.data[i])));
        i += 1;
    }
    let phase, f2 = sig_phase_deg(hr.data, hi.data);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f2; }
    return grid, flt_arr(copy(mag)), phase, no_fault();
}

// sig_zinv_pair turns two lists of coefficients in the powers of z
// into lists of the same count in the powers of one over z. The
// shorter list gets zeros at the front, as SciPy holds it.
fn sig_zinv_pair(num: f64[:], den: f64[:], tag: const u8[:]) -> f64[>..], f64[>..], Fault {
    if num.len < 1 || den.len < 1 {
        return [], [], fault(F_VALUE, tag, "a list of coefficients holds no element");
    }
    if !all_finite(num) || !all_finite(den) {
        return [], [], fault(F_VALUE, tag, "a coefficient is not finite");
    }
    var np_: f64[>..] = [];
    var dp: f64[>..] = [];
    let diff = num.len - den.len;
    var i: i64 = 0;
    while i < -diff { np_.push(0.0); i += 1; }
    i = 0;
    while i < num.len { np_.push(num[i]); i += 1; }
    i = 0;
    while i < diff { dp.push(0.0); i += 1; }
    i = 0;
    while i < den.len { dp.push(den[i]); i += 1; }
    return np_, dp, no_fault();
}

// dfreqresp gives the answer of a discrete system of two lists of
// coefficients over a grid of frequencies. The coefficients come from
// the highest power of z to the lowest. The answer sits on the unit
// circle, from zero up to but not including pi. When the caller gives
// no grid, the grid holds ten thousand points, as SciPy holds it. The
// results are the frequencies, the real parts and the imaginary parts.
fn dfreqresp(num: f64[:], den: f64[:], w: f64[:] = [], n: i64 = 10000, whole: bool = false) -> Arr, Arr, Arr, Fault {
    // The coefficients go into the scale of the filter first, as the
    // form of a discrete system does.
    var nb: f64[>..] = [];
    var na: f64[>..] = [];
    let f1 = form_normalize(num, den, "dfreqresp", nb, na);
    if f1.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f1; }
    let np_, dp, f = sig_zinv_pair(nb, na, "dfreqresp");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
    var grid = copy_slice(w);
    if grid.len == 0 {
        let count = n;
        let f2 = sig_grid_fault(count, "dfreqresp");
        if f2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f2; }
        var top: f64 = 3.141592653589793;
        if whole { top = 6.283185307179586; }
        var g: f64[>..] = [];
        var i: i64 = 0;
        while i < count { g.push(top * i / count); i += 1; }
        grid = copy(g);
    }
    if grid.len > LTI_GRID_MAX {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_RANGE, "dfreqresp", "the grid holds more points than the limit");
    }
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    var i: i64 = 0;
    while i < grid.len {
        let z = Cplx { re: mathshim::cosine(grid[i]), im: -mathshim::sine(grid[i]) };
        let h = c_div(sig_poly_asc(np_, z), sig_poly_asc(dp, z));
        hr.push(h.re);
        hi.push(h.im);
        i += 1;
    }
    return flt_arr(copy(grid)), flt_arr(copy(hr)), flt_arr(copy(hi)), no_fault();
}

// dbode gives the size and the phase of the answer of a discrete
// system of two lists of coefficients. The size is in decibel and the
// phase is in degrees, smooth across the turns. The frequencies come
// back divided by the step of the sample. When the caller gives no
// grid, the grid holds one hundred points from zero up to but not
// including pi, as SciPy holds it.
fn dbode(num: f64[:], den: f64[:], w: f64[:] = [], n: i64 = 100, dt: f64 = 1.0) -> Arr, Arr, Arr, Fault {
    if !(dt > 0.0) {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "dbode", "the step of the sample holds a value of zero or below");
    }
    let grid, hr, hi, f = dfreqresp(num, den, w, n, false);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
    var wr: f64[>..] = [];
    var mag: f64[>..] = [];
    var i: i64 = 0;
    while i < grid.data.len {
        wr.push(grid.data[i] / dt);
        mag.push(20.0 * mathshim::log10_of(mathshim::hypot_of(hr.data[i], hi.data[i])));
        i += 1;
    }
    let phase, f2 = sig_phase_deg(hr.data, hi.data);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f2; }
    return flt_arr(copy(wr)), flt_arr(copy(mag)), phase, no_fault();
}

// disc_response reads the answer of a discrete system of two lists of
// coefficients to an input. The coefficients come from the highest
// power of z to the lowest, as scipy.signal holds them. The lists go
// into the powers of one over z, where the shorter list gets zeros at
// the front, and the answer follows the difference of the past values.
fn disc_response(num: f64[:], den: f64[:], u: f64[:], tag: const u8[:]) -> f64[>..], Fault {
    // A numerator longer than the denominator holds no state form.
    if num.len > den.len {
        return [], fault(F_SHAPE, tag, "the numerator holds more coefficients than the denominator");
    }
    // The form of the coefficients of the filter holds the rules and
    // the scale. The delay of the powers of one over z stays, because
    // the zeros at the front go back after the turn to that form.
    var nb: f64[>..] = [];
    var na: f64[>..] = [];
    let fn_ = form_normalize(num, den, tag, nb, na);
    if fn_.kind != F_NONE { return [], fn_; }
    // The form may drop the leading zeros of the denominator, so the
    // check of the length runs again.
    if nb.len > na.len {
        return [], fault(F_SHAPE, tag, "the numerator holds more coefficients than the denominator");
    }
    var np_: f64[>..] = [];
    var i: i64 = 0;
    while i < na.len - nb.len { np_.push(0.0); i += 1; }
    i = 0;
    while i < nb.len { np_.push(nb[i]); i += 1; }
    let m = np_.len;
    let k = na.len;
    var y: f64[>..] = [];
    var s: i64 = 0;
    while s < u.len {
        var v = np_[0] * u[s];
        i = 1;
        while i < m {
            if s - i >= 0 { v += np_[i] * u[s - i]; }
            i += 1;
        }
        i = 1;
        while i < k {
            if s - i >= 0 { v -= na[i] * y[s - i]; }
            i += 1;
        }
        y.push(v);
        s += 1;
    }
    return y, no_fault();
}

// dstep gives the answer of a discrete system to a held input of one.
// The coefficients come from the highest power of z to the lowest.
// The result holds the samples and the values of the answer. The count
// of the samples is n, as SciPy holds it.
fn dstep(num: f64[:], den: f64[:], n: i64 = 100, dt: f64 = 1.0) -> Arr, Arr, Fault {
    if n < 1 || n > LTI_GRID_MAX {
        return empty_arr(), empty_arr(), fault(F_RANGE, "dstep", "the count of the samples stands outside one to the limit");
    }
    if !(dt > 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "dstep", "the step of the sample holds a value of zero or below");
    }
    var u: f64[>..] = [];
    var i: i64 = 0;
    while i < n { u.push(1.0); i += 1; }
    let y, f = disc_response(num, den, u, "dstep");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    var t: f64[>..] = [];
    i = 0;
    while i < n { t.push(i * dt); i += 1; }
    return flt_arr(copy(t)), flt_arr(copy(y)), no_fault();
}

// dimpulse gives the answer of a discrete system to one hit at the
// first sample. The coefficients come from the highest power of z to
// the lowest. The result holds the samples and the values of the
// answer. The count of the samples is n, as SciPy holds it.
fn dimpulse(num: f64[:], den: f64[:], n: i64 = 100, dt: f64 = 1.0) -> Arr, Arr, Fault {
    if n < 1 || n > LTI_GRID_MAX {
        return empty_arr(), empty_arr(), fault(F_RANGE, "dimpulse", "the count of the samples stands outside one to the limit");
    }
    if !(dt > 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "dimpulse", "the step of the sample holds a value of zero or below");
    }
    var u: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        if i == 0 { u.push(1.0); } else { u.push(0.0); }
        i += 1;
    }
    let y, f = disc_response(num, den, u, "dimpulse");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    var t: f64[>..] = [];
    i = 0;
    while i < n { t.push(i * dt); i += 1; }
    return flt_arr(copy(t)), flt_arr(copy(y)), no_fault();
}
