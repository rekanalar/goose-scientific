// The set-up of the ODE solver: the tableaus, the events and the dense output rules.
//
namespace scientific;

import mathshim;

// ivp_rms gives the root of the mean square of a list.
fn ivp_rms(v: f64[:]) -> f64 {
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < v.len { s += v[i] * v[i]; i += 1; }
    return mathshim::root(s / v.len);
}

// ivp_tableau fills the coefficients of a method. The count of stages
// leaves out the stage of the new point. The method RK45 is the pair
// of Dormand and Prince. The method RK23 is the pair of Bogacki and
// Shampine. The method DOP853 is the pair of Dormand and Prince of
// the order eight. It holds two error vectors.
fn ivp_tableau(kind: i32, c: f64[>..]&, a: f64[>..]&, b: f64[>..]&, e: f64[>..]&, e3: f64[>..]&) -> i64 {
    if kind == 2 {
        dop853_c(c);
        dop853_a(a);
        dop853_b(b);
        dop853_e5(e);
        dop853_e3(e3);
        return 12;
    }
    if kind == 0 {
        c.push(0.0); c.push(0.2); c.push(0.3); c.push(0.8); c.push(8.0 / 9.0); c.push(1.0);
        let rows: f64[36] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
            3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0,
            44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0,
            19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0,
            9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0];
        var i: i64 = 0;
        while i < 36 { a.push(rows[i]); i += 1; }
        b.push(35.0 / 384.0); b.push(0.0); b.push(500.0 / 1113.0); b.push(125.0 / 192.0); b.push(-2187.0 / 6784.0); b.push(11.0 / 84.0);
        e.push(-71.0 / 57600.0); e.push(0.0); e.push(71.0 / 16695.0); e.push(-71.0 / 1920.0); e.push(17253.0 / 339200.0); e.push(-22.0 / 525.0); e.push(1.0 / 40.0);
        return 6;
    }
    c.push(0.0); c.push(0.5); c.push(0.75);
    let rows: f64[9] = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.75, 0.0];
    var i: i64 = 0;
    while i < 9 { a.push(rows[i]); i += 1; }
    b.push(2.0 / 9.0); b.push(1.0 / 3.0); b.push(4.0 / 9.0);
    e.push(5.0 / 72.0); e.push(-1.0 / 12.0); e.push(-1.0 / 9.0); e.push(1.0 / 8.0);
    return 3;
}

// solve_ivp integrates the system y' = f(t, y) from t0 to t1 with the
// start y0. The method is "RK45" or "RK23". The step follows the
// control of SciPy, so the steps equal the steps of SciPy. The result
// holds the times of the steps in one array and the values in a
// matrix of one row per time. The tolerances hold the relative and the
// absolute error of one step. A count of steps above 1000000 is a
// fault.
fn solve_ivp<F: fn(f64, f64[:]) -> f64[:]>(t0: f64, t1: f64, y0: f64[:], rtol: f64 = 0.001, atol: f64 = 1.0e-6, method: const u8[:] = "RK45", max_step: f64 = 1.0e308, t_eval: f64[:] = [], first_step: f64 = 0.0) -> Arr, Arr, Fault {
    let t, y, et, ei, ey, dn, f = ivp_run(t0, t1, y0, rtol, atol, method, max_step, t_eval, 0, [], [], false, first_step, false) { tt, yy => F(tt, copy(yy)) };
    return t, y, f;
}

// solve_ivp_jac integrates like solve_ivp with the methods BDF and
// Radau, and takes the Jacobian from the caller. The block gets the
// time, the state and a mode. The mode zero asks for the derivative of
// n values. The mode one asks for the Jacobian: n times n values in
// row-major order, where row i holds the partial derivatives of the
// component i. The methods RK45, RK23 and DOP853 ignore the Jacobian.
fn solve_ivp_jac<G: fn(f64, f64[:], i64) -> f64[:]>(t0: f64, t1: f64, y0: f64[:], rtol: f64 = 0.001, atol: f64 = 1.0e-6, method: const u8[:] = "BDF", max_step: f64 = 1.0e308, t_eval: f64[:] = [], first_step: f64 = 0.0) -> Arr, Arr, Fault {
    let n = y0.len;
    let t, y, et, ei, ey, dn, f = ivp_run(t0, t1, y0, rtol, atol, method, max_step, t_eval, 0, [], [], false, first_step, true) { tt, yy => if yy.len == n + 1 { G(tt, yy[0..n], 1) } else { G(tt, copy(yy), 0) } };
    return t, y, f;
}

// solve_ivp_events integrates like solve_ivp and finds the events. The
// block gives the derivative first and then nev event functions. An
// event happens where its function changes sign. The array terminal
// flags the events that end the run. The array direction holds one
// value per event: above zero for a rise, below zero for a fall, zero
// for both. It gives the times and the states, the times, the numbers
// and the states of the events. A terminal event ends the run at its
// time.
fn solve_ivp_events<F: fn(f64, f64[:]) -> f64[:]>(t0: f64, t1: f64, y0: f64[:], nev: i64, terminal: bool[:], direction: f64[:], rtol: f64 = 0.001, atol: f64 = 1.0e-6, method: const u8[:] = "RK45", max_step: f64 = 1.0e308, t_eval: f64[:] = [], first_step: f64 = 0.0) -> Arr, Arr, Arr, Arr, Arr, Fault {
    if nev < 1 || nev > 64 || terminal.len != nev || direction.len != nev {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "solve_ivp_events", "the flags and the directions do not hold one value per event");
    }
    let t, y, et, ei, ey, dn, f = ivp_run(t0, t1, y0, rtol, atol, method, max_step, t_eval, nev, terminal, direction, false, first_step, false) { tt, yy => F(tt, copy(yy)) };
    return t, y, et, ei, ey, f;
}

// solve_ivp_dense integrates like solve_ivp and keeps the polynomial of
// each step. It gives the times, the states and the dense output. Read
// the dense output with dense_eval.
fn solve_ivp_dense<F: fn(f64, f64[:]) -> f64[:]>(t0: f64, t1: f64, y0: f64[:], rtol: f64 = 0.001, atol: f64 = 1.0e-6, method: const u8[:] = "RK45", max_step: f64 = 1.0e308, first_step: f64 = 0.0) -> Arr, Arr, Dense, Fault {
    let t, y, et, ei, ey, dn, f = ivp_run(t0, t1, y0, rtol, atol, method, max_step, [], 0, [], [], true, first_step, false) { tt, yy => F(tt, copy(yy)) };
    return t, y, dn, f;
}

// dense_eval gives the states of a dense output at the times x. A time
// outside the steps takes the polynomial of the nearest step. The
// result holds one row per time.
fn dense_eval(d: Dense&, x: f64[:]) -> Arr, Fault {
    if d.steps < 1 {
        return empty_arr(), fault(F_VALUE, "dense_eval", "the dense output holds no step");
    }
    let st = d.steps;
    var rows = d.ns + 1;
    if d.kind == 2 { rows = 16; }
    if d.kind == 3 { rows = 9; }
    if d.kind == 4 { rows = 3; }
    var ht: f64[>..] = [];
    var hh: f64[>..] = [];
    var hy: f64[>..] = [];
    var hk: f64[>..] = [];
    var q: i64 = 0;
    while q < st { ht.push(d.buf[q]); hh.push(d.buf[st + q]); q += 1; }
    q = 0;
    while q < st * d.n { hy.push(d.buf[2 * st + q]); q += 1; }
    q = 0;
    while q < st * rows * d.n { hk.push(d.buf[2 * st + st * d.n + q]); q += 1; }
    let t, y, f = ivp_dense(x, ht, hh, hy, hk, d.kind, d.ns, d.n);
    return y, f;
}

// ivp_pm gives the matrix of the dense polynomial of a method and the
// count of its powers.
fn ivp_pm(rk45: bool) -> f64[>..], i64 {
    if rk45 {
        let pm: f64[>..] = [1.0, -2.8535800653862835, 3.0717434641059005, -1.1270175653862835,
              0.0, 0.0, 0.0, 0.0,
              0.0, 4.023133379230305, -6.249321565289, 2.675424484351598,
              0.0, -3.7324019615885042, 10.068970589843675, -5.685526961588504,
              0.0, 2.5548038301849423, -6.399112377351017, 3.5219323679207912,
              0.0, -1.3744241142186024, 3.272657752246729, -1.7672812570757455,
              0.0, 1.3824689317781436, -3.764937863556287, 2.382468931778144];
        return pm, 4;
    }
    let pm: f64[>..] = [1.0, -1.3333333333333333, 0.5555555555555556,
              0.0, 1.0, -0.6666666666666666,
              0.0, 1.3333333333333333, -0.8888888888888888,
              0.0, -1.0, 1.0];
    return pm, 3;
}

// ivp_point gives the state inside one step, at the fraction xs of the
// step. The stages of the step are in k, and the last derivative is in
// fl.
fn ivp_point(rk45: bool, ns: i64, n: i64, y: f64[:], h: f64, k: f64[:], fl: f64[:], xs: f64) -> f64[>..] {
    let pm, pc = ivp_pm(rk45);
    let rows = ns + 1;
    var pw: f64[>..] = [];
    var pp: f64 = 1.0;
    var c: i64 = 0;
    while c < pc { pp = pp * xs; pw.push(pp); c += 1; }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var acc: f64 = 0.0;
        c = 0;
        while c < pc {
            var q: f64 = 0.0;
            var r: i64 = 0;
            while r < rows {
                var kv: f64 = 0.0;
                if r < ns { kv = k[r * n + i]; } else { kv = fl[i]; }
                q += kv * pm[r * pc + c];
                r += 1;
            }
            acc += q * pw[c];
            c += 1;
        }
        out.push(y[i] + h * acc);
        i += 1;
    }
    return out;
}

// ivp_point_kind gives the state inside one step, at the fraction xs
// of the step. The rows of the stages are in kall. The method of the
// order eight uses its own interpolation.
fn ivp_point_kind(kind: i32, ns: i64, n: i64, y: f64[:], ynew: f64[:], h: f64, kall: f64[:], xs: f64) -> f64[>..] {
    if kind == 2 {
        return dop853_point(n, y, ynew, h, kall, xs);
    }
    if kind == 4 {
        return rad_dense(n, y, kall, xs);
    }
    if kind == 3 {
        return bdf_poly(n, kall, kall[6 * n] + (xs - 1.0) * h);
    }
    var k: f64[>..] = [];
    var fl: f64[>..] = [];
    var i: i64 = 0;
    while i < ns * n { k.push(kall[i]); i += 1; }
    i = 0;
    while i < n { fl.push(kall[ns * n + i]); i += 1; }
    return ivp_point(kind == 0, ns, n, y, h, k, fl, xs);
}

// dop853_point gives the state inside one step of the method of the
// order eight, at the fraction xs of the step. The rows of the stages
// are in k. The first row stands at the start of the step, the row of
// the number twelve holds the derivative at the end of the step and
// the last three rows serve the interpolation.
fn dop853_point(n: i64, y: f64[:], ynew: f64[:], h: f64, k: f64[:], xs: f64) -> f64[>..] {
    var dm: f64[>..] = [];
    dop853_d(dm);
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        let dy = ynew[i] - y[i];
        let fo = k[i];
        let fnv = k[12 * n + i];
        // The four rows of the matrix of the interpolation.
        var f3: f64 = 0.0;
        var f4: f64 = 0.0;
        var f5: f64 = 0.0;
        var f6: f64 = 0.0;
        var r: i64 = 0;
        while r < 4 {
            var q: f64 = 0.0;
            var j: i64 = 0;
            while j < 16 { q += k[j * n + i] * dm[r * 16 + j]; j += 1; }
            if r == 0 { f3 = q; } else if r == 1 { f4 = q; } else if r == 2 { f5 = q; } else { f6 = q; }
            r += 1;
        }
        var acc = h * f6;
        acc = acc * xs + h * f5;
        acc = acc * (1.0 - xs) + h * f4;
        acc = acc * xs + h * f3;
        acc = acc * (1.0 - xs) + 2.0 * dy - h * (fnv + fo);
        acc = acc * xs + h * fo - dy;
        acc = acc * (1.0 - xs) + dy;
        acc = acc * xs;
        out.push(y[i] + acc);
        i += 1;
    }
    return out;
}

// Dense holds the steps of a run. It gives the values between the
// steps with the polynomial of each step. The buffer holds the step
// starts, the step sizes, the states and the stages, one after the
// other.
struct Dense {
    kind: i32,
    ns: i64,
    n: i64,
    steps: i64,
    buf: f64[>..]
}

// dense_none gives a dense output that holds no step.
fn dense_none() -> Dense {
    return Dense { kind: 0, ns: 0, n: 0, steps: 0, buf: [] };
}

