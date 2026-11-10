// The quadrature and the ordinary differential equations.
//
// A routine takes the function of the caller as a parameter of a
// generic type. The caller passes the function as a block at the end
// of the call. quad follows QUADPACK: it cuts the interval by the
// Gauss and Kronrod rule of 21 points and bisects the interval of the
// largest error. solve_ivp follows the steps of SciPy for the methods
// RK45 and RK23, so the steps equal the steps of SciPy.
namespace scientific;

import mathshim;

const GK_NODES: f64[11] = [0.995657163025808080735527280689003, 0.973906528517171720077964012084452, 0.930157491355708226001207180059508, 0.865063366688984510732096688423493, 0.780817726586416897063717578345042, 0.679409568299024406234327365114874, 0.562757134668604683339000099272694, 0.433395394129247190799265943165784, 0.294392862701460198131126603103866, 0.148874338981631210884826001129720, 0.0];
const GK_WEIGHTS: f64[11] = [0.011694638867371874278064396062192, 0.032558162307964727478818972459390, 0.054755896574351996031381300244580, 0.075039674810919952767043140916190, 0.093125454583697605535065465083366, 0.109387158802297641899210590325805, 0.123491976262065851077958109831074, 0.134709217311473325928054001771707, 0.142775938577060080797094273138717, 0.147739104901338491374841515972068, 0.149445554002916905664936468389821];
const GAUSS_WEIGHTS: f64[5] = [0.066671344308688137593568809893332, 0.149451349150580593145776339657697, 0.219086362515982043995534934228163, 0.269266719309996355091226921569469, 0.295524224714752870173892994651338];

// The largest count of intervals that quad takes.
const MAX_QUAD_LIMIT: i64 = 10000;

fn quad_is_inf(v: f64) -> bool {
    return v - v != 0.0 && v == v;
}

// quad_eval gives the integrand at the point t. The mode 0 is the
// plain function. The modes 1 and 2 map the range of t from zero to one
// to the rays above and below lo.
fn quad_eval<F: fn(f64) -> f64>(mode: i64, lo: f64, t: f64) -> f64 {
    if mode == 0 { return F(t); }
    let u = (1.0 - t) / t;
    if mode == 1 { return F(lo + u) / (t * t); }
    return F(lo - u) / (t * t);
}

// gk_combine rates the rule of 21 points from the values of the
// integrand. The values fv[base] to fv[base + 10] hold the left nodes,
// the last of them the center. The values fv[base + 11] to
// fv[base + 20] hold the right nodes. It gives the integral and the
// error estimate of QUADPACK.
fn gk_combine(fv: f64[:], base: i64, half: f64) -> f64, f64 {
    let fc = fv[base + 10];
    var resk = GK_WEIGHTS[10] * fc;
    var resg: f64 = 0.0;
    var resabs = mathshim::abs_of(resk);
    var j: i64 = 0;
    while j < 10 {
        let f1 = fv[base + j];
        let f2 = fv[base + 11 + j];
        resk += GK_WEIGHTS[j] * (f1 + f2);
        resabs += GK_WEIGHTS[j] * (mathshim::abs_of(f1) + mathshim::abs_of(f2));
        if j % 2 == 1 { resg += GAUSS_WEIGHTS[j / 2] * (f1 + f2); }
        j += 1;
    }
    let reskh = resk * 0.5;
    var resasc = GK_WEIGHTS[10] * mathshim::abs_of(fc - reskh);
    j = 0;
    while j < 10 {
        resasc += GK_WEIGHTS[j] * (mathshim::abs_of(fv[base + j] - reskh) + mathshim::abs_of(fv[base + 11 + j] - reskh));
        j += 1;
    }
    let result = resk * half;
    resabs = resabs * mathshim::abs_of(half);
    resasc = resasc * mathshim::abs_of(half);
    var err = mathshim::abs_of((resk - resg) * half);
    if resasc != 0.0 && err != 0.0 {
        var r = 200.0 * err / resasc;
        r = r * mathshim::root(r);
        if r < 1.0 { err = resasc * r; } else { err = resasc; }
    }
    if resabs > 2.2250738585072014e-308 / (50.0 * 2.220446049250313e-16) {
        let floor = 50.0 * 2.220446049250313e-16 * resabs;
        if floor > err { err = floor; }
    }
    return result, err;
}

// quad_rule applies the rule of 21 points to [a, b]. It gives the
// integral and the error estimate of QUADPACK.
fn quad_rule<F: fn(f64) -> f64>(mode: i64, lo: f64, a: f64, b: f64) -> f64, f64 {
    let center = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    // One call site of quad_eval keeps the count of copies of F small.
    var fv: f64[>..] = [];
    var j: i64 = 0;
    while j < 11 {
        fv.push(quad_eval(mode, lo, center - half * GK_NODES[j]) { x => F(x) });
        j += 1;
    }
    j = 0;
    while j < 10 {
        fv.push(quad_eval(mode, lo, center + half * GK_NODES[j]) { x => F(x) });
        j += 1;
    }
    return gk_combine(fv, 0, half);
}

// quad_run integrates one range by the bisection of the interval of
// the largest error. It gives the value, the error and the fault.
fn quad_run<F: fn(f64) -> f64>(mode: i64, lo: f64, a: f64, b: f64, epsabs: f64, epsrel: f64, limit: i64) -> f64, f64, Fault {
    var ea: f64[>..] = [a];
    var eb: f64[>..] = [b];
    var ev: f64[>..] = [0.0];
    var ee: f64[>..] = [0.0];
    var worst: i64 = -1;
    var total: f64 = 0.0;
    var toterr: f64 = 1.0e300;
    // One call site of quad_rule keeps the count of copies of F small.
    // The first pass rates the whole range. Each later pass splits the
    // interval of the largest error and rates the two halves.
    while worst < 0 || (toterr > epsabs && toterr > epsrel * mathshim::abs_of(total)) {
        if worst >= 0 && ea.len >= limit {
            return total, toterr, fault(F_LIMIT, "quad", "the count of intervals reached the limit before the error fell below the tolerance");
        }
        var pa: f64[>..] = [];
        var pb: f64[>..] = [];
        if worst < 0 {
            pa.push(a);
            pb.push(b);
        } else {
            let lo2 = ea[worst];
            let hi2 = eb[worst];
            let mid = 0.5 * (lo2 + hi2);
            pa.push(lo2);
            pb.push(mid);
            pa.push(mid);
            pb.push(hi2);
        }
        var pv: f64[>..] = [];
        var pe: f64[>..] = [];
        var k: i64 = 0;
        while k < pa.len {
            let v, r = quad_rule(mode, lo, pa[k], pb[k]) { x => F(x) };
            pv.push(v);
            pe.push(r);
            k += 1;
        }
        if worst < 0 {
            ev[0] = pv[0];
            ee[0] = pe[0];
        } else {
            eb[worst] = pb[0];
            ev[worst] = pv[0];
            ee[worst] = pe[0];
            ea.push(pa[1]);
            eb.push(pb[1]);
            ev.push(pv[1]);
            ee.push(pe[1]);
        }
        total = 0.0;
        toterr = 0.0;
        var i: i64 = 0;
        while i < ev.len { total += ev[i]; toterr += ee[i]; i += 1; }
        worst = 0;
        i = 1;
        while i < ee.len { if ee[i] > ee[worst] { worst = i; } i += 1; }
    }
    return total, toterr, no_fault();
}

// quad integrates a function from a to b. The range may reach to
// infinity at either end, with an infinite value for the end. It gives
// the value and the estimate of the error. The fault F_LIMIT means that
// the count of intervals reached the limit. The estimate of the error
// is not equal to the estimate of SciPy, because the bisection of the
// package holds no extrapolation.
fn quad<F: fn(f64) -> f64>(a: f64, b: f64, epsabs: f64 = 1.49e-8, epsrel: f64 = 1.49e-8, limit: i64 = 50) -> f64, f64, Fault {
    if a != a || b != b {
        return 0.0, 0.0, fault(F_VALUE, "quad", "a limit holds the missing value");
    }
    if !(epsabs >= 0.0) || !(epsrel >= 0.0) || (epsabs == 0.0 && epsrel < 5.0e-29) {
        return 0.0, 0.0, fault(F_VALUE, "quad", "the tolerance holds a value that no result can meet");
    }
    if limit < 1 || limit > MAX_QUAD_LIMIT {
        return 0.0, 0.0, fault(F_RANGE, "quad", "the limit stands outside one to the largest count");
    }
    if a == b { return 0.0, 0.0, no_fault(); }
    var lo = a;
    var hi = b;
    var sgn: f64 = 1.0;
    if a > b { lo = b; hi = a; sgn = -1.0; }
    let ia = quad_is_inf(lo);
    let ib = quad_is_inf(hi);
    if !ia && !ib {
        let v, e, f = quad_run(0, 0.0, lo, hi, epsabs, epsrel, limit) { x => F(x) };
        return sgn * v, e, f;
    }
    if ia && ib {
        let v1, e1, f1 = quad_run(1, 0.0, 0.0, 1.0, epsabs, epsrel, limit) { x => F(x) };
        if f1.kind != F_NONE { return sgn * v1, e1, f1; }
        let v2, e2, f2 = quad_run(2, 0.0, 0.0, 1.0, epsabs, epsrel, limit) { x => F(x) };
        return sgn * (v1 + v2), e1 + e2, f2;
    }
    if ib {
        let v, e, f = quad_run(1, lo, 0.0, 1.0, epsabs, epsrel, limit) { x => F(x) };
        return sgn * v, e, f;
    }
    let v, e, f = quad_run(2, hi, 0.0, 1.0, epsabs, epsrel, limit) { x => F(x) };
    return sgn * v, e, f;
}

// fixed_quad integrates a function from a to b by the Gauss and
// Legendre rule of n points. The rule is exact for a polynomial of
// the degree 2 n - 1.
fn fixed_quad<F: fn(f64) -> f64>(a: f64, b: f64, n: i64 = 5) -> f64, Fault {
    if n < 1 || n > 512 {
        return 0.0, fault(F_RANGE, "fixed_quad", "the count of points stands outside one to 512");
    }
    if !(a - a == 0.0) || !(b - b == 0.0) {
        return 0.0, fault(F_VALUE, "fixed_quad", "a limit is not finite");
    }
    var total: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        // The root i of the Legendre polynomial by the method of Newton.
        var x = mathshim::cosine(3.141592653589793 * (i + 0.75) / (n + 0.5));
        var dp: f64 = 1.0;
        var turn: i64 = 0;
        while turn < 100 {
            var p0: f64 = 1.0;
            var p1 = x;
            var k: i64 = 2;
            if n == 1 { p1 = x; }
            while k <= n {
                let p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
                p0 = p1;
                p1 = p2;
                k += 1;
            }
            if n == 1 { dp = 1.0; p0 = 1.0; } else { dp = n * (x * p1 - p0) / (x * x - 1.0); }
            let step = p1 / dp;
            x = x - step;
            if mathshim::abs_of(step) < 1.0e-16 { break; }
            turn += 1;
        }
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        total += w * F(0.5 * (b - a) * x + 0.5 * (b + a));
        i += 1;
    }
    return total * 0.5 * (b - a), no_fault();
}

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
fn solve_ivp<F: fn(f64, f64[:]) -> f64[:]>(t0: f64, t1: f64, y0: f64[:], rtol: f64 = 0.001, atol: f64 = 1.0e-6, method: const u8[:] = "RK45", max_step: f64 = 1.0e308, t_eval: f64[:] = []) -> Arr, Arr, Fault {
    let t, y, et, ei, ey, dn, f = ivp_run(t0, t1, y0, rtol, atol, method, max_step, t_eval, 0, [], [], false) { tt, yy => F(tt, copy(yy)) };
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
fn solve_ivp_events<F: fn(f64, f64[:]) -> f64[:]>(t0: f64, t1: f64, y0: f64[:], nev: i64, terminal: bool[:], direction: f64[:], rtol: f64 = 0.001, atol: f64 = 1.0e-6, method: const u8[:] = "RK45", max_step: f64 = 1.0e308, t_eval: f64[:] = []) -> Arr, Arr, Arr, Arr, Arr, Fault {
    if nev < 1 || nev > 64 || terminal.len != nev || direction.len != nev {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "solve_ivp_events", "the flags and the directions do not hold one value per event");
    }
    let t, y, et, ei, ey, dn, f = ivp_run(t0, t1, y0, rtol, atol, method, max_step, t_eval, nev, terminal, direction, false) { tt, yy => F(tt, copy(yy)) };
    return t, y, et, ei, ey, f;
}

// solve_ivp_dense integrates like solve_ivp and keeps the polynomial of
// each step. It gives the times, the states and the dense output. Read
// the dense output with dense_eval.
fn solve_ivp_dense<F: fn(f64, f64[:]) -> f64[:]>(t0: f64, t1: f64, y0: f64[:], rtol: f64 = 0.001, atol: f64 = 1.0e-6, method: const u8[:] = "RK45", max_step: f64 = 1.0e308) -> Arr, Arr, Dense, Fault {
    let t, y, et, ei, ey, dn, f = ivp_run(t0, t1, y0, rtol, atol, method, max_step, [], 0, [], [], true) { tt, yy => F(tt, copy(yy)) };
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

// ivp_run integrates the system y' = f(t, y) from t0 to t1. The block
// gives n values of the derivative, then nev values of the event
// functions. It gives the times and the states, the times, the numbers
// and the states of the events, and the dense output of the steps.
fn ivp_run<F: fn(f64, f64[:]) -> f64[:]>(t0: f64, t1: f64, y0: f64[:], rtol: f64, atol: f64, method: const u8[:], max_step: f64, t_eval: f64[:], nev: i64, terminal: bool[:], direction: f64[:], want_dense: bool) -> Arr, Arr, Arr, Arr, Arr, Dense, Fault {
    if method != "RK45" && method != "RK23" && method != "DOP853" {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_VALUE, "solve_ivp", "the method holds an unknown name");
    }
    if !(t0 - t0 == 0.0) || !(t1 - t1 == 0.0) || t0 == t1 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_VALUE, "solve_ivp", "the times are not finite and different");
    }
    if y0.len < 1 || y0.len > 4096 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_SHAPE, "solve_ivp", "the start holds no value or too many");
    }
    if !(rtol > 0.0) || !(atol >= 0.0) || !(max_step > 0.0) {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_VALUE, "solve_ivp", "a tolerance or the largest step holds a value out of range");
    }
    var tol = rtol;
    if tol < 2.220446049250313e-14 { tol = 2.220446049250313e-14; }
    let n = y0.len;
    var dir: f64 = 1.0;
    if t1 < t0 { dir = -1.0; }
    // The times of t_eval must lie in the span and must be sorted.
    var qi: i64 = 0;
    while qi < t_eval.len {
        let te = t_eval[qi];
        if !(te - te == 0.0) || dir * (te - t0) < 0.0 || dir * (t1 - te) < 0.0 {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_VALUE, "solve_ivp", "a time of t_eval is not finite or lies outside the span");
        }
        if qi > 0 && dir * (te - t_eval[qi - 1]) < 0.0 {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_VALUE, "solve_ivp", "the times of t_eval do not follow the direction");
        }
        qi += 1;
    }
    var hist_t: f64[>..] = [];
    var hist_h: f64[>..] = [];
    var hist_y: f64[>..] = [];
    var hist_k: f64[>..] = [];
    // The kind of the method. The number zero is RK45, the number one
    // is RK23 and the number two is DOP853.
    var kind: i32 = 1;
    if method == "RK45" { kind = 0; }
    if method == "DOP853" { kind = 2; }
    var order: f64 = 3.0;
    if kind == 0 { order = 5.0; }
    if kind == 2 { order = 8.0; }
    // The exponent of the control is minus one over the order of the
    // error estimate plus one.
    var cc: f64[>..] = [];
    var aa: f64[>..] = [];
    var bb: f64[>..] = [];
    var ee: f64[>..] = [];
    var e3: f64[>..] = [];
    let ns = ivp_tableau(kind, cc, aa, bb, ee, e3);
    // The count of the rows of the stages that the dense output holds.
    // The method of the order eight holds three more rows for the
    // interpolation.
    var rows: i64 = ns + 1;
    if kind == 2 { rows = 16; }
    var y: f64[>..] = [];
    var i: i64 = 0;
    while i < n { y.push(y0[i]); i += 1; }
    var t = t0;
    let f00 = F(t, copy(y));
    if f00.len != n + nev {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
    }
    var f: f64[>..] = [];
    i = 0;
    while i < n { f.push(f00[i]); i += 1; }
    // The first step.
    var gold: f64[>..] = [];
    i = 0;
    while i < nev { gold.push(f00[n + i]); i += 1; }
    var ev_t: f64[>..] = [];
    var ev_i: f64[>..] = [];
    var ev_y: f64[>..] = [];
    var stop = false;
    var scale: f64[>..] = [];
    var q1: f64[>..] = [];
    var q2: f64[>..] = [];
    i = 0;
    while i < n { scale.push(atol + mathshim::abs_of(y[i]) * tol); q1.push(y[i] / scale[i]); q2.push(f[i] / scale[i]); i += 1; }
    let d0 = ivp_rms(q1);
    let d1 = ivp_rms(q2);
    var h0: f64 = 0.000001;
    if !(d0 < 0.00001 || d1 < 0.00001) { h0 = 0.01 * d0 / d1; }
    // The step of the trial does not leave the span, as SciPy does.
    let span = mathshim::abs_of(t1 - t0);
    if h0 > span { h0 = span; }
    var y1: f64[>..] = [];
    i = 0;
    while i < n { y1.push(y[i] + h0 * dir * f[i]); i += 1; }
    let f1 = F(t + h0 * dir, copy(y1));
    if f1.len != n + nev {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
    }
    var q3: f64[>..] = [];
    i = 0;
    while i < n { q3.push((f1[i] - f[i]) / scale[i]); i += 1; }
    let d2 = ivp_rms(q3) / h0;
    var h1: f64 = 0.0;
    if d1 <= 1.0e-15 && d2 <= 1.0e-15 {
        h1 = h0 * 0.001;
        if h1 < 0.000001 { h1 = 0.000001; }
    } else {
        var big = d1;
        if d2 > big { big = d2; }
        h1 = mathshim::power(0.01 / big, 1.0 / order);
    }
    var h_abs = 100.0 * h0;
    if h1 < h_abs { h_abs = h1; }
    if span < h_abs { h_abs = span; }
    if max_step < h_abs { h_abs = max_step; }
    var ts: f64[>..] = [t];
    var ys: f64[>..] = [];
    i = 0;
    while i < n { ys.push(y[i]); i += 1; }
    var rejected = false;
    var steps: i64 = 0;
    while dir * (t1 - t) > 0.0 && !stop {
        steps += 1;
        if steps > 1000000 {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_LIMIT, "solve_ivp", "the count of steps reached the limit");
        }
        var min_step = 10.0 * mathshim::abs_of(t) * 2.220446049250313e-16;
        if min_step < 5.0e-323 { min_step = 5.0e-323; }
        if h_abs < min_step { h_abs = min_step; }
        if h_abs > max_step { h_abs = max_step; }
        var accepted = false;
        var kacc: f64[>..] = [];
        var fnew: f64[>..] = [];
        var gnew: f64[>..] = [];
        var ynew: f64[>..] = [];
        var tnew: f64 = t;
        while !accepted {
            if h_abs < min_step {
                return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_VALUE, "solve_ivp", "the step fell below the smallest step");
            }
            var h = h_abs * dir;
            tnew = t + h;
            if dir * (tnew - t1) > 0.0 { tnew = t1; }
            h = tnew - t;
            h_abs = mathshim::abs_of(h);
            // The stages. K holds one row of n values per stage.
            var k: f64[>..] = [];
            i = 0;
            while i < n { k.push(f[i]); i += 1; }
            var s: i64 = 1;
            while s < ns {
                var ytmp: f64[>..] = [];
                i = 0;
                while i < n {
                    var acc: f64 = 0.0;
                    var j: i64 = 0;
                    while j < s { acc += k[j * n + i] * aa[s * ns + j]; j += 1; }
                    ytmp.push(y[i] + h * acc);
                    i += 1;
                }
                let ks = F(t + cc[s] * h, copy(ytmp));
                if ks.len != n + nev {
                    return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
                }
                i = 0;
                while i < n { k.push(ks[i]); i += 1; }
                s += 1;
            }
            ynew = [];
            i = 0;
            while i < n {
                var acc: f64 = 0.0;
                var j: i64 = 0;
                while j < ns { acc += k[j * n + i] * bb[j]; j += 1; }
                ynew.push(y[i] + h * acc);
                i += 1;
            }
            let fn1 = F(tnew, copy(ynew));
            if fn1.len != n + nev {
                return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
            }
            fnew = [];
            gnew = [];
            i = 0;
            while i < n { fnew.push(fn1[i]); i += 1; }
            i = 0;
            while i < nev { gnew.push(fn1[n + i]); i += 1; }
            var err: f64 = 0.0;
            if kind == 2 {
                // The method of the order eight holds two error vectors.
                // The estimate of the error takes the fifth order and
                // the third order together, as SciPy does.
                var s5: f64 = 0.0;
                var s3: f64 = 0.0;
                i = 0;
                while i < n {
                    var a5: f64 = 0.0;
                    var a3: f64 = 0.0;
                    var j: i64 = 0;
                    while j < ns { a5 += k[j * n + i] * ee[j]; a3 += k[j * n + i] * e3[j]; j += 1; }
                    a5 += fnew[i] * ee[ns];
                    a3 += fnew[i] * e3[ns];
                    var big = mathshim::abs_of(y[i]);
                    if mathshim::abs_of(ynew[i]) > big { big = mathshim::abs_of(ynew[i]); }
                    let sc = atol + big * tol;
                    let d5 = a5 / sc;
                    let d3 = a3 / sc;
                    s5 += d5 * d5;
                    s3 += d3 * d3;
                    i += 1;
                }
                if !(s5 == 0.0 && s3 == 0.0) {
                    err = mathshim::abs_of(h) * s5 / mathshim::root((s5 + 0.01 * s3) * n);
                }
            } else {
                var en: f64[>..] = [];
                i = 0;
                while i < n {
                    var acc: f64 = 0.0;
                    var j: i64 = 0;
                    while j < ns { acc += k[j * n + i] * ee[j]; j += 1; }
                    acc += fnew[i] * ee[ns];
                    en.push(h * acc);
                    i += 1;
                }
                var sq: f64[>..] = [];
                i = 0;
                while i < n {
                    var big = mathshim::abs_of(y[i]);
                    if mathshim::abs_of(ynew[i]) > big { big = mathshim::abs_of(ynew[i]); }
                    sq.push(en[i] / (atol + big * tol));
                    i += 1;
                }
                err = ivp_rms(sq);
            }
            if err < 1.0 {
                kacc = copy_slice(k);
                var factor: f64 = 10.0;
                if err != 0.0 {
                    factor = 0.9 * mathshim::power(err, -1.0 / order);
                    if factor > 10.0 { factor = 10.0; }
                }
                if rejected && factor > 1.0 { factor = 1.0; }
                h_abs = h_abs * factor;
                accepted = true;
                rejected = false;
            } else {
                var factor = 0.9 * mathshim::power(err, -1.0 / order);
                if factor < 0.2 { factor = 0.2; }
                h_abs = h_abs * factor;
                rejected = true;
            }
        }
        // The rows of the stages that the interpolation of the step
        // uses. The method of the order eight holds three more rows.
        var kall: f64[>..] = [];
        var kex: f64[>..] = [];
        if kind == 2 && (t_eval.len > 0 || want_dense || nev > 0) {
            let hh = tnew - t;
            var exa: f64[>..] = [];
            var exc: f64[>..] = [];
            dop853_extra(exa, exc);
            var s: i64 = 13;
            while s < 16 {
                var base: i64 = 0;
                if s == 14 { base = 13; }
                if s == 15 { base = 27; }
                var ytmp: f64[>..] = [];
                i = 0;
                while i < n {
                    var acc: f64 = 0.0;
                    var j: i64 = 0;
                    while j < s {
                        var kv: f64 = 0.0;
                        if j < ns { kv = kacc[j * n + i]; }
                        else if j == ns { kv = fnew[i]; }
                        else { kv = kex[(j - ns - 1) * n + i]; }
                        acc += kv * exa[base + j];
                        j += 1;
                    }
                    ytmp.push(y[i] + hh * acc);
                    i += 1;
                }
                let ks = F(t + exc[s - 13] * hh, copy(ytmp));
                if ks.len != n + nev {
                    return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
                }
                i = 0;
                while i < n { kex.push(ks[i]); i += 1; }
                s += 1;
            }
        }
        i = 0;
        while i < ns * n { kall.push(kacc[i]); i += 1; }
        i = 0;
        while i < n { kall.push(fnew[i]); i += 1; }
        i = 0;
        while i < kex.len { kall.push(kex[i]); i += 1; }
        if t_eval.len > 0 || want_dense {
            hist_t.push(t);
            hist_h.push(tnew - t);
            i = 0;
            while i < n { hist_y.push(y[i]); i += 1; }
            i = 0;
            while i < kall.len { hist_k.push(kall[i]); i += 1; }
        }
        if nev > 0 {
            // The events of the step. A change of sign of an event
            // function in the direction that the caller asks for.
            var hit_t: f64[>..] = [];
            var hit_i: f64[>..] = [];
            var e: i64 = 0;
            while e < nev {
                let g0 = gold[e];
                let g1 = gnew[e];
                let up = g0 <= 0.0 && g1 >= 0.0;
                let down = g0 >= 0.0 && g1 <= 0.0;
                var want = false;
                if direction[e] > 0.0 { want = up; }
                else if direction[e] < 0.0 { want = down; }
                else { want = up || down; }
                if want {
                    // The root by bisection of the polynomial of the step.
                    var lo: f64 = 0.0;
                    var hi: f64 = 1.0;
                    var glo = g0;
                    var it: i64 = 0;
                    while it < 200 {
                        let mid = 0.5 * (lo + hi);
                        if mid == lo || mid == hi { break; }
                        let ym = ivp_point_kind(kind, ns, n, y, ynew, tnew - t, kall, mid);
                        let gm = F(t + mid * (tnew - t), copy(ym));
                        let gv = gm[n + e];
                        if (glo <= 0.0 && gv >= 0.0) || (glo >= 0.0 && gv <= 0.0) { hi = mid; } else { lo = mid; glo = gv; }
                        it += 1;
                    }
                    hit_t.push(t + hi * (tnew - t));
                    hit_i.push(e);
                }
                e += 1;
            }
            // The hits in the order of the direction of time.
            var order: f64[>..] = [];
            var taken: f64[>..] = [];
            i = 0;
            while i < hit_t.len { taken.push(0.0); i += 1; }
            var c: i64 = 0;
            while c < hit_t.len {
                var pick: i64 = -1;
                i = 0;
                while i < hit_t.len {
                    if taken[i] == 0.0 && (pick < 0 || dir * hit_t[i] < dir * hit_t[pick]) { pick = i; }
                    i += 1;
                }
                taken[pick] = 1.0;
                order.push(pick);
                c += 1;
            }
            var oi: i64 = 0;
            while oi < order.len && !stop {
                let h = order[oi] as i64;
                let th = hit_t[h];
                let eh = hit_i[h] as i64;
                let fr = (th - t) / (tnew - t);
                let yh = ivp_point_kind(kind, ns, n, y, ynew, tnew - t, kall, fr);
                ev_t.push(th);
                ev_i.push(eh);
                i = 0;
                while i < n { ev_y.push(yh[i]); i += 1; }
                if terminal[eh] {
                    stop = true;
                    tnew = th;
                    ynew = copy_slice(yh);
                    fnew = [];
                    var fz = F(th, copy(yh));
                    i = 0;
                    while i < n { fnew.push(fz[i]); i += 1; }
                }
                oi += 1;
            }
            gold = copy_slice(gnew);
        }
        t = tnew;
        y = copy(ynew);
        f = copy(fnew);
        ts.push(t);
        i = 0;
        while i < n { ys.push(y[i]); i += 1; }
    }
    var dn = dense_none();
    if want_dense {
        var buf: f64[>..] = [];
        var q: i64 = 0;
        while q < hist_t.len { buf.push(hist_t[q]); q += 1; }
        q = 0;
        while q < hist_h.len { buf.push(hist_h[q]); q += 1; }
        q = 0;
        while q < hist_y.len { buf.push(hist_y[q]); q += 1; }
        q = 0;
        while q < hist_k.len { buf.push(hist_k[q]); q += 1; }
        dn = Dense { kind: kind, ns: ns, n: n, steps: hist_t.len, buf: copy(buf) };
    }
    var evt = empty_arr();
    var evi = empty_arr();
    var evy = empty_arr();
    if ev_t.len > 0 {
        evt = line_arr(copy(ev_t));
        evi = line_arr(copy(ev_i));
        evy = mat2(copy(ev_y), ev_t.len, n);
    }
    if t_eval.len > 0 {
        // The times after a terminal event drop out.
        var kept: f64[>..] = [];
        var qq: i64 = 0;
        while qq < t_eval.len {
            if dir * (t_eval[qq] - t) <= 0.0 { kept.push(t_eval[qq]); }
            qq += 1;
        }
        let rt, ry, rf = ivp_dense(kept, hist_t, hist_h, hist_y, hist_k, kind, ns, n);
        return rt, ry, evt, evi, evy, dn, rf;
    }
    return line_arr(copy(ts)), mat2(copy(ys), ts.len, n), evt, evi, evy, dn, no_fault();
}

// ivp_dense gives the values at the times of t_eval from the dense
// output of the steps. The polynomial of each step uses the stages of
// the step, as SciPy does for RK45 and RK23.
fn ivp_dense(te: f64[:], ht: f64[:], hh: f64[:], hy: f64[:], hk: f64[:], kind: i32, ns: i64, n: i64) -> Arr, Arr, Fault {
    var pm: f64[>..] = [];
    var pc: i64 = 3;
    if kind == 0 {
        pc = 4;
        pm = [1.0, -2.8535800653862835, 3.0717434641059005, -1.1270175653862835,
              0.0, 0.0, 0.0, 0.0,
              0.0, 4.023133379230305, -6.249321565289, 2.675424484351598,
              0.0, -3.7324019615885042, 10.068970589843675, -5.685526961588504,
              0.0, 2.5548038301849423, -6.399112377351017, 3.5219323679207912,
              0.0, -1.3744241142186024, 3.272657752246729, -1.7672812570757455,
              0.0, 1.3824689317781436, -3.764937863556287, 2.382468931778144];
    } else if kind == 1 {
        pm = [1.0, -1.3333333333333333, 0.5555555555555556,
              0.0, 1.0, -0.6666666666666666,
              0.0, 1.3333333333333333, -0.8888888888888888,
              0.0, -1.0, 1.0];
    }
    let rows = ns + 1;
    var ts: f64[>..] = [];
    var ys: f64[>..] = [];
    var si: i64 = 0;
    var j: i64 = 0;
    while j < te.len {
        let x = te[j];
        // The first step whose end lies at x or after it.
        while si + 1 < ht.len && (hh[si] > 0.0 && ht[si] + hh[si] < x || hh[si] < 0.0 && ht[si] + hh[si] > x) { si += 1; }
        let xs = (x - ht[si]) / hh[si];
        if kind == 2 {
            // The method of the order eight. The rows of the stages of
            // the step and the state at its end.
            var kb: f64[>..] = [];
            var i0: i64 = 0;
            while i0 < 16 * n { kb.push(hk[si * 16 * n + i0]); i0 += 1; }
            var bb: f64[>..] = [];
            dop853_b(bb);
            var yo: f64[>..] = [];
            var yn: f64[>..] = [];
            var i: i64 = 0;
            while i < n {
                yo.push(hy[si * n + i]);
                var acc: f64 = 0.0;
                var r: i64 = 0;
                while r < 12 { acc += kb[r * n + i] * bb[r]; r += 1; }
                yn.push(hy[si * n + i] + hh[si] * acc);
                i += 1;
            }
            let yv = dop853_point(n, yo, yn, hh[si], kb, xs);
            ts.push(x);
            i = 0;
            while i < n { ys.push(yv[i]); i += 1; }
            j += 1;
            continue;
        }
        var pw: f64[>..] = [];
        var pp: f64 = 1.0;
        var c: i64 = 0;
        while c < pc { pp = pp * xs; pw.push(pp); c += 1; }
        ts.push(x);
        var i: i64 = 0;
        while i < n {
            var acc: f64 = 0.0;
            c = 0;
            while c < pc {
                var q: f64 = 0.0;
                var r: i64 = 0;
                while r < rows { q += hk[si * rows * n + r * n + i] * pm[r * pc + c]; r += 1; }
                acc += q * pw[c];
                c += 1;
            }
            ys.push(hy[si * n + i] + hh[si] * acc);
            i += 1;
        }
        j += 1;
    }
    return line_arr(copy(ts)), mat2(copy(ys), ts.len, n), no_fault();
}

// quad_finite tells whether a number is neither infinite nor missing.
fn quad_finite(v: f64) -> bool { return v - v == 0.0; }

// dblquad_inner integrates the function over y for one x, between two
// finite limits.
fn dblquad_inner<F: fn(f64, f64) -> f64>(x: f64, lo: f64, hi: f64, epsabs: f64, epsrel: f64, limit: i64) -> f64, f64, Fault {
    let v, e, f = quad(lo, hi, epsabs, epsrel, limit) { y => F(y, x) };
    return v, e, f;
}

// dblquad gives the double integral of a function f(y, x) over x from a
// to b and over y from g(x) to h(x). The limits may be infinite. The limits
// of y are two named functions of the caller. The function f is a block.
// The call gives the value and the sum of the errors of the inner
// integrals. A fault of an inner integral stops the call.
fn dblquad<G: fn(f64) -> f64, H: fn(f64) -> f64, F: fn(f64, f64) -> f64>(a: f64, b: f64, epsabs: f64 = 1.49e-8, epsrel: f64 = 1.49e-8, limit: i64 = 50) -> f64, f64, Fault {
    if a != a || b != b {
        return 0.0, 0.0, fault(F_VALUE, "dblquad", "a limit of x holds the missing value");
    }
    if !(epsabs >= 0.0) || !(epsrel >= 0.0) || (epsabs == 0.0 && epsrel < 5.0e-29) {
        return 0.0, 0.0, fault(F_VALUE, "dblquad", "the tolerance holds a value that no result can meet");
    }
    if limit < 1 || limit > MAX_QUAD_LIMIT {
        return 0.0, 0.0, fault(F_RANGE, "dblquad", "the limit stands outside one to the largest count");
    }
    if a == b { return 0.0, 0.0, no_fault(); }
    var err: f64 = 0.0;
    var bad: Fault = no_fault();
    let v, e, f = quad(a, b, epsabs, epsrel, limit) { x =>
        let g = G(x);
        let h = H(x);
        var r: f64 = 0.0;
        if g != g || h != h {
            bad = fault(F_VALUE, "dblquad", "a limit of y holds the missing value");
        } else {
            let iv, ie, ifl = dblquad_inner(x, g, h, epsabs, epsrel, limit) { y, xx => F(y, xx) };
            if ifl.kind != F_NONE { bad = ifl; }
            err += ie;
            r = iv;
        }
        r
    };
    if bad.kind != F_NONE { return 0.0, 0.0, bad; }
    return v, e + err, f;
}

// odeint integrates dy/dt = f(y, t) from the start y0 and gives the
// state at each time in t, one row per time. The first row is y0. The
// function is a block { y, t => ... }, with the arguments in the order
// of SciPy odeint. Each segment between two times runs the RK45 method
// of solve_ivp with the tolerances rtol and atol. The times rise or
// fall without change of direction.
fn odeint<F: fn(f64[:], f64) -> f64[:]>(y0: f64[:], t: f64[:], rtol: f64 = 1.49012e-8, atol: f64 = 1.49012e-8) -> Arr, Fault {
    let n = y0.len;
    let m = t.len;
    if n < 1 || m < 2 {
        return empty_arr(), fault(F_SHAPE, "odeint", "the start or the times hold too few values");
    }
    if m * n > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "odeint", "the result holds more elements than the limit");
    }
    var i: i64 = 1;
    var up = t[1] > t[0];
    while i < m {
        if !(t[i] - t[i] == 0.0) || t[i] == t[i - 1] || (t[i] > t[i - 1]) != up {
            return empty_arr(), fault(F_VALUE, "odeint", "the times do not move in one direction");
        }
        i += 1;
    }
    var out: f64[>..] = [];
    var cur: f64[>..] = [];
    i = 0;
    while i < n { out.push(y0[i]); cur.push(y0[i]); i += 1; }
    var k: i64 = 1;
    while k < m {
        let ts, ys, f = solve_ivp(t[k - 1], t[k], cur, rtol, atol, "RK45", 1.0e308) { tt, yy => F(copy(yy), tt) };
        if f.kind != F_NONE { return empty_arr(), fault(f.kind, "odeint", f.detail); }
        let rows = ts.data.len;
        i = 0;
        while i < n { cur[i] = ys.data[(rows - 1) * n + i]; out.push(cur[i]); i += 1; }
        k += 1;
    }
    return mat2(copy(out), m, n), no_fault();
}

// tplquad_mid integrates the function over y and z for one x. The
// limits of y are given as numbers, already found by the caller.
fn tplquad_mid<Q: fn(f64, f64) -> f64, R: fn(f64, f64) -> f64, F: fn(f64, f64, f64) -> f64>(x: f64, g: f64, h: f64, epsabs: f64, epsrel: f64, limit: i64) -> f64, f64, Fault {
    if !(quad_finite(g) && quad_finite(h)) {
        return 0.0, 0.0, fault(F_VALUE, "tplquad", "a limit of y holds a value that is not finite");
    }
    var a = g;
    var b = h;
    var sgn: f64 = 1.0;
    if g > h { a = h; b = g; sgn = -1.0; }
    var err: f64 = 0.0;
    var bad: Fault = no_fault();
    let v, e, f = quad_run(0, 0.0, a, b, epsabs, epsrel, limit) { y =>
        let q = Q(x, y);
        let r = R(x, y);
        var out: f64 = 0.0;
        if quad_finite(q) && quad_finite(r) {
            if q == r { out = 0.0; }
            else {
                var z0 = q;
                var z1 = r;
                var zs: f64 = 1.0;
                if q > r { z0 = r; z1 = q; zs = -1.0; }
                let iv, ie, ifl = quad_run(0, 0.0, z0, z1, epsabs, epsrel, limit) { z => F(z, y, x) };
                if ifl.kind != F_NONE { bad = ifl; }
                err += ie;
                out = zs * iv;
            }
        } else {
            bad = fault(F_VALUE, "tplquad", "a limit of z holds a value that is not finite");
        }
        out
    };
    if bad.kind != F_NONE { return 0.0, 0.0, bad; }
    return sgn * v, e + err, f;
}

// tplquad gives the triple integral of a function f(z, y, x) over x from
// a to b, over y from g(x) to h(x) and over z from q(x, y) to r(x, y).
// The limits are named functions of the caller. The function f is a
// block. The call gives the value and the sum of the errors of the
// inner integrals. A fault of an inner integral stops the call.
fn tplquad<G: fn(f64) -> f64, H: fn(f64) -> f64, Q: fn(f64, f64) -> f64, R: fn(f64, f64) -> f64, F: fn(f64, f64, f64) -> f64>(a: f64, b: f64, epsabs: f64 = 1.49e-8, epsrel: f64 = 1.49e-8, limit: i64 = 50) -> f64, f64, Fault {
    if a != a || b != b || !quad_finite(a) || !quad_finite(b) {
        return 0.0, 0.0, fault(F_VALUE, "tplquad", "a limit of x holds a value that is not finite");
    }
    if !(epsabs >= 0.0) || !(epsrel >= 0.0) || (epsabs == 0.0 && epsrel < 5.0e-29) {
        return 0.0, 0.0, fault(F_VALUE, "tplquad", "the tolerance holds a value that no result can meet");
    }
    if limit < 1 || limit > MAX_QUAD_LIMIT {
        return 0.0, 0.0, fault(F_RANGE, "tplquad", "the limit stands outside one to the largest count");
    }
    if a == b { return 0.0, 0.0, no_fault(); }
    var lo = a;
    var hi = b;
    var sgn: f64 = 1.0;
    if a > b { lo = b; hi = a; sgn = -1.0; }
    var err: f64 = 0.0;
    var bad: Fault = no_fault();
    let v, e, f = quad_run(0, 0.0, lo, hi, epsabs, epsrel, limit) { x =>
        let g = G(x);
        let h = H(x);
        var out: f64 = 0.0;
        if quad_finite(g) && quad_finite(h) {
            let iv, ie, ifl = tplquad_mid(x, g, h, epsabs, epsrel, limit, Q, R) { z, y, xx => F(z, y, xx) };
            if ifl.kind != F_NONE { bad = ifl; }
            err += ie;
            out = iv;
        } else {
            bad = fault(F_VALUE, "tplquad", "a limit of y holds a value that is not finite");
        }
        out
    };
    if bad.kind != F_NONE { return 0.0, 0.0, bad; }
    return sgn * v, e + err, f;
}
