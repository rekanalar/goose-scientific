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

// quad_rule applies the rule of 21 points to [a, b]. It gives the
// integral and the error estimate of QUADPACK.
fn quad_rule<F: fn(f64) -> f64>(mode: i64, lo: f64, a: f64, b: f64) -> f64, f64 {
    let center = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let fc = quad_eval(mode, lo, center) { x => F(x) };
    var resk = GK_WEIGHTS[10] * fc;
    var resg: f64 = 0.0;
    var resabs = mathshim::abs_of(resk);
    var fv1: f64[>..] = [];
    var fv2: f64[>..] = [];
    var j: i64 = 0;
    while j < 10 {
        let dx = half * GK_NODES[j];
        let f1 = quad_eval(mode, lo, center - dx) { x => F(x) };
        let f2 = quad_eval(mode, lo, center + dx) { x => F(x) };
        fv1.push(f1);
        fv2.push(f2);
        resk += GK_WEIGHTS[j] * (f1 + f2);
        resabs += GK_WEIGHTS[j] * (mathshim::abs_of(f1) + mathshim::abs_of(f2));
        if j % 2 == 1 { resg += GAUSS_WEIGHTS[j / 2] * (f1 + f2); }
        j += 1;
    }
    let reskh = resk * 0.5;
    var resasc = GK_WEIGHTS[10] * mathshim::abs_of(fc - reskh);
    j = 0;
    while j < 10 {
        resasc += GK_WEIGHTS[j] * (mathshim::abs_of(fv1[j] - reskh) + mathshim::abs_of(fv2[j] - reskh));
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

// quad_run integrates one range by the bisection of the interval of
// the largest error. It gives the value, the error and the fault.
fn quad_run<F: fn(f64) -> f64>(mode: i64, lo: f64, a: f64, b: f64, epsabs: f64, epsrel: f64, limit: i64) -> f64, f64, Fault {
    var ea: f64[>..] = [a];
    var eb: f64[>..] = [b];
    var ev: f64[>..] = [];
    var ee: f64[>..] = [];
    let v0, e0 = quad_rule(mode, lo, a, b) { x => F(x) };
    ev.push(v0);
    ee.push(e0);
    var total = v0;
    var toterr = e0;
    while toterr > epsabs && toterr > epsrel * mathshim::abs_of(total) {
        if ea.len >= limit {
            return total, toterr, fault(F_LIMIT, "quad", "the count of intervals reached the limit before the error fell below the tolerance");
        }
        var worst: i64 = 0;
        var i: i64 = 1;
        while i < ee.len { if ee[i] > ee[worst] { worst = i; } i += 1; }
        let lo2 = ea[worst];
        let hi2 = eb[worst];
        let mid = 0.5 * (lo2 + hi2);
        let v1, r1 = quad_rule(mode, lo, lo2, mid) { x => F(x) };
        let v2, r2 = quad_rule(mode, lo, mid, hi2) { x => F(x) };
        eb[worst] = mid;
        ev[worst] = v1;
        ee[worst] = r1;
        ea.push(mid);
        eb.push(hi2);
        ev.push(v2);
        ee.push(r2);
        total = 0.0;
        toterr = 0.0;
        i = 0;
        while i < ev.len { total += ev[i]; toterr += ee[i]; i += 1; }
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
// Shampine.
fn ivp_tableau(rk45: bool, c: f64[>..]&, a: f64[>..]&, b: f64[>..]&, e: f64[>..]&) -> i64 {
    if rk45 {
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
fn solve_ivp<F: fn(f64, f64[:]) -> f64[:]>(t0: f64, t1: f64, y0: f64[:], rtol: f64 = 0.001, atol: f64 = 1.0e-6, method: const u8[:] = "RK45", max_step: f64 = 1.0e308) -> Arr, Arr, Fault {
    if method != "RK45" && method != "RK23" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "solve_ivp", "the method holds an unknown name");
    }
    if !(t0 - t0 == 0.0) || !(t1 - t1 == 0.0) || t0 == t1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "solve_ivp", "the times are not finite and different");
    }
    if y0.len < 1 || y0.len > 4096 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "solve_ivp", "the start holds no value or too many");
    }
    if !(rtol > 0.0) || !(atol >= 0.0) || !(max_step > 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "solve_ivp", "a tolerance or the largest step holds a value out of range");
    }
    var tol = rtol;
    if tol < 2.220446049250313e-14 { tol = 2.220446049250313e-14; }
    let n = y0.len;
    var dir: f64 = 1.0;
    if t1 < t0 { dir = -1.0; }
    let rk45 = method == "RK45";
    var order: f64 = 3.0;
    if rk45 { order = 5.0; }
    // The exponent of the control is minus one over the order of the
    // error estimate plus one.
    var cc: f64[>..] = [];
    var aa: f64[>..] = [];
    var bb: f64[>..] = [];
    var ee: f64[>..] = [];
    let ns = ivp_tableau(rk45, cc, aa, bb, ee);
    var y: f64[>..] = [];
    var i: i64 = 0;
    while i < n { y.push(y0[i]); i += 1; }
    var t = t0;
    let f00 = F(t, copy(y));
    if f00.len != n {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
    }
    var f: f64[>..] = [];
    i = 0;
    while i < n { f.push(f00[i]); i += 1; }
    // The first step.
    var scale: f64[>..] = [];
    var q1: f64[>..] = [];
    var q2: f64[>..] = [];
    i = 0;
    while i < n { scale.push(atol + mathshim::abs_of(y[i]) * tol); q1.push(y[i] / scale[i]); q2.push(f[i] / scale[i]); i += 1; }
    let d0 = ivp_rms(q1);
    let d1 = ivp_rms(q2);
    var h0: f64 = 0.000001;
    if !(d0 < 0.00001 || d1 < 0.00001) { h0 = 0.01 * d0 / d1; }
    var y1: f64[>..] = [];
    i = 0;
    while i < n { y1.push(y[i] + h0 * dir * f[i]); i += 1; }
    let f1 = F(t + h0 * dir, copy(y1));
    if f1.len != n {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
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
    let span = mathshim::abs_of(t1 - t0);
    if span < h_abs { h_abs = span; }
    if max_step < h_abs { h_abs = max_step; }
    var ts: f64[>..] = [t];
    var ys: f64[>..] = [];
    i = 0;
    while i < n { ys.push(y[i]); i += 1; }
    var rejected = false;
    var steps: i64 = 0;
    while dir * (t1 - t) > 0.0 {
        steps += 1;
        if steps > 1000000 {
            return empty_arr(), empty_arr(), fault(F_LIMIT, "solve_ivp", "the count of steps reached the limit");
        }
        var min_step = 10.0 * mathshim::abs_of(t) * 2.220446049250313e-16;
        if min_step < 5.0e-323 { min_step = 5.0e-323; }
        if h_abs < min_step { h_abs = min_step; }
        if h_abs > max_step { h_abs = max_step; }
        var accepted = false;
        var fnew: f64[>..] = [];
        var ynew: f64[>..] = [];
        var tnew: f64 = t;
        while !accepted {
            if h_abs < min_step {
                return empty_arr(), empty_arr(), fault(F_VALUE, "solve_ivp", "the step fell below the smallest step");
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
                if ks.len != n {
                    return empty_arr(), empty_arr(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
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
            if fn1.len != n {
                return empty_arr(), empty_arr(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
            }
            fnew = [];
            i = 0;
            while i < n { fnew.push(fn1[i]); i += 1; }
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
            let err = ivp_rms(sq);
            if err < 1.0 {
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
        t = tnew;
        y = copy(ynew);
        f = copy(fnew);
        ts.push(t);
        i = 0;
        while i < n { ys.push(y[i]); i += 1; }
    }
    return line_arr(copy(ts)), mat2(copy(ys), ts.len, n), no_fault();
}

// quad_finite tells whether a number is neither infinite nor missing.
fn quad_finite(v: f64) -> bool { return v - v == 0.0; }

// dblquad_inner integrates the function over y for one x, between two
// finite limits.
fn dblquad_inner<F: fn(f64, f64) -> f64>(x: f64, lo: f64, hi: f64, epsabs: f64, epsrel: f64, limit: i64) -> f64, f64, Fault {
    if lo == hi { return 0.0, 0.0, no_fault(); }
    var a = lo;
    var b = hi;
    var sgn: f64 = 1.0;
    if lo > hi { a = hi; b = lo; sgn = -1.0; }
    let v, e, f = quad_run(0, 0.0, a, b, epsabs, epsrel, limit) { y => F(y, x) };
    return sgn * v, e, f;
}

// dblquad gives the double integral of a function f(y, x) over x from a
// to b and over y from g(x) to h(x). The limits are finite. The limits
// of y are two named functions of the caller. The function f is a block.
// The call gives the value and the sum of the errors of the inner
// integrals. A fault of an inner integral stops the call.
fn dblquad<G: fn(f64) -> f64, H: fn(f64) -> f64, F: fn(f64, f64) -> f64>(a: f64, b: f64, epsabs: f64 = 1.49e-8, epsrel: f64 = 1.49e-8, limit: i64 = 50) -> f64, f64, Fault {
    if a != a || b != b || !quad_finite(a) || !quad_finite(b) {
        return 0.0, 0.0, fault(F_VALUE, "dblquad", "a limit of x holds a value that is not finite");
    }
    if !(epsabs >= 0.0) || !(epsrel >= 0.0) || (epsabs == 0.0 && epsrel < 5.0e-29) {
        return 0.0, 0.0, fault(F_VALUE, "dblquad", "the tolerance holds a value that no result can meet");
    }
    if limit < 1 || limit > MAX_QUAD_LIMIT {
        return 0.0, 0.0, fault(F_RANGE, "dblquad", "the limit stands outside one to the largest count");
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
        var r: f64 = 0.0;
        if !quad_finite(g) || !quad_finite(h) {
            bad = fault(F_VALUE, "dblquad", "a limit of y holds a value that is not finite");
        } else {
            let iv, ie, ifl = dblquad_inner(x, g, h, epsabs, epsrel, limit) { y, xx => F(y, xx) };
            if ifl.kind != F_NONE { bad = ifl; }
            err += ie;
            r = iv;
        }
        r
    };
    if bad.kind != F_NONE { return 0.0, 0.0, bad; }
    return sgn * v, e + err, f;
}
