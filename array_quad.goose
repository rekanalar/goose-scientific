// The quadrature: the Gauss and Kronrod rule, the nodes and the weights.
//
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

