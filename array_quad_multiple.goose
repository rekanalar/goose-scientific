// The dense output of the ODE solver, odeint and the multiple integrals.
//
namespace scientific;

import mathshim;

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
        if kind == 4 {
            var kq: f64[>..] = [];
            var iq: i64 = 0;
            while iq < 3 * n { kq.push(hk[si * 3 * n + iq]); iq += 1; }
            var yo: f64[>..] = [];
            iq = 0;
            while iq < n { yo.push(hy[si * n + iq]); iq += 1; }
            let yv = rad_dense(n, yo, kq, xs);
            ts.push(x);
            iq = 0;
            while iq < n { ys.push(yv[iq]); iq += 1; }
            j += 1;
            continue;
        }
        if kind == 3 {
            var kb: f64[>..] = [];
            var i0: i64 = 0;
            while i0 < 9 * n { kb.push(hk[si * 9 * n + i0]); i0 += 1; }
            let yv = bdf_poly(n, kb, x);
            ts.push(x);
            var i: i64 = 0;
            while i < n { ys.push(yv[i]); i += 1; }
            j += 1;
            continue;
        }
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
