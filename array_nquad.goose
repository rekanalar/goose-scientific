// Integrals of N variables and of vector-valued functions. nquad nests
// the adaptive rule of quad over N ranges. A block cannot call itself,
// so the nesting runs as a stack of states, one state for each range.
// quad_vec rates the Gauss and Kronrod rule of 21 points on a vector.
namespace scientific;

import mathshim;

// The largest count of ranges that nquad takes.
const MAX_NQUAD_RANGES: i64 = 8;

// The largest count of evaluations of the integrand that nquad takes.
const MAX_NQUAD_EVALS: i64 = 20000000;

// The largest product of the count of values and the count of
// intervals that quad_vec holds.
const MAX_QUAD_VEC_CELLS: i64 = 20000000;

// quad_map gives the point and the weight of the node t. The mode 0 is
// a finite range. The modes 1 and 2 map the rays above and below lo.
// The mode 3 maps the whole line from the range of t from -1 to 1.
fn quad_map(mode: i64, lo: f64, t: f64) -> f64, f64 {
    if mode == 0 { return t, 1.0; }
    if mode == 3 {
        let d = 1.0 - t * t;
        return t / d, (1.0 + t * t) / (d * d);
    }
    let u = (1.0 - t) / t;
    if mode == 1 { return lo + u, 1.0 / (t * t); }
    return lo - u, 1.0 / (t * t);
}

// quad_tolerance_fault checks the tolerances and the limit.
fn quad_tolerance_fault(tag: const u8[:], epsabs: f64, epsrel: f64, limit: i64) -> Fault {
    if !(epsabs >= 0.0) || !(epsrel >= 0.0) || (epsabs == 0.0 && epsrel < 5.0e-29) {
        return fault(F_VALUE, tag, "the tolerance holds a value that no result can meet");
    }
    if limit < 1 || limit > MAX_QUAD_LIMIT {
        return fault(F_RANGE, tag, "the limit stands outside one to the largest count");
    }
    return no_fault();
}

// nquad_call gives the integrand at a point. The point is a view, so
// the block takes a slice on every version of the compiler.
fn nquad_call<F: fn(f64[:]) -> f64>(point: f64[:]) -> f64 { return F(point); }

// nquad integrates a function of N variables. The row ranges holds the
// pairs lo and hi, one pair for each variable. The first pair belongs
// to the innermost variable, and the function takes the point in the
// same order: the first value of the point is the first variable. An
// end may be infinite. The call gives the value and the error estimate
// of the outermost integral. The tolerances and the limit hold for
// every range, as in quad. At most eight ranges and twenty million
// evaluations are allowed.
fn nquad<F: fn(f64[:]) -> f64>(ranges: f64[:], epsabs: f64 = 1.49e-8, epsrel: f64 = 1.49e-8, limit: i64 = 50) -> f64, f64, Fault {
    if ranges.len == 0 || ranges.len % 2 != 0 {
        return 0.0, 0.0, fault(F_SHAPE, "nquad", "the ranges hold no pair or an odd count of values");
    }
    let n = ranges.len / 2;
    if n > MAX_NQUAD_RANGES {
        return 0.0, 0.0, fault(F_LIMIT, "nquad", "the count of ranges is above the limit of eight");
    }
    let tf = quad_tolerance_fault("nquad", epsabs, epsrel, limit);
    if tf.kind != F_NONE { return 0.0, 0.0, tf; }
    var mode: i64[>..] = [];
    var sgn: f64[>..] = [];
    var lo: f64[>..] = [];
    var ra: f64[>..] = [];
    var rb: f64[>..] = [];
    var l: i64 = 0;
    while l < n {
        let a = ranges[2 * l];
        let b = ranges[2 * l + 1];
        if a != a || b != b {
            return 0.0, 0.0, fault(F_VALUE, "nquad", "a limit holds the missing value");
        }
        if a == b { return 0.0, 0.0, no_fault(); }
        var s: f64 = 1.0;
        var x = a;
        var y = b;
        if a > b { x = b; y = a; s = -1.0; }
        let ia = quad_is_inf(x);
        let ib = quad_is_inf(y);
        var m: i64 = 0;
        var o: f64 = 0.0;
        var u: f64 = 0.0;
        var v: f64 = 1.0;
        if ia && ib { m = 3; u = -1.0; }
        else if ib { m = 1; o = x; }
        else if ia { m = 2; o = y; }
        else { u = x; v = y; }
        mode.push(m);
        sgn.push(s);
        lo.push(o);
        ra.push(u);
        rb.push(v);
        l += 1;
    }
    let cap = limit + 1;
    var ea: f64[>..] = [];
    var eb: f64[>..] = [];
    var ev: f64[>..] = [];
    var ee: f64[>..] = [];
    var i: i64 = 0;
    while i < n * cap { ea.push(0.0); eb.push(0.0); ev.push(0.0); ee.push(0.0); i += 1; }
    var fv: f64[>..] = [];
    var wt: f64[>..] = [];
    var pa: f64[>..] = [];
    var pb: f64[>..] = [];
    var rv: f64[>..] = [];
    var re: f64[>..] = [];
    var point: f64[>..] = [];
    var cnt: i64[>..] = [];
    var jn: i64[>..] = [];
    var np: i64[>..] = [];
    var pk: i64[>..] = [];
    var worst: i64[>..] = [];
    i = 0;
    while i < n {
        point.push(0.0);
        wt.push(1.0);
        cnt.push(0);
        jn.push(0);
        np.push(1);
        pk.push(0);
        worst.push(-1);
        i += 1;
    }
    i = 0;
    while i < 21 * n { fv.push(0.0); i += 1; }
    i = 0;
    while i < 2 * n { pa.push(0.0); pb.push(0.0); rv.push(0.0); re.push(0.0); i += 1; }
    var cur = n - 1;
    var fresh = true;
    var evals: i64 = 0;
    var result: f64 = 0.0;
    var rerr: f64 = 0.0;
    var done = false;
    while !done {
        let c = cur;
        if fresh {
            cnt[c] = 0;
            worst[c] = -1;
            np[c] = 1;
            pk[c] = 0;
            jn[c] = 0;
            pa[2 * c] = ra[c];
            pb[2 * c] = rb[c];
            fresh = false;
        }
        let jj = jn[c];
        if jj < 21 {
            let p = pk[c];
            let a = pa[2 * c + p];
            let b = pb[2 * c + p];
            let center = 0.5 * (a + b);
            let half = 0.5 * (b - a);
            var t = center - half * GK_NODES[jj % 11];
            if jj >= 11 { t = center + half * GK_NODES[jj - 11]; }
            let x, w = quad_map(mode[c], lo[c], t);
            point[c] = x;
            wt[c] = w;
            if c == 0 {
                evals += 1;
                if evals > MAX_NQUAD_EVALS {
                    return 0.0, 0.0, fault(F_LIMIT, "nquad", "the count of evaluations reached the limit");
                }
                fv[21 * c + jj] = nquad_call(point) { p => F(p) } * w;
                jn[c] = jj + 1;
            } else {
                cur = c - 1;
                fresh = true;
            }
        } else {
            let p = pk[c];
            let half = 0.5 * (pb[2 * c + p] - pa[2 * c + p]);
            let v, e = gk_combine(fv, 21 * c, half);
            rv[2 * c + p] = v;
            re[2 * c + p] = e;
            pk[c] = p + 1;
            jn[c] = 0;
            if pk[c] >= np[c] {
                let ob = c * cap;
                if worst[c] < 0 {
                    ea[ob] = ra[c];
                    eb[ob] = rb[c];
                    ev[ob] = rv[2 * c];
                    ee[ob] = re[2 * c];
                    cnt[c] = 1;
                } else {
                    let w = worst[c];
                    eb[ob + w] = pb[2 * c];
                    ev[ob + w] = rv[2 * c];
                    ee[ob + w] = re[2 * c];
                    let k = cnt[c];
                    ea[ob + k] = pa[2 * c + 1];
                    eb[ob + k] = pb[2 * c + 1];
                    ev[ob + k] = rv[2 * c + 1];
                    ee[ob + k] = re[2 * c + 1];
                    cnt[c] = k + 1;
                }
                var total: f64 = 0.0;
                var toterr: f64 = 0.0;
                var big: i64 = 0;
                var q: i64 = 0;
                while q < cnt[c] {
                    total += ev[ob + q];
                    toterr += ee[ob + q];
                    if ee[ob + q] > ee[ob + big] { big = q; }
                    q += 1;
                }
                worst[c] = big;
                if toterr <= epsabs || toterr <= epsrel * mathshim::abs_of(total) {
                    let res = sgn[c] * total;
                    if c == n - 1 {
                        result = res;
                        rerr = toterr;
                        done = true;
                    } else {
                        cur = c + 1;
                        let d = cur;
                        fv[21 * d + jn[d]] = res * wt[d];
                        jn[d] = jn[d] + 1;
                    }
                } else if cnt[c] >= limit {
                    return 0.0, 0.0, fault(F_LIMIT, "nquad", "the count of intervals reached the limit before the error fell below the tolerance");
                } else {
                    let mid = 0.5 * (ea[ob + big] + eb[ob + big]);
                    pa[2 * c] = ea[ob + big];
                    pb[2 * c] = mid;
                    pa[2 * c + 1] = mid;
                    pb[2 * c + 1] = eb[ob + big];
                    np[c] = 2;
                    pk[c] = 0;
                }
            }
        }
    }
    return result, rerr, no_fault();
}

// qv_norm gives the norm of the m values of v from the offset off. The
// flag maxnorm picks the largest absolute value. The other norm is the
// Euclidean norm.
fn qv_norm(v: f64[:], off: i64, m: i64, maxnorm: bool) -> f64 {
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < m {
        let a = mathshim::abs_of(v[off + i]);
        if maxnorm { if a > s { s = a; } } else { s += a * a; }
        i += 1;
    }
    if maxnorm { return s; }
    return mathshim::root(s);
}

// qv_rule rates the rule of 21 points on [a, b] for a function of the
// caller that gives a vector. It fills res with the integral and gives
// the error estimate. The count m is -1 before the first
// call, and the call sets it. A vector of another length is a fault.
fn qv_rule<F: fn(f64, f64[>..]&)>(mode: i64, lo: f64, a: f64, b: f64, m: i64&, res: f64[>..]&, maxnorm: bool, fv: f64[>..]&) -> f64, Fault {
    let center = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    var j: i64 = 0;
    while j < 21 {
        var t = center - half * GK_NODES[j % 11];
        if j >= 11 { t = center + half * GK_NODES[j - 11]; }
        let x, w = quad_map(mode, lo, t);
        var out: f64[>..] = [];
        F(x, &out);
        if m < 0 {
            if out.len == 0 {
                return 0.0, fault(F_SHAPE, "quad_vec", "the function gives a vector with no value");
            }
            m = out.len;
            if m * 21 > MAX_QUAD_VEC_CELLS {
                return 0.0, fault(F_LIMIT, "quad_vec", "the vector holds too many values");
            }
            var z: i64 = 0;
            while z < m * 21 { fv.push(0.0); z += 1; }
        }
        while res.len < m { res.push(0.0); }
        if out.len != m {
            return 0.0, fault(F_SHAPE, "quad_vec", "the function changes the length of its vector");
        }
        var k: i64 = 0;
        while k < m { fv[j * m + k] = out[k] * w; k += 1; }
        j += 1;
    }
    var errv: f64[>..] = [];
    var k: i64 = 0;
    while k < m {
        let fc = fv[10 * m + k];
        var resk = GK_WEIGHTS[10] * fc;
        var resg: f64 = 0.0;
        j = 0;
        while j < 10 {
            let s = fv[j * m + k] + fv[(11 + j) * m + k];
            resk += GK_WEIGHTS[j] * s;
            if j % 2 == 1 { resg += GAUSS_WEIGHTS[j / 2] * s; }
            j += 1;
        }
        res[k] = resk * half;
        errv.push((resk - resg) * half);
        k += 1;
    }
    return qv_norm(errv, 0, m, maxnorm), no_fault();
}

// quad_vec integrates a function that gives a vector. The block takes
// the point and a row to fill: { x, out => out.push(a); out.push(b); }.
// The row holds the same count of values for every point. The range may
// reach to infinity at either end. The norm is "2" or "max". The
// bisection stops when the error of the norm falls to the larger of
// epsabs and epsrel times the norm of the value. The call gives the
// vector, the error estimate and the fault. The fault F_LIMIT means
// that the count of intervals reached the limit.
fn quad_vec<F: fn(f64, f64[>..]&)>(a: f64, b: f64, epsabs: f64 = 1.0e-200, epsrel: f64 = 1.0e-8, limit: i64 = 10000, norm: const u8[:] = "2") -> Arr, f64, Fault {
    if a != a || b != b {
        return empty_arr(), 0.0, fault(F_VALUE, "quad_vec", "a limit holds the missing value");
    }
    var maxnorm = false;
    if norm == "max" { maxnorm = true; }
    else if norm != "2" {
        return empty_arr(), 0.0, fault(F_VALUE, "quad_vec", "the norm is not 2 or max");
    }
    if !(epsabs >= 0.0) || !(epsrel >= 0.0) {
        return empty_arr(), 0.0, fault(F_VALUE, "quad_vec", "the tolerance holds a value that no result can meet");
    }
    if limit < 1 || limit > MAX_QUAD_LIMIT {
        return empty_arr(), 0.0, fault(F_RANGE, "quad_vec", "the limit stands outside one to the largest count");
    }
    var lo = a;
    var hi = b;
    var sgn: f64 = 1.0;
    if a > b { lo = b; hi = a; sgn = -1.0; }
    let ia = quad_is_inf(lo);
    let ib = quad_is_inf(hi);
    var mode: i64 = 0;
    var off: f64 = 0.0;
    var ra = lo;
    var rb = hi;
    if ia && ib { mode = 3; ra = -1.0; rb = 1.0; }
    else if ib { mode = 1; off = lo; ra = 0.0; rb = 1.0; }
    else if ia { mode = 2; off = hi; ra = 0.0; rb = 1.0; }
    var m: i64 = -1;
    var fv: f64[>..] = [];
    var ea: f64[>..] = [ra];
    var eb: f64[>..] = [rb];
    var ee: f64[>..] = [];
    var ev: f64[>..] = [];
    var total: f64[>..] = [];
    var part: f64[>..] = [];
    var toterr: f64 = 0.0;
    var worst: i64 = 0;
    var first = true;
    var tnorm: f64 = 0.0;
    if lo == hi {
        // The function still gives the count of values.
        var out: f64[>..] = [];
        F(lo, &out);
        var z: f64[>..] = [];
        var q: i64 = 0;
        while q < out.len { z.push(0.0); q += 1; }
        if out.len == 0 { return empty_arr(), 0.0, fault(F_SHAPE, "quad_vec", "the function gives a vector with no value"); }
        return line_arr(copy(z)), 0.0, no_fault();
    }
    var pa: f64[>..] = [ra];
    var pb: f64[>..] = [rb];
    while first || (toterr > epsabs && toterr > epsrel * tnorm) {
        if !first && ea.len >= limit {
            return empty_arr(), toterr, fault(F_LIMIT, "quad_vec", "the count of intervals reached the limit before the error fell below the tolerance");
        }
        if !first {
            let w = worst;
            let mid = 0.5 * (ea[w] + eb[w]);
            pa = [ea[w], mid];
            pb = [mid, eb[w]];
        }
        var k: i64 = 0;
        var newv: f64[>..] = [];
        var newe: f64[>..] = [];
        while k < pa.len {
            part = [];
            let e, fl = qv_rule(mode, off, pa[k], pb[k], m, part, maxnorm, fv) { x, o => F(x, o) };
            if fl.kind != F_NONE { return empty_arr(), 0.0, fl; }
            var z: i64 = 0;
            while z < m { newv.push(part[z]); z += 1; }
            newe.push(e);
            k += 1;
        }
        if first {
            var z: i64 = 0;
            while z < m { total.push(0.0); ev.push(newv[z]); total[z] = newv[z]; z += 1; }
            ee.push(newe[0]);
            toterr = newe[0];
        } else {
            let w = worst;
            var z: i64 = 0;
            while z < m {
                total[z] = total[z] - ev[w * m + z] + newv[z] + newv[m + z];
                ev[w * m + z] = newv[z];
                ev.push(newv[m + z]);
                z += 1;
            }
            toterr = toterr - ee[w] + newe[0] + newe[1];
            ea.push(pa[1]);
            eb[w] = pb[0];
            eb.push(pb[1]);
            ee[w] = newe[0];
            ee.push(newe[1]);
        }
        first = false;
        tnorm = qv_norm(total, 0, m, maxnorm);
        worst = 0;
        var i: i64 = 1;
        while i < ee.len { if ee[i] > ee[worst] { worst = i; } i += 1; }
        if ea.len * m > MAX_QUAD_VEC_CELLS {
            return empty_arr(), toterr, fault(F_LIMIT, "quad_vec", "the vector holds too many values");
        }
    }
    var out: f64[>..] = [];
    var z: i64 = 0;
    while z < m { out.push(sgn * total[z]); z += 1; }
    return line_arr(copy(out)), toterr, no_fault();
}
