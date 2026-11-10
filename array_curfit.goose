// The smoothing spline of Dierckx. The port follows the routines
// fpcurf, fpbspl, fpback, fpdisc, fpknot, fpgivs, fprota and fprati
// of the FITPACK library. The loop counts hold the one-based
// arithmetic of the original, and every read of a two dimensional
// buffer uses its column-major order.
namespace scientific;

import mathshim;

// curfit_fit gives the knots, the coefficients, the weighted sum of
// squared residuals and the code of the smoothing spline of the data
// with a smoothing factor above zero. The points of x must rise, the
// weights must hold positive numbers, and the degree runs from one
// to five.
fn curfit_fit(x: f64[:], y: f64[:], w: f64[:], k: i64, s: f64) -> f64[>..], f64[>..], f64, i64, Fault {
    let m = x.len;
    let k1 = k + 1;
    var et: f64[>..] = [];
    var ec: f64[>..] = [];
    if k < 1 || k > 5 {
        return et, ec, 0.0, 10, fault(F_RANGE, "splrep", "the degree is outside one to five");
    }
    if m < k1 {
        return et, ec, 0.0, 10, fault(F_SHAPE, "splrep", "the data hold too few points for the degree");
    }
    var i: i64 = 0;
    while i < m {
        if !(w[i] - w[i] == 0.0) || w[i] <= 0.0 {
            return et, ec, 0.0, 10, fault(F_VALUE, "splrep", "a weight is not a positive number");
        }
        if !(x[i] - x[i] == 0.0) || !(y[i] - y[i] == 0.0) {
            return et, ec, 0.0, 10, fault(F_VALUE, "splrep", "a point or a value is not finite");
        }
        if i > 0 && x[i-1] >= x[i] {
            return et, ec, 0.0, 10, fault(F_VALUE, "splrep", "the points of x fall or hold the same value twice");
        }
        i += 1;
    }
    if !(s - s == 0.0) || s < 0.0 {
        return et, ec, 0.0, 10, fault(F_VALUE, "splrep", "the smoothing factor is not finite or holds a value below zero");
    }
    let nest = m + k1;
    let n, t, c, fp, ier, f = fpcurf(x, y, w, k, s, nest, x[0], x[m-1]);
    if f.kind != F_NONE { return et, ec, 0.0, 10, f; }
    var tt: f64[>..] = [];
    var cc: f64[>..] = [];
    i = 0;
    while i < n { tt.push(t[i]); i += 1; }
    i = 0;
    while i < n - k1 { cc.push(c[i]); i += 1; }
    return tt, cc, fp, ier, no_fault();
}

// fpcurf is the port of fpcurf in the computation mode iopt is zero.
// It gives the count of knots, the knots, the coefficients, the
// weighted sum of squared residuals and the code of the original.
fn fpcurf(x: f64[:], y: f64[:], w: f64[:], k: i64, s: f64, nest: i64, xb: f64, xe: f64) -> i64, f64[>..], f64[>..], f64, i64, Fault {
    let m = x.len;
    let k1 = k + 1;
    let k2 = k1 + 1;
    let maxit: i64 = 20;
    let con1: f64 = 0.1;
    let con9: f64 = 0.9;
    let con4: f64 = 0.04;
    let nmin = 2 * k1;
    let acc = 0.001 * s;
    let nmax = m + k1;
    var t: f64[>..] = [];
    var c: f64[>..] = [];
    var fpint: f64[>..] = [];
    var z: f64[>..] = [];
    var a: f64[>..] = [];
    var g: f64[>..] = [];
    var b: f64[>..] = [];
    var q: f64[>..] = [];
    var nrdata: i64[>..] = [];
    var i: i64 = 0;
    while i < nest { t.push(0.0); c.push(0.0); fpint.push(0.0); z.push(0.0); nrdata.push(0); i += 1; }
    i = 0;
    while i < nest * k1 { a.push(0.0); i += 1; }
    i = 0;
    while i < nest * k2 { g.push(0.0); b.push(0.0); i += 1; }
    i = 0;
    while i < m * k1 { q.push(0.0); i += 1; }
    var n = nmin;
    var fpold: f64 = 0.0;
    var nplus: i64 = 0;
    var fp0: f64 = 0.0;
    var fp: f64 = 0.0;
    var fpms: f64 = 0.0;
    var ier: i64 = -2;
    nrdata[0] = m - 2;
    var phase_two = false;
    var iter: i64 = 0;
    while !phase_two {
        iter += 1;
        if iter > m { phase_two = true; break; }
        if n == nmin { ier = -2; }
        var nrint = n - nmin + 1;
        let nk1 = n - k1;
        i = 0;
        while i < k1 {
            t[i] = xb;
            t[n - 1 - i] = xe;
            i += 1;
        }
        fp = 0.0;
        i = 0;
        while i < nk1 {
            z[i] = 0.0;
            var j: i64 = 0;
            while j < k1 { a[j * nest + i] = 0.0; j += 1; }
            i += 1;
        }
        var l: i64 = k1;
        var it: i64 = 0;
        while it < m {
            let xi = x[it];
            let wi = w[it];
            var yi = y[it] * wi;
            while !(xi < t[l] || l == nk1) { l += 1; }
            let h = fpbspl_of(t, k, xi, l);
            i = 0;
            while i < k1 {
                q[i * m + it] = h[i];
                h[i] = h[i] * wi;
                i += 1;
            }
            var j: i64 = l - k1;
            i = 0;
            while i < k1 {
                j += 1;
                let piv = h[i];
                if piv != 0.0 {
                    let at = j - 1;
                    var ww = a[at];
                    var store = piv;
                    if store < 0.0 { store = -store; }
                    var dd: f64 = 0.0;
                    if store >= ww { dd = store * mathshim::root(1.0 + (ww / piv) * (ww / piv)); }
                    if store < ww { dd = ww * mathshim::root(1.0 + (piv / ww) * (piv / ww)); }
                    let cosg = ww / dd;
                    let sing = piv / dd;
                    a[at] = dd;
                    var stor1 = yi;
                    var stor2 = z[j - 1];
                    z[j - 1] = cosg * stor2 + sing * stor1;
                    yi = cosg * stor1 - sing * stor2;
                    if i < k1 - 1 {
                        var i2: i64 = 1;
                        var i1: i64 = i + 1;
                        while i1 < k1 {
                            i2 += 1;
                            stor1 = h[i1];
                            stor2 = a[(i2 - 1) * nest + (j - 1)];
                            a[(i2 - 1) * nest + (j - 1)] = cosg * stor2 + sing * stor1;
                            h[i1] = cosg * stor1 - sing * stor2;
                            i1 += 1;
                        }
                    }
                }
                i += 1;
            }
            fp = fp + yi * yi;
            it += 1;
        }
        if ier == -2 { fp0 = fp; }
        fpint[n - 1] = fp0;
        fpint[n - 2] = fpold;
        nrdata[n - 1] = nplus;
        fpback_of(a, z, nk1, k1, c, nest);
        fpms = fp - s;
        if fpms > -acc && fpms < acc {
            return n, t, c, fp, ier, no_fault();
        }
        if fpms < 0.0 { phase_two = true; break; }
        if n == nmax { return n, t, c, fp, -1, no_fault(); }
        if n == nest { return n, t, c, fp, 1, no_fault(); }
        if ier != 0 {
            nplus = 1;
            ier = 0;
        } else {
            let npl1a = nplus * 2;
            var npl1 = npl1a;
            if fpold - fp > acc { npl1 = ((nplus as f64) * fpms / (fpold - fp)) as i64; }
            var mx = npl1;
            if nplus / 2 > mx { mx = nplus / 2; }
            if 1 > mx { mx = 1; }
            if npl1a < mx { nplus = npl1a; } else { nplus = mx; }
        }
        fpold = fp;
        var fpart: f64 = 0.0;
        var ii: i64 = 1;
        l = k2;
        var fresh = false;
        it = 0;
        while it < m {
            if !(x[it] < t[l - 1] || l > nk1) {
                fresh = true;
                l += 1;
            }
            var term: f64 = 0.0;
            var l0 = l - k2;
            i = 0;
            while i < k1 {
                l0 += 1;
                term = term + c[l0 - 1] * q[i * m + it];
                i += 1;
            }
            let d0 = w[it] * (term - y[it]);
            term = d0 * d0;
            fpart = fpart + term;
            if fresh {
                let half = term * 0.5;
                fpint[ii - 1] = fpart - half;
                ii += 1;
                fpart = half;
                fresh = false;
            }
            it += 1;
        }
        fpint[nrint - 1] = fpart;
        var restart_interp = false;
        var lp: i64 = 0;
        while lp < nplus {
            let rn = fpknot_of(x, t, fpint, nrdata, n, nrint, nest, 1);
            n = rn.n;
            nrint = rn.nrint;
            if n == nmax { restart_interp = true; break; }
            if n == nest { break; }
            lp += 1;
        }
        if restart_interp {
            let mk1 = m - k1;
            let k3 = k / 2;
            i = k2;
            var j: i64 = k3 + 2;
            if k3 * 2 != k {
                var l2: i64 = 0;
                while l2 < mk1 { t[i - 1] = x[j - 1]; i += 1; j += 1; l2 += 1; }
            } else {
                var l2: i64 = 0;
                while l2 < mk1 { t[i - 1] = (x[j - 1] + x[j - 2]) * 0.5; i += 1; j += 1; l2 += 1; }
            }
            iter = 0;
            continue;
        }
    }
    if ier == -2 {
        return n, t, c, fp0, ier, no_fault();
    }
    // Part two: the smoothing spline of the accepted set of knots.
    let nk1 = n - k1;
    let n8 = n - nmin;
    fpdisc_of(t, n, k2, b, nest);
    var p1: f64 = 0.0;
    var f1 = fp0 - s;
    var p3: f64 = -1.0;
    var f3 = fpms;
    var p: f64 = 0.0;
    i = 0;
    while i < nk1 { p = p + a[i]; i += 1; }
    p = (nk1 as f64) / p;
    var ich1 = false;
    var ich3 = false;
    iter = 0;
    while iter < maxit {
        iter += 1;
        let pinv = 1.0 / p;
        i = 0;
        while i < nk1 {
            c[i] = z[i];
            g[(k2 - 1) * nest + i] = 0.0;
            var j: i64 = 0;
            while j < k1 {
                g[j * nest + i] = a[j * nest + i];
                j += 1;
            }
            i += 1;
        }
        var it: i64 = 0;
        while it < n8 {
            var h: f64[>..] = [];
            i = 0;
            while i < k2 { h.push(b[i * nest + it] * pinv); i += 1; }
            var yi: f64 = 0.0;
            var j: i64 = it;
            while j < nk1 {
                let piv = h[0];
                let at = j;
                var ww = g[at];
                var store = piv;
                if store < 0.0 { store = -store; }
                var dd: f64 = 0.0;
                if store >= ww { dd = store * mathshim::root(1.0 + (ww / piv) * (ww / piv)); }
                if store < ww { dd = ww * mathshim::root(1.0 + (piv / ww) * (piv / ww)); }
                let cosg = ww / dd;
                let sing = piv / dd;
                g[at] = dd;
                var stor1 = yi;
                var stor2 = c[j];
                c[j] = cosg * stor2 + sing * stor1;
                yi = cosg * stor1 - sing * stor2;
                if j == nk1 - 1 { break; }
                var i2 = k1;
                if j >= n8 { i2 = nk1 - j - 1; }
                i = 0;
                while i < i2 {
                    let i1 = i + 1;
                    stor1 = h[i1];
                    stor2 = g[i1 * nest + j];
                    g[i1 * nest + j] = cosg * stor2 + sing * stor1;
                    h[i1] = cosg * stor1 - sing * stor2;
                    h[i] = h[i1];
                    i += 1;
                }
                h[i2] = 0.0;
                j += 1;
            }
            it += 1;
        }
        fpback_of(g, c, nk1, k2, c, nest);
        fp = 0.0;
        var l = k2;
        it = 0;
        while it < m {
            if !(x[it] < t[l - 1] || l > nk1) {
                l += 1;
            }
            var term: f64 = 0.0;
            var l0 = l - k2;
            i = 0;
            while i < k1 {
                l0 += 1;
                term = term + c[l0 - 1] * q[i * m + it];
                i += 1;
            }
            fp = fp + (w[it] * (term - y[it])) * (w[it] * (term - y[it]));
            it += 1;
        }
        fpms = fp - s;
        if fpms > -acc && fpms < acc {
            return n, t, c, fp, 0, no_fault();
        }
        if iter == maxit {
            return n, t, c, fp, 3, no_fault();
        }
        let p2 = p;
        let f2 = fpms;
        var step_small = false;
        if !ich3 {
            if f2 - f3 > acc {
                if f2 < 0.0 { ich3 = true; }
            } else {
                p3 = p2;
                f3 = f2;
                p = p * con4;
                if p <= p1 { p = p1 * con9 + p2 * con1; }
                step_small = true;
            }
        }
        if step_small { continue; }
        step_small = false;
        if !ich1 {
            if f1 - f2 > acc {
                if f2 > 0.0 { ich1 = true; }
            } else {
                p1 = p2;
                f1 = f2;
                p = p / con4;
                if p3 < 0.0 { step_small = true; }
                else {
                    if p >= p3 { p = p2 * con1 + p3 * con9; }
                    step_small = true;
                }
            }
        }
        if step_small { continue; }
        if f2 >= f1 || f2 <= f3 {
            return n, t, c, fp, 2, no_fault();
        }
        let rr = fprati_of(p1, f1, p2, f2, p3, f3);
        p = rr.p;
        p1 = rr.p1;
        f1 = rr.f1;
        p3 = rr.p3;
        f3 = rr.f3;
    }
    return n, t, c, fp, 3, no_fault();
}

// fpbspl_of gives the k plus one non-zero b-splines of the degree at
// the point x in the knot interval l.
fn fpbspl_of(t: f64[:], k: i64, x: f64, l: i64) -> f64[>..] {
    var h: f64[>..] = [];
    h.push(1.0);
    var j: i64 = 1;
    while j <= k {
        var hh: f64[>..] = [];
        var i2: i64 = 0;
        while i2 < j { hh.push(h[i2]); i2 += 1; }
        h[0] = 0.0;
        var i: i64 = 1;
        while i <= j {
            let li = l + i;
            let lj = li - j;
            let f = hh[i - 1] / (t[li - 1] - t[lj - 1]);
            h[i - 1] = h[i - 1] + f * (t[li - 1] - x);
            if i == j {
                h.push(f * (x - t[lj - 1]));
            } else {
                h[i] = f * (x - t[lj - 1]);
            }
            i += 1;
        }
        j += 1;
    }
    return h;
}

// fpback_of solves the system of the upper triangular matrix a of
// the bandwidth k. The solution goes into c.
fn fpback_of(a: f64[:], z: f64[:], n: i64, k: i64, c: f64[>..]&, nest: i64) {
    let k1 = k - 1;
    c[n - 1] = z[n - 1] / a[n - 1];
    var i = n - 2;
    if i < 0 { return; }
    var j: i64 = 2;
    while j <= n {
        var store = z[i];
        var i1 = k1;
        if j <= k1 { i1 = j - 1; }
        var m = i;
        var l: i64 = 1;
        while l <= i1 {
            m += 1;
            store = store - c[m] * a[l * nest + i];
            l += 1;
        }
        c[i] = store / a[i];
        i -= 1;
        j += 1;
    }
}

// fpdisc_of gives the jumps of the kth derivative of the b-splines
// at the inner knots.
fn fpdisc_of(t: f64[:], n: i64, k2: i64, b: f64[>..]&, nest: i64) {
    let k1 = k2 - 1;
    let k = k1 - 1;
    let nk1 = n - k1;
    let nrint = nk1 - k;
    let fac = nrint as f64 / (t[nk1] - t[k1 - 1]);
    var h: f64[>..] = [];
    var i: i64 = 0;
    while i < 14 { h.push(0.0); i += 1; }
    var l: i64 = k2;
    while l <= nk1 {
        let lmk = l - k1;
        var j: i64 = 1;
        while j <= k1 {
            let ik = j + k1;
            let lj = l + j;
            let lk = lj - k2;
            h[j - 1] = t[l - 1] - t[lk - 1];
            h[ik - 1] = t[l - 1] - t[lj - 1];
            j += 1;
        }
        var lp = lmk;
        j = 1;
        while j <= k2 {
            var jk = j;
            var prod = h[j - 1];
            var i2: i64 = 1;
            while i2 <= k {
                jk += 1;
                prod = prod * h[jk - 1] * fac;
                i2 += 1;
            }
            let lk = lp + k1;
            b[(j - 1) * nest + (lmk - 1)] = (t[lk - 1] - t[lp - 1]) / prod;
            lp += 1;
            j += 1;
        }
        l += 1;
    }
}

// fpknot_of adds one knot where the sum of the squared residuals of
// the knot interval is the largest. The two counts come back as one
// pair.
struct KnotCount { n: i64, nrint: i64 }

fn fpknot_of(x: f64[:], t: f64[>..]&, fpint: f64[>..]&, nrdata: i64[>..]&, n_in: i64, nrint_in: i64, nest: i64, istart: i64) -> KnotCount {
    var n = n_in;
    var nrint = nrint_in;
    let k = (n - nrint - 1) / 2;
    var fpmax: f64 = 0.0;
    var jbegin = istart;
    var number: i64 = 0;
    var maxpt: i64 = 0;
    var maxbeg: i64 = 0;
    var j: i64 = 1;
    while j <= nrint {
        let jpoint = nrdata[j - 1];
        if !(fpmax >= fpint[j - 1] || jpoint == 0) {
            fpmax = fpint[j - 1];
            number = j;
            maxpt = jpoint;
            maxbeg = jbegin;
        }
        jbegin = jbegin + jpoint + 1;
        j += 1;
    }
    let ihalf = maxpt / 2 + 1;
    let nrx = maxbeg + ihalf;
    let next = number + 1;
    if next <= nrint {
        j = next;
        while j <= nrint {
            let jj = next + nrint - j;
            fpint[jj] = fpint[jj - 1];
            nrdata[jj] = nrdata[jj - 1];
            let jk = jj + k;
            t[jk] = t[jk - 1];
            j += 1;
        }
    }
    nrdata[number - 1] = ihalf - 1;
    nrdata[next - 1] = maxpt - ihalf;
    let am = maxpt as f64;
    var an = nrdata[number - 1] as f64;
    fpint[number - 1] = fpmax * an / am;
    an = nrdata[next - 1];
    fpint[next - 1] = fpmax * an / am;
    let jk = next + k;
    t[jk - 1] = x[nrx - 1];
    n += 1;
    nrint += 1;
    return KnotCount { n: n, nrint: nrint };
}

// fprati_of gives the next p of the iteration of the part two. It
// holds f1 above zero and f3 below zero when it comes back.
struct RatioStep { p: f64, p1: f64, f1: f64, p3: f64, f3: f64 }

fn fprati_of(p1_in: f64, f1_in: f64, p2: f64, f2: f64, p3_in: f64, f3_in: f64) -> RatioStep {
    var p1 = p1_in;
    var f1 = f1_in;
    var p3 = p3_in;
    var f3 = f3_in;
    var p: f64 = 0.0;
    if p3 > 0.0 {
        let h1 = f1 * (f2 - f3);
        let h2 = f2 * (f3 - f1);
        let h3 = f3 * (f1 - f2);
        p = -(p1 * p2 * h3 + p2 * p3 * h1 + p3 * p1 * h2) / (p1 * h1 + p2 * h2 + p3 * h3);
    } else {
        p = (p1 * (f1 - f3) * f2 - p2 * (f2 - f3) * f1) / ((f1 - f2) * f3);
    }
    if f2 < 0.0 {
        p3 = p2;
        f3 = f2;
    } else {
        p1 = p2;
        f1 = f2;
    }
    return RatioStep { p: p, p1: p1, f1: f1, p3: p3, f3: f3 };
}
