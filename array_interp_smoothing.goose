// The smoothing spline of SciPy: make_smoothing_spline solves the
// weighted fit with the penalty of the second derivative, in the
// basis of natural splines, and make_smoothing_spline_gcv finds the
// factor from the generalized cross validation. generate_knots gives
// the knots of the smoothing fit of Dierckx: it adds knots until the
// weighted sum of the squared residuals falls below the factor s.
// The names follow scipy.interpolate.make_smoothing_spline and
// scipy.interpolate.generate_knots.
namespace scientific;

import mathshim;

// sm_window_coef gives the coefficient i of the divided difference of
// order m minus one over the window xs. The coefficient is one over
// the product of the steps to the other points.
fn sm_window_coef(xs: f64[:], m: i64, i: i64) -> f64 {
    var pp: f64 = 1.0;
    var q: i64 = 0;
    while q < m {
        if q != i { pp = pp * (xs[i] - xs[q]); }
        q += 1;
    }
    return 1.0 / pp;
}

// sm_bands builds the design of the natural spline and the penalty in
// the dense matrices xd and ed of n by n. The design holds the values
// of the natural basis at the points. The penalty holds the divided
// differences of the points over the weights.
fn sm_bands(x: f64[:], w: f64[:], xd: f64[>..]&, ed: f64[>..]&) {
    let n = x.len;
    var i: i64 = 0;
    while i < n * n { xd.push(0.0); ed.push(0.0); i += 1; }
    // The knots of the B-spline basis of the natural spline.
    let t = sm_smooth_knots(x);
    // The values of every basis function at every point, in the dense
    // matrix of n rows and nb columns.
    let nb = n + 2;
    var dv: f64[>..] = [];
    var q: i64 = 0;
    while q < nb {
        var unit: f64[>..] = [];
        var p: i64 = 0;
        while p < nb { if p == q { unit.push(1.0); } else { unit.push(0.0); } p += 1; }
        var j: i64 = 0;
        while j < n {
            dv.push(bs_value(t, unit, 3, x[j]));
            j += 1;
        }
        q += 1;
    }
    // The value of the basis q at the point j. The values of one
    // basis take n places in the list, so the place of the basis q
    // at the point j is q times n plus j.
    i = 2;
    while i < n - 2 {
        xd[(i - 1) * n + i] = dv[(i + 1) * n + (i - 1)];
        xd[i * n + i] = dv[(i + 1) * n + i];
        xd[(i + 1) * n + i] = dv[(i + 1) * n + (i + 1)];
        i += 1;
    }
    // The corners of the left end.
    xd[0 * n + 1] = dv[0 * n + 0];
    xd[0 * n + 0] = (x[2] + x[1] - 2.0 * x[0]) * dv[0 * n + 0];
    xd[1 * n + 1] = dv[1 * n + 1] + dv[2 * n + 1];
    xd[1 * n + 0] = (x[2] - x[0]) * dv[1 * n + 1];
    xd[2 * n + 1] = dv[2 * n + 2];
    // The corners of the right end.
    xd[(n - 3) * n + (n - 2)] = dv[(nb - 3) * n + (n - 3)];
    xd[(n - 2) * n + (n - 1)] = (x[n - 1] - x[n - 3]) * dv[(nb - 2) * n + (n - 2)];
    xd[(n - 2) * n + (n - 2)] = dv[(nb - 3) * n + (n - 2)] + dv[(nb - 2) * n + (n - 2)];
    xd[(n - 1) * n + (n - 1)] = (2.0 * x[n - 1] - x[n - 2] - x[n - 3]) * dv[(nb - 1) * n + (n - 1)];
    xd[(n - 1) * n + (n - 2)] = dv[(nb - 1) * n + (n - 1)];
    // The penalty. The five bands of the column j hold the divided
    // difference of order four over the window of five points.
    var win: f64[>..] = [];
    ed[0 * n + 0] = sm_window_coef(x, 3, 0) / w[0];
    ed[1 * n + 0] = sm_window_coef(x, 3, 1) / w[1];
    ed[2 * n + 0] = sm_window_coef(x, 3, 2) / w[2];
    win = [x[0], x[1], x[2], x[3]];
    i = 0;
    while i < 4 { ed[i * n + 1] = sm_window_coef(win, 4, i) / w[i]; i += 1; }
    i = 2;
    while i < n - 2 {
        win = [x[i - 2], x[i - 1], x[i], x[i + 1], x[i + 2]];
        var r: i64 = 0;
        while r < 5 {
            ed[(i - 2 + r) * n + i] = (x[i + 2] - x[i - 2]) * sm_window_coef(win, 5, r) / w[i - 2 + r];
            r += 1;
        }
        i += 1;
    }
    win = [x[n - 4], x[n - 3], x[n - 2], x[n - 1]];
    i = 0;
    while i < 4 { ed[(n - 4 + i) * n + (n - 2)] = -sm_window_coef(win, 4, i) / w[n - 4 + i]; i += 1; }
    win = [x[n - 3], x[n - 2], x[n - 1]];
    i = 0;
    while i < 3 { ed[(n - 3 + i) * n + (n - 1)] = sm_window_coef(win, 3, i) / w[n - 3 + i]; i += 1; }
    // The penalty of SciPy carries the factor six.
    i = 0;
    while i < n * n { ed[i] = 6.0 * ed[i]; i += 1; }
}

// sm_check_input holds the rules of the data of a smoothing spline.
fn sm_check_input(x: f64[:], y: f64[:], w: f64[:], tag: const u8[:]) -> Fault {
    let n = x.len;
    if n < 5 {
        return fault(F_SHAPE, tag, "the data need five points at least");
    }
    // An empty weight list stands for no weights, as the empty w of
    // SciPy stands for the unit weights.
    if y.len != n || (w.len != 0 && w.len != n) {
        return fault(F_SHAPE, tag, "the values and the weights need the count of the points");
    }
    var i: i64 = 0;
    while i < n {
        if !(x[i] - x[i] == 0.0) || !(y[i] - y[i] == 0.0) {
            return fault(F_VALUE, tag, "a point or a value is not finite");
        }
        if w.len > 0 && (!(w[i] - w[i] == 0.0) || w[i] <= 0.0) {
            return fault(F_VALUE, tag, "a weight is not a positive number");
        }
        if i > 0 && x[i] <= x[i - 1] {
            return fault(F_VALUE, tag, "the points do not rise");
        }
        i += 1;
    }
    return no_fault();
}

// sm_smooth_knots gives the knots of the smoothing spline of the
// points.
fn sm_smooth_knots(x: f64[:]) -> f64[>..] {
    let n = x.len;
    var t: f64[>..] = [];
    var i: i64 = 0;
    while i < 3 { t.push(x[0]); i += 1; }
    i = 0;
    while i < n { t.push(x[i]); i += 1; }
    i = 0;
    while i < 3 { t.push(x[n - 1]); i += 1; }
    return t;
}

// sm_combine fills out with xd plus lam times ed, over every place
// of the matrix.
fn sm_combine(xd: f64[:], ed: f64[:], lam: f64, out: f64[>..]&) {
    var i: i64 = 0;
    while i < xd.len { out.push(xd[i] + lam * ed[i]); i += 1; }
}

// sm_coefficients solves the fit for the factor lam and moves the
// coefficients of the natural basis back to the basis of B-splines.
fn sm_coefficients(x: f64[:], y: f64[:], w: f64[:], lam: f64, xd: f64[:], ed: f64[:]) -> BSpl, Fault {
    let n = x.len;
    var m: f64[>..] = [];
    sm_combine(xd, ed, lam, m);
    var sy: i64[..8] = [];
    sy.push(n);
    var sb: i64[..8] = [];
    sb.push(n);
    let sol, f = solve(mat2(copy(m), n, n), Arr { ndim: 1, shape: sb, data: copy_slice(y) });
    if f.kind != F_NONE {
        return bspl_empty(), fault(F_VALUE, "make_smoothing_spline", "the system of the fit holds no solution");
    }
    let c = sol.data;
    let t = sm_smooth_knots(x);
    // The rule of the thesis of Zemlyanoy, equation 2.2.10: the first
    // two and the last two coefficients of the basis of B-splines mix
    // the coefficients of the ends of the natural basis.
    var co: f64[>..] = [];
    co.push(c[0] * (t[5] + t[4] - 2.0 * t[3]) + c[1]);
    co.push(c[0] * (t[5] - t[3]) + c[1]);
    var i: i64 = 1;
    while i < n - 1 { co.push(c[i]); i += 1; }
    co.push(c[n - 1] * (t[n + 2] - t[n]) + c[n - 2]);
    co.push(c[n - 1] * (2.0 * t[n + 2] - t[n + 1] - t[n]) + c[n - 2]);
    return bs_make(3, t, co), no_fault();
}

// sm_weights gives the weights of the fit. An empty weight list
// stands for the unit weight of every point.
fn sm_weights(w: f64[:], n: i64) -> f64[>..] {
    if w.len == n { return copy_slice(w); }
    var ww: f64[>..] = [];
    var i: i64 = 0;
    while i < n { ww.push(1.0); i += 1; }
    return ww;
}

// make_smoothing_spline follows scipy.interpolate.make_smoothing_spline
// with a factor lam: it holds the weighted sum of the squared
// distances and the square of the second derivative at its smallest.
// The weights hold one for an empty list.
fn make_smoothing_spline(x: f64[:], y: f64[:], w: f64[:], lam: f64) -> BSpl, Fault {
    let f = sm_check_input(x, y, w, "make_smoothing_spline");
    if f.kind != F_NONE { return bspl_empty(), f; }
    if !(lam - lam == 0.0) || lam < 0.0 {
        return bspl_empty(), fault(F_VALUE, "make_smoothing_spline", "the factor is not a number above zero or zero");
    }
    let ww = sm_weights(w, x.len);
    var xd: f64[>..] = [];
    var ed: f64[>..] = [];
    sm_bands(x, ww, xd, ed);
    return sm_coefficients(x, y, ww, lam, xd, ed);
}

// sm_xtx builds the product of the transpose of a and the weights and
// b, in a dense matrix of n by n.
fn sm_xtx(a: f64[:], b: f64[:], w: f64[:], n: i64, out: f64[>..]&) {
    var i: i64 = 0;
    while i < n * n { out.push(0.0); i += 1; }
    var r: i64 = 0;
    while r < n {
        var q: i64 = 0;
        while q < n {
            var s: f64 = 0.0;
            var j: i64 = 0;
            while j < n {
                s += a[r * n + j] * w[j] * b[j * n + q];
                j += 1;
            }
            out[r * n + q] = s;
            q += 1;
        }
        r += 1;
    }
}

// sm_gcv gives the value of the generalized cross validation at the
// factor lam, in the form of the thesis of Zemlyanoy, equation 1.3.2.
fn sm_gcv(lam: f64, xd: f64[:], ed: f64[:], xtx: f64[:], xte: f64[:], y: f64[:], n: i64) -> f64 {
    var m: f64[>..] = [];
    sm_combine(xd, ed, lam, m);
    var sb: i64[..8] = [];
    sb.push(n);
    let sol, f = solve(mat2(copy(m), n, n), Arr { ndim: 1, shape: sb, data: copy_slice(y) });
    if f.kind != F_NONE { return 1.0e308; }
    // The numerator: the norm of the penalty times the coefficients.
    var run: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        var s: f64 = 0.0;
        var j: i64 = 0;
        while j < n { s += ed[j * n + i] * sol.data[j]; j += 1; }
        run += s * s;
        i += 1;
    }
    let numer = lam * lam * run / (n as f64);
    // The denominator: one less the trace of the hat matrix, squared.
    var lhs: f64[>..] = [];
    var xt: f64[>..] = [];
    i = 0;
    while i < n * n { lhs.push(xtx[i] + lam * xte[i]); xt.push(xtx[i]); i += 1; }
    let tr, f2 = solve(mat2(copy(lhs), n, n), mat2(copy(xt), n, n));
    if f2.kind != F_NONE { return 1.0e308; }
    var trace: f64 = 0.0;
    i = 0;
    while i < n { trace += tr.data[i * n + i]; i += 1; }
    let denom = (1.0 - trace / (n as f64)) * (1.0 - trace / (n as f64));
    if denom == 0.0 { return 1.0e308; }
    return numer / denom;
}

// sm_bounded_min holds the bounded search of the minimum of a smooth
// function between a and b. It mixes steps of the golden section with
// steps of the parabola, as the bounded search of SciPy does.
fn sm_bounded_min(a: f64, b: f64, tol: f64, maxiter: i64, xd: f64[:], ed: f64[:], xtx: f64[:], xte: f64[:], y: f64[:], n: i64) -> f64 {
    let sqrt_eps = 1.4832396974191326e-08;
    let golden = 0.5 * (3.0 - 2.2360679774997896);
    var fulc = a + golden * (b - a);
    var nfc = fulc;
    var xf = fulc;
    var rat: f64 = 0.0;
    var e: f64 = 0.0;
    var fx = sm_gcv(xf, xd, ed, xtx, xte, y, n);
    var ffulc = fx;
    var fnfc = fx;
    var fu: f64 = 0.0;
    var lo = a;
    var hi = b;
    var num: i64 = 1;
    var xm = 0.5 * (lo + hi);
    var tol1 = sqrt_eps * mathshim::abs_of(xf) + tol / 3.0;
    var tol2 = 2.0 * tol1;
    while mathshim::abs_of(xf - xm) > (tol2 - 0.5 * (hi - lo)) {
        var golden_step = true;
        if mathshim::abs_of(e) > tol1 {
            golden_step = false;
            let r = (xf - nfc) * (fx - ffulc);
            var q = (xf - fulc) * (fx - fnfc);
            var p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 { p = -p; }
            q = mathshim::abs_of(q);
            let rsave = e;
            e = rat;
            if mathshim::abs_of(p) < mathshim::abs_of(0.5 * q * rsave) && p > q * (lo - xf) && p < q * (hi - xf) {
                rat = p / q;
                var x = xf + rat;
                if (x - lo) < tol2 || (hi - x) < tol2 {
                    var si: f64 = 0.0;
                    if xm - xf > 0.0 { si = 1.0; } else if xm - xf < 0.0 { si = -1.0; } else { si = 1.0; }
                    rat = tol1 * si;
                }
            } else {
                golden_step = true;
            }
        }
        if golden_step {
            if xf >= xm { e = lo - xf; } else { e = hi - xf; }
            rat = golden * e;
        }
        var si: f64 = 0.0;
        if rat > 0.0 { si = 1.0; } else if rat < 0.0 { si = -1.0; } else { si = 1.0; }
        var step = mathshim::abs_of(rat);
        if step < tol1 { step = tol1; }
        let x = xf + si * step;
        fu = sm_gcv(x, xd, ed, xtx, xte, y, n);
        num += 1;
        if fu <= fx {
            if x >= xf { lo = xf; } else { hi = xf; }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf { lo = x; } else { hi = x; }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (lo + hi);
        tol1 = sqrt_eps * mathshim::abs_of(xf) + tol / 3.0;
        tol2 = 2.0 * tol1;
        if num >= maxiter { break; }
    }
    return xf;
}

// make_smoothing_spline_gcv follows make_smoothing_spline with the
// factor found from the generalized cross validation, as the default
// of SciPy does.
fn make_smoothing_spline_gcv(x: f64[:], y: f64[:], w: f64[:]) -> BSpl, Fault {
    let f = sm_check_input(x, y, w, "make_smoothing_spline_gcv");
    if f.kind != F_NONE { return bspl_empty(), f; }
    let n = x.len;
    let ww = sm_weights(w, n);
    var xd: f64[>..] = [];
    var ed: f64[>..] = [];
    sm_bands(x, ww, xd, ed);
    var xtx: f64[>..] = [];
    sm_xtx(xd, xd, ww, n, xtx);
    var xte: f64[>..] = [];
    sm_xtx(xd, ed, ww, n, xte);
    // The factor comes from the bounded search over the interval from
    // zero to the count of the points, with the step 1.0e-5, as the
    // bounded search of SciPy holds.
    let lam = sm_bounded_min(0.0, n as f64, 1.0e-5, 500, xd, ed, xtx, xte, y, n);
    return sm_coefficients(x, y, ww, lam, xd, ed);
}

// gk_lsq_residual gives the residuals and the weighted sum of the
// squares of the least squares spline of the knots t.
fn gk_lsq_residual(x: f64[:], y: f64[:], w: f64[:], t: f64[:], k: i64, res: f64[>..]&) -> f64, Fault {
    let b, f = make_lsq_spline(x, y, t, k, w);
    if f.kind != F_NONE { return 0.0, f; }
    let v, f2 = bspline_eval(b, x);
    if f2.kind != F_NONE { return 0.0, f2; }
    var fp: f64 = 0.0;
    var i: i64 = 0;
    while i < x.len {
        // The residuals carry the square of the weighted distance, as
        // the rule of the knot search holds.
        let d = w[i] * (v.data[i] - y[i]);
        res.push(d * d);
        fp += d * d;
        i += 1;
    }
    return fp, no_fault();
}

// gk_add_knot picks the new knot of the rule of Dierckx. The points
// fall into the runs between the knots. The sum of the squared
// residuals of a run counts the boundary point with the half into each
// side. The run with the largest sum takes the new knot at the middle
// point of the run, and a run needs two points at least.
fn gk_add_knot(x: f64[:], t: f64[:], k: i64, res: f64[:]) -> f64, Fault {
    let n = t.len;
    let nc = n - k - 1;
    var interval: i64 = k + 1;
    var ix: i64[>..] = [];
    var fparts: f64[>..] = [];
    var fpart: f64 = 0.0;
    ix.push(0);
    var i: i64 = 0;
    while i < x.len {
        fpart += res[i];
        if x[i] >= t[interval] && interval < nc {
            // The point on the boundary splits with the half between
            // the two runs.
            let carry = res[i] / 2.0;
            fpart -= carry;
            fparts.push(fpart);
            fpart = carry;
            interval += 1;
            ix.push(i);
        }
        i += 1;
    }
    ix.push(x.len - 1);
    fparts.push(fpart);
    var best: i64 = -1;
    var score: f64 = -1.0e100;
    var r: i64 = 0;
    while r < fparts.len {
        if ix[r + 1] - ix[r] > 1 && fparts[r] > score {
            best = r;
            score = fparts[r];
        }
        r += 1;
    }
    if best < 0 {
        return 0.0, fault(F_VALUE, "generate_knots", "no span holds a point for a new knot");
    }
    return x[(ix[best] + ix[best + 1] + 1) / 2], no_fault();
}

// generate_knots follows scipy.interpolate.generate_knots: it adds
// knots until the weighted sum of the squared residuals of the least
// squares spline falls below the factor s, and gives the knots that
// the fit accepts at the end.
fn generate_knots(x: f64[:], y: f64[:], w: f64[:], k: i64, s: f64) -> f64[>..], Fault {
    if k < 1 || k > 5 {
        return [], fault(F_RANGE, "generate_knots", "the degree is outside one to five");
    }
    if !(s - s == 0.0) || s < 0.0 {
        return [], fault(F_VALUE, "generate_knots", "the factor is not a number above zero or zero");
    }
    if x.len != y.len || x.len < k + 1 {
        return [], fault(F_SHAPE, "generate_knots", "the data need one equal count");
    }
    if w.len != 0 && w.len != x.len {
        return [], fault(F_SHAPE, "generate_knots", "the weights need the count of the points");
    }
    // The weights hold one for an empty list.
    var ww: f64[>..] = [];
    var i0: i64 = 0;
    while i0 < x.len { ww.push(1.0); i0 += 1; }
    if w.len == x.len { ww = copy_slice(w); }
    var i: i64 = 0;
    while i < x.len {
        if !(x[i] - x[i] == 0.0) || !(y[i] - y[i] == 0.0) || !(ww[i] - ww[i] == 0.0) || ww[i] <= 0.0 {
            return [], fault(F_VALUE, "generate_knots", "a point, a value or a weight is not a usable number");
        }
        if i > 0 && x[i] < x[i - 1] {
            return [], fault(F_VALUE, "generate_knots", "the points fall");
        }
        i += 1;
    }
    let acc = s * 0.001;
    let m = x.len;
    var nest = m + k + 1;
    if nest < 2 * k + 3 { nest = 2 * k + 3; }
    let nmin = 2 * (k + 1);
    let nmax = m + k + 1;
    var t: f64[>..] = [];
    i = 0;
    while i < k + 1 { t.push(x[0]); i += 1; }
    i = 0;
    while i < k + 1 { t.push(x[m - 1]); i += 1; }
    var fp: f64 = 0.0;
    var nplus: i64 = 1;
    var turn: i64 = 0;
    while turn < m {
        let fpold = fp;
        var res: f64[>..] = [];
        let nfp, f = gk_lsq_residual(x, y, ww, t, k, res);
        if f.kind != F_NONE { return [], f; }
        fp = nfp;
        let fpms = fp - s;
        if mathshim::abs_of(fpms) < acc || fpms < 0.0 {
            return t, no_fault();
        }
        if t.len == nmin {
            nplus = 1;
        } else {
            let delta = fpold - fp;
            var npl1: i64 = nplus * 2;
            if delta > acc { npl1 = (nplus * fpms / delta) as i64; }
            var low: i64 = nplus / 2;
            if npl1 > low { low = npl1; }
            if low < 1 { low = 1; }
            var up: i64 = nplus * 2;
            if low < up { up = low; }
            nplus = up;
        }
        var j: i64 = 0;
        while j < nplus {
            let nk, fa = gk_add_knot(x, t, k, res);
            if fa.kind != F_NONE { return [], fa; }
            var at: i64 = 0;
            while at < t.len && t[at] < nk { at += 1; }
            var t2: f64[>..] = [];
            var q: i64 = 0;
            while q < t.len {
                if q == at { t2.push(nk); }
                t2.push(t[q]);
                q += 1;
            }
            if at >= t.len { t2.push(nk); }
            t = copy(t2);
            if t.len >= nmax {
                return bs_knots(x, k), no_fault();
            }
            if t.len >= nest {
                return t, no_fault();
            }
            if j < nplus - 1 {
                var res2: f64[>..] = [];
                let fp2, f2 = gk_lsq_residual(x, y, ww, t, k, res2);
                if f2.kind != F_NONE { return [], f2; }
                res = copy(res2);
            }
            j += 1;
        }
        turn += 1;
    }
    return t, no_fault();
}
