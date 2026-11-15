// The smoothing spline filters of SciPy.
//
// cspline1d and qspline1d give the coefficients of a spline over a
// signal. cspline1d_eval and qspline1d_eval give the values of the
// spline at new places. cspline2d and qspline2d give the coefficients
// of a spline over a matrix. symiirorder1 and symiirorder2 run a
// smoothing filter forward and then backward with a mirror at both
// ends. spline_filter smooths a matrix with the cubic spline of the
// fall-off lmbda. The formulas follow SciPy 1.18.1, in
// scipy/signal/_spline_filters.py and _splinemodule.h.
namespace scientific;

import mathshim;

const SPLINE_PI: f64 = 3.141592653589793;

// spline_ic1 gives the start value of the filter of order one. The
// value is the sum that the mirror of the signal adds. The sum stops
// when the next term falls below the precision.
fn spline_ic1(z1: f64, x: f64[:], p: f64) -> f64, Fault {
    let n = x.len;
    if n < 1 { return 0.0, fault(F_VALUE, "symiirorder1", "the sum of the mirror needs one value or more"); }
    var y: f64 = x[0];
    var pw: f64 = 1.0;
    let prec = p * p;
    var k: i64 = 0;
    var run = true;
    while run {
        pw *= z1;
        y += pw * x[k];
        let err = pw * pw;
        k += 1;
        run = err > prec && k < n;
    }
    if k >= n { return 0.0, fault(F_VALUE, "symiirorder1", "the sum of the mirror did not converge"); }
    return y, no_fault();
}

// spline_hc gives the answer of a filter of order two in polar form
// to a step. It runs forward.
fn spline_hc(k: i64, cs: f64, r: f64, omega: f64) -> f64 {
    if k < 0 { return 0.0; }
    if omega == 0.0 {
        return cs * mathshim::power(r, k as f64) * (k + 1) as f64;
    }
    if omega == SPLINE_PI {
        var sgn: f64 = 1.0;
        if k % 2 == 1 { sgn = -1.0; }
        return cs * mathshim::power(r, k as f64) * (k + 1) as f64 * sgn;
    }
    return cs * mathshim::power(r, k as f64) * mathshim::sine(omega * (k + 1) as f64)
        / mathshim::sine(omega);
}

// spline_hs gives the answer of a filter of order two in polar form
// to a step. It runs backward.
fn spline_hs(k: i64, cs: f64, rsq: f64, omega: f64) -> f64 {
    var ak = k;
    if ak < 0 { ak = -ak; }
    let rsupk = mathshim::power(rsq, ak as f64 / 2.0);
    if omega == 0.0 || omega == SPLINE_PI {
        let c0 = (1.0 + rsq) / ((1.0 - rsq) * (1.0 - rsq) * (1.0 - rsq)) * cs * cs;
        var gamma = (1.0 - rsq) / (1.0 + rsq);
        if omega == SPLINE_PI && ak % 2 == 1 { gamma = -gamma; }
        return c0 * rsupk * (1.0 + gamma * ak as f64);
    }
    let c0 = cs * cs * (1.0 + rsq) / (1.0 - rsq)
        / (1.0 - 2.0 * rsq * mathshim::cosine(2.0 * omega) + rsq * rsq);
    let gamma = (1.0 - rsq) / (1.0 + rsq) / mathshim::tan_of(omega);
    return c0 * rsupk * (mathshim::cosine(omega * ak as f64)
        + gamma * mathshim::sine(omega * ak as f64));
}

// spline_ic2_fwd gives the two start values of the second order
// filter that runs forward. The sums stop when the next term falls
// below the precision.
fn spline_ic2_fwd(r: f64, omega: f64, x: f64[:], p: f64) -> f64, f64, Fault {
    let n = x.len;
    let rsq = r * r;
    let cs = 1.0 - 2.0 * r * mathshim::cosine(omega) + rsq;
    var f0: f64 = spline_hc(0, cs, r, omega) * x[0];
    let prec = p * p;
    var k: i64 = 0;
    var run = true;
    while run {
        let diff = spline_hc(k + 1, cs, r, omega);
        f0 += diff * x[k];
        let err = diff * diff;
        k += 1;
        run = err > prec && k < n;
    }
    if k >= n { return 0.0, 0.0, fault(F_VALUE, "symiirorder2", "the sum of the mirror did not converge"); }
    var f1: f64 = spline_hc(0, cs, r, omega) * x[1] + spline_hc(1, cs, r, omega) * x[0];
    k = 0;
    run = true;
    while run {
        let diff = spline_hc(k + 2, cs, r, omega);
        f1 += diff * x[k];
        let err = diff * diff;
        k += 1;
        run = err > prec && k < n;
    }
    if k >= n { return 0.0, 0.0, fault(F_VALUE, "symiirorder2", "the sum of the mirror did not converge"); }
    return f0, f1, no_fault();
}

// spline_ic2_bwd gives the two start values of the second order
// filter that runs backward. SciPy compares the square of the term
// with the unsquared precision.
fn spline_ic2_bwd(r: f64, omega: f64, x: f64[:], p: f64) -> f64, f64, Fault {
    let n = x.len;
    let rsq = r * r;
    let cs = 1.0 - 2.0 * r * mathshim::cosine(omega) + rsq;
    var b0: f64 = 0.0;
    var k: i64 = 0;
    var run = true;
    while run {
        let diff = spline_hs(k, cs, rsq, omega) + spline_hs(k + 1, cs, rsq, omega);
        b0 += diff * x[n - 1 - k];
        let err = diff * diff;
        k += 1;
        run = err > p && k < n;
    }
    if k >= n { return 0.0, 0.0, fault(F_VALUE, "symiirorder2", "the sum of the mirror did not converge"); }
    var b1: f64 = 0.0;
    k = 0;
    run = true;
    while run {
        let diff = spline_hs(k - 1, cs, rsq, omega) + spline_hs(k + 2, cs, rsq, omega);
        b1 += diff * x[n - 1 - k];
        let err = diff * diff;
        k += 1;
        run = err > p && k < n;
    }
    if k >= n { return 0.0, 0.0, fault(F_VALUE, "symiirorder2", "the sum of the mirror did not converge"); }
    return b0, b1, no_fault();
}

// spline_sym1_core runs the filter of order one over one line. The
// result holds one value for each value of the line.
fn spline_sym1_core(x: f64[:], c0: f64, z1: f64, p: f64, out: f64[>..]&) -> Fault {
    let n = x.len;
    let y0, f = spline_ic1(z1, x, p);
    if f.kind != F_NONE { return f; }
    var y: f64[>..] = [];
    y.push(y0);
    var k: i64 = 1;
    while k < n {
        y.push(x[k] + z1 * y[k - 1]);
        k += 1;
    }
    let last = -c0 / (z1 - 1.0) * y[n - 1];
    var i: i64 = 0;
    while i < n { out.push(0.0); i += 1; }
    out[n - 1] = last;
    if n >= 2 {
        out[n - 2] = c0 * y[n - 2] + z1 * last;
        i = n - 3;
        while i >= 0 {
            out[i] = c0 * y[i] + z1 * out[i + 1];
            i -= 1;
        }
    }
    return no_fault();
}

// spline_sym2_core runs the filter of order two over one line. The
// result holds one value for each value of the line.
fn spline_sym2_core(x: f64[:], r: f64, omega: f64, p: f64, out: f64[>..]&) -> Fault {
    let n = x.len;
    if n < 2 {
        return fault(F_SHAPE, "symiirorder2", "the signal needs two values or more");
    }
    let rsq = r * r;
    let a2 = 2.0 * r * mathshim::cosine(omega);
    let a3 = -rsq;
    let cs = 1.0 - 2.0 * r * mathshim::cosine(omega) + rsq;
    let f0, f1, f = spline_ic2_fwd(r, omega, x, p);
    if f.kind != F_NONE { return f; }
    var s0: f64 = a3 * f0 + a2 * f1;
    var s1: f64 = a3 * f1;
    var y: f64[>..] = [];
    y.push(f0);
    y.push(f1);
    var k: i64 = 2;
    while k < n {
        let v = cs * x[k] + s0;
        y.push(v);
        s0 = a2 * v + s1;
        s1 = a3 * v;
        k += 1;
    }
    let b0, b1, fb = spline_ic2_bwd(r, omega, x, p);
    if fb.kind != F_NONE { return fb; }
    s0 = a3 * b0 + a2 * b1;
    s1 = a3 * b1;
    var i: i64 = 0;
    while i < n { out.push(0.0); i += 1; }
    out[n - 2] = b1;
    out[n - 1] = b0;
    i = n - 3;
    while i >= 0 {
        let v = cs * y[i] + s0;
        out[i] = v;
        s0 = a2 * v + s1;
        s1 = a3 * v;
        i -= 1;
    }
    return no_fault();
}

// symiirorder1 runs the smoothing filter of order one forward and
// then backward. The filter is c0 over the product of the factors
// one minus z1 over z and one minus z1 times z.
fn symiirorder1(signal: f64[:], c0: f64, z1: f64, precision: f64 = -1.0) -> Arr, Fault {
    if !(z1 > -1.0) || !(z1 < 1.0) {
        return empty_arr(), fault(F_VALUE, "symiirorder1", "|z1| must be less than 1.0");
    }
    if signal.len < 1 {
        return empty_arr(), fault(F_SHAPE, "symiirorder1", "the signal holds no value");
    }
    var p = precision;
    if p <= 0.0 || p > 1.0 { p = 1.0e-11; }
    var out: f64[>..] = [];
    let f = spline_sym1_core(signal, c0, z1, p, out);
    if f.kind != F_NONE { return empty_arr(), f; }
    return line_arr(copy(out)), no_fault();
}

// symiirorder2 runs the smoothing filter of order two forward and
// then backward. The filter is cs over the product of the factors
// one minus a2 over z minus a3 over z squared and its mirror.
fn symiirorder2(signal: f64[:], r: f64, omega: f64, precision: f64 = -1.0) -> Arr, Fault {
    if r >= 1.0 {
        return empty_arr(), fault(F_VALUE, "symiirorder2", "r must be less than 1.0");
    }
    if signal.len < 1 {
        return empty_arr(), fault(F_SHAPE, "symiirorder2", "the signal holds no value");
    }
    var p = precision;
    if p <= 0.0 || p > 1.0 { p = 1.0e-11; }
    var out: f64[>..] = [];
    let f = spline_sym2_core(signal, r, omega, p, out);
    if f.kind != F_NONE { return empty_arr(), f; }
    return line_arr(copy(out)), no_fault();
}

// spline_root_from_lambda gives the pole and the angle of the
// smoothing filter of the fall-off lamb.
fn spline_root_from_lambda(lamb: f64) -> f64, f64 {
    let tmp = mathshim::root(3.0 + 144.0 * lamb);
    let xi = 1.0 - 96.0 * lamb + 24.0 * lamb * tmp;
    let omega = mathshim::arctangent_of(mathshim::root((144.0 * lamb - 1.0) / xi));
    let tmp2 = mathshim::root(xi);
    let r = (24.0 * lamb - 1.0 - tmp2) / (24.0 * lamb)
        * mathshim::root(48.0 * lamb + 24.0 * lamb * tmp) / tmp2;
    return r, omega;
}

// spline_coeff_smooth gives the pole and the angle of the smoothing
// coefficients of the fall-off lamb.
fn spline_coeff_smooth(lamb: f64, tag: const u8[:]) -> f64, f64, Fault {
    let tmp = mathshim::root(3.0 + 144.0 * lamb);
    let xi = 1.0 - 96.0 * lamb + 24.0 * lamb * tmp;
    if 144.0 * lamb - 1.0 < 0.0 || xi < 0.0 {
        return 0.0, 0.0, fault(F_VALUE, tag, "the fall-off stands outside the range of the smoothing spline");
    }
    let omeg = mathshim::arctangent2_of(mathshim::root(144.0 * lamb - 1.0), mathshim::root(xi));
    var rho = (24.0 * lamb - 1.0 - mathshim::root(xi)) / (24.0 * lamb);
    rho = rho * mathshim::root((48.0 * lamb + 24.0 * lamb * tmp) / xi);
    return rho, omeg, no_fault();
}

// spline_pole_coeff gives the spline coefficients of a signal from
// the pole zc. SciPy scales the cubic form by six and the quadratic
// form by eight.
fn spline_pole_coeff(signal: f64[:], zc: f64, scale: f64, tag: const u8[:]) -> Arr, Fault {
    let n = signal.len;
    if n < 1 {
        return empty_arr(), fault(F_SHAPE, tag, "the signal holds no value");
    }
    var psum: f64 = 0.0;
    var pw: f64 = 1.0;
    var k: i64 = 0;
    while k < n {
        psum += pw * signal[k];
        pw *= zc;
        k += 1;
    }
    if n == 1 {
        // SciPy forgets the scale for a signal of one value.
        var one: f64[>..] = [];
        one.push(zc / (zc - 1.0) * (signal[0] + zc * signal[0]));
        return line_arr(copy(one)), no_fault();
    }
    var y: f64[>..] = [];
    y.push(signal[0] + zc * psum);
    k = 1;
    while k < n {
        y.push(signal[k] + zc * y[k - 1]);
        k += 1;
    }
    let last = zc / (zc - 1.0) * y[n - 1];
    var out: f64[>..] = [];
    k = 0;
    while k < n { out.push(0.0); k += 1; }
    out[n - 1] = last;
    out[n - 2] = -zc * y[n - 2] + zc * last;
    var i: i64 = n - 3;
    while i >= 0 {
        out[i] = -zc * y[i] + zc * out[i + 1];
        i -= 1;
    }
    k = 0;
    while k < n {
        out[k] *= scale;
        k += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// spline_smooth_coeff gives the coefficients of the smoothing cubic
// spline of the fall-off lamb. SciPy keeps two extra start values.
fn spline_smooth_coeff(signal: f64[:], lamb: f64, tag: const u8[:]) -> Arr, Fault {
    let n = signal.len;
    if n < 2 {
        return empty_arr(), fault(F_SHAPE, tag, "the smoothing spline needs two values or more");
    }
    let rho, omega, f = spline_coeff_smooth(lamb, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let rsq = rho * rho;
    let cs = 1.0 - 2.0 * rho * mathshim::cosine(omega) + rsq;
    let a1 = -2.0 * rho * mathshim::cosine(omega);
    let a2 = rsq;
    var zi2: f64 = spline_hc(0, cs, rho, omega) * signal[0];
    var zi1: f64 = spline_hc(0, cs, rho, omega) * signal[0]
        + spline_hc(1, cs, rho, omega) * signal[1];
    var k: i64 = 0;
    while k < n {
        zi2 += spline_hc(k + 1, cs, rho, omega) * signal[k];
        zi1 += spline_hc(k + 2, cs, rho, omega) * signal[k];
        k += 1;
    }
    // The start state of the forward filter. SciPy builds it with
    // lfiltic, which negates the product of the tail of a.
    var s0: f64 = -a1 * zi1 - a2 * zi2;
    var s1: f64 = -a2 * zi1;
    var yp: f64[>..] = [];
    yp.push(zi2);
    yp.push(zi1);
    k = 2;
    while k < n {
        let v = cs * signal[k] + s0;
        yp.push(v);
        s0 = -a1 * v + s1;
        s1 = -a2 * v;
        k += 1;
    }
    var zi2b: f64 = 0.0;
    var zi1b: f64 = 0.0;
    k = 0;
    while k < n {
        zi2b += (spline_hs(k, cs, rsq, omega) + spline_hs(k + 1, cs, rsq, omega))
            * signal[n - 1 - k];
        zi1b += (spline_hs(k - 1, cs, rsq, omega) + spline_hs(k + 2, cs, rsq, omega))
            * signal[n - 1 - k];
        k += 1;
    }
    s0 = -a1 * zi1b - a2 * zi2b;
    s1 = -a2 * zi1b;
    var out: f64[>..] = [];
    k = 0;
    while k < n { out.push(0.0); k += 1; }
    k = n - 3;
    while k >= 0 {
        let v = cs * yp[k] + s0;
        out[k] = v;
        s0 = -a1 * v + s1;
        s1 = -a2 * v;
        k -= 1;
    }
    out[n - 1] = zi2b;
    out[n - 2] = zi1b;
    return line_arr(copy(out)), no_fault();
}

// cspline1d gives the cubic spline coefficients of a signal. The
// fall-off lamb of zero gives the interpolating spline.
fn cspline1d(signal: f64[:], lamb: f64 = 0.0) -> Arr, Fault {
    if lamb != 0.0 {
        return spline_smooth_coeff(signal, lamb, "cspline1d");
    }
    return spline_pole_coeff(signal, -2.0 + mathshim::root(3.0), 6.0, "cspline1d");
}

// qspline1d gives the quadratic spline coefficients of a signal. The
// fall-off lamb must hold zero.
fn qspline1d(signal: f64[:], lamb: f64 = 0.0) -> Arr, Fault {
    if lamb != 0.0 {
        return empty_arr(), fault(F_VALUE, "qspline1d", "the smoothing quadratic spline is not supported");
    }
    return spline_pole_coeff(signal, -3.0 + 2.0 * mathshim::root(2.0), 8.0, "qspline1d");
}

// spline_cubic_basis gives the value of the cubic spline basis at the
// place t.
fn spline_cubic_basis(t: f64) -> f64 {
    var a = t;
    if a < 0.0 { a = -a; }
    if a >= 2.0 { return 0.0; }
    if a < 1.0 { return (4.0 - 6.0 * a * a + 3.0 * a * a * a) / 6.0; }
    let b = 2.0 - a;
    return b * b * b / 6.0;
}

// spline_quad_basis gives the value of the quadratic spline basis at
// the place t.
fn spline_quad_basis(t: f64) -> f64 {
    var a = t;
    if a < 0.0 { a = -a; }
    if a >= 1.5 { return 0.0; }
    if a < 0.5 { return 0.75 - a * a; }
    let b = 1.5 - a;
    return b * b / 2.0;
}

// spline_eval_core gives the values of a spline at new places. A
// place outside the grid folds with the mirror until it stands
// inside. The cubic basis holds four terms, the quadratic three.
fn spline_eval_core(cj: f64[:], newx: f64[:], dx: f64, x0: f64, quad: bool,
                    tag: const u8[:]) -> Arr, Fault {
    let n = cj.len;
    if n < 1 {
        return empty_arr(), fault(F_VALUE, tag, "the coefficients hold no value");
    }
    if dx == 0.0 {
        return empty_arr(), fault(F_VALUE, tag, "the step of the old grid must not be zero");
    }
    var out: f64[>..] = [];
    let count: i64 = 3;
    var count4: i64 = count;
    if !quad { count4 = 4; }
    var half: f64 = 1.5;
    if !quad { half = 2.0; }
    var q: i64 = 0;
    while q < newx.len {
        var t = (newx[q] - x0) / dx;
        var gcount: i64 = 0;
        let hi = (n - 1) as f64;
        while (t < 0.0 || t > hi) && gcount < 4096 {
            if t < 0.0 {
                t = -t;
            } else {
                t = 2.0 * hi - t;
            }
            gcount += 1;
        }
        let jlower = (mathshim::floor_of(t - half) as i64) + 1;
        var v: f64 = 0.0;
        var i: i64 = 0;
        while i < count4 {
            let thisj = jlower + i;
            var indj = thisj;
            if indj < 0 { indj = 0; }
            if indj > n - 1 { indj = n - 1; }
            if quad {
                v += cj[indj] * spline_quad_basis(t - thisj as f64);
            } else {
                v += cj[indj] * spline_cubic_basis(t - thisj as f64);
            }
            i += 1;
        }
        out.push(v);
        q += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// cspline1d_eval gives the values of the cubic spline at new places.
// The old grid holds the places x0 plus j times dx.
fn cspline1d_eval(cj: f64[:], newx: f64[:], dx: f64 = 1.0, x0: f64 = 0.0) -> Arr, Fault {
    return spline_eval_core(cj, newx, dx, x0, false, "cspline1d_eval");
}

// qspline1d_eval gives the values of the quadratic spline at new
// places. The old grid holds the places x0 plus j times dx.
fn qspline1d_eval(cj: f64[:], newx: f64[:], dx: f64 = 1.0, x0: f64 = 0.0) -> Arr, Fault {
    return spline_eval_core(cj, newx, dx, x0, true, "qspline1d_eval");
}

// spline_sym_axis runs the symmetric filter over each line of a
// matrix. With rows true the lines are the rows, else the columns.
fn spline_sym_axis(a, cubic: bool, c0: f64, z1: f64, r: f64, omega: f64, p: f64,
                   rows: bool, tag: const u8[:]) -> Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, tag, "the matrix must hold two dimensions");
    }
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    let nrows = a.shape[0];
    let ncols = a.shape[1];
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < nrows * ncols { out.push(0.0); i += 1; }
    var lines = ncols;
    if rows { lines = nrows; }
    var li: i64 = 0;
    while li < lines {
        var line: f64[>..] = [];
        var t: i64 = 0;
        var n = ncols;
        if !rows { n = nrows; }
        while t < n {
            if rows {
                line.push(a.data[li * ncols + t]);
            } else {
                line.push(a.data[t * ncols + li]);
            }
            t += 1;
        }
        var res: f64[>..] = [];
        var f: Fault = no_fault();
        if cubic { f = spline_sym1_core(line[..], c0, z1, p, res); }
        else { f = spline_sym2_core(line[..], r, omega, p, res); }
        if f.kind != F_NONE { return empty_arr(), f; }
        t = 0;
        while t < n {
            if rows {
                out[li * ncols + t] = res[t];
            } else {
                out[t * ncols + li] = res[t];
            }
            t += 1;
        }
        li += 1;
    }
    return arr_of(copy(out), a.shape), no_fault();
}

// cspline2d gives the cubic spline coefficients of a matrix. The
// fall-off lamb at or below one over 144 gives the interpolating
// spline. Above that the call smooths.
fn cspline2d(signal, lamb: f64 = 0.0, precision: f64 = -1.0) -> Arr, Fault {
    if signal.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "cspline2d", "the matrix must hold two dimensions");
    }
    let fs = check_stack(signal.ndim, signal.shape, signal.data.len, 1, "cspline2d");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    var p = precision;
    if p < 0.0 || p >= 1.0 { p = 1.0e-6; }
    if lamb <= 1.0 / 144.0 {
        let r = -2.0 + mathshim::root(3.0);
        let mid, f1 = spline_sym_axis(signal, true, -r * 6.0, r, 0.0, 0.0, p, true, "cspline2d");
        if f1.kind != F_NONE { return empty_arr(), f1; }
        return spline_sym_axis(mid, true, -r * 6.0, r, 0.0, 0.0, p, false, "cspline2d");
    }

    let r, omega = spline_root_from_lambda(lamb);
    let mid, f1 = spline_sym_axis(signal, false, 0.0, 0.0, r, omega, p, true, "cspline2d");
    if f1.kind != F_NONE { return empty_arr(), f1; }
    return spline_sym_axis(mid, false, 0.0, 0.0, r, omega, p, false, "cspline2d");
}

// qspline2d gives the quadratic spline coefficients of a matrix. The
// fall-off lamb must be negative or zero.
fn qspline2d(signal, lamb: f64 = 0.0, precision: f64 = -1.0) -> Arr, Fault {
    if signal.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "qspline2d", "the matrix must hold two dimensions");
    }
    let fs = check_stack(signal.ndim, signal.shape, signal.data.len, 1, "qspline2d");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    if lamb > 0.0 {
        return empty_arr(), fault(F_VALUE, "qspline2d", "the fall-off must be negative or zero");
    }
    var p = precision;
    if p < 0.0 || p >= 1.0 { p = 1.0e-6; }
    let r = -3.0 + 2.0 * mathshim::root(2.0);
    let mid, f1 = spline_sym_axis(signal, true, -r * 8.0, r, 0.0, 0.0, p, true, "qspline2d");
    if f1.kind != F_NONE { return empty_arr(), f1; }
    return spline_sym_axis(mid, true, -r * 8.0, r, 0.0, 0.0, p, false, "qspline2d");
}

// spline_filter smooths a matrix with the cubic spline of the
// fall-off lmbda. SciPy builds the row filter in single precision,
// so the taps hold the single precision values of one over six and
// four over six.
fn spline_filter(a, lmbda: f64 = 5.0) -> Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "spline_filter", "the matrix must hold two dimensions");
    }
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, "spline_filter");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    let ck, f = cspline2d(a, lmbda);
    if f.kind != F_NONE { return empty_arr(), f; }
    var taps: f64[>..] = [];
    taps.push(0.1666666716337204);
    taps.push(0.6666666865348816);
    taps.push(0.1666666716337204);
    return sepfir2d(ck, taps[..], taps[..]);
}
