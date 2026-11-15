// The design of a filter by the least squares of the error.
//
// firls gives the taps of the filter whose weighted error over the
// bands is the smallest. The filter has a linear phase, so the count
// of taps must be odd. The call follows scipy.signal.firls: it builds
// the integrals of the cosine over the bands, assembles the matrix of
// the Toeplitz and the Hankel form, builds the right side from the
// wanted values over each band and solves the system.
namespace scientific;

import mathshim;

// The largest count of taps of the design. The matrix of the system
// holds the square of half the count of taps.
const FIRLS_TAPS_MAX: i64 = 1025;

// firls_sinc gives the sine of the angle over the angle, and one at an
// angle of zero.
fn firls_sinc(x: f64) -> f64 {
    if x == 0.0 { return 1.0; }
    return mathshim::sine(3.141592653589793 * x) / (3.141592653589793 * x);
}

// firls_val gives the wanted value of one band at one frequency. The
// band holds the slope and the offset of the wanted values. The value
// at a frequency of zero uses the rule of the limit.
fn firls_val(f: f64, m: f64, c: f64, n: i64) -> f64 {
    var v = f * (m * f + c) * firls_sinc(f * n as f64);
    if n == 0 {
        v -= m * f * f / 2.0;
    } else {
        let w = 3.141592653589793 * n as f64;
        v += m * mathshim::cosine(w * f) / (w * w);
    }
    return v;
}

// firls gives the taps of the filter of the least squares error. The
// count of taps must be odd. The bands hold two edges for each band,
// in the units of the rate. The desired values hold one value for
// each edge of the bands. The weight holds one value for each band,
// or it stays empty for a weight of one. The results are the taps.
fn firls(numtaps: i64, bands: f64[:], desired: f64[:], weight: f64[:] = [], fs: f64 = 2.0) -> Arr, Fault {
    if numtaps < 1 || numtaps % 2 == 0 {
        return empty_arr(), fault(F_VALUE, "firls", "the count of taps must be odd and above zero");
    }
    if numtaps > FIRLS_TAPS_MAX {
        return empty_arr(), fault(F_LIMIT, "firls", "the count of taps is above the limit");
    }
    if !(fs > 0.0) || fs - fs != 0.0 {
        return empty_arr(), fault(F_VALUE, "firls", "the rate of the samples must be a finite value above zero");
    }
    if bands.len < 2 || bands.len % 2 != 0 {
        return empty_arr(), fault(F_VALUE, "firls", "the bands must hold pairs of edges");
    }
    let nbands = bands.len / 2;
    if desired.len != bands.len {
        return empty_arr(), fault(F_SHAPE, "firls", "the wanted values must hold one value for each edge of the bands");
    }
    if weight.len != 0 && weight.len != nbands {
        return empty_arr(), fault(F_SHAPE, "firls", "the weight must hold one value for each band");
    }
    let nyq = fs / 2.0;
    // The edges of the bands run from zero to one of the rate.
    var lo: f64[>..] = [];
    var hi: f64[>..] = [];
    var i: i64 = 0;
    while i < nbands {
        let e1 = bands[2 * i] / nyq;
        let e2 = bands[2 * i + 1] / nyq;
        if e1 < 0.0 || e1 > 1.0 || e2 < 0.0 || e2 > 1.0 {
            return empty_arr(), fault(F_VALUE, "firls", "an edge of a band lies outside zero and one of the rate");
        }
        if !(e2 > e1) {
            return empty_arr(), fault(F_VALUE, "firls", "a band holds no width");
        }
        if i > 0 && e1 < lo[i - 1] {
            return empty_arr(), fault(F_VALUE, "firls", "the bands do not rise");
        }
        if i > 0 && hi[i - 1] > e1 {
            return empty_arr(), fault(F_VALUE, "firls", "the bands overlap");
        }
        lo.push(e1);
        hi.push(e2);
        i += 1;
    }
    i = 0;
    while i < desired.len {
        if desired[i] < 0.0 {
            return empty_arr(), fault(F_VALUE, "firls", "a wanted value holds a value below zero");
        }
        i += 1;
    }
    var wts: f64[>..] = [];
    i = 0;
    while i < nbands {
        if weight.len == 0 {
            wts.push(1.0);
        } else {
            if weight[i] < 0.0 {
                return empty_arr(), fault(F_VALUE, "firls", "a weight holds a value below zero");
            }
            wts.push(weight[i]);
        }
        i += 1;
    }
    let m = (numtaps - 1) / 2;
    // The integrals of the cosine over the bands, with the weights.
    var q: f64[>..] = [];
    var n: i64 = 0;
    while n < numtaps {
        var v: f64 = 0.0;
        i = 0;
        while i < nbands {
            v += wts[i] * (hi[i] * firls_sinc(hi[i] * n as f64) - lo[i] * firls_sinc(lo[i] * n as f64));
            i += 1;
        }
        q.push(v);
        n += 1;
    }
    // The matrix of the system holds a Toeplitz and a Hankel form.
    let q0: f64[>..] = [];
    let qm: f64[>..] = [];
    i = 0;
    while i <= m { q0.push(q[i]); i += 1; }
    i = m;
    while i < numtaps { qm.push(q[i]); i += 1; }
    let q1, fs1 = toeplitz(q0);
    if fs1.kind != F_NONE { return empty_arr(), fs1; }
    let q2, fs2 = hankel(q0, qm);
    if fs2.kind != F_NONE { return empty_arr(), fs2; }
    let side = m + 1;
    var qm2: f64[>..] = [];
    i = 0;
    while i < side * side { qm2.push(q1.data[i] + q2.data[i]); i += 1; }
    let qmat = mat2(copy(qm2), side, side);
    // The right side of the system comes from the wanted values.
    var rhs: f64[>..] = [];
    n = 0;
    while n <= m {
        var v: f64 = 0.0;
        i = 0;
        while i < nbands {
            let slope = (desired[2 * i + 1] - desired[2 * i]) / (hi[i] - lo[i]);
            let off = desired[2 * i] - lo[i] * slope;
            v += wts[i] * (firls_val(hi[i], slope, off, n) - firls_val(lo[i], slope, off, n));
            i += 1;
        }
        rhs.push(v);
        n += 1;
    }
    // The matrix of the system is a Gram matrix, so a solve of the
    // Cholesky form holds the answer with more care than the solve of
    // the LU form. A factor that holds no pivot falls back on the
    // solve of the package, and a system that holds no solution falls
    // back on the least squares of the package.
    var chol: f64[>..] = [];
    var ok_chol = true;
    i = 0;
    while i < side * side { chol.push(0.0); i += 1; }
    var r: i64 = 0;
    while r < side && ok_chol {
        var c: i64 = 0;
        var sum: f64 = 0.0;
        while c < r {
            var d = qm2[r * side + c];
            var k: i64 = 0;
            while k < c { d -= chol[r * side + k] * chol[c * side + k]; k += 1; }
            let l = d / chol[c * side + c];
            chol[r * side + c] = l;
            sum += l * l;
            c += 1;
        }
        let d = qm2[r * side + r] - sum;
        if d <= 0.0 || d - d != 0.0 {
            ok_chol = false;
            break;
        }
        chol[r * side + r] = mathshim::root(d);
        r += 1;
    }
    var av: f64[>..] = [];
    if ok_chol {
        // The forward substitution on L, then the back substitution
        // on the transpose.
        var y: f64[>..] = [];
        i = 0;
        while i < side { y.push(rhs[i]); i += 1; }
        r = 0;
        while r < side {
            var c: i64 = 0;
            while c < r {
                y[r] -= chol[r * side + c] * y[c];
                c += 1;
            }
            y[r] /= chol[r * side + r];
            r += 1;
        }
        r = side - 1;
        while r >= 0 {
            var c = r + 1;
            while c < side {
                y[r] -= chol[c * side + r] * y[c];
                c += 1;
            }
            y[r] /= chol[r * side + r];
            r -= 1;
        }
        av = copy(y);
    } else {
        let lv, fl = solve(qmat, flt_arr(copy(rhs)));
        if fl.kind != F_NONE {
            let bv, fs4 = lstsq(qmat, flt_arr(copy(rhs)));
            if fs4.kind != F_NONE { return empty_arr(), fs4; }
            av = copy_slice(bv.data);
        } else {
            av = copy_slice(lv.data);
        }
    }
    var taps: f64[>..] = [];
    i = m;
    while i >= 1 { taps.push(av[i]); i -= 1; }
    taps.push(2.0 * av[0]);
    i = 1;
    while i <= m { taps.push(av[i]); i += 1; }
    if !all_finite(taps) {
        return empty_arr(), fault(F_VALUE, "firls", "a tap of the filter is not finite");
    }
    return flt_arr(copy(taps)), no_fault();
}
