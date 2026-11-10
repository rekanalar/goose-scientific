// Digital filters: the Butterworth design, the linear filter, the
// zero phase filter, the cascade of second order sections and the
// analytic signal.
//
// The package holds real numbers only. The design builds the
// polynomials of the analog prototype, turns them into the digital
// filter with the bilinear rule and gives the coefficients with real
// arithmetic. The coefficients match those of SciPy.
namespace scientific;

import mathshim;

const PI_VALUE: f64 = 3.141592653589793;

// flt_binom gives the binomial coefficient of n over k.
fn flt_binom(n: i64, k: i64) -> f64 {
    var r: f64 = 1.0;
    var i: i64 = 1;
    while i <= k {
        r = r * (n - k + i) / i;
        i += 1;
    }
    return r;
}

// flt_pmul multiplies two polynomials. The coefficients run from the
// highest power. The caller passes an empty result.
fn flt_pmul(a: f64[:], b: f64[:], out: f64[>..]&) {
    var i: i64 = 0;
    while i < a.len + b.len - 1 { out.push(0.0); i += 1; }
    i = 0;
    while i < a.len {
        var j: i64 = 0;
        while j < b.len {
            out[i + j] = out[i + j] + a[i] * b[j];
            j += 1;
        }
        i += 1;
    }
}

// flt_peval gives the value of a polynomial at a point.
fn flt_peval(c: f64[:], x: f64) -> f64 {
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < c.len { t = t * x + c[i]; i += 1; }
    return t;
}

// flt_shift_pow gives the polynomial (z + c) to the power n.
fn flt_shift_pow(n: i64, c: f64, out: f64[>..]&) {
    var p: f64 = 1.0;
    var j: i64 = 0;
    while j <= n { out.push(0.0); j += 1; }
    j = 0;
    while j <= n {
        out[j] = flt_binom(n, j) * p;
        p = p * c;
        j += 1;
    }
}

// flt_quad_pow gives the polynomial (s^2 + w2) to the power m.
fn flt_quad_pow(m: i64, w2: f64, out: f64[>..]&) {
    var j: i64 = 0;
    while j <= 2 * m { out.push(0.0); j += 1; }
    var p: f64 = 1.0;
    j = 0;
    while j <= m {
        out[2 * j] = flt_binom(m, j) * p;
        p = p * w2;
        j += 1;
    }
}

// flt_proto gives the polynomial of the analog Butterworth prototype
// of the unit cut.
fn flt_proto(order: i64, out: f64[>..]&) {
    var c: f64[>..] = [1.0];
    if order % 2 == 1 {
        var t: f64[>..] = [];
        flt_pmul(c, [1.0, 1.0], t);
        c = copy(t);
    }
    var m = order - 1;
    while m >= 1 {
        if (order - 1 - m) % 2 == 0 {
            var t: f64[>..] = [];
            let k = 2.0 * mathshim::cosine(PI_VALUE * m / (2.0 * order));
            flt_pmul(c, [1.0, k, 1.0], t);
            c = copy(t);
        }
        m -= 1;
    }
    var i: i64 = 0;
    while i < c.len { out.push(c[i]); i += 1; }
}

// flt_digital maps the analog denominator Q(s) to the digital one with
// the bilinear rule s = 4 (z - 1) / (z + 1).
fn flt_digital(q: f64[:], out: f64[>..]&) {
    let m = q.len - 1;
    var i: i64 = 0;
    while i <= m { out.push(0.0); i += 1; }
    // i counts the power of (z + 1), m - i the power of 4 (z - 1).
    var pw: f64[>..] = [];
    i = 0;
    while i <= m { pw.push(1.0); i += 1; }
    i = 1;
    while i <= m { pw[i] = pw[i - 1] * 4.0; i += 1; }
    i = 0;
    while i <= m {
        let p = m - i;
        var u: f64[>..] = [];
        var v: f64[>..] = [];
        var t: f64[>..] = [];
        flt_shift_pow(p, -1.0, u);
        flt_shift_pow(i, 1.0, v);
        flt_pmul(u, v, t);
        var j: i64 = 0;
        while j <= m {
            out[j] = out[j] + q[i] * pw[p] * t[j];
            j += 1;
        }
        i += 1;
    }
}

// butter designs a digital Butterworth filter. The cut is a fraction
// of the Nyquist rate, in the open range of zero to one. A band type
// takes two cuts. The type is "lowpass", "highpass", "bandpass" or
// "bandstop". The result holds the numerator and the denominator.
fn butter(order: i64, wn: f64[:], btype: const u8[:]) -> Arr, Arr, Fault {
    if order < 1 || order > 32 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "butter", "the order holds a value outside one to 32");
    }
    var band = false;
    if btype == "bandpass" || btype == "bandstop" { band = true; }
    else if btype != "lowpass" && btype != "highpass" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "butter", "the type of the filter holds an unknown name");
    }
    var need: i64 = 1;
    if band { need = 2; }
    if wn.len != need {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "butter", "the count of cuts does not fit the type");
    }
    var i: i64 = 0;
    var w: f64[>..] = [];
    while i < need {
        if !(wn[i] > 0.0 && wn[i] < 1.0) {
            return empty_arr(), empty_arr(), fault(F_RANGE, "butter", "a cut holds a value outside zero to one");
        }
        w.push(4.0 * mathshim::tan_of(PI_VALUE * wn[i] / 2.0));
        i += 1;
    }
    if band && !(wn[0] < wn[1]) {
        return empty_arr(), empty_arr(), fault(F_RANGE, "butter", "the low cut holds a value that is not below the high cut");
    }
    var p: f64[>..] = [];
    flt_proto(order, p);
    var q: f64[>..] = [];
    var num: f64[>..] = [];
    let n = order;
    if btype == "lowpass" {
        var wp: f64 = 1.0;
        i = 0;
        while i <= n { q.push(p[i] * wp); wp = wp * w[0]; i += 1; }
        flt_shift_pow(n, 1.0, num);
        i = 0;
        var s: f64 = 1.0;
        while i < n { s = s * w[0]; i += 1; }
        i = 0;
        while i <= n { num[i] = num[i] * s; i += 1; }
    } else if btype == "highpass" {
        var wp: f64 = 1.0;
        i = 0;
        while i <= n { q.push(p[n - i] * wp); wp = wp * w[0]; i += 1; }
        flt_shift_pow(n, -1.0, num);
        var s: f64 = 1.0;
        i = 0;
        while i < n { s = s * 4.0; i += 1; }
        i = 0;
        while i <= n { num[i] = num[i] * s; i += 1; }
    } else {
        let bw = w[1] - w[0];
        let w2 = w[0] * w[1];
        i = 0;
        while i <= 2 * n { q.push(0.0); i += 1; }
        var j: i64 = 0;
        while j <= n {
            // The prototype term with the substituted power.
            var sq: f64[>..] = [];
            var bwp: f64 = 1.0;
            var e: i64 = 0;
            var lead = j;
            if btype == "bandstop" { lead = n - j; }
            if btype == "bandpass" { flt_quad_pow(n - j, w2, sq); e = j; }
            else { flt_quad_pow(j, w2, sq); e = n - j; }
            i = 0;
            while i < e { bwp = bwp * bw; i += 1; }
            i = 0;
            while i < e { sq.push(0.0); i += 1; }
            i = 0;
            while i < sq.len {
                q[lead + i] = q[lead + i] + p[j] * bwp * sq[i];
                i += 1;
            }
            j += 1;
        }
        i = 0;
        while i <= 2 * n { num.push(0.0); i += 1; }
        if btype == "bandpass" {
            var s: f64 = 1.0;
            i = 0;
            while i < n { s = s * bw * 4.0; i += 1; }
            j = 0;
            while j <= n {
                var sg: f64 = 1.0;
                if j % 2 == 1 { sg = -1.0; }
                num[2 * j] = s * flt_binom(n, j) * sg;
                j += 1;
            }
        } else {
            let zc = (16.0 - w2) / (16.0 + w2);
            var acc: f64[>..] = [1.0];
            i = 0;
            while i < n {
                var t: f64[>..] = [];
                flt_pmul(acc, [1.0, -2.0 * zc, 1.0], t);
                acc = copy(t);
                i += 1;
            }
            var s: f64 = 1.0;
            i = 0;
            while i < n { s = s * (16.0 + w2); i += 1; }
            i = 0;
            while i < acc.len { num[i] = acc[i] * s; i += 1; }
        }
    }
    var d: f64[>..] = [];
    flt_digital(q, d);
    let d0 = d[0];
    var bo: f64[>..] = [];
    var ao: f64[>..] = [];
    i = 0;
    while i < num.len { bo.push(num[i] / d0); i += 1; }
    i = 0;
    while i < d.len { ao.push(d[i] / d0); i += 1; }
    let s1: i64[..8] = [];
    s1.push(bo.len);
    let s2: i64[..8] = [];
    s2.push(ao.len);
    return Arr { ndim: 1, shape: s1, data: bo }, Arr { ndim: 1, shape: s2, data: ao }, no_fault();
}

// flt_prepare pads the numerator and the denominator to one length and
// divides both by the first denominator coefficient.
fn flt_prepare(b: f64[:], a: f64[:], pb: f64[>..]&, pa: f64[>..]&, tag: const u8[:]) -> Fault {
    if b.len < 1 || a.len < 1 {
        return fault(F_VALUE, tag, "a coefficient list holds no element");
    }
    if a[0] == 0.0 {
        return fault(F_VALUE, tag, "the first denominator coefficient holds zero");
    }
    var n = b.len;
    if a.len > n { n = a.len; }
    var i: i64 = 0;
    while i < n {
        if i < b.len { pb.push(b[i] / a[0]); } else { pb.push(0.0); }
        if i < a.len { pa.push(a[i] / a[0]); } else { pa.push(0.0); }
        i += 1;
    }
    return no_fault();
}

// flt_run runs the filter in the transposed direct form. The state
// holds one value less than the coefficients and carries over.
fn flt_run(pb: f64[:], pa: f64[:], x: f64[:], z: f64[>..]&, y: f64[>..]&) {
    let n = pb.len;
    var t: i64 = 0;
    while t < x.len {
        let xi = x[t];
        var yi = xi * pb[0];
        if n > 1 { yi = yi + z[0]; }
        var i: i64 = 0;
        while i + 1 < n {
            var nz: f64 = 0.0;
            if i + 2 < n { nz = z[i + 1]; }
            z[i] = nz + pb[i + 1] * xi - pa[i + 1] * yi;
            i += 1;
        }
        y.push(yi);
        t += 1;
    }
}

fn flt_arr(v: f64[>..]) -> Arr {
    let s: i64[..8] = [];
    s.push(v.len);
    return Arr { ndim: 1, shape: s, data: v };
}

// lfilter filters a signal with the filter of the coefficients b and
// a. The state starts at zero.
fn lfilter(b: f64[:], a: f64[:], x: f64[:]) -> Arr, Fault {
    var pb: f64[>..] = [];
    var pa: f64[>..] = [];
    let f = flt_prepare(b, a, pb, pa, "lfilter");
    if f.kind != F_NONE { return empty_arr(), f; }
    var z: f64[>..] = [];
    var i: i64 = 0;
    while i < pb.len { z.push(0.0); i += 1; }
    var y: f64[>..] = [];
    flt_run(pb, pa, x, z, y);
    return flt_arr(copy(y)), no_fault();
}

// flt_zi gives the state of the filter at rest for a constant input
// of one.
fn flt_zi(pb: f64[:], pa: f64[:], zi: f64[>..]&) -> Fault {
    let n = pb.len;
    var sb: f64 = 0.0;
    var sa: f64 = 0.0;
    var i: i64 = 0;
    while i < n { sb = sb + pb[i]; sa = sa + pa[i]; i += 1; }
    if sa == 0.0 {
        return fault(F_VALUE, "lfilter_zi", "the denominator sums to zero, so the filter holds no steady state");
    }
    let gain = sb / sa;
    i = 0;
    while i < n { zi.push(0.0); i += 1; }
    var acc: f64 = 0.0;
    i = n - 1;
    while i >= 1 {
        acc = acc + pb[i] - gain * pa[i];
        zi[i - 1] = acc;
        i -= 1;
    }
    return no_fault();
}

// lfilter_zi gives the initial state for a step input of one.
fn lfilter_zi(b: f64[:], a: f64[:]) -> Arr, Fault {
    var pb: f64[>..] = [];
    var pa: f64[>..] = [];
    let f = flt_prepare(b, a, pb, pa, "lfilter_zi");
    if f.kind != F_NONE { return empty_arr(), f; }
    var zi: f64[>..] = [];
    let g = flt_zi(pb, pa, zi);
    if g.kind != F_NONE { return empty_arr(), g; }
    if zi.len < 2 { return empty_arr(), fault(F_VALUE, "lfilter_zi", "the filter holds no state"); }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i + 1 < zi.len { out.push(zi[i]); i += 1; }
    return flt_arr(copy(out)), no_fault();
}

// filtfilt filters a signal forward and backward, so the phase
// shift is zero. The signal is extended at both ends with an odd
// reflection of three times the length of the coefficients.
fn filtfilt(b: f64[:], a: f64[:], x: f64[:]) -> Arr, Fault {
    var pb: f64[>..] = [];
    var pa: f64[>..] = [];
    let f = flt_prepare(b, a, pb, pa, "filtfilt");
    if f.kind != F_NONE { return empty_arr(), f; }
    let edge = 3 * pb.len;
    if x.len <= edge {
        return empty_arr(), fault(F_SHAPE, "filtfilt", "the signal is not longer than three times the coefficients");
    }
    var zi: f64[>..] = [];
    let g = flt_zi(pb, pa, zi);
    if g.kind != F_NONE { return empty_arr(), g; }
    var ext: f64[>..] = [];
    var i: i64 = 0;
    while i < edge { ext.push(2.0 * x[0] - x[edge - i]); i += 1; }
    i = 0;
    while i < x.len { ext.push(x[i]); i += 1; }
    i = 0;
    while i < edge { ext.push(2.0 * x[x.len - 1] - x[x.len - 2 - i]); i += 1; }
    var z: f64[>..] = [];
    i = 0;
    while i < zi.len { z.push(zi[i] * ext[0]); i += 1; }
    var y: f64[>..] = [];
    flt_run(pb, pa, ext, z, y);
    var r: f64[>..] = [];
    i = 0;
    while i < y.len { r.push(y[y.len - 1 - i]); i += 1; }
    z = [];
    i = 0;
    while i < zi.len { z.push(zi[i] * r[0]); i += 1; }
    var y2: f64[>..] = [];
    flt_run(pb, pa, r, z, y2);
    var out: f64[>..] = [];
    i = 0;
    while i < x.len { out.push(y2[y2.len - 1 - edge - i]); i += 1; }
    return flt_arr(copy(out)), no_fault();
}

// sosfilt filters a signal with a cascade of second order sections.
// Each row of the matrix holds b0, b1, b2, a0, a1 and a2. The first
// denominator coefficient of each row must hold one.
fn sosfilt(sos, x: f64[:]) -> Arr, Fault {
    if sos.ndim != 2 || sos.shape[1] != 6 {
        return empty_arr(), fault(F_SHAPE, "sosfilt", "the sections do not hold a matrix of six columns");
    }
    var cur: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len { cur.push(x[i]); i += 1; }
    var s: i64 = 0;
    while s < sos.shape[0] {
        let r = s * 6;
        if sos.data[r + 3] != 1.0 {
            return empty_arr(), fault(F_VALUE, "sosfilt", "a section holds a first denominator coefficient that is not one");
        }
        var pb: f64[>..] = [sos.data[r], sos.data[r + 1], sos.data[r + 2]];
        var pa: f64[>..] = [1.0, sos.data[r + 4], sos.data[r + 5]];
        var z: f64[>..] = [0.0, 0.0, 0.0];
        var y: f64[>..] = [];
        flt_run(pb, pa, cur, z, y);
        cur = copy(y);
        s += 1;
    }
    return flt_arr(copy(cur)), no_fault();
}

// hilbert gives the analytic signal of a real signal. The result holds
// the real parts, which equal the signal, and the imaginary parts.
fn hilbert(x: f64[:]) -> Arr, Arr, Fault {
    let n = x.len;
    let fr, fi, f = fft(x);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    var re: f64[>..] = [];
    var im: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var h: f64 = 0.0;
        if i == 0 { h = 1.0; }
        else if n % 2 == 0 && i == n / 2 { h = 1.0; }
        else if i < (n + 1) / 2 { h = 2.0; }
        re.push(fr.data[i] * h);
        im.push(fi.data[i] * h);
        i += 1;
    }
    return ifft(re, im);
}
