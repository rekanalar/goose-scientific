// The waveforms, the vector strength and the maximum length sequence.
//
// chirp, gausspulse, sawtooth and sweep_poly build a wave on a list of
// times. gammatone designs the gammatone filter. vectorstrength tells
// how the events of a signal line up with a period. max_len_seq gives
// the maximum length sequence of a count of bits.
//
// The formulas follow SciPy 1.18.1 (scipy.signal).
namespace scientific;

import mathshim;

const WAVE_TAU: f64 = 6.283185307179586;

// chirp gives the cosine of a frequency that sweeps from f0 at the time
// zero to f1 at the time t1. The method names are "linear",
// "quadratic", "logarithmic" and "hyperbolic". A vertex at zero picks
// the shape of a quadratic sweep. phi comes in degrees.
fn chirp(t: f64[:], f0: f64, t1: f64, f1: f64, method: const u8[:] = "linear",
         vertex_zero: bool = true, phi: f64 = 0.0) -> Arr, Fault {
    if t.len < 1 || !all_finite(t) {
        return empty_arr(), fault(F_VALUE, "chirp", "the times must be a finite list");
    }
    if mathshim::is_finite_of(f0) == 0.0 || mathshim::is_finite_of(f1) == 0.0 || mathshim::is_finite_of(t1) == 0.0 || mathshim::is_finite_of(phi) == 0.0 {
        return empty_arr(), fault(F_VALUE, "chirp", "the frequencies must be finite");
    }
    var kind: i64 = 0;
    var beta: f64 = 0.0;
    var degenerate = false;
    if method == "linear" || method == "lin" || method == "li" {
        kind = 0;
        if t1 == 0.0 {
            return empty_arr(), fault(F_VALUE, "chirp", "t1 holds zero");
        }
        beta = (f1 - f0) / t1;
    } else if method == "quadratic" || method == "quad" || method == "q" {
        kind = 1;
        if t1 == 0.0 {
            return empty_arr(), fault(F_VALUE, "chirp", "t1 holds zero");
        }
        beta = (f1 - f0) / (t1 * t1);
    } else if method == "logarithmic" || method == "log" {
        kind = 2;
        if f0 == 0.0 || f1 == 0.0 || (f0 < 0.0) != (f1 < 0.0) {
            return empty_arr(), fault(F_VALUE, "chirp", "a logarithmic sweep needs two frequencies of one sign");
        }
        if f0 == f1 {
            degenerate = true;
        } else {
            beta = t1 / mathshim::ln(f1 / f0);
        }
    } else if method == "hyperbolic" {
        kind = 3;
        if f0 == 0.0 || f1 == 0.0 {
            return empty_arr(), fault(F_VALUE, "chirp", "a hyperbolic sweep needs two frequencies of no zero");
        }
        if f0 == f1 {
            degenerate = true;
        }
    } else {
        return empty_arr(), fault(F_VALUE, "chirp", "the method is unknown");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < t.len {
        let x = t[i];
        var ph: f64 = 0.0;
        if kind == 0 {
            ph = WAVE_TAU * (f0 * x + 0.5 * beta * x * x);
        } else if kind == 1 {
            if vertex_zero {
                ph = WAVE_TAU * (f0 * x + beta * x * x * x / 3.0);
            } else {
                ph = WAVE_TAU * (f1 * x + beta * ((t1 - x) * (t1 - x) * (t1 - x) - t1 * t1 * t1) / 3.0);
            }
        } else if kind == 2 {
            if degenerate {
                ph = WAVE_TAU * f0 * x;
            } else {
                ph = WAVE_TAU * beta * f0 * (mathshim::power(f1 / f0, x / t1) - 1.0);
            }
        } else {
            if degenerate {
                ph = WAVE_TAU * f0 * x;
            } else {
                let sing = -f1 * t1 / (f0 - f1);
                var u = 1.0 - x / sing;
                if u < 0.0 { u = -u; }
                ph = WAVE_TAU * (-sing * f0) * mathshim::ln(u);
            }
        }
        d.push(mathshim::cosine(ph + phi * 0.017453292519943295));
        i += 1;
    }
    return line_arr(copy(d)), no_fault();
}

// gausspulse gives a Gaussian modulated sinusoid. The fractional
// bandwidth bw is measured at the level bwr in dB. The part is "i" for
// the real wave, "q" for the quadrature wave or "env" for the envelope.
fn gausspulse(t: f64[:], fc: f64 = 1000.0, bw: f64 = 0.5, bwr: f64 = -6.0,
              part: const u8[:] = "i") -> Arr, Fault {
    let a, f0 = wave_gauss_a(fc, bw, bwr);
    if f0.kind != F_NONE {
        return empty_arr(), f0;
    }
    if t.len < 1 || !all_finite(t) {
        return empty_arr(), fault(F_VALUE, "gausspulse", "the times must be a finite list");
    }
    if part != "i" && part != "q" && part != "env" {
        return empty_arr(), fault(F_VALUE, "gausspulse", "the part is unknown");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < t.len {
        let x = t[i];
        let g = mathshim::power_e(-a * x * x);
        if part == "env" {
            d.push(g);
        } else {
            let w = mathshim::cosine(WAVE_TAU * fc * x);
            if part == "i" {
                d.push(g * w);
            } else {
                d.push(g * mathshim::sine(WAVE_TAU * fc * x));
            }
        }
        i += 1;
    }
    return line_arr(copy(d)), no_fault();
}

// gausspulse_cutoff gives the time where the envelope of a pulse falls
// to the level tpr in dB.
fn gausspulse_cutoff(fc: f64, bw: f64 = 0.5, bwr: f64 = -6.0, tpr: f64 = -60.0) -> f64, Fault {
    let a, f0 = wave_gauss_a(fc, bw, bwr);
    if f0.kind != F_NONE {
        return 0.0/0.0, f0;
    }
    let tref = mathshim::power(10.0, tpr / 20.0);
    return mathshim::root(-mathshim::ln(tref) / a), no_fault();
}

// wave_gauss_a holds the factor of the Gaussian and the checks.
fn wave_gauss_a(fc: f64, bw: f64, bwr: f64) -> f64, Fault {
    if fc < 0.0 || mathshim::is_finite_of(fc) == 0.0 || mathshim::is_finite_of(bw) == 0.0 || mathshim::is_finite_of(bwr) == 0.0 {
        return 0.0/0.0, fault(F_VALUE, "gausspulse", "the centre and the bandwidth must be finite");
    }
    if bw <= 0.0 {
        return 0.0/0.0, fault(F_VALUE, "gausspulse", "the bandwidth must be above zero");
    }
    if bwr >= 0.0 {
        return 0.0/0.0, fault(F_VALUE, "gausspulse", "the reference level must be below zero");
    }
    let ref = mathshim::power(10.0, bwr / 20.0);
    let p = 3.141592653589793 * fc * bw;
    return -(p * p) / (4.0 * mathshim::ln(ref)), no_fault();
}

// sawtooth gives a wave of period 2 pi that rises from -1 to 1. The
// width is the fraction of the period that rises. A width of 0.5 gives
// a triangle. A width outside 0 and 1 gives NaN values, as SciPy does.
fn sawtooth(t: f64[:], width: f64 = 1.0) -> Arr, Fault {
    if t.len < 1 || !all_finite(t) || mathshim::is_finite_of(width) == 0.0 {
        return empty_arr(), fault(F_VALUE, "sawtooth", "the times and the width must be finite");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < t.len {
        var y: f64 = 0.0 / 0.0;
        if width >= 0.0 && width <= 1.0 {
            let tmod = mathshim::mod_of(t[i], WAVE_TAU);
            if tmod < width * WAVE_TAU {
                y = tmod / (3.141592653589793 * width) - 1.0;
            } else {
                y = (3.141592653589793 * (width + 1.0) - tmod) / (3.141592653589793 * (1.0 - width));
            }
        }
        d.push(y);
        i += 1;
    }
    return line_arr(copy(d)), no_fault();
}

// sweep_poly gives a cosine whose frequency follows a polynomial. The
// coefficients run from the highest power down. phi comes in degrees.
fn sweep_poly(t: f64[:], poly: f64[:], phi: f64 = 0.0) -> Arr, Fault {
    if t.len < 1 || !all_finite(t) {
        return empty_arr(), fault(F_VALUE, "sweep_poly", "the times must be a finite list");
    }
    if poly.len < 1 || !all_finite(poly) || mathshim::is_finite_of(phi) == 0.0 {
        return empty_arr(), fault(F_VALUE, "sweep_poly", "the coefficients must be a finite list");
    }
    // polyint: the coefficients of the integral of the polynomial.
    var k: f64[>..] = [];
    let n = poly.len;
    var i: i64 = 0;
    while i < n {
        k.push(poly[i] / (n - i) as f64);
        i += 1;
    }
    k.push(0.0);
    var d: f64[>..] = [];
    i = 0;
    while i < t.len {
        let ph = WAVE_TAU * polyval(copy(k), t[i]) + phi * 0.017453292519943295;
        d.push(mathshim::cosine(ph));
        i += 1;
    }
    return line_arr(copy(d)), no_fault();
}

// gammatone designs the filter of a gammatone at one frequency. The
// type is "fir" for a tapped delay line or "iir" for a recursive filter.
// The order goes from 1 to 24. numtaps of 0 picks the default count.
fn gammatone(freq: f64, ftype: const u8[:] = "fir", order: i64 = 4,
             numtaps: i64 = 0, fs: f64 = 2.0) -> Arr, Arr, Fault {
    if mathshim::is_finite_of(freq) == 0.0 || freq <= 0.0 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "gammatone", "the frequency must be above zero");
    }
    if mathshim::is_finite_of(fs) == 0.0 || fs <= 0.0 || freq >= fs / 2.0 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "gammatone", "the frequency must stay below the half of the rate");
    }
    if order < 1 || order > 24 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "gammatone", "the order must go from 1 to 24");
    }
    if ftype != "fir" && ftype != "iir" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "gammatone", "the type is unknown");
    }
    if ftype == "fir" {
        var nt = numtaps;
        if nt == 0 {
            nt = (fs * 0.015) as i64;
            if nt < 15 { nt = 15; }
        }
        if nt < 1 || nt > MAX_DSP_LENGTH {
            return empty_arr(), empty_arr(), fault(F_LIMIT, "gammatone", "the tap count is out of range");
        }
        let bw = 1.019 * (freq / 9.26449 + 24.7);
        var fact: f64 = 1.0;
        var j: i64 = 2;
        while j <= order - 1 { fact *= j as f64; j += 1; }
        let scale = 2.0 * mathshim::power(WAVE_TAU * bw, order as f64) / fact / fs;
        var b: f64[>..] = [];
        var i: i64 = 0;
        while i < nt {
            let tt = i as f64 / fs;
            let tt2 = mathshim::power(tt, (order - 1) as f64);
            b.push(tt2 * mathshim::power_e(-WAVE_TAU * bw * tt) * mathshim::cosine(WAVE_TAU * freq * tt) * scale);
            i += 1;
        }
        var az: f64[>..] = [];
        az.push(1.0);
        let s: i64[..8] = [];
        return line_arr(copy(b)), Arr { ndim: 1, shape: s, data: az }, no_fault();
    }
    // The recursive filter of order 4, from Hohmann 2002.
    let T = 1.0 / fs;
    let bw = WAVE_TAU * 1.019 * (freq / 9.26449 + 24.7);
    let fr = 2.0 * freq * 3.141592653589793 * T;
    let bwT = bw * T;
    let cr = mathshim::cosine(fr);
    let sn = mathshim::sine(fr);
    let g3 = mathshim::root(3.0 + mathshim::power(2.0, 1.5)) * sn;
    let g4 = mathshim::root(3.0 - mathshim::power(2.0, 1.5)) * sn;
    // The first factor turns by twice the rate of the filter, the
    // second turns by the rate and fades.
    let g1 = Cplx { re: -2.0 * T * mathshim::cosine(2.0 * fr),
                    im: -2.0 * T * mathshim::sine(2.0 * fr) };
    let g2 = Cplx { re: 2.0 * T * mathshim::power_e(-bwT) * cr,
                    im: 2.0 * T * mathshim::power_e(-bwT) * sn };
    let t1 = c_add(g1, c_scale(g2, cr - g4));
    let t2 = c_add(g1, c_scale(g2, cr + g4));
    let t3 = c_add(g1, c_scale(g2, cr - g3));
    let t4 = c_add(g1, c_scale(g2, cr + g3));
    var gg = c_mul(c_mul(t1, t2), c_mul(t3, t4));
    // The fifth factor turns by twice the rate of the filter.
    let g5 = Cplx { re: mathshim::cosine(2.0 * fr), im: mathshim::sine(2.0 * fr) };
    let e1 = mathshim::power_e(-bwT);
    let e2 = mathshim::power_e(-2.0 * bwT);
    let d_re = -2.0 * e2 - 2.0 * g5.re + 2.0 * (1.0 + g5.re) * e1;
    let d_im = -2.0 * g5.im + 2.0 * g5.im * e1;
    let den = Cplx { re: d_re, im: d_im };
    let dd = c_mul(c_mul(den, den), c_mul(den, den));
    let gv = c_abs(c_div(gg, dd));
    if !(gv > 0.0) || mathshim::is_finite_of(gv) == 0.0 {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "gammatone", "the gain is out of range");
    }
    let e3 = mathshim::power_e(-3.0 * bwT);
    let e4 = mathshim::power_e(-4.0 * bwT);
    let e5 = mathshim::power_e(-5.0 * bwT);
    let e6 = mathshim::power_e(-6.0 * bwT);
    let e7 = mathshim::power_e(-7.0 * bwT);
    let e8 = mathshim::power_e(-8.0 * bwT);
    let T4 = T * T * T * T;
    let c2 = mathshim::cosine(2.0 * fr);
    let c3 = mathshim::cosine(3.0 * fr);
    let c4 = mathshim::cosine(4.0 * fr);
    var b: f64[>..] = [];
    b.push(T4 / gv);
    b.push(-4.0 * T4 * cr * e1 / gv);
    b.push(6.0 * T4 * c2 * e2 / gv);
    b.push(-4.0 * T4 * c3 * e3 / gv);
    b.push(T4 * c4 * e4 / gv);
    var a: f64[>..] = [];
    a.push(1.0);
    a.push(-8.0 * cr * e1);
    a.push(4.0 * (4.0 + 3.0 * c2) * e2);
    a.push(-8.0 * (6.0 * cr + c3) * e3);
    a.push(2.0 * (18.0 + 16.0 * c2 + c4) * e4);
    a.push(-8.0 * (6.0 * cr + c3) * e5);
    a.push(4.0 * (4.0 + 3.0 * c2) * e6);
    a.push(-8.0 * cr * e7);
    a.push(e8);
    return line_arr(copy(b)), line_arr(copy(a)), no_fault();
}

// vectorstrength gives the strength and the phase of the events of each
// period. The phase comes in radians. A period of zero is a fault.
fn vectorstrength(events: f64[:], period: f64[:]) -> Arr, Arr, Fault {
    if events.len < 1 || !all_finite(events) || period.len < 1 || !all_finite(period) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "vectorstrength", "the lists must be finite");
    }
    var str_d: f64[>..] = [];
    var ph_d: f64[>..] = [];
    var p: i64 = 0;
    while p < period.len {
        if period[p] <= 0.0 {
            return empty_arr(), empty_arr(), fault(F_VALUE, "vectorstrength", "a period must be above zero");
        }
        var sum_re: f64 = 0.0;
        var sum_im: f64 = 0.0;
        var e: i64 = 0;
        while e < events.len {
            let a = WAVE_TAU * events[e] / period[p];
            sum_re += mathshim::cosine(a);
            sum_im += mathshim::sine(a);
            e += 1;
        }
        sum_re /= events.len as f64;
        sum_im /= events.len as f64;
        str_d.push(mathshim::root(sum_re * sum_re + sum_im * sum_im));
        ph_d.push(mathshim::arctangent2_of(sum_im, sum_re));
        p += 1;
    }
    return line_arr(copy(str_d)), line_arr(copy(ph_d)), no_fault();
}

// The taps of the maximum length sequences of the orders 2 to 32, and
// the count of taps of each order.
const MLS_TAPS: f64[>..] = [1.0, 2.0, 3.0, 3.0, 5.0, 6.0, 7.0, 6.0, 1.0, 5.0,
                            7.0, 9.0, 11.0, 10.0, 4.0, 12.0, 11.0, 8.0, 13.0,
                            12.0, 2.0, 14.0, 15.0, 13.0, 4.0, 14.0, 11.0, 18.0,
                            17.0, 14.0, 17.0, 19.0, 21.0, 18.0, 23.0, 22.0, 17.0,
                            22.0, 25.0, 24.0, 20.0, 26.0, 25.0, 22.0, 25.0, 27.0,
                            29.0, 28.0, 7.0, 28.0, 31.0, 30.0, 10.0];
const MLS_TAP_N: f64[>..] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0,
                             3.0, 3.0, 3.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0,
                             1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 1.0,
                             3.0];

// max_len_seq gives the maximum length sequence of nbits bits. The
// sequence holds the values 0 and 1. The list of taps may stay empty
// for the taps of the table. A length of -1 gives the full period. The
// second result holds the final state of the register.
fn max_len_seq(nbits: i64, length: i64 = -1, taps: f64[:] = []) -> Arr, Arr, Fault {
    if nbits < 2 || nbits > 32 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "max_len_seq", "the bit count must go from 2 to 32");
    }
    var tl: i64[>..] = [];
    if taps.len > 0 {
        var i: i64 = 0;
        while i < taps.len {
            let tv = taps[i];
            if tv != mathshim::trunc_of(tv) || tv < 0.0 || tv > nbits as f64 {
                return empty_arr(), empty_arr(), fault(F_VALUE, "max_len_seq", "a tap must be a place of the register");
            }
            let ti = tv as i64;
            var seen = false;
            var k: i64 = 0;
            while k < tl.len {
                if tl[k] == ti { seen = true; }
                k += 1;
            }
            if !seen { tl.push(ti); }
            i += 1;
        }
    } else {
        var off: i64 = 0;
        var o: i64 = 2;
        while o < nbits {
            off += MLS_TAP_N[o - 2] as i64;
            o += 1;
        }
        let cnt = MLS_TAP_N[nbits - 2] as i64;
        var i2: i64 = 0;
        while i2 < cnt {
            tl.push(MLS_TAPS[off + i2] as i64);
            i2 += 1;
        }
    }
    if tl.len == 0 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "max_len_seq", "the list of taps holds no tap");
    }
    var n = length;
    if n < 0 {
        var p: f64 = 1.0;
        var b: i64 = 0;
        while b < nbits { p *= 2.0; b += 1; }
        n = p as i64 - 1;
    }
    if n > MAX_DSP_LENGTH {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "max_len_seq", "the length is out of range");
    }
    var state: i64[>..] = [];
    var i4: i64 = 0;
    while i4 < nbits { state.push(1); i4 += 1; }
    var seq: f64[>..] = [];
    var idx2: i64 = 0;
    var i5: i64 = 0;
    while i5 < n {
        var fb = state[idx2];
        seq.push(fb as f64);
        var t3: i64 = 0;
        while t3 < tl.len {
            fb ^= state[(tl[t3] + idx2) % nbits];
            t3 += 1;
        }
        state[idx2] = fb;
        idx2 = (idx2 + 1) % nbits;
        i5 += 1;
    }
    var out: f64[>..] = [];
    var b2: i64 = 0;
    while b2 < nbits {
        out.push(state[(idx2 + b2) % nbits] as f64);
        b2 += 1;
    }
    return line_arr(copy(seq)), line_arr(copy(out)), no_fault();
}
