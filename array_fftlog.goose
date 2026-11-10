// The fast Hankel transform of scipy.fft: fht, ifht and fhtoffset.
//
// The transform acts on a signal that lies on a logarithmic grid. It
// runs the real FFT, multiplies the spectrum by coefficients from the
// complex log gamma function, and runs the inverse FFT. The functions
// reject a bias or an order that makes the transform singular.
namespace scientific;

import mathshim;

// cl_log gives the real and the imaginary part of the principal
// logarithm of the complex number x + iy.
fn cl_log(x: f64, y: f64) -> f64, f64 {
    return mathshim::ln(mathshim::hypot_of(x, y)), mathshim::arctangent2_of(y, x);
}

// cl_gamma_right gives the log gamma function for a real part of at
// least one half. It shifts the argument to a real part of 15 and then
// sums the series of Stirling.
fn cl_gamma_right(x0: f64, y: f64) -> f64, f64 {
    var x = x0;
    var sr: f64 = 0.0;
    var si: f64 = 0.0;
    while x < 15.0 {
        let lr, li = cl_log(x, y);
        sr += lr;
        si += li;
        x += 1.0;
    }
    let lr, li = cl_log(x, y);
    let ar = x - 0.5;
    var re = ar * lr - y * li - x + 0.9189385332046727;
    var im = ar * li + y * lr - y;
    let den = x * x + y * y;
    let wr = x / den;
    let wi = -y / den;
    let w2r = wr * wr - wi * wi;
    let w2i = 2.0 * wr * wi;
    let c: f64[7] = [1.0 / 12.0, -1.0 / 360.0, 1.0 / 1260.0, -1.0 / 1680.0, 1.0 / 1188.0, -691.0 / 360360.0, 1.0 / 156.0];
    var tr = wr;
    var ti = wi;
    var k: i64 = 0;
    while k < 7 {
        re += c[k] * tr;
        im += c[k] * ti;
        let nr = tr * w2r - ti * w2i;
        let ni = tr * w2i + ti * w2r;
        tr = nr;
        ti = ni;
        k += 1;
    }
    return re - sr, im - si;
}

// cl_log_sin gives the logarithm of sin(w) for w = a + ib. A large b
// uses the leading exponential, so the sine and cosine cannot overflow.
fn cl_log_sin(a: f64, b: f64) -> f64, f64 {
    if b > 30.0 { return b - 0.6931471805599453, 1.5707963267948966 - a; }
    if b < -30.0 { return -b - 0.6931471805599453, a - 1.5707963267948966; }
    return cl_log(mathshim::sine(a) * mathshim::cosh_of(b), mathshim::cosine(a) * mathshim::sinh_of(b));
}

// cl_gamma gives the log gamma function of x + iy. A pole gives not a
// number. A real part below one half uses the reflection formula. The
// imaginary part may differ from SciPy by a multiple of 2 pi.
fn cl_gamma(x: f64, y: f64) -> f64, f64 {
    if x >= 0.5 { return cl_gamma_right(x, y); }
    if y == 0.0 && x == mathshim::floor_of(x) { return (x - x) / (x - x), 0.0; }
    let pi = 3.141592653589793;
    let xr = x - 2.0 * mathshim::floor_of(x / 2.0);
    let sr, si = cl_log_sin(pi * xr, pi * y);
    let gr, gi = cl_gamma_right(1.0 - x, -y);
    return 1.1447298858494002 - sr - gr, -si - gi;
}

// gamma_sign gives the sign of the gamma function of a real number
// that is not a pole.
fn gamma_sign(x: f64) -> f64 {
    if x > 0.0 { return 1.0; }
    if mathshim::mod_of(mathshim::floor_of(x), 2.0) == 0.0 { return 1.0; }
    return -1.0;
}

// is_pole tells whether the gamma function has a pole at x.
fn is_pole(x: f64) -> bool {
    return x <= 0.0 && x == mathshim::floor_of(x);
}

// pochhammer gives the rising factorial gamma(a + m) / gamma(a).
fn pochhammer(a: f64, m: f64) -> f64 {
    let b = a + m;
    let pa = is_pole(a);
    let pb = is_pole(b);
    if pa && pb {
        var p: f64 = 1.0;
        if m >= 0.0 {
            var k: f64 = 0.0;
            while k < m && p - p == 0.0 { p = p * (a + k); k += 1.0; }
            return p;
        }
        var k: f64 = m;
        while k < 0.0 && p - p == 0.0 { p = p * (a + k); k += 1.0; }
        return 1.0 / p;
    }
    if pa { return 0.0; }
    if pb { return 1.0 / 0.0; }
    let rb, ib = cl_gamma(b, 0.0);
    let ra, ia = cl_gamma(a, 0.0);
    return gamma_sign(a) * gamma_sign(b) * mathshim::power_e(rb - ra);
}

// fht_coeff gives the multipliers of the spectrum, one pair of parts
// for each frequency from zero to half the length.
fn fht_coeff(n: i64, dln: f64, mu: f64, offset: f64, bias: f64, inverse: bool, ur: f64[>..]&, ui: f64[>..]&) -> Fault {
    let xp = (mu + 1.0 + bias) / 2.0;
    let xm = (mu + 1.0 - bias) / 2.0;
    let h = n / 2;
    let pi = 3.141592653589793;
    var k: i64 = 0;
    while k <= h {
        let y = pi * h / (n * dln) * k / h;
        let pr, pim = cl_gamma(xp, y);
        let mr, mim = cl_gamma(xm, y);
        let mag = pr - mr + 0.6931471805599453 * bias;
        let ph = pim + mim + y * 2.0 * (0.6931471805599453 - offset);
        let e = mathshim::power_e(mag);
        ur.push(e * mathshim::cosine(ph));
        ui.push(e * mathshim::sine(ph));
        k += 1;
    }
    if n % 2 == 0 { ui[h] = 0.0; }
    if !(ur[0] - ur[0] == 0.0) {
        ur[0] = mathshim::power(2.0, bias) * pochhammer(xm, xp - xm);
        ui[0] = 0.0;
    }
    if ur[0] - ur[0] != 0.0 && !inverse {
        return fault(F_VALUE, "fht", "the transform is singular; change the bias");
    }
    if ur[0] == 0.0 && inverse {
        return fault(F_VALUE, "ifht", "the inverse transform is singular; change the bias");
    }
    return no_fault();
}

// fht_apply runs the transform with the coefficients. The result is
// the reversed inverse transform of the product.
fn fht_apply(name: const u8[:], a: f64[:], dln: f64, mu: f64, offset: f64, bias: f64, inverse: bool) -> Arr, Fault {
    let n = a.len;
    if n < 1 {
        return empty_arr(), fault(F_SHAPE, name, "the signal holds no values");
    }
    if !(dln - dln == 0.0) || dln == 0.0 || !(mu - mu == 0.0) || !(offset - offset == 0.0) || !(bias - bias == 0.0) {
        return empty_arr(), fault(F_VALUE, name, "dln must be finite and not zero, and mu, offset and bias must be finite");
    }
    var ur: f64[>..] = [];
    var ui: f64[>..] = [];
    let cf = fht_coeff(n, dln, mu, offset, bias, inverse, ur, ui);
    if cf.kind != F_NONE { return empty_arr(), fault(cf.kind, name, cf.detail); }
    let jc = (n - 1) / 2.0;
    var x: f64[>..] = [];
    var j: i64 = 0;
    while j < n {
        var v = a[j];
        if bias != 0.0 {
            if !inverse { v = v * mathshim::power_e(-bias * (j - jc) * dln); }
            else { v = v * mathshim::power_e(bias * ((j - jc) * dln + offset)); }
        }
        x.push(v);
        j += 1;
    }
    let sr, si, ff = rfft(x);
    if ff.kind != F_NONE { return empty_arr(), ff; }
    var pr: f64[>..] = [];
    var pi: f64[>..] = [];
    var k: i64 = 0;
    while k < sr.data.len {
        let re = sr.data[k];
        let im = si.data[k];
        if !inverse {
            pr.push(re * ur[k] - im * ui[k]);
            pi.push(re * ui[k] + im * ur[k]);
        } else if ur[k] - ur[k] != 0.0 {
            pr.push(0.0);
            pi.push(0.0);
        } else {
            let den = ur[k] * ur[k] + ui[k] * ui[k];
            pr.push((re * ur[k] - im * ui[k]) / den);
            pi.push((re * ui[k] + im * ur[k]) / den);
        }
        k += 1;
    }
    let back, bf = irfft(pr, pi, n);
    if bf.kind != F_NONE { return empty_arr(), bf; }
    var out: f64[>..] = [];
    j = 0;
    while j < n {
        var v = back.data[n - 1 - j];
        if bias != 0.0 {
            if !inverse { v = v * mathshim::power_e(-bias * ((j - jc) * dln + offset)); }
            else { v = v / mathshim::power_e(-bias * (j - jc) * dln); }
        }
        out.push(v);
        j += 1;
    }
    let s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: out }, no_fault();
}

// fht gives the fast Hankel transform of a signal on a logarithmic grid
// of the step dln. The argument mu is the order, offset is the offset of
// the logarithm of the grid, and bias is the power law bias.
fn fht(a: f64[:], dln: f64, mu: f64, offset: f64, bias: f64) -> Arr, Fault {
    return fht_apply("fht", a, dln, mu, offset, bias, false);
}

// ifht gives the inverse of fht for the same arguments.
fn ifht(a: f64[:], dln: f64, mu: f64, offset: f64, bias: f64) -> Arr, Fault {
    return fht_apply("ifht", a, dln, mu, offset, bias, true);
}

// fhtoffset gives the offset nearest to the initial one that makes the
// transform of the step dln and the order mu low ringing.
fn fhtoffset(dln: f64, mu: f64, initial: f64, bias: f64) -> f64, Fault {
    if !(dln - dln == 0.0) || dln == 0.0 || !(mu - mu == 0.0) || !(initial - initial == 0.0) || !(bias - bias == 0.0) {
        return 0.0, fault(F_VALUE, "fhtoffset", "dln must be finite and not zero, and mu, initial and bias must be finite");
    }
    let xp = (mu + 1.0 + bias) / 2.0;
    let xm = (mu + 1.0 - bias) / 2.0;
    let y = 3.141592653589793 / (2.0 * dln);
    let pr, pim = cl_gamma(xp, y);
    let mr, mim = cl_gamma(xm, y);
    let arg = (0.6931471805599453 - initial) / dln + (pim + mim) / 3.141592653589793;
    return initial + (arg - mathshim::round_of(arg)) * dln, no_fault();
}
