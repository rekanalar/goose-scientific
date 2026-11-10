// The windows of scipy.signal.windows that need a list of parameters or
// a solve: general_cosine, general_hamming, general_gaussian, taylor,
// chebwin and kaiser_bessel_derived. A window with sym false is the
// periodic window of the Fourier transform.
namespace scientific;

import mathshim;

// ww_arr wraps the values of a window in an array.
fn ww_arr(d: f64[:]) -> Arr {
    return line_arr(copy_slice(d));
}

// ww_acosh gives the inverse hyperbolic cosine of a number of one or
// more.
fn ww_acosh(x: f64) -> f64 {
    return mathshim::ln(x + mathshim::root(x * x - 1.0));
}

// ww_check tests the length of a window.
fn ww_check(m: i64, tag: const u8[:]) -> Fault {
    if m < 1 || m > MAX_ELEMENTS - 1 {
        return fault(F_RANGE, tag, "the length stands outside its range");
    }
    return no_fault();
}

// general_cosine gives a window that sums cosine terms with the
// weights a. The terms run from the angle -pi to pi.
fn general_cosine(m: i64, a: f64[:], sym: bool = true) -> Arr, Fault {
    let f = ww_check(m, "general_cosine");
    if f.kind != F_NONE { return empty_arr(), f; }
    if a.len < 1 || a.len > 64 {
        return empty_arr(), fault(F_SHAPE, "general_cosine", "the weights need one to 64 values");
    }
    var i: i64 = 0;
    while i < a.len {
        if !(a[i] - a[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "general_cosine", "a weight is not finite");
        }
        i += 1;
    }
    var out: f64[>..] = [];
    if m == 1 { out.push(1.0); return ww_arr(out), no_fault(); }
    var len = m;
    if !sym { len = m + 1; }
    i = 0;
    while i < m {
        let fac = -3.141592653589793 + 6.283185307179586 * i / (len - 1);
        var t: f64 = 0.0;
        var k: i64 = 0;
        while k < a.len { t += a[k] * mathshim::cosine(k * fac); k += 1; }
        out.push(t);
        i += 1;
    }
    return ww_arr(out), no_fault();
}

// general_hamming gives the window alpha minus (1 - alpha) cos.
fn general_hamming(m: i64, alpha: f64, sym: bool = true) -> Arr, Fault {
    if !(alpha - alpha == 0.0) {
        return empty_arr(), fault(F_VALUE, "general_hamming", "alpha is not finite");
    }
    return general_cosine(m, [alpha, 1.0 - alpha], sym);
}

// general_gaussian gives the window exp(-0.5 |n / sig|^(2 p)).
fn general_gaussian(m: i64, p: f64, sig: f64, sym: bool = true) -> Arr, Fault {
    let f = ww_check(m, "general_gaussian");
    if f.kind != F_NONE { return empty_arr(), f; }
    if !(sig > 0.0) || !(p > 0.0) || !(sig - sig == 0.0) || !(p - p == 0.0) {
        return empty_arr(), fault(F_VALUE, "general_gaussian", "p and sig need finite values above zero");
    }
    var out: f64[>..] = [];
    if m == 1 { out.push(1.0); return ww_arr(out), no_fault(); }
    var len = m;
    if !sym { len = m + 1; }
    var i: i64 = 0;
    while i < m {
        let n = i - (len - 1.0) / 2.0;
        out.push(mathshim::power_e(-0.5 * mathshim::power(mathshim::abs_of(n / sig), 2.0 * p)));
        i += 1;
    }
    return ww_arr(out), no_fault();
}

// taylor gives the window of Taylor with nbar nearly constant side
// lobes at the level sll below the main lobe. The flag norm scales the
// peak to one.
fn taylor(m: i64, nbar: i64 = 4, sll: f64 = 30.0, norm: bool = true, sym: bool = true) -> Arr, Fault {
    let f = ww_check(m, "taylor");
    if f.kind != F_NONE { return empty_arr(), f; }
    if nbar < 1 || nbar > 1000 || !(sll > 0.0) || !(sll < 1000.0) {
        return empty_arr(), fault(F_RANGE, "taylor", "nbar or sll stands outside its range");
    }
    var out: f64[>..] = [];
    if m == 1 { out.push(1.0); return ww_arr(out), no_fault(); }
    var len = m;
    if !sym { len = m + 1; }
    let pi = 3.141592653589793;
    let b = mathshim::power(10.0, sll / 20.0);
    let aa = ww_acosh(b) / pi;
    let nb = nbar * 1.0;
    let s2 = nb * nb / (aa * aa + (nb - 0.5) * (nb - 0.5));
    var fm: f64[>..] = [];
    var mi: i64 = 0;
    while mi < nbar - 1 {
        let m1 = (mi + 1) * 1.0;
        var numer: f64 = 1.0;
        if mi % 2 == 1 { numer = -1.0; }
        var denom: f64 = 2.0;
        var j: i64 = 0;
        while j < nbar - 1 {
            let mj = (j + 1) * 1.0;
            numer = numer * (1.0 - m1 * m1 / s2 / (aa * aa + (mj - 0.5) * (mj - 0.5)));
            if j != mi { denom = denom * (1.0 - m1 * m1 / (mj * mj)); }
            j += 1;
        }
        fm.push(numer / denom);
        mi += 1;
    }
    let mm = len * 1.0;
    var scale: f64 = 1.0;
    if norm {
        var w0: f64 = 1.0;
        var k: i64 = 0;
        while k < fm.len {
            w0 += 2.0 * fm[k] * mathshim::cosine(2.0 * pi * (k + 1) * ((mm - 1.0) / 2.0 - mm / 2.0 + 0.5) / mm);
            k += 1;
        }
        scale = 1.0 / w0;
    }
    var i: i64 = 0;
    while i < m {
        var w: f64 = 1.0;
        var k: i64 = 0;
        while k < fm.len {
            w += 2.0 * fm[k] * mathshim::cosine(2.0 * pi * (k + 1) * (i - mm / 2.0 + 0.5) / mm);
            k += 1;
        }
        out.push(w * scale);
        i += 1;
    }
    return ww_arr(out), no_fault();
}

// chebwin gives the window of Dolph and Chebyshev with the side lobes
// at at decibels below the main lobe.
fn chebwin(m: i64, at: f64, sym: bool = true) -> Arr, Fault {
    let f = ww_check(m, "chebwin");
    if f.kind != F_NONE { return empty_arr(), f; }
    if !(at - at == 0.0) {
        return empty_arr(), fault(F_VALUE, "chebwin", "the attenuation is not finite");
    }
    var out: f64[>..] = [];
    if m == 1 { out.push(1.0); return ww_arr(out), no_fault(); }
    if m > 1000000 {
        return empty_arr(), fault(F_LIMIT, "chebwin", "the length stands above one million");
    }
    var len = m;
    if !sym { len = m + 1; }
    let pi = 3.141592653589793;
    let order = len - 1.0;
    let val = mathshim::power(10.0, mathshim::abs_of(at) / 20.0);
    if !(val >= 1.0) || mathshim::is_infinite_of(val) != 0.0 {
        return empty_arr(), fault(F_RANGE, "chebwin", "the attenuation stands outside its range");
    }
    let beta = mathshim::cosh_of(ww_acosh(val) / order);
    var pr: f64[>..] = [];
    var pi_: f64[>..] = [];
    var k: i64 = 0;
    while k < len {
        let x = beta * mathshim::cosine(pi * k / len);
        var p: f64 = 0.0;
        if x > 1.0 { p = mathshim::cosh_of(order * ww_acosh(x)); }
        else if x < -1.0 {
            var sg: f64 = -1.0;
            if len % 2 == 1 { sg = 1.0; }
            p = sg * mathshim::cosh_of(order * ww_acosh(-x));
        } else { p = mathshim::cosine(order * mathshim::arccosine_of(x)); }
        if len % 2 == 1 {
            pr.push(p);
            pi_.push(0.0);
        } else {
            pr.push(p * mathshim::cosine(pi / len * k));
            pi_.push(p * mathshim::sine(pi / len * k));
        }
        k += 1;
    }
    let wr, wi, ft = fft_parts(pr, pi_);
    if ft.kind != F_NONE { return empty_arr(), ft; }
    var w: f64[>..] = [];
    if len % 2 == 1 {
        let n = (len + 1) / 2;
        var i: i64 = n - 1;
        while i >= 1 { w.push(wr.data[i]); i -= 1; }
        i = 0;
        while i < n { w.push(wr.data[i]); i += 1; }
    } else {
        let n = len / 2 + 1;
        var i: i64 = n - 1;
        while i >= 1 { w.push(wr.data[i]); i -= 1; }
        i = 1;
        while i < n { w.push(wr.data[i]); i += 1; }
    }
    var mx = w[0];
    var q: i64 = 1;
    while q < w.len { if w[q] > mx { mx = w[q]; } q += 1; }
    var o: i64 = 0;
    while o < m { out.push(w[o] / mx); o += 1; }
    return ww_arr(out), no_fault();
}

// kaiser_bessel_derived gives the window of Kaiser and Bessel for an
// even length. The window is defined for the symmetric form only.
fn kaiser_bessel_derived(m: i64, beta: f64) -> Arr, Fault {
    let f = ww_check(m, "kaiser_bessel_derived");
    if f.kind != F_NONE { return empty_arr(), f; }
    if m % 2 == 1 {
        return empty_arr(), fault(F_VALUE, "kaiser_bessel_derived", "the window needs an even length");
    }
    if !(beta >= 0.0 && beta <= 700.0) {
        return empty_arr(), fault(F_RANGE, "kaiser_bessel_derived", "beta stands outside zero to 700");
    }
    let kw, fk = kaiser(m / 2 + 1, beta);
    if fk.kind != F_NONE { return empty_arr(), fk; }
    var cs: f64[>..] = [];
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < kw.data.len { s += kw.data[i]; cs.push(s); i += 1; }
    var half: f64[>..] = [];
    i = 0;
    while i < cs.len - 1 { half.push(mathshim::root(cs[i] / cs[cs.len - 1])); i += 1; }
    var out: f64[>..] = copy_slice(half);
    i = half.len - 1;
    while i >= 0 { out.push(half[i]); i -= 1; }
    return ww_arr(out), no_fault();
}
