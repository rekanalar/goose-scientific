// The frequency responses of a filter.
//
// The calls of this file give the response of an analog or a digital
// filter over a grid of frequencies. The names follow the calls of
// scipy.signal. An analog response sits at the point j w, a digital
// response at the unit circle. The grid of the analog calls comes
// from the roots of the two polynomials, as in SciPy.
//
// The package holds real numbers only, so a response comes back in
// three arrays: the frequencies, the real parts and the imaginary
// parts. The complex roots come in two lists, the real parts and the
// imaginary parts, as the calls of the forms do.
namespace scientific;

import mathshim;

// The largest count of the points of a grid.
const RESP_GRID_MAX: i64 = 65536;

// sig_grid_fault checks the count of the points of a grid.
fn sig_grid_fault(n: i64, tag: const u8[:]) -> Fault {
    if n < 1 || n > RESP_GRID_MAX {
        return fault(F_RANGE, tag, "the count of the points holds a value outside one to 65536");
    }
    return no_fault();
}

// sig_snaps roots of a real polynomial can hold a small imaginary part
// from the method. A size below the tolerance counts as zero, as the
// roots of SciPy do.
fn sig_snaps(im: f64, re: f64) -> f64 {
    if mathshim::abs_of(im) <= 1.0e-10 * (1.0 + mathshim::abs_of(re)) { return 0.0; }
    return im;
}

// sig_findfreqs_zp gives the grid of the analog calls from the poles
// and the zeros.
fn sig_findfreqs_zp(pr: f64[:], pi_: f64[:], zr: f64[:], zi: f64[:], n: i64, tag: const u8[:]) -> Arr, Fault {
    let f = sig_grid_fault(n, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    // The pole with an imaginary part of zero or above, and the zeros
    // inside the range with an imaginary part of zero or above.
    var lo: f64 = 0.0;
    var hi: f64 = 0.0;
    var first = true;
    var i: i64 = 0;
    while i < pr.len {
        let im = sig_snaps(pi_[i], pr[i]);
        if im >= 0.0 {
            let h = 3.0 * mathshim::abs_of(pr[i] + sig_tiny(pi_[i], pr[i])) + 1.5 * im;
            let l = mathshim::abs_of(pr[i] + sig_tiny(pi_[i], pr[i])) + 2.0 * im;
            if first { hi = h; lo = l; first = false; }
            else {
                if h > hi { hi = h; }
                if l < lo { lo = l; }
            }
        }
        i += 1;
    }
    i = 0;
    while i < zr.len {
        let im = sig_snaps(zi[i], zr[i]);
        let size = mathshim::hypot_of(zr[i], zi[i]);
        if im >= 0.0 && size < 1.0e5 {
            let h = 3.0 * mathshim::abs_of(zr[i] + sig_tiny(zi[i], zr[i])) + 1.5 * im;
            let l = mathshim::abs_of(zr[i] + sig_tiny(zi[i], zr[i])) + 2.0 * im;
            if first { hi = h; lo = l; first = false; }
            else {
                if h > hi { hi = h; }
                if l < lo { lo = l; }
            }
        }
        i += 1;
    }
    if first {
        return empty_arr(), fault(F_VALUE, tag, "the grid holds no frequency");
    }
    let hfreq = mathshim::round_of(mathshim::log10_of(hi) + 0.5);
    let lfreq = mathshim::round_of(mathshim::log10_of(0.1 * lo) - 0.5);
    let w, fw = logspace(lfreq, hfreq, n, 10.0);
    if fw.kind != F_NONE { return empty_arr(), fw; }
    return w, no_fault();
}

// sig_tiny gives one when the size of a root is tiny, else zero. The
// roots near the origin move the decade of the grid, as in SciPy.
fn sig_tiny(im: f64, re: f64) -> f64 {
    let size = mathshim::hypot_of(re, im);
    if size < 1.0e-10 { return 1.0; }
    return 0.0;
}

// findfreqs gives the grid of the frequencies of the analog calls.
// The grid runs over the decades that hold the roots of the numerator
// and the denominator, as in SciPy. The count of the points is n.
fn findfreqs(num: f64[:], den: f64[:], n: i64) -> Arr, Fault {
    if num.len < 1 || den.len < 1 {
        return empty_arr(), fault(F_VALUE, "findfreqs", "a coefficient list holds no element");
    }
    let zre, zim, fz = roots(num);
    if fz.kind != F_NONE { return empty_arr(), fz; }
    let pre, pim, fp = roots(den);
    if fp.kind != F_NONE { return empty_arr(), fp; }
    // The list of the poles of SciPy holds the point -1000 when the
    // denominator holds no root.
    var pr: f64[>..] = [];
    var pi_: f64[>..] = [];
    if pre.data.len == 0 {
        pr.push(-1000.0);
        pi_.push(0.0);
    } else {
        var i: i64 = 0;
        while i < pre.data.len { pr.push(pre.data[i]); pi_.push(pim.data[i]); i += 1; }
    }
    var zr: f64[>..] = [];
    var zi: f64[>..] = [];
    var i: i64 = 0;
    while i < zre.data.len { zr.push(zre.data[i]); zi.push(zim.data[i]); i += 1; }
    return sig_findfreqs_zp(pr, pi_, zr, zi, n, "findfreqs");
}

// sig_poly_at gives the value of a polynomial with real coefficients
// at the analog point j w.
fn sig_poly_at(c: f64[:], w: f64) -> Cplx {
    let z = Cplx { re: 0.0, im: w };
    return c_value(c, z);
}

// sig_section_at gives the value of a section polynomial at the point
// of the unit circle with the angle om. The list holds the
// coefficients with the lowest power first.
fn sig_section_at(c: f64[:], om: f64) -> Cplx {
    var re: f64 = 0.0;
    var im: f64 = 0.0;
    var k: i64 = 0;
    while k < c.len {
        re += c[k] * mathshim::cosine(om * k);
        im -= c[k] * mathshim::sine(om * k);
        k += 1;
    }
    return Cplx { re: re, im: im };
}

// sig_prod_from_roots gives the product of the distances of the point
// to the roots.
fn sig_prod_from_roots(rr: Cplx[:], s: Cplx) -> Cplx {
    var t = Cplx { re: 1.0, im: 0.0 };
    var i: i64 = 0;
    while i < rr.len { t = c_mul(t, c_sub(s, rr[i])); i += 1; }
    return t;
}

// freqs gives the response of an analog filter at n points. The grid
// holds the decades of the roots, as in SciPy. The response comes back
// as the frequencies, the real parts and the imaginary parts.
fn freqs(b: f64[:], a: f64[:], n: i64 = 200) -> Arr, Arr, Arr, Fault {
    let w, f = findfreqs(b, a, n);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    var i: i64 = 0;
    while i < w.data.len {
        let num = sig_poly_at(b, w.data[i]);
        let den = sig_poly_at(a, w.data[i]);
        if den.re == 0.0 && den.im == 0.0 {
            return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "freqs", "the denominator holds zero at a point");
        }
        let h = c_div(num, den);
        hr.push(h.re);
        hi.push(h.im);
        i += 1;
    }
    return w, flt_arr(copy(hr)), flt_arr(copy(hi)), no_fault();
}

// freqs_zpk gives the response of an analog filter from its zeros,
// its poles and its gain, at n points. The roots come in two lists
// each: the real parts and the imaginary parts.
fn freqs_zpk(zr: f64[:], zi: f64[:], pr: f64[:], pi_: f64[:], k: f64, n: i64 = 200) -> Arr, Arr, Arr, Fault {
    let fz = form_roots_check(zr, zi, "freqs_zpk");
    if fz.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), fz; }
    let fp = form_roots_check(pr, pi_, "freqs_zpk");
    if fp.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), fp; }
    let w, f = sig_findfreqs_zp(pr, pi_, zr, zi, n, "freqs_zpk");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
    let z = rs_zip(zr, zi);
    let p = rs_zip(pr, pi_);
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    var i: i64 = 0;
    while i < w.data.len {
        let s = Cplx { re: 0.0, im: w.data[i] };
        let den = sig_prod_from_roots(p, s);
        if den.re == 0.0 && den.im == 0.0 {
            return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "freqs_zpk", "a pole sits on the point j w");
        }
        let h = c_scale(c_div(sig_prod_from_roots(z, s), den), k);
        hr.push(h.re);
        hi.push(h.im);
        i += 1;
    }
    return w, flt_arr(copy(hr)), flt_arr(copy(hi)), no_fault();
}

// sig_unit_grid gives the grid of the digital calls: n points from
// zero up to, but not including, pi or 2 pi.
fn sig_unit_grid(n: i64, whole: bool, tag: const u8[:]) -> Arr, Fault {
    let f = sig_grid_fault(n, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    var top: f64 = 3.141592653589793;
    if whole { top = 6.283185307179586; }
    var w: f64[>..] = [];
    var i: i64 = 0;
    while i < n { w.push(top * i / n); i += 1; }
    return flt_arr(copy(w)), no_fault();
}

// freqz_sos gives the response of a digital filter given as sections
// of the order two, at n points. Each row of the sections holds b0,
// b1, b2, a0, a1 and a2. The response is the product of the
// responses of the sections.
fn freqz_sos(sos: Arr, n: i64 = 512, whole: bool = false) -> Arr, Arr, Arr, Fault {
    let w, f = sig_unit_grid(n, whole, "freqz_sos");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
    if sos.ndim != 2 || sos.shape[1] != 6 {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "freqz_sos", "the sections do not hold six coefficients per row");
    }
    if sos.shape[0] < 1 {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "freqz_sos", "the sections hold no row");
    }
    // The sections are read by their shape, so the buffer must agree
    // with the shape.
    let fs = check_stack(sos.ndim, sos.shape, sos.data.len, 1, "freqz_sos");
    if fs.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), fs; }
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        hr.push(1.0);
        hi.push(0.0);
        i += 1;
    }
    var q: i64 = 0;
    while q < sos.shape[0] {
        var b: f64[>..] = [];
        var a: f64[>..] = [];
        var j: i64 = 0;
        while j < 3 { b.push(sos.data[q * 6 + j]); a.push(sos.data[q * 6 + 3 + j]); j += 1; }
        i = 0;
        while i < n {
            let om = w.data[i];
            let num = sig_section_at(b, om);
            let den = sig_section_at(a, om);
            if den.re == 0.0 && den.im == 0.0 {
                return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "freqz_sos", "a section holds a zero at a point");
            }
            let h = c_div(num, den);
            let pr = hr[i] * h.re - hi[i] * h.im;
            let pi_ = hr[i] * h.im + hi[i] * h.re;
            hr[i] = pr;
            hi[i] = pi_;
            i += 1;
        }
        q += 1;
    }
    return w, flt_arr(copy(hr)), flt_arr(copy(hi)), no_fault();
}

// freqz_zpk gives the response of a digital filter from its zeros,
// its poles and its gain, at n points. The roots come in two lists
// each: the real parts and the imaginary parts. The point of the map
// uses the angle with a positive sign, as SciPy does.
fn freqz_zpk(zr: f64[:], zi: f64[:], pr: f64[:], pi_: f64[:], k: f64, n: i64 = 512, whole: bool = false) -> Arr, Arr, Arr, Fault {
    let fz = form_roots_check(zr, zi, "freqz_zpk");
    if fz.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), fz; }
    let fp = form_roots_check(pr, pi_, "freqz_zpk");
    if fp.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), fp; }
    let w, f = sig_unit_grid(n, whole, "freqz_zpk");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
    let z = rs_zip(zr, zi);
    let p = rs_zip(pr, pi_);
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        let zm = Cplx { re: mathshim::cosine(w.data[i]), im: mathshim::sine(w.data[i]) };
        let den = sig_prod_from_roots(p, zm);
        if den.re == 0.0 && den.im == 0.0 {
            return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "freqz_zpk", "a pole sits on the unit circle");
        }
        let h = c_scale(c_div(sig_prod_from_roots(z, zm), den), k);
        hr.push(h.re);
        hi.push(h.im);
        i += 1;
    }
    return w, flt_arr(copy(hr)), flt_arr(copy(hi)), no_fault();
}

// group_delay gives the group delay of a digital filter at n points,
// in radians per sample. The delay at a singular point holds zero.
fn group_delay(b: f64[:], a: f64[:], n: i64 = 512, whole: bool = false) -> Arr, Arr, Fault {
    if b.len < 1 || a.len < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "group_delay", "a coefficient list holds no element");
    }
    let f = sig_grid_fault(n, "group_delay");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    if !all_finite(b) || !all_finite(a) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "group_delay", "a coefficient is not finite");
    }
    // c is the product of the numerator with the reversed denominator,
    // and cr holds the coefficients with their index, as SciPy does.
    var arev: f64[>..] = [];
    var j = a.len - 1;
    while j >= 0 { arev.push(a[j]); j -= 1; }
    var c: f64[>..] = [];
    flt_pmul(b, arev, c);
    // cr holds the coefficients with their index.
    var cr: f64[>..] = [];
    var i: i64 = 0;
    while i < c.len { cr.push(c[i] * i); i += 1; }
    var top: f64 = 3.141592653589793;
    if whole { top = 6.283185307179586; }
    var w: f64[>..] = [];
    var gd: f64[>..] = [];
    i = 0;
    while i < n {
        let om = top * i / n;
        w.push(om);
        let z = Cplx { re: mathshim::cosine(om), im: -mathshim::sine(om) };
        // num holds z times the derivative, den holds the polynomial.
        let num = sig_poly_unit(cr, z);
        let den = sig_poly_unit(c, z);
        let h = c_div(num, den);
        var v = h.re - a.len + 1;
        if mathshim::is_finite_of(v) == 0.0 { v = 0.0; }
        gd.push(v);
        i += 1;
    }
    return flt_arr(copy(w)), flt_arr(copy(gd)), no_fault();
}

// sig_poly_unit gives the value of a polynomial with real
// coefficients at the point z of the unit circle.
fn sig_poly_unit(c: f64[:], z: Cplx) -> Cplx {
    var t = Cplx { re: 0.0, im: 0.0 };
    var i: i64 = c.len - 1;
    while i >= 0 { t = c_mul(t, z); t = c_add(t, Cplx { re: c[i], im: 0.0 }); i -= 1; }
    return t;
}
