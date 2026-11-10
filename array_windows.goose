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

// dp_count gives the count of eigenvalues of the tridiagonal matrix
// that lie below x, by the sequence of Sturm.
fn dp_count(d: f64[:], e: f64[:], x: f64) -> i64 {
    var cnt: i64 = 0;
    var q: f64 = d[0] - x;
    if q < 0.0 { cnt += 1; }
    var i: i64 = 1;
    while i < d.len {
        if q == 0.0 { q = 1.0e-300; }
        q = d[i] - x - e[i - 1] * e[i - 1] / q;
        if q < 0.0 { cnt += 1; }
        i += 1;
    }
    return cnt;
}

// dp_vector solves the tridiagonal system (T - lambda I) v = b by
// elimination and gives v.
fn dp_solve(d: f64[:], e: f64[:], lam: f64, b: f64[:]) -> f64[>..] {
    let n = d.len;
    var diag: f64[>..] = [];
    var rhs: f64[>..] = copy_slice(b);
    var i: i64 = 0;
    while i < n { diag.push(d[i] - lam); i += 1; }
    i = 1;
    while i < n {
        if diag[i - 1] == 0.0 { diag[i - 1] = 1.0e-300; }
        let r = e[i - 1] / diag[i - 1];
        diag[i] = diag[i] - r * e[i - 1];
        rhs[i] = rhs[i] - r * rhs[i - 1];
        i += 1;
    }
    var v: f64[>..] = copy_slice(rhs);
    if diag[n - 1] == 0.0 { diag[n - 1] = 1.0e-300; }
    v[n - 1] = rhs[n - 1] / diag[n - 1];
    i = n - 2;
    while i >= 0 {
        if diag[i] == 0.0 { diag[i] = 1.0e-300; }
        v[i] = (rhs[i] - e[i] * v[i + 1]) / diag[i];
        i -= 1;
    }
    return v;
}

// dpss gives the discrete prolate spheroidal sequences, the windows of
// Slepian. The value nw is the product of the time and the half
// bandwidth. A kmax of zero gives the first window as a vector. A
// positive kmax gives the first kmax windows as rows. The norm is
// "approximate", "subsample", "2" or "" for the default of SciPy. The
// eigenvalue problem is tridiagonal, solved by bisection and inverse
// iteration.
fn dpss(m: i64, nw: f64, kmax: i64 = 0, sym: bool = true, norm: const u8[:] = "") -> Arr, Fault {
    let w, r, f = dp_run(m, nw, kmax, sym, norm, false);
    return w, f;
}

// dpss_ratios gives the windows of dpss and the concentration ratios of
// the windows. The ratios are a vector of kmax numbers, or an array of
// no dimension when kmax is zero.
fn dpss_ratios(m: i64, nw: f64, kmax: i64 = 0, sym: bool = true, norm: const u8[:] = "") -> Arr, Arr, Fault {
    return dp_run(m, nw, kmax, sym, norm, true);
}

// dp_ratio gives the share of the energy of one window inside the band.
// It sums the autocorrelation of the window times the kernel of the
// band. The FFT gives the autocorrelation.
fn dp_ratio(v: f64[:], w: f64) -> f64, Fault {
    let n = v.len;
    var size: i64 = 1;
    while size < 2 * n - 1 { size = size * 2; }
    var re: f64[>..] = [];
    var im: f64[>..] = [];
    var i: i64 = 0;
    while i < size {
        if i < n { re.push(v[i]); } else { re.push(0.0); }
        im.push(0.0);
        i += 1;
    }
    let xr, xi, f1 = fft_parts(re, im);
    if f1.kind != F_NONE { return 0.0, f1; }
    var pw: f64[>..] = [];
    var zero: f64[>..] = [];
    i = 0;
    while i < size {
        pw.push(xr.data[i] * xr.data[i] + xi.data[i] * xi.data[i]);
        zero.push(0.0);
        i += 1;
    }
    let cr, ci, f2 = fft_parts(pw, zero);
    if f2.kind != F_NONE { return 0.0, f2; }
    var t: f64 = 2.0 * w * cr.data[0] / size;
    i = 1;
    while i < n {
        t += 4.0 * w * mathshim::sinc_of(2.0 * w * i) * cr.data[i] / size;
        i += 1;
    }
    return t, no_fault();
}

fn dp_run(m: i64, nw: f64, kmax: i64, sym: bool, norm: const u8[:], want: bool) -> Arr, Arr, Fault {
    let f = ww_check(m, "dpss");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    if m > 100000 {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "dpss", "the length stands above 100000");
    }
    if norm != "" && norm != "approximate" && norm != "subsample" && norm != "2" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "dpss", "the norm must be approximate, subsample or 2");
    }
    if !(nw > 0.0) || !(nw < m / 2.0) {
        return empty_arr(), empty_arr(), fault(F_RANGE, "dpss", "nw must lie above zero and below half the length");
    }
    var single = false;
    var k = kmax;
    if k == 0 { single = true; k = 1; }
    if k < 0 || k > m {
        return empty_arr(), empty_arr(), fault(F_RANGE, "dpss", "kmax must lie from one to the length");
    }
    var len = m;
    if !sym { len = m + 1; }
    let pi = 3.141592653589793;
    let w = nw / len;
    var d: f64[>..] = [];
    var e: f64[>..] = [];
    var i: i64 = 0;
    let cw = mathshim::cosine(2.0 * pi * w);
    while i < len {
        let t = (len - 1.0 - 2.0 * i) / 2.0;
        d.push(t * t * cw);
        if i > 0 { e.push(i * (len - i) / 2.0); }
        i += 1;
    }
    if len == 1 {
        var one: f64[>..] = [1.0];
        var ro: f64[>..] = [1.0];
        var rsh: i64[..8] = [];
        if !single { rsh.push(1); }
        return line_arr(copy(one)), Arr { ndim: rsh.len, shape: rsh, data: ro }, no_fault();
    }
    // The bounds of Gershgorin.
    var lo0: f64 = d[0];
    var hi0: f64 = d[0];
    i = 0;
    while i < len {
        var r: f64 = 0.0;
        if i > 0 { r += mathshim::abs_of(e[i - 1]); }
        if i < len - 1 { r += mathshim::abs_of(e[i]); }
        if d[i] - r < lo0 { lo0 = d[i] - r; }
        if d[i] + r > hi0 { hi0 = d[i] + r; }
        i += 1;
    }
    var rows: f64[>..] = [];
    var kk: i64 = 0;
    while kk < k {
        let j = len - 1 - kk;
        var lo = lo0;
        var hi = hi0;
        var it: i64 = 0;
        while it < 200 {
            let mid = 0.5 * (lo + hi);
            if mid == lo || mid == hi { break; }
            if dp_count(d, e, mid) > j { hi = mid; } else { lo = mid; }
            it += 1;
        }
        let lam = 0.5 * (lo + hi);
        let shift = lam + 1.0e-13 * (1.0 + mathshim::abs_of(lam));
        var v: f64[>..] = [];
        i = 0;
        while i < len { v.push(1.0 + 0.37 * mathshim::sine(1.7 * i + 0.3)); i += 1; }
        var iter: i64 = 0;
        while iter < 6 {
            var nv = dp_solve(d, e, shift, v);
            // Remove the parts along the earlier vectors.
            var p: i64 = 0;
            while p < kk {
                var dot: f64 = 0.0;
                i = 0;
                while i < len { dot += nv[i] * rows[p * len + i]; i += 1; }
                i = 0;
                while i < len { nv[i] = nv[i] - dot * rows[p * len + i]; i += 1; }
                p += 1;
            }
            var nrm: f64 = 0.0;
            i = 0;
            while i < len { nrm += nv[i] * nv[i]; i += 1; }
            nrm = mathshim::root(nrm);
            i = 0;
            while i < len { nv[i] = nv[i] / nrm; i += 1; }
            v = copy_slice(nv);
            iter += 1;
        }
        // The signs follow the convention of Percival and Walden.
        if kk % 2 == 0 {
            var sum: f64 = 0.0;
            i = 0;
            while i < len { sum += v[i]; i += 1; }
            if sum < 0.0 { i = 0; while i < len { v[i] = -v[i]; i += 1; } }
        } else {
            var thresh: f64 = 1.0e-7;
            if 1.0 / len > thresh { thresh = 1.0 / len; }
            i = 0;
            while i < len && v[i] * v[i] <= thresh { i += 1; }
            if i < len && v[i] < 0.0 { i = 0; while i < len { v[i] = -v[i]; i += 1; } }
        }
        i = 0;
        while i < len { rows.push(v[i]); i += 1; }
        kk += 1;
    }
    var ratios: f64[>..] = [];
    if want {
        kk = 0;
        while kk < k {
            var row: f64[>..] = [];
            i = 0;
            while i < len { row.push(rows[kk * len + i]); i += 1; }
            let rt, fr = dp_ratio(row, w);
            if fr.kind != F_NONE { return empty_arr(), empty_arr(), fr; }
            ratios.push(rt);
            kk += 1;
        }
    }
    var rshape: i64[..8] = [];
    if !single { rshape.push(k); }
    var nrm_mode = norm;
    if nrm_mode == "" {
        if single { nrm_mode = "approximate"; } else { nrm_mode = "2"; }
    }
    if nrm_mode != "2" {
        var mx = rows[0];
        i = 1;
        while i < rows.len { if rows[i] > mx { mx = rows[i]; } i += 1; }
        var corr: f64 = 1.0 / mx;
        if len % 2 == 0 && nrm_mode == "approximate" { corr = corr * len * len / (len * len + nw); }
        if len % 2 == 0 && nrm_mode == "subsample" {
            // The first window shifted by a fraction of a sample.
            var pr: f64[>..] = [];
            var pz: f64[>..] = [];
            i = 0;
            while i < len { pr.push(rows[i] * corr); pz.push(0.0); i += 1; }
            let sr, si, fs = fft_parts(pr, pz);
            if fs.kind != F_NONE { return empty_arr(), empty_arr(), fs; }
            var tot: f64 = sr.data[0];
            var q: i64 = 1;
            while q <= len / 2 {
                let ang = 3.141592653589793 * (1.0 - 1.0 / len) * q;
                tot += 2.0 * (sr.data[q] * mathshim::cosine(ang) - si.data[q] * mathshim::sine(ang));
                q += 1;
            }
            corr = corr * len / tot;
        }
        i = 0;
        while i < rows.len { rows[i] = rows[i] * corr; i += 1; }
    }
    var out: f64[>..] = [];
    kk = 0;
    while kk < k {
        i = 0;
        while i < m { out.push(rows[kk * len + i]); i += 1; }
        kk += 1;
    }
    let rarr = Arr { ndim: rshape.len, shape: rshape, data: copy(ratios) };
    if single { return ww_arr(out), rarr, no_fault(); }
    var sh: i64[..8] = [];
    sh.push(k);
    sh.push(m);
    return Arr { ndim: 2, shape: sh, data: out }, rarr, no_fault();
}
