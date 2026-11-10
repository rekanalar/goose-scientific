// The cosine and sine transforms of the types 1 to 4, the transforms of
// a Hermitian signal, the real transforms of many axes and the length
// that the transform runs fastest at. The cosine and sine transforms
// run on the complex transform of the package, with the lengths 2N,
// 2(N - 1) and 2(N + 1), so the cost grows as N log N.
namespace scientific;

import mathshim;

const PI_DCT: f64 = 3.141592653589793;

// dct_kind_check checks the number of the type.
fn dct_kind_check(kind: i64, tag: const u8[:]) -> Fault {
    if kind < 1 || kind > 4 {
        return fault(F_VALUE, tag, "the type holds a value outside one to four");
    }
    return no_fault();
}

// norm_code gives 0 for backward, 1 for ortho, 2 for forward and -1 for
// another name.
fn norm_code(norm: const u8[:]) -> i64 {
    if norm == "backward" { return 0; }
    if norm == "ortho" { return 1; }
    if norm == "forward" { return 2; }
    return -1;
}

// dct_core gives the cosine transform of a type with no scale.
fn dct_core(x: f64[:], kind: i64, out: f64[>..]&) -> Fault {
    let n = x.len;
    var re: f64[>..] = [];
    var im: f64[>..] = [];
    var i: i64 = 0;
    if kind == 1 {
        if n < 2 { return fault(F_SHAPE, "dct", "the type one needs two points"); }
        while i < n { re.push(x[i]); i += 1; }
        i = 0;
        while i < n - 2 { re.push(x[n - 2 - i]); i += 1; }
        i = 0;
        while i < re.len { im.push(0.0); i += 1; }
        let fr, fi, f = fft_parts(re, im);
        if f.kind != F_NONE { return f; }
        i = 0;
        while i < n { out.push(fr.data[i]); i += 1; }
        return no_fault();
    }
    if kind == 2 {
        while i < n { re.push(x[i]); i += 1; }
        i = 0;
        while i < n { re.push(x[n - 1 - i]); i += 1; }
        i = 0;
        while i < 2 * n { im.push(0.0); i += 1; }
        let fr, fi, f = fft_parts(re, im);
        if f.kind != F_NONE { return f; }
        i = 0;
        while i < n {
            let th = PI_DCT * i / (2.0 * n);
            out.push(fr.data[i] * mathshim::cosine(th) + fi.data[i] * mathshim::sine(th));
            i += 1;
        }
        return no_fault();
    }
    // The types three and four feed the weighted input, turned by the
    // phase, into the transform of 2N points.
    while i < n {
        let ph = PI_DCT * i / (2.0 * n);
        var w: f64 = 1.0;
        if kind == 3 && i > 0 { w = 2.0; }
        re.push(w * x[i] * mathshim::cosine(ph));
        im.push(-w * x[i] * mathshim::sine(ph));
        i += 1;
    }
    i = 0;
    while i < n { re.push(0.0); im.push(0.0); i += 1; }
    let fr, fi, f = fft_parts(re, im);
    if f.kind != F_NONE { return f; }
    i = 0;
    while i < n {
        if kind == 3 { out.push(fr.data[i]); }
        else {
            let psi = PI_DCT * (2.0 * i + 1.0) / (4.0 * n);
            out.push(2.0 * (fr.data[i] * mathshim::cosine(psi) + fi.data[i] * mathshim::sine(psi)));
        }
        i += 1;
    }
    return no_fault();
}

// dct_ortho gives the cosine transform with the scale that makes the
// transform orthonormal.
fn dct_ortho(x: f64[:], kind: i64, out: f64[>..]&) -> Fault {
    let n = x.len;
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < n { v.push(x[i]); i += 1; }
    if kind == 1 {
        let r2 = mathshim::root(2.0);
        if n < 2 { return fault(F_SHAPE, "dct", "the type one needs two points"); }
        v[0] = v[0] * r2;
        v[n - 1] = v[n - 1] * r2;
    }
    if kind == 3 {
        v[0] = v[0] * mathshim::root(1.0 / n);
        i = 1;
        while i < n { v[i] = v[i] * mathshim::root(1.0 / (2.0 * n)); i += 1; }
    }
    var y: f64[>..] = [];
    let f = dct_core(v, kind, y);
    if f.kind != F_NONE { return f; }
    i = 0;
    while i < n {
        var s: f64 = 1.0;
        if kind == 1 {
            s = mathshim::root(1.0 / (2.0 * (n - 1)));
            if i == 0 || i == n - 1 { s = s / mathshim::root(2.0); }
        } else if kind == 2 {
            s = mathshim::root(1.0 / (2.0 * n));
            if i == 0 { s = mathshim::root(1.0 / (4.0 * n)); }
        } else if kind == 4 {
            s = mathshim::root(1.0 / (2.0 * n));
        }
        out.push(y[i] * s);
        i += 1;
    }
    return no_fault();
}

// dst_core gives the sine transform of a type with no scale, or the
// ortho scale when the flag holds.
fn dst_core(x: f64[:], kind: i64, ortho: bool, out: f64[>..]&) -> Fault {
    let n = x.len;
    var i: i64 = 0;
    if kind == 1 {
        var re: f64[>..] = [0.0];
        var im: f64[>..] = [];
        while i < n { re.push(x[i]); i += 1; }
        re.push(0.0);
        i = 0;
        while i < n { re.push(-x[n - 1 - i]); i += 1; }
        i = 0;
        while i < re.len { im.push(0.0); i += 1; }
        let fr, fi, f = fft_parts(re, im);
        if f.kind != F_NONE { return f; }
        var s: f64 = 1.0;
        if ortho { s = mathshim::root(1.0 / (2.0 * (n + 1))); }
        i = 0;
        while i < n { out.push(-fi.data[i + 1] * s); i += 1; }
        return no_fault();
    }
    var v: f64[>..] = [];
    var sign: f64 = 1.0;
    i = 0;
    while i < n {
        if kind == 3 { v.push(x[n - 1 - i]); }
        else { v.push(sign * x[i]); }
        sign = -sign;
        i += 1;
    }
    var z: f64[>..] = [];
    var f: Fault = no_fault();
    if kind == 3 {
        if ortho { f = dct_ortho(v, 3, z); } else { f = dct_core(v, 3, z); }
    } else {
        if ortho { f = dct_ortho(v, kind, z); } else { f = dct_core(v, kind, z); }
    }
    if f.kind != F_NONE { return f; }
    i = 0;
    sign = 1.0;
    while i < n {
        if kind == 3 { out.push(sign * z[i]); }
        else { out.push(z[n - 1 - i]); }
        sign = -sign;
        i += 1;
    }
    return no_fault();
}

// fwd_scale gives the scale of the forward transform for the norm
// "forward": one over 2N, with N - 1 for the cosine of type one and
// N + 1 for the sine of type one.
fn fwd_scale(n: i64, kind: i64, sine: bool) -> f64 {
    if kind == 1 && !sine { return 1.0 / (2.0 * (n - 1)); }
    if kind == 1 { return 1.0 / (2.0 * (n + 1)); }
    return 1.0 / (2.0 * n);
}

fn inverse_kind(kind: i64) -> i64 {
    if kind == 2 { return 3; }
    if kind == 3 { return 2; }
    return kind;
}

// sct runs a cosine or sine transform, forward or inverse, with a norm.
fn sct(x: f64[:], kind: i64, norm: const u8[:], sine: bool, inverse: bool, tag: const u8[:]) -> Arr, Fault {
    let c = dct_kind_check(kind, tag);
    if c.kind != F_NONE { return empty_arr(), c; }
    let nc = norm_code(norm);
    if nc < 0 {
        return empty_arr(), fault(F_VALUE, tag, "the norm holds an unknown name");
    }
    if x.len < 1 {
        return empty_arr(), fault(F_VALUE, tag, "the signal holds no element");
    }
    if x.len > MAX_ELEMENTS / 4 {
        return empty_arr(), fault(F_LIMIT, tag, "the signal holds more elements than the limit");
    }
    var k = kind;
    if inverse { k = inverse_kind(kind); }
    var y: f64[>..] = [];
    var f: Fault = no_fault();
    if nc == 1 {
        if sine { f = dst_core(x, k, true, y); }
        else { f = dct_ortho(x, k, y); }
    } else {
        if sine { f = dst_core(x, k, false, y); }
        else { f = dct_core(x, k, y); }
    }
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, tag, f.detail); }
    // The scale of the norms backward and forward.
    var s: f64 = 1.0;
    if nc == 0 && inverse { s = fwd_scale(x.len, k, sine); }
    if nc == 2 && !inverse { s = fwd_scale(x.len, k, sine); }
    if s != 1.0 {
        var i: i64 = 0;
        while i < y.len { y[i] = y[i] * s; i += 1; }
    }
    return line_arr(copy(y)), no_fault();
}

// dct gives the cosine transform of a row. The type kind runs from one
// to four. The norm is "backward", "ortho" or "forward". The type one
// needs two points.
fn dct(x: f64[:], kind: i64 = 2, norm: const u8[:] = "backward") -> Arr, Fault {
    return sct(x, kind, norm, false, false, "dct");
}

// idct gives the inverse of the cosine transform of a type.
fn idct(x: f64[:], kind: i64 = 2, norm: const u8[:] = "backward") -> Arr, Fault {
    return sct(x, kind, norm, false, true, "idct");
}

// dst gives the sine transform of a row. The type kind runs from one to
// four.
fn dst(x: f64[:], kind: i64 = 2, norm: const u8[:] = "backward") -> Arr, Fault {
    return sct(x, kind, norm, true, false, "dst");
}

// idst gives the inverse of the sine transform of a type.
fn idst(x: f64[:], kind: i64 = 2, norm: const u8[:] = "backward") -> Arr, Fault {
    return sct(x, kind, norm, true, true, "idst");
}

// hfft gives the real transform of a signal that holds only the
// positive half of a Hermitian spectrum. The length n of the result is
// 2 (m - 1) for m points when n is zero.
fn hfft(re: f64[:], im: f64[:], n: i64 = 0) -> Arr, Fault {
    var len = n;
    if len == 0 { len = 2 * (re.len - 1); }
    if len < 1 {
        return empty_arr(), fault(F_VALUE, "hfft", "the length holds a value below one");
    }
    var neg: f64[>..] = [];
    var i: i64 = 0;
    while i < im.len { neg.push(-im[i]); i += 1; }
    let r, f = irfft(re, neg, len);
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, "hfft", f.detail); }
    var d: f64[>..] = [];
    i = 0;
    while i < r.data.len { d.push(r.data[i] * len); i += 1; }
    return line_arr(copy(d)), no_fault();
}

// ihfft gives the positive half of the Hermitian spectrum of a real
// signal, scaled by the inverse of the length. The result holds the
// real parts and the imaginary parts.
fn ihfft(x: f64[:]) -> Arr, Arr, Fault {
    let fr, fi, f = rfft(x);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), fault(f.kind, "ihfft", f.detail); }
    var a: f64[>..] = [];
    var b: f64[>..] = [];
    var i: i64 = 0;
    while i < fr.data.len {
        a.push(fr.data[i] / x.len);
        b.push(-fi.data[i] / x.len);
        i += 1;
    }
    return line_arr(copy(a)), line_arr(copy(b)), no_fault();
}

// smooth_over tells whether a number holds no prime factor above the
// largest one.
fn smooth_over(n: i64, largest: i64) -> bool {
    var m = n;
    var p: i64 = 2;
    while p <= largest {
        var prime = true;
        var q: i64 = 2;
        while q * q <= p { if p % q == 0 { prime = false; } q += 1; }
        if prime { while m % p == 0 { m = m / p; } }
        p += 1;
    }
    return m == 1;
}

// next_fast_len gives the smallest length at or above the target that
// the transform runs fast at. A real transform takes the factors 2, 3
// and 5. A complex transform takes the factors up to 11, as SciPy
// does.
fn next_fast_len(target: i64, real: bool = false) -> i64, Fault {
    if target < 1 || target > 1099511627776 {
        return 0, fault(F_RANGE, "next_fast_len", "the target stands outside one to 2^40");
    }
    var largest: i64 = 11;
    if real { largest = 5; }
    var n = target;
    while !smooth_over(n, largest) { n += 1; }
    return n, no_fault();
}

// along_axis transforms every line of an array along one axis with the
// complex transform. The caller passes copies of the parts.
fn along_axis(dr: f64[>..]&, di: f64[>..]&, shape: i64[:], axis: i64, inverse: bool) -> Fault {
    let width = shape[axis];
    var inner: i64 = 1;
    var outer: i64 = 1;
    var i: i64 = 0;
    while i < shape.len {
        if i < axis { outer *= shape[i]; }
        if i > axis { inner *= shape[i]; }
        i += 1;
    }
    var o: i64 = 0;
    while o < outer {
        var q: i64 = 0;
        while q < inner {
            var lr: f64[>..] = [];
            var li: f64[>..] = [];
            var k: i64 = 0;
            while k < width {
                let at = (o * width + k) * inner + q;
                lr.push(dr[at]);
                li.push(di[at]);
                k += 1;
            }
            var tr: Arr = empty_arr();
            var ti: Arr = empty_arr();
            var f: Fault = no_fault();
            if inverse { tr, ti, f = ifft(lr, li); } else { tr, ti, f = fft_parts(lr, li); }
            if f.kind != F_NONE { return f; }
            k = 0;
            while k < width {
                let at = (o * width + k) * inner + q;
                dr[at] = tr.data[k];
                di[at] = ti.data[k];
                k += 1;
            }
            q += 1;
        }
        o += 1;
    }
    return no_fault();
}

// rfftn gives the transform of a real array along every axis. The last
// axis keeps its positive half only, so it holds n / 2 + 1 places.
fn rfftn(a) -> Arr, Arr, Fault {
    var dr = copy_slice(a.data);
    var di: f64[>..] = [];
    var i: i64 = 0;
    while i < dr.len { di.push(0.0); i += 1; }
    var axis: i64 = 0;
    while axis < a.ndim {
        let f = along_axis(dr, di, a.shape, axis, false);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), fault(f.kind, "rfftn", f.detail); }
        axis += 1;
    }
    let last = a.shape[a.ndim - 1];
    let keep = last / 2 + 1;
    let rows = a.data.len / last;
    var outr: f64[>..] = [];
    var outi: f64[>..] = [];
    var r: i64 = 0;
    while r < rows {
        var c: i64 = 0;
        while c < keep { outr.push(dr[r * last + c]); outi.push(di[r * last + c]); c += 1; }
        r += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    s[a.ndim - 1] = keep;
    return Arr { ndim: a.ndim, shape: s, data: outr }, Arr { ndim: a.ndim, shape: s, data: outi }, no_fault();
}

// rfft2 gives the transform of a real matrix. The last axis keeps its
// positive half.
fn rfft2(a) -> Arr, Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "rfft2", "the operand is not a matrix");
    }
    return rfftn(a);
}

// irfftn gives the real array of a transform that holds the positive
// half of the last axis. The length of the last axis of the result is
// n_last, or 2 (m - 1) for m places when n_last is zero.
fn irfftn(re, im, n_last: i64 = 0) -> Arr, Fault {
    if re.ndim != im.ndim {
        return empty_arr(), fault(F_SHAPE, "irfftn", "the parts hold a different count of dimensions");
    }
    var i: i64 = 0;
    while i < re.ndim {
        if re.shape[i] != im.shape[i] {
            return empty_arr(), fault(F_SHAPE, "irfftn", "the parts hold different shapes");
        }
        i += 1;
    }
    let m = re.shape[re.ndim - 1];
    var len = n_last;
    if len == 0 { len = 2 * (m - 1); }
    if len < 1 || len / 2 + 1 != m {
        return empty_arr(), fault(F_SHAPE, "irfftn", "the last axis does not hold the length");
    }
    var dr = copy_slice(re.data);
    var di = copy_slice(im.data);
    var axis: i64 = 0;
    while axis < re.ndim - 1 {
        let f = along_axis(dr, di, re.shape, axis, true);
        if f.kind != F_NONE { return empty_arr(), fault(f.kind, "irfftn", f.detail); }
        axis += 1;
    }
    let rows = re.data.len / m;
    var d: f64[>..] = [];
    var r: i64 = 0;
    while r < rows {
        var lr: f64[>..] = [];
        var li: f64[>..] = [];
        var c: i64 = 0;
        while c < m { lr.push(dr[r * m + c]); li.push(di[r * m + c]); c += 1; }
        let line, f = irfft(lr, li, len);
        if f.kind != F_NONE { return empty_arr(), fault(f.kind, "irfftn", f.detail); }
        c = 0;
        while c < len { d.push(line.data[c]); c += 1; }
        r += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < re.ndim { s.push(re.shape[i]); i += 1; }
    s[re.ndim - 1] = len;
    return Arr { ndim: re.ndim, shape: s, data: d }, no_fault();
}

// irfft2 gives the real matrix of a transform of a real matrix.
fn irfft2(re, im, n_last: i64 = 0) -> Arr, Fault {
    if re.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "irfft2", "the operand is not a matrix");
    }
    return irfftn(re, im, n_last);
}

// sct_axes runs a cosine or sine transform along each listed axis, in
// the order of the list. An empty list runs no transform and gives the
// array back, as SciPy does. A negative axis counts from the end.
fn sct_axes(a, axes: i64[:], kind: i64, norm: const u8[:], sine: bool, inverse: bool, tag: const u8[:]) -> Arr, Fault {
    let c = dct_kind_check(kind, tag);
    if c.kind != F_NONE { return empty_arr(), c; }
    if norm_code(norm) < 0 {
        return empty_arr(), fault(F_VALUE, tag, "the norm holds an unknown name");
    }
    var ax: i64[>..] = [];
    var i: i64 = 0;
    while i < axes.len {
        let k = fix_axis(axes[i], a.ndim);
        if k < 0 { return empty_arr(), fault(F_RANGE, tag, "an axis stands outside the shape"); }
        var j: i64 = 0;
        while j < ax.len {
            if ax[j] == k { return empty_arr(), fault(F_VALUE, tag, "an axis repeats"); }
            j += 1;
        }
        ax.push(k);
        i += 1;
    }
    var d: f64[>..] = [];
    i = 0;
    while i < a.data.len { d.push(a.data[i]); i += 1; }
    i = 0;
    while i < ax.len {
        let axis = ax[i];
        let width = a.shape[axis];
        let inner = product_after_of(a.shape, axis);
        let outer = product_before_of(a.shape, axis);
        var o: i64 = 0;
        while o < outer {
            var q: i64 = 0;
            while q < inner {
                var line: f64[>..] = [];
                var k: i64 = 0;
                while k < width { line.push(d[(o * width + k) * inner + q]); k += 1; }
                let t, f = sct(line, kind, norm, sine, inverse, tag);
                if f.kind != F_NONE { return empty_arr(), f; }
                k = 0;
                while k < width { d[(o * width + k) * inner + q] = t.data[k]; k += 1; }
                q += 1;
            }
            o += 1;
        }
        i += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}

// all_axes lists every axis of an array.
fn all_axes(nd: i64) -> i64[>..] {
    var v: i64[>..] = [];
    var i: i64 = 0;
    while i < nd { v.push(i); i += 1; }
    return v;
}

// dctn gives the cosine transform along the axes. With no axes given
// (the list holds a single value of 1000), it runs along every axis.
fn dctn(a, axes: i64[:] = [1000], kind: i64 = 2, norm: const u8[:] = "backward") -> Arr, Fault {
    if axes.len == 1 && axes[0] == 1000 { return sct_axes(a, all_axes(a.ndim), kind, norm, false, false, "dctn"); }
    return sct_axes(a, axes, kind, norm, false, false, "dctn");
}

// idctn gives the inverse cosine transform along the axes.
fn idctn(a, axes: i64[:] = [1000], kind: i64 = 2, norm: const u8[:] = "backward") -> Arr, Fault {
    if axes.len == 1 && axes[0] == 1000 { return sct_axes(a, all_axes(a.ndim), kind, norm, false, true, "idctn"); }
    return sct_axes(a, axes, kind, norm, false, true, "idctn");
}

// dstn gives the sine transform along the axes.
fn dstn(a, axes: i64[:] = [1000], kind: i64 = 2, norm: const u8[:] = "backward") -> Arr, Fault {
    if axes.len == 1 && axes[0] == 1000 { return sct_axes(a, all_axes(a.ndim), kind, norm, true, false, "dstn"); }
    return sct_axes(a, axes, kind, norm, true, false, "dstn");
}

// idstn gives the inverse sine transform along the axes.
fn idstn(a, axes: i64[:] = [1000], kind: i64 = 2, norm: const u8[:] = "backward") -> Arr, Fault {
    if axes.len == 1 && axes[0] == 1000 { return sct_axes(a, all_axes(a.ndim), kind, norm, true, true, "idstn"); }
    return sct_axes(a, axes, kind, norm, true, true, "idstn");
}

// hfftn gives the real array of a signal that holds the positive half
// of a Hermitian spectrum along the last axis, as scipy.fft.hfftn does.
// It is the inverse real transform of the conjugate, scaled by the
// count of places of the result. The length of the last axis of the
// result is n_last, or 2 (m - 1) when n_last is zero.
fn hfftn(re, im, n_last: i64 = 0) -> Arr, Fault {
    if re.ndim != im.ndim || re.data.len != im.data.len {
        return empty_arr(), fault(F_SHAPE, "hfftn", "the parts hold different shapes");
    }
    let m = re.shape[re.ndim - 1];
    var len = n_last;
    if len == 0 { len = 2 * (m - 1); }
    if len < 1 {
        return empty_arr(), fault(F_VALUE, "hfftn", "the length holds a value below one");
    }
    // The last axis keeps len / 2 + 1 places: SciPy crops or pads it.
    let need = len / 2 + 1;
    let rows = re.data.len / m;
    var cr: f64[>..] = [];
    var ci: f64[>..] = [];
    var r: i64 = 0;
    while r < rows {
        var c: i64 = 0;
        while c < need {
            if c < m {
                cr.push(re.data[r * m + c]);
                ci.push(-im.data[r * m + c]);
            } else {
                cr.push(0.0);
                ci.push(0.0);
            }
            c += 1;
        }
        r += 1;
    }
    var s: i64[..8] = [];
    var i: i64 = 0;
    while i < re.ndim { s.push(re.shape[i]); i += 1; }
    s[re.ndim - 1] = need;
    let xr = Arr { ndim: re.ndim, shape: s, data: copy(cr) };
    let xi = Arr { ndim: re.ndim, shape: s, data: copy(ci) };
    let out, f = irfftn(xr, xi, len);
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, "hfftn", f.detail); }
    var d: f64[>..] = [];
    i = 0;
    while i < out.data.len { d.push(out.data[i] * out.data.len); i += 1; }
    return Arr { ndim: out.ndim, shape: out.shape, data: copy(d) }, no_fault();
}

// hfft2 is hfftn of a matrix.
fn hfft2(re, im, n_last: i64 = 0) -> Arr, Fault {
    if re.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "hfft2", "the operand is not a matrix");
    }
    return hfftn(re, im, n_last);
}

// ihfftn gives the positive half, along the last axis, of the
// Hermitian spectrum of a real array. It is the conjugate of the real
// transform, divided by the count of places.
fn ihfftn(a) -> Arr, Arr, Fault {
    let r, i, f = rfftn(a);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), fault(f.kind, "ihfftn", f.detail); }
    var dr: f64[>..] = [];
    var di: f64[>..] = [];
    var k: i64 = 0;
    while k < r.data.len {
        dr.push(r.data[k] / a.data.len);
        di.push(-i.data[k] / a.data.len);
        k += 1;
    }
    return Arr { ndim: r.ndim, shape: r.shape, data: copy(dr) }, Arr { ndim: i.ndim, shape: i.shape, data: copy(di) }, no_fault();
}

// ihfft2 is ihfftn of a matrix.
fn ihfft2(a) -> Arr, Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "ihfft2", "the operand is not a matrix");
    }
    return ihfftn(a);
}
