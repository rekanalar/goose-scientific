// The Fourier transforms of real and Hermitian signals along one axis.
//
namespace scientific;

import mathshim;

// hfft_axis is hfft along one axis of an array. The input holds the
// positive half of a Hermitian spectrum along the axis. The length n
// of the result is two times the length of the axis minus two when n
// is zero. The axis is cropped or padded to n / 2 + 1 places. The
// norms backward, ortho and forward follow SciPy.
fn hfft_axis(re, im, n: i64 = 0, axis: i64 = -1, norm: const u8[:] = "backward") -> Arr, Fault {
    if re.ndim != im.ndim || re.data.len != im.data.len {
        return empty_arr(), fault(F_SHAPE, "hfft_axis", "the parts hold different shapes");
    }
    let nc = norm_code(norm);
    if nc < 0 { return empty_arr(), fault(F_VALUE, "hfft_axis", "the norm holds an unknown name"); }
    let k = fix_axis(axis, re.ndim);
    if k < 0 { return empty_arr(), fault(F_RANGE, "hfft_axis", "an axis stands outside the shape"); }
    let w = re.shape[k];
    var len = n;
    if len == 0 { len = 2 * (w - 1); }
    if len < 1 { return empty_arr(), fault(F_VALUE, "hfft_axis", "the length holds a value below one"); }
    let need = len / 2 + 1;
    var ishape: i64[..8] = [];
    var oshape: i64[..8] = [];
    var i: i64 = 0;
    while i < re.ndim { ishape.push(re.shape[i]); oshape.push(re.shape[i]); i += 1; }
    ishape[k] = need;
    oshape[k] = len;
    var cr: f64[>..] = [];
    var ci: f64[>..] = [];
    hx_copy(re.data, re.shape, ishape, cr);
    hx_copy(im.data, im.shape, ishape, ci);
    let inner = product_after_of(oshape, k);
    let outer = product_before_of(oshape, k);
    var scale = 1.0;
    if nc == 0 { scale = len as f64; }
    if nc == 1 { scale = mathshim::root(len as f64); }
    var d: f64[>..] = [];
    var total: i64 = 1;
    i = 0;
    while i < re.ndim { total *= oshape[i]; i += 1; }
    i = 0;
    while i < total { d.push(0.0); i += 1; }
    var o: i64 = 0;
    while o < outer {
        var q: i64 = 0;
        while q < inner {
            var lr: f64[>..] = [];
            var li: f64[>..] = [];
            var j: i64 = 0;
            while j < need {
                lr.push(cr[(o * need + j) * inner + q]);
                li.push(-ci[(o * need + j) * inner + q]);
                j += 1;
            }
            let t, f = irfft(lr, li, len);
            if f.kind != F_NONE { return empty_arr(), f; }
            j = 0;
            while j < len {
                d[(o * len + j) * inner + q] = t.data[j] * scale;
                j += 1;
            }
            q += 1;
        }
        o += 1;
    }
    return Arr { ndim: re.ndim, shape: oshape, data: d }, no_fault();
}

// ihfft_axis is ihfft along one axis of an array. The length n of the
// transform is the length of the axis when n is zero. The axis is
// cropped or padded to n places, and the result holds n / 2 + 1
// places. The norms backward, ortho and forward follow SciPy.
fn ihfft_axis(a, n: i64 = 0, axis: i64 = -1, norm: const u8[:] = "backward") -> Arr, Arr, Fault {
    let nc = norm_code(norm);
    if nc < 0 { return empty_arr(), empty_arr(), fault(F_VALUE, "ihfft_axis", "the norm holds an unknown name"); }
    let k = fix_axis(axis, a.ndim);
    if k < 0 { return empty_arr(), empty_arr(), fault(F_RANGE, "ihfft_axis", "an axis stands outside the shape"); }
    var len = n;
    if len == 0 { len = a.shape[k]; }
    if len < 1 { return empty_arr(), empty_arr(), fault(F_VALUE, "ihfft_axis", "the length holds a value below one"); }
    let count = len / 2 + 1;
    var ishape: i64[..8] = [];
    var oshape: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim { ishape.push(a.shape[i]); oshape.push(a.shape[i]); i += 1; }
    ishape[k] = len;
    oshape[k] = count;
    var buf: f64[>..] = [];
    hx_copy(a.data, a.shape, ishape, buf);
    let inner = product_after_of(oshape, k);
    let outer = product_before_of(oshape, k);
    var scale = 1.0 / (len as f64);
    if nc == 1 { scale = 1.0 / mathshim::root(len as f64); }
    if nc == 2 { scale = 1.0; }
    var dr: f64[>..] = [];
    var di: f64[>..] = [];
    var total: i64 = 1;
    i = 0;
    while i < a.ndim { total *= oshape[i]; i += 1; }
    i = 0;
    while i < total { dr.push(0.0); di.push(0.0); i += 1; }
    var o: i64 = 0;
    while o < outer {
        var q: i64 = 0;
        while q < inner {
            var line: f64[>..] = [];
            var j: i64 = 0;
            while j < len { line.push(buf[(o * len + j) * inner + q]); j += 1; }
            let fr, fi, f = rfft(line);
            if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
            j = 0;
            while j < count {
                let at = (o * count + j) * inner + q;
                dr[at] = fr.data[j] * scale;
                di[at] = -fi.data[j] * scale;
                j += 1;
            }
            q += 1;
        }
        o += 1;
    }
    return Arr { ndim: a.ndim, shape: oshape, data: dr }, Arr { ndim: a.ndim, shape: oshape, data: di }, no_fault();
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

