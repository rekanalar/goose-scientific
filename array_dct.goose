// The cosine and sine transforms of the types 1 to 4 and their axes.
//
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

// sct_axis_len runs the cosine or sine transform along one axis of an
// array and holds the length of the caller. A length of zero keeps
// the size of the axis. The axis is cropped or padded to the length.
fn sct_axis_len(a, axis: i64, n: i64, kind: i64, norm: const u8[:], sine: bool, inverse: bool, tag: const u8[:]) -> Arr, Fault {
    let c = dct_kind_check(kind, tag);
    if c.kind != F_NONE { return empty_arr(), c; }
    if norm_code(norm) < 0 { return empty_arr(), fault(F_VALUE, tag, "the norm holds an unknown name"); }
    let k = fix_axis(axis, a.ndim);
    if k < 0 { return empty_arr(), fault(F_RANGE, tag, "an axis stands outside the shape"); }
    var len = n;
    if len == 0 { len = a.shape[k]; }
    if len < 1 { return empty_arr(), fault(F_VALUE, tag, "the length holds a value below one"); }
    var oshape: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim { oshape.push(a.shape[i]); i += 1; }
    oshape[k] = len;
    var buf: f64[>..] = [];
    hx_copy(a.data, a.shape, oshape, buf);
    let inner = product_after_of(oshape, k);
    let outer = product_before_of(oshape, k);
    var o: i64 = 0;
    while o < outer {
        var q: i64 = 0;
        while q < inner {
            var line: f64[>..] = [];
            var j: i64 = 0;
            while j < len { line.push(buf[(o * len + j) * inner + q]); j += 1; }
            let t, f = sct(line, kind, norm, sine, inverse, tag);
            if f.kind != F_NONE { return empty_arr(), f; }
            j = 0;
            while j < len { buf[(o * len + j) * inner + q] = t.data[j]; j += 1; }
            q += 1;
        }
        o += 1;
    }
    return Arr { ndim: a.ndim, shape: oshape, data: buf }, no_fault();
}

// dct_axis is dct along one axis of an array. The length n crops or
// pads the axis. A length of zero keeps the size of the axis.
fn dct_axis(a, kind: i64 = 2, n: i64 = 0, axis: i64 = -1, norm: const u8[:] = "backward") -> Arr, Fault {
    return sct_axis_len(a, axis, n, kind, norm, false, false, "dct_axis");
}

// idct_axis is idct along one axis of an array.
fn idct_axis(a, kind: i64 = 2, n: i64 = 0, axis: i64 = -1, norm: const u8[:] = "backward") -> Arr, Fault {
    return sct_axis_len(a, axis, n, kind, norm, false, true, "idct_axis");
}

// dst_axis is dst along one axis of an array.
fn dst_axis(a, kind: i64 = 2, n: i64 = 0, axis: i64 = -1, norm: const u8[:] = "backward") -> Arr, Fault {
    return sct_axis_len(a, axis, n, kind, norm, true, false, "dst_axis");
}

// idst_axis is idst along one axis of an array.
fn idst_axis(a, kind: i64 = 2, n: i64 = 0, axis: i64 = -1, norm: const u8[:] = "backward") -> Arr, Fault {
    return sct_axis_len(a, axis, n, kind, norm, true, true, "idst_axis");
}

