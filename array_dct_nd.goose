// The transforms over several axes: dctn, dstn, hfftn and their inverses.
//
namespace scientific;

import mathshim;

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

// hx_axes resolves the axes of a Hermitian transform. The list [1000]
// means every axis, or the last len(s) axes when s holds sizes.
fn hx_axes(nd: i64, s_len: i64, axes: i64[:], tag: const u8[:]) -> i64[>..], Fault {
    var ax: i64[>..] = [];
    if axes.len == 1 && axes[0] == 1000 {
        var k: i64 = nd;
        if s_len > 0 { k = s_len; }
        if k > nd { return ax, fault(F_SHAPE, tag, "s holds more sizes than the array holds axes"); }
        var i: i64 = nd - k;
        while i < nd { ax.push(i); i += 1; }
    } else {
        var i: i64 = 0;
        while i < axes.len {
            let k = fix_axis(axes[i], nd);
            if k < 0 { return ax, fault(F_RANGE, tag, "an axis stands outside the shape"); }
            var j: i64 = 0;
            while j < ax.len {
                if ax[j] == k { return ax, fault(F_VALUE, tag, "an axis repeats"); }
                j += 1;
            }
            ax.push(k);
            i += 1;
        }
    }
    if ax.len == 0 { return ax, fault(F_VALUE, tag, "no axis to transform"); }
    if s_len > 0 && s_len != ax.len { return ax, fault(F_SHAPE, tag, "s and the axes differ in count"); }
    return ax, no_fault();
}

// hx_copy copies a block of one shape into a block of another shape.
// It crops what is too long and pads what is too short with zero.
fn hx_copy(src: f64[:], sshape: i64[:], dshape: i64[:], out: f64[>..]&) {
    let k = dshape.len;
    var n: i64 = 1;
    var i: i64 = 0;
    while i < k { n *= dshape[i]; i += 1; }
    var p: i64 = 0;
    while p < n {
        var rest = p;
        var off: i64 = 0;
        var stride: i64 = 1;
        var inside = true;
        i = k - 1;
        while i >= 0 {
            let idx = rest % dshape[i];
            rest = rest / dshape[i];
            if idx >= sshape[i] { inside = false; }
            off += idx * stride;
            stride *= sshape[i];
            i -= 1;
        }
        if inside { out.push(src[off]); } else { out.push(0.0); }
        p += 1;
    }
}

// hx_sizes gives the output sizes of the transformed axes. A size of -1
// keeps the size of the input. Without s the last axis of hfftn holds
// two times its length minus two.
fn hx_sizes(tshape: i64[:], s: i64[:], half_in: bool, tag: const u8[:]) -> i64[>..], Fault {
    var o: i64[>..] = [];
    let k = tshape.len;
    var i: i64 = 0;
    while i < k {
        var v = tshape[i];
        if s.len > 0 {
            if s[i] < -1 || s[i] == 0 { return o, fault(F_VALUE, tag, "s holds a size below one"); }
            if s[i] > 0 { v = s[i]; }
        } else if i == k - 1 && half_in {
            v = 2 * (tshape[i] - 1);
        }
        if v < 1 { return o, fault(F_VALUE, tag, "the length of the last axis is below one"); }
        o.push(v);
        i += 1;
    }
    return o, no_fault();
}

// hfftn_ex is hfftn with the arguments s, axes and norm of SciPy.
// s holds the output size of each axis in axes, and -1 keeps the size.
// The list [1000] for axes means every axis, or the last len(s) axes.
// The norm backward leaves the result as it is. The norm forward
// divides by the count of output places. The norm ortho divides by its root.
fn hfftn_ex(re, im, s: i64[:] = [], axes: i64[:] = [1000], norm: const u8[:] = "backward") -> Arr, Fault {
    if re.ndim != im.ndim || re.data.len != im.data.len {
        return empty_arr(), fault(F_SHAPE, "hfftn_ex", "the parts hold different shapes");
    }
    let nc = norm_code(norm);
    if nc < 0 { return empty_arr(), fault(F_VALUE, "hfftn_ex", "the norm holds an unknown name"); }
    let ax, f = hx_axes(re.ndim, s.len, axes, "hfftn_ex");
    if f.kind != F_NONE { return empty_arr(), f; }
    let k = ax.len;
    var dst: i64[>..] = [];
    var i: i64 = re.ndim - k;
    while i < re.ndim { dst.push(i); i += 1; }
    let mr, f1 = moveaxis_many(re, ax, dst);
    if f1.kind != F_NONE { return empty_arr(), fault(f1.kind, "hfftn_ex", f1.detail); }
    let mi, f2 = moveaxis_many(im, ax, dst);
    if f2.kind != F_NONE { return empty_arr(), fault(f2.kind, "hfftn_ex", f2.detail); }
    var ts: i64[>..] = [];
    var batch: i64 = 1;
    i = 0;
    while i < mr.ndim {
        if i >= mr.ndim - k { ts.push(mr.shape[i]); } else { batch *= mr.shape[i]; }
        i += 1;
    }
    let o, f3 = hx_sizes(ts, s, true, "hfftn_ex");
    if f3.kind != F_NONE { return empty_arr(), f3; }
    var total: i64 = 1;
    var cap: i64 = 1;
    i = 0;
    while i < k { total *= o[i]; i += 1; }
    // The leading axes are cropped or padded; the last axis keeps its size.
    var mid: i64[>..] = [];
    i = 0;
    while i < k { if i < k - 1 { mid.push(o[i]); } else { mid.push(ts[i]); } i += 1; }
    i = 0;
    while i < k { cap *= mid[i]; i += 1; }
    let blk = mr.data.len / batch;
    var scale = 1.0;
    if nc == 1 { scale = 1.0 / mathshim::root(total as f64); }
    if nc == 2 { scale = 1.0 / (total as f64); }
    var d: f64[>..] = [];
    var b: i64 = 0;
    while b < batch {
        var cr: f64[>..] = [];
        var ci: f64[>..] = [];
        hx_copy(mr.data[b * blk..(b + 1) * blk], ts, mid, cr);
        hx_copy(mi.data[b * blk..(b + 1) * blk], ts, mid, ci);
        var sh: i64[..8] = [];
        i = 0;
        while i < k { sh.push(mid[i]); i += 1; }
        let xr = Arr { ndim: k, shape: sh, data: copy(cr) };
        let xi = Arr { ndim: k, shape: sh, data: copy(ci) };
        let out, f4 = hfftn(xr, xi, o[k - 1]);
        if f4.kind != F_NONE { return empty_arr(), fault(f4.kind, "hfftn_ex", f4.detail); }
        i = 0;
        while i < out.data.len { d.push(out.data[i] * scale); i += 1; }
        b += 1;
    }
    var rs: i64[..8] = [];
    i = 0;
    while i < mr.ndim - k { rs.push(mr.shape[i]); i += 1; }
    i = 0;
    while i < k { rs.push(o[i]); i += 1; }
    let res = Arr { ndim: mr.ndim, shape: rs, data: copy(d) };
    let back, f5 = moveaxis_many(res, dst, ax);
    if f5.kind != F_NONE { return empty_arr(), fault(f5.kind, "hfftn_ex", f5.detail); }
    return back, no_fault();
}

// ihfftn_ex is ihfftn with the arguments s, axes and norm of SciPy.
// The norm backward divides by the count of input places. The norm
// forward does not scale. The norm ortho divides by its root.
fn ihfftn_ex(a, s: i64[:] = [], axes: i64[:] = [1000], norm: const u8[:] = "backward") -> Arr, Arr, Fault {
    let nc = norm_code(norm);
    if nc < 0 { return empty_arr(), empty_arr(), fault(F_VALUE, "ihfftn_ex", "the norm holds an unknown name"); }
    let ax, f = hx_axes(a.ndim, s.len, axes, "ihfftn_ex");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let k = ax.len;
    var dst: i64[>..] = [];
    var i: i64 = a.ndim - k;
    while i < a.ndim { dst.push(i); i += 1; }
    let m, f1 = moveaxis_many(a, ax, dst);
    if f1.kind != F_NONE { return empty_arr(), empty_arr(), fault(f1.kind, "ihfftn_ex", f1.detail); }
    var ts: i64[>..] = [];
    var batch: i64 = 1;
    i = 0;
    while i < m.ndim {
        if i >= m.ndim - k { ts.push(m.shape[i]); } else { batch *= m.shape[i]; }
        i += 1;
    }
    let o, f3 = hx_sizes(ts, s, false, "ihfftn_ex");
    if f3.kind != F_NONE { return empty_arr(), empty_arr(), f3; }
    var total: i64 = 1;
    i = 0;
    while i < k { total *= o[i]; i += 1; }
    var scale = 1.0;
    if nc == 1 { scale = mathshim::root(total as f64); }
    if nc == 2 { scale = total as f64; }
    let blk = m.data.len / batch;
    var dr: f64[>..] = [];
    var di: f64[>..] = [];
    var rshape: i64[..8] = [];
    var b: i64 = 0;
    while b < batch {
        var c: f64[>..] = [];
        hx_copy(m.data[b * blk..(b + 1) * blk], ts, o, c);
        var sh: i64[..8] = [];
        i = 0;
        while i < k { sh.push(o[i]); i += 1; }
        let x = Arr { ndim: k, shape: sh, data: copy(c) };
        let r, im, f4 = ihfftn(x);
        if f4.kind != F_NONE { return empty_arr(), empty_arr(), fault(f4.kind, "ihfftn_ex", f4.detail); }
        i = 0;
        while i < r.data.len { dr.push(r.data[i] * scale); di.push(im.data[i] * scale); i += 1; }
        rshape = r.shape;
        b += 1;
    }
    var rs: i64[..8] = [];
    i = 0;
    while i < m.ndim - k { rs.push(m.shape[i]); i += 1; }
    i = 0;
    while i < k { rs.push(rshape[i]); i += 1; }
    let rr = Arr { ndim: m.ndim, shape: rs, data: copy(dr) };
    let ri = Arr { ndim: m.ndim, shape: rs, data: copy(di) };
    let br, g1 = moveaxis_many(rr, dst, ax);
    if g1.kind != F_NONE { return empty_arr(), empty_arr(), fault(g1.kind, "ihfftn_ex", g1.detail); }
    let bi, g2 = moveaxis_many(ri, dst, ax);
    if g2.kind != F_NONE { return empty_arr(), empty_arr(), fault(g2.kind, "ihfftn_ex", g2.detail); }
    return br, bi, no_fault();
}
