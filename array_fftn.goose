// The transforms of more than one dimension, the shifts of every axis
// and the frequencies of the real transform.
//
// A transform of many axes runs the transform of one dimension along
// each axis in turn, as NumPy does. The input holds two arrays of one
// shape: the real parts and the imaginary parts.
namespace scientific;

import mathshim;

// transform_axes runs the transform along every axis of the arrays.
// The flag picks the inverse, which divides by the length of each
// axis.
fn transform_axes(re, im, inverse: bool, who: const u8[:]) -> Arr, Arr, Fault {
    if re.ndim != im.ndim {
        return empty_arr(), empty_arr(), fault(F_SHAPE, who, "the parts hold a different count of dimensions");
    }
    var i: i64 = 0;
    while i < re.ndim {
        if re.shape[i] != im.shape[i] {
            return empty_arr(), empty_arr(), fault(F_SHAPE, who, "the parts hold different shapes");
        }
        i += 1;
    }
    var dr: f64[>..] = [];
    var di: f64[>..] = [];
    i = 0;
    while i < re.data.len { dr.push(re.data[i]); di.push(im.data[i]); i += 1; }
    var axis: i64 = 0;
    while axis < re.ndim {
        let width = re.shape[axis];
        let inner = product_after_of(re.shape, axis);
        let outer = product_before_of(re.shape, axis);
        var o: i64 = 0;
        while o < outer {
            var q: i64 = 0;
            while q < inner {
                var line_re: f64[>..] = [];
                var line_im: f64[>..] = [];
                var k: i64 = 0;
                while k < width {
                    let at = (o * width + k) * inner + q;
                    line_re.push(dr[at]);
                    line_im.push(di[at]);
                    k += 1;
                }
                var tr: Arr = empty_arr();
                var ti: Arr = empty_arr();
                var f: Fault = no_fault();
                if inverse { tr, ti, f = ifft(line_re, line_im); }
                else { tr, ti, f = fft_parts(line_re, line_im); }
                if f.kind != F_NONE { return empty_arr(), empty_arr(), fault(f.kind, who, f.detail); }
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
        axis += 1;
    }
    return Arr { ndim: re.ndim, shape: re.shape, data: dr }, Arr { ndim: re.ndim, shape: re.shape, data: di }, no_fault();
}

// fftn gives the transform of an array along every axis.
fn fftn(re, im) -> Arr, Arr, Fault {
    return transform_axes(re, im, false, "fftn");
}

// ifftn gives the inverse transform along every axis. The result
// holds the scale of the lengths of all axes.
fn ifftn(re, im) -> Arr, Arr, Fault {
    return transform_axes(re, im, true, "ifftn");
}

// hx_block runs one block of a transform of many axes. The block is
// cropped or padded to the sizes in o, transformed, and scaled.
fn hx_block(re, im, o: i64[:], inverse: bool, scale: f64, who: const u8[:], outr: f64[>..]&, outi: f64[>..]&) -> Fault {
    var cr: f64[>..] = [];
    var ci: f64[>..] = [];
    hx_copy(re.data, re.shape, o, cr);
    hx_copy(im.data, im.shape, o, ci);
    var sh: i64[..8] = [];
    var i: i64 = 0;
    while i < o.len { sh.push(o[i]); i += 1; }
    let xr = Arr { ndim: o.len, shape: sh, data: copy(cr) };
    let xi = Arr { ndim: o.len, shape: sh, data: copy(ci) };
    let tr, ti, f = transform_axes(xr, xi, inverse, who);
    if f.kind != F_NONE { return f; }
    i = 0;
    while i < tr.data.len {
        outr.push(tr.data[i] * scale);
        outi.push(ti.data[i] * scale);
        i += 1;
    }
    return no_fault();
}

// hx_run resolves the arguments of a transform of many axes, moves the
// axes to the end, and runs one block per batch. The flag picks the
// inverse. The caller gives the scale of the norm.
fn hx_run(re, im, s: i64[:], axes: i64[:], norm: const u8[:], inverse: bool, who: const u8[:]) -> Arr, Arr, Fault {
    if re.ndim != im.ndim || re.data.len != im.data.len {
        return empty_arr(), empty_arr(), fault(F_SHAPE, who, "the parts hold different shapes");
    }
    let nc = norm_code(norm);
    if nc < 0 { return empty_arr(), empty_arr(), fault(F_VALUE, who, "the norm holds an unknown name"); }
    let ax, f = hx_axes(re.ndim, s.len, axes, who);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let k = ax.len;
    var dst: i64[>..] = [];
    var i: i64 = re.ndim - k;
    while i < re.ndim { dst.push(i); i += 1; }
    let mr, f1 = moveaxis_many(re, ax, dst);
    if f1.kind != F_NONE { return empty_arr(), empty_arr(), fault(f1.kind, who, f1.detail); }
    let mi, f2 = moveaxis_many(im, ax, dst);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), fault(f2.kind, who, f2.detail); }
    var ts: i64[>..] = [];
    var batch: i64 = 1;
    i = 0;
    while i < mr.ndim {
        if i >= mr.ndim - k { ts.push(mr.shape[i]); } else { batch *= mr.shape[i]; }
        i += 1;
    }
    let o, f3 = hx_sizes(ts, s, false, who);
    if f3.kind != F_NONE { return empty_arr(), empty_arr(), f3; }
    var total: i64 = 1;
    i = 0;
    while i < k { total *= o[i]; i += 1; }
    var scale = 1.0;
    if inverse {
        if nc == 1 { scale = mathshim::root(total as f64); }
        if nc == 2 { scale = total as f64; }
    } else {
        if nc == 1 { scale = 1.0 / mathshim::root(total as f64); }
        if nc == 2 { scale = 1.0 / (total as f64); }
    }
    let blk = mr.data.len / batch;
    var tsh: i64[..8] = [];
    i = 0;
    while i < k { tsh.push(ts[i]); i += 1; }
    var sr: i64[..8] = [];
    i = 0;
    while i < mr.ndim - k { sr.push(mr.shape[i]); i += 1; }
    i = 0;
    while i < k { sr.push(o[i]); i += 1; }
    var outr: f64[>..] = [];
    var outi: f64[>..] = [];
    var b: i64 = 0;
    while b < batch {
        var br: f64[>..] = [];
        var bi: f64[>..] = [];
        var q: i64 = 0;
        while q < blk { br.push(mr.data[b * blk + q]); bi.push(mi.data[b * blk + q]); q += 1; }
        let xr = Arr { ndim: k, shape: tsh, data: copy(br) };
        let xi = Arr { ndim: k, shape: tsh, data: copy(bi) };
        let fb = hx_block(xr, xi, o, inverse, scale, who, outr, outi);
        if fb.kind != F_NONE { return empty_arr(), empty_arr(), fault(fb.kind, who, fb.detail); }
        b += 1;
    }
    let rr = Arr { ndim: mr.ndim, shape: sr, data: copy(outr) };
    let ri = Arr { ndim: mr.ndim, shape: sr, data: copy(outi) };
    let br2, g1 = moveaxis_many(rr, dst, ax);
    if g1.kind != F_NONE { return empty_arr(), empty_arr(), fault(g1.kind, who, g1.detail); }
    let bi2, g2 = moveaxis_many(ri, dst, ax);
    if g2.kind != F_NONE { return empty_arr(), empty_arr(), fault(g2.kind, who, g2.detail); }
    return br2, bi2, no_fault();
}

// fftn_ex is fftn with the arguments s, axes and norm of SciPy. A size
// of -1 in s keeps the size of the input. The list [1000] for axes
// means every axis, or the last len(s) axes. The norm backward leaves
// the result as it is. The norm ortho divides by the root of the count
// of transformed places. The norm forward divides by the count.
fn fftn_ex(re, im, s: i64[:] = [], axes: i64[:] = [1000], norm: const u8[:] = "backward") -> Arr, Arr, Fault {
    return hx_run(re, im, s, axes, norm, false, "fftn_ex");
}

// ifftn_ex is ifftn with the arguments s, axes and norm of SciPy. The
// norm backward divides by the count of transformed places, which the
// transform itself does. The norm ortho divides by its root. The norm
// forward does not scale.
fn ifftn_ex(re, im, s: i64[:] = [], axes: i64[:] = [1000], norm: const u8[:] = "backward") -> Arr, Arr, Fault {
    return hx_run(re, im, s, axes, norm, true, "ifftn_ex");
}

// fft2c gives the transform of a matrix that holds two parts.
fn fft2c(re, im) -> Arr, Arr, Fault {
    if re.ndim != 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "fft2c", "the operand holds a count of dimensions that is not two");
    }
    return transform_axes(re, im, false, "fft2c");
}

// ifft2 gives the inverse transform of a matrix that holds two parts.
fn ifft2(re, im) -> Arr, Arr, Fault {
    if re.ndim != 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "ifft2", "the operand holds a count of dimensions that is not two");
    }
    return transform_axes(re, im, true, "ifft2");
}

// shift_axes rolls every axis. The forward shift rolls by the half of
// the length, so the element at the index p comes from p plus the
// rest of the length. The inverse rolls back, so it reads p plus the
// half. The two differ when the length is odd.
fn shift_axes(a, inverse: bool) -> Arr, Fault {
    var d: f64[>..] = [];
    var pos: i64[..8] = [];
    var strides: i64[>..] = [];
    strides_of(a.shape, strides);
    var i: i64 = 0;
    while i < a.ndim { pos.push(0); i += 1; }
    var n: i64 = 0;
    while n < a.data.len {
        var src: i64 = 0;
        i = 0;
        while i < a.ndim {
            let w = a.shape[i];
            var shift = w - w / 2;
            if inverse { shift = w / 2; }
            var k = pos[i] + shift;
            if k >= w { k -= w; }
            src += k * strides[i];
            i += 1;
        }
        d.push(a.data[src]);
        i = a.ndim - 1;
        while i >= 0 {
            pos[i] += 1;
            if pos[i] < a.shape[i] { break; }
            pos[i] = 0;
            i -= 1;
        }
        n += 1;
    }
    return Arr { ndim: a.ndim, shape: a.shape, data: d }, no_fault();
}

// ifftshift undoes fftshift on every axis. The two agree when the
// length is even.
fn ifftshift(a) -> Arr, Fault {
    return shift_axes(a, true);
}

// rfftfreq gives the frequencies of the real transform of a given
// length: the non-negative ones, from zero to half the rate.
fn rfftfreq(n: i64, d: f64) -> Arr, Fault {
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "rfftfreq", "the length holds a value below one");
    }
    if d == 0.0 {
        return empty_arr(), fault(F_VALUE, "rfftfreq", "the step holds zero");
    }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i <= n / 2 { out.push(i / (n * d)); i += 1; }
    var s: i64[..8] = [];
    s.push(out.len);
    return Arr { ndim: 1, shape: s, data: out }, no_fault();
}
