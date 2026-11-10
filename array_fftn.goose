// The transforms of more than one dimension, the shifts of every axis
// and the frequencies of the real transform.
//
// A transform of many axes runs the transform of one dimension along
// each axis in turn, as NumPy does. The input holds two arrays of one
// shape: the real parts and the imaginary parts.
namespace scientific;

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
