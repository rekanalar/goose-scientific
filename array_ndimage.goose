// The linear filters of arrays of N dimensions, as in scipy.ndimage:
// correlate1d, convolve1d, uniform_filter1d, gaussian_filter1d,
// correlate, convolve, uniform_filter and gaussian_filter.
//
// Every filter reads the values outside the array with one function of
// the index, so the modes live in one place. The mode "reflect" repeats
// the edge value. The mode "mirror" does not. The names "grid-mirror",
// "grid-constant" and "grid-wrap" equal "reflect", "constant" and "wrap".
//
// The calls "correlate" and "convolve" of this package take two signals
// of one dimension. The filters of N dimensions are named correlate_nd
// and convolve_nd.
namespace scientific;

import mathshim;

// The cost of a call is the count of multiplications. A larger cost
// gives a fault. The radius of a Gaussian stays below ND_RADIUS values.
const ND_COST: i64 = 1000000000;
const ND_RADIUS: f64 = 1000000.0;
const ND_ORDER: i64 = 64;

// nd_mode gives the number of a mode name, or -1.
fn nd_mode(name: const u8[:]) -> i64 {
    if name == "reflect" || name == "grid-mirror" { return 0; }
    if name == "constant" || name == "grid-constant" { return 1; }
    if name == "nearest" { return 2; }
    if name == "mirror" { return 3; }
    if name == "wrap" || name == "grid-wrap" { return 4; }
    return -1;
}

// nd_src gives the index in 0 to n - 1 that stands for the index i. The
// constant mode gives -1 outside the array.
fn nd_src(i: i64, n: i64, mode: i64) -> i64 {
    if i >= 0 && i < n { return i; }
    if mode == 1 { return -1; }
    if mode == 2 {
        if i < 0 { return 0; }
        return n - 1;
    }
    if mode == 4 {
        var w = i % n;
        if w < 0 { w += n; }
        return w;
    }
    if mode == 3 {
        if n == 1 { return 0; }
        let p = 2 * n - 2;
        var m = i % p;
        if m < 0 { m += p; }
        if m >= n { m = p - m; }
        return m;
    }
    let p = 2 * n;
    var m = i % p;
    if m < 0 { m += p; }
    if m >= n { m = p - 1 - m; }
    return m;
}

// nd_check checks the array and the mode. It gives the number of the mode.
fn nd_check(a, mode: const u8[:], tag: const u8[:]) -> i64, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if f.kind != F_NONE { return -1, f; }
    let m = nd_mode(mode);
    if m < 0 { return -1, fault(F_VALUE, tag, "the mode holds an unknown name"); }
    return m, no_fault();
}

// nd_axis gives the axis in 0 to ndim - 1, or -1 for an axis outside.
fn nd_axis(axis: i64, ndim: i64) -> i64 {
    if axis < -ndim || axis >= ndim { return -1; }
    if axis < 0 { return axis + ndim; }
    return axis;
}

fn nd_zeros(n: i64) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n { d.push(0.0); i += 1; }
    return d;
}

// nd_val gives the value of the line at an index, with the border rule.
fn nd_val(line: f64[:], k: i64, mode: i64, cval: f64) -> f64 {
    let j = nd_src(k, line.len, mode);
    if j < 0 { return cval; }
    return line[j];
}

// nd_taps gives the weighted sum of the line. The output at i is the
// sum over j of w[j] times the line at i + j - shift.
fn nd_taps(line: f64[:], w: f64[:], shift: i64, mode: i64, cval: f64, res: f64[>..]&) {
    let n = line.len;
    var i: i64 = 0;
    while i < n {
        var s: f64 = 0.0;
        let first = i - shift;
        var j: i64 = 0;
        if first >= 0 && first + w.len <= n {
            while j < w.len { s += w[j] * line[first + j]; j += 1; }
        } else {
            while j < w.len { s += w[j] * nd_val(line, first + j, mode, cval); j += 1; }
        }
        res[i] = s;
        i += 1;
    }
}

// nd_box gives the mean of a window of size values. The window of the
// output at i starts at i - shift. A running sum moves the window. A
// line with a value that is not finite takes the direct sum, so a NaN
// spoils only the windows that hold it.
fn nd_box(line: f64[:], size: i64, shift: i64, mode: i64, cval: f64, res: f64[>..]&) {
    let n = line.len;
    var finite = cval - cval == 0.0;
    var i: i64 = 0;
    while i < n {
        if !(line[i] - line[i] == 0.0) { finite = false; }
        i += 1;
    }
    let scale = 1.0 / (size as f64);
    var sum: f64 = 0.0;
    i = 0;
    while i < n {
        let first = i - shift;
        if !finite || i == 0 {
            sum = 0.0;
            var j: i64 = 0;
            while j < size { sum += nd_val(line, first + j, mode, cval); j += 1; }
        } else {
            sum += nd_val(line, first + size - 1, mode, cval) - nd_val(line, first - 1, mode, cval);
        }
        res[i] = sum * scale;
        i += 1;
    }
}

// nd_less orders two values. A NaN is the largest value.
fn nd_less(a: f64, b: f64) -> bool {
    return a < b || (a == a && !(b == b));
}

// nd_extreme gives the minimum or the maximum of a window of size values.
// The window of the output at i starts at i - shift.
fn nd_extreme(line: f64[:], size: i64, shift: i64, mode: i64, cval: f64, high: bool, res: f64[>..]&) {
    let n = line.len;
    var i: i64 = 0;
    while i < n {
        let first = i - shift;
        var m = nd_val(line, first, mode, cval);
        var j: i64 = 1;
        while j < size {
            let v = nd_val(line, first + j, mode, cval);
            if high { if nd_less(m, v) { m = v; } } else { if nd_less(v, m) { m = v; } }
            j += 1;
        }
        res[i] = m;
        i += 1;
    }
}

// nd_along filters every line of src along one axis. The operation 1 gives
// the mean of a window, 2 its minimum and 3 its maximum. The operation 0
// gives the weighted sum with the weights w. The
// buffer out holds as many zeros as src holds values.
fn nd_along(src: f64[:], shape: i64[:], axis: i64, w: f64[:], op: i64, size: i64, shift: i64, mode: i64, cval: f64, out: f64[>..]&) {
    let n = shape[axis];
    var inner: i64 = 1;
    var k: i64 = axis + 1;
    while k < shape.len { inner *= shape[k]; k += 1; }
    let outer = src.len / (n * inner);
    var line: f64[>..] = nd_zeros(n);
    var res: f64[>..] = nd_zeros(n);
    var o: i64 = 0;
    while o < outer {
        var q: i64 = 0;
        while q < inner {
            let base = o * n * inner + q;
            var i: i64 = 0;
            while i < n { line[i] = src[base + i * inner]; i += 1; }
            if op == 1 {
                nd_box(line, size, shift, mode, cval, res);
            } else if op == 2 || op == 3 {
                nd_extreme(line, size, shift, mode, cval, op == 3, res);
            } else {
                nd_taps(line, w, shift, mode, cval, res);
            }
            i = 0;
            while i < n { out[base + i * inner] = res[i]; i += 1; }
            q += 1;
        }
        o += 1;
    }
}

// nd_origin_ok tells whether an origin is valid for a count of weights.
fn nd_origin_ok(origin: i64, len: i64) -> bool {
    return origin >= -(len / 2) && origin <= (len - 1) / 2;
}

// nd_weights1d filters along one axis with the weights. It reverses
// them for a convolution, as scipy.ndimage.convolve1d does.
fn nd_weights1d(a, w: f64[:], axis: i64, mode: const u8[:], cval: f64, origin: i64, convolve: bool, tag: const u8[:]) -> Arr, Fault {
    let m, f = nd_check(a, mode, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let ax = nd_axis(axis, a.ndim);
    if ax < 0 { return empty_arr(), fault(F_RANGE, tag, "the axis stands outside the array"); }
    if w.len < 1 { return empty_arr(), fault(F_VALUE, tag, "the weights hold no value"); }
    var o = origin;
    var wk: f64[>..] = [];
    var j: i64 = 0;
    while j < w.len {
        if convolve { wk.push(w[w.len - 1 - j]); } else { wk.push(w[j]); }
        j += 1;
    }
    if convolve {
        o = -origin;
        if w.len % 2 == 0 { o -= 1; }
    }
    if !nd_origin_ok(o, w.len) { return empty_arr(), fault(F_RANGE, tag, "the origin stands outside the weights"); }
    if a.data.len > ND_COST / w.len {
        return empty_arr(), fault(F_LIMIT, tag, "the cost of the filter is above the limit");
    }
    var out: f64[>..] = nd_zeros(a.data.len);
    nd_along(a.data, a.shape, ax, wk, 0, 0, w.len / 2 + o, m, cval, out);
    return arr_of(copy(out), a.shape), no_fault();
}

// correlate1d correlates the array with the weights along one axis.
fn correlate1d(a, weights: f64[:], axis: i64 = -1, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64 = 0) -> Arr, Fault {
    return nd_weights1d(a, weights, axis, mode, cval, origin, false, "correlate1d");
}

// convolve1d convolves the array with the weights along one axis.
fn convolve1d(a, weights: f64[:], axis: i64 = -1, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64 = 0) -> Arr, Fault {
    return nd_weights1d(a, weights, axis, mode, cval, origin, true, "convolve1d");
}

// nd_box_axis applies the mean of a window along one axis. It gives the
// result of the axis in out, or a fault.
fn nd_box_axis(src: f64[:], shape: i64[:], axis: i64, op: i64, size: i64, origin: i64, mode: i64, cval: f64, tag: const u8[:], out: f64[>..]&) -> Fault {
    if size < 1 { return fault(F_VALUE, tag, "the size holds a value below one"); }
    if origin < -(size / 2) || origin > size - 1 - size / 2 {
        return fault(F_RANGE, tag, "the origin stands outside the window");
    }
    let lines = src.len / shape[axis];
    if lines > ND_COST / size {
        return fault(F_LIMIT, tag, "the cost of the filter is above the limit");
    }
    nd_along(src, shape, axis, [], op, size, size / 2 + origin, mode, cval, out);
    return no_fault();
}

// uniform_filter1d gives the mean of a window of size values along one axis.
fn uniform_filter1d(a, size: i64, axis: i64 = -1, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64 = 0) -> Arr, Fault {
    let m, f = nd_check(a, mode, "uniform_filter1d");
    if f.kind != F_NONE { return empty_arr(), f; }
    let ax = nd_axis(axis, a.ndim);
    if ax < 0 { return empty_arr(), fault(F_RANGE, "uniform_filter1d", "the axis stands outside the array"); }
    var out: f64[>..] = nd_zeros(a.data.len);
    let g = nd_box_axis(a.data, a.shape, ax, 1, size, origin, m, cval, "uniform_filter1d", out);
    if g.kind != F_NONE { return empty_arr(), g; }
    return arr_of(copy(out), a.shape), no_fault();
}

// nd_gauss_kernel builds the weights of a Gaussian or of its derivative,
// in the order that a correlation needs: the first weight is for the
// value at the largest offset. A derivative of order k is q(x) phi(x),
// where q holds a polynomial that follows from the rule of the
// derivative of a product.
fn nd_gauss_kernel(sigma: f64, order: i64, radius: i64, out: f64[>..]&) {
    let s2 = sigma * sigma;
    var phi: f64[>..] = [];
    var sum: f64 = 0.0;
    var x: i64 = -radius;
    while x <= radius {
        let p = mathshim::power_e(-0.5 / s2 * ((x * x) as f64));
        phi.push(p);
        sum += p;
        x += 1;
    }
    var q: f64[>..] = nd_zeros(order + 1);
    q[0] = 1.0;
    var step: i64 = 0;
    while step < order {
        var nq: f64[>..] = nd_zeros(order + 1);
        var j: i64 = 0;
        while j <= order {
            var v: f64 = 0.0;
            if j + 1 <= order { v += ((j + 1) as f64) * q[j + 1]; }
            if j >= 1 { v -= q[j - 1] / s2; }
            nq[j] = v;
            j += 1;
        }
        q = copy(nq);
        step += 1;
    }
    var i: i64 = 0;
    while i <= 2 * radius {
        let off = radius - i;
        let xf = off as f64;
        var poly: f64 = 0.0;
        var j: i64 = order;
        while j >= 0 { poly = poly * xf + q[j]; j -= 1; }
        let wt = phi[radius + off] / sum;
        if order == 0 { out.push(wt); } else { out.push(poly * wt); }
        i += 1;
    }
}

// nd_gauss_axis filters src along one axis with a Gaussian. It gives the
// result in out, or a fault.
fn nd_gauss_axis(src: f64[:], shape: i64[:], axis: i64, sigma: f64, order: i64, truncate: f64, mode: i64, cval: f64, tag: const u8[:], out: f64[>..]&) -> Fault {
    if !(sigma - sigma == 0.0) || !(sigma > 0.0) {
        return fault(F_VALUE, tag, "the sigma holds a value that is not above zero");
    }
    // A sigma whose square underflows would give a NaN in every weight.
    let e = 0.5 / (sigma * sigma);
    if !(e - e == 0.0) {
        return fault(F_VALUE, tag, "the sigma holds a value that is too small to square");
    }
    if !(truncate - truncate == 0.0) || truncate < 0.0 {
        return fault(F_VALUE, tag, "the truncation holds a value that is not finite or is below zero");
    }
    if order < 0 { return fault(F_VALUE, tag, "the order holds a value below zero"); }
    if order > ND_ORDER { return fault(F_LIMIT, tag, "the order is above the limit"); }
    let rf = truncate * sigma + 0.5;
    if rf > ND_RADIUS { return fault(F_LIMIT, tag, "the radius of the kernel is above the limit"); }
    let radius = rf as i64;
    let len = 2 * radius + 1;
    if src.len > ND_COST / len { return fault(F_LIMIT, tag, "the cost of the filter is above the limit"); }
    var w: f64[>..] = [];
    nd_gauss_kernel(sigma, order, radius, w);
    nd_along(src, shape, axis, w, 0, 0, radius, mode, cval, out);
    return no_fault();
}

// gaussian_filter1d filters the array with a Gaussian, or with one of its
// derivatives, along one axis. The radius is truncate times sigma plus one half.
fn gaussian_filter1d(a, sigma: f64, axis: i64 = -1, order: i64 = 0, mode: const u8[:] = "reflect", cval: f64 = 0.0, truncate: f64 = 4.0) -> Arr, Fault {
    let m, f = nd_check(a, mode, "gaussian_filter1d");
    if f.kind != F_NONE { return empty_arr(), f; }
    let ax = nd_axis(axis, a.ndim);
    if ax < 0 { return empty_arr(), fault(F_RANGE, "gaussian_filter1d", "the axis stands outside the array"); }
    var out: f64[>..] = nd_zeros(a.data.len);
    let g = nd_gauss_axis(a.data, a.shape, ax, sigma, order, truncate, m, cval, "gaussian_filter1d", out);
    if g.kind != F_NONE { return empty_arr(), g; }
    return arr_of(copy(out), a.shape), no_fault();
}

// nd_direct correlates the data with an array of weights. The shift of
// each axis is the offset of the centre of the weights. The buffer out
// holds as many zeros as src holds values.
fn nd_direct(src: f64[:], shape: i64[:], w: f64[:], wshape: i64[:], shift: i64[:], mode: i64, cval: f64, out: f64[>..]&) {
    let nd = shape.len;
    var ix: i64[..8] = [];
    var ox: i64[..8] = [];
    var d: i64 = 0;
    while d < nd { ix.push(0); ox.push(0); d += 1; }
    var p: i64 = 0;
    while p < src.len {
        var s: f64 = 0.0;
        var q: i64 = 0;
        d = 0;
        while d < nd { ox[d] = 0; d += 1; }
        while q < w.len {
            if w[q] != 0.0 {
                var flat: i64 = 0;
                var outside = false;
                d = 0;
                while d < nd && !outside {
                    let m = nd_src(ix[d] + ox[d] - shift[d], shape[d], mode);
                    if m < 0 { outside = true; } else { flat = flat * shape[d] + m; }
                    d += 1;
                }
                if outside { s += w[q] * cval; } else { s += w[q] * src[flat]; }
            }
            d = nd - 1;
            while d >= 0 {
                ox[d] += 1;
                if ox[d] < wshape[d] { d = -1; } else { ox[d] = 0; d -= 1; }
            }
            q += 1;
        }
        out[p] = s;
        d = nd - 1;
        while d >= 0 {
            ix[d] += 1;
            if ix[d] < shape[d] { d = -1; } else { ix[d] = 0; d -= 1; }
        }
        p += 1;
    }
}

// nd_list reads a list of one value for each axis. An empty list gives
// the default. A list of one value holds for every axis.
fn nd_list(v: i64[:], ndim: i64, default: i64, out: i64[..8]&) -> bool {
    var d: i64 = 0;
    if v.len == 0 { while d < ndim { out.push(default); d += 1; } return true; }
    if v.len == 1 { while d < ndim { out.push(v[0]); d += 1; } return true; }
    if v.len != ndim { return false; }
    while d < ndim { out.push(v[d]); d += 1; }
    return true;
}

// nd_weightsn correlates or convolves with an array of weights.
fn nd_weightsn(a, weights, mode: const u8[:], cval: f64, origin: i64[:], convolve: bool, tag: const u8[:]) -> Arr, Fault {
    let m, f = nd_check(a, mode, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let fw = check_stack(weights.ndim, weights.shape, weights.data.len, 1, tag);
    if fw.kind != F_NONE { return empty_arr(), fw; }
    if weights.ndim != a.ndim {
        return empty_arr(), fault(F_SHAPE, tag, "the weights hold another count of dimensions than the array");
    }
    var og: i64[..8] = [];
    if !nd_list(origin, a.ndim, 0, og) {
        return empty_arr(), fault(F_SHAPE, tag, "the origin holds a count that is not one for each axis");
    }
    var shift: i64[..8] = [];
    var d: i64 = 0;
    while d < a.ndim {
        var o = og[d];
        if convolve {
            o = -o;
            if weights.shape[d] % 2 == 0 { o -= 1; }
        }
        if !nd_origin_ok(o, weights.shape[d]) {
            return empty_arr(), fault(F_RANGE, tag, "the origin stands outside the weights");
        }
        shift.push(weights.shape[d] / 2 + o);
        d += 1;
    }
    if a.data.len > ND_COST / weights.data.len {
        return empty_arr(), fault(F_LIMIT, tag, "the cost of the filter is above the limit");
    }
    var wk: f64[>..] = [];
    var j: i64 = 0;
    while j < weights.data.len {
        if convolve { wk.push(weights.data[weights.data.len - 1 - j]); } else { wk.push(weights.data[j]); }
        j += 1;
    }
    var out: f64[>..] = nd_zeros(a.data.len);
    nd_direct(a.data, a.shape, wk, weights.shape, shift, m, cval, out);
    return arr_of(copy(out), a.shape), no_fault();
}

// correlate_nd correlates the array with an array of weights that holds
// the count of dimensions of the array. The origin holds none, one, or one
// value for each axis.
fn correlate_nd(a, weights, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return nd_weightsn(a, weights, mode, cval, origin, false, "correlate_nd");
}

// convolve_nd convolves the array with an array of weights.
fn convolve_nd(a, weights, mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    return nd_weightsn(a, weights, mode, cval, origin, true, "convolve_nd");
}

// uniform_filter gives the mean of a box along every axis. The size and
// the origin hold none, one, or one value for each axis.
fn uniform_filter(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    let m, f = nd_check(a, mode, "uniform_filter");
    if f.kind != F_NONE { return empty_arr(), f; }
    var sz: i64[..8] = [];
    if size.len == 0 || !nd_list(size, a.ndim, 3, sz) {
        return empty_arr(), fault(F_SHAPE, "uniform_filter", "the size holds a count that is not one for each axis");
    }
    var og: i64[..8] = [];
    if !nd_list(origin, a.ndim, 0, og) {
        return empty_arr(), fault(F_SHAPE, "uniform_filter", "the origin holds a count that is not one for each axis");
    }
    var cur: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { cur.push(a.data[i]); i += 1; }
    var d: i64 = 0;
    while d < a.ndim {
        var nxt: f64[>..] = nd_zeros(a.data.len);
        let g = nd_box_axis(cur, a.shape, d, 1, sz[d], og[d], m, cval, "uniform_filter", nxt);
        if g.kind != F_NONE { return empty_arr(), g; }
        cur = copy(nxt);
        d += 1;
    }
    return arr_of(copy(cur), a.shape), no_fault();
}

// gaussian_filter filters the array with a Gaussian along every axis. The
// sigma holds one value or one for each axis. The order holds none, one,
// or one value for each axis. An axis with a sigma of 1.0e-15 or less stays
// as it is, as in SciPy.
fn gaussian_filter(a, sigma: f64[:], order: i64[:] = [], mode: const u8[:] = "reflect", cval: f64 = 0.0, truncate: f64 = 4.0) -> Arr, Fault {
    let m, f = nd_check(a, mode, "gaussian_filter");
    if f.kind != F_NONE { return empty_arr(), f; }
    if sigma.len != 1 && sigma.len != a.ndim {
        return empty_arr(), fault(F_SHAPE, "gaussian_filter", "the sigma holds a count that is not one for each axis");
    }
    var od: i64[..8] = [];
    if !nd_list(order, a.ndim, 0, od) {
        return empty_arr(), fault(F_SHAPE, "gaussian_filter", "the order holds a count that is not one for each axis");
    }
    var cur: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { cur.push(a.data[i]); i += 1; }
    var d: i64 = 0;
    while d < a.ndim {
        var sg = sigma[0];
        if sigma.len > 1 { sg = sigma[d]; }
        if !(sg - sg == 0.0) {
            return empty_arr(), fault(F_VALUE, "gaussian_filter", "the sigma holds a value that is not finite");
        }
        if sg > 0.000000000000001 {
            var nxt: f64[>..] = nd_zeros(a.data.len);
            let g = nd_gauss_axis(cur, a.shape, d, sg, od[d], truncate, m, cval, "gaussian_filter", nxt);
            if g.kind != F_NONE { return empty_arr(), g; }
            cur = copy(nxt);
        }
        d += 1;
    }
    return arr_of(copy(cur), a.shape), no_fault();
}
