// The products of arrays and the windows: the dot product, the matrix product and the windows.
//
// The products of arrays, the windows and the old functions of the
// polynomials. The old functions take the coefficients from the
// highest power, as polyval takes them.
namespace scientific;

import mathshim;

// vdot gives the dot product of the flattened arrays. The arrays hold
// the same count of elements.
fn vdot(a, b) -> f64, Fault {
    if a.data.len != b.data.len {
        return 0.0, fault(F_SHAPE, "vdot", "the arrays hold different counts of elements");
    }
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < a.data.len { t += a.data[i] * b.data[i]; i += 1; }
    return t, no_fault();
}

// vecdot sums the products of two arrays along an axis. The other
// dimensions broadcast. The result drops the axis.
fn vecdot(a, b, axis: i64 = -1) -> Arr, Fault {
    if a.ndim < 1 || b.ndim < 1 {
        return empty_arr(), fault(F_SHAPE, "vecdot", "an operand holds no dimension");
    }
    let ax = fix_axis(axis, a.ndim);
    let bx = fix_axis(axis, b.ndim);
    if ax < 0 || bx < 0 {
        return empty_arr(), fault(F_RANGE, "vecdot", "the axis stands outside the shape");
    }
    if a.shape[ax] != b.shape[bx] {
        return empty_arr(), fault(F_SHAPE, "vecdot", "the lengths of the axis do not agree");
    }
    let width = a.shape[ax];
    // The dimensions that the arrays keep when they drop the axis.
    var ab: i64[..8] = [];
    var bb: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim { if i != ax { ab.push(a.shape[i]); } i += 1; }
    i = 0;
    while i < b.ndim { if i != bx { bb.push(b.shape[i]); } i += 1; }
    let cb, fb = batch_shape(ab, bb);
    if fb.kind != F_NONE { return empty_arr(), fb; }
    var astr: i64[>..] = [];
    var bstr: i64[>..] = [];
    batch_strides(a.shape, ax, astr);
    batch_strides(b.shape, bx, bstr);
    let astep = product_after_of(a.shape, ax);
    let bstep = product_after_of(b.shape, bx);
    let batch = batch_count(cb);
    var idx: i64[>..] = [];
    i = 0;
    while i < cb.len { idx.push(0); i += 1; }
    var d: f64[>..] = [];
    var bi: i64 = 0;
    while bi < batch {
        batch_step(cb, idx, bi);
        let aoff = batch_off(ab, astr, idx, cb.len);
        let boff = batch_off(bb, bstr, idx, cb.len);
        var t: f64 = 0.0;
        var k: i64 = 0;
        while k < width {
            t = t + a.data[aoff + k * astep] * b.data[boff + k * bstep];
            k += 1;
        }
        d.push(t);
        bi += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < cb.len { s.push(cb[i]); i += 1; }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// matvec multiplies a matrix by a row. The last two dimensions of the
// first operand give the matrices. The row is the last dimension of the
// second operand. The other dimensions broadcast.
fn matvec(a, b) -> Arr, Fault {
    if a.ndim < 2 || b.ndim < 1 {
        return empty_arr(), fault(F_SHAPE, "matvec", "the operands are not a matrix and a row");
    }
    let m = a.shape[a.ndim - 2];
    let n = a.shape[a.ndim - 1];
    if b.shape[b.ndim - 1] != n {
        return empty_arr(), fault(F_SHAPE, "matvec", "the inner dimensions do not agree");
    }
    // The dimensions that the arrays keep before the matrix.
    var ab: i64[..8] = [];
    var bb: i64[..8] = [];
    var i: i64 = 0;
    while i < a.ndim - 2 { ab.push(a.shape[i]); i += 1; }
    i = 0;
    while i < b.ndim - 1 { bb.push(b.shape[i]); i += 1; }
    let cb, fb = batch_shape(ab, bb);
    if fb.kind != F_NONE { return empty_arr(), fb; }
    // The steps of the batch dimensions.
    var astr: i64[>..] = [];
    var bstr: i64[>..] = [];
    i = 0;
    while i < ab.len { astr.push(0); i += 1; }
    i = 0;
    while i < bb.len { bstr.push(0); i += 1; }
    var acc: i64 = m * n;
    i = ab.len - 1;
    while i >= 0 { astr[i] = acc; acc = acc * ab[i]; i -= 1; }
    acc = n;
    i = bb.len - 1;
    while i >= 0 { bstr[i] = acc; acc = acc * bb[i]; i -= 1; }
    var batch: i64 = 1;
    i = 0;
    while i < cb.len {
        if cb[i] > 0 && batch > MAX_ELEMENTS / cb[i] {
            return empty_arr(), fault(F_LIMIT, "matvec", "the result holds more elements than the limit");
        }
        batch = batch * cb[i];
        i += 1;
    }
    var idx: i64[>..] = [];
    i = 0;
    while i < cb.len { idx.push(0); i += 1; }
    var d: f64[>..] = [];
    var bi: i64 = 0;
    while bi < batch {
        batch_step(cb, idx, bi);
        let aoff = batch_off(ab, astr, idx, cb.len);
        let boff = batch_off(bb, bstr, idx, cb.len);
        var r: i64 = 0;
        while r < m {
            var t: f64 = 0.0;
            var p: i64 = 0;
            while p < n {
                t = t + a.data[aoff + r * n + p] * b.data[boff + p];
                p += 1;
            }
            d.push(t);
            r += 1;
        }
        bi += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < cb.len { s.push(cb[i]); i += 1; }
    s.push(m);
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// vecmat multiplies a row by a matrix. The result is a row.
fn vecmat(v, a) -> Arr, Fault {
    if a.ndim != 2 || v.ndim != 1 {
        return empty_arr(), fault(F_SHAPE, "vecmat", "the operands are not a row and a matrix");
    }
    let r, f = dot(v, a);
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, "vecmat", f.detail); }
    return ravel(r);
}

// vander gives the matrix of the powers of a row: one column per power.
// The count n of columns of zero means the length of the row. The
// powers fall from left to right unless increasing is true.
fn vander(x: f64[:], n: i64 = 0, increasing: bool = false) -> Arr, Fault {
    var cols = n;
    if cols == 0 { cols = x.len; }
    if x.len < 1 || cols < 1 || x.len * cols > MAX_ELEMENTS {
        return empty_arr(), fault(F_RANGE, "vander", "the row or the count of columns stands outside its range");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len {
        var j: i64 = 0;
        while j < cols {
            var p = cols - 1 - j;
            if increasing { p = j; }
            var t: f64 = 1.0;
            var q: i64 = 0;
            while q < p { t *= x[i]; q += 1; }
            d.push(t);
            j += 1;
        }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(x.len);
    s.push(cols);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// tensordot sums the products of two arrays over the listed axes. The
// axes of a pair at the same place in the two lists must hold the same
// length. The result holds the free axes of a, then the free axes of
// b. A result with no axis is a row of one element.
fn tensordot(a, b, axes_a: i64[:], axes_b: i64[:]) -> Arr, Fault {
    if axes_a.len != axes_b.len {
        return empty_arr(), fault(F_SHAPE, "tensordot", "the lists of axes hold different counts");
    }
    var pa: i64[..8] = [];
    var pb: i64[..8] = [];
    var used_a: i64 = 0;
    var used_b: i64 = 0;
    var kdim: i64 = 1;
    var i: i64 = 0;
    // The summed axes of a come last, and those of b come first.
    var sum_a: i64[..8] = [];
    var sum_b: i64[..8] = [];
    while i < axes_a.len {
        let x = fix_axis(axes_a[i], a.ndim);
        let y = fix_axis(axes_b[i], b.ndim);
        if x < 0 || y < 0 {
            return empty_arr(), fault(F_RANGE, "tensordot", "an axis stands outside the shape");
        }
        if used_a & (1 << x) != 0 || used_b & (1 << y) != 0 {
            return empty_arr(), fault(F_VALUE, "tensordot", "an axis repeats");
        }
        if a.shape[x] != b.shape[y] {
            return empty_arr(), fault(F_SHAPE, "tensordot", "two summed axes hold different lengths");
        }
        used_a |= 1 << x;
        used_b |= 1 << y;
        sum_a.push(x);
        sum_b.push(y);
        kdim *= a.shape[x];
        i += 1;
    }
    var fa: i64[..8] = [];
    var fb: i64[..8] = [];
    var m: i64 = 1;
    var n: i64 = 1;
    i = 0;
    while i < a.ndim {
        if used_a & (1 << i) == 0 { pa.push(i); fa.push(a.shape[i]); m *= a.shape[i]; }
        i += 1;
    }
    i = 0;
    while i < sum_a.len { pa.push(sum_a[i]); pb.push(sum_b[i]); i += 1; }
    i = 0;
    while i < b.ndim {
        if used_b & (1 << i) == 0 { pb.push(i); fb.push(b.shape[i]); n *= b.shape[i]; }
        i += 1;
    }
    if fa.len + fb.len > MAX_DIMS {
        return empty_arr(), fault(F_LIMIT, "tensordot", "the result holds too many dimensions");
    }
    let at, f1 = transpose_perm(a, pa);
    if f1.kind != F_NONE { return empty_arr(), f1; }
    let bt, f2 = transpose_perm(b, pb);
    if f2.kind != F_NONE { return empty_arr(), f2; }
    let am, f3 = reshape(at, [m, kdim]);
    if f3.kind != F_NONE { return empty_arr(), f3; }
    let bm, f4 = reshape(bt, [kdim, n]);
    if f4.kind != F_NONE { return empty_arr(), f4; }
    let c, f5 = matmul(am, bm);
    if f5.kind != F_NONE { return empty_arr(), f5; }
    var s: i64[..8] = [];
    i = 0;
    while i < fa.len { s.push(fa[i]); i += 1; }
    i = 0;
    while i < fb.len { s.push(fb[i]); i += 1; }
    if s.len == 0 { s.push(1); }
    return Arr { ndim: s.len, shape: s, data: copy(c.data) }, no_fault();
}

// tensordot_n sums over the last n axes of a and the first n axes of b.
// NumPy takes n of two by default.
fn tensordot_n(a, b, n: i64) -> Arr, Fault {
    if n < 0 || n > a.ndim || n > b.ndim {
        return empty_arr(), fault(F_RANGE, "tensordot", "the count of axes stands outside the shapes");
    }
    var xa: i64[..8] = [];
    var xb: i64[..8] = [];
    var i: i64 = 0;
    while i < n { xa.push(a.ndim - n + i); xb.push(i); i += 1; }
    return tensordot(a, b, xa, xb);
}

// The windows. The symmetric window of m points comes first.

// win_cos sums the cosine terms of a window of the family of Hann.
fn win_cos(n: i64, m: i64, a: f64[:]) -> f64 {
    var t: f64 = 0.0;
    var k: i64 = 0;
    var sign: f64 = 1.0;
    while k < a.len {
        t += sign * a[k] * mathshim::cosine(6.283185307179586 * k * n / (m - 1));
        sign = -sign;
        k += 1;
    }
    return t;
}

// window_kind gives the number of a window name, or -1.
fn window_kind(name: const u8[:]) -> i64 {
    if name == "boxcar" { return 0; }
    if name == "triang" { return 1; }
    if name == "bartlett" { return 2; }
    if name == "hann" || name == "hanning" { return 3; }
    if name == "hamming" { return 4; }
    if name == "blackman" { return 5; }
    if name == "blackmanharris" { return 6; }
    if name == "nuttall" { return 7; }
    if name == "flattop" { return 8; }
    if name == "bohman" { return 9; }
    if name == "barthann" { return 10; }
    if name == "cosine" { return 11; }
    if name == "lanczos" { return 12; }
    if name == "parzen" { return 13; }
    if name == "kaiser" { return 14; }
    if name == "gaussian" { return 15; }
    if name == "tukey" { return 16; }
    if name == "exponential" { return 17; }
    return -1;
}

// win_one gives the value of a symmetric window of m points at the
// place n. The parameter p serves the kaiser, gaussian, tukey and
// exponential windows.
fn win_one(kind: i64, n: i64, m: i64, p: f64) -> f64 {
    if m == 1 { return 1.0; }
    let x = n as f64;
    let mm = m as f64;
    let half = (mm - 1.0) / 2.0;
    if kind == 0 { return 1.0; }
    if kind == 1 {
        var k = n + 1;
        if m % 2 == 1 {
            if k > (m + 1) / 2 { k = m - n; }
            return 2.0 * k / (mm + 1.0);
        }
        if k > m / 2 { k = m - n; }
        return (2.0 * k - 1.0) / mm;
    }
    if kind == 2 {
        if x <= half { return 2.0 * x / (mm - 1.0); }
        return 2.0 - 2.0 * x / (mm - 1.0);
    }
    if kind == 3 { return win_cos(n, m, [0.5, 0.5]); }
    if kind == 4 { return win_cos(n, m, [0.54, 0.46]); }
    if kind == 5 { return win_cos(n, m, [0.42, 0.5, 0.08]); }
    if kind == 6 { return win_cos(n, m, [0.35875, 0.48829, 0.14128, 0.01168]); }
    if kind == 7 { return win_cos(n, m, [0.3635819, 0.4891775, 0.1365995, 0.0106411]); }
    if kind == 8 { return win_cos(n, m, [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]); }
    if kind == 9 {
        if n == 0 || n == m - 1 { return 0.0; }
        let fac = mathshim::abs_of(-1.0 + 2.0 * x / (mm - 1.0));
        return (1.0 - fac) * mathshim::cosine(3.141592653589793 * fac) + mathshim::sine(3.141592653589793 * fac) / 3.141592653589793;
    }
    if kind == 10 {
        let fac = mathshim::abs_of(x / (mm - 1.0) - 0.5);
        return 0.62 - 0.48 * fac + 0.38 * mathshim::cosine(6.283185307179586 * fac);
    }
    if kind == 11 { return mathshim::sine(3.141592653589793 * (x + 0.5) / mm); }
    if kind == 12 { return mathshim::sinc_of(2.0 * x / (mm - 1.0) - 1.0); }
    if kind == 13 {
        let na = mathshim::abs_of(x - half);
        let r = na / (mm / 2.0);
        if na <= (mm - 1.0) / 4.0 { return 1.0 - 6.0 * r * r + 6.0 * r * r * r; }
        let u = 1.0 - r;
        return 2.0 * u * u * u;
    }
    if kind == 14 {
        let r = (x - half) / half;
        var inside = 1.0 - r * r;
        if inside < 0.0 { inside = 0.0; }
        return mathshim::i0_of(p * mathshim::root(inside)) / mathshim::i0_of(p);
    }
    if kind == 15 {
        let c = x - half;
        return mathshim::power_e(-c * c / (2.0 * p * p));
    }
    if kind == 16 {
        if p <= 0.0 { return 1.0; }
        if p >= 1.0 { return win_cos(n, m, [0.5, 0.5]); }
        let width = mathshim::trunc_of(p * (mm - 1.0) / 2.0) as i64;
        if n <= width {
            return 0.5 * (1.0 + mathshim::cosine(3.141592653589793 * (-1.0 + 2.0 * x / (p * (mm - 1.0)))));
        }
        if n >= m - width - 1 {
            return 0.5 * (1.0 + mathshim::cosine(3.141592653589793 * (-2.0 / p + 1.0 + 2.0 * x / (p * (mm - 1.0)))));
        }
        return 1.0;
    }
    return mathshim::power_e(-mathshim::abs_of(x - half) / p);
}

// window_build builds a window of m points. A periodic window drops the
// last point of the symmetric window of m + 1 points.
fn window_build(kind: i64, m: i64, p: f64, periodic: bool, tag: const u8[:]) -> Arr, Fault {
    if m < 1 || m > MAX_ELEMENTS - 1 {
        return empty_arr(), fault(F_RANGE, tag, "the length stands outside its range");
    }
    var len = m;
    if periodic { len = m + 1; }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < m { d.push(win_one(kind, i, len, p)); i += 1; }
    var s: i64[..8] = [];
    s.push(m);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// hanning gives the window of Hann of m points.
fn hanning(m: i64) -> Arr, Fault { return window_build(3, m, 0.0, false, "hanning"); }

// hamming gives the window of Hamming of m points.
fn hamming(m: i64) -> Arr, Fault { return window_build(4, m, 0.0, false, "hamming"); }

// blackman gives the window of Blackman of m points.
fn blackman(m: i64) -> Arr, Fault { return window_build(5, m, 0.0, false, "blackman"); }

// bartlett gives the triangular window of m points with zero ends.
fn bartlett(m: i64) -> Arr, Fault { return window_build(2, m, 0.0, false, "bartlett"); }

// kaiser gives the window of Kaiser of m points. The shape beta lies
// between 0 and 700, because the Bessel function overflows above.
fn kaiser(m: i64, beta: f64) -> Arr, Fault {
    if !(beta >= 0.0 && beta <= 700.0) {
        return empty_arr(), fault(F_RANGE, "kaiser", "beta stands outside zero to 700");
    }
    return window_build(14, m, beta, false, "kaiser");
}

// get_window gives a window by its name. The names are boxcar, triang,
// bartlett, hann, hamming, blackman, blackmanharris, nuttall, flattop,
// bohman, barthann, cosine, lanczos, parzen, kaiser, gaussian, tukey
// and exponential. A periodic window (fftbins true) serves the Fourier
// transform. The parameter holds beta for kaiser, the spread for
// gaussian, the share of the taper for tukey (0.5 if absent) and the
// decay for exponential.
fn get_window(name: const u8[:], nx: i64, fftbins: bool = true, param: f64 = 0.0 / 0.0) -> Arr, Fault {
    let kind = window_kind(name);
    if kind < 0 {
        return empty_arr(), fault(F_VALUE, "get_window", "the name holds an unknown window");
    }
    var p = param;
    if kind == 16 && p != p { p = 0.5; }
    if kind == 14 && !(p >= 0.0 && p <= 700.0) {
        return empty_arr(), fault(F_RANGE, "get_window", "beta stands outside zero to 700");
    }
    if (kind == 15 || kind == 17) && !(p > 0.0) {
        return empty_arr(), fault(F_VALUE, "get_window", "the window needs a parameter above zero");
    }
    return window_build(kind, nx, p, fftbins, "get_window");
}

// The old functions of the polynomials.

