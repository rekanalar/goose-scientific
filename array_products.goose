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

// vecdot sums the products of two arrays of one shape along an axis.
// The result drops the axis. A row gives a row of one element. The
// package does not broadcast the two shapes.
fn vecdot(a, b, axis: i64 = -1) -> Arr, Fault {
    if a.ndim != b.ndim {
        return empty_arr(), fault(F_SHAPE, "vecdot", "the arrays hold different counts of dimensions");
    }
    var i: i64 = 0;
    while i < a.ndim {
        if a.shape[i] != b.shape[i] {
            return empty_arr(), fault(F_SHAPE, "vecdot", "the shapes differ");
        }
        i += 1;
    }
    let ax = fix_axis(axis, a.ndim);
    if ax < 0 {
        return empty_arr(), fault(F_RANGE, "vecdot", "the axis stands outside the shape");
    }
    let width = a.shape[ax];
    let inner = product_after_of(a.shape, ax);
    let outer = product_before_of(a.shape, ax);
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var j: i64 = 0;
        while j < inner {
            var t: f64 = 0.0;
            var k: i64 = 0;
            while k < width {
                let at = o * width * inner + k * inner + j;
                t += a.data[at] * b.data[at];
                k += 1;
            }
            d.push(t);
            j += 1;
        }
        o += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < a.ndim { if i != ax { s.push(a.shape[i]); } i += 1; }
    if s.len == 0 { s.push(1); }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// matvec multiplies a matrix by a row. The result is a row.
fn matvec(a, v) -> Arr, Fault {
    if a.ndim != 2 || v.ndim != 1 {
        return empty_arr(), fault(F_SHAPE, "matvec", "the operands are not a matrix and a row");
    }
    let r, f = dot(a, v);
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, "matvec", f.detail); }
    return ravel(r);
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

// ein_label turns a letter into a number of 0 to 51. Another character
// gives -1.
fn ein_label(c: i64) -> i64 {
    if c >= 65 && c <= 90 { return c - 65; }
    if c >= 97 && c <= 122 { return c - 97 + 26; }
    return -1;
}

// einsum_run sums the products of one or two arrays by a rule of
// letters, as einsum of NumPy does. The caller passes the first array
// again for the second place when the rule holds one operand.
fn einsum_run(spec: const u8[:], a, b, nops: i64) -> Arr, Fault {
    var labs: i64[..16] = [];
    var cut: i64[..4] = [];
    var outl: i64[..16] = [];
    var part: i64 = 0;
    var arrow = false;
    cut.push(0);
    var i: i64 = 0;
    while i < spec.len {
        let c = spec[i] as i64;
        if c == 32 { i += 1; continue; }
        if c == 44 {
            if arrow || part + 1 >= nops {
                return empty_arr(), fault(F_VALUE, "einsum", "the rule holds more operands than the call");
            }
            part += 1;
            cut.push(labs.len);
        } else if c == 45 {
            if arrow || i + 1 >= spec.len || spec[i + 1] != 62 {
                return empty_arr(), fault(F_VALUE, "einsum", "the rule holds a broken arrow");
            }
            arrow = true;
            i += 1;
        } else {
            let l = ein_label(c);
            if l < 0 {
                return empty_arr(), fault(F_VALUE, "einsum", "the rule holds a character that is no letter");
            }
            if arrow { outl.push(l); }
            else {
                if labs.len >= 16 {
                    return empty_arr(), fault(F_LIMIT, "einsum", "the rule holds too many letters");
                }
                labs.push(l);
            }
        }
        i += 1;
    }
    if part + 1 != nops {
        return empty_arr(), fault(F_VALUE, "einsum", "the rule holds fewer operands than the call");
    }
    cut.push(labs.len);
    if cut.len < 2 || cut[1] - cut[0] != a.ndim || (nops == 2 && cut[2] - cut[1] != b.ndim) {
        return empty_arr(), fault(F_SHAPE, "einsum", "the letters do not match the dimensions");
    }
    // The distinct letters and their lengths.
    var ids: i64[..8] = [];
    var size: i64[..8] = [];
    var count: i64[..8] = [];
    var op: i64 = 0;
    while op < nops {
        var p = cut[op];
        while p < cut[op + 1] {
            var dim = a.shape[p - cut[op]];
            if op == 1 { dim = b.shape[p - cut[op]]; }
            var at: i64 = -1;
            var q: i64 = 0;
            while q < ids.len { if ids[q] == labs[p] { at = q; } q += 1; }
            if at < 0 {
                if ids.len >= 8 {
                    return empty_arr(), fault(F_LIMIT, "einsum", "the rule holds too many distinct letters");
                }
                ids.push(labs[p]);
                size.push(dim);
                count.push(1);
            } else {
                count[at] += 1;
                if size[at] == 1 { size[at] = dim; }
                else if dim != 1 && dim != size[at] {
                    return empty_arr(), fault(F_SHAPE, "einsum", "a letter holds two different lengths");
                }
            }
            p += 1;
        }
        op += 1;
    }
    // The letters of the result.
    var ord: i64[..8] = [];
    if arrow {
        var q: i64 = 0;
        while q < outl.len {
            var at: i64 = -1;
            var r: i64 = 0;
            while r < ids.len { if ids[r] == outl[q] { at = r; } r += 1; }
            if at < 0 {
                return empty_arr(), fault(F_VALUE, "einsum", "a letter of the result stands in no operand");
            }
            r = 0;
            while r < ord.len {
                if ord[r] == at {
                    return empty_arr(), fault(F_VALUE, "einsum", "a letter of the result repeats");
                }
                r += 1;
            }
            ord.push(at);
            q += 1;
        }
    } else {
        // The letters that stand once, in the order of the alphabet.
        var l: i64 = 0;
        while l < 52 {
            var q: i64 = 0;
            while q < ids.len {
                if ids[q] == l && count[q] == 1 { ord.push(q); }
                q += 1;
            }
            l += 1;
        }
    }
    let nout = ord.len;
    var rest: i64[..8] = [];
    var q: i64 = 0;
    while q < ids.len {
        var in_out = false;
        var r: i64 = 0;
        while r < nout { if ord[r] == q { in_out = true; } r += 1; }
        if !in_out { rest.push(q); }
        q += 1;
    }
    var all: i64[..8] = [];
    q = 0;
    while q < nout { all.push(ord[q]); q += 1; }
    q = 0;
    while q < rest.len { all.push(rest[q]); q += 1; }
    // The strides of every operand for every letter of the loop.
    var sa: i64[>..] = [];
    strides_of(a.shape, sa);
    var sb: i64[>..] = [];
    strides_of(b.shape, sb);
    var stride_a: i64[..8] = [];
    var stride_b: i64[..8] = [];
    var total: i64 = 1;
    var o_stride: i64[..8] = [];
    var ototal: i64 = 1;
    q = nout - 1;
    while q >= 0 { o_stride.push(0); q -= 1; }
    q = 0;
    while q < all.len {
        var ta: i64 = 0;
        var tb: i64 = 0;
        var p: i64 = 0;
        while p < cut[1] - cut[0] {
            if labs[p] == ids[all[q]] && a.shape[p] != 1 { ta += sa[p]; }
            p += 1;
        }
        if nops == 2 {
            p = 0;
            while p < cut[2] - cut[1] {
                if labs[cut[1] + p] == ids[all[q]] && b.shape[p] != 1 { tb += sb[p]; }
                p += 1;
            }
        }
        stride_a.push(ta);
        stride_b.push(tb);
        total *= size[all[q]];
        if total > MAX_ELEMENTS {
            return empty_arr(), fault(F_LIMIT, "einsum", "the loop holds more steps than the limit");
        }
        q += 1;
    }
    var os: i64[..8] = [];
    var run: i64 = 1;
    q = nout - 1;
    var rev: i64[..8] = [];
    while q >= 0 { rev.push(run); run *= size[ord[q]]; q -= 1; }
    ototal = run;
    q = 0;
    while q < nout { os.push(rev[nout - 1 - q]); q += 1; }
    var d: f64[>..] = [];
    q = 0;
    while q < ototal { d.push(0.0); q += 1; }
    var pos: i64[..8] = [];
    q = 0;
    while q < all.len { pos.push(0); q += 1; }
    var step: i64 = 0;
    while step < total {
        var ia: i64 = 0;
        var ib: i64 = 0;
        var io: i64 = 0;
        q = 0;
        while q < all.len {
            ia += pos[q] * stride_a[q];
            ib += pos[q] * stride_b[q];
            if q < nout { io += pos[q] * os[q]; }
            q += 1;
        }
        var t = a.data[ia];
        if nops == 2 { t = t * b.data[ib]; }
        d[io] = d[io] + t;
        q = all.len - 1;
        while q >= 0 {
            pos[q] += 1;
            if pos[q] < size[all[q]] { break; }
            pos[q] = 0;
            q -= 1;
        }
        step += 1;
    }
    var s: i64[..8] = [];
    q = 0;
    while q < nout { s.push(size[ord[q]]); q += 1; }
    if s.len == 0 { s.push(1); }
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// einsum sums the products of two arrays by a rule of letters, such as
// "ij,jk->ik". A letter that repeats inside one operand walks its
// diagonal. A result with no letter is a row of one element. The rule
// holds no ellipsis.
fn einsum(spec: const u8[:], a, b) -> Arr, Fault {
    return einsum_run(spec, a, b, 2);
}

// einsum1 applies a rule of letters to one array, such as "ii" or
// "ij->ji".
fn einsum1(spec: const u8[:], a) -> Arr, Fault {
    return einsum_run(spec, a, a, 1);
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

// poly gives the coefficients of the monic polynomial of the roots,
// from the highest power.
fn poly(roots: f64[:]) -> Arr, Fault {
    if roots.len < 1 {
        return empty_arr(), fault(F_VALUE, "poly", "the list of roots holds no value");
    }
    if roots.len > MAX_ROOTS_DEGREE {
        return empty_arr(), fault(F_LIMIT, "poly", "the count of roots holds a value above the limit");
    }
    var c: f64[>..] = [1.0];
    var i: i64 = 0;
    while i < roots.len {
        var t: f64[>..] = [];
        var j: i64 = 0;
        while j <= c.len { t.push(0.0); j += 1; }
        j = 0;
        while j < c.len {
            t[j] = t[j] + c[j];
            t[j + 1] = t[j + 1] - roots[i] * c[j];
            j += 1;
        }
        c = copy(t);
        i += 1;
    }
    return line_arr(copy(c)), no_fault();
}

// poly_cplx gives the coefficients of the monic polynomial of complex
// roots, as the real parts and the imaginary parts.
fn poly_cplx(re: f64[:], im: f64[:]) -> Arr, Arr, Fault {
    if re.len < 1 || re.len != im.len {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "poly", "the parts hold no value or differ in count");
    }
    if re.len > MAX_ROOTS_DEGREE {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "poly", "the count of roots holds a value above the limit");
    }
    var c: Cplx[>..] = [Cplx { re: 1.0, im: 0.0 }];
    var i: i64 = 0;
    while i < re.len {
        var t: Cplx[>..] = [];
        var j: i64 = 0;
        while j <= c.len { t.push(Cplx { re: 0.0, im: 0.0 }); j += 1; }
        j = 0;
        while j < c.len {
            t[j] = c_add(t[j], c[j]);
            t[j + 1] = c_sub(t[j + 1], c_mul(Cplx { re: re[i], im: im[i] }, c[j]));
            j += 1;
        }
        c = copy(t);
        i += 1;
    }
    var r: f64[>..] = [];
    var m: f64[>..] = [];
    i = 0;
    while i < c.len { r.push(c[i].re); m.push(c[i].im); i += 1; }
    return line_arr(copy(r)), line_arr(copy(m)), no_fault();
}

// polyadd adds two polynomials from the highest power. The result
// keeps the leading zeros, as NumPy keeps them.
fn polyadd(a: f64[:], b: f64[:]) -> Arr, Fault {
    return poly_sum(a, b, 1.0, "polyadd");
}

// polysub subtracts the second polynomial from the first.
fn polysub(a: f64[:], b: f64[:]) -> Arr, Fault {
    return poly_sum(a, b, -1.0, "polysub");
}

fn poly_sum(a: f64[:], b: f64[:], sign: f64, tag: const u8[:]) -> Arr, Fault {
    if a.len < 1 || b.len < 1 {
        return empty_arr(), fault(F_VALUE, tag, "a polynomial holds no coefficient");
    }
    var n = a.len;
    if b.len > n { n = b.len; }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var v: f64 = 0.0;
        let ia = i - (n - a.len);
        let ib = i - (n - b.len);
        if ia >= 0 { v = a[ia]; }
        if ib >= 0 { v = v + sign * b[ib]; }
        d.push(v);
        i += 1;
    }
    return line_arr(copy(d)), no_fault();
}

// polymul multiplies two polynomials from the highest power.
fn polymul(a: f64[:], b: f64[:]) -> Arr, Fault {
    if a.len < 1 || b.len < 1 {
        return empty_arr(), fault(F_VALUE, "polymul", "a polynomial holds no coefficient");
    }
    var d: f64[>..] = [];
    flt_pmul(a, b, d);
    return line_arr(copy(d)), no_fault();
}

// polydiv divides two polynomials from the highest power and gives the
// quotient and the remainder. The remainder drops its leading zeros.
fn polydiv(u: f64[:], v: f64[:]) -> Arr, Arr, Fault {
    if u.len < 1 || v.len < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "polydiv", "a polynomial holds no coefficient");
    }
    if v[0] == 0.0 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "polydiv", "the leading coefficient of the divisor holds zero");
    }
    let m = u.len;
    let n = v.len;
    if m < n {
        return line_arr([0.0]), line_arr(copy_slice(u)), no_fault();
    }
    var r: f64[>..] = [];
    var i: i64 = 0;
    while i < m { r.push(u[i]); i += 1; }
    var q: f64[>..] = [];
    i = 0;
    while i <= m - n {
        let f = r[i] / v[0];
        q.push(f);
        var j: i64 = 0;
        while j < n { r[i + j] = r[i + j] - f * v[j]; j += 1; }
        i += 1;
    }
    // The remainder holds the last n - 1 places, without leading zeros.
    var rem: f64[>..] = [];
    i = m - n + 1;
    while i < m { rem.push(r[i]); i += 1; }
    var first: i64 = 0;
    while first < rem.len - 1 && rem[first] == 0.0 { first += 1; }
    var out: f64[>..] = [];
    i = first;
    while i < rem.len { out.push(rem[i]); i += 1; }
    if out.len == 0 { out.push(0.0); }
    return line_arr(copy(q)), line_arr(copy(out)), no_fault();
}

fn copy_slice(u: f64[:]) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < u.len { d.push(u[i]); i += 1; }
    return d;
}

// polyder gives the derivative of order m from the highest power. A
// constant gives [0], where NumPy gives an empty array.
fn polyder(p: f64[:], m: i64 = 1) -> Arr, Fault {
    if p.len < 1 || m < 1 {
        return empty_arr(), fault(F_VALUE, "polyder", "the polynomial holds no coefficient or the order is below one");
    }
    var c = copy_slice(p);
    var turn: i64 = 0;
    while turn < m && c.len > 1 {
        var d: f64[>..] = [];
        var i: i64 = 0;
        while i + 1 < c.len { d.push(c[i] * (c.len - 1 - i)); i += 1; }
        c = copy(d);
        turn += 1;
    }
    if turn < m { c = [0.0]; }
    return line_arr(copy(c)), no_fault();
}

// polyint gives the integral of order m from the highest power. The
// list k holds the constants of the integration, one per turn, with
// zero for any constant that the list leaves out.
fn polyint(p: f64[:], m: i64 = 1, k: f64[:] = []) -> Arr, Fault {
    if p.len < 1 || m < 1 || m > 64 {
        return empty_arr(), fault(F_VALUE, "polyint", "the polynomial holds no coefficient or the order stands outside one to 64");
    }
    if k.len > m {
        return empty_arr(), fault(F_VALUE, "polyint", "the list of constants is longer than the order");
    }
    var c = copy_slice(p);
    var turn: i64 = 0;
    while turn < m {
        var d: f64[>..] = [];
        var i: i64 = 0;
        while i < c.len { d.push(c[i] / (c.len - i)); i += 1; }
        if turn < k.len { d.push(k[turn]); } else { d.push(0.0); }
        c = copy(d);
        turn += 1;
    }
    return line_arr(copy(c)), no_fault();
}
