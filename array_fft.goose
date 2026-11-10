// The discrete Fourier transform.
//
// The package holds real numbers only. The transform works on two
// arrays: one holds the real parts and one holds the imaginary parts.
// The forward transform takes the negative sign, as NumPy does.
namespace scientific;

import mathshim;

// is_power_of_two tells whether a number holds a power of two.
fn is_power_of_two(n: i64) -> bool {
    if n < 1 { return false; }
    var v = n;
    while v % 2 == 0 { v /= 2; }
    return v == 1;
}

// twiddle holds the root of unity for a step. The angle is
// -2 * pi * step / n.
fn twiddle(step: i64, n: i64) -> f64, f64 {
    let angle = -6.283185307179586 * step / n;
    return mathshim::cosine(angle), mathshim::sine(angle);
}

// twiddle_sign holds the root of unity for a step. The backward
// transform takes the positive sign.
fn twiddle_sign(step: i64, n: i64, inverse: bool) -> f64, f64 {
    if inverse { return twiddle(-step, n); }
    return twiddle(step, n);
}

// fft_radix2 takes the transform of a signal whose length is a power
// of two. The result holds the real parts and the imaginary parts.
// bit_reverse puts the elements in the order of the reversed bits of
// their place. The transform needs this order.
fn bit_reverse(re: f64[:], im: f64[:], n: i64) {
    // The count of the bits of the place.
    var bits: i64 = 0;
    var t = n;
    while t > 1 {
        t /= 2;
        bits += 1;
    }
    var i: i64 = 0;
    while i < n {
        let j = reverse_bits(i, bits);
        if j > i {
            let tr = re[i];
            let ti = im[i];
            re[i] = re[j];
            im[i] = im[j];
            re[j] = tr;
            im[j] = ti;
        }
        i += 1;
    }
}

// reverse_bits gives the place with the reversed bits.
fn reverse_bits(value: i64, bits: i64) -> i64 {
    var r: i64 = 0;
    var v = value;
    var i: i64 = 0;
    while i < bits {
        r = r * 2 + v % 2;
        v /= 2;
        i += 1;
    }
    return r;
}

// fft_radix2 takes the transform of a signal whose length is a power
// of two. The method runs the turns in the order of the size. The
// result holds the real parts and the imaginary parts.
fn fft_radix2(re: f64[:], im: f64[:], n: i64, inverse: bool) {
    bit_reverse(re, im, n);
    // The size of the block grows from two to the length.
    var size: i64 = 2;
    while size <= n {
        let half = size / 2;
        let step = n / size;
        var i: i64 = 0;
        while i < n {
            var k: i64 = 0;
            while k < half {
                let c, s = twiddle_sign(k * step, n, inverse);
                let a = i + k;
                let b = a + half;
                let tr = re[b] * c - im[b] * s;
                let ti = re[b] * s + im[b] * c;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] = re[a] + tr;
                im[a] = im[a] + ti;
                k += 1;
            }
            i += size;
        }
        size *= 2;
    }
}

// fft_bluestein takes the transform of a signal whose length holds a
// factor that is not a power of two. The method turns the transform
// into a convolution.
fn fft_bluestein(re: f64[:], im: f64[:], n: i64) -> Fault {
    // The length of the convolution, a power of two.
    var m: i64 = 1;
    while m < 2 * n { m *= 2; }
    // The chirp of the signal.
    var chirp_re: f64[>..] = [];
    var chirp_im: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        // The square of the index, taken so that the angle stays
        // small.
        let j = (i * i) % (2 * n);
        let c, s = twiddle(j, 2 * n);
        chirp_re.push(c);
        chirp_im.push(s);
        i += 1;
    }
    // The signal takes the chirp.
    var ar: f64[>..] = [];
    var ai: f64[>..] = [];
    i = 0;
    while i < m { ar.push(0.0); ai.push(0.0); i += 1; }
    i = 0;
    while i < n {
        ar[i] = re[i] * chirp_re[i] - im[i] * chirp_im[i];
        ai[i] = re[i] * chirp_im[i] + im[i] * chirp_re[i];
        i += 1;
    }
    // The kernel of the convolution.
    var br: f64[>..] = [];
    var bi: f64[>..] = [];
    i = 0;
    while i < m { br.push(0.0); bi.push(0.0); i += 1; }
    br[0] = chirp_re[0];
    bi[0] = -chirp_im[0];
    i = 1;
    while i < n {
        br[i] = chirp_re[i];
        bi[i] = -chirp_im[i];
        br[m - i] = chirp_re[i];
        bi[m - i] = -chirp_im[i];
        i += 1;
    }
    fft_radix2(ar, ai, m, false);
    fft_radix2(br, bi, m, false);
    // The product of the two transforms.
    i = 0;
    while i < m {
        let tr = ar[i] * br[i] - ai[i] * bi[i];
        let ti = ar[i] * bi[i] + ai[i] * br[i];
        ar[i] = tr;
        ai[i] = ti;
        i += 1;
    }
    // The product goes back with the positive sign. The convolution
    // takes the scale of the length.
    fft_radix2(ar, ai, m, true);
    i = 0;
    while i < m {
        ar[i] = ar[i] / m;
        ai[i] = ai[i] / m;
        i += 1;
    }
    // The result takes the chirp back.
    i = 0;
    while i < n {
        re[i] = ar[i] * chirp_re[i] - ai[i] * chirp_im[i];
        im[i] = ar[i] * chirp_im[i] + ai[i] * chirp_re[i];
        i += 1;
    }
    return no_fault();
}


// fft gives the transform of a real signal. The result holds the real
// parts and the imaginary parts in two arrays.
fn fft(re: f64[:]) -> Arr, Arr, Fault {
    let n = re.len;
    var im: f64[>..] = [];
    var i: i64 = 0;
    while i < n { im.push(0.0); i += 1; }
    return fft_parts(re, im);
}

// fft_parts gives the transform of a signal that holds two parts.
fn fft_parts(re: f64[:], im: f64[:]) -> Arr, Arr, Fault {
    let n = re.len;
    if im.len != n {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "fft", "the parts hold a different count");
    }
    if n < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "fft", "the signal holds no element");
    }
    if n > MAX_ELEMENTS {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "fft", "the signal holds more elements than the limit");
    }
    var out_re: f64[>..] = [];
    var out_im: f64[>..] = [];
    var i: i64 = 0;
    while i < n { out_re.push(re[i]); out_im.push(im[i]); i += 1; }
    if is_power_of_two(n) {
        fft_radix2(out_re, out_im, n, false);
    } else {
        let f = fft_bluestein(out_re, out_im, n);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    }
    let s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: out_re }, Arr { ndim: 1, shape: s, data: out_im }, no_fault();
}

// ifft gives the signal of a transform. The parts of the result come
// with the scale of the length.
fn ifft(re: f64[:], im: f64[:]) -> Arr, Arr, Fault {
    let n = re.len;
    if n < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "ifft", "the transform holds no element");
    }
    if im.len != n {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "ifft", "the parts hold a different count");
    }
    var out_re: f64[>..] = [];
    var out_im: f64[>..] = [];
    var i: i64 = 0;
    while i < n { out_re.push(re[i]); out_im.push(im[i]); i += 1; }
    if is_power_of_two(n) {
        fft_radix2(out_re, out_im, n, true);
    } else {
        // The Bluestein method takes the forward transform. The parts
        // take the opposite sign, and the result takes it back.
        var sr: f64[>..] = [];
        var si: f64[>..] = [];
        i = 0;
        while i < n { sr.push(out_re[i]); si.push(-out_im[i]); i += 1; }
        let f = fft_bluestein(sr, si, n);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
        out_re = copy(sr);
        i = 0;
        while i < n {
            out_im[i] = -si[i];
            i += 1;
        }
    }
    // The signal comes back with the scale of the length.
    i = 0;
    while i < n {
        out_re[i] = out_re[i] / n;
        out_im[i] = out_im[i] / n;
        i += 1;
    }
    let s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: out_re }, Arr { ndim: 1, shape: s, data: out_im }, no_fault();
}

// fftfreq gives the frequencies of a transform of a given length. The
// frequencies come in the order of NumPy: the positive ones, then the
// negative ones.
fn fftfreq(n: i64, d: f64) -> Arr, Fault {
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "fftfreq", "the length holds a value below one");
    }
    if d == 0.0 {
        return empty_arr(), fault(F_VALUE, "fftfreq", "the step holds zero");
    }
    var d2: f64[>..] = [];
    var i: i64 = 0;
    while i < n { d2.push(0.0); i += 1; }
    // The positive frequencies come first, then the negative ones.
    // The count of the positive ones follows NumPy.
    var count = (n - 1) / 2 + 1;
    i = 0;
    while i < n {
        var k = i;
        if i >= count { k = i - count - n / 2; }
        d2[i] = k / (n * d);
        i += 1;
    }
    let s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: d2 }, no_fault();
}

// fftshift moves the zero frequency to the middle of the
// frequencies.
fn fftshift(a) -> Arr, Fault {
    let n = a.data.len;
    var d: f64[>..] = [];
    var half = n / 2;
    var i: i64 = 0;
    while i < n {
        var k = i + half;
        if k >= n { k = k - n; }
        d.push(a.data[k]);
        i += 1;
    }
    let s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// fft2 gives the transform of a matrix along both axes.
fn fft2(a) -> Arr, Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "fft2", "the operand holds a count of dimensions that is not two");
    }
    let rows = a.shape[0];
    let cols = a.shape[1];
    // The transform of each row.
    var re: f64[>..] = [];
    var im: f64[>..] = [];
    var r: i64 = 0;
    while r < rows {
        var line_re: f64[>..] = [];
        var line_im: f64[>..] = [];
        var c: i64 = 0;
        while c < cols {
            line_re.push(a.data[r * cols + c]);
            line_im.push(0.0);
            c += 1;
        }
        let lr, li, f = fft(line_re);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
        c = 0;
        while c < cols {
            re.push(lr.data[c]);
            im.push(li.data[c]);
            c += 1;
        }
        r += 1;
    }
    // The transform of each column.
    var out_re: f64[>..] = [];
    var out_im: f64[>..] = [];
    var c2: i64 = 0;
    while c2 < cols {
        var line_re: f64[>..] = [];
        var line_im: f64[>..] = [];
        r = 0;
        while r < rows {
            line_re.push(re[r * cols + c2]);
            line_im.push(im[r * cols + c2]);
            r += 1;
        }
        let lr, li, f = fft_parts(line_re, line_im);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
        r = 0;
        while r < rows {
            out_re.push(lr.data[r]);
            out_im.push(li.data[r]);
            r += 1;
        }
        c2 += 1;
    }
    // The result holds the rows in the order of the columns.
    var final_re: f64[>..] = [];
    var final_im: f64[>..] = [];
    r = 0;
    while r < rows {
        c2 = 0;
        while c2 < cols {
            final_re.push(out_re[c2 * rows + r]);
            final_im.push(out_im[c2 * rows + r]);
            c2 += 1;
        }
        r += 1;
    }
    let s: i64[..8] = [];
    s.push(rows);
    s.push(cols);
    return Arr { ndim: 2, shape: s, data: final_re }, Arr { ndim: 2, shape: s, data: final_im }, no_fault();
}

// rfft gives the transform of a real signal with only the positive
// frequencies. The count of the frequencies is the length divided by
// two, plus one.
fn rfft(re: f64[:]) -> Arr, Arr, Fault {
    let n = re.len;
    let lr, li, f = fft(re);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let count = n / 2 + 1;
    var out_re: f64[>..] = [];
    var out_im: f64[>..] = [];
    var i: i64 = 0;
    while i < count {
        out_re.push(lr.data[i]);
        out_im.push(li.data[i]);
        i += 1;
    }
    let s: i64[..8] = [];
    s.push(count);
    return Arr { ndim: 1, shape: s, data: out_re }, Arr { ndim: 1, shape: s, data: out_im }, no_fault();
}

// irfft gives the signal of a transform that holds only the positive
// frequencies. The length of the signal comes from the argument.
fn irfft(re: f64[:], im: f64[:], n: i64) -> Arr, Fault {
    if re.len != im.len {
        return empty_arr(), fault(F_SHAPE, "irfft", "the parts hold a different count");
    }
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "irfft", "the length holds a value below one");
    }
    let count = re.len;
    // The positive half of a signal of the length n holds the count
    // of n divided by two, plus one. A different count is a fault.
    if count != n / 2 + 1 {
        return empty_arr(), fault(F_SHAPE, "irfft", "the count of the parts does not hold the length");
    }
    // The negative frequencies come from the positive ones. The whole
    // spectrum holds n places.
    var full_re: f64[>..] = [];
    var full_im: f64[>..] = [];
    var i: i64 = 0;
    while i < n { full_re.push(0.0); full_im.push(0.0); i += 1; }
    i = 0;
    while i < count {
        full_re[i] = re[i];
        full_im[i] = im[i];
        i += 1;
    }
    // The middle place of an even length holds its own value.
    var nyquist = false;
    if n % 2 == 0 { nyquist = true; }
    i = 1;
    while i < count {
        if !(nyquist && i == n / 2) {
            full_re[n - i] = re[i];
            full_im[n - i] = -im[i];
        }
        i += 1;
    }
    let out_re, out_im, f = ifft(full_re, full_im);
    if f.kind != F_NONE { return empty_arr(), f; }
    let s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: copy(out_re.data) }, no_fault();
}
