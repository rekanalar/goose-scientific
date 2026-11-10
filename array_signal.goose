// The functions of a signal.
//
// The functions take the values of a signal. The transform of the
// package takes the parts of a signal by their frequency.
namespace scientific;

import mathshim;

// convolve gives the overlap of two signals. The method is the full
// overlap, as NumPy holds under its default. The kernel turns
// around, as NumPy does. The result holds a count that is the sum of
// the two counts minus one.
fn convolve(a: f64[:], kernel: f64[:]) -> Arr, Fault {
    let n = a.len;
    let m = kernel.len;
    if n < 1 || m < 1 {
        return empty_arr(), fault(F_VALUE, "convolve", "a signal holds no value");
    }
    if n + m - 1 > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "convolve", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n + m - 1 {
        var t: f64 = 0.0;
        // The product of the two signals at every shift. The value of
        // the kernel at the place of the shift comes in the order of
        // the kernel, as NumPy does.
        let lo = i - m + 1;
        var hi = i;
        if hi > n - 1 { hi = n - 1; }
        var j = lo;
        if j < 0 { j = 0; }
        while j <= hi {
            t += a[j] * kernel[i - j];
            j += 1;
        }
        d.push(t);
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// correlate gives the match of a pattern in a signal. The match is
// the overlap with the pattern turned around.
fn correlate(a: f64[:], pattern: f64[:]) -> Arr, Fault {
    let n = a.len;
    let m = pattern.len;
    if n < 1 || m < 1 {
        return empty_arr(), fault(F_VALUE, "correlate", "a signal holds no value");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n + m - 1 {
        var t: f64 = 0.0;
        var k: i64 = 0;
        while k < m {
            let at = i - (m - 1 - k);
            if at >= 0 && at < n {
                t += a[at] * pattern[k];
            }
            k += 1;
        }
        d.push(t);
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// detrend takes the straight line that the least squares of the
// package fits under a signal, and gives the signal with the line
// taken out. The result holds the values of the signal about zero.
fn detrend(y: f64[:]) -> Arr, Fault {
    let n = y.len;
    if n < 2 {
        return empty_arr(), fault(F_VALUE, "detrend", "the signal holds fewer than two values");
    }
    // The design matrix holds a column of ones and a column of the
    // places of the values.
    var design: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        design.push(1.0);
        design.push(i);
        i += 1;
    }
    var ds: i64[..8] = [];
    ds.push(n);
    ds.push(2);
    let design_arr = Arr { ndim: 2, shape: ds, data: copy(design) };
    var ys: i64[..8] = [];
    ys.push(n);
    var ybuf: f64[>..] = [];
    i = 0;
    while i < n { ybuf.push(y[i]); i += 1; }
    let y_arr = Arr { ndim: 1, shape: ys, data: copy(ybuf) };
    // The least squares of the package gives the line.
    let coef, fc = lstsq(design_arr, y_arr);
    if fc.kind != F_NONE { return empty_arr(), fc; }
    // The signal holds the values with the line taken out.
    var d: f64[>..] = [];
    i = 0;
    while i < n {
        d.push(y[i] - (coef.data[0] + coef.data[1] * i));
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// welch gives the size of the parts of a signal by their frequency.
// The signal is cut into parts that overlap by half. Every part
// takes the window of Hann. The transform of the package takes every
// part, and the sizes of the parts come as the mean of the squares
// of the values of the transform. The result holds the sizes of the
// parts of the frequency from zero to half of the count of a part.
fn welch(y: f64[:], size: i64) -> Arr, Fault {
    if size < 2 {
        return empty_arr(), fault(F_VALUE, "welch", "the size of a part holds fewer than two");
    }
    let n = y.len;
    if n < size {
        return empty_arr(), fault(F_VALUE, "welch", "the signal holds fewer values than a part");
    }
    // The window of Hann over a part.
    var window: f64[>..] = [];
    var i: i64 = 0;
    while i < size {
        window.push(0.5 - 0.5 * mathshim::cosine(6.283185307179586 * i / size));
        i += 1;
    }
    // The parts start at every half of the size, so they overlap by
    // half.
    let hop = size / 2;
    var count: i64 = 0;
    var start: i64 = 0;
    while start + size <= n {
        count += 1;
        start += hop;
    }
    if count < 1 {
        return empty_arr(), fault(F_VALUE, "welch", "the signal holds no whole part");
    }
    // The sizes of the parts of the frequency. The transform of a
    // part holds a count of the size, and the sizes of the positive
    // frequencies hold half of it plus one.
    let bins = size / 2 + 1;
    var power: f64[>..] = [];
    i = 0;
    while i < bins { power.push(0.0); i += 1; }
    start = 0;
    while start + size <= n {
        // The part takes the window.
        var part: f64[>..] = [];
        i = 0;
        while i < size {
            part.push(y[start + i] * window[i]);
            i += 1;
        }
        let re, im, f = fft(part);
        if f.kind != F_NONE { return empty_arr(), f; }
        i = 0;
        while i < bins {
            power[i] += re.data[i] * re.data[i] + im.data[i] * im.data[i];
            i += 1;
        }
        start += hop;
    }
    // The mean of the squares of the values of the transform. The
    // scale of the window of Hann divides the sizes. The parts of
    // the frequency above zero hold twice the size, as SciPy holds
    // for a spectrum of one side.
    var scale: f64 = 0.0;
    i = 0;
    while i < size { scale += window[i] * window[i]; i += 1; }
    var d: f64[>..] = [];
    i = 0;
    while i < bins {
        var twice: f64 = 1.0;
        if i > 0 && i < bins - 1 { twice = 2.0; }
        if size % 2 == 1 && i == bins - 1 { twice = 2.0; }
        d.push(twice * power[i] / (count * scale));
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}
