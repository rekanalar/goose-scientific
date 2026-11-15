// The fast convolution and the upsample filter.
//
// oaconvolve convolves two signals by the overlap add method. It cuts
// the longer signal into blocks and adds the tails of the blocks.
// upfirdn puts zeros between the samples of a signal, filters it and
// keeps every down-th value.
//
// The formulas follow SciPy 1.18.1. The source of the convolution is
// scipy/signal/_signaltools.py. The source of the upsample filter is
// scipy/signal/_upfirdn.py and scipy/signal/_upfirdn_apply.pyx.
namespace scientific;

import mathshim;

// oa_opt_size gives the length of a block of the overlap add method.
// The rule of SciPy comes from the Lambert W function. The call finds
// the root of the same equation with a few steps of Newton's rule.
fn oa_opt_size(k: i64) -> i64 {
    if k < 1 { return 1; }
    // The root stands where N equals k times the logarithm of two times
    // N times the root of e. The rule of Newton walks to it.
    var n: f64 = (k as f64) * 4.0;
    var i: i64 = 0;
    while i < 60 {
        let f = n - (k as f64) * mathshim::ln(2.0 * n * 2.718281828459045);
        let df = 1.0 - (k as f64) / n;
        if df == 0.0 { break; }
        let step = f / df;
        n -= step;
        if n < 1.0 { n = 1.0; }
        if step < 0.0 && step > -1.0e-9 { break; }
        i += 1;
    }
    let r = mathshim::ceil_of(n);
    if r < 1.0 { return 1; }
    return r as i64;
}

// oa_block_len gives the length of a block of the overlap add method.
// The call gives zero when the blocks give no gain.
fn oa_block_len(s1: i64, s2: i64) -> i64 {
    if s1 < 1 || s2 < 1 { return 0; }
    if s1 == s2 || s1 == 1 || s2 == 1 { return 0; }
    var big = s1;
    var small = s2;
    if small > big {
        big = s2;
        small = s1;
    }
    if small * 2 >= big { return 0; }
    let opt = oa_opt_size(small - 1);
    // The transform of the package likes a power of two.
    var blk: i64 = 1;
    while blk < opt { blk *= 2; }
    if blk >= big { return 0; }
    return blk;
}

// oaconvolve convolves two signals by the overlap add method. The
// mode is "full", "same" for the length of the first signal, or
// "valid" for the values that need no padding.
fn oaconvolve(in1: f64[:], in2: f64[:], mode: const u8[:] = "full") -> Arr, Fault {
    if in1.len < 1 || in2.len < 1 {
        return empty_arr(), fault(F_SHAPE, "oaconvolve", "a signal holds no value");
    }
    if mode != "full" && mode != "same" && mode != "valid" {
        return empty_arr(), fault(F_VALUE, "oaconvolve", "the mode holds an unknown name");
    }
    let n1 = in1.len;
    let n2 = in2.len;
    let total = n1 + n2 - 1;
    if total > MAX_DSP_LENGTH {
        return empty_arr(), fault(F_LIMIT, "oaconvolve", "the result is longer than the limit");
    }
    // The longer signal gives the blocks. The shorter one gives the
    // kernel of each block.
    var long = in1;
    var short = in2;
    if n2 > n1 {
        long = in2;
        short = in1;
    }
    let nlong = long.len;
    let nshort = short.len;
    var acc: f64[>..] = [];
    var i: i64 = 0;
    while i < total { acc.push(0.0); i += 1; }
    let blk = oa_block_len(nlong, nshort);
    if blk > 0 {
        // The blocks tile the long signal without an overlap. Every
        // block holds step values of the signal and takes zeros up to
        // the length of the transform. The tails of the block results
        // add the values of the next block, as the overlap add method
        // does.
        let step = blk - nshort + 1;
        var at: i64 = 0;
        while at < nlong {
            var part: f64[>..] = [];
            var k: i64 = 0;
            while k < blk {
                if at + k < nlong && k < step {
                    part.push(long[at + k]);
                } else {
                    part.push(0.0);
                }
                k += 1;
            }
            let cp, cf = fftconvolve(part[..], short, "full");
            if cf.kind != F_NONE { return empty_arr(), cf; }
            k = 0;
            while k < cp.data.len && at + k < total {
                acc[at + k] += cp.data[k];
                k += 1;
            }
            at += step;
        }
    } else {
        // One block covers the whole signal. The ordinary convolution
        // gives the same values.
        let full, f = fftconvolve(in1, in2, "full");
        if f.kind != F_NONE { return empty_arr(), f; }
        i = 0;
        while i < full.data.len {
            acc[i] = full.data[i];
            i += 1;
        }
    }
    var out: f64[>..] = [];
    i = 0;
    while i < total { out.push(acc[i]); i += 1; }
    return oa_apply_mode(out, n1, n2, mode, "oaconvolve");
}

// oa_apply_mode cuts the values of the full overlap that the mode
// asks for. The tag names the call that holds the fault.
fn oa_apply_mode(full: f64[:], n1: i64, n2: i64, mode: const u8[:],
                 tag: const u8[:]) -> Arr, Fault {
    let total = n1 + n2 - 1;
    if mode == "full" {
        var out: f64[>..] = [];
        var i: i64 = 0;
        while i < full.len { out.push(full[i]); i += 1; }
        return line_arr(copy(out)), no_fault();
    }
    var size = n1;
    if mode == "valid" {
        size = n1 - n2 + 1;
        if n2 > n1 { size = n2 - n1 + 1; }
    }
    let start = (total - size) / 2;
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < size {
        if start + i < 0 || start + i >= full.len {
            return empty_arr(), fault(F_VALUE, tag, "the mode asks for values outside the overlap");
        }
        out.push(full[start + i]);
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// upfirdn_ext gives the value of a signal at a place outside the
// signal. The mode names the way the signal extends. The places
// below zero need the left rule, the places at or above the length
// need the right rule.
fn upfirdn_ext(x: f64[:], idx: i64, mode: const u8[:], cval: f64, left: bool) -> f64 {
    let n = x.len;
    if mode == "constant" { return cval; }
    if mode == "edge" {
        if left { return x[0]; }
        return x[n - 1];
    }
    if mode == "symmetric" {
        var j: i64 = 0;
        if left { j = (-idx - 1) % (2 * n); } else { j = idx % (2 * n); }
        if j < n { return x[j]; }
        return x[n - 1 - (j - n)];
    }
    if mode == "reflect" {
        // The mode reflects the signal about its edge. The left side
        // starts at the second value and the right side starts at the
        // value before the last one. The period holds twice the count
        // of the values less one.
        if n < 2 { return x[0]; }
        let period = 2 * (n - 1);
        var j: i64 = 0;
        if left {
            j = (-idx - 1) % period;
            if j < n - 1 { return x[j + 1]; }
            return x[period - 1 - j];
        }
        j = idx % period;
        if j <= n - 1 { return x[j]; }
        return x[period - j];
    }
    if mode == "wrap" {
        var j: i64 = 0;
        if left { j = (-idx - 1) % n; } else { j = idx % n; }
        if left { return x[n - 1 - j]; }
        return x[j];
    }
    if mode == "smooth" {
        if left { return x[0] + idx as f64 * (x[1] - x[0]); }
        return x[n - 1] + (idx - n + 1) as f64 * (x[n - 1] - x[n - 2]);
    }
    if mode == "line" {
        let slope = (x[n - 1] - x[0]) / (n - 1) as f64;
        if left { return x[0] + idx as f64 * slope; }
        return x[n - 1] + (idx - n + 1) as f64 * slope;
    }
    if mode == "antisymmetric" {
        if left {
            let j = (-idx - 1) % (2 * n);
            if j < n { return -x[j]; }
            return x[n - 1 - (j - n)];
        }
        if idx < 2 * n { return -x[n - 1 - (idx - n)]; }
        let j2 = idx % (2 * n);
        if j2 < n { return x[j2]; }
        return -x[n - 1 - (j2 - n)];
    }
    if mode == "antireflect" {
        if left {
            if -idx < n { return x[0] - (x[-idx] - x[0]); }
            let le = x[0] + (x[0] - x[n - 1]) * ((-idx - 1) / (n - 1)) as f64;
            let j = (-idx - 1) % (2 * (n - 1));
            if j < n - 1 { return le - (x[j + 1] - x[0]); }
            return le - (x[n - 1] - x[n - 2 - (j - (n - 1))]);
        }
        if idx < 2 * n - 1 {
            return x[n - 1] - (x[n - 2 - (idx - n)] - x[n - 1]);
        }
        let re = x[n - 1] + (x[n - 1] - x[0]) * ((idx / (n - 1)) - 1) as f64;
        let j = idx % (2 * (n - 1));
        if j < n - 1 { return re + (x[j] - x[0]); }
        return re + (x[n - 1] - x[n - 1 - (j - (n - 1))]);
    }
    return cval;
}

// upfirdn_line filters one line of a signal with the taps h. The
// line first gets up minus one zeros between its samples. Then the
// filter runs. Then the call keeps every down-th value.
fn upfirdn_line(h: f64[:], x: f64[:], up: i64, down: i64, mode: const u8[:],
                 cval: f64, out: f64[>..]&) {
    let len_x = x.len;
    let len_h = h.len;
    let out_len = (((len_x - 1) * up + len_h) - 1) / down + 1;
    var i: i64 = 0;
    while i < out_len {
        let n = i * down;
        var s: f64 = 0.0;
        // The loop starts at the taps that give m a multiple of up.
        var j: i64 = n % up;
        while j < len_h {
            let m = n - j;
            let k = m / up;
            if k >= 0 && k < len_x {
                s += h[j] * x[k];
            } else if k < 0 {
                s += h[j] * upfirdn_ext(x, k, mode, cval, true);
            } else {
                s += h[j] * upfirdn_ext(x, k, mode, cval, false);
            }
            j += up;
        }
        out.push(s);
        i += 1;
    }
}

// upfirdn filters a signal with the taps h. The axis names the
// direction of the signal in the array. The signal first gets up
// minus one zeros between its samples. Then the filter runs along the
// axis. Then the call keeps every down-th value.
fn upfirdn(h: f64[:], x, up: i64 = 1, down: i64 = 1, axis: i64 = -1,
           mode: const u8[:] = "constant", cval: f64 = 0.0) -> Arr, Fault {
    let tag = "upfirdn";
    if h.len < 1 {
        return empty_arr(), fault(F_VALUE, tag, "the taps hold no value");
    }
    if x.ndim < 1 {
        return empty_arr(), fault(F_SHAPE, tag, "the signal holds no dimension");
    }
    let fs = check_stack(x.ndim, x.shape, x.data.len, 1, tag);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    if x.data.len < 1 {
        return empty_arr(), fault(F_SHAPE, tag, "the signal holds no value");
    }
    if up < 1 || down < 1 {
        return empty_arr(), fault(F_VALUE, tag, "the rates must be one or more");
    }
    if up > 100000 || down > 100000 {
        return empty_arr(), fault(F_LIMIT, tag, "a rate stands outside the limit");
    }
    if mode != "constant" && mode != "symmetric" && mode != "reflect" && mode != "edge"
        && mode != "wrap" && mode != "smooth" && mode != "antisymmetric"
        && mode != "antireflect" && mode != "line" {
        return empty_arr(), fault(F_VALUE, tag, "the mode holds an unknown name");
    }
    var ax = axis;
    if ax < 0 { ax = ax + x.ndim; }
    if ax < 0 || ax >= x.ndim {
        return empty_arr(), fault(F_RANGE, tag, "the axis stands outside the shape");
    }
    let n = x.shape[ax];
    let out_len = (((n - 1) * up + h.len) - 1) / down + 1;
    // The count of the values before the axis and after it.
    var outer: i64 = 1;
    var inner: i64 = 1;
    var i: i64 = 0;
    while i < ax { outer *= x.shape[i]; i += 1; }
    i = ax + 1;
    while i < x.ndim { inner *= x.shape[i]; i += 1; }
    if outer * inner * out_len > MAX_DSP_LENGTH {
        return empty_arr(), fault(F_LIMIT, tag, "the result is longer than the limit");
    }
    var out: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        // The values of one part of the axis. The call keeps them in
        // the order of the inner values first.
        var part: f64[>..] = [];
        var q: i64 = 0;
        while q < inner {
            var line: f64[>..] = [];
            var k: i64 = 0;
            while k < n {
                line.push(x.data[o * n * inner + k * inner + q]);
                k += 1;
            }
            upfirdn_line(h, line[..], up, down, mode, cval, part);
            q += 1;
        }
        // The values of the axis come between the inner values.
        var k: i64 = 0;
        while k < out_len {
            q = 0;
            while q < inner {
                out.push(part[q * out_len + k]);
                q += 1;
            }
            k += 1;
        }
        o += 1;
    }
    var shape: i64[..8] = [];
    i = 0;
    while i < x.ndim {
        if i == ax {
            shape.push(out_len);
        } else {
            shape.push(x.shape[i]);
        }
        i += 1;
    }
    return arr_of(copy(out), shape), no_fault();
}
