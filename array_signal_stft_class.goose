// The short time transform of a plan.
//
// A plan holds the window, the step and the rate of the transform.
// The forward call gives the transform of a signal. The inverse call
// gives the signal back. The plan holds the dual window that the
// inverse call needs.
//
// The rules follow SciPy 1.18.1 (scipy.signal.ShortTimeFFT). The
// source is scipy/signal/_short_time_fft.py.
namespace scientific;

import mathshim;

// StftPlan holds the plan of a short time transform. It replaces the
// class ShortTimeFFT of SciPy. The window and the dual window hold
// one value for each place of the window. A phase shift below zero
// means that the call applies no shift.
struct StftPlan {
    hop: i64,
    fs: f64,
    fft_mode: const u8[:],
    mfft: i64,
    scale_to: const u8[:],
    phase_shift: f64,
    win_len: i64,
    buf: f64[>..],
}

// stft_plan_empty is the plan that every fault path returns.
fn stft_plan_empty() -> StftPlan {
    return StftPlan { hop: 0, fs: 0.0, fft_mode: "", mfft: 0, scale_to: "", phase_shift: 0.0, win_len: 0, buf: [] };
}

// stft_shift rolls a buffer of the length n by k places. A positive k
// moves the values up, as NumPy roll does.
fn stft_shift(src: f64[:], k: i64, out: f64[>..]&) {
    let n = src.len;
    var s = k % n;
    if s < 0 { s += n; }
    var i: i64 = 0;
    while i < n {
        var at = i - s;
        if at < 0 { at += n; }
        out.push(src[at]);
        i += 1;
    }
}

// stft_all_zero tells whether every value of a buffer from the place
// from stands at zero. A place below zero counts from the end, as a
// slice of NumPy does. A place at or above the length gives true.
fn stft_all_zero(w: f64[:], from: i64) -> bool {
    var at = from;
    if at < 0 { at = at + w.len; }
    if at < 0 { at = 0; }
    var i: i64 = at;
    while i < w.len {
        if w[i] != 0.0 { return false; }
        i += 1;
    }
    return true;
}

// stft_all_zero_head tells whether every value of a buffer up to the
// place to stands at zero. A place below zero cuts the tail, as a
// slice of NumPy does.
fn stft_all_zero_head(w: f64[:], to: i64) -> bool {
    var at = to;
    if at < 0 { at = at + w.len; }
    if at > w.len { at = w.len; }
    if at < 0 { at = 0; }
    var i: i64 = 0;
    while i < at {
        if w[i] != 0.0 { return false; }
        i += 1;
    }
    return true;
}

// stft_dual_window gives the canonical dual window of a window and a
// step. The window must hold an overlap that covers every place.
fn stft_dual_window(win: f64[:], hop: i64, tag: const u8[:]) -> f64[>..], Fault {
    let n = win.len;
    if hop > n {
        var e: f64[>..] = [];
        return e, fault(F_VALUE, tag, "the step is larger than the window, so the transform is not invertible");
    }
    var sq: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        sq.push(win[i] * win[i]);
        i += 1;
    }
    let dd = sig_step_sum(sq, hop);
    var biggest: f64 = 0.0;
    i = 0;
    while i < n {
        if dd[i] > biggest { biggest = dd[i]; }
        i += 1;
    }
    let floor = biggest * 1.0e-15;
    i = 0;
    while i < n {
        if dd[i] < floor {
            var e: f64[>..] = [];
            return e, fault(F_VALUE, tag, "the transform is not invertible for this window and step");
        }
        i += 1;
    }
    var out: f64[>..] = [];
    i = 0;
    while i < n {
        out.push(win[i] / dd[i]);
        i += 1;
    }
    return out, no_fault();
}

// stft_plan holds the plan of a short time transform. The mode is
// "onesided", "twosided" or "centered". The count mfft of zero takes
// the length of the window. The scale is "", "spectrum" or "psd".
// The spectrum scale divides the window by its sum, as the magnitude
// scale of SciPy does.
fn stft_plan(win: f64[:], hop: i64, fs: f64, fft_mode: const u8[:] = "onesided",
             mfft: i64 = 0, scale: const u8[:] = "", phase_shift: f64 = 0.0) -> StftPlan, Fault {
    let tag = "stft_plan";
    if win.len < 1 {
        return stft_plan_empty(), fault(F_SHAPE, tag, "the window holds no value");
    }
    if !all_finite(win) {
        return stft_plan_empty(), fault(F_VALUE, tag, "the window holds a value that is not finite");
    }
    if hop < 1 || hop > MAX_DSP_LENGTH {
        return stft_plan_empty(), fault(F_RANGE, tag, "the step must be one or more");
    }
    if !(fs - fs == 0.0) || fs <= 0.0 {
        return stft_plan_empty(), fault(F_VALUE, tag, "the rate must be above zero");
    }
    if fft_mode != "onesided" && fft_mode != "twosided" && fft_mode != "centered" {
        return stft_plan_empty(), fault(F_VALUE, tag, "the mode holds an unknown name");
    }
    if scale != "" && scale != "spectrum" && scale != "psd" {
        return stft_plan_empty(), fault(F_VALUE, tag, "the scale holds an unknown name");
    }
    var count = mfft;
    if count <= 0 { count = win.len; }
    if count < win.len || count > MAX_DSP_LENGTH {
        return stft_plan_empty(), fault(F_RANGE, tag, "the length of the transform must hold the length of the window or more");
    }
    var w: f64[>..] = [];
    var i: i64 = 0;
    while i < win.len { w.push(win[i]); i += 1; }
    // The scale of the spectrum divides the window by its sum. The
    // scale of the power divides it by the root of its power over the
    // step of the time.
    if scale == "spectrum" {
        var s: f64 = 0.0;
        i = 0;
        while i < w.len { s += w[i]; i += 1; }
        if s == 0.0 {
            return stft_plan_empty(), fault(F_VALUE, tag, "the sum of the window is zero, so the scale of the spectrum fails");
        }
        i = 0;
        while i < w.len { w[i] = w[i] / mathshim::abs_of(s); i += 1; }
    } else if scale == "psd" {
        var s: f64 = 0.0;
        i = 0;
        while i < w.len { s += w[i] * w[i]; i += 1; }
        if s <= 0.0 {
            return stft_plan_empty(), fault(F_VALUE, tag, "the power of the window is zero, so the scale of the power fails");
        }
        let fac = 1.0 / mathshim::root(s * fs);
        i = 0;
        while i < w.len { w[i] = w[i] * fac; i += 1; }
    }
    let dual, f = stft_dual_window(w[..], hop, tag);
    if f.kind != F_NONE { return stft_plan_empty(), f; }
    var buf: f64[>..] = [];
    var i2: i64 = 0;
    while i2 < w.len { buf.push(w[i2]); i2 += 1; }
    i2 = 0;
    while i2 < dual.len { buf.push(dual[i2]); i2 += 1; }
    return StftPlan { hop: hop, fs: fs, fft_mode: fft_mode, mfft: count,
                      scale_to: scale, phase_shift: phase_shift, win_len: w.len, buf: buf }, no_fault();
}

// stft_pre_padding gives the smallest slice of a plan. The zeroth
// slice stands over the zeroth sample of the signal.
fn stft_pre_padding(plan: StftPlan&) -> i64 {
    let m = plan.win_len;
    let mid = m / 2;
    let n0 = -mid;
    var p: i64 = 0;
    var n_ = n0;
    while n_ > n0 - m - 1 {
        let next = n_ - plan.hop;
        if next + m <= 0 || stft_all_zero(plan.buf[..], next) {
            return -p;
        }
        n_ -= plan.hop;
        p += 1;
    }
    return 0;
}

// stft_post_padding gives the largest slice of a plan for a signal of
// n samples.
fn stft_post_padding(plan: StftPlan&, n: i64) -> i64, Fault {
    let m = plan.win_len;
    let mid = m / 2;
    if n < m - mid {
        return 0, fault(F_SHAPE, "stft_forward", "the signal holds fewer values than half of the window");
    }
    let q1 = n / plan.hop;
    let k1 = q1 * plan.hop - mid;
    var q = q1;
    var k_ = k1;
    while k_ < n + m {
        let next = k_ + plan.hop;
        if next >= n || stft_all_zero_head(plan.buf[..], n - next) {
            return q + 1, no_fault();
        }
        k_ += plan.hop;
        q += 1;
    }
    return 0, fault(F_LIMIT, "stft_forward", "the slices of the signal did not end");
}

// stft_fft gives the transform of one windowed part of a signal. The
// part holds the count of the window. The call writes the values at
// the place off of the buffers of the result. The values of one part
// stand stride places apart, since a part holds one column of the
// matrix of the transform.
fn stft_fft(plan: StftPlan&, part_re: f64[:], part_im: f64[:], re: f64[>..]&,
            im: f64[>..]&, off: i64, stride: i64) -> Fault {
    let m = plan.win_len;
    let n = plan.mfft;
    var pr: f64[>..] = [];
    var pi: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        if i < part_re.len {
            pr.push(part_re[i]);
            pi.push(part_im[i]);
        } else {
            pr.push(0.0);
            pi.push(0.0);
        }
        i += 1;
    }
    // The roll of the phase moves the values of the part before the
    // transform, as SciPy does.
    if plan.phase_shift >= 0.0 {
        let ps = ((plan.phase_shift as i64) + m / 2) % m;
        if ps != 0 {
            var sr: f64[>..] = [];
            var si: f64[>..] = [];
            stft_shift(pr[..], -ps, sr);
            stft_shift(pi[..], -ps, si);
            pr = copy(sr);
            pi = copy(si);
        }
    }
    if plan.fft_mode == "onesided" {
        let or_, oi, f = rfft(pr[..]);
        if f.kind != F_NONE { return f; }
        i = 0;
        while i < or_.data.len {
            re[off + i * stride] = or_.data[i];
            im[off + i * stride] = oi.data[i];
            i += 1;
        }
        return no_fault();
    }
    let or_, oi, f = fft_parts(pr[..], pi[..]);
    if f.kind != F_NONE { return f; }
    if plan.fft_mode == "twosided" {
        i = 0;
        while i < n {
            re[off + i * stride] = or_.data[i];
            im[off + i * stride] = oi.data[i];
            i += 1;
        }
        return no_fault();
    }
    // The mode of the centre moves the zero frequency to the middle.
    var sr: f64[>..] = [];
    var si: f64[>..] = [];
    stft_shift(or_.data[..], (n + 1) / 2, sr);
    stft_shift(oi.data[..], (n + 1) / 2, si);
    i = 0;
    while i < n {
        re[off + i * stride] = sr[i];
        im[off + i * stride] = si[i];
        i += 1;
    }
    return no_fault();
}

// stft_ifft gives the part of a signal from one slice of a transform.
// The result holds the count of the window. The mode of two sides
// gives the real part of the signal.
fn stft_ifft(plan: StftPlan&, sre: f64[:], sim: f64[:], out: f64[>..]&) -> Fault {
    let m = plan.win_len;
    let n = plan.mfft;
    var xr: f64[>..] = [];
    var i: i64 = 0;
    if plan.fft_mode == "onesided" {
        let or_, f = irfft(sre, sim, n);
        if f.kind != F_NONE { return f; }
        i = 0;
        while i < n { xr.push(or_.data[i]); i += 1; }
    } else {
        var sr: f64[>..] = [];
        var si: f64[>..] = [];
        if plan.fft_mode == "centered" {
            stft_shift(sre, n / 2, sr);
            stft_shift(sim, n / 2, si);
        } else {
            i = 0;
            while i < n { sr.push(sre[i]); si.push(sim[i]); i += 1; }
        }
        let or_, oi, f = ifft(sr[..], si[..]);
        if f.kind != F_NONE { return f; }
        i = 0;
        while i < n { xr.push(or_.data[i]); i += 1; }
    }
    if plan.phase_shift >= 0.0 {
        let ps = ((plan.phase_shift as i64) + m / 2) % m;
        if ps != 0 {
            var sr: f64[>..] = [];
            stft_shift(xr[..], ps, sr);
            xr = copy(sr);
        }
    }
    i = 0;
    while i < m {
        out.push(xr[i] * plan.buf[plan.win_len + i]);
        i += 1;
    }
    return no_fault();
}

// stft_forward gives the short time transform of a signal. The axis
// names the direction of the signal and must be the last one. The
// result holds the real and the imaginary parts of the matrix of the
// transform side by side, with one row per frequency and one column
// per part of the signal.
fn stft_forward(plan: StftPlan&, x, axis: i64 = -1) -> CArr, Fault {
    let tag = "stft_forward";
    let fs = check_stack(x.ndim, x.shape, x.data.len, 1, tag);
    if fs.kind != F_NONE { return carr_empty(), fs; }
    if x.ndim < 1 {
        return carr_empty(), fault(F_SHAPE, tag, "the signal holds no dimension");
    }
    var ax = axis;
    if ax < 0 { ax = ax + x.ndim; }
    if ax != x.ndim - 1 {
        return carr_empty(), fault(F_RANGE, tag, "the axis must name the last direction");
    }
    let n = x.shape[x.ndim - 1];
    if !all_finite(x.data) {
        return carr_empty(), fault(F_VALUE, tag, "the signal holds a value that is not finite");
    }
    let m = plan.win_len;
    if n < m - m / 2 {
        return carr_empty(), fault(F_SHAPE, tag, "the signal holds fewer values than half of the window");
    }
    let pmax, fp = stft_post_padding(plan, n);
    if fp.kind != F_NONE { return carr_empty(), fp; }
    let qmin = stft_pre_padding(plan);
    let slices = pmax - qmin;
    let count = plan.mfft;
    var bins = count;
    if plan.fft_mode == "onesided" { bins = count / 2 + 1; }
    var rows: i64 = 1;
    var i: i64 = 0;
    while i < x.ndim - 1 { rows *= x.shape[i]; i += 1; }
    if rows * bins * slices > MAX_ELEMENTS {
        return carr_empty(), fault(F_LIMIT, tag, "the transform holds more elements than the limit");
    }
    var dre: f64[>..] = [];
    var dim: f64[>..] = [];
    i = 0;
    while i < rows * bins * slices { dre.push(0.0); dim.push(0.0); i += 1; }
    var r: i64 = 0;
    while r < rows {
        var s: i64 = 0;
        while s < slices {
            let start = (qmin + s) * plan.hop - m / 2;
            var pr: f64[>..] = [];
            var j: i64 = 0;
            while j < m {
                let at = start + j;
                var v: f64 = 0.0;
                if at >= 0 && at < n { v = x.data[r * n + at]; }
                pr.push(v * plan.buf[j]);
                j += 1;
            }
            var pi: f64[>..] = [];
            j = 0;
            while j < m { pi.push(0.0); j += 1; }
            let off = r * bins * slices + s;
            let f = stft_fft(plan, pr[..], pi[..], dre, dim, off, slices);
            if f.kind != F_NONE { return carr_empty(), f; }
            s += 1;
        }
        r += 1;
    }
    var shape: i64[..8] = [];
    i = 0;
    while i < x.ndim - 1 { shape.push(x.shape[i]); i += 1; }
    shape.push(bins);
    shape.push(slices);
    // The values of the two parts stand side by side, as CArr holds
    // them. The call folds the two buffers into one.
    var data: f64[>..] = [];
    i = 0;
    while i < dre.len {
        data.push(dre[i]);
        data.push(dim[i]);
        i += 1;
    }
    return carr_make(shape, copy(data)), no_fault();
}

// stft_inverse gives the signal of a short time transform. The axis
// names the direction of the signal in the result and must be the
// last one. The transform must come from stft_forward of the same
// plan. The mode of one side gives the real signal. The modes of two
// sides give the real part of the signal.
fn stft_inverse(plan: StftPlan&, s, axis: i64 = -1) -> Arr, Fault {
    let tag = "stft_inverse";
    let fs = check_stack(s.ndim, s.shape, s.data.len, 2, tag);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    if s.ndim < 2 {
        return empty_arr(), fault(F_SHAPE, tag, "the transform must hold two dimensions or more");
    }
    var ax = axis;
    if ax < 0 { ax = ax + s.ndim; }
    if ax != s.ndim - 1 {
        return empty_arr(), fault(F_RANGE, tag, "the axis must name the last direction");
    }
    let bins = s.shape[s.ndim - 2];
    let slices = s.shape[s.ndim - 1];
    let m = plan.win_len;
    let count = plan.mfft;
    var want = count;
    if plan.fft_mode == "onesided" { want = count / 2 + 1; }
    if bins != want {
        return empty_arr(), fault(F_SHAPE, tag, "the transform holds another count of frequencies than the plan");
    }
    let pmin = stft_pre_padding(plan);
    let qmax = slices + pmin;
    let k1 = (qmax - 1) * plan.hop + m - m / 2;
    let q1raw, fp = stft_post_padding(plan, k1);
    if fp.kind != F_NONE { return empty_arr(), fp; }
    var q1 = q1raw;
    if qmax < q1 { q1 = qmax; }
    // The nearest place of the signal above k1 that a slice covers.
    var kq1 = k1;
    if k1 % plan.hop != 0 { kq1 = (k1 / plan.hop + 1) * plan.hop; }
    let npts = kq1 + m - m / 2;
    if npts > MAX_DSP_LENGTH {
        return empty_arr(), fault(F_LIMIT, tag, "the signal would be too long");
    }
    var rows: i64 = 1;
    var i: i64 = 0;
    while i < s.ndim - 2 { rows *= s.shape[i]; i += 1; }
    var dr: f64[>..] = [];
    var r: i64 = 0;
    while r < rows {
        var x: f64[>..] = [];
        i = 0;
        while i < npts { x.push(0.0); i += 1; }
        var q = pmin;
        while q < q1 {
            var sre: f64[>..] = [];
            var sim: f64[>..] = [];
            let base = 2 * (r * bins * slices + q - pmin);
            var j: i64 = 0;
            while j < bins {
                sre.push(s.data[base + 2 * j * slices]);
                sim.push(s.data[base + 2 * j * slices + 1]);
                j += 1;
            }
            var xs: f64[>..] = [];
            let f = stft_ifft(plan, sre[..], sim[..], xs);
            if f.kind != F_NONE { return empty_arr(), f; }
            var i0 = q * plan.hop - m / 2;
            var i1 = i0 + m;
            if npts < i1 { i1 = npts; }
            var j0: i64 = 0;
            if i0 < 0 {
                j0 = -i0;
                i0 = 0;
            }
            var t: i64 = 0;
            while t < i1 - i0 {
                x[i0 + t] += xs[j0 + t];
                t += 1;
            }
            q += 1;
        }
        i = 0;
        while i < k1 { dr.push(x[i]); i += 1; }
        r += 1;
    }
    var shape: i64[..8] = [];
    i = 0;
    while i < s.ndim - 2 { shape.push(s.shape[i]); i += 1; }
    shape.push(k1);
    return arr_of(copy(dr), shape), no_fault();
}
