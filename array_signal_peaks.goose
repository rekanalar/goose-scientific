// The peaks of a signal and the widths of the peaks.
//
// argrelmax, argrelmin and argrelextrema give the places of the local
// maxima, of the local minima and of the local extrema of a signal.
// peak_widths gives the width of each peak at a level of its height.
// ricker gives the wavelet of a width. cwt gives the transform of a
// signal by wavelets of many widths. find_peaks_cwt finds the peaks
// of a signal through that transform.
//
// The formulas follow SciPy 1.18.1 (scipy.signal).
namespace scientific;

import mathshim;

// ricker gives the wavelet of a width. The count points holds the
// count of the values of the wavelet. The wavelet is even and holds
// its middle at the middle of the list.
fn ricker(points: i64, width: f64) -> Arr, Fault {
    if points < 1 {
        return empty_arr(), fault(F_VALUE, "ricker", "the count of the values must hold one value or more");
    }
    if !(width > 0.0) || mathshim::is_finite_of(width) == 0.0 {
        return empty_arr(), fault(F_VALUE, "ricker", "the width must hold a value above zero");
    }
    let amp = 2.0 / (mathshim::root(3.0 * width) * mathshim::root(mathshim::root(3.141592653589793)));
    let wsq = width * width;
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < points {
        let x = i as f64 - (points - 1) as f64 / 2.0;
        let xsq = x * x;
        out.push(amp * (1.0 - xsq / wsq) * mathshim::power_e(-xsq / (2.0 * wsq)));
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// cwt gives the transform of a signal by a wavelet of every width. The
// result holds a row for every width. The rows hold the values of the
// signal after the convolution with the wavelet of that width.
fn cwt(data: f64[:], widths: f64[:]) -> Arr, Fault {
    let tag = "cwt";
    if data.len < 1 || !all_finite(data) {
        return empty_arr(), fault(F_VALUE, tag, "the signal must be a finite list");
    }
    if widths.len < 1 {
        return empty_arr(), fault(F_VALUE, tag, "the widths must hold one value or more");
    }
    var i: i64 = 0;
    while i < widths.len {
        if !(widths[i] > 0.0) {
            return empty_arr(), fault(F_VALUE, tag, "a width must hold a value above zero");
        }
        i += 1;
    }
    var out: f64[>..] = [];
    var r: i64 = 0;
    while r < widths.len {
        let width = widths[r];
        var points = (10.0 * width) as i64;
        if points > data.len { points = data.len; }
        if points < 1 { points = 1; }
        let w, f = ricker(points, width);
        if f.kind != F_NONE { return empty_arr(), f; }
        // The wavelet turns around, so the convolution gives the same
        // values as the convolution with the wavelet.
        var kern: f64[>..] = [];
        var j: i64 = points - 1;
        while j >= 0 { kern.push(w.data[j]); j -= 1; }
        let row, f2 = convolve(data, copy(kern));
        if f2.kind != F_NONE { return empty_arr(), f2; }
        // The row holds the middle of the full result.
        let start = (row.data.len - data.len) / 2;
        j = 0;
        while j < data.len { out.push(row.data[start + j]); j += 1; }
        r += 1;
    }
    var s: i64[..8] = [];
    s.push(widths.len);
    s.push(data.len);
    return Arr { ndim: 2, shape: s, data: out }, no_fault();
}

// argrelmax gives the places of the local maxima of a signal, as SciPy
// argrelmax does. The order holds the count of the places on each
// side that must hold a smaller value. A place of the edge of the
// signal never holds a maximum.
fn argrelmax(x: f64[:], order: i64 = 1) -> Arr, Fault {
    return argrelextrema(x, order, true);
}

// argrelmin gives the places of the local minima of a signal, as SciPy
// argrelmin does. The options follow argrelmax.
fn argrelmin(x: f64[:], order: i64 = 1) -> Arr, Fault {
    return argrelextrema(x, order, false);
}

// argrelextrema gives the places of the local maxima of a signal when
// the flag is true, and the places of the local minima when the flag
// is false. The order holds the count of the places on each side that
// must hold a smaller value.
fn argrelextrema(x: f64[:], order: i64 = 1, maximum: bool = true) -> Arr, Fault {
    let tag = "argrelextrema";
    if x.len < 1 {
        return empty_arr(), fault(F_SHAPE, tag, "the signal must hold one value or more");
    }
    if order < 1 {
        return empty_arr(), fault(F_VALUE, tag, "the order must hold one value or more");
    }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len {
        var ok = true;
        var s: i64 = 1;
        while s <= order && ok {
            var plus = i + s;
            if plus > x.len - 1 { plus = x.len - 1; }
            var minus = i - s;
            if minus < 0 { minus = 0; }
            if maximum {
                if !(x[i] > x[plus] && x[i] > x[minus]) { ok = false; }
            } else {
                if !(x[i] < x[plus] && x[i] < x[minus]) { ok = false; }
            }
            s += 1;
        }
        if ok { out.push(i as f64); }
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// peak_widths gives the width of each peak of a signal, as SciPy
// peak_widths does. The peaks hold the places of the peaks. The
// height holds the part of the prominence of the peak that the width
// is measured at. The call gives the widths, the heights of the
// lines, and the places of the two ends of each line.
fn peak_widths(x: f64[:], peaks: f64[:], rel_height: f64 = 0.5) -> Arr, Arr, Arr, Arr, Fault {
    let tag = "peak_widths";
    if x.len < 1 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the signal must hold one value or more");
    }
    if peaks.len < 1 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the peaks must hold one value or more");
    }
    if rel_height < 0.0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, tag, "the height must hold a value of no minus");
    }
    var widths: f64[>..] = [];
    var heights: f64[>..] = [];
    var lefts: f64[>..] = [];
    var rights: f64[>..] = [];
    var k: i64 = 0;
    while k < peaks.len {
        let pk = peaks[k];
        if !(pk >= 0.0) || pk >= x.len || mathshim::trunc_of(pk) != pk {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_RANGE, tag, "a peak lies outside the signal");
        }
        let p = mathshim::trunc_of(pk) as i64;
        // The bases of the peak lie at the lowest places on each side.
        var i_min: i64 = 0;
        var i_max: i64 = x.len - 1;
        var left = p;
        var lmin = x[p];
        var i: i64 = p;
        while i_min <= i && x[i] <= x[p] {
            if x[i] < lmin { lmin = x[i]; left = i; }
            i -= 1;
        }
        var right = p;
        var rmin = x[p];
        i = p;
        while i <= i_max && x[i] <= x[p] {
            if x[i] < rmin { rmin = x[i]; right = i; }
            i += 1;
        }
        var base = lmin;
        if rmin > base { base = rmin; }
        let prom = x[p] - base;
        let h = x[p] - prom * rel_height;
        // The place of the end on the left, with the line between the
        // two places.
        i = p;
        while left < i && h < x[i] { i -= 1; }
        var li = i as f64;
        if x[i] < h && x[i + 1] != x[i] {
            li += (h - x[i]) / (x[i + 1] - x[i]);
        }
        i = p;
        while i < right && h < x[i] { i += 1; }
        var ri = i as f64;
        if x[i] < h && x[i - 1] != x[i] {
            ri -= (h - x[i]) / (x[i - 1] - x[i]);
        }
        widths.push(ri - li);
        heights.push(h);
        lefts.push(li);
        rights.push(ri);
        k += 1;
    }
    return line_arr(copy(widths)), line_arr(copy(heights)), line_arr(copy(lefts)),
           line_arr(copy(rights)), no_fault();
}

// find_peaks_cwt finds the peaks of a signal through the transform by
// wavelets, as SciPy find_peaks_cwt does. The widths hold the widths
// of the wavelets. The peak lies where a ridge of the transform holds
// a strong value over many widths. The call gives the places of the
// peaks.
fn find_peaks_cwt(vector: f64[:], widths: f64[:], min_snr: f64 = 1.0,
                   noise_perc: f64 = 10.0) -> Arr, Fault {
    let tag = "find_peaks_cwt";
    if vector.len < 1 || !all_finite(vector) {
        return empty_arr(), fault(F_VALUE, tag, "the signal must be a finite list");
    }
    if widths.len < 1 {
        return empty_arr(), fault(F_VALUE, tag, "the widths must hold one value or more");
    }
    var i: i64 = 0;
    while i < widths.len {
        if !(widths[i] > 0.0) {
            return empty_arr(), fault(F_VALUE, tag, "a width must hold a value above zero");
        }
        i += 1;
    }
    if !(noise_perc >= 0.0 && noise_perc <= 100.0) {
        return empty_arr(), fault(F_VALUE, tag, "the part of the noise must hold a value from zero to a hundred");
    }
    let cw, f = cwt(vector, widths);
    if f.kind != F_NONE { return empty_arr(), f; }
    let n = vector.len;
    let rows = widths.len;
    // The places of the maxima of each row of the transform.
    var relmax: bool[>..] = [];
    i = 0;
    while i < rows * n { relmax.push(false); i += 1; }
    var r: i64 = 0;
    while r < rows {
        var c: i64 = 0;
        while c < n {
            var ok = true;
            if c > 0 && cw.data[r * n + c - 1] >= cw.data[r * n + c] { ok = false; }
            if ok && c < n - 1 && cw.data[r * n + c + 1] >= cw.data[r * n + c] { ok = false; }
            if ok { relmax[r * n + c] = true; }
            c += 1;
        }
        r += 1;
    }
    // The row of the transform that holds the highest maxima starts the
    // ridges.
    var start_row: i64 = -1;
    r = 0;
    while r < rows {
        var any = false;
        var c: i64 = 0;
        while c < n {
            if relmax[r * n + c] { any = true; }
            c += 1;
        }
        if any { start_row = r; }
        r += 1;
    }
    if start_row < 0 { return line_arr([]), no_fault(); }
    // The ridges hold the rows and the columns of their places. Each
    // line holds the count of its places, the row and the column of
    // its last place, and the count of the places without a new one.
    var line_count: i64[>..] = [];
    var line_row: i64[>..] = [];
    var line_col: i64[>..] = [];
    var line_gap: i64[>..] = [];
    var line_alive: bool[>..] = [];
    var c: i64 = 0;
    while c < n {
        if relmax[start_row * n + c] {
            line_count.push(1);
            line_row.push(start_row);
            line_col.push(c);
            line_gap.push(0);
            line_alive.push(true);
        }
        c += 1;
    }
    var gap_thresh = mathshim::trunc_of(widths[0]) as i64;
    if widths[0] > gap_thresh as f64 { gap_thresh += 1; }
    r = start_row - 1;
    while r >= 0 {
        // The places of the lines before this row hold the last places
        // of the previous row.
        var prev_col: i64[>..] = [];
        var prev_line: i64[>..] = [];
        var ln: i64 = 0;
        while ln < line_count.len {
            if line_alive[ln] {
                prev_col.push(line_col[ln]);
                prev_line.push(ln);
            }
            ln += 1;
        }
        ln = 0;
        while ln < line_count.len {
            if line_alive[ln] { line_gap[ln] += 1; }
            ln += 1;
        }
        c = 0;
        while c < n {
            if relmax[r * n + c] {
                // The line whose last place lies closest takes this
                // place when the distance is short enough.
                var best: i64 = -1;
                var best_dist: f64 = 0.0;
                ln = 0;
                while ln < prev_col.len {
                    var d = c as f64 - prev_col[ln] as f64;
                    if d < 0.0 { d = -d; }
                    if best < 0 || d < best_dist {
                        best = ln;
                        best_dist = d;
                    }
                    ln += 1;
                }
                var max_dist = widths[r] / 4.0;
                if best >= 0 && best_dist <= max_dist {
                    // The line that takes the place keeps its number.
                    let found = prev_line[best];
                    line_count[found] += 1;
                    line_row[found] = r;
                    line_col[found] = c;
                    line_gap[found] = 0;
                } else {
                    line_count.push(1);
                    line_row.push(r);
                    line_col.push(c);
                    line_gap.push(0);
                    line_alive.push(true);
                }
            }
            c += 1;
        }
        // The lines with too many places without a new one end.
        ln = 0;
        while ln < line_count.len {
            if line_alive[ln] && line_gap[ln] > gap_thresh { line_alive[ln] = false; }
            ln += 1;
        }
        r -= 1;
    }
    // The noise of each place of the first row of the transform.
    var noise: f64[>..] = [];
    let window = n as f64 / 20.0;
    var wsize = mathshim::trunc_of(window) as i64;
    if window > wsize as f64 { wsize += 1; }
    let hf = wsize / 2;
    let odd = wsize % 2;
    c = 0;
    while c < n {
        var start = c - hf;
        if start < 0 { start = 0; }
        var stop = c + hf + odd;
        if stop > n { stop = n; }
        var part: f64[>..] = [];
        var k: i64 = start;
        while k < stop { part.push(cw.data[k]); k += 1; }
        noise.push(sig_percentile(copy(part), noise_perc));
        c += 1;
    }
    // The ridges that are long enough and hold a strong value give the
    // peaks.
    let min_length = rows as f64 / 4.0;
    var min_len = mathshim::trunc_of(min_length) as i64;
    if min_length > min_len as f64 { min_len += 1; }
    var out: f64[>..] = [];
    var ln: i64 = 0;
    while ln < line_count.len {
        if line_count[ln] >= min_len {
            let row0 = line_row[ln];
            let col0 = line_col[ln];
            var snr = cw.data[row0 * n + col0] / noise[col0];
            if snr < 0.0 { snr = -snr; }
            if snr >= min_snr { out.push(col0 as f64); }
        }
        ln += 1;
    }
    // The places of the peaks stand in order.
    var ord = order_places(copy(out), 1, out.len);
    var sorted: f64[>..] = [];
    var o: i64 = 0;
    while o < out.len { sorted.push(out[ord[o]]); o += 1; }
    return line_arr(copy(sorted)), no_fault();
}

// sig_percentile gives the part of the values of a list that lies
// below the given part, as SciPy scoreatpercentile does.
fn sig_percentile(v: f64[>..], part: f64) -> f64 {
    if v.len < 1 { return 0.0 / 0.0; }
    var s = copy(v);
    var i: i64 = 1;
    while i < s.len {
        let key = s[i];
        var j = i - 1;
        while j >= 0 && s[j] > key { s[j + 1] = s[j]; j -= 1; }
        s[j + 1] = key;
        i += 1;
    }
    let pos = part / 100.0 * ((s.len - 1) as f64);
    let lo = mathshim::trunc_of(pos) as i64;
    let hi = lo + 1;
    if hi > s.len - 1 { return s[s.len - 1]; }
    return s[lo] + (pos - lo as f64) * (s[hi] - s[lo]);
}
