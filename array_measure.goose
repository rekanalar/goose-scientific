// The measurements of labeled regions, as in scipy.ndimage: sum_labels,
// mean_labels, variance_labels, standard_deviation_labels, minimum_labels,
// maximum_labels, median_labels, minimum_position, maximum_position,
// extrema, center_of_mass, histogram_labels, find_objects and
// labeled_comprehension.
//
// A labels array holds whole numbers and has the shape of the data. An
// index list names the labels to measure. The result holds one entry for
// each index entry, in the order of the list, and a repeated entry repeats
// its result. An empty index list stands for None: the result holds one
// entry for all values with a label above zero. An empty labels array
// stands for None: all values form one region, and the index list must be
// empty. A label that is absent from the labels array gives 0.0 for the
// sums, the extrema and the median, 0 for a position, and NaN for the mean,
// the variance and the center.
//
// The calls minimum_labels, maximum_labels, median_labels and the positions
// treat NaN as the largest value, so the result does not depend on the order
// of the values. SciPy gives a result there that depends on the path of the
// call. A tie for a position keeps the first place.
namespace scientific;

import mathshim;

// ms_sort_index sorts the positions in ord by the values of idx. It is a
// heap sort, so no input gives a slow case.
fn ms_sift(ord: i64[>..]&, idx: i64[:], start: i64, end: i64) {
    var root = start;
    var go = true;
    while go {
        var child = 2 * root + 1;
        if child >= end {
            go = false;
        } else {
            if child + 1 < end && idx[ord[child]] < idx[ord[child + 1]] { child += 1; }
            if idx[ord[root]] < idx[ord[child]] {
                let t = ord[root];
                ord[root] = ord[child];
                ord[child] = t;
                root = child;
            } else {
                go = false;
            }
        }
    }
}

// ms_prepare reads the labels and the index. It gives the slot of each value
// in gid, or -1. It gives the slot of each index entry in slot. The count of
// slots is the count of distinct index entries, or 1 for no index.
fn ms_prepare(a, labels, index: i64[:], tag: const u8[:], gid: i64[>..]&, slot: i64[>..]&) -> i64, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fa.kind != F_NONE { return 0, fa; }
    let n = a.data.len;
    if index.len > MAX_ELEMENTS {
        return 0, fault(F_LIMIT, tag, "the index list holds more entries than the limit");
    }
    var i: i64 = 0;
    if labels.ndim == 0 {
        if index.len != 0 {
            return 0, fault(F_VALUE, tag, "an index list needs a labels array");
        }
        while i < n { gid.push(0); i += 1; }
        slot.push(0);
        return 1, no_fault();
    }
    let fl = check_stack(labels.ndim, labels.shape, labels.data.len, 1, tag);
    if fl.kind != F_NONE { return 0, fl; }
    if labels.ndim != a.ndim || labels.data.len != n {
        return 0, fault(F_SHAPE, tag, "the labels hold another shape than the data");
    }
    var d: i64 = 0;
    while d < a.ndim {
        if labels.shape[d] != a.shape[d] { return 0, fault(F_SHAPE, tag, "the labels hold another shape than the data"); }
        d += 1;
    }
    var lab: i64[>..] = [];
    i = 0;
    while i < n {
        let x = labels.data[i];
        if !(x >= -9.0e18 && x <= 9.0e18) || (x as i64) as f64 != x {
            return 0, fault(F_VALUE, tag, "the labels hold a value that is not a whole number");
        }
        lab.push(x as i64);
        i += 1;
    }
    if index.len == 0 {
        i = 0;
        while i < n { if lab[i] > 0 { gid.push(0); } else { gid.push(-1); } i += 1; }
        slot.push(0);
        return 1, no_fault();
    }
    var ord: i64[>..] = [];
    i = 0;
    while i < index.len { ord.push(i); i += 1; }
    var s = index.len / 2 - 1;
    while s >= 0 { ms_sift(ord, index, s, index.len); s -= 1; }
    var end = index.len - 1;
    while end > 0 {
        let t = ord[0];
        ord[0] = ord[end];
        ord[end] = t;
        ms_sift(ord, index, 0, end);
        end -= 1;
    }
    var keys: i64[>..] = [];
    i = 0;
    while i < index.len { slot.push(0); i += 1; }
    i = 0;
    while i < index.len {
        let v = index[ord[i]];
        if keys.len == 0 || keys[keys.len - 1] != v { keys.push(v); }
        slot[ord[i]] = keys.len - 1;
        i += 1;
    }
    i = 0;
    while i < n {
        var lo: i64 = 0;
        var hi: i64 = keys.len - 1;
        var found: i64 = -1;
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            if keys[mid] == lab[i] { found = mid; lo = hi + 1; } else if keys[mid] < lab[i] { lo = mid + 1; } else { hi = mid - 1; }
        }
        gid.push(found);
        i += 1;
    }
    return keys.len, no_fault();
}

// ms_out gives the result of one value for each slot as an array with one
// entry for each index entry.
fn ms_out(res: f64[:], slot: i64[:], width: i64, rows: bool = false) -> Arr {
    var data: f64[>..] = [];
    var j: i64 = 0;
    while j < slot.len {
        var w: i64 = 0;
        while w < width { data.push(res[slot[j] * width + w]); w += 1; }
        j += 1;
    }
    if width == 1 && !rows { return arr_of(copy(data), [slot.len]); }
    return arr_of(copy(data), [slot.len, width]);
}

// ms_sums gives the count and the sum of each slot.
fn ms_sums(a, gid: i64[:], ns: i64, cnt: f64[>..]&, sum: f64[>..]&) {
    var p: i64 = 0;
    while p < a.data.len {
        if gid[p] >= 0 { cnt[gid[p]] += 1.0; sum[gid[p]] += a.data[p]; }
        p += 1;
    }
}

// ms_basic gives the sum of the values of each slot. The kind 1 divides the
// sum by the count, so it gives the mean.
fn ms_basic(a, labels, index: i64[:], kind: i64, tag: const u8[:]) -> Arr, Fault {
    var gid: i64[>..] = [];
    var slot: i64[>..] = [];
    let ns, f = ms_prepare(a, labels, index, tag, gid, slot);
    if f.kind != F_NONE { return empty_arr(), f; }
    var cnt: f64[>..] = nd_zeros(ns);
    var sum: f64[>..] = nd_zeros(ns);
    ms_sums(a, gid, ns, cnt, sum);
    if kind == 1 {
        var s: i64 = 0;
        while s < ns { sum[s] = sum[s] / cnt[s]; s += 1; }
    }
    return ms_out(sum, slot, 1), no_fault();
}

// sum_labels gives the sum of the values of each label.
fn sum_labels(a, labels, index: i64[:] = []) -> Arr, Fault {
    return ms_basic(a, labels, index, 0, "sum_labels");
}

// mean_labels gives the mean of the values of each label. A label without
// a value gives NaN.
fn mean_labels(a, labels, index: i64[:] = []) -> Arr, Fault {
    return ms_basic(a, labels, index, 1, "mean_labels");
}

// ms_variance gives the variance of each slot. The kind 1 gives the root.
fn ms_variance(a, labels, index: i64[:], root: bool, tag: const u8[:]) -> Arr, Fault {
    var gid: i64[>..] = [];
    var slot: i64[>..] = [];
    let ns, f = ms_prepare(a, labels, index, tag, gid, slot);
    if f.kind != F_NONE { return empty_arr(), f; }
    var cnt: f64[>..] = nd_zeros(ns);
    var sum: f64[>..] = nd_zeros(ns);
    ms_sums(a, gid, ns, cnt, sum);
    var s: i64 = 0;
    while s < ns { sum[s] = sum[s] / cnt[s]; s += 1; }
    var sq: f64[>..] = nd_zeros(ns);
    var p: i64 = 0;
    while p < a.data.len {
        if gid[p] >= 0 {
            let dv = a.data[p] - sum[gid[p]];
            sq[gid[p]] += dv * dv;
        }
        p += 1;
    }
    s = 0;
    while s < ns {
        sq[s] = sq[s] / cnt[s];
        if root { sq[s] = mathshim::root(sq[s]); }
        s += 1;
    }
    return ms_out(sq, slot, 1), no_fault();
}

// variance_labels gives the variance of the values of each label, with the
// divisor of the count.
fn variance_labels(a, labels, index: i64[:] = []) -> Arr, Fault {
    return ms_variance(a, labels, index, false, "variance_labels");
}

// standard_deviation_labels gives the root of the variance of each label.
fn standard_deviation_labels(a, labels, index: i64[:] = []) -> Arr, Fault {
    return ms_variance(a, labels, index, true, "standard_deviation_labels");
}

// ms_extreme finds the smallest and the largest value of each slot. A tie
// keeps the first place in row-major order. NaN is the largest value.
fn ms_extreme(a, gid: i64[:], ns: i64, mn: f64[>..]&, mx: f64[>..]&, pmn: i64[>..]&, pmx: i64[>..]&) {
    var seen: i64[>..] = [];
    var s: i64 = 0;
    while s < ns { seen.push(0); pmn.push(0); pmx.push(0); s += 1; }
    var p: i64 = 0;
    while p < a.data.len {
        let g = gid[p];
        if g >= 0 {
            let x = a.data[p];
            if seen[g] == 0 {
                seen[g] = 1;
                mn[g] = x; mx[g] = x; pmn[g] = p; pmx[g] = p;
            } else {
                if nd_less(x, mn[g]) { mn[g] = x; pmn[g] = p; }
                if nd_less(mx[g], x) { mx[g] = x; pmx[g] = p; }
            }
        }
        p += 1;
    }
}

// ms_positions turns flat places into coordinates. It gives one row for
// each slot.
fn ms_positions(shape: i64[:], flat: i64[:], ns: i64) -> f64[>..] {
    let nd = shape.len;
    var out: f64[>..] = nd_zeros(ns * nd);
    var s: i64 = 0;
    while s < ns {
        var rem = flat[s];
        var d: i64 = nd - 1;
        while d >= 0 { out[s * nd + d] = (rem % shape[d]) as f64; rem /= shape[d]; d -= 1; }
        s += 1;
    }
    return out;
}

// ms_extreme_call runs the extremes. The kind is 0 for the minimum, 1 for
// the maximum, 2 for the place of the minimum and 3 for the place of the
// maximum.
fn ms_extreme_call(a, labels, index: i64[:], kind: i64, tag: const u8[:]) -> Arr, Fault {
    var gid: i64[>..] = [];
    var slot: i64[>..] = [];
    let ns, f = ms_prepare(a, labels, index, tag, gid, slot);
    if f.kind != F_NONE { return empty_arr(), f; }
    var mn: f64[>..] = nd_zeros(ns);
    var mx: f64[>..] = nd_zeros(ns);
    var pmn: i64[>..] = [];
    var pmx: i64[>..] = [];
    ms_extreme(a, gid, ns, mn, mx, pmn, pmx);
    if kind == 0 { return ms_out(mn, slot, 1), no_fault(); }
    if kind == 1 { return ms_out(mx, slot, 1), no_fault(); }
    if slot.len > MAX_ELEMENTS / a.ndim {
        return empty_arr(), fault(F_LIMIT, tag, "the result holds more elements than the limit");
    }
    var pos: f64[>..] = [];
    if kind == 2 { pos = ms_positions(a.shape, pmn, ns); } else { pos = ms_positions(a.shape, pmx, ns); }
    return ms_out(pos, slot, a.ndim, true), no_fault();
}

// minimum_labels gives the smallest value of each label.
fn minimum_labels(a, labels, index: i64[:] = []) -> Arr, Fault {
    return ms_extreme_call(a, labels, index, 0, "minimum_labels");
}

// maximum_labels gives the largest value of each label.
fn maximum_labels(a, labels, index: i64[:] = []) -> Arr, Fault {
    return ms_extreme_call(a, labels, index, 1, "maximum_labels");
}

// minimum_position gives the first place of the smallest value of each label,
// one row of coordinates for each index entry.
fn minimum_position(a, labels, index: i64[:] = []) -> Arr, Fault {
    return ms_extreme_call(a, labels, index, 2, "minimum_position");
}

// maximum_position gives the first place of the largest value of each label.
fn maximum_position(a, labels, index: i64[:] = []) -> Arr, Fault {
    return ms_extreme_call(a, labels, index, 3, "maximum_position");
}

// extrema gives the smallest value, the largest value and their places.
fn extrema(a, labels, index: i64[:] = []) -> Arr, Arr, Arr, Arr, Fault {
    let mn, f1 = minimum_labels(a, labels, index);
    if f1.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), f1; }
    let mx, f2 = maximum_labels(a, labels, index);
    let pn, f3 = minimum_position(a, labels, index);
    let px, f4 = maximum_position(a, labels, index);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), f2; }
    if f3.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), f3; }
    if f4.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), f4; }
    return mn, mx, pn, px, no_fault();
}

// ms_grouped gathers the values of every slot into one buffer. The start of
// slot s is off[s], and it holds cnt[s] values.
fn ms_grouped(a, gid: i64[:], ns: i64, off: i64[>..]&, cnt: i64[>..]&, vals: f64[>..]&) {
    var s: i64 = 0;
    while s < ns { cnt.push(0); s += 1; }
    var p: i64 = 0;
    while p < a.data.len { if gid[p] >= 0 { cnt[gid[p]] += 1; } p += 1; }
    var run: i64 = 0;
    s = 0;
    while s < ns { off.push(run); run += cnt[s]; s += 1; }
    var fill: i64[>..] = [];
    s = 0;
    while s < ns { fill.push(0); s += 1; }
    p = 0;
    while p < run { vals.push(0.0); p += 1; }
    p = 0;
    while p < a.data.len {
        let g = gid[p];
        if g >= 0 { vals[off[g] + fill[g]] = a.data[p]; fill[g] += 1; }
        p += 1;
    }
}

// median_labels gives the median of the values of each label. An even count
// gives the mean of the two middle values.
fn median_labels(a, labels, index: i64[:] = []) -> Arr, Fault {
    var gid: i64[>..] = [];
    var slot: i64[>..] = [];
    let ns, f = ms_prepare(a, labels, index, "median_labels", gid, slot);
    if f.kind != F_NONE { return empty_arr(), f; }
    var off: i64[>..] = [];
    var cnt: i64[>..] = [];
    var vals: f64[>..] = [];
    ms_grouped(a, gid, ns, off, cnt, vals);
    var res: f64[>..] = nd_zeros(ns);
    var s: i64 = 0;
    while s < ns {
        let t = cnt[s];
        if t > 0 {
            var buf: f64[>..] = nd_zeros(t);
            var i: i64 = 0;
            while i < t { buf[i] = vals[off[s] + i]; i += 1; }
            let hi = nd_pick(buf, t, t / 2);
            if t % 2 == 1 {
                res[s] = hi;
            } else {
                i = 0;
                while i < t { buf[i] = vals[off[s] + i]; i += 1; }
                let lo = nd_pick(buf, t, t / 2 - 1);
                res[s] = (lo + hi) / 2.0;
            }
        }
        s += 1;
    }
    return ms_out(res, slot, 1), no_fault();
}

// center_of_mass gives the mean place of each label, weighted by the values.
// It gives one row of coordinates for each index entry. A weight sum of zero
// gives NaN or an infinity, as the division of floats does.
fn center_of_mass(a, labels, index: i64[:] = []) -> Arr, Fault {
    var gid: i64[>..] = [];
    var slot: i64[>..] = [];
    let ns, f = ms_prepare(a, labels, index, "center_of_mass", gid, slot);
    if f.kind != F_NONE { return empty_arr(), f; }
    let nd = a.ndim;
    if slot.len > MAX_ELEMENTS / nd {
        return empty_arr(), fault(F_LIMIT, "center_of_mass", "the result holds more elements than the limit");
    }
    var cnt: f64[>..] = nd_zeros(ns);
    var norm: f64[>..] = nd_zeros(ns);
    ms_sums(a, gid, ns, cnt, norm);
    var acc: f64[>..] = nd_zeros(ns * nd);
    var ix: i64[..8] = [];
    var d: i64 = 0;
    while d < nd { ix.push(0); d += 1; }
    var p: i64 = 0;
    while p < a.data.len {
        let g = gid[p];
        if g >= 0 {
            d = 0;
            while d < nd { acc[g * nd + d] += a.data[p] * (ix[d] as f64); d += 1; }
        }
        gf_next(ix, a.shape);
        p += 1;
    }
    var s: i64 = 0;
    while s < ns {
        d = 0;
        while d < nd { acc[s * nd + d] = acc[s * nd + d] / norm[s]; d += 1; }
        s += 1;
    }
    return ms_out(acc, slot, nd, true), no_fault();
}

// histogram_labels counts the values of each label in the bins between the
// lowest and the highest edge. The edges are bins + 1 equal steps. The last
// bin holds its upper edge. A value outside the range, or NaN, falls in no
// bin. A label without a value gives a row of zeros, where SciPy gives None.
// Without an index list the result holds one row of bins, as SciPy gives it.
fn histogram_labels(a, lo: f64, hi: f64, bins: i64, labels, index: i64[:] = []) -> Arr, Fault {
    var gid: i64[>..] = [];
    var slot: i64[>..] = [];
    let ns, f = ms_prepare(a, labels, index, "histogram_labels", gid, slot);
    if f.kind != F_NONE { return empty_arr(), f; }
    if bins < 1 { return empty_arr(), fault(F_VALUE, "histogram_labels", "the count of bins is below one"); }
    if !(lo - lo == 0.0) || !(hi - hi == 0.0) || lo > hi {
        return empty_arr(), fault(F_VALUE, "histogram_labels", "the range is not finite or its low edge is above its high edge");
    }
    if ns > MAX_ELEMENTS / bins || slot.len > MAX_ELEMENTS / bins {
        return empty_arr(), fault(F_LIMIT, "histogram_labels", "the result holds more elements than the limit");
    }
    var edges: f64[>..] = [];
    let step = (hi - lo) / (bins as f64);
    if !(step - step == 0.0) { return empty_arr(), fault(F_VALUE, "histogram_labels", "the width of a bin is not finite"); }
    var e: i64 = 0;
    while e < bins { edges.push(lo + (e as f64) * step); e += 1; }
    edges.push(hi);
    var res: f64[>..] = nd_zeros(ns * bins);
    var p: i64 = 0;
    while p < a.data.len {
        let g = gid[p];
        let x = a.data[p];
        if g >= 0 && x >= edges[0] && x <= edges[bins] {
            var b: i64 = -1;
            if x == edges[bins] {
                b = bins - 1;
            } else {
                var l2: i64 = 0;
                var h2: i64 = bins;
                while l2 < h2 {
                    let mid = l2 + (h2 - l2) / 2;
                    if edges[mid] <= x { l2 = mid + 1; } else { h2 = mid; }
                }
                b = l2 - 1;
            }
            if b >= 0 { res[g * bins + b] += 1.0; }
        }
        p += 1;
    }
    // Without an index list the result is one row of bins, as SciPy gives it.
    if index.len == 0 {
        var flat: f64[>..] = [];
        var b: i64 = 0;
        while b < bins { flat.push(res[b]); b += 1; }
        return arr_of(copy(flat), [bins]), no_fault();
    }
    return ms_out(res, slot, bins, true), no_fault();
}

// find_objects gives the bounding box of each label from 1 up to the largest
// label, or up to max_label when it is above zero. One row holds the start
// of each axis and then the stop of each axis. A label without a value gives
// a row of -1, where SciPy gives None. Labels below one are ignored.
fn find_objects(labels, max_label: i64 = 0) -> Arr, Fault {
    let fl = check_stack(labels.ndim, labels.shape, labels.data.len, 1, "find_objects");
    if fl.kind != F_NONE { return empty_arr(), fl; }
    let nd = labels.ndim;
    let n = labels.data.len;
    var top: i64 = 0;
    var p: i64 = 0;
    while p < n {
        let x = labels.data[p];
        if !(x >= -9.0e18 && x <= 9.0e18) || (x as i64) as f64 != x {
            return empty_arr(), fault(F_VALUE, "find_objects", "the labels hold a value that is not a whole number");
        }
        if (x as i64) > top { top = x as i64; }
        p += 1;
    }
    if max_label < 0 { return empty_arr(), fault(F_VALUE, "find_objects", "max_label is below zero"); }
    var m = top;
    if max_label > 0 { m = max_label; }
    if m < 1 { return arr_of([], [0, 2 * nd]), no_fault(); }
    if m > MAX_ELEMENTS / (2 * nd) {
        return empty_arr(), fault(F_LIMIT, "find_objects", "the result holds more elements than the limit");
    }
    var box: i64[>..] = [];
    var q: i64 = 0;
    while q < m * 2 * nd { box.push(-1); q += 1; }
    var ix: i64[..8] = [];
    var d: i64 = 0;
    while d < nd { ix.push(0); d += 1; }
    p = 0;
    while p < n {
        let l = labels.data[p] as i64;
        if l >= 1 && l <= m {
            let r = (l - 1) * 2 * nd;
            d = 0;
            while d < nd {
                if box[r + d] < 0 || ix[d] < box[r + d] { box[r + d] = ix[d]; }
                if ix[d] + 1 > box[r + nd + d] { box[r + nd + d] = ix[d] + 1; }
                d += 1;
            }
        }
        gf_next(ix, labels.shape);
        p += 1;
    }
    var out: f64[>..] = [];
    q = 0;
    while q < box.len { out.push(box[q] as f64); q += 1; }
    return arr_of(copy(out), [m, 2 * nd]), no_fault();
}

// labeled_comprehension runs the block over the values of each label. The
// block takes the values as a buffer and gives one number. A label without a
// value gives the default.
fn labeled_comprehension<F>(a, labels, index: i64[:] = [], default: f64 = 0.0) -> Arr, Fault {
    var gid: i64[>..] = [];
    var slot: i64[>..] = [];
    let ns, f = ms_prepare(a, labels, index, "labeled_comprehension", gid, slot);
    if f.kind != F_NONE { return empty_arr(), f; }
    var off: i64[>..] = [];
    var cnt: i64[>..] = [];
    var vals: f64[>..] = [];
    ms_grouped(a, gid, ns, off, cnt, vals);
    var res: f64[>..] = nd_zeros(ns);
    var s: i64 = 0;
    while s < ns {
        let t = cnt[s];
        if t > 0 {
            var buf: f64[>..] = nd_zeros(t);
            var i: i64 = 0;
            while i < t { buf[i] = vals[off[s] + i]; i += 1; }
            res[s] = F(copy(buf));
        } else {
            res[s] = default;
        }
        s += 1;
    }
    return ms_out(res, slot, 1), no_fault();
}
