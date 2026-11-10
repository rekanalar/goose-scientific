// The sorting, the search and the order statistics.
//
// The sort is stable and gives a copy. The array of the caller stays
// as it was.
namespace scientific;

import mathshim;

// sort_values orders the values of a slice and gives the positions
// that hold the order. The positions give the sort of the values.
fn sort_values(values: f64[:]) -> Arr, Arr, Fault {
    let n = values.len;
    var pos: f64[>..] = [];
    var i: i64 = 0;
    while i < n { pos.push(i); i += 1; }
    // An insertion sort keeps the order of two equal elements.
    i = 1;
    while i < n {
        let key = pos[i];
        var j = i - 1;
        while j >= 0 {
            let other = pos[j];
            if values[other as i64] > values[key as i64] {
                pos[j + 1] = other;
                j -= 1;
            } else { break; }
        }
        pos[j + 1] = key;
        i += 1;
    }
    var d: f64[>..] = [];
    i = 0;
    while i < n { d.push(values[pos[i] as i64]); i += 1; }
    var s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: d }, Arr { ndim: 1, shape: s, data: pos }, no_fault();
}

// sort orders the elements of an array from the smallest to the
// largest.
fn sort(a) -> Arr, Fault {
    let v, p, f = sort_values(a.data);
    if f.kind != F_NONE { return empty_arr(), f; }
    return v, no_fault();
}

// argsort gives the flat positions that order the elements.
fn argsort(a) -> Arr, Fault {
    let v, p, f = sort_values(a.data);
    if f.kind != F_NONE { return empty_arr(), f; }
    return p, no_fault();
}

// sort_axis orders the elements along one axis.
fn sort_axis(a, axis: i64) -> Arr, Fault {
    if axis < 0 || axis >= a.ndim {
        return empty_arr(), fault(F_RANGE, "sort_axis", "the axis stands outside the shape");
    }
    let width = a.shape[axis];
    let inner = product_after_of(a.shape, axis);
    let outer = product_before_of(a.shape, axis);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { d.push(a.data[i]); i += 1; }
    var o: i64 = 0;
    while o < outer {
        var j: i64 = 0;
        while j < inner {
            // The values of one line, and their places in it.
            var line: f64[>..] = [];
            var k: i64 = 0;
            while k < width {
                line.push(a.data[o * width * inner + k * inner + j]);
                k += 1;
            }
            let sorted, sp, f = sort_values(line);
            if f.kind != F_NONE { return empty_arr(), f; }
            k = 0;
            while k < width {
                d[o * width * inner + k * inner + j] = sorted.data[k];
                k += 1;
            }
            j += 1;
        }
        o += 1;
    }
    return Arr { ndim: a.ndim, shape: same_shape(a), data: d }, no_fault();
}

// unique gives the distinct values in the order of their first
// appearance.
fn unique(a) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        let x = a.data[i];
        var seen = false;
        var j: i64 = 0;
        while j < d.len {
            if d[j] == x { seen = true; break; }
            j += 1;
        }
        if !seen { d.push(x); }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// unique_sorted gives the distinct values in the order of their value.
fn unique_sorted(a) -> Arr, Fault {
    let v, f = sort(a);
    if f.kind != F_NONE { return empty_arr(), f; }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < v.data.len {
        if d.len == 0 || d[d.len - 1] != v.data[i] { d.push(v.data[i]); }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// searchsorted gives the place where a value fits in a sorted array.
// A value that equals an element goes before it, as in NumPy.
fn searchsorted(a, values: f64[:]) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < values.len {
        let x = values[i];
        var lo: i64 = 0;
        var hi = a.data.len;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if a.data[mid] < x { lo = mid + 1; } else { hi = mid; }
        }
        d.push(lo as f64);
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// histogram counts the elements that fall in each of a number of equal
// bins. The bins run from the smallest value to the largest value.
fn histogram(a, bins: i64) -> Arr, Arr, Fault {
    if bins < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "histogram", "the count of bins holds a value below one");
    }
    let lo, f = min(a);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let hi, f2 = max(a);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), f2; }
    if hi == lo {
        return empty_arr(), empty_arr(), fault(F_VALUE, "histogram", "every value holds the same number");
    }
    var counts: f64[>..] = [];
    var i: i64 = 0;
    while i < bins { counts.push(0.0); i += 1; }
    let span = (hi - lo) / bins as f64;
    i = 0;
    while i < a.data.len {
        let k = (a.data[i] - lo) / span;
        var b = k as i64;
        if b >= bins { b = bins - 1; }
        if b < 0 { b = 0; }
        counts[b] += 1.0;
        i += 1;
    }
    var edges: f64[>..] = [];
    i = 0;
    while i <= bins { edges.push(lo + span * i as f64); i += 1; }
    var sc: i64[..8] = [];
    sc.push(bins);
    var se: i64[..8] = [];
    se.push(bins + 1);
    return Arr { ndim: 1, shape: sc, data: counts }, Arr { ndim: 1, shape: se, data: edges }, no_fault();
}

// histogram_edges gives the edges of the bins of a histogram.
fn histogram_edges(a, bins: i64) -> Arr, Fault {
    let c, e, f = histogram(a, bins);
    if f.kind != F_NONE { return empty_arr(), f; }
    return e, no_fault();
}

// bincount counts the occurrences of the whole numbers of an array.
fn bincount(a, minlength: i64) -> Arr, Fault {
    var high: i64 = minlength - 1;
    var i: i64 = 0;
    while i < a.data.len {
        let v = a.data[i];
        if v != v || v < 0.0 || v != mathshim::trunc_of(v) {
            return empty_arr(), fault(F_VALUE, "bincount", "a value holds no whole number");
        }
        let k = v as i64;
        if k > high { high = k; }
        i += 1;
    }
    var d: f64[>..] = [];
    i = 0;
    while i <= high { d.push(0.0); i += 1; }
    i = 0;
    while i < a.data.len {
        let k = a.data[i] as i64;
        d[k] += 1.0;
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// digitize gives the bin of each number. The edges must rise.
fn digitize(values: f64[:], edges: f64[:]) -> Arr, Fault {
    var i: i64 = 1;
    while i < edges.len {
        if edges[i] < edges[i - 1] {
            return empty_arr(), fault(F_VALUE, "digitize", "the edges do not rise");
        }
        i += 1;
    }
    var d: f64[>..] = [];
    i = 0;
    while i < values.len {
        let x = values[i];
        var lo: i64 = 0;
        var hi = edges.len;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if x >= edges[mid] { lo = mid + 1; } else { hi = mid; }
        }
        d.push(lo);
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// median gives the middle value. An even count gives the mean of the
// two middle values, as NumPy does.
fn median(a) -> f64, Fault {
    let v, f = sort(a);
    if f.kind != F_NONE { return 0.0, f; }
    let n = v.data.len;
    if n == 0 { return 0.0, fault(F_VALUE, "median", "the array holds no element"); }
    if n % 2 == 1 { return v.data[n / 2], no_fault(); }
    return (v.data[n / 2 - 1] + v.data[n / 2]) / 2.0, no_fault();
}

// percentile gives a value of a percentage. The percentage lies
// between zero and one hundred.
fn percentile(a, q: f64) -> f64, Fault {
    if q < 0.0 || q > 100.0 {
        return 0.0, fault(F_VALUE, "percentile", "the percentage lies outside the range");
    }
    let v, f = sort(a);
    if f.kind != F_NONE { return 0.0, f; }
    let n = v.data.len;
    if n == 0 { return 0.0, fault(F_VALUE, "percentile", "the array holds no element"); }
    let pos = q / 100.0 * (n - 1) as f64;
    let lo = pos as i64;
    let hi = lo + 1;
    if hi >= n { return v.data[n - 1], no_fault(); }
    let frac = pos - lo as f64;
    return v.data[lo] + (v.data[hi] - v.data[lo]) * frac, no_fault();
}

// roll moves the elements along one axis by a number of places.
fn roll(a, count: i64, axis: i64) -> Arr, Fault {
    if axis < 0 || axis >= a.ndim {
        return empty_arr(), fault(F_RANGE, "roll", "the axis stands outside the shape");
    }
    let width = a.shape[axis];
    if width == 0 { return empty_arr(), fault(F_VALUE, "roll", "the axis holds no element"); }
    let inner = product_after_of(a.shape, axis);
    let outer = product_before_of(a.shape, axis);
    var shift = count % width;
    if shift < 0 { shift += width; }
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var k: i64 = 0;
        while k < width {
            var j: i64 = 0;
            while j < inner {
                let src = o * width * inner + ((k - shift + width) % width) * inner + j;
                d.push(a.data[src]);
                j += 1;
            }
            k += 1;
        }
        o += 1;
    }
    return Arr { ndim: a.ndim, shape: same_shape(a), data: d }, no_fault();
}
