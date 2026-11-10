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

// histogram2d counts the pairs of two arrays in a grid of equal bins.
// The first array gives the row of the grid and the second one gives
// the column. The counts sit in the order of the rows, as NumPy gives
// them. A value on the right edge of the range falls in the last bin.
fn histogram2d(x, y, xbins: i64 = 10, ybins: i64 = 10) -> Arr, Arr, Arr, Fault {
    if xbins < 1 || ybins < 1 {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "histogram2d", "a count of bins holds a value below one");
    }
    if x.ndim != 1 || y.ndim != 1 {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "histogram2d", "an input holds not one dimension");
    }
    if x.data.len != y.data.len {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "histogram2d", "the inputs hold a different count of values");
    }
    let lox, flo = min(x);
    if flo.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), flo; }
    let hix, fhi = max(x);
    if fhi.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), fhi; }
    let loy, flo2 = min(y);
    if flo2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), flo2; }
    let hiy, fhi2 = max(y);
    if fhi2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), fhi2; }
    if hix == lox || hiy == loy {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "histogram2d", "every value of one array holds the same number");
    }
    let spanx = (hix - lox) / xbins as f64;
    let spany = (hiy - loy) / ybins as f64;
    var counts: f64[>..] = [];
    var i: i64 = 0;
    while i < xbins * ybins { counts.push(0.0); i += 1; }
    i = 0;
    while i < x.data.len {
        var bx = ((x.data[i] - lox) / spanx) as i64;
        if bx >= xbins { bx = xbins - 1; }
        if bx < 0 { bx = 0; }
        var by = ((y.data[i] - loy) / spany) as i64;
        if by >= ybins { by = ybins - 1; }
        if by < 0 { by = 0; }
        counts[bx * ybins + by] += 1.0;
        i += 1;
    }
    var sc: i64[..8] = [];
    sc.push(xbins); sc.push(ybins);
    var sx: i64[..8] = [];
    sx.push(xbins + 1);
    var sy: i64[..8] = [];
    sy.push(ybins + 1);
    var ex: f64[>..] = [];
    var ey: f64[>..] = [];
    i = 0;
    while i <= xbins { ex.push(lox + spanx * i as f64); i += 1; }
    i = 0;
    while i <= ybins { ey.push(loy + spany * i as f64); i += 1; }
    return Arr { ndim: 2, shape: sc, data: counts }, Arr { ndim: 1, shape: sx, data: ex }, Arr { ndim: 1, shape: sy, data: ey }, no_fault();
}

// histogramdd counts the rows of points in a grid of equal bins. Each
// row holds one point, and each column holds one dimension. The count
// of bins is the same for every dimension. The counts sit in the order
// of the rows of the grid, so the last dimension runs fastest, as
// NumPy gives them. The edges come back as a grid of one row per
// dimension.
fn histogramdd(points, bins: i64 = 10) -> Arr, Arr, Fault {
    if bins < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "histogramdd", "the count of bins holds a value below one");
    }
    if points.ndim != 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "histogramdd", "the points hold not two dimensions");
    }
    let dims = points.shape[1];
    if dims > MAX_DIMS {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "histogramdd", "the points hold more dimensions than the limit");
    }
    // The counts hold a power of the bins. A count that passes the
    // limit of the elements is a fault.
    var total: i64 = 1;
    var k: i64 = 0;
    while k < dims {
        total *= bins;
        if total > MAX_ELEMENTS {
            return empty_arr(), empty_arr(), fault(F_LIMIT, "histogramdd", "the counts hold more elements than the limit");
        }
        k += 1;
    }
    var lo: f64[>..] = [];
    var hi: f64[>..] = [];
    var span: f64[>..] = [];
    var j: i64 = 0;
    while j < dims {
        var col: f64[>..] = [];
        var i: i64 = 0;
        while i < points.shape[0] { col.push(points.data[i * dims + j]); i += 1; }
        var sh: i64[..8] = [];
        sh.push(col.len);
        let c = Arr { ndim: 1, shape: sh, data: copy(col) };
        let l, fl = min(c);
        if fl.kind != F_NONE { return empty_arr(), empty_arr(), fl; }
        let h, fh = max(c);
        if fh.kind != F_NONE { return empty_arr(), empty_arr(), fh; }
        if h == l {
            return empty_arr(), empty_arr(), fault(F_VALUE, "histogramdd", "every value of one column holds the same number");
        }
        lo.push(l); hi.push(h); span.push((h - l) / bins as f64);
        j += 1;
    }
    var counts: f64[>..] = [];
    var i: i64 = 0;
    while i < total { counts.push(0.0); i += 1; }
    i = 0;
    while i < points.shape[0] {
        var flat: i64 = 0;
        j = 0;
        while j < dims {
            var b = ((points.data[i * dims + j] - lo[j]) / span[j]) as i64;
            if b >= bins { b = bins - 1; }
            if b < 0 { b = 0; }
            flat = flat * bins + b;
            j += 1;
        }
        counts[flat] += 1.0;
        i += 1;
    }
    var sc: i64[..8] = [];
    j = 0;
    while j < dims { sc.push(bins); j += 1; }
    var se: i64[..8] = [];
    se.push(dims); se.push(bins + 1);
    var edges: f64[>..] = [];
    j = 0;
    while j < dims {
        i = 0;
        while i <= bins { edges.push(lo[j] + span[j] * i as f64); i += 1; }
        j += 1;
    }
    return Arr { ndim: sc.len, shape: sc, data: counts }, Arr { ndim: 2, shape: se, data: edges }, no_fault();
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

// interp takes the values between the points of a rising row. A
// point below the first point takes the first value. A point above
// the last point takes the last value. A row of points that does not
// rise is a fault. A row of one point holds that value everywhere.
fn interp(values: f64[:], points: f64[:], xs: f64[:]) -> Arr, Fault {
    if points.len < 1 {
        return empty_arr(), fault(F_VALUE, "interp", "the points hold no element");
    }
    var i: i64 = 1;
    while i < points.len {
        if points[i] < points[i - 1] {
            return empty_arr(), fault(F_VALUE, "interp", "the points do not rise");
        }
        i += 1;
    }
    if points.len == 1 {
        var one: f64[>..] = [];
        i = 0;
        while i < xs.len { one.push(values[0]); i += 1; }
        var s0: i64[..8] = [];
        s0.push(one.len);
        return Arr { ndim: 1, shape: s0, data: one }, no_fault();
    }
    var d: f64[>..] = [];
    i = 0;
    while i < xs.len {
        let x = xs[i];
        if x <= points[0] { d.push(values[0]); }
        else if x >= points[points.len - 1] { d.push(values[points.len - 1]); }
        else {
            var lo: i64 = 0;
            var hi = points.len;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if points[mid] <= x { lo = mid + 1; } else { hi = mid; }
            }
            let k = lo - 1;
            let t = (x - points[k]) / (points[k + 1] - points[k]);
            d.push(values[k] + (values[k + 1] - values[k]) * t);
        }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// gradient gives the derivative of a row of values with a spacing.
// The ends take the difference of the one side. The middle takes the
// difference of the two neighbours, over twice the spacing. A
// spacing of zero is a fault.
fn gradient(a, spacing: f64) -> Arr, Fault {
    if a.ndim != 1 {
        return empty_arr(), fault(F_SHAPE, "gradient", "the array holds a count of dimensions that is not one");
    }
    if spacing == 0.0 {
        return empty_arr(), fault(F_VALUE, "gradient", "the spacing holds zero");
    }
    let n = a.data.len;
    if n < 2 {
        return empty_arr(), fault(F_VALUE, "gradient", "the array holds fewer than two elements");
    }
    var d: f64[>..] = [];
    d.push((a.data[1] - a.data[0]) / spacing);
    var i: i64 = 1;
    while i < n - 1 {
        d.push((a.data[i + 1] - a.data[i - 1]) / (2.0 * spacing));
        i += 1;
    }
    d.push((a.data[n - 1] - a.data[n - 2]) / spacing);
    var s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}
