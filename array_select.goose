// Selection and order: the choice among arrays, the cut by a condition,
// the sort by several keys, the partition by position, the set of the
// values that stand in one array only and the unique functions that
// give the places, the inverse and the counts.
namespace scientific;

// whole_place checks that a value is a whole number that fits a place.
fn whole_place(v: f64) -> bool {
    return v == v && v == mathshim::trunc_of(v) && v < 4503599627370496.0 && v > -4503599627370496.0;
}

// key_less tells whether the element i comes before the element j. The
// keys sit one row after the other, n values per key. The last key is
// the main key, as in NumPy. A NaN comes last.
fn key_less(keys: f64[:], nkeys: i64, n: i64, i: i64, j: i64) -> bool {
    var k = nkeys - 1;
    while k >= 0 {
        let x = keys[k * n + i];
        let y = keys[k * n + j];
        if less_for_set(x, y) { return true; }
        if less_for_set(y, x) { return false; }
        k -= 1;
    }
    return false;
}

// order_places gives the places that sort n elements by their keys. The
// sort is a stable merge sort, so the cost grows as n log n.
fn order_places(keys: f64[:], nkeys: i64, n: i64) -> i64[>..] {
    var src: i64[>..] = [];
    var dst: i64[>..] = [];
    var i: i64 = 0;
    while i < n { src.push(i); dst.push(0); i += 1; }
    var w: i64 = 1;
    while w < n {
        var lo: i64 = 0;
        while lo < n {
            var mid = lo + w;
            if mid > n { mid = n; }
            var hi = lo + 2 * w;
            if hi > n { hi = n; }
            var a = lo;
            var b = mid;
            var o = lo;
            while a < mid && b < hi {
                if key_less(keys, nkeys, n, src[b], src[a]) { dst[o] = src[b]; b += 1; }
                else { dst[o] = src[a]; a += 1; }
                o += 1;
            }
            while a < mid { dst[o] = src[a]; a += 1; o += 1; }
            while b < hi { dst[o] = src[b]; b += 1; o += 1; }
            lo += 2 * w;
        }
        let t = copy(src);
        src = copy(dst);
        dst = copy(t);
        w *= 2;
    }
    return src;
}

fn line_arr(v: f64[>..]) -> Arr {
    var s: i64[..8] = [];
    s.push(v.len);
    return Arr { ndim: 1, shape: s, data: v };
}

// lexsort gives the places that sort by several keys. The matrix holds
// one key per row. The last row is the main key.
fn lexsort(keys) -> Arr, Fault {
    if keys.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "lexsort", "the keys do not hold a matrix");
    }
    let pos = order_places(keys.data, keys.shape[0], keys.shape[1]);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < pos.len { d.push(pos[i]); i += 1; }
    return line_arr(copy(d)), no_fault();
}

// argpartition gives the places that partition every row, along the
// last axis, around the element at the place kth. The place holds the
// element of the sorted row, the places before it hold elements that do
// not exceed it, and the places after it hold elements that are not
// below it. The order inside each side is the stable order.
fn argpartition(a, kth: i64) -> Arr, Fault {
    let w = a.shape[a.ndim - 1];
    var k = kth;
    if k < 0 { k += w; }
    if k < 0 || k >= w {
        return empty_arr(), fault(F_RANGE, "argpartition", "kth stands outside the axis");
    }
    let rows = a.data.len / w;
    var d: f64[>..] = [];
    var r: i64 = 0;
    while r < rows {
        var row: f64[>..] = [];
        var i: i64 = 0;
        while i < w { row.push(a.data[r * w + i]); i += 1; }
        let pos = order_places(row, 1, w);
        i = 0;
        while i < w { d.push(pos[i]); i += 1; }
        r += 1;
    }
    var s: i64[..8] = [];
    var q: i64 = 0;
    while q < a.ndim { s.push(a.shape[q]); q += 1; }
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}

// compress_axis keeps the entries of an axis where the condition holds
// a value that is not zero. A condition shorter than the axis drops the
// entries after it. A condition longer than the axis is a fault. A
// result with no element is a fault.
fn compress_axis(cond: f64[:], a, axis: i64) -> Arr, Fault {
    let ax = fix_axis(axis, a.ndim);
    if ax < 0 {
        return empty_arr(), fault(F_RANGE, "compress", "the axis stands outside the shape");
    }
    let width = a.shape[ax];
    if cond.len > width {
        return empty_arr(), fault(F_SHAPE, "compress", "the condition is longer than the axis");
    }
    var kept: i64 = 0;
    var i: i64 = 0;
    while i < cond.len { if cond[i] != 0.0 { kept += 1; } i += 1; }
    if kept == 0 {
        return empty_arr(), fault(F_VALUE, "compress", "the result holds no element");
    }
    let inner = product_after_of(a.shape, ax);
    let outer = product_before_of(a.shape, ax);
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var k: i64 = 0;
        while k < cond.len {
            if cond[k] != 0.0 {
                var j: i64 = 0;
                while j < inner { d.push(a.data[o * width * inner + k * inner + j]); j += 1; }
            }
            k += 1;
        }
        o += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    s[ax] = kept;
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}

// compress keeps the elements of the flattened array where the
// condition holds a value that is not zero.
fn compress(cond: f64[:], a) -> Arr, Fault {
    let x, f = ravel(a);
    if f.kind != F_NONE { return empty_arr(), f; }
    return compress_axis(cond, x, 0);
}

// extract gives the elements where the condition array holds a value
// that is not zero, in the order of the flattened arrays.
fn extract(cond, a) -> Arr, Fault {
    return compress(cond.data, a);
}

// choose builds an array from the entries of the stacked choices. The
// choices stand along the first axis. The index array holds the shape
// of one choice and tells which choice gives each element. The mode is
// "raise", "wrap" or "clip" for an index outside the choices. The
// package does not broadcast the index.
fn choose(index, choices, mode: const u8[:] = "raise") -> Arr, Fault {
    if mode != "raise" && mode != "wrap" && mode != "clip" {
        return empty_arr(), fault(F_VALUE, "choose", "the mode holds an unknown name");
    }
    if choices.ndim != index.ndim + 1 {
        return empty_arr(), fault(F_SHAPE, "choose", "the choices do not hold one axis more than the index");
    }
    var i: i64 = 0;
    while i < index.ndim {
        if choices.shape[i + 1] != index.shape[i] {
            return empty_arr(), fault(F_SHAPE, "choose", "a choice does not hold the shape of the index");
        }
        i += 1;
    }
    let k = choices.shape[0];
    let n = index.data.len;
    var d: f64[>..] = [];
    i = 0;
    while i < n {
        let v = index.data[i];
        if !whole_place(v) {
            return empty_arr(), fault(F_VALUE, "choose", "an index holds no whole number");
        }
        var c = v as i64;
        if mode == "wrap" {
            c = c % k;
            if c < 0 { c += k; }
        } else if mode == "clip" {
            if c < 0 { c = 0; }
            if c >= k { c = k - 1; }
        } else if c < 0 || c >= k {
            return empty_arr(), fault(F_RANGE, "choose", "an index stands outside the choices");
        }
        d.push(choices.data[c * n + i]);
        i += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < index.ndim { s.push(index.shape[i]); i += 1; }
    return Arr { ndim: index.ndim, shape: s, data: d }, no_fault();
}

// piecewise gives a value for each element by the first rule of the
// conditions that holds, the last one that holds wins, as in NumPy.
// The conditions stand along the first axis and hold the shape of x.
// The values hold one constant per condition, and one more for the
// elements that no condition holds. The default is zero. The package
// takes constants only, not functions.
fn piecewise(x, conds, values: f64[:]) -> Arr, Fault {
    if conds.ndim != x.ndim + 1 {
        return empty_arr(), fault(F_SHAPE, "piecewise", "the conditions do not hold one axis more than x");
    }
    var i: i64 = 0;
    while i < x.ndim {
        if conds.shape[i + 1] != x.shape[i] {
            return empty_arr(), fault(F_SHAPE, "piecewise", "a condition does not hold the shape of x");
        }
        i += 1;
    }
    let k = conds.shape[0];
    if values.len != k && values.len != k + 1 {
        return empty_arr(), fault(F_SHAPE, "piecewise", "the values do not match the conditions");
    }
    let n = x.data.len;
    var d: f64[>..] = [];
    i = 0;
    while i < n {
        var v: f64 = 0.0;
        if values.len == k + 1 { v = values[k]; }
        var c: i64 = 0;
        var any = false;
        while c < k {
            if conds.data[c * n + i] != 0.0 { v = values[c]; any = true; }
            c += 1;
        }
        if !any && values.len == k + 1 { v = values[k]; }
        d.push(v);
        i += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < x.ndim { s.push(x.shape[i]); i += 1; }
    return Arr { ndim: x.ndim, shape: s, data: d }, no_fault();
}

// setxor1d gives the sorted values that stand in one array only. A NaN
// that stands in both arrays stays twice, as NumPy keeps it. A result
// with no element is a fault.
fn setxor1d(a, b) -> Arr, Fault {
    let ua, fa = unique_set(a.data);
    if fa.kind != F_NONE { return empty_arr(), fa; }
    let ub, fb = unique_set(b.data);
    if fb.kind != F_NONE { return empty_arr(), fb; }
    var all: f64[>..] = [];
    var i: i64 = 0;
    while i < ua.data.len { all.push(ua.data[i]); i += 1; }
    i = 0;
    while i < ub.data.len { all.push(ub.data[i]); i += 1; }
    let aux = sort_set(all);
    var d: f64[>..] = [];
    i = 0;
    while i < aux.len {
        // A value stays when it differs from both neighbours.
        let before = i == 0 || aux[i - 1] != aux[i];
        let after = i == aux.len - 1 || aux[i + 1] != aux[i];
        if before && after { d.push(aux[i]); }
        i += 1;
    }
    if d.len == 0 {
        return empty_arr(), fault(F_VALUE, "setxor1d", "the result holds no element");
    }
    return line_arr(copy(d)), no_fault();
}

// unique_all gives the distinct values in order, the place of the
// first appearance of each, the inverse that rebuilds the array from
// the values, and the count of each value. The inverse holds the shape
// of the array. Each NaN counts as its own value, as NumPy counts it.
fn unique_all(a) -> Arr, Arr, Arr, Arr, Fault {
    let n = a.data.len;
    let pos = order_places(a.data, 1, n);
    var vals: f64[>..] = [];
    var first: f64[>..] = [];
    var counts: f64[>..] = [];
    var inv: f64[>..] = [];
    var i: i64 = 0;
    while i < n { inv.push(0.0); i += 1; }
    i = 0;
    while i < n {
        let x = a.data[pos[i]];
        if vals.len == 0 || vals[vals.len - 1] != x {
            vals.push(x);
            first.push(pos[i]);
            counts.push(0.0);
        }
        counts[counts.len - 1] = counts[counts.len - 1] + 1.0;
        inv[pos[i]] = vals.len - 1;
        i += 1;
    }
    var s: i64[..8] = [];
    i = 0;
    while i < a.ndim { s.push(a.shape[i]); i += 1; }
    let iv = Arr { ndim: a.ndim, shape: s, data: copy(inv) };
    return line_arr(copy(vals)), line_arr(copy(first)), iv, line_arr(copy(counts)), no_fault();
}

// unique_values gives the distinct values in order. Each NaN stays.
fn unique_values(a) -> Arr, Fault {
    let v, p, i, c, f = unique_all(a);
    if f.kind != F_NONE { return empty_arr(), f; }
    return v, no_fault();
}

// unique_counts gives the distinct values in order and the count of
// each.
fn unique_counts(a) -> Arr, Arr, Fault {
    let v, p, i, c, f = unique_all(a);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    return v, c, no_fault();
}

// unique_inverse gives the distinct values in order and the inverse
// that rebuilds the array from them.
fn unique_inverse(a) -> Arr, Arr, Fault {
    let v, p, i, c, f = unique_all(a);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    return v, i, no_fault();
}
