// The sets of numbers.
//
// A set holds each value once. The values come in the order of their
// value. A NaN comes after every other value, as NumPy puts it.
namespace scientific;

// less_for_set tells whether a value comes before another in a set. A
// NaN comes after every value, so no value comes before it.
fn less_for_set(a: f64, b: f64) -> bool {
    if a != a { return false; }
    if b != b { return true; }
    return a < b;
}

// same_for_set tells whether two values stand for one value of a set.
// Two NaN stand for one value, as NumPy holds them in a unique array.
fn same_for_set(a: f64, b: f64) -> bool {
    if a != a && b != b { return true; }
    return a == b;
}

// sort_set orders the values of a slice for a set. The sort is stable
// and puts a NaN at the end.
fn sort_set(values: f64[:]) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < values.len { d.push(values[i]); i += 1; }
    i = 1;
    while i < d.len {
        let key = d[i];
        var j = i - 1;
        while j >= 0 {
            if !less_for_set(key, d[j]) { break; }
            d[j + 1] = d[j];
            j -= 1;
        }
        d[j + 1] = key;
        i += 1;
    }
    return d;
}

// unique_set gives the distinct values of an ordered row. Each value
// comes once. Two NaN come once, as NumPy holds them.
fn unique_set(values: f64[:]) -> Arr, Fault {
    let sorted = sort_set(values);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < sorted.len {
        if d.len == 0 || !same_for_set(d[d.len - 1], sorted[i]) { d.push(sorted[i]); }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// holds_value tells whether a sorted row holds a value. A NaN holds no
// value, not even another NaN, as NumPy holds it.
fn holds_value(values: f64[:], x: f64) -> bool {
    if x != x { return false; }
    var lo: i64 = 0;
    var hi = values.len;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if values[mid] < x { lo = mid + 1; } else { hi = mid; }
    }
    if lo < values.len && values[lo] == x { return true; }
    return false;
}

// union1d gives the values of two rows in one row. Each value comes
// once, in the order of its value.
fn union1d(a, b) -> Arr, Fault {
    // The two rows together hold more elements than the limit, so the
    // union is a fault. The check keeps the memory inside the limit.
    if a.data.len + b.data.len > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "union1d", "the rows hold more elements than the limit");
    }
    var all: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { all.push(a.data[i]); i += 1; }
    i = 0;
    while i < b.data.len { all.push(b.data[i]); i += 1; }
    return unique_set(all);
}

// intersect1d gives the values that two rows hold in common. Each
// value comes once, in the order of its value.
fn intersect1d(a, b) -> Arr, Fault {
    let other = sort_set(b.data);
    var common: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        if holds_value(other, a.data[i]) { common.push(a.data[i]); }
        i += 1;
    }
    return unique_set(common);
}

// setdiff1d gives the values of the first row that the second row
// holds no value of. Each value comes once, in the order of its value.
fn setdiff1d(a, b) -> Arr, Fault {
    let other = sort_set(b.data);
    var rest: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        if !holds_value(other, a.data[i]) { rest.push(a.data[i]); }
        i += 1;
    }
    return unique_set(rest);
}

// in1d tells for each value of a flat row whether another row holds
// it. The result holds 1.0 and 0.0.
fn in1d(a, b, invert: bool = false) -> Arr, Fault {
    let other = sort_set(b.data);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        var found = holds_value(other, a.data[i]);
        if invert { found = !found; }
        if found { d.push(1.0); } else { d.push(0.0); }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(d.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// isin tells for each element of an array whether another array holds
// its value. The result has the shape of the first array and holds
// 1.0 and 0.0.
fn isin(a, b, invert: bool = false) -> Arr, Fault {
    let other = sort_set(b.data);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        var found = holds_value(other, a.data[i]);
        if invert { found = !found; }
        if found { d.push(1.0); } else { d.push(0.0); }
        i += 1;
    }
    return Arr { ndim: a.ndim, shape: same_shape(a), data: d }, no_fault();
}
