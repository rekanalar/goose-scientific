// The count of the business days between two dates, as busday_count
// of NumPy.
//
// The count holds the days that the weekmask of the calendar allows
// and that no holiday covers. The last date does not count. A count
// over a pair in the wrong order gives a negative count. NumPy
// corrects that pair in a way that drops one day: the correction of
// the issue 23197 of NumPy. This file holds the same correction, so
// the counts match NumPy 2.5.4.
//
// The count of whole weeks is a multiplication, not a walk over the
// days. Only the rest below a week is a walk, and a rest holds at most
// six days.
namespace scientific;

// bus_c_rem gives the rest of a division in the way of the C language:
// the sign of the first value stays. Goose gives the rest in the way of
// mathematics, so this call computes the rest by hand.
fn bus_c_rem(a: i64, b: i64) -> i64 {
    return a - (a / b) * b;
}

// bus_count_holidays gives the count of the holidays from a first day
// up to a last day that does not count.
fn bus_count_holidays(cal, begin: i64, end: i64) -> i64 {
    return bus_hol_ge(cal, end) - bus_hol_ge(cal, begin);
}

// busday_count_one gives the count of the business days of a pair of
// dates. The last date does not count. A date that is not a time is a
// fault of kind F_VALUE.
fn busday_count_one(begin: f64, end: f64, cal) -> i64, Fault {
    let tag = "busday_count";
    if cal.busdays == 0 {
        return 0, fault(F_VALUE, tag, "the weekmask holds no business day");
    }
    let b, nat1, f1 = date_of(begin, tag);
    if f1.kind != F_NONE { return 0, f1; }
    let e, nat2, f2 = date_of(end, tag);
    if f2.kind != F_NONE { return 0, f2; }
    if nat1 || nat2 {
        return 0, fault(F_VALUE, tag, "a date is not a time");
    }
    var lo = b;
    var hi = e;
    var swapped = false;
    if lo > hi {
        let t = lo;
        lo = hi;
        hi = t;
        swapped = true;
        lo += 1;
        hi += 1;
    }
    if lo == hi { return 0, no_fault(); }
    if hi - lo > MAX_BUS_SPAN {
        return 0, fault(F_LIMIT, tag, "the span of the pair stands above the limit of 3652058 days");
    }
    var count = -bus_count_holidays(cal, lo, hi);
    let weeks = (hi - lo) / 7;
    count += weeks * cal.busdays;
    lo += weeks * 7;
    if lo < hi {
        var dow = date_weekday(lo);
        while lo < hi {
            if cal.mask[dow] == 1 { count += 1; }
            lo += 1;
            dow += 1;
            if dow == 7 { dow = 0; }
        }
    }
    if swapped { count = -count; }
    return count, no_fault();
}

// busday_count gives the count of the business days of two lists of
// dates, one pair at a time. The lists spread over each other in the
// way of NumPy. Each date is a whole number of days since 1970-01-01.
// A date that is not a time is a fault of kind F_VALUE. A date outside
// the years 1 to 9999 is a fault of kind F_RANGE. A span above the
// limit of 3652058 days is a fault of kind F_LIMIT.
fn busday_count(begin: Arr, end: Arr, cal) -> Arr, Fault {
    let tag = "busday_count";
    let f = check_stack(begin.ndim, begin.shape, begin.data.len, 1, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let f2 = check_stack(end.ndim, end.shape, end.data.len, 1, tag);
    if f2.kind != F_NONE { return empty_arr(), f2; }
    let shape, sf = broadcast_shapes(begin.shape, end.shape);
    if sf.kind != F_NONE { return empty_arr(), sf; }
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    bcast_indices(shape, begin.shape, end.shape, ia, ib);
    var data: f64[>..] = [];
    var i: i64 = 0;
    let n = numel(shape);
    while i < n {
        let c, cf = busday_count_one(begin.data[ia[i]], end.data[ib[i]], cal);
        if cf.kind != F_NONE { return empty_arr(), cf; }
        data.push(c as f64);
        i += 1;
    }
    var s: i64[..8] = [];
    var k: i64 = 0;
    while k < shape.len { s.push(shape[k]); k += 1; }
    return Arr { ndim: shape.len, shape: s, data: data }, no_fault();
}
