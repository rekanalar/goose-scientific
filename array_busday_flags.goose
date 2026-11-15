// The flags of the dates: the test of a business day, and the test of
// a date that is not a time, as is_busday and isnat of NumPy.
//
// A date that is a business day stands on a day of the week that the
// weekmask allows, and no holiday of the calendar covers it. A date
// that is not a time is no business day.
namespace scientific;

// is_busday_one tells whether one date is a business day of a
// calendar. A date that is not a time is no business day, as in NumPy.
fn is_busday_one(day: f64, cal) -> bool {
    let d, nat, f = date_of(day, "is_busday");
    if f.kind != F_NONE { return false; }
    if nat { return false; }
    return bus_is_day(cal, d);
}

// is_busday tells for every date of a list whether the date is a
// business day of a calendar. A date that is not a whole number of
// days, or that is not finite, is a fault of kind F_VALUE. A date
// outside the years 1 to 9999 is a fault of kind F_RANGE.
fn is_busday(dates: Arr, cal) -> Arr, Fault {
    let tag = "is_busday";
    let f = check_stack(dates.ndim, dates.shape, dates.data.len, 1, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    var data: f64[>..] = [];
    var i: i64 = 0;
    while i < dates.data.len {
        let d, nat, df = date_of(dates.data[i], tag);
        if df.kind != F_NONE { return empty_arr(), df; }
        var on = 0.0;
        if !nat {
            if bus_is_day(cal, d) { on = 1.0; }
        }
        data.push(on);
        i += 1;
    }
    var s: i64[..8] = [];
    var k: i64 = 0;
    while k < dates.shape.len { s.push(dates.shape[k]); k += 1; }
    return Arr { ndim: dates.ndim, shape: s, data: data }, no_fault();
}

// isnat_one tells whether one date is a date that is not a time.
fn isnat_one(day: f64) -> bool {
    return day != day;
}

// isnat tells for every date of a list whether the date is a date that
// is not a time. A value that is not finite, but a number, is a fault
// of kind F_VALUE. A date that is not a whole number of days is a
// fault of kind F_VALUE too.
fn isnat(dates: Arr) -> Arr, Fault {
    let tag = "isnat";
    let f = check_stack(dates.ndim, dates.shape, dates.data.len, 1, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    var data: f64[>..] = [];
    var i: i64 = 0;
    while i < dates.data.len {
        let v = dates.data[i];
        var on = 0.0;
        if v != v { on = 1.0; }
        else if v - v != 0.0 || v != mathshim::trunc_of(v) {
            return empty_arr(), fault(F_VALUE, tag, "the value is not a whole number of days");
        }
        data.push(on);
        i += 1;
    }
    var s: i64[..8] = [];
    var k: i64 = 0;
    while k < dates.shape.len { s.push(dates.shape[k]); k += 1; }
    return Arr { ndim: dates.ndim, shape: s, data: data }, no_fault();
}
