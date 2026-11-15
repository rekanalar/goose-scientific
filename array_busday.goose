// The calendar of the business days: the days of the week that are
// business days, and the days that are not business days because of a
// holiday.
//
// The weekmask of NumPy holds seven signs. The first sign stands for
// Monday. A weekmask may also name the days of the week. Each name
// holds three letters. White space separates the names.
//
// The holidays of NumPy are a list of dates. The package holds them as
// counts of days. A holiday on a day that the weekmask excludes is not
// a holiday, as in NumPy. A calendar holds each holiday once and holds
// the holidays in the order of the days.
//
// The name of a day of the week of NumPy starts with a capital letter.
// The names are Mon, Tue, Wed, Thu, Fri, Sat and Sun.
namespace scientific;

// The limits of the calendar of the business days. A span of days
// stands above the limit of 3652058 days. That is one day above the
// widest span of the years 1 to 9999, so the limit guards against a
// bad count and not against a real pair. A count of holidays stands
// above the limit of 65536. A walk of a roll stands above the limit of
// 1000000 days.
const MAX_BUS_SPAN: i64 = 3652058;
const MAX_BUS_HOLIDAYS: i64 = 65536;
const MAX_BUS_ROLL: i64 = 1000000;

// The rolls of a day that is not a business day. The roll of NumPy
// names the way the day moves to a business day.
const BUS_ROLL_RAISE: i64 = 0;
const BUS_ROLL_NAT: i64 = 1;
const BUS_ROLL_FORWARD: i64 = 2;
const BUS_ROLL_FOLLOWING: i64 = 3;
const BUS_ROLL_BACKWARD: i64 = 4;
const BUS_ROLL_PRECEDING: i64 = 5;
const BUS_ROLL_MODFOLLOWING: i64 = 6;
const BUS_ROLL_MODPRECEDING: i64 = 7;

// BusCal holds a calendar of the business days. The mask holds seven
// signs. The first sign stands for Monday. The busdays count the signs
// that stand for a business day. The holidays hold the counts of the
// days of the holidays in the order of the days.
struct BusCal { mask: i64[..7], busdays: i64, holidays: f64[>..] }

// bus_zero_mask gives a mask of seven signs that hold no business day.
fn bus_zero_mask() -> i64[..7] {
    var mask: i64[..7] = [];
    var i: i64 = 0;
    while i < 7 { mask.push(0); i += 1; }
    return mask;
}

// buscal_empty gives a calendar that holds no business day and no
// holiday.
fn buscal_empty() -> BusCal {
    return BusCal { mask: bus_zero_mask(), busdays: 0, holidays: [] };
}

// buscal_default gives the calendar of NumPy: the days from Monday to
// Friday are business days, and the calendar holds no holiday.
fn buscal_default() -> BusCal {
    var mask = bus_zero_mask();
    var i: i64 = 0;
    while i < 5 { mask[i] = 1; i += 1; }
    return BusCal { mask: mask, busdays: 5, holidays: [] };
}

// bus_space tells whether a byte is white space.
fn bus_space(c: u8) -> bool {
    if c == 32 { return true; }
    if c == 9 { return true; }
    if c == 10 { return true; }
    if c == 11 { return true; }
    if c == 12 { return true; }
    if c == 13 { return true; }
    return false;
}

// bus_day_index gives the place of a day of the week. Monday stands at
// the place zero. A name that NumPy does not hold gives the place -1.
// NumPy reads three letters, so a name of two letters is no name.
fn bus_day_index(c0: u8, c1: u8, c2: u8) -> i64 {
    if c0 == 77 && c1 == 111 && c2 == 110 { return 0; }
    if c0 == 84 && c1 == 117 && c2 == 101 { return 1; }
    if c0 == 87 && c1 == 101 && c2 == 100 { return 2; }
    if c0 == 84 && c1 == 104 && c2 == 117 { return 3; }
    if c0 == 70 && c1 == 114 && c2 == 105 { return 4; }
    if c0 == 83 && c1 == 97 && c2 == 116 { return 5; }
    if c0 == 83 && c1 == 117 && c2 == 110 { return 6; }
    return -1;
}

// bus_weekmask gives the mask of a weekmask of NumPy. A text of seven
// signs gives a mask of seven signs. Any other text names the days of
// the week. A text that names a day that NumPy does not hold is a
// fault of kind F_VALUE.
fn bus_weekmask(text: const u8[:], tag: const u8[:]) -> i64[..7], Fault {
    var mask = bus_zero_mask();
    if text.len == 7 {
        var ok = true;
        var i: i64 = 0;
        while i < 7 {
            let c = text[i];
            if c == 48 { mask[i] = 0; }
            else if c == 49 { mask[i] = 1; }
            else { ok = false; }
            i += 1;
        }
        if ok { return mask, no_fault(); }
    }
    mask = bus_zero_mask();
    var i: i64 = 0;
    while i < text.len {
        while i < text.len && bus_space(text[i]) { i += 1; }
        if i >= text.len { break; }
        if i + 2 >= text.len {
            return mask, fault(F_VALUE, tag, "the weekmask names a day that holds less than three letters");
        }
        let at = bus_day_index(text[i], text[i + 1], text[i + 2]);
        if at < 0 {
            return mask, fault(F_VALUE, tag, "the weekmask names a day that NumPy does not hold");
        }
        mask[at] = 1;
        i += 3;
    }
    return mask, no_fault();
}

// bus_holidays_of gives the holidays of a calendar. The list holds the
// counts of the days of the holidays. It holds each holiday once, and
// it holds the holidays in the order of the days.
fn bus_holidays_of(mask: i64[..7], holidays: Arr&, tag: const u8[:]) -> f64[>..], Fault {
    let sorted, f = bus_sort_holidays(holidays, mask, tag);
    if f.kind != F_NONE { return [], f; }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < sorted.data.len {
        if out.len == 0 || out[out.len - 1] != sorted.data[i] { out.push(sorted.data[i]); }
        i += 1;
    }
    return out, no_fault();
}

// bus_sort_holidays gives the valid holidays of a list, in the order of
// the days. A holiday on a day that the weekmask excludes falls away,
// as in NumPy. A holiday that is not a time falls away, as in NumPy. A
// count of holidays above the limit is a fault of kind F_LIMIT.
fn bus_sort_holidays(holidays: Arr&, mask: i64[..7], tag: const u8[:]) -> Arr, Fault {
    // A list of no holidays holds no holiday, and NumPy accepts it. The
    // check of the shape rejects a dimension of no length, so the empty
    // list leaves the check early.
    if holidays.ndim == 1 && holidays.data.len == 0 {
        var es: i64[..8] = [];
        es.push(0);
        return Arr { ndim: 1, shape: es, data: [] }, no_fault();
    }
    let f = check_stack(holidays.ndim, holidays.shape, holidays.data.len, 1, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    if holidays.data.len > MAX_BUS_HOLIDAYS {
        return empty_arr(), fault(F_LIMIT, tag, "the count of the holidays stands above the limit of 65536");
    }
    var days: f64[>..] = [];
    var i: i64 = 0;
    while i < holidays.data.len {
        let day, nat, df = date_of(holidays.data[i], tag);
        if df.kind != F_NONE { return empty_arr(), df; }
        if !nat {
            if mask[date_weekday(day)] == 1 { days.push(day as f64); }
        }
        i += 1;
    }
    let sorted = sort_set(days[..]);
    var s: i64[..8] = [];
    s.push(sorted.len);
    return Arr { ndim: 1, shape: s, data: sorted }, no_fault();
}

// busdaycalendar gives a calendar of the business days. The weekmask
// names the days of the week that are business days. The holidays hold
// the days that are not business days. A weekmask that holds no
// business day is a fault of kind F_VALUE, as in NumPy.
fn busdaycalendar(weekmask: const u8[:], holidays: Arr) -> BusCal, Fault {
    let tag = "busdaycalendar";
    let mask, mf = bus_weekmask(weekmask, tag);
    if mf.kind != F_NONE { return buscal_empty(), mf; }
    var busdays: i64 = 0;
    var i: i64 = 0;
    while i < 7 {
        busdays += mask[i];
        i += 1;
    }
    if busdays == 0 {
        return buscal_empty(), fault(F_VALUE, tag, "the weekmask holds no business day");
    }
    let hol, hf = bus_holidays_of(mask, holidays, tag);
    if hf.kind != F_NONE { return buscal_empty(), hf; }
    return BusCal { mask: mask, busdays: busdays, holidays: hol }, no_fault();
}

// buscal_weekmask gives the seven signs of the mask of a calendar. The
// first sign stands for Monday.
fn buscal_weekmask(cal) -> i64[..7] {
    var out = bus_zero_mask();
    var i: i64 = 0;
    while i < 7 {
        out[i] = cal.mask[i];
        i += 1;
    }
    return out;
}

// buscal_holidays gives the counts of the days of the holidays of a
// calendar. The list holds each holiday once, in the order of the days.
fn buscal_holidays(cal) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < cal.holidays.len {
        out.push(cal.holidays[i]);
        i += 1;
    }
    return out;
}

// bus_hol_ge gives the place of the first holiday that stands on a day
// or after it.
fn bus_hol_ge(cal, day: i64) -> i64 {
    var lo: i64 = 0;
    var hi: i64 = cal.holidays.len;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cal.holidays[mid] < day as f64 { lo = mid + 1; } else { hi = mid; }
    }
    return lo;
}

// bus_hol_gt gives the place of the first holiday that stands after a
// day.
fn bus_hol_gt(cal, day: i64) -> i64 {
    var lo: i64 = 0;
    var hi: i64 = cal.holidays.len;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cal.holidays[mid] <= day as f64 { lo = mid + 1; } else { hi = mid; }
    }
    return lo;
}

// bus_is_holiday tells whether a day stands on a holiday of a
// calendar.
fn bus_is_holiday(cal, day: i64) -> bool {
    let at = bus_hol_ge(cal, day);
    if at >= cal.holidays.len { return false; }
    return cal.holidays[at] == day as f64;
}

// bus_is_day tells whether a day is a business day of a calendar.
fn bus_is_day(cal, day: i64) -> bool {
    if cal.mask[date_weekday(day)] != 1 { return false; }
    if bus_is_holiday(cal, day) { return false; }
    return true;
}

// bus_roll_code gives the number of a roll of NumPy. A roll that NumPy
// does not hold is a fault of kind F_VALUE.
fn bus_roll_code(roll: const u8[:], tag: const u8[:]) -> i64, Fault {
    if roll == "raise" { return BUS_ROLL_RAISE, no_fault(); }
    if roll == "nat" { return BUS_ROLL_NAT, no_fault(); }
    if roll == "forward" { return BUS_ROLL_FORWARD, no_fault(); }
    if roll == "following" { return BUS_ROLL_FOLLOWING, no_fault(); }
    if roll == "backward" { return BUS_ROLL_BACKWARD, no_fault(); }
    if roll == "preceding" { return BUS_ROLL_PRECEDING, no_fault(); }
    if roll == "modifiedfollowing" { return BUS_ROLL_MODFOLLOWING, no_fault(); }
    if roll == "modifiedpreceding" { return BUS_ROLL_MODPRECEDING, no_fault(); }
    return -1, fault(F_VALUE, tag, "the roll is not a roll of NumPy");
}

// bus_walk_forward gives the first business day on a day or after it.
// The walk stops at the limit of 1000000 days, and it gives a fault of
// kind F_LIMIT then.
fn bus_walk_forward(day: i64, cal, tag: const u8[:]) -> i64, Fault {
    var d = day;
    var steps: i64 = 0;
    var go = true;
    while go {
        d += 1;
        steps += 1;
        if steps > MAX_BUS_ROLL {
            return 0, fault(F_LIMIT, tag, "the roll walks more days than the limit of 1000000");
        }
        if bus_is_day(cal, d) { go = false; }
    }
    return d, no_fault();
}

// bus_walk_backward gives the first business day on a day or before it.
// The walk stops at the limit of 1000000 days, and it gives a fault of
// kind F_LIMIT then.
fn bus_walk_backward(day: i64, cal, tag: const u8[:]) -> i64, Fault {
    var d = day;
    var steps: i64 = 0;
    var go = true;
    while go {
        d -= 1;
        steps += 1;
        if steps > MAX_BUS_ROLL {
            return 0, fault(F_LIMIT, tag, "the roll walks more days than the limit of 1000000");
        }
        if bus_is_day(cal, d) { go = false; }
    }
    return d, no_fault();
}

// bus_roll_day gives the business day of a roll. The second result
// tells that the day is not a time. The roll of NumPy moves a day that
// is not a business day to the next business day, to the business day
// before it, or to the business day of the same month. The rolls of
// NumPy compare the month alone, so the year of the month does not
// count.
fn bus_roll_day(day: i64, code: i64, cal, tag: const u8[:]) -> i64, bool, Fault {
    if bus_is_day(cal, day) { return day, false, no_fault(); }
    if code == BUS_ROLL_NAT { return 0, true, no_fault(); }
    if code == BUS_ROLL_RAISE {
        return 0, false, fault(F_VALUE, tag, "the date is not a business day");
    }
    if code == BUS_ROLL_FORWARD || code == BUS_ROLL_FOLLOWING {
        let d, f = bus_walk_forward(day, cal, tag);
        if f.kind != F_NONE { return 0, false, f; }
        return d, false, no_fault();
    }
    if code == BUS_ROLL_BACKWARD || code == BUS_ROLL_PRECEDING {
        let d, f = bus_walk_backward(day, cal, tag);
        if f.kind != F_NONE { return 0, false, f; }
        return d, false, no_fault();
    }
    if code == BUS_ROLL_MODFOLLOWING {
        let d, f = bus_walk_forward(day, cal, tag);
        if f.kind != F_NONE { return 0, false, f; }
        if date_month(d) == date_month(day) { return d, false, no_fault(); }
        let b, bf = bus_walk_backward(day, cal, tag);
        if bf.kind != F_NONE { return 0, false, bf; }
        return b, false, no_fault();
    }
    let d, f = bus_walk_backward(day, cal, tag);
    if f.kind != F_NONE { return 0, false, f; }
    if date_month(d) == date_month(day) { return d, false, no_fault(); }
    let b, bf = bus_walk_forward(day, cal, tag);
    if bf.kind != F_NONE { return 0, false, bf; }
    return b, false, no_fault();
}
