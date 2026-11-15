// The civil calendar of the package: the count of days since
// 1970-01-01, the date of such a count, and the day of the week.
//
// A date is a whole number of days. The day zero is 1970-01-01. The
// days come from the proleptic Gregorian calendar. The package holds
// no time of day, so a date holds no hour and no minute.
//
// The count of days of a date of a year, a month and a day comes from
// the algorithm of Howard Hinnant. The same algorithm gives the date
// of a count of days. The algorithm works for every year, but the
// package holds the years 1 to 9999 only.
namespace scientific;

import mathshim;

// The first day and the last day that the package holds. The first day
// is 0001-01-01 and the last day is 9999-12-31.
const DATE_MIN_DAY: i64 = -719162;
const DATE_MAX_DAY: i64 = 2932896;

// CivilDate holds a date of the civil calendar.
struct CivilDate { year: i64, month: i64, day: i64 }

// civil_days gives the count of days since 1970-01-01 of a date of a
// year, a month and a day of the proleptic Gregorian calendar.
fn civil_days(y: i64, m: i64, d: i64) -> i64 {
    var yy = y;
    if m <= 2 { yy = yy - 1; }
    var era = yy;
    if yy < 0 { era = yy - 399; }
    era = era / 400;
    let yoe = yy - era * 400;
    var mp = m;
    if m > 2 { mp = m - 3; } else { mp = m + 9; }
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

// civil_of gives the year, the month and the day of a count of days
// since 1970-01-01.
fn civil_of(days: i64, out: CivilDate&) {
    let z = days + 719468;
    var era = z;
    if z < 0 { era = z - 146096; }
    era = era / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    var m = mp;
    if mp < 10 { m = mp + 3; } else { m = mp - 9; }
    out.year = y;
    out.month = m;
    out.day = d;
    if m <= 2 { out.year = y + 1; }
}

// date_weekday gives the day of the week of a count of days. The count
// starts at Monday, so the weekmask of NumPy and the day of the week
// use the same order.
fn date_weekday(day: i64) -> i64 {
    return (day - 4) % 7;
}

// date_month gives the month of a count of days. The modified rolls of
// NumPy compare the month alone, so this call gives the month alone.
fn date_month(day: i64) -> i64 {
    var c = CivilDate { year: 0, month: 0, day: 0 };
    civil_of(day, c);
    return c.month;
}

// date_of gives the count of days of a date of a caller. A value that
// is not a number is the day that is not a time, and the second result
// tells it from a real day. A value that is not a whole number of days,
// or that is not finite, is a fault of kind F_VALUE. A day outside the
// years 1 to 9999 is a fault of kind F_RANGE.
fn date_of(value: f64, tag: const u8[:]) -> i64, bool, Fault {
    if value != value { return 0, true, no_fault(); }
    if value - value != 0.0 {
        return 0, false, fault(F_VALUE, tag, "the date is not a finite number");
    }
    if value != mathshim::trunc_of(value) {
        return 0, false, fault(F_VALUE, tag, "the date is not a whole number of days");
    }
    if value < DATE_MIN_DAY as f64 || value > DATE_MAX_DAY as f64 {
        return 0, false, fault(F_RANGE, tag, "the year of the date stands outside 1 to 9999");
    }
    return value as i64, false, no_fault();
}

// date_in_range tells whether a count of days stands inside the years
// 1 to 9999.
fn date_in_range(day: i64) -> bool {
    if day < DATE_MIN_DAY || day > DATE_MAX_DAY { return false; }
    return true;
}

// date_days gives the count of days since 1970-01-01 of a date of a
// year, a month and a day. A year outside 1 to 9999 is a fault of kind
// F_RANGE. A month outside 1 to 12 and a day outside the month are a
// fault of kind F_RANGE too.
fn date_days(y: i64, m: i64, d: i64) -> f64, Fault {
    let tag = "date_days";
    if y < 1 || y > 9999 {
        return 0.0, fault(F_RANGE, tag, "the year of the date stands outside 1 to 9999");
    }
    if m < 1 || m > 12 {
        return 0.0, fault(F_RANGE, tag, "the month of the date stands outside 1 to 12");
    }
    var length = 31;
    if m == 4 || m == 6 || m == 9 || m == 11 { length = 30; }
    if m == 2 {
        let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
        length = 28;
        if leap { length = 29; }
    }
    if d < 1 || d > length {
        return 0.0, fault(F_RANGE, tag, "the day of the date stands outside the month");
    }
    return civil_days(y, m, d) as f64, no_fault();
}

// date_parts gives the year, the month and the day of a date. A date
// that is not a time is a fault of kind F_VALUE. A year outside 1 to
// 9999 is a fault of kind F_RANGE.
fn date_parts(day: f64) -> i64, i64, i64, Fault {
    let tag = "date_parts";
    let d, nat, f = date_of(day, tag);
    if f.kind != F_NONE { return 0, 0, 0, f; }
    if nat { return 0, 0, 0, fault(F_VALUE, tag, "the date is not a time"); }
    var c = CivilDate { year: 0, month: 0, day: 0 };
    civil_of(d, c);
    return c.year, c.month, c.day, no_fault();
}
