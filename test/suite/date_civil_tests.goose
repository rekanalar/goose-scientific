// The tests of the civil calendar of the dates. Run them with
// SCIENTIFIC_SUITES=date_civil.
//
// The calls match the pins of NumPy 2.5.4: the count of days of a date,
// the date of a count of days and the day of the week. A date is a whole
// number of days since 1970-01-01. The day of the week starts at Monday,
// so Monday is 0.
namespace date_civil_tests;

import scientific from "../..";
import date_civil_pins;

fn nat() -> f64 {
    return 0.0 / 0.0;
}

fn test_civil_days() {
    let n = date_civil_pins::DAYS.len / 4;
    var i: i64 = 0;
    while i < n {
        let r = i * 4;
        let y = date_civil_pins::DAYS[r] as i64;
        let m = date_civil_pins::DAYS[r + 1] as i64;
        let d = date_civil_pins::DAYS[r + 2] as i64;
        let want = date_civil_pins::DAYS[r + 3] as i64;
        assert(scientific::civil_days(y, m, d) == want);
        i += 1;
    }
}

fn test_civil_days_outside() {
    // The algorithm holds for the years that the package refuses.
    let n = date_civil_pins::OUTSIDE.len / 4;
    var i: i64 = 0;
    while i < n {
        let r = i * 4;
        let y = date_civil_pins::OUTSIDE[r] as i64;
        let m = date_civil_pins::OUTSIDE[r + 1] as i64;
        let d = date_civil_pins::OUTSIDE[r + 2] as i64;
        let want = date_civil_pins::OUTSIDE[r + 3] as i64;
        assert(scientific::civil_days(y, m, d) == want);
        var c = scientific::CivilDate { year: 0, month: 0, day: 0 };
        scientific::civil_of(want, c);
        assert(c.year == y && c.month == m && c.day == d);
        i += 1;
    }
}

fn test_civil_of() {
    let n = date_civil_pins::PARTS.len / 5;
    var i: i64 = 0;
    while i < n {
        let r = i * 5;
        let day = date_civil_pins::PARTS[r] as i64;
        var c = scientific::CivilDate { year: 0, month: 0, day: 0 };
        scientific::civil_of(day, c);
        assert(c.year == date_civil_pins::PARTS[r + 1] as i64);
        assert(c.month == date_civil_pins::PARTS[r + 2] as i64);
        assert(c.day == date_civil_pins::PARTS[r + 3] as i64);
        // The date and its count of days make a round trip.
        assert(scientific::civil_days(c.year, c.month, c.day) == day);
        i += 1;
    }
}

fn test_date_weekday() {
    let n = date_civil_pins::PARTS.len / 5;
    var i: i64 = 0;
    while i < n {
        let r = i * 5;
        let day = date_civil_pins::PARTS[r] as i64;
        assert(scientific::date_weekday(day) == date_civil_pins::PARTS[r + 4] as i64);
        i += 1;
    }
    // 1970-01-01 is a Thursday and 1969-12-31 is a Wednesday.
    assert(scientific::date_weekday(0) == 3);
    assert(scientific::date_weekday(-1) == 2);
}

fn test_date_month() {
    let n = date_civil_pins::PARTS.len / 5;
    var i: i64 = 0;
    while i < n {
        let r = i * 5;
        let day = date_civil_pins::PARTS[r] as i64;
        assert(scientific::date_month(day) == date_civil_pins::PARTS[r + 2] as i64);
        i += 1;
    }
}

fn test_date_in_range() {
    assert(scientific::date_in_range(scientific::DATE_MIN_DAY));
    assert(scientific::date_in_range(scientific::DATE_MAX_DAY));
    assert(scientific::date_in_range(0));
    assert(!scientific::date_in_range(scientific::DATE_MIN_DAY - 1));
    assert(!scientific::date_in_range(scientific::DATE_MAX_DAY + 1));
    // The limits are the days of 0001-01-01 and 9999-12-31 in NumPy.
    assert(scientific::DATE_MIN_DAY == date_civil_pins::DAYS[3] as i64);
    assert(scientific::DATE_MAX_DAY == date_civil_pins::DAYS[7] as i64);
}

fn test_date_of() {
    let d, nat_flag, f = scientific::date_of(20771.0, "t");
    assert(f.kind == scientific::F_NONE && !nat_flag && d == 20771);
    let d1, n1, f1 = scientific::date_of(-719162.0, "t");
    assert(f1.kind == scientific::F_NONE && !n1 && d1 == -719162);
    let d2, n2, f2 = scientific::date_of(2932896.0, "t");
    assert(f2.kind == scientific::F_NONE && !n2 && d2 == 2932896);
    // A date that is not a time is no fault. The second result tells it.
    let d3, n3, f3 = scientific::date_of(nat(), "t");
    assert(f3.kind == scientific::F_NONE && n3 && d3 == 0);
    // A value that is not finite is a fault of kind F_VALUE.
    let d4, n4, f4 = scientific::date_of(1.0 / 0.0, "t");
    assert(f4.kind == scientific::F_VALUE && !n4);
    let d5, n5, f5 = scientific::date_of(-1.0 / 0.0, "t");
    assert(f5.kind == scientific::F_VALUE && !n5);
    // A value that is not a whole number of days is a fault of kind F_VALUE.
    let d6, n6, f6 = scientific::date_of(20771.5, "t");
    assert(f6.kind == scientific::F_VALUE && !n6);
    // A day outside the years 1 to 9999 is a fault of kind F_RANGE.
    let d7, n7, f7 = scientific::date_of(-719163.0, "t");
    assert(f7.kind == scientific::F_RANGE && !n7);
    let d8, n8, f8 = scientific::date_of(2932897.0, "t");
    assert(f8.kind == scientific::F_RANGE && !n8);
}

fn test_date_days() {
    let n = date_civil_pins::DAYS.len / 4;
    var i: i64 = 0;
    while i < n {
        let r = i * 4;
        let y = date_civil_pins::DAYS[r] as i64;
        let m = date_civil_pins::DAYS[r + 1] as i64;
        let d = date_civil_pins::DAYS[r + 2] as i64;
        let day, f = scientific::date_days(y, m, d);
        assert(f.kind == scientific::F_NONE);
        assert(day == date_civil_pins::DAYS[r + 3]);
        i += 1;
    }
}

fn test_date_days_refused() {
    // NumPy refuses each date of the list, so the package gives a fault.
    let n = date_civil_pins::BAD.len / 3;
    var i: i64 = 0;
    while i < n {
        let r = i * 3;
        let y = date_civil_pins::BAD[r] as i64;
        let m = date_civil_pins::BAD[r + 1] as i64;
        let d = date_civil_pins::BAD[r + 2] as i64;
        let day, f = scientific::date_days(y, m, d);
        assert(f.kind == scientific::F_RANGE);
        assert(day == 0.0);
        i += 1;
    }
    // The package holds the years 1 to 9999 only. NumPy holds more.
    let a, fa = scientific::date_days(0, 1, 1);
    assert(fa.kind == scientific::F_RANGE);
    let b, fb = scientific::date_days(10000, 1, 1);
    assert(fb.kind == scientific::F_RANGE);
    let c, fc = scientific::date_days(-1, 12, 31);
    assert(fc.kind == scientific::F_RANGE);
}

fn test_date_parts() {
    let n = date_civil_pins::PARTS.len / 5;
    var i: i64 = 0;
    while i < n {
        let r = i * 5;
        let y, m, d, f = scientific::date_parts(date_civil_pins::PARTS[r]);
        assert(f.kind == scientific::F_NONE);
        assert(y == date_civil_pins::PARTS[r + 1] as i64);
        assert(m == date_civil_pins::PARTS[r + 2] as i64);
        assert(d == date_civil_pins::PARTS[r + 3] as i64);
        i += 1;
    }
}

fn test_date_parts_faults() {
    // A date that is not a time is a fault of kind F_VALUE.
    let y, m, d, f = scientific::date_parts(nat());
    assert(f.kind == scientific::F_VALUE);
    let y1, m1, d1, f1 = scientific::date_parts(1.0 / 0.0);
    assert(f1.kind == scientific::F_VALUE);
    let y2, m2, d2, f2 = scientific::date_parts(0.5);
    assert(f2.kind == scientific::F_VALUE);
    // A day outside the years 1 to 9999 is a fault of kind F_RANGE.
    let y3, m3, d3, f3 = scientific::date_parts(-719163.0);
    assert(f3.kind == scientific::F_RANGE);
    let y4, m4, d4, f4 = scientific::date_parts(2932897.0);
    assert(f4.kind == scientific::F_RANGE);
}
