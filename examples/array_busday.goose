// The business days: the count of the work days between two dates, and
// the date that stands some work days away. Run it with:
// goose run examples/array_busday.goose
import std;
import scientific from "..";

fn one_of(values: f64[>..]) -> scientific::Arr {
    var s: i64[..8] = [];
    s.push(values.len);
    return scientific::Arr { ndim: 1, shape: s, data: values };
}

fn main() {
    // A date is a whole number of days since 1970-01-01.
    let d0, f0 = scientific::date_days(2026, 11, 13);
    let d1, f1 = scientific::date_days(2026, 12, 25);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);
    print("2026-11-13 is day ", d0, " and 2026-12-25 is day ", d1, "\n");

    // The count of the business days between two dates.
    var b: f64[>..] = [];
    b.push(d0);
    var e: f64[>..] = [];
    e.push(d1);
    let c, fc = scientific::busday_count(one_of(copy(b)), one_of(copy(e)),
        scientific::buscal_default());
    assert(fc.kind == scientific::F_NONE);
    print("the count of the business days is ", c.data[0], "\n");

    // The date that stands three business days away.
    var o: f64[>..] = [];
    o.push(3.0);
    let r, fr = scientific::busday_offset(one_of(copy(b)), one_of(copy(o)),
        "forward", scientific::buscal_default());
    assert(fr.kind == scientific::F_NONE);
    let y, m, day, fd = scientific::date_parts(r.data[0]);
    assert(fd.kind == scientific::F_NONE);
    print("three business days later is ", y, "-", m, "-", day, "\n");

    // The roll moves a day of the weekend to a business day.
    var sat: f64[>..] = [];
    sat.push(d0 + 1.0);
    let r2, fr2 = scientific::busday_offset(one_of(copy(sat)), one_of(copy(o)),
        "forward", scientific::buscal_default());
    assert(fr2.kind == scientific::F_NONE);
    print("Saturday rolls to day ", r2.data[0], "\n");

    // A calendar holds the days of the week and the holidays.
    var hol: f64[>..] = [];
    hol.push(d1);
    let cal, fcal = scientific::busdaycalendar("1111100", one_of(copy(hol)));
    assert(fcal.kind == scientific::F_NONE);
    var dd: f64[>..] = [];
    dd.push(d1);
    let is, fis = scientific::is_busday(one_of(copy(dd)), cal);
    assert(fis.kind == scientific::F_NONE);
    print("2026-12-25 is a business day of the calendar: ", is.data[0] == 1.0, "\n");

    // The text of a date, and the unit of a date type.
    let t, ft = scientific::datetime_as_string_one(d0);
    assert(ft.kind == scientific::F_NONE);
    print("the text of the day is ", t, "\n");
    let unit, count, fu = scientific::datetime_data("M8[5ms]");
    assert(fu.kind == scientific::F_NONE);
    print("M8[5ms] holds the unit ", unit, " and the count ", count, "\n");
    return 0;
}
