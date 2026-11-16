// The tests of the business days and the dates. Run them with
// SCIENTIFIC_SUITES=array_busday.
//
// The calls match the pins of NumPy 2.5.4: busday_count, busday_offset,
// busdaycalendar, is_busday, isnat, datetime_as_string and
// datetime_data. Each date is a whole number of days since 1970-01-01.
// A date that is not a time is a NaN. A count of -1 in a pin means
// that NumPy raised an error for the case.
//
// The pins hold a code for the calendar of each case. This file holds
// the mask and the holidays of each code, and the same list lives in
// the generator test/gen_array_busday.py.
namespace array_busday_tests;

import scientific from "../..";
import array_busday_pins;

// ERR marks a case where NumPy raised an error. The generator holds
// the same code.
const ERR: f64 = -999999999.0;

// text_at gives the text of a case in a byte list and a span list.
fn text_at(bytes: f64[:], spans: f64[:], i: i64) -> u8[>..] {
    var out: u8[>..] = [];
    let at = spans[2 * i] as i64;
    let n = spans[2 * i + 1] as i64;
    var k: i64 = 0;
    while k < n { out.push(bytes[at + k] as u8); k += 1; }
    return out;
}

// MASKS holds the weekmask text of each calendar code. The generator
// holds the same five texts.
fn mask_at(code: i64) -> const u8[:] {
    if code == 0 { return "1111100"; }
    if code == 1 { return "1111110"; }
    if code == 2 { return "1100011"; }
    if code == 3 { return "SatSun"; }
    return "Sun TueWed  Thu Fri";
}

// cal_of gives the calendar of a code. The holidays of the codes 1 and
// 2 are 2026-12-25 and 2026-01-01, as day numbers.
fn cal_of(code: i64) -> scientific::BusCal {
    var none: f64[>..] = [];
    if code == 1 {
        var h1: f64[>..] = [];
        h1.push(20812.0);
        let c, f = scientific::busdaycalendar(mask_at(code), one(h1));
        assert(f.kind == scientific::F_NONE);
        return c;
    }
    if code == 2 {
        var h2: f64[>..] = [];
        h2.push(20812.0);
        h2.push(20454.0);
        let c, f = scientific::busdaycalendar(mask_at(code), one(h2));
        assert(f.kind == scientific::F_NONE);
        return c;
    }
    let c, f = scientific::busdaycalendar(mask_at(code), one(none));
    assert(f.kind == scientific::F_NONE);
    return c;
}

// one wraps a row of values in an array of one dimension.
fn one(values: f64[:]) -> scientific::Arr {
    var s: i64[..8] = [];
    s.push(values.len);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < values.len { d.push(values[i]); i += 1; }
    return scientific::Arr { ndim: 1, shape: s, data: d };
}

fn nat() -> f64 {
    return 0.0 / 0.0;
}

fn day_at(rows: f64[:], i: i64, at: i64) -> f64 {
    let v = rows[at];
    if v <= ERR { return nat(); }
    return v;
}

fn roll_at(code: i64) -> const u8[:] {
    if code == 0 { return "raise"; }
    if code == 1 { return "nat"; }
    if code == 2 { return "forward"; }
    if code == 3 { return "following"; }
    if code == 4 { return "backward"; }
    if code == 5 { return "preceding"; }
    if code == 6 { return "modifiedfollowing"; }
    return "modifiedpreceding";
}

fn test_busday_count() {
    let n = array_busday_pins::COUNT.len / 4;
    var i: i64 = 0;
    while i < n {
        let r = i * 4;
        let cal = cal_of(array_busday_pins::COUNT[r] as i64);
        let b = day_at(array_busday_pins::COUNT, i, r + 1);
        let e = day_at(array_busday_pins::COUNT, i, r + 2);
        let want = array_busday_pins::COUNT[r + 3] as i64;
        var bd: f64[>..] = [];
        bd.push(b);
        var ed: f64[>..] = [];
        ed.push(e);
        let out, f = scientific::busday_count(one(bd), one(ed), cal);
        if want <= ERR {
            assert(f.kind == scientific::F_VALUE || f.kind == scientific::F_RANGE
                || f.kind == scientific::F_LIMIT);
        } else {
            assert(f.kind == scientific::F_NONE);
            assert(out.data.len == 1);
            if out.data[0] != want {
                print("count case ", i, " got ", out.data[0], " want ", want, "\n");
            }
            assert(out.data[0] == want);
        }
        i += 1;
    }
}

fn test_busday_count_one() {
    let c, f = scientific::busday_count_one(20454.0, 20484.0, scientific::buscal_default());
    assert(f.kind == scientific::F_NONE && c == 22);
    // The reversed pair of the issue 23197 of NumPy loses one day.
    let c2, f2 = scientific::busday_count_one(20484.0, 20454.0, scientific::buscal_default());
    assert(f2.kind == scientific::F_NONE && c2 == -21);
    // A date that is not a time is a fault.
    let c3, f3 = scientific::busday_count_one(nat(), 20484.0, scientific::buscal_default());
    assert(f3.kind == scientific::F_VALUE);
}

fn test_busday_offset() {
    let n = array_busday_pins::OFFSET.len / 6;
    var i: i64 = 0;
    while i < n {
        let r = i * 6;
        let cal = cal_of(array_busday_pins::OFFSET[r] as i64);
        let code = array_busday_pins::OFFSET[r + 1] as i64;
        let d = day_at(array_busday_pins::OFFSET, i, r + 2);
        let o = array_busday_pins::OFFSET[r + 3] as i64;
        let want_nat = array_busday_pins::OFFSET[r + 4] == 1.0;
        let want = array_busday_pins::OFFSET[r + 5];
        var dd: f64[>..] = [];
        dd.push(d);
        var oo: f64[>..] = [];
        oo.push(o);
        let out, f = scientific::busday_offset(one(dd), one(oo), roll_at(code), cal);
        if want <= ERR && !want_nat {
            assert(f.kind == scientific::F_VALUE || f.kind == scientific::F_RANGE
                || f.kind == scientific::F_LIMIT);
        } else {
            assert(f.kind == scientific::F_NONE);
            assert(out.data.len == 1);
            if want_nat {
                assert(out.data[0] != out.data[0]);
            } else {
                if out.data[0] != want {
                    print("offset case ", i, " got ", out.data[0], " want ", want, "\n");
                }
                assert(out.data[0] == want);
            }
        }
        i += 1;
    }
}

fn test_busday_offset_one() {
    // A business day moves by whole weeks and walks the rest.
    let d, f = scientific::busday_offset_one(20470.0, 3, 2, scientific::buscal_default());
    assert(f.kind == scientific::F_NONE && d == 20475.0);
    // A day of the weekend rolls forward first. The day 20471 is the
    // eighteenth of January of 2026, a Sunday, and the roll gives the
    // nineteenth, the day 20472.
    let d2, f2 = scientific::busday_offset_one(20471.0, 0, 2, scientific::buscal_default());
    assert(f2.kind == scientific::F_NONE && d2 == 20472.0);
    // A date that is not a time stays a date that is not a time.
    let d3, f3 = scientific::busday_offset_one(nat(), 3, 2, scientific::buscal_default());
    assert(f3.kind == scientific::F_NONE && d3 != d3);
    // The roll raise refuses a date that is not a business day.
    let d4, f4 = scientific::busday_offset_one(20471.0, 0, 0, scientific::buscal_default());
    assert(f4.kind == scientific::F_VALUE);
    // The roll nat gives a date that is not a time.
    let d5, f5 = scientific::busday_offset_one(20471.0, 0, 1, scientific::buscal_default());
    assert(f5.kind == scientific::F_NONE && d5 != d5);
}

fn test_is_busday() {
    let n = array_busday_pins::ISBUSDAY.len / 4;
    var i: i64 = 0;
    while i < n {
        let r = i * 4;
        let cal = cal_of(array_busday_pins::ISBUSDAY[r] as i64);
        let d = day_at(array_busday_pins::ISBUSDAY, i, r + 1);
        let nat_flag = array_busday_pins::ISBUSDAY[r + 2] == 1.0;
        let want = array_busday_pins::ISBUSDAY[r + 3] == 1.0;
        var dd: f64[>..] = [];
        dd.push(if nat_flag { nat() } else { d });
        let out, f = scientific::is_busday(one(dd), cal);
        assert(f.kind == scientific::F_NONE);
        assert(out.data.len == 1);
        assert((out.data[0] == 1.0) == want);
        i += 1;
    }
}

fn test_is_busday_one() {
    assert(scientific::is_busday_one(20473.0, scientific::buscal_default()));
    assert(!scientific::is_busday_one(20471.0, scientific::buscal_default()));
    assert(!scientific::is_busday_one(nat(), scientific::buscal_default()));
}

fn test_isnat() {
    let n = array_busday_pins::ISNAT.len / 2;
    var i: i64 = 0;
    while i < n {
        let r = i * 2;
        let d = day_at(array_busday_pins::ISNAT, i, r);
        let want = array_busday_pins::ISNAT[r + 1] == 1.0;
        var dd: f64[>..] = [];
        dd.push(d);
        let out, f = scientific::isnat(one(dd));
        assert(f.kind == scientific::F_NONE);
        assert((out.data[0] == 1.0) == want);
        i += 1;
    }
}

fn test_isnat_one() {
    assert(scientific::isnat_one(nat()));
    assert(!scientific::isnat_one(20470.0));
}

fn test_busdaycalendar() {
    let n = array_busday_pins::CAL.len;
    var i: i64 = 0;
    var hol_at: i64 = 0;
    while i < n {
        let code = array_busday_pins::CAL[i] as i64;
        let want_mask = text_at(array_busday_pins::CAL_BYTES,
            array_busday_pins::CAL_SPANS, i);
        let count = array_busday_pins::CAL_HOL[hol_at] as i64;
        var c = cal_of(code);
        let mask = scientific::buscal_weekmask(c);
        var k: i64 = 0;
        while k < 7 {
            let digit = (want_mask[k] as i64) - ('0');
            assert(mask[k] == digit);
            k += 1;
        }
        let hol = scientific::buscal_holidays(c);
        assert(hol.len == count);
        var j: i64 = 0;
        while j < count {
            assert(hol[j] == array_busday_pins::CAL_HOL[hol_at + 1 + j]);
            j += 1;
        }
        hol_at += 1 + count;
        i += 1;
    }
}

fn test_busdaycalendar_weekmask_named() {
    // The named form of the weekmask names the days of the week.
    var none: f64[>..] = [];
    let c, f = scientific::busdaycalendar("SatSun", one(none));
    assert(f.kind == scientific::F_NONE);
    let m = scientific::buscal_weekmask(c);
    assert(m[5] == 1 && m[6] == 1 && m[0] == 0);
    // A weekmask that holds no business day is a fault, as in NumPy.
    let c2, f2 = scientific::busdaycalendar("0000000", one(none));
    assert(f2.kind == scientific::F_VALUE && c2.busdays == 0);
    // A text that names no day of the week is a fault.
    let c3, f3 = scientific::busdaycalendar("XatSun", one(none));
    assert(f3.kind == scientific::F_VALUE);
    // A holiday on a day that the weekmask excludes falls away.
    // 2026-12-25 is a Friday and 2026-12-26 is a Saturday.
    var h: f64[>..] = [];
    h.push(20812.0);
    h.push(20813.0);
    let c4, f4 = scientific::busdaycalendar("1111100", one(h));
    assert(f4.kind == scientific::F_NONE);
    let hol4 = scientific::buscal_holidays(c4);
    assert(hol4.len == 1);
}

fn test_datetime_as_string() {
    let n = array_busday_pins::TEXT_DATE.len;
    var i: i64 = 0;
    while i < n {
        let d = day_at(array_busday_pins::TEXT_DATE, i, i);
        var dd: f64[>..] = [];
        dd.push(d);
        let out, f = scientific::datetime_as_string(one(dd));
        assert(f.kind == scientific::F_NONE);
        let at = array_busday_pins::TEXT_SPANS[2 * i] as i64;
        let len = array_busday_pins::TEXT_SPANS[2 * i + 1] as i64;
        assert(out.len == len);
        var k: i64 = 0;
        while k < len {
            assert(out[k] == array_busday_pins::TEXT_BYTES[at + k] as i64);
            k += 1;
        }
        i += 1;
    }
}

fn test_datetime_as_string_one() {
    // The day 20471 is the eighteenth of January of 2026.
    let t, f = scientific::datetime_as_string_one(20471.0);
    assert(f.kind == scientific::F_NONE && t == "2026-01-18");
    let t2, f2 = scientific::datetime_as_string_one(nat());
    assert(f2.kind == scientific::F_NONE && t2 == "NaT");
}

fn test_datetime_data() {
    // The specs live here, in the order of the pins. The generator
    // holds the same list.
    let n = array_busday_pins::DATA.len / 2;
    var i: i64 = 0;
    while i < n {
        let spec = spec_at(i);
        let want_code = array_busday_pins::DATA[2 * i] as i64;
        let want_count = array_busday_pins::DATA[2 * i + 1] as i64;
        let unit, count, f = scientific::datetime_data(spec);
        if want_code < 0 {
            if !(f.kind == scientific::F_VALUE || f.kind == scientific::F_LIMIT) {
                print("datetime_data case ", i, " fault ", f.kind, " want a fault\n");
            }
            assert(f.kind == scientific::F_VALUE || f.kind == scientific::F_LIMIT);
        } else {
            if f.kind != scientific::F_NONE || unit != unit_at(want_code) || count != want_count {
                print("datetime_data case ", i, " fault ", f.kind, " unit ", unit, " count ", count, "\n");
            }
            assert(f.kind == scientific::F_NONE);
            assert(unit == unit_at(want_code) && count == want_count);
        }
        i += 1;
    }
}

fn spec_at(i: i64) -> const u8[:] {
    if i == 0 { return "M8[D]"; }
    if i == 1 { return "M8[s]"; }
    if i == 2 { return "M8[ms]"; }
    if i == 3 { return "M8[5ms]"; }
    if i == 4 { return "M8[10D]"; }
    if i == 5 { return "M8[Y]"; }
    if i == 6 { return "M8[W]"; }
    if i == 7 { return "M8[us]"; }
    if i == 8 { return "M8[ns]"; }
    if i == 9 { return "m8[2W]"; }
    if i == 10 { return "datetime64[7s]"; }
    if i == 11 { return "timedelta64[3as]"; }
    if i == 12 { return "M8"; }
    if i == 13 { return "m8"; }
    if i == 14 { return "datetime64"; }
    if i == 15 { return "timedelta64"; }
    if i == 16 { return "M8[]"; }
    if i == 17 { return "M8[ms/1000]"; }
    if i == 18 { return "M8[13]"; }
    if i == 19 { return "M8[xx]"; }
    if i == 20 { return "i8"; }
    if i == 21 { return "M8[5us]"; }
    return "M8[10000000000ms]";
}

fn unit_at(code: i64) -> const u8[:] {
    if code == 0 { return "Y"; }
    if code == 1 { return "M"; }
    if code == 2 { return "W"; }
    if code == 3 { return "D"; }
    if code == 4 { return "h"; }
    if code == 5 { return "m"; }
    if code == 6 { return "s"; }
    if code == 7 { return "ms"; }
    if code == 8 { return "us"; }
    if code == 9 { return "ns"; }
    if code == 10 { return "ps"; }
    if code == 11 { return "fs"; }
    if code == 12 { return "as"; }
    return "generic";
}

fn test_datetime_data_one() {
    let unit, count, f = scientific::datetime_data("M8[5ms]");
    assert(f.kind == scientific::F_NONE && unit == "ms" && count == 5);
    let unit2, count2, f2 = scientific::datetime_data("M8");
    assert(f2.kind == scientific::F_NONE && unit2 == "generic" && count2 == 1);
    let unit3, count3, f3 = scientific::datetime_data("i8");
    assert(f3.kind == scientific::F_VALUE);
}
