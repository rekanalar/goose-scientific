// The text of a date, and the unit of a date type: datetime_as_string
// and datetime_data of NumPy.
//
// A date is a whole number of days since 1970-01-01, so the text of a
// date is the form YYYY-MM-DD, as NumPy writes a date of the unit D. A
// date that is not a time gives the text NaT. The package holds the
// years 1 to 9999, so the year has four digits.
//
// datetime_data reads the name of a date type, as in M8[5ms], and
// gives the unit and the count of the unit. A name without a unit has
// the unit generic and the count 1. The form with a divisor, as in
// M8[ms/1000], turns into a multiple of a smaller unit by the table of
// NumPy: M8[ms/1000] gives the unit us and the count 1. A divisor that
// no smaller unit holds as a whole multiple is a fault of kind F_VALUE.
namespace scientific;

// date_digits writes a whole number with a fixed count of digits.
fn date_digits(out: u8[>..]&, value: i64, width: i64) {
    var v = value;
    var tmp: u8[>..] = [];
    while tmp.len < width {
        tmp.push(('0' as i64 + v % 10) as u8);
        v = v / 10;
    }
    var k = tmp.len - 1;
    while k >= 0 { out.push(tmp[k]); k -= 1; }
}

// datetime_as_string_one writes the text of one date. The form is
// YYYY-MM-DD. A date that is not a time gives NaT. A date that is not
// a whole number of days is a fault of kind F_VALUE, and a date outside
// the years 1 to 9999 is a fault of kind F_RANGE.
fn datetime_as_string_one(day: f64) -> u8[>..], Fault {
    var out: u8[>..] = [];
    let d, nat, f = date_of(day, "datetime_as_string");
    if f.kind != F_NONE { return out, f; }
    if nat { out.append("NaT"); return out, no_fault(); }
    var c = CivilDate { year: 0, month: 0, day: 0 };
    civil_of(d, c);
    date_digits(out, c.year, 4);
    out.push('-');
    date_digits(out, c.month, 2);
    out.push('-');
    date_digits(out, c.day, 2);
    return out, no_fault();
}

// datetime_as_string writes the text of each date of a list. A line
// feed stands between the texts.
fn datetime_as_string(dates: Arr) -> u8[>..], Fault {
    var out: u8[>..] = [];
    let f = check_stack(dates.ndim, dates.shape, dates.data.len, 1, "datetime_as_string");
    if f.kind != F_NONE { return out, f; }
    var i: i64 = 0;
    while i < dates.data.len {
        if i > 0 { out.push(10); }
        let t, tf = datetime_as_string_one(dates.data[i]);
        if tf.kind != F_NONE { var none: u8[>..] = []; return none, tf; }
        out.append(t);
        i += 1;
    }
    return out, no_fault();
}

// date_unit_name gives the name of a unit, or an empty text when the
// text is no unit of NumPy.
fn date_unit_name(t: const u8[:]) -> const u8[:] {
    if t == "Y" { return "Y"; }
    if t == "M" { return "M"; }
    if t == "W" { return "W"; }
    if t == "D" { return "D"; }
    if t == "h" { return "h"; }
    if t == "m" { return "m"; }
    if t == "s" { return "s"; }
    if t == "ms" { return "ms"; }
    if t == "us" { return "us"; }
    if t == "ns" { return "ns"; }
    if t == "ps" { return "ps"; }
    if t == "fs" { return "fs"; }
    if t == "as" { return "as"; }
    if t == "generic" { return "generic"; }
    return "";
}

// date_type_head tells where the unit part of a name starts. A name
// starts with M8, m8, datetime64 or timedelta64. The call gives -1
// for any other name.
fn date_type_head(t: const u8[:]) -> i64 {
    if t.len >= 2 && (t[0] == 'M' || t[0] == 'm') && t[1] == '8' { return 2; }
    if t.len >= 10 && t[0] == 'd' && t[1] == 'a' && t[2] == 't' && t[3] == 'e'
        && t[4] == 't' && t[5] == 'i' && t[6] == 'm' && t[7] == 'e'
        && t[8] == '6' && t[9] == '4' { return 10; }
    if t.len >= 11 && t[0] == 't' && t[1] == 'i' && t[2] == 'm' && t[3] == 'e'
        && t[4] == 'd' && t[5] == 'e' && t[6] == 'l' && t[7] == 't'
        && t[8] == 'a' && t[9] == '6' && t[10] == '4' { return 11; }
    return -1;
}

// date_unit_at gives the name of the unit with the index i, in the
// order Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as.
fn date_unit_at(i: i64) -> const u8[:] {
    if i == 0 { return "Y"; }
    if i == 1 { return "M"; }
    if i == 2 { return "W"; }
    if i == 3 { return "D"; }
    if i == 4 { return "h"; }
    if i == 5 { return "m"; }
    if i == 6 { return "s"; }
    if i == 7 { return "ms"; }
    if i == 8 { return "us"; }
    if i == 9 { return "ns"; }
    if i == 10 { return "ps"; }
    if i == 11 { return "fs"; }
    return "as";
}

// date_unit_index gives the index of a unit name, or -1 for a name
// that is no unit of a fixed size. The unit generic has no index.
fn date_unit_index(name: const u8[:]) -> i64 {
    var i: i64 = 0;
    while i < 13 {
        if date_unit_at(i) == name { return i; }
        i += 1;
    }
    return -1;
}

// date_divide turns the unit with the index base and a divisor into a
// multiple of a smaller unit, with the table of NumPy. The result is
// the index of the smaller unit and the factor of the count, or -1 when
// no smaller unit holds the divisor as a whole multiple.
fn date_divide(base: i64, den: i64) -> i64, i64 {
    var tries: i64[..3] = [];
    var to: i64[..3] = [];
    if base == 0 { tries.push(12); tries.push(52); tries.push(365); to.push(1); to.push(2); to.push(3); }
    else if base == 1 { tries.push(4); tries.push(30); tries.push(720); to.push(2); to.push(3); to.push(4); }
    else if base == 2 { tries.push(7); tries.push(168); tries.push(10080); to.push(3); to.push(4); to.push(5); }
    else if base == 3 { tries.push(24); tries.push(1440); tries.push(86400); to.push(4); to.push(5); to.push(6); }
    else if base == 4 { tries.push(60); tries.push(3600); to.push(5); to.push(6); }
    else if base == 5 { tries.push(60); tries.push(60000); to.push(6); to.push(7); }
    else if base <= 10 { tries.push(1000); tries.push(1000000); to.push(base + 1); to.push(base + 2); }
    else if base == 11 { tries.push(1000); to.push(12); }
    var i: i64 = 0;
    while i < tries.len {
        if tries[i] % den == 0 { return to[i], tries[i] / den; }
        i += 1;
    }
    return -1, 0;
}

// datetime_data gives the unit and the count of the unit of a date
// type name, as np.datetime_data does. A name like M8[5ms] gives the
// unit ms and the count 5. A name like M8[ms/1000] gives the unit us
// and the count 1. A name without brackets gives the unit generic and
// the count 1. A name that NumPy does not hold is a fault of kind
// F_VALUE. A count above 2147483647 is a fault of kind F_LIMIT.
fn datetime_data(spec: const u8[:]) -> const u8[:], i64, Fault {
    let tag = "datetime_data";
    let head = date_type_head(spec);
    if head < 0 { return "", 0, fault(F_VALUE, tag, "the name is not a date type"); }
    if spec.len == head { return "generic", 1, no_fault(); }
    if spec[head] != '[' || spec[spec.len - 1] != ']' {
        return "", 0, fault(F_VALUE, tag, "the unit stands between square brackets");
    }
    var i = head + 1;
    let stop = spec.len - 1;
    var count: i64 = 0;
    var digits: i64 = 0;
    while i < stop && spec[i] >= '0' && spec[i] <= '9' {
        count = count * 10 + (spec[i] - '0') as i64;
        digits += 1;
        if count > 2147483647 {
            return "", 0, fault(F_LIMIT, tag, "the count of the unit stands above 2147483647");
        }
        i += 1;
    }
    if digits == 0 { count = 1; }
    if digits > 0 && count == 0 {
        return "", 0, fault(F_VALUE, tag, "the count of the unit is zero");
    }
    var unit: u8[>..] = [];
    while i < stop && spec[i] != '/' { unit.push(spec[i]); i += 1; }
    let name = date_unit_name(unit[..]);
    if name.len == 0 {
        return "", 0, fault(F_VALUE, tag, "the unit is not a unit of NumPy");
    }
    if i >= stop { return name, count, no_fault(); }
    // The divisor: the unit turns into a multiple of a smaller unit.
    i += 1;
    var den: i64 = 0;
    var dd: i64 = 0;
    while i < stop && spec[i] >= '0' && spec[i] <= '9' {
        den = den * 10 + (spec[i] - '0') as i64;
        dd += 1;
        if den > 2147483647 {
            return "", 0, fault(F_LIMIT, tag, "the divisor stands above 2147483647");
        }
        i += 1;
    }
    if dd == 0 || i != stop || den == 0 {
        return "", 0, fault(F_VALUE, tag, "the divisor holds no whole number above zero");
    }
    let base = date_unit_index(name);
    if base < 0 {
        return "", 0, fault(F_VALUE, tag, "a generic unit holds no divisor");
    }
    let to, mult = date_divide(base, den);
    if to < 0 {
        return "", 0, fault(F_VALUE, tag, "the divisor is not a multiple of a smaller unit");
    }
    if count > 2147483647 / mult {
        return "", 0, fault(F_LIMIT, tag, "the count of the unit stands above 2147483647");
    }
    return date_unit_at(to), count * mult, no_fault();
}
