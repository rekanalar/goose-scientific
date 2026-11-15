// The text of a float: format_float_positional and
// format_float_scientific of NumPy.
//
// NumPy prints a float with the Dragon4 algorithm. Its unique mode gives
// the fewest digits that read back as the same number. Its other mode
// gives the digits of the exact value, rounded half to even. This file
// does both. The unique digits come from the shortest form of the
// standard library, and the exact digits come from a decimal expansion
// of the number, built with whole-number steps. The options of NumPy
// stand in one struct, because Goose has no named arguments.
namespace scientific;

import json;

// The limit of every count of digits and every pad. A float holds at
// most 1074 digits after the point, so a larger count adds zeros only.
const TEXT_LIMIT: i64 = 4096;

// FloatFormat holds the options of the two calls. A precision of -1
// stands for no precision. A pad of 0 stands for no pad. An exp_digits
// of -1 stands for the default of two digits. The trim is one of the
// bytes k, ., 0 and -.
struct FloatFormat {
    precision: i64, unique: bool, fractional: bool, trim: u8, sign: bool,
    pad_left: i64, pad_right: i64, min_digits: i64, exp_digits: i64
}

// float_format gives the defaults of NumPy: the unique digits, no
// precision, trim k, no sign and no pad.
fn float_format() -> FloatFormat {
    return FloatFormat {
        precision: -1, unique: true, fractional: true, trim: 'k', sign: false,
        pad_left: 0, pad_right: 0, min_digits: 0, exp_digits: -1
    };
}

// Dec holds the digits of a positive number: the value is 0.d1 d2 ...
// times 10^point. Only the first n digits count. A count of zero is
// the number zero.
struct Dec { n: i64, point: i64, carried: bool, d: u8[>..] }

// dec_new gives the empty digits of the number zero.
fn dec_new() -> Dec {
    return Dec { n: 0, point: 1, carried: false, d: [] };
}

// dec_at gives the digit at a place, and zero outside the digits.
fn dec_at(v: Dec&, i: i64) -> i64 {
    if i < 0 || i >= v.n { return 0; }
    return v.d[i] as i64;
}

// dec_strip drops the zeros at the end of the digits.
fn dec_strip(v: Dec&) {
    while v.n > 0 && v.d[v.n - 1] == 0 { v.n -= 1; }
}

// dec_mul multiplies the little endian digits by a small factor.
fn dec_mul(le: u8[>..]&, f: i64) {
    var carry: i64 = 0;
    var i: i64 = 0;
    while i < le.len {
        let t = (le[i] as i64) * f + carry;
        le[i] = (t % 10) as u8;
        carry = t / 10;
        i += 1;
    }
    while carry > 0 {
        le.push((carry % 10) as u8);
        carry = carry / 10;
    }
}

// dec_exact gives the exact digits of a finite number above zero. The
// number is m * 2^e with m below 2^53, so the digits of m times a power
// of two or of five are the digits of the number.
fn dec_exact(x: f64, v: Dec&) {
    var m = x;
    var e: i64 = 0;
    while m >= 9007199254740992.0 { m = m / 2.0; e += 1; }
    while m < 4503599627370496.0 { m = m * 2.0; e -= 1; }
    var le: u8[>..] = [];
    var w = m as i64;
    while w > 0 {
        le.push((w % 10) as u8);
        w = w / 10;
    }
    var shift: i64 = 0;
    if e >= 0 {
        var k = e;
        while k >= 20 { dec_mul(le, 1048576); k -= 20; }
        if k > 0 { dec_mul(le, 1 << k); }
    } else {
        // m * 2^e = m * 5^(-e) / 10^(-e)
        shift = -e;
        var k = -e;
        while k >= 9 { dec_mul(le, 1953125); k -= 9; }
        while k > 0 { dec_mul(le, 5); k -= 1; }
    }
    v.d.clear();
    var i = le.len - 1;
    while i >= 0 { v.d.push(le[i]); i -= 1; }
    v.n = le.len;
    v.point = le.len - shift;
    dec_strip(v);
}

// dec_shortest gives the fewest digits that read back as the number.
// The standard library writes them as text, and this call reads the
// text back into digits.
fn dec_shortest(x: f64, v: Dec&) {
    var text: u8[>..] = [];
    json::format_number(text, x);
    v.d.clear();
    var ints: i64 = 0;
    var seen_point = false;
    var i: i64 = 0;
    while i < text.len && text[i] != 'e' {
        if text[i] == '.' { seen_point = true; }
        else {
            v.d.push(text[i] - '0');
            if !seen_point { ints += 1; }
        }
        i += 1;
    }
    var ex: i64 = 0;
    if i < text.len {
        i += 1;
        var neg = false;
        if i < text.len && text[i] == '-' { neg = true; i += 1; }
        while i < text.len { ex = ex * 10 + ((text[i] - '0') as i64); i += 1; }
        if neg { ex = -ex; }
    }
    var lead: i64 = 0;
    while lead < v.d.len && v.d[lead] == 0 { lead += 1; }
    // The digits move to the front in a fresh buffer, so the buffer
    // holds no stale digit behind the count.
    var fresh: u8[>..] = [];
    var j: i64 = 0;
    while lead + j < v.d.len { fresh.push(v.d[lead + j]); j += 1; }
    v.d = copy(fresh);
    v.n = fresh.len;
    v.point = ints + ex - lead;
    dec_strip(v);
}

// dec_bump rounds up. The first m digits of v are the digits that stay.
// The last of them grows by one, the nines behind it fall away, and a
// run of nines at the front gives a one at a place above. The drop flag
// takes the NumPy form: the digits that the carry ran through leave the
// count. Without the flag they turn to zeros and the count stays.
fn dec_bump(v: Dec&, m: i64, drop: bool) {
    var i = m - 1;
    while i >= 0 && v.d[i] == 9 { i -= 1; }
    if i < 0 {
        v.d.clear();
        v.d.push(1);
        v.n = 1;
        v.point += 1;
        v.carried = true;
        return;
    }
    v.d[i] = ((v.d[i] as i64) + 1) as u8;
    if drop { v.n = i + 1; v.carried = true; return; }
    var k = i + 1;
    while k < m { v.d[k] = 0; k += 1; }
    v.n = m;
}

// dec_round_at rounds the digits at a place. The place is the lowest
// position that stays, counted in places after the point, so a place
// of -3 keeps the digits at -1 and -2 and -3. The digits below the
// place decide it: more than half rounds up, less than half rounds
// down, and exactly half rounds to the even digit. The drop flag goes
// to dec_bump.
fn dec_round_at(v: Dec&, place: i64, drop: bool) {
    let keep = v.point - place;
    if keep < 0 {
        // The number is below the place by more than one digit, so its
        // value is below half of the step of the place.
        v.n = 0;
        v.point = place;
        return;
    }
    if keep == 0 {
        // The number itself is the tail that the place rounds.
        if v.n == 0 { v.point = place; return; }
        let d = v.d[0];
        if d > 5 { dec_up(v, place); return; }
        if d < 5 { v.n = 0; v.point = place; return; }
        var more = false;
        var i: i64 = 1;
        while i < v.n { if v.d[i] != 0 { more = true; } i += 1; }
        if more { dec_up(v, place); return; }
        // The number is exactly half. No digit stays, and zero is even.
        v.n = 0;
        v.point = place;
        return;
    }
    if keep >= v.n { return; }
    let d = v.d[keep];
    if d > 5 { dec_bump(v, keep, drop); return; }
    if d < 5 { v.n = keep; return; }
    var more = false;
    var i = keep + 1;
    while i < v.n { if v.d[i] != 0 { more = true; } i += 1; }
    if more { dec_bump(v, keep, drop); return; }
    // The number is exactly half. The last digit that stays decides.
    if v.d[keep - 1] % 2 == 1 { dec_bump(v, keep, drop); return; }
    v.n = keep;
}

// dec_up rounds a number below a place up to a one at the place.
fn dec_up(v: Dec&, place: i64) {
    v.d.clear();
    v.d.push(1);
    v.n = 1;
    v.point = place + 1;
    v.carried = true;
}

// frac_count gives the count of the digits after the point.
fn frac_count(v: Dec&) -> i64 {
    let c = v.n - v.point;
    if c < 0 { return 0; }
    return c;
}

// short_fits tells whether the shortest digits stay inside the count
// of digits that the precision allows. The scientific mode counts the
// digits before the point, so one digit more is allowed there.
fn short_fits(v: Dec&, o: FloatFormat, sci: bool) -> bool {
    if sci { return v.n <= o.precision + 1; }
    if !o.fractional { return v.n <= o.precision; }
    return frac_count(v) <= o.precision;
}

// short_long tells whether the shortest digits hold the count of
// digits that min_digits asks for.
fn short_long(v: Dec&, o: FloatFormat, sci: bool) -> bool {
    if sci { return v.n >= o.min_digits + 1; }
    if !o.fractional { return v.n >= o.min_digits; }
    return frac_count(v) >= o.min_digits;
}

// cut_place gives the lowest position of the digits that stay. The
// least flag chooses the cut of min_digits, which asks for digits,
// against the cut of the precision, which caps them. The scientific
// mode and the form without fractions count significant digits, so
// the place hangs on the point of the digits that get cut.
fn cut_place(v: Dec&, o: FloatFormat, sci: bool, least: bool) -> i64 {
    if sci {
        var c = if least { o.min_digits + 1 } else { o.precision + 1 };
        if c < 1 { c = 1; }
        return v.point - c;
    }
    if !o.fractional {
        var c = if least { o.min_digits } else { o.precision };
        if c < 1 { c = 1; }
        return v.point - c;
    }
    let c = if least { o.min_digits } else { o.precision };
    return -c;
}

// float_digits chooses the digits of a finite number by the options.
// The unique mode keeps the shortest digits, unless a precision cuts
// them or min_digits asks for more. The other mode rounds the exact
// digits half to even at the cut. The form without fractions follows
// the NumPy rule: the carry drops the digits it runs through.
fn float_digits(x: f64, o: FloatFormat, sci: bool, v: Dec&) {
    v.n = 0;
    v.point = 1;
    v.carried = false;
    if x == 0.0 { return; }
    let ax = if x < 0.0 { -x } else { x };
    // The carry drops the digits it ran through in the scientific form
    // and in the form without fractions.
    let drop = sci || !o.fractional;
    if !o.unique {
        dec_exact(ax, v);
        dec_round_at(v, cut_place(v, o, sci, false), drop);
        return;
    }
    dec_shortest(ax, v);
    if o.precision >= 0 && !short_fits(v, o, sci) {
        dec_exact(ax, v);
        dec_round_at(v, cut_place(v, o, sci, false), drop);
        return;
    }
    if o.min_digits > 0 && !short_long(v, o, sci) {
        // The exact digits take the rounding that min_digits asks for,
        // and the scientific form pads its digits to the count. The
        // zeros that come last do not change the number.
        dec_exact(ax, v);
        dec_round_at(v, cut_place(v, o, sci, true), drop);
        // A carry drops its digits and the trim pads them back. Digits
        // that ran out stay as printed zeros.
        if sci && !v.carried {
            let need = o.min_digits + 1;
            if v.n < need {
                var fresh: u8[>..] = [];
                var i: i64 = 0;
                while i < v.n { fresh.push(v.d[i]); i += 1; }
                while fresh.len < need { fresh.push(0); }
                v.d = copy(fresh);
                v.n = fresh.len;
            }
        }
        return;
    }
}

fn text_bad(tag: const u8[:], why: const u8[:]) -> Fault {
    return fault(F_VALUE, tag, why);
}

// float_check tests the options. A fault names the first bad one.
fn float_check(tag: const u8[:], o: FloatFormat, sci: bool) -> Fault {
    if o.trim != 'k' && o.trim != '.' && o.trim != '0' && o.trim != '-' {
        return text_bad(tag, "the trim is not k, a point, 0 or a minus");
    }
    if o.precision < -1 { return text_bad(tag, "the precision is below zero"); }
    if o.pad_left < 0 || o.pad_right < 0 || o.min_digits < 0 || o.exp_digits < -1 {
        return text_bad(tag, "a pad, min_digits or exp_digits is below zero");
    }
    if !o.unique && o.precision < 0 { return text_bad(tag, "a precision is needed when unique is false"); }
    if !sci && !o.fractional && o.precision == 0 {
        return text_bad(tag, "the precision must be above zero when fractional is false");
    }
    if o.min_digits > 0 && o.precision > 0 && o.min_digits > o.precision {
        return text_bad(tag, "min_digits must not exceed the precision");
    }
    if o.precision > TEXT_LIMIT || o.pad_left > TEXT_LIMIT || o.pad_right > TEXT_LIMIT
        || o.min_digits > TEXT_LIMIT || o.exp_digits > TEXT_LIMIT {
        return fault(F_LIMIT, tag, "a count of digits or a pad is above the limit of 4096");
    }
    return no_fault();
}

// wanted gives the count of the digits after the point that the
// options ask for. The unique mode pads to min_digits only, and the
// form without fractions counts the digits in all, so the digits
// before the point come off.
fn wanted(o: FloatFormat, sci: bool, whole: i64) -> i64 {
    var add: i64 = 0;
    if o.unique { add = o.min_digits; } else { add = o.precision; }
    if add < 0 { return 0; }
    if !sci && !o.fractional { return add - whole; }
    return add;
}

fn push_spaces(out: u8[>..]&, count: i64) {
    var i: i64 = 0;
    while i < count { out.push(' '); i += 1; }
}

// float_text writes a finite or infinite number. The sci flag chooses
// the scientific form.
fn float_text(x: f64, o: FloatFormat, sci: bool, tag: const u8[:]) -> u8[>..], Fault {
    var out: u8[>..] = [];
    let f = float_check(tag, o, sci);
    if f.kind != F_NONE { return out, f; }
    let neg = x < 0.0 || (x == 0.0 && 1.0 / x < 0.0);
    if x != x { out.append("nan"); return out, no_fault(); }
    if x - x != 0.0 {
        if neg { out.push('-'); } else if o.sign { out.push('+'); }
        out.append("inf");
        return out, no_fault();
    }
    var v = dec_new();
    float_digits(x, o, sci, v);
    // The digits before the point and after it.
    var whole: u8[>..] = [];
    var frac: u8[>..] = [];
    var exp: i64 = 0;
    if sci {
        var first: u8 = 0;
        if v.n > 0 { first = v.d[0]; }
        whole.push((('0' as i64) + (first as i64)) as u8);
        var i: i64 = 1;
        while i < v.n { frac.push((('0' as i64) + (v.d[i] as i64)) as u8); i += 1; }
        exp = v.point - 1;
    } else if v.point > 0 {
        var i: i64 = 0;
        while i < v.point {
            whole.push((('0' as i64) + dec_at(v, i)) as u8);
            i += 1;
        }
        var j = v.point;
        while j < v.n {
            frac.push((('0' as i64) + (v.d[j] as i64)) as u8);
            j += 1;
        }
    } else {
        whole.push('0');
        var z: i64 = 0;
        while z < -v.point { frac.push('0'); z += 1; }
        var i: i64 = 0;
        while i < v.n { frac.push((('0' as i64) + (v.d[i] as i64)) as u8); i += 1; }
    }
    // The point and the digits after it. The point falls away in the
    // trim of a minus when no digit stays behind it. The scientific
    // form keeps its point when the number has digits behind its first
    // one, even when the trim empties them.
    var keep = frac.len;
    var base = keep > 0;
    if sci { base = v.n > 1; }
    var point = base || o.trim != '-';
    if o.trim == '0' && keep == 0 {
        frac.push('0');
        keep = 1;
        point = true;
    } else if o.trim == 'k' {
        let need = wanted(o, sci, whole.len);
        while keep < need { frac.push('0'); keep += 1; }
    }
    if o.trim != 'k' && keep > 0 {
        while keep > 0 && frac[keep - 1] == '0' { keep -= 1; }
        if o.trim == '0' && keep == 0 {
            frac.push('0');
            keep = 1;
            point = true;
        }
        if o.trim == '-' && keep == 0 && !sci { point = false; }
    }
    // The pad at the left goes in front of the sign.
    var head = whole.len;
    if neg || o.sign { head += 1; }
    if o.pad_left > head { push_spaces(out, o.pad_left - head); }
    if neg { out.push('-'); } else if o.sign { out.push('+'); }
    out.append(whole);
    if point { out.push('.'); }
    var i: i64 = 0;
    while i < keep { out.push(frac[i]); i += 1; }
    if sci {
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        var e = if exp < 0 { -exp } else { exp };
        var ed: u8[>..] = [];
        while e > 0 { ed.push((('0' as i64) + (e % 10)) as u8); e = e / 10; }
        // The exponent holds two digits by default, and NumPy holds it
        // at five. A count of zero gives the sign alone.
        var width = if o.exp_digits < 0 { 2 } else { o.exp_digits };
        if width > 5 { width = 5; }
        while ed.len < width { ed.push('0'); }
        var k = ed.len - 1;
        while k >= 0 { out.push(ed[k]); k -= 1; }
    } else {
        // The pad at the right. A trimmed point still holds its place,
        // so it takes a space of its own.
        if o.pad_right > keep {
            var count = o.pad_right - keep;
            if !point && keep == 0 { count += 1; }
            push_spaces(out, count);
        }
    }
    return out, no_fault();
}

// format_float_positional writes a number as a decimal text without an
// exponent, as NumPy does. The options stand in a FloatFormat.
fn format_float_positional(x: f64, o: FloatFormat) -> u8[>..], Fault {
    return float_text(x, o, false, "format_float_positional");
}

// format_float_scientific writes a number as a decimal text with an
// exponent, as NumPy does. The fractional option does not apply.
fn format_float_scientific(x: f64, o: FloatFormat) -> u8[>..], Fault {
    return float_text(x, o, true, "format_float_scientific");
}
