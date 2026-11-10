// The functions of scipy.constants: the lookup in the table of the
// physical constants, the change of temperature scale and the change
// between wavelength and frequency. The numbers live in the constants
// module.
namespace scientific;

// lower_byte gives the lower case of an ASCII letter.
fn lower_byte(b: u8) -> u8 {
    if b >= 65 && b <= 90 { return b + 32; }
    return b;
}

// same_text tells whether two texts hold the same letters when the
// case does not count.
fn same_text(a: const u8[:], b: const u8[:]) -> bool {
    if a.len != b.len { return false; }
    var i: i64 = 0;
    while i < a.len {
        if lower_byte(a[i]) != lower_byte(b[i]) { return false; }
        i += 1;
    }
    return true;
}

// line_start gives the index where the line of a number starts in a
// text of lines. The end of the text gives its length.
fn line_start(text: const u8[:], number: i64) -> i64 {
    var at: i64 = 0;
    var line: i64 = 0;
    while line < number && at < text.len {
        if text[at] == 10 { line += 1; }
        at += 1;
    }
    return at;
}

// line_end gives the index of the end of the line that starts at an
// index.
fn line_end(text: const u8[:], start: i64) -> i64 {
    var at = start;
    while at < text.len && text[at] != 10 { at += 1; }
    return at;
}

// constant_index gives the place of a name in the table. A name that
// the table lacks gives -1. The case of the name counts, as it does
// in SciPy.
fn constant_index(key: const u8[:]) -> i64 {
    var number: i64 = 0;
    var at: i64 = 0;
    let names = constants::NAMES;
    while number < constants::COUNT {
        let end = line_end(names, at);
        if end - at == key.len {
            var same = true;
            var i: i64 = 0;
            while i < key.len {
                if names[at + i] != key[i] { same = false; break; }
                i += 1;
            }
            if same { return number; }
        }
        at = end + 1;
        number += 1;
    }
    return -1;
}

// value gives the value of a physical constant by its name. A name
// that the table lacks is a fault, as SciPy raises an error.
fn value(key: const u8[:]) -> f64, Fault {
    let i = constant_index(key);
    if i < 0 { return 0.0, fault(F_VALUE, "value", "the table holds no constant of this name"); }
    return constants::VALUES[i], no_fault();
}

// unit gives the unit of a physical constant by its name.
fn unit(key: const u8[:]) -> const u8[:], Fault {
    let i = constant_index(key);
    if i < 0 { return "", fault(F_VALUE, "unit", "the table holds no constant of this name"); }
    let start = line_start(constants::UNITS, i);
    return constants::UNITS[start..line_end(constants::UNITS, start)], no_fault();
}

// precision gives the relative uncertainty of a physical constant: the
// uncertainty divided by the value. The sign of the value stays, so a
// constant below zero gives a result below zero, as SciPy gives it.
fn precision(key: const u8[:]) -> f64, Fault {
    let i = constant_index(key);
    if i < 0 { return 0.0, fault(F_VALUE, "precision", "the table holds no constant of this name"); }
    return constants::UNCERTAINTIES[i] / constants::VALUES[i], no_fault();
}

// find gives the names of the physical constants that hold a text,
// one name per line, in sorted order. The case does not count. An
// empty text gives every name.
fn find(sub: const u8[:]) -> u8[>..] {
    var out: u8[>..] = [];
    var number: i64 = 0;
    var at: i64 = 0;
    let names = constants::NAMES;
    while number < constants::COUNT {
        let end = line_end(names, at);
        var hit = sub.len == 0;
        var s: i64 = at;
        while !hit && s + sub.len <= end {
            var same = true;
            var i: i64 = 0;
            while i < sub.len {
                if lower_byte(names[s + i]) != lower_byte(sub[i]) { same = false; break; }
                i += 1;
            }
            if same { hit = true; }
            s += 1;
        }
        if hit {
            if out.len > 0 { out.push(10); }
            var k = at;
            while k < end { out.push(names[k]); k += 1; }
        }
        at = end + 1;
        number += 1;
    }
    return out;
}

// scale_code gives the number of a temperature scale: 0 Celsius, 1
// Kelvin, 2 Fahrenheit, 3 Rankine. The short names C, K, F and R
// count too. The case does not count. Any other name gives -1.
fn scale_code(name: const u8[:]) -> i64 {
    if same_text(name, "celsius") || same_text(name, "c") { return 0; }
    if same_text(name, "kelvin") || same_text(name, "k") { return 1; }
    if same_text(name, "fahrenheit") || same_text(name, "f") { return 2; }
    if same_text(name, "rankine") || same_text(name, "r") { return 3; }
    return -1;
}

// convert_temperature changes the scale of temperatures. The steps are
// the steps of SciPy, so the digits agree.
fn convert_temperature(a, old_scale: const u8[:], new_scale: const u8[:]) -> Arr, Fault {
    let from = scale_code(old_scale);
    let to = scale_code(new_scale);
    if from < 0 || to < 0 {
        return empty_arr(), fault(F_VALUE, "convert_temperature", "a scale is not Celsius, Kelvin, Fahrenheit or Rankine");
    }
    let zero = constants::zero_Celsius;
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len {
        let v = a.data[i];
        var kelvin = v;
        if from == 0 { kelvin = v + zero; }
        else if from == 2 { kelvin = (v - 32.0) * 5.0 / 9.0 + zero; }
        else if from == 3 { kelvin = v * 5.0 / 9.0; }
        var r = kelvin;
        if to == 0 { r = kelvin - zero; }
        else if to == 2 { r = (kelvin - zero) * 9.0 / 5.0 + 32.0; }
        else if to == 3 { r = kelvin * 9.0 / 5.0; }
        d.push(r);
        i += 1;
    }
    return Arr { ndim: a.ndim, shape: same_shape(a), data: d }, no_fault();
}

// lambda2nu gives the frequencies of the wavelengths: the speed of
// light divided by each wavelength.
fn lambda2nu(a) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { d.push(constants::c / a.data[i]); i += 1; }
    return Arr { ndim: a.ndim, shape: same_shape(a), data: d }, no_fault();
}

// nu2lambda gives the wavelengths of the frequencies: the speed of
// light divided by each frequency.
fn nu2lambda(a) -> Arr, Fault {
    return lambda2nu(a);
}
