// The text of whole numbers and of type codes: base_repr, binary_repr,
// typename and mintypecode of NumPy.
namespace scientific;

import array_dtype;

const DIGITS36: const u8[:] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

// The type codes in the order of their size, as NumPy keeps them. The
// first code of a set of codes wins.
const CODES_BY_SIZE: const u8[:] = "GDFgdfQqLlIiHhBb?";

// magnitude_of gives the absolute value of an i64 as a u64. The value
// -2^63 has a magnitude that an i64 cannot hold.
fn magnitude_of(number: i64) -> u64 {
    if number >= 0 { return number as u64; }
    return 0 - (number as u64);
}

// bit_length gives the count of bits of a magnitude. A zero has none.
fn bit_length(mag: u64) -> i64 {
    var n: i64 = 0;
    var m = mag;
    while m > 0 { n += 1; m = m >> 1; }
    return n;
}

// bits_of gives the binary text of a magnitude, with no zeros in front.
fn bits_of(mag: u64) -> u8[>..] {
    var out: u8[>..] = [];
    var i = bit_length(mag) - 1;
    while i >= 0 {
        out.push((('0') + (((mag >> (i as u64)) & 1) as i64)) as u8);
        i -= 1;
    }
    return out;
}

// base_repr writes a whole number in a base from 2 to 36. The digits
// are 0 to 9 and then A to Z. A number below zero gets a minus sign.
// The padding adds that many zeros in front of the digits, after the
// sign. A padding below zero adds none. The number zero with a
// padding gives the zeros alone, as NumPy gives them.
fn base_repr(number: i64, base: i64 = 2, padding: i64 = 0) -> u8[>..], Fault {
    var out: u8[>..] = [];
    if base < 2 { return out, fault(F_VALUE, "base_repr", "the base is below 2"); }
    if base > 36 { return out, fault(F_VALUE, "base_repr", "the base is above 36"); }
    if padding > TEXT_LIMIT { return out, fault(F_LIMIT, "base_repr", "the padding is above the limit of 4096"); }
    var le: u8[>..] = [];
    var num = magnitude_of(number);
    while num > 0 {
        le.push(DIGITS36[(num % (base as u64)) as i64]);
        num = num / (base as u64);
    }
    // The padding goes in front of the digits, and the sign in front of
    // the padding. NumPy counts a padding below zero as a padding that
    // holds no zeros, and the number zero then has no text at all.
    var zeros = padding;
    if zeros < 0 { zeros = 0; }
    var i: i64 = 0;
    while i < zeros { le.push('0'); i += 1; }
    if number < 0 { le.push('-'); }
    if le.len == 0 {
        if padding != 0 { return out, no_fault(); }
        out.push('0');
        return out, no_fault();
    }
    var k = le.len - 1;
    while k >= 0 { out.push(le[k]); k -= 1; }
    return out, no_fault();
}

// binary_repr writes a whole number in binary. A number below zero
// gets a minus sign. A width gives the two's complement of a negative
// number in that many bits, and gives a number of zero or more the
// zeros in front that it lacks. A width of -1 gives no width. A width
// too small for the number is a fault, as in NumPy. The number zero
// with a width below one gives the zeros of that width, and with no
// width gives a single zero.
fn binary_repr(number: i64, width: i64 = -1) -> u8[>..], Fault {
    var out: u8[>..] = [];
    if width > TEXT_LIMIT { return out, fault(F_LIMIT, "binary_repr", "the width is above the limit of 4096"); }
    if number == 0 {
        if width < 0 {
            if width == -1 { out.push('0'); }
            return out, no_fault();
        }
        if width == 0 { out.push('0'); return out, no_fault(); }
        var i: i64 = 0;
        while i < width { out.push('0'); i += 1; }
        return out, no_fault();
    }
    if number > 0 {
        let bits = bits_of(number as u64);
        let bw = bits.len;
        if width >= 0 && width < bw {
            return out, fault(F_VALUE, "binary_repr", "the width is too small for the number");
        }
        var outw = bw;
        if width > bw { outw = width; }
        var i = outw - bw;
        while i > 0 { out.push('0'); i -= 1; }
        out.append(bits);
        return out, no_fault();
    }
    // A number below zero with no width gets the digits of its size.
    let mag = magnitude_of(number);
    if width == -1 {
        out.push('-');
        out.append(bits_of(mag));
        return out, no_fault();
    }
    // With a width, the number gets the two's complement. The width of
    // the size loses one bit when the size is a power of two, as NumPy
    // does.
    var poswidth = bit_length(mag);
    if mag == ((1 as u64) << ((poswidth - 1) as u64)) { poswidth -= 1; }
    var twocomp: u64 = 0;
    if poswidth >= 63 {
        twocomp = ((1 as u64) << 63) + (((1 as u64) << 63) - mag);
    } else {
        twocomp = ((1 as u64) << ((poswidth + 1) as u64)) - mag;
    }
    let bits = bits_of(twocomp);
    let bw = bits.len;
    if width < bw {
        return out, fault(F_VALUE, "binary_repr", "the width is too small for the number");
    }
    var outw = bw;
    if width > bw { outw = width; }
    var i = outw - bw;
    while i > 0 { out.push('1'); i -= 1; }
    out.append(bits);
    return out, no_fault();
}

// typename gives the description that NumPy keeps for a type code, such
// as "double precision" for d. NumPy 2.5 deprecates the call, and this
// call gives no warning. A code that NumPy does not know is a fault.
// The names of the 14 types of the package are names as well, and each
// of them gives itself back.
fn typename(char_: const u8[:]) -> const u8[:], Fault {
    if char_ == "S1" { return "character", no_fault(); }
    if char_ == "?" { return "bool", no_fault(); }
    if char_ == "b" { return "signed char", no_fault(); }
    if char_ == "B" { return "unsigned char", no_fault(); }
    if char_ == "h" { return "short", no_fault(); }
    if char_ == "H" { return "unsigned short", no_fault(); }
    if char_ == "i" { return "integer", no_fault(); }
    if char_ == "I" { return "unsigned integer", no_fault(); }
    if char_ == "l" { return "long integer", no_fault(); }
    if char_ == "L" { return "unsigned long integer", no_fault(); }
    if char_ == "q" { return "long long integer", no_fault(); }
    if char_ == "Q" { return "unsigned long long integer", no_fault(); }
    if char_ == "f" { return "single precision", no_fault(); }
    if char_ == "d" { return "double precision", no_fault(); }
    if char_ == "g" { return "long precision", no_fault(); }
    if char_ == "F" { return "complex single precision", no_fault(); }
    if char_ == "D" { return "complex double precision", no_fault(); }
    if char_ == "G" { return "complex long double precision", no_fault(); }
    if char_ == "S" { return "string", no_fault(); }
    if char_ == "U" { return "unicode", no_fault(); }
    if char_ == "V" { return "void", no_fault(); }
    if char_ == "O" { return "object", no_fault(); }
    if dt_index(char_) >= 0 { return char_, no_fault(); }
    return "", fault(F_VALUE, "typename", "the type code is not one that NumPy names");
}

// mintypecode gives the code of the smallest type, taken from a set of
// codes, that holds every type in typechars. It takes the codes of
// NumPy for a text of single letters. A letter outside the set is
// ignored. A text with no letter of the set gives the default. The two
// letters F and d together give D. The letter G has no type in the
// package, so a set that holds it gives a fault.
fn mintypecode(typechars: const u8[:], typeset: const u8[:] = "GDFgdf", default_: u8 = 'd') -> u8, Fault {
    var has_f = false;
    var has_d = false;
    var best: i64 = CODES_BY_SIZE.len;
    var any = false;
    for c in typechars {
        if !has_in(typeset, c) { continue; }
        any = true;
        if c == 'G' {
            return default_, fault(F_VALUE, "mintypecode", "the package has no type for the code G");
        }
        var at: i64 = 0;
        while at < CODES_BY_SIZE.len && CODES_BY_SIZE[at] != c { at += 1; }
        if at >= CODES_BY_SIZE.len {
            return default_, fault(F_VALUE, "mintypecode", "the set holds a code that NumPy does not order");
        }
        if c == 'F' { has_f = true; }
        if c == 'd' { has_d = true; }
        if at < best { best = at; }
    }
    if !any { return default_, no_fault(); }
    if has_f && has_d { return 'D', no_fault(); }
    return CODES_BY_SIZE[best], no_fault();
}

// has_in tells whether a text holds a byte.
fn has_in(text: const u8[:], c: u8) -> bool {
    for t in text { if t == c { return true; } }
    return false;
}
