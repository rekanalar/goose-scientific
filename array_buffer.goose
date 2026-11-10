// The input of a file or of raw bytes: frombuffer and fromfile.
//
// NumPy reads the bytes of a buffer or a file as numbers of the
// machine. The package holds one type of number, so the bytes decode
// as numbers of 64 bits, least significant byte first, as NumPy
// decodes them. The decode runs in the package, because the standard
// library of both supported Goose versions holds no read of the bits.
namespace scientific;

import os;

// f64_of_bits builds the number of a pattern of bits. The rule of the
// machine holds: one part holds the sign, one part holds the power of
// two, and one part holds the digits. A power of 2047 with no digits
// holds an infinity, and with digits holds the missing value. A power
// of zero holds a number below the smallest normal.
fn f64_of_bits(bits: u64) -> f64 {
    let sign = bits >> 63;
    let exp = (bits >> 52) & 2047;
    let frac = bits & 4503599627370495;
    var v: f64 = 0.0;
    if exp == 2047 {
        if frac == 0 {
            v = 1.0 / 0.0;
        } else {
            v = 0.0 / 0.0;
        }
    } else if exp > 0 {
        // The digits give the part above one. The number holds 52
        // digits, so the product and the sum lose no digit.
        let mant = 1.0 + frac * 2.220446049250313e-16;
        v = mant * power_of_two(exp - 1023.0);
    } else {
        // The numbers below the smallest normal hold no part above
        // one. The digits carry the whole value.
        v = frac * 4.9406564584124654e-324;
    }
    if sign == 1 { return -v; }
    return v;
}

// power_of_two gives a power of two. The power of the standard
// library holds every power of two exactly.
fn power_of_two(e: f64) -> f64 {
    if e == 0.0 { return 1.0; }
    return mathshim::power(2.0, e);
}

// bits_of reads 8 bytes of a buffer as one pattern of bits, least
// significant byte first. The caller checks the range.
fn bits_of(data: const u8[:], at: i64) -> u64 {
    var bits: u64 = data[at];
    var k: i64 = 1;
    while k < 8 {
        bits = bits | (data[at + k] as u64 << (8 * k));
        k += 1;
    }
    return bits;
}

// frombuffer decodes the bytes of a buffer as numbers of 64 bits. An
// offset must sit on a number. The count of -1 takes every number of
// the rest. A count above the rest is a fault, as NumPy refuses it.
fn frombuffer(data: const u8[:], count: i64 = -1, offset: i64 = 0) -> Arr, Fault {
    if offset < 0 || offset > data.len {
        return empty_arr(), fault(F_VALUE, "frombuffer", "the offset holds a value outside the buffer");
    }
    let rem = data.len - offset;
    if offset % 8 != 0 || rem % 8 != 0 {
        return empty_arr(), fault(F_VALUE, "frombuffer", "the buffer holds a count of bytes that no count of numbers holds");
    }
    let avail = rem / 8;
    if count > avail {
        return empty_arr(), fault(F_VALUE, "frombuffer", "the buffer holds fewer numbers than the count asks for");
    }
    var n = avail;
    if count >= 0 { n = count; }
    if n > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "frombuffer", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        d.push(f64_of_bits(bits_of(data, offset + i * 8)));
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// fromfile decodes the bytes of a file as numbers of 64 bits. A file
// that does not read is a fault. A count above the file takes what
// the file holds, as NumPy gives it. A rest of a few bytes at the end
// drops, as NumPy drops it.
fn fromfile(path: const u8[:], count: i64 = -1) -> Arr, Fault {
    var bytes: u8[>..] = [];
    if !read_file(path, bytes) {
        return empty_arr(), fault(F_VALUE, "fromfile", "the file does not read");
    }
    var n = bytes.len / 8;
    if count >= 0 && count < n { n = count; }
    return frombuffer(bytes[..n * 8], n, 0);
}
