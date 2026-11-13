// The pack of the bits and the unpack of the bits.
//
// The pack puts the values of one axis into bytes. A value stands for a
// bit when it is not zero. The first bit of the axis stands in the high
// place of the byte in the order big, and in the low place in the order
// little. The last byte holds the bits that are left and zeros after
// them.
//
// The unpack takes the bytes of one axis and gives the bits. A byte must
// be a whole number from 0 to 255.
//
// The default axis is the last axis. NumPy lays the whole array flat
// when no axis is named. The caller reaches that form with flatten.
namespace scientific;

// The count of the bits in one byte.
const BITS_PER_BYTE: i64 = 8;

// bits_order checks the order of the bits. It gives 0 for the order big
// and 1 for the order little. An order that does not exist is a fault.
fn bits_order(order: const u8[:], tag: const u8[:]) -> i64, Fault {
    if order == "big" { return 0, no_fault(); }
    if order == "little" { return 1, no_fault(); }
    return -1, fault(F_VALUE, tag, "the order of the bits does not exist");
}

// bits_shift gives the place of the bit k of a byte. The order big puts
// the first bit in the high place, and the order little puts it in the
// low place.
fn bits_shift(k: i64, ord: i64) -> i64 {
    if ord == 1 { return k; }
    return BITS_PER_BYTE - 1 - k;
}

// bits_byte gives the byte of the bits of one axis. The first bit of the
// axis stands in the high place of the byte in the order big, and in the
// low place in the order little.
fn bits_byte(a, outer: i64, width: i64, inner: i64, at: i64, j: i64, ord: i64) -> i64 {
    var byte: i64 = 0;
    let base = outer * width * inner + j;
    var k: i64 = 0;
    while k < BITS_PER_BYTE {
        let idx = at * BITS_PER_BYTE + k;
        if idx < width {
            if a.data[base + idx * inner] != 0.0 {
                byte = byte + (1 << bits_shift(k, ord));
            }
        }
        k += 1;
    }
    return byte;
}

// packbits puts the values of one axis into bytes. The axis may be
// negative, and the default axis is the last axis.
fn packbits(a, axis: i64 = -1, bitorder: const u8[:] = "big") -> Arr, Fault {
    return pack_bits(a, axis, bitorder);
}

// pack_bits puts the values of one axis into bytes.
fn pack_bits(a, axis: i64, order: const u8[:]) -> Arr, Fault {
    let ord, fo = bits_order(order, "packbits");
    if fo.kind != F_NONE { return empty_arr(), fo; }
    // The array must agree with its own shape before any value is read.
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, "packbits");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    let ax = fix_axis(axis, a.ndim);
    if ax < 0 {
        return empty_arr(), fault(F_RANGE, "packbits", "the axis stands outside the shape");
    }
    let width = a.shape[ax];
    let inner = product_after_of(a.shape, ax);
    let outer = product_before_of(a.shape, ax);
    let nbytes = (width + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
    // The count of the elements of the result must not pass the limit.
    if nbytes > MAX_ELEMENTS / (inner * outer) {
        return empty_arr(), fault(F_LIMIT, "packbits", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        var b: i64 = 0;
        while b < nbytes {
            var j: i64 = 0;
            while j < inner {
                d.push(bits_byte(a, o, width, inner, b, j, ord) as f64);
                j += 1;
            }
            b += 1;
        }
        o += 1;
    }
    var s = same_shape(a);
    s[ax] = nbytes;
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}

// bits_check_bytes checks that every value of the array is a byte.
fn bits_check_bytes(a, tag: const u8[:]) -> Fault {
    var i: i64 = 0;
    while i < a.data.len {
        let v = a.data[i];
        if v < 0.0 || v > 255.0 { return fault(F_VALUE, tag, "a byte lies outside zero to 255"); }
        if !whole_place(v) { return fault(F_VALUE, tag, "a byte is not a whole number"); }
        i += 1;
    }
    return no_fault();
}

// unpackbits takes the bytes of an array and gives the bits. The count
// gives the count of the bits along the axis. A count of minus one
// gives all of them, and a count above that number gives zeros after
// them.
fn unpackbits(a, axis: i64 = -1, count: i64 = -1, bitorder: const u8[:] = "big") -> Arr, Fault {
    return unpack_bits(a, axis, count, bitorder);
}

// unpack_bits takes the bytes of one axis and gives the bits.
fn unpack_bits(a, axis: i64, count: i64, order: const u8[:]) -> Arr, Fault {
    let ord, fo = bits_order(order, "unpackbits");
    if fo.kind != F_NONE { return empty_arr(), fo; }
    if count < -1 {
        return empty_arr(), fault(F_RANGE, "unpackbits", "the count of the bits lies below minus one");
    }
    // The array must agree with its own shape before any value is read.
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, "unpackbits");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    let ax = fix_axis(axis, a.ndim);
    if ax < 0 {
        return empty_arr(), fault(F_RANGE, "unpackbits", "the axis stands outside the shape");
    }
    let fb = bits_check_bytes(a, "unpackbits");
    if fb.kind != F_NONE { return empty_arr(), fb; }
    let width = a.shape[ax];
    let inner = product_after_of(a.shape, ax);
    let outer = product_before_of(a.shape, ax);
    var nbits = width * BITS_PER_BYTE;
    if count >= 0 { nbits = count; }
    if nbits > MAX_ELEMENTS / (inner * outer) {
        return empty_arr(), fault(F_LIMIT, "unpackbits", "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var o: i64 = 0;
    while o < outer {
        let base = o * width * inner;
        var p: i64 = 0;
        while p < nbits {
            // The byte and the bit of the column stay the same for
            // every value of the inner axis.
            let bi = p / BITS_PER_BYTE;
            let k = p % BITS_PER_BYTE;
            // The mask of the bit stays the same for every value of the
            // inner axis.
            let mask = 1 << bits_shift(k, ord);
            var j: i64 = 0;
            while j < inner {
                var value: f64 = 0.0;
                if bi < width {
                    let byte = a.data[base + bi * inner + j] as i64;
                    if (byte & mask) != 0 { value = 1.0; }
                }
                d.push(value);
                j += 1;
            }
            p += 1;
        }
        o += 1;
    }
    var s = same_shape(a);
    s[ax] = nbits;
    return Arr { ndim: a.ndim, shape: s, data: d }, no_fault();
}


