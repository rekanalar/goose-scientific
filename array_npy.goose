// The binary files of NumPy: save, load, savez and savez_compressed.
//
// NumPy stores an array in the .npy format: a magic, a header with the
// type and the shape, and the raw numbers. The package holds one type
// of number, so it writes and reads the type float64, least significant
// byte first, and it refuses any other type. The header is read by
// hand: the reader finds the three keys and reads the numbers that it
// needs, and it never evaluates the header as code.
//
// A .npz file is a zip file of .npy members. The standard library of
// both supported Goose versions holds no deflate, so savez_compressed
// writes stored members, the same bytes as savez. NumPy reads them
// both. The bits of a number are built in the package, because the
// standard library of one supported version has no call for them.
namespace scientific;

import os;

const NPY_MAGIC: const u8[:] = "\x93NUMPY";
const NPY_MAX_HEADER: i64 = 10000;

fn npy_put_u16(out: u8[>..]&, v: i64) {
    out.push((v & 255) as u8);
    out.push(((v >> 8) & 255) as u8);
}

fn npy_put_u32(out: u8[>..]&, v: i64) {
    npy_put_u16(out, v & 65535);
    npy_put_u16(out, (v >> 16) & 65535);
}

// npy_bits gives the pattern of bits of a number: one part holds the
// sign, one part holds the power of two, and one part holds the digits.
// The digits come from the number in steps of two, because the standard
// library of one supported version holds no call for the bits.
fn npy_bits(x: f64) -> u64 {
    var sign: u64 = 0;
    if x < 0.0 || (x == 0.0 && 1.0 / x < 0.0) { sign = 1; }
    if x != x { return (sign << 63) | 9221120237041090560; }
    var ax = if x < 0.0 { -x } else { x };
    if ax - ax != 0.0 { return (sign << 63) | 9218868437227405312; }
    if ax == 0.0 { return sign << 63; }
    // A step of two never rounds, so y holds the digits of the number.
    var y = ax;
    var k: i64 = 0;
    while y < 1.0 { y = y * 2.0; k -= 1; }
    while y >= 2.0 { y = y / 2.0; k += 1; }
    if k < -1022 {
        // A number below the smallest normal holds its digits in the
        // fraction alone, in steps of 2^-1074.
        var s = k + 1074;
        var m = y;
        while s > 0 { m = m * 2.0; s -= 1; }
        return (sign << 63) | (m as! u64);
    }
    let frac = ((y - 1.0) * 4503599627370496.0) as! u64;
    return (sign << 63) | (((k + 1023) as u64) << 52) | frac;
}

// npy_put_f64 writes a number as 8 bytes, least significant first.
fn npy_put_f64(out: u8[>..]&, x: f64) {
    let b = npy_bits(x);
    var k: i64 = 0;
    while k < 8 {
        out.push(((b >> (8 * k)) & 255) as u8);
        k += 1;
    }
}

fn npy_put_int(out: u8[>..]&, v: i64) {
    var le: u8[>..] = [];
    var n = v;
    if n == 0 { le.push('0'); }
    while n > 0 { le.push((('0') + n % 10) as u8); n = n / 10; }
    var k = le.len - 1;
    while k >= 0 { out.push(le[k]); k -= 1; }
}

// npy_bytes writes an array as a .npy file of version 1.0. The header
// holds the three keys in the order of NumPy and fills to a multiple of
// 64 bytes, and 21 spaces of spare room leave the room that NumPy leaves.
fn npy_bytes(a: Arr&, tag: const u8[:]) -> u8[>..], Fault {
    var out: u8[>..] = [];
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fs.kind != F_NONE { return out, fs; }
    var h: u8[>..] = [];
    h.append("{'descr': '<f8', 'fortran_order': False, 'shape': (");
    var d: i64 = 0;
    while d < a.ndim {
        if d > 0 { h.append(", "); }
        npy_put_int(h, a.shape[d]);
        d += 1;
    }
    if a.ndim == 1 { h.push(','); }
    h.append("), }");
    var first: i64 = 1;
    var w = a.shape[0];
    while w >= 10 { w = w / 10; first += 1; }
    var sp: i64 = 21 - first;
    while sp > 0 { h.push(' '); sp -= 1; }
    let hlen = h.len + 1;
    let padlen = 64 - ((10 + hlen) % 64);
    out.append(NPY_MAGIC);
    out.push(1);
    out.push(0);
    npy_put_u16(out, hlen + padlen);
    out.append(h);
    var p: i64 = 0;
    while p < padlen { out.push(' '); p += 1; }
    out.push('\n');
    var i: i64 = 0;
    while i < a.data.len { npy_put_f64(out, a.data[i]); i += 1; }
    return out, no_fault();
}

// npy_path refuses a path with a NUL byte.
fn npy_path(path: const u8[:], tag: const u8[:]) -> Fault {
    var k: i64 = 0;
    while k < path.len {
        if path[k] == 0 { return fault(F_VALUE, tag, "the path holds a NUL byte"); }
        k += 1;
    }
    return no_fault();
}

// save writes an array to a .npy file. The file holds the type float64.
fn save(path: const u8[:], a: Arr&) -> bool, Fault {
    let fp = npy_path(path, "save");
    if fp.kind != F_NONE { return false, fp; }
    let bytes, f = npy_bytes(a, "save");
    if f.kind != F_NONE { return false, f; }
    if !write_file(path, bytes) { return false, fault(F_VALUE, "save", "the file does not write"); }
    return true, no_fault();
}

// npy_find gives the place after a key in a header, or -1.
fn npy_find(h: const u8[:], key: const u8[:]) -> i64 {
    var i: i64 = 0;
    while i + key.len <= h.len {
        var k: i64 = 0;
        while k < key.len && h[i + k] == key[k] { k += 1; }
        if k == key.len { return i + key.len; }
        i += 1;
    }
    return -1;
}

// npy_skip steps over the spaces of a header. A place of -1 stays.
fn npy_skip(h: const u8[:], at: i64) -> i64 {
    if at < 0 { return at; }
    var i = at;
    while i < h.len && (h[i] == ' ' || h[i] == '\t') { i += 1; }
    return i;
}

// npy_has tells whether a header holds a word at a place.
fn npy_has(h: const u8[:], at: i64, word: const u8[:]) -> bool {
    if at < 0 || at + word.len > h.len { return false; }
    var k: i64 = 0;
    while k < word.len { if h[at + k] != word[k] { return false; } k += 1; }
    return true;
}

// npy_read decodes the bytes of a .npy file of the type float64. A file
// of any other type, of the Fortran order, or with a header that claims
// more numbers than the bytes hold is a fault.
fn npy_read(b: const u8[:], tag: const u8[:]) -> Arr, Fault {
    if b.len < 10 || !npy_has(b, 0, NPY_MAGIC) {
        return empty_arr(), fault(F_VALUE, tag, "the bytes hold no magic of a .npy file");
    }
    var hstart: i64 = 10;
    var hlen: i64 = b[8] | ((b[9] as i64) << 8);
    if b[6] == 2 || b[6] == 3 {
        if b.len < 12 { return empty_arr(), fault(F_VALUE, tag, "the header is cut"); }
        hlen = hlen | ((b[10] as i64) << 16) | ((b[11] as i64) << 24);
        hstart = 12;
    } else if b[6] != 1 {
        return empty_arr(), fault(F_VALUE, tag, "the version is not 1, 2 or 3");
    }
    if hlen > NPY_MAX_HEADER || hstart + hlen > b.len {
        return empty_arr(), fault(F_LIMIT, tag, "the header is above its limit or past the bytes");
    }
    let h = b[hstart..hstart + hlen];
    var at = npy_find(h, "'descr':");
    at = npy_skip(h, at);
    if at < 0 || !(npy_has(h, at, "'<f8'") || npy_has(h, at, "'=f8'")) {
        return empty_arr(), fault(F_VALUE, tag, "the type is not float64");
    }
    at = npy_skip(h, npy_find(h, "'fortran_order':"));
    if !npy_has(h, at, "False") {
        return empty_arr(), fault(F_VALUE, tag, "the file holds the Fortran order");
    }
    at = npy_find(h, "'shape':");
    if at < 0 { return empty_arr(), fault(F_VALUE, tag, "the header holds no shape"); }
    at = npy_skip(h, at);
    if at >= h.len || h[at] != '(' { return empty_arr(), fault(F_VALUE, tag, "the shape holds no tuple"); }
    at += 1;
    var shape: i64[..8] = [];
    while true {
        at = npy_skip(h, at);
        if at >= h.len { return empty_arr(), fault(F_VALUE, tag, "the shape is cut"); }
        if h[at] == ')' { break; }
        var n: i64 = 0;
        var digits: i64 = 0;
        while at < h.len && h[at] >= '0' && h[at] <= '9' {
            n = n * 10 + (h[at] - '0');
            if n > MAX_ELEMENTS { return empty_arr(), fault(F_LIMIT, tag, "a length is above the limit"); }
            digits += 1;
            at += 1;
        }
        if digits == 0 { return empty_arr(), fault(F_VALUE, tag, "the shape holds a bad length"); }
        if shape.len >= MAX_DIMS { return empty_arr(), fault(F_LIMIT, tag, "the shape holds more dimensions than the limit"); }
        shape.push(n);
        at = npy_skip(h, at);
        if at < h.len && h[at] == ',' { at += 1; }
    }
    let fc = check_shape(shape);
    if fc.kind != F_NONE { return empty_arr(), fault(fc.kind, tag, fc.detail); }
    let n = numel(shape);
    let body = hstart + hlen;
    // The header must not claim more numbers than the bytes hold.
    if n > (b.len - body) / 8 {
        return empty_arr(), fault(F_LIMIT, tag, "the header claims more numbers than the file holds");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        d.push(f64_of_bits(bits_of(b, body + i * 8)));
        i += 1;
    }
    return Arr { ndim: shape.len, shape: shape, data: d }, no_fault();
}

// load reads an array from a .npy file of the type float64.
fn load(path: const u8[:]) -> Arr, Fault {
    let fp = npy_path(path, "load");
    if fp.kind != F_NONE { return empty_arr(), fp; }
    var bytes: u8[>..] = [];
    if !read_file(path, bytes) { return empty_arr(), fault(F_VALUE, "load", "the file does not read"); }
    if bytes.len > TEXT_MAX_BYTES { return empty_arr(), fault(F_LIMIT, "load", "the file holds more bytes than the limit"); }
    return npy_read(bytes[..], "load");
}

// npz_crc gives the CRC-32 of the bytes of a zip member.
fn npz_crc(b: const u8[:]) -> i64 {
    var crc: u32 = 4294967295;
    var i: i64 = 0;
    while i < b.len {
        crc = crc ^ b[i];
        var k: i64 = 0;
        while k < 8 {
            if (crc & 1) == 1 { crc = (crc >> 1) ^ 3988292384; } else { crc = crc >> 1; }
            k += 1;
        }
        i += 1;
    }
    return (crc ^ 4294967295);
}

// npz_name_ok holds the rule for a member name: one to 64 letters,
// digits and underscores. It keeps a path out of the archive.
fn npz_name_ok(name: const u8[:]) -> bool {
    if name.len < 1 || name.len > 64 { return false; }
    var i: i64 = 0;
    while i < name.len {
        let c = name[i];
        let ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        if !ok { return false; }
        i += 1;
    }
    return true;
}

// npz_write writes a zip file of stored members. The members are the
// .npy bytes of the bodies, named name.npy. A zip holds at most one
// member whose name takes 64 bytes, and one archive of 2 GB.
fn npz_write(path: const u8[:], names: const u8[:][>..]&, bodies: const u8[:][>..]&, tag: const u8[:]) -> bool, Fault {
    let fp = npy_path(path, tag);
    if fp.kind != F_NONE { return false, fp; }
    var out: u8[>..] = [];
    var central: u8[>..] = [];
    var count: i64 = 0;
    var m: i64 = 0;
    while m < bodies.len {
        if !npz_name_ok(names[m]) {
            return false, fault(F_VALUE, tag, "a name holds a byte outside letters, digits and underscore");
        }
        if out.len + bodies[m].len > 2147483647 {
            return false, fault(F_LIMIT, tag, "the archive holds more bytes than a zip file holds");
        }
        let crc = npz_crc(bodies[m]);
        let nlen = names[m].len + 4;
        let offset = out.len;
        // The local header and the central entry hold the same fields:
        // version 20, no flags, stored, the time and date of 1980.
        npy_put_u32(out, 67324752);
        npy_put_u16(out, 20); npy_put_u16(out, 0); npy_put_u16(out, 0);
        npy_put_u16(out, 0); npy_put_u16(out, 33);
        npy_put_u32(out, crc); npy_put_u32(out, bodies[m].len); npy_put_u32(out, bodies[m].len);
        npy_put_u16(out, nlen); npy_put_u16(out, 0);
        out.append(names[m]); out.append(".npy"); out.append(bodies[m]);
        npy_put_u32(central, 33639248);
        npy_put_u16(central, 20); npy_put_u16(central, 20); npy_put_u16(central, 0);
        npy_put_u16(central, 0); npy_put_u16(central, 0); npy_put_u16(central, 33);
        npy_put_u32(central, crc); npy_put_u32(central, bodies[m].len); npy_put_u32(central, bodies[m].len);
        npy_put_u16(central, nlen); npy_put_u16(central, 0); npy_put_u16(central, 0);
        npy_put_u16(central, 0); npy_put_u16(central, 0); npy_put_u32(central, 0);
        npy_put_u32(central, offset);
        central.append(names[m]); central.append(".npy");
        count += 1;
        m += 1;
    }
    let cdir = out.len;
    out.append(central);
    npy_put_u32(out, 101010256);
    npy_put_u16(out, 0); npy_put_u16(out, 0);
    npy_put_u16(out, count); npy_put_u16(out, count);
    npy_put_u32(out, central.len); npy_put_u32(out, cdir); npy_put_u16(out, 0);
    if !write_file(path, out) { return false, fault(F_VALUE, tag, "the file does not write"); }
    return true, no_fault();
}

// savez writes one named array to a .npz file. NumPy takes the name
// arr_0 for an array with no name, so a caller passes that name.
fn savez(path: const u8[:], name: const u8[:], a: Arr&) -> bool, Fault {
    let body, f = npy_bytes(a, "savez");
    if f.kind != F_NONE { return false, f; }
    var names: const u8[:][>..] = [];
    names.push(name);
    var bodies: const u8[:][>..] = [];
    bodies.push(body[..]);
    return npz_write(path, names, bodies, "savez");
}

// savez for two named arrays.
fn savez(path: const u8[:], name1: const u8[:], a1: Arr&, name2: const u8[:], a2: Arr&) -> bool, Fault {
    let body1, f1 = npy_bytes(a1, "savez");
    if f1.kind != F_NONE { return false, f1; }
    let body2, f2 = npy_bytes(a2, "savez");
    if f2.kind != F_NONE { return false, f2; }
    var names: const u8[:][>..] = [];
    names.push(name1);
    names.push(name2);
    var bodies: const u8[:][>..] = [];
    bodies.push(body1[..]);
    bodies.push(body2[..]);
    return npz_write(path, names, bodies, "savez");
}

// savez_compressed writes the same file as savez. The members stay
// stored, because the standard library holds no deflate.
fn savez_compressed(path: const u8[:], name: const u8[:], a: Arr&) -> bool, Fault {
    let body, f = npy_bytes(a, "savez_compressed");
    if f.kind != F_NONE { return false, f; }
    var names: const u8[:][>..] = [];
    names.push(name);
    var bodies: const u8[:][>..] = [];
    bodies.push(body[..]);
    return npz_write(path, names, bodies, "savez_compressed");
}

fn savez_compressed(path: const u8[:], name1: const u8[:], a1: Arr&, name2: const u8[:], a2: Arr&) -> bool, Fault {
    let body1, f1 = npy_bytes(a1, "savez_compressed");
    if f1.kind != F_NONE { return false, f1; }
    let body2, f2 = npy_bytes(a2, "savez_compressed");
    if f2.kind != F_NONE { return false, f2; }
    var names: const u8[:][>..] = [];
    names.push(name1);
    names.push(name2);
    var bodies: const u8[:][>..] = [];
    bodies.push(body1[..]);
    bodies.push(body2[..]);
    return npz_write(path, names, bodies, "savez_compressed");
}
