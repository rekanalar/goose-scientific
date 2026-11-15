// The archive of zip files and the members of NumPy inside it.
//
// The reader of this file finds the record of the end of an archive,
// and walks the directory in the middle of it. It takes the body of
// every member from the head of that member. A member that is
// compressed gives a fault, because the standard library holds no
// deflate.
//
// The members of an archive of SciPy hold a .npy file each. The head
// of a .npy file names its type and its shape, and the body holds the
// numbers. The reader of this file takes the types of NumPy. The
// values hold float64. The indices and the shape hold int32 or int64.
// The name of a format holds bytes. The head is read by hand and is
// never run as code.
//
// The writer of this file writes the same heads and the same bodies.
// The archive of the package and the archive of SciPy therefore hold
// the same bytes.
namespace scientific;

const NPZ_EOCD: i64 = 101010256;
const NPZ_CENTRAL: i64 = 33639248;
const NPZ_LOCAL: i64 = 67324752;
const NPZ_MAX_MEMBERS: i64 = 256;
const NPZ_MAX_NAME: i64 = 64;

// npz_get_u16 gives the whole number of two bytes, least significant
// first.
fn npz_get_u16(b: const u8[:], at: i64) -> i64 {
    return (b[at] as i64) | ((b[at + 1] as i64) << 8);
}

// npz_get_u32 gives the whole number of four bytes, least significant
// first.
fn npz_get_u32(b: const u8[:], at: i64) -> i64 {
    return npz_get_u16(b, at) | (npz_get_u16(b, at + 2) << 16);
}

// npz_eocd finds the record of the end of the archive. The record can
// stand up to 64 KiB before the end, because a zip file can hold a
// note behind it. The search runs over that room alone.
fn npz_eocd(b: const u8[:], tag: const u8[:]) -> i64, Fault {
    if b.len < 22 {
        return -1, fault(F_VALUE, tag, "the bytes are too short for an archive");
    }
    var at = b.len - 22;
    var low: i64 = 0;
    if at > 65557 { low = at - 65557; }
    while at >= low {
        if npz_get_u32(b, at) == NPZ_EOCD { return at, no_fault(); }
        at -= 1;
    }
    return -1, fault(F_VALUE, tag, "the bytes hold no record of the end of an archive");
}

// zip_name_ok holds the rule for the name of a member: letters,
// digits and underscores, then the suffix .npy. The rule keeps a path
// out of the archive.
fn zip_name_ok(n: const u8[:]) -> bool {
    if n.len < 5 || n.len > NPZ_MAX_NAME + 4 { return false; }
    let k = n.len - 4;
    if !(n[k] == '.' && n[k + 1] == 'n' && n[k + 2] == 'p' && n[k + 3] == 'y') {
        return false;
    }
    return npz_name_ok(n[..k]);
}

// npz_entries walks the directory in the middle of the archive. It
// gives the start and the length of every name, the kind of the
// storage, the length of every body and the place of every head. The
// names are checked before any body is read.
fn npz_entries(b: const u8[:], at: i64, tag: const u8[:]) -> i64[>..], i64[>..], i64[>..], i64[>..], i64[>..], Fault {
    var ns: i64[>..] = [];
    var nl: i64[>..] = [];
    var me: i64[>..] = [];
    var us: i64[>..] = [];
    var of: i64[>..] = [];
    let count = npz_get_u16(b, at + 10);
    if count > NPZ_MAX_MEMBERS {
        return ns, nl, me, us, of, fault(F_LIMIT, tag, "the archive holds more members than the limit");
    }
    var p = npz_get_u32(b, at + 16);
    let end = p + npz_get_u32(b, at + 12);
    if end > b.len {
        return ns, nl, me, us, of, fault(F_LIMIT, tag, "the directory stands outside the bytes");
    }
    var k: i64 = 0;
    while k < count {
        if p < 0 || p + 46 > b.len {
            return ns, nl, me, us, of, fault(F_LIMIT, tag, "the directory is cut");
        }
        if npz_get_u32(b, p) != NPZ_CENTRAL {
            return ns, nl, me, us, of, fault(F_VALUE, tag, "the directory holds no head of a member");
        }
        let n = npz_get_u16(b, p + 28);
        let el = npz_get_u16(b, p + 30);
        let cl = npz_get_u16(b, p + 32);
        let nm = npz_get_u16(b, p + 10);
        let sz = npz_get_u32(b, p + 24);
        let off = npz_get_u32(b, p + 42);
        // The room of the name is checked before the name is read.
        if p + 46 + n > b.len {
            return ns, nl, me, us, of, fault(F_LIMIT, tag, "the name of a member stands outside the bytes");
        }
        if !zip_name_ok(b[p + 46..p + 46 + n]) {
            return ns, nl, me, us, of, fault(F_VALUE, tag, "a name holds a byte outside letters, digits and underscore");
        }
        if sz < 0 || sz > MAX_ELEMENTS * 8 {
            return ns, nl, me, us, of, fault(F_LIMIT, tag, "a member holds more bytes than the limit");
        }
        if off < 0 || off + 30 > b.len {
            return ns, nl, me, us, of, fault(F_LIMIT, tag, "the head of a member stands outside the bytes");
        }
        ns.push(p + 46);
        nl.push(n);
        me.push(nm);
        us.push(sz);
        of.push(off);
        p += 46 + n + el + cl;
        k += 1;
    }
    return ns, nl, me, us, of, no_fault();
}

// npz_member gives the body of the member with the given name. The
// body stands behind the head of the member, whose own name and extra
// room come from the head. A member that is compressed gives a fault,
// because the standard library holds no deflate.
fn npz_member(b: const u8[:], ns: i64[:], nl: i64[:], me: i64[:], us: i64[:],
              of: i64[:], want: const u8[:], tag: const u8[:]) -> u8[>..], Fault {
    var out: u8[>..] = [];
    var k: i64 = 0;
    while k < ns.len {
        if nl[k] == want.len + 4 {
            var ok = true;
            var q: i64 = 0;
            while ok && q < want.len {
                if b[ns[k] + q] != want[q] { ok = false; }
                q += 1;
            }
            let tail = ns[k] + want.len;
            if ok && !(b[tail] == '.' && b[tail + 1] == 'n' && b[tail + 2] == 'p' && b[tail + 3] == 'y') {
                ok = false;
            }
            if ok {
                if me[k] != 0 {
                    return out, fault(F_VALUE, tag, "the member holds a compressed member");
                }
                let off = of[k];
                if npz_get_u32(b, off) != NPZ_LOCAL {
                    return out, fault(F_VALUE, tag, "the head of a member holds no mark");
                }
                let ds = off + 30 + npz_get_u16(b, off + 26) + npz_get_u16(b, off + 28);
                if ds + us[k] > b.len {
                    return out, fault(F_LIMIT, tag, "the body of a member stands outside the bytes");
                }
                var q2: i64 = 0;
                while q2 < us[k] { out.push(b[ds + q2]); q2 += 1; }
                return out, no_fault();
            }
        }
        k += 1;
    }
    return out, fault(F_VALUE, tag, "the archive holds no member with the name");
}

// npz_npy_meta reads the head of a member: the kind of the type, the
// shape and the place of the body. The kinds are 0 float64, 1 int32,
// 2 int64 and 3 bytes of the type that holds no order. The head is
// read by hand and is never run as code.
fn npz_npy_meta(b: const u8[:], tag: const u8[:]) -> i64, i64[>..], i64, Fault {
    var shape: i64[>..] = [];
    if b.len < 10 || !npy_has(b, 0, NPY_MAGIC) {
        return 0, shape, 0, fault(F_VALUE, tag, "a member holds no magic of a .npy file");
    }
    var hstart: i64 = 10;
    var hlen: i64 = npz_get_u16(b, 8);
    if b[6] == 2 || b[6] == 3 {
        if b.len < 12 {
            return 0, shape, 0, fault(F_VALUE, tag, "a head is cut");
        }
        hlen = hlen | (npz_get_u16(b, 10) << 16);
        hstart = 12;
    } else if b[6] != 1 {
        return 0, shape, 0, fault(F_VALUE, tag, "a member holds another version than 1, 2 or 3");
    }
    if hlen > NPY_MAX_HEADER || hstart + hlen > b.len {
        return 0, shape, 0, fault(F_LIMIT, tag, "a head is above its limit or past the bytes");
    }
    let h = b[hstart..hstart + hlen];
    let at = npy_skip(h, npy_find(h, "'descr':"));
    if at < 0 || at >= h.len || h[at] != '\'' {
        return 0, shape, 0, fault(F_VALUE, tag, "a head holds no type");
    }
    var e = at + 1;
    while e < h.len && h[e] != '\'' { e += 1; }
    if e >= h.len {
        return 0, shape, 0, fault(F_VALUE, tag, "a head holds no end of the type");
    }
    let d = h[at + 1..e];
    var kind: i64 = -1;
    if d.len == 3 && (d == "<f8" || d == "=f8") { kind = 0; }
    if d.len == 3 && d == "<i4" { kind = 1; }
    if d.len == 3 && d == "<i8" { kind = 2; }
    if d.len >= 2 && d[0] == '|' && d[1] == 'S' { kind = 3; }
    if kind < 0 {
        return 0, shape, 0, fault(F_VALUE, tag, "a member holds a type that the package does not read");
    }
    let fo = npy_skip(h, npy_find(h, "'fortran_order':"));
    if !npy_has(h, fo, "False") {
        return 0, shape, 0, fault(F_VALUE, tag, "a member holds the Fortran order");
    }
    let st = npy_skip(h, npy_find(h, "'shape':"));
    if st < 0 || st >= h.len || h[st] != '(' {
        return 0, shape, 0, fault(F_VALUE, tag, "a head holds no shape");
    }
    var p = st + 1;
    while true {
        p = npy_skip(h, p);
        if p >= h.len {
            return 0, shape, 0, fault(F_VALUE, tag, "a shape is cut");
        }
        if h[p] == ')' { break; }
        var n: i64 = 0;
        var digits: i64 = 0;
        while p < h.len && h[p] >= '0' && h[p] <= '9' {
            n = n * 10 + ((h[p] - '0') as i64);
            if n > MAX_ELEMENTS {
                return 0, shape, 0, fault(F_LIMIT, tag, "a length is above the limit");
            }
            digits += 1;
            p += 1;
        }
        if digits == 0 {
            return 0, shape, 0, fault(F_VALUE, tag, "a shape holds a bad length");
        }
        if shape.len >= MAX_DIMS {
            return 0, shape, 0, fault(F_LIMIT, tag, "a shape holds more dimensions than the limit");
        }
        shape.push(n);
        p = npy_skip(h, p);
        if p < h.len && h[p] == ',' { p += 1; }
    }
    return kind, shape, hstart + hlen, no_fault();
}

// npz_vals_f64 gives the values of a member of float64.
fn npz_vals_f64(b: const u8[:], body: i64, count: i64, tag: const u8[:]) -> f64[>..], Fault {
    var out: f64[>..] = [];
    if count < 0 || body + count * 8 > b.len {
        return out, fault(F_LIMIT, tag, "a member holds fewer numbers than its shape");
    }
    var i: i64 = 0;
    while i < count {
        out.push(f64_of_bits(bits_of(b, body + i * 8)));
        i += 1;
    }
    return out, no_fault();
}

// npz_vals_i64 gives the values of a member of int32 or int64 as whole
// numbers. Both types are read, because NumPy holds the indices of a
// small matrix in int32 and the shape in int64.
fn npz_vals_i64(b: const u8[:], body: i64, count: i64, kind: i64, tag: const u8[:]) -> f64[>..], Fault {
    var out: f64[>..] = [];
    let w = if kind == 1 { 4 } else { 8 };
    if count < 0 || body + count * w > b.len {
        return out, fault(F_LIMIT, tag, "a member holds fewer numbers than its shape");
    }
    var i: i64 = 0;
    while i < count {
        var v: i64 = 0;
        if kind == 1 {
            v = npz_get_u32(b, body + i * 4);
            if v >= 2147483648 { v -= 4294967296; }
        } else {
            v = npz_get_u32(b, body + i * 8) | (npz_get_u32(b, body + i * 8 + 4) << 32);
        }
        out.push(v as f64);
        i += 1;
    }
    return out, no_fault();
}

// npz_put_i64 writes a whole number as eight bytes, least significant
// first.
fn npz_put_i64(out: u8[>..]&, v: i64) {
    npy_put_u32(out, v & 4294967295);
    npy_put_u32(out, v >> 32);
}

// npz_head gives the head of a member: the magic, the version, the
// length and the three keys in the order of NumPy, filled with spaces
// to the next multiple of 64 bytes.
fn npz_head(descr: const u8[:], shape: i64[:]) -> u8[>..] {
    var h: u8[>..] = [];
    h.append("{'descr': '");
    h.append(descr);
    h.append("', 'fortran_order': False, 'shape': (");
    var d: i64 = 0;
    while d < shape.len {
        if d > 0 { h.append(", "); }
        npy_put_int(h, shape[d]);
        d += 1;
    }
    if shape.len == 1 { h.push(','); }
    h.append("), }");
    let hlen = h.len + 1;
    let padlen = 64 - ((10 + hlen) % 64);
    var out: u8[>..] = [];
    out.append(NPY_MAGIC);
    out.push(1);
    out.push(0);
    npy_put_u16(out, hlen + padlen);
    out.append(h);
    var p: i64 = 0;
    while p < padlen { out.push(' '); p += 1; }
    out.push('\n');
    return out;
}

// npz_body_i gives the body bytes of a list of whole numbers, in the
// type that the rule of the index type of SciPy names: int32 while
// every index fits, int64 when one stands above it.
fn npz_body_i(vals: f64[:], wide: bool) -> u8[>..] {
    var out: u8[>..] = [];
    var i: i64 = 0;
    while i < vals.len {
        if wide { npz_put_i64(out, vals[i] as i64); } else { npy_put_u32(out, vals[i] as i64); }
        i += 1;
    }
    return out;
}

// npz_body_f64 gives the body bytes of a list of values of float64.
fn npz_body_f64(vals: f64[:]) -> u8[>..] {
    var out: u8[>..] = [];
    var i: i64 = 0;
    while i < vals.len { npy_put_f64(out, vals[i]); i += 1; }
    return out;
}
