// The tests of the group array_text. Run them with SCIENTIFIC_SUITES=array_text.
//
// The text calls match the pins of NumPy 2.5.4: format_float_positional,
// format_float_scientific, base_repr, binary_repr, typename and
// mintypecode. Each case of a pin list holds a start and a length in a
// byte list. A length of -1 means that NumPy raised an error.
namespace array_text_tests;

import scientific from "../..";
import array_text_pins;

fn same(got: const u8[:], bytes: f64[:], spans: f64[:], i: i64) -> bool {
    let at = spans[2 * i] as i64;
    let n = spans[2 * i + 1] as i64;
    if got.len != n { return false; }
    var k: i64 = 0;
    while k < n {
        if got[k] as i64 != bytes[at + k] as i64 { return false; }
        k += 1;
    }
    return true;
}

fn text_of(bytes: f64[:], spans: f64[:], i: i64) -> u8[>..] {
    var out: u8[>..] = [];
    let at = spans[2 * i] as i64;
    let n = spans[2 * i + 1] as i64;
    var k: i64 = 0;
    while k < n { out.push(bytes[at + k] as u8); k += 1; }
    return out;
}

fn is_text(got: const u8[:], want: const u8[:]) -> bool {
    return got == want;
}

fn value_of(code: i64, x: f64) -> f64 {
    if code == 1 { return 0.0 / 0.0; }
    if code == 2 { return 1.0 / 0.0; }
    if code == 3 { return -1.0 / 0.0; }
    return x;
}

fn options_of(rows: f64[:], r: i64) -> scientific::FloatFormat {
    var o = scientific::float_format();
    o.precision = rows[r + 2] as i64;
    o.unique = rows[r + 3] == 1.0;
    o.fractional = rows[r + 4] == 1.0;
    o.trim = rows[r + 5] as u8;
    o.sign = rows[r + 6] == 1.0;
    o.pad_left = rows[r + 7] as i64;
    o.pad_right = rows[r + 8] as i64;
    o.min_digits = rows[r + 9] as i64;
    o.exp_digits = rows[r + 10] as i64;
    return o;
}

fn check_floats(rows: f64[:], bytes: f64[:], spans: f64[:], sci: bool) {
    let count = rows.len / 11;
    var i: i64 = 0;
    while i < count {
        let r = i * 11;
        let x = value_of(rows[r + 1] as i64, rows[r]);
        let o = options_of(rows, r);
        var out: u8[>..] = [];
        var f = scientific::no_fault();
        if sci { out, f = scientific::format_float_scientific(x, o); }
        else { out, f = scientific::format_float_positional(x, o); }
        let n = spans[2 * i + 1] as i64;
        if n < 0 {
            assert(f.kind == scientific::F_VALUE);
        } else {
            if f.kind != scientific::F_NONE || !same(out, bytes, spans, i) {
                let want = text_of(bytes, spans, i);
                print("case ", i, " x=", x, " prec=", o.precision, " uniq=", o.unique,
                      " frac=", o.fractional, " trim=", o.trim as i64, " sign=", o.sign,
                      " pl=", o.pad_left, " pr=", o.pad_right, " md=", o.min_digits,
                      " ed=", o.exp_digits, " fault=", f.kind, "\n");
                print("  got: '", out, "'\n");
                print("  want:'", want, "'\n");
            }
            assert(f.kind == scientific::F_NONE);
            assert(same(out, bytes, spans, i));
        }
        i += 1;
    }
}

fn test_format_float_positional() {
    check_floats(array_text_pins::POS_ROWS, array_text_pins::POS_BYTES, array_text_pins::POS_SPANS, false);
    var o = scientific::float_format();
    let a, fa = scientific::format_float_positional(0.1, o);
    assert(fa.kind == scientific::F_NONE && is_text(a, "0.1"));
    let b, fb = scientific::format_float_positional(1.0 / 3.0, o);
    assert(fb.kind == scientific::F_NONE && is_text(b, "0.3333333333333333"));
    let c, fc = scientific::format_float_positional(1.0e16, o);
    assert(fc.kind == scientific::F_NONE && is_text(c, "10000000000000000."));
    let d, fd = scientific::format_float_positional(1.0e-7, o);
    assert(fd.kind == scientific::F_NONE && is_text(d, "0.0000001"));
    o.precision = 3;
    let e, fe = scientific::format_float_positional(1.0e-7, o);
    assert(fe.kind == scientific::F_NONE && is_text(e, "0.000"));
    o = scientific::float_format();
    o.trim = '0';
    let f, ff = scientific::format_float_positional(3.0, o);
    assert(ff.kind == scientific::F_NONE && is_text(f, "3.0"));
}

fn test_format_float_scientific() {
    check_floats(array_text_pins::SCI_ROWS, array_text_pins::SCI_BYTES, array_text_pins::SCI_SPANS, true);
    var o = scientific::float_format();
    let a, fa = scientific::format_float_scientific(0.1, o);
    assert(fa.kind == scientific::F_NONE && is_text(a, "1.e-01"));
    let b, fb = scientific::format_float_scientific(123456.789, o);
    assert(fb.kind == scientific::F_NONE && is_text(b, "1.23456789e+05"));
    let c, fc = scientific::format_float_scientific(3.0, o);
    assert(fc.kind == scientific::F_NONE && is_text(c, "3.e+00"));
    let d, fd = scientific::format_float_scientific(0.0, o);
    assert(fd.kind == scientific::F_NONE && is_text(d, "0.e+00"));
    o.exp_digits = 3;
    let e, fe = scientific::format_float_scientific(1.5, o);
    assert(fe.kind == scientific::F_NONE && is_text(e, "1.5e+000"));
    o = scientific::float_format();
    o.precision = 2;
    let f, ff = scientific::format_float_scientific(123456.789, o);
    assert(ff.kind == scientific::F_NONE && is_text(f, "1.23e+05"));
}

fn test_format_float_limits() {
    var o = scientific::float_format();
    o.precision = 5000;
    let t, f = scientific::format_float_positional(1.5, o);
    assert(f.kind == scientific::F_LIMIT);
    o.precision = 4096;
    o.unique = false;
    let u, g = scientific::format_float_positional(1.5, o);
    assert(g.kind == scientific::F_NONE && u.len == 4096 + 2);
    o = scientific::float_format();
    o.trim = 'x';
    let w, h = scientific::format_float_scientific(1.5, o);
    assert(h.kind == scientific::F_VALUE);
    o = scientific::float_format();
    o.precision = -2;
    let i, j = scientific::format_float_positional(1.5, o);
    assert(j.kind == scientific::F_VALUE);
    o = scientific::float_format();
    o.unique = false;
    let k, l = scientific::format_float_positional(1.5, o);
    assert(l.kind == scientific::F_VALUE);
    o = scientific::float_format();
    o.fractional = false;
    o.precision = 0;
    let m, n = scientific::format_float_positional(1.5, o);
    assert(n.kind == scientific::F_VALUE);
    o = scientific::float_format();
    o.min_digits = 3;
    o.precision = 2;
    let p, q = scientific::format_float_positional(1.5, o);
    assert(q.kind == scientific::F_VALUE);
}

fn test_base_repr() {
    let rows = array_text_pins::BASE_ROWS;
    var i: i64 = 0;
    while i < rows.len / 3 {
        let out, f = scientific::base_repr(rows[3 * i], rows[3 * i + 1], rows[3 * i + 2]);
        let n = array_text_pins::BASE_SPANS[2 * i + 1] as i64;
        if n < 0 {
            assert(f.kind == scientific::F_VALUE);
        } else {
            if f.kind != scientific::F_NONE || !same(out, array_text_pins::BASE_BYTES, array_text_pins::BASE_SPANS, i) {
                print("base case ", i, " n=", rows[3 * i], " base=", rows[3 * i + 1], " pad=", rows[3 * i + 2], "\n");
                print("  got: '", out, "'\n");
                print("  want:'", text_of(array_text_pins::BASE_BYTES, array_text_pins::BASE_SPANS, i), "'\n");
            }
            assert(f.kind == scientific::F_NONE);
            assert(same(out, array_text_pins::BASE_BYTES, array_text_pins::BASE_SPANS, i));
        }
        i += 1;
    }
    let a, fa = scientific::base_repr(255, 16, 0);
    assert(fa.kind == scientific::F_NONE && is_text(a, "FF"));
    let b, fb = scientific::base_repr(35, 36, 0);
    assert(fb.kind == scientific::F_NONE && is_text(b, "Z"));
    let c, fc = scientific::base_repr(-5, 2, 3);
    assert(fc.kind == scientific::F_NONE && is_text(c, "-000101"));
    let d, fd = scientific::base_repr(7, 1, 0);
    assert(fd.kind == scientific::F_VALUE);
    let e, fe = scientific::base_repr(7, 37, 0);
    assert(fe.kind == scientific::F_VALUE);
    let g, fg = scientific::base_repr(7, 2, 5000);
    assert(fg.kind == scientific::F_LIMIT);
}

fn test_binary_repr() {
    let rows = array_text_pins::BIN_ROWS;
    var i: i64 = 0;
    while i < rows.len / 2 {
        let out, f = scientific::binary_repr(rows[2 * i], rows[2 * i + 1]);
        let n = array_text_pins::BIN_SPANS[2 * i + 1] as i64;
        if n < 0 {
            assert(f.kind == scientific::F_VALUE);
        } else {
            if f.kind != scientific::F_NONE || !same(out, array_text_pins::BIN_BYTES, array_text_pins::BIN_SPANS, i) {
                print("binary case ", i, " n=", rows[2 * i], " width=", rows[2 * i + 1], "\n");
                print("  got: '", out, "'\n");
                print("  want:'", text_of(array_text_pins::BIN_BYTES, array_text_pins::BIN_SPANS, i), "'\n");
            }
            assert(f.kind == scientific::F_NONE);
            assert(same(out, array_text_pins::BIN_BYTES, array_text_pins::BIN_SPANS, i));
        }
        i += 1;
    }
    let a, fa = scientific::binary_repr(5);
    assert(fa.kind == scientific::F_NONE && is_text(a, "101"));
    let b, fb = scientific::binary_repr(-5, 8);
    assert(fb.kind == scientific::F_NONE && is_text(b, "11111011"));
    let c, fc = scientific::binary_repr(0, 4);
    assert(fc.kind == scientific::F_NONE && is_text(c, "0000"));
    let d, fd = scientific::binary_repr(-6, 4);
    assert(fd.kind == scientific::F_NONE && is_text(d, "1010"));
    let e, fe = scientific::binary_repr(3, 1);
    assert(fe.kind == scientific::F_VALUE);
    let g, fg = scientific::binary_repr(3, 2);
    assert(fg.kind == scientific::F_NONE && is_text(g, "11"));
    let h, fh = scientific::binary_repr(7, 5000);
    assert(fh.kind == scientific::F_LIMIT);
}

fn test_typename() {
    let spans = array_text_pins::TYPENAME_IN_SPANS;
    var i: i64 = 0;
    while i < spans.len / 2 {
        let code = text_of(array_text_pins::TYPENAME_IN_BYTES, spans, i);
        let out, f = scientific::typename(code);
        let n = array_text_pins::TYPENAME_OUT_SPANS[2 * i + 1] as i64;
        if n < 0 {
            assert(f.kind == scientific::F_VALUE);
        } else {
            if f.kind != scientific::F_NONE || !same(out, array_text_pins::TYPENAME_OUT_BYTES, array_text_pins::TYPENAME_OUT_SPANS, i) {
                print("typename case ", i, " code=", code, "\n");
                print("  got: '", out, "'\n");
                print("  want:'", text_of(array_text_pins::TYPENAME_OUT_BYTES, array_text_pins::TYPENAME_OUT_SPANS, i), "'\n");
            }
            assert(f.kind == scientific::F_NONE);
            assert(same(out, array_text_pins::TYPENAME_OUT_BYTES, array_text_pins::TYPENAME_OUT_SPANS, i));
        }
        i += 1;
    }
    let a, fa = scientific::typename("d");
    assert(fa.kind == scientific::F_NONE && is_text(a, "double precision"));
    let b, fb = scientific::typename("G");
    assert(fb.kind == scientific::F_NONE && is_text(b, "complex long double precision"));
    let c, fc = scientific::typename("x");
    assert(fc.kind == scientific::F_VALUE);
    // The names of the 14 types of the package give themselves back.
    let names = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
                 "uint32", "uint64", "float16", "float32", "float64", "complex64", "complex128"];
    var k: i64 = 0;
    while k < names.len {
        let out, f = scientific::typename(names[k]);
        assert(f.kind == scientific::F_NONE && is_text(out, names[k]));
        k += 1;
    }
}

fn test_mintypecode() {
    let spans = array_text_pins::MT_CHARS_SPANS;
    var i: i64 = 0;
    while i < spans.len / 2 {
        let chars = text_of(array_text_pins::MT_CHARS_BYTES, spans, i);
        let set = text_of(array_text_pins::MT_SET_BYTES, array_text_pins::MT_SET_SPANS, i);
        let def = text_of(array_text_pins::MT_DEFAULT_BYTES, array_text_pins::MT_DEFAULT_SPANS, i);
        let c, f = scientific::mintypecode(chars, set, def[0]);
        let n = array_text_pins::MT_OUT_SPANS[2 * i + 1] as i64;
        if n < 0 {
            assert(f.kind == scientific::F_VALUE);
        } else {
            let want = array_text_pins::MT_OUT_BYTES[array_text_pins::MT_OUT_SPANS[2 * i] as i64] as i64;
            assert(f.kind == scientific::F_NONE && c as i64 == want);
        }
        i += 1;
    }
    let a, fa = scientific::mintypecode("df");
    assert(fa.kind == scientific::F_NONE && a == 'd');
    let b, fb = scientific::mintypecode("ff");
    assert(fb.kind == scientific::F_NONE && b == 'f');
    let c, fc = scientific::mintypecode("");
    assert(fc.kind == scientific::F_NONE && c == 'd');
    let d, fd = scientific::mintypecode("G");
    assert(fd.kind == scientific::F_VALUE);
    let e, fe = scientific::mintypecode("x", "x", 'd');
    assert(fe.kind == scientific::F_VALUE);
}
