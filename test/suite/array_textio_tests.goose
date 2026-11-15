// The tests of the group array_textio. Run them with
// SCIENTIFIC_SUITES=array_textio.
//
// The pins hold the text and the bytes that NumPy 2.5.4 gives for the
// same calls. Each test writes its files beside the test program and
// removes them after the check. A value of a pin holds a code: 0
// finite, 1 the missing value, 2 the infinity, 3 the negative
// infinity.
namespace array_textio_tests;

import scientific from "../..";
import os;
import array_textio_pins;

var TEXTIO_BAD: i64 = 0;

fn num_of(code: i64, x: f64) -> f64 {
    if code == 1 { return 0.0 / 0.0; }
    if code == 2 { return 1.0 / 0.0; }
    if code == 3 { return -1.0 / 0.0; }
    return x;
}

fn val_at(vals: f64[:], pair: i64) -> f64 {
    return num_of(vals[2 * pair + 1] as i64, vals[2 * pair]);
}

// same_val compares one result value with one pin value. The missing
// value matches itself and the infinities match their sign.
fn same_val(got: f64, vals: f64[:], pair: i64) -> bool {
    let code = vals[2 * pair + 1] as i64;
    if code == 1 { return got != got; }
    if code == 2 { return got > 0.0 && got - got != 0.0; }
    if code == 3 { return got < 0.0 && got - got != 0.0; }
    return got == vals[2 * pair];
}

// span_text builds the bytes of one pin run.
fn span_text(bytes: f64[:], spans: f64[:], i: i64) -> u8[>..] {
    var out: u8[>..] = [];
    let at = spans[2 * i] as i64;
    let n = spans[2 * i + 1] as i64;
    var k: i64 = 0;
    while k < n { out.push(bytes[at + k] as u8); k += 1; }
    return out;
}

fn same_text(got: const u8[:], want: const u8[:]) -> bool {
    return got == want;
}

// case_arr builds the array of a case from the value pairs.
fn case_arr(vals: f64[:], start: i64, n: i64, ndim: i64,
            d0: i64, d1: i64, d2: i64) -> scientific::Arr {
    var data: f64[>..] = [];
    var k: i64 = 0;
    while k < n { data.push(val_at(vals, start + k)); k += 1; }
    var s: i64[..8] = [];
    if ndim >= 1 { s.push(d0); }
    if ndim >= 2 { s.push(d1); }
    if ndim >= 3 { s.push(d2); }
    return scientific::Arr { ndim: ndim, shape: s, data: data };
}

// opt_print builds the print options of a case. A field of -1 keeps
// the default of NumPy.
fn opt_print(opts: f64[:], r: i64) -> scientific::PrintOptions {
    var o = scientific::print_options();
    if opts[r] >= 0 { o.precision = opts[r] as i64; }
    if opts[r + 1] >= 0 { o.threshold = opts[r + 1] as i64; }
    if opts[r + 2] >= 0 { o.edgeitems = opts[r + 2] as i64; }
    if opts[r + 3] >= 0 { o.linewidth = opts[r + 3] as i64; }
    if opts[r + 4] >= 0 { o.suppress = opts[r + 4] > 0; }
    let fm = opts[r + 5] as i64;
    if fm == 1 { o.floatmode = 'f'; }
    if fm == 2 { o.floatmode = 'u'; }
    if fm == 3 { o.floatmode = 'e'; }
    let sg = opts[r + 6] as i64;
    if sg == 1 { o.sign = '+'; }
    if sg == 2 { o.sign = ' '; }
    return o;
}

// check_prints runs the print cases of one kind.
fn check_prints(kind_want: i64, name: const u8[:]) {
    let rows = array_textio_pins::P_CASES;
    let opts = array_textio_pins::P_OPTS;
    let vals = array_textio_pins::VALUES;
    let bytes = array_textio_pins::BYTES;
    let spans = array_textio_pins::SPANS;
    let count = rows.len / 10;
    var i: i64 = 0;
    while i < count {
        let r = i * 10;
        if rows[r] as i64 != kind_want { i += 1; continue; }
        let a = case_arr(vals, rows[r + 6] as i64, rows[r + 5] as i64,
                         rows[r + 1] as i64, rows[r + 2] as i64, rows[r + 3] as i64,
                         rows[r + 4] as i64);
        let o = opt_print(opts, (rows[r + 7] as i64) * 7);
        var out: u8[>..] = [];
        var f = scientific::no_fault();
        if kind_want == 1 { out, f = scientific::array2string(a, o); }
        if kind_want == 2 { out, f = scientific::array_repr(a, o); }
        if kind_want == 3 { out, f = scientific::array_str(a, o); }
        let want = span_text(bytes, spans, rows[r + 8] as i64);
        if f.kind != scientific::F_NONE || !same_text(out, want) {
            print(name, " case ", i, " fault ", f.kind, "\n");
            print("  got: '", out, "'\n");
            print(" want: '", want, "'\n");
            TEXTIO_BAD += 1;
        }
        i += 1;
    }
}

fn test_array_textio_string() {
    check_prints(1, "array2string");
    assert(TEXTIO_BAD == 0);
}

fn test_array_textio_repr() {
    check_prints(2, "array_repr");
    assert(TEXTIO_BAD == 0);
}

fn test_array_textio_str() {
    check_prints(3, "array_str");
    assert(TEXTIO_BAD == 0);
}

// opt_load fills the fields of the options of a read that hold no
// text. The caller sets the delimiter and the comment mark itself, in
// its own scope, because the options hold views of the texts.
fn opt_load(o: scientific::TxtOptions&, opts: f64[:], r: i64, usecols: f64[:],
            fillcode: i64, fillval: f64) {
    o.skiprows = opts[r + 2] as i64;
    o.max_rows = opts[r + 3] as i64;
    let nuse = opts[r + 4] as i64;
    let ustart = opts[r + 5] as i64;
    var k: i64 = 0;
    while k < nuse { o.usecols.push(usecols[ustart + k] as i64); k += 1; }
    o.fill = num_of(fillcode, fillval);
}

fn check_shape(a: scientific::Arr&, ndim: i64, d0: i64, d1: i64, d2: i64) -> bool {
    if a.ndim != ndim { return false; }
    var s: i64[3] = [d0, d1, d2];
    var k: i64 = 0;
    while k < ndim {
        if a.shape[k] != s[k] { return false; }
        k += 1;
    }
    return true;
}

// check_loads runs the read cases of one kind.
fn check_loads(kind_want: i64, name: const u8[:]) {
    let rows = array_textio_pins::L_CASES;
    let opts = array_textio_pins::L_OPTS;
    let usecols = array_textio_pins::L_USECOLS;
    let vals = array_textio_pins::VALUES;
    let bytes = array_textio_pins::BYTES;
    let spans = array_textio_pins::SPANS;
    let count = rows.len / 13;
    // The texts of the options live as long as the options, so the
    // buffers that hold them stand outside the loop.
    var tdel: u8[>..] = [];
    var tcom: u8[>..] = [];
    var i: i64 = 0;
    while i < count {
        let r = i * 13;
        if rows[r] as i64 != kind_want { i += 1; continue; }
        let inp = span_text(bytes, spans, rows[r + 7] as i64);
        assert(write_file("sci_textio_in.txt", inp));
        let lrow = (rows[r + 8] as i64) * 6;
        var o = scientific::txt_options();
        let di = opts[lrow] as i64;
        if di >= 0 {
            tdel = span_text(bytes, spans, di);
            o.delimiter = tdel[..];
        }
        let ci = opts[lrow + 1] as i64;
        if ci >= 0 {
            tcom = span_text(bytes, spans, ci);
            o.comments = tcom[..];
        }
        opt_load(o, opts, lrow, usecols, rows[r + 9] as i64, rows[r + 10]);
        var a: scientific::Arr = scientific::empty_arr();
        var f = scientific::no_fault();
        if kind_want == 1 {
            a, f = scientific::loadtxt("sci_textio_in.txt", o);
        } else {
            a, f = scientific::genfromtxt("sci_textio_in.txt", o);
        }
        let n = rows[r + 5] as i64;
        let start = rows[r + 6] as i64;
        var ok = f.kind == scientific::F_NONE;
        if ok {
            ok = check_shape(a, rows[r + 1] as i64, rows[r + 2] as i64,
                             rows[r + 3] as i64, rows[r + 4] as i64);
        }
        var k: i64 = 0;
        while ok && k < n {
            if !same_val(a.data[k], vals, start + k) { ok = false; }
            k += 1;
        }
        if !ok {
            print(name, " case ", i, " fault ", f.kind, " ndim ", a.ndim, "\n");
            TEXTIO_BAD += 1;
        }
        assert(delete_file("sci_textio_in.txt"));
        i += 1;
    }
}

fn test_array_textio_loadtxt() {
    check_loads(1, "loadtxt");
    assert(TEXTIO_BAD == 0);
}

fn test_array_textio_genfromtxt() {
    check_loads(2, "genfromtxt");
    assert(TEXTIO_BAD == 0);
}

fn test_array_textio_savetxt() {
    let rows = array_textio_pins::S_CASES;
    let opts = array_textio_pins::S_OPTS;
    let vals = array_textio_pins::VALUES;
    let bytes = array_textio_pins::BYTES;
    let spans = array_textio_pins::SPANS;
    let count = rows.len / 10;
    // The texts of the options live as long as the options, so the
    // buffers that hold them stand outside the loop.
    var tdel: u8[>..] = [];
    var thdr: u8[>..] = [];
    var tftr: u8[>..] = [];
    var tmrk: u8[>..] = [];
    var i: i64 = 0;
    while i < count {
        let r = i * 10;
        let a = case_arr(vals, rows[r + 6] as i64, rows[r + 5] as i64,
                         rows[r + 1] as i64, rows[r + 2] as i64, rows[r + 3] as i64,
                         rows[r + 4] as i64);
        let orow = (rows[r + 7] as i64) * 6;
        var o = scientific::save_options();
        let fc = opts[orow] as i64;
        if fc == 1 { o.format = 'f'; }
        if fc == 2 { o.format = 'd'; }
        o.precision = opts[orow + 1] as i64;
        tdel = span_text(bytes, spans, opts[orow + 2] as i64);
        o.delimiter = tdel[..];
        if opts[orow + 3] >= 0 {
            thdr = span_text(bytes, spans, opts[orow + 3] as i64);
            o.header = thdr[..];
        }
        if opts[orow + 4] >= 0 {
            tftr = span_text(bytes, spans, opts[orow + 4] as i64);
            o.footer = tftr[..];
        }
        tmrk = span_text(bytes, spans, opts[orow + 5] as i64);
        o.mark = tmrk[..];
        let ok, f = scientific::savetxt("sci_textio_out.txt", a, o);
        var got: u8[>..] = [];
        var read_ok = false;
        if ok && f.kind == scientific::F_NONE {
            read_ok = read_file("sci_textio_out.txt", got);
        }
        let want = span_text(bytes, spans, rows[r + 8] as i64);
        if !read_ok || !same_text(got, want) {
            print("savetxt case ", i, " fault ", f.kind, "\n");
            print("  got: '", got, "'\n want: '", want, "'\n");
            TEXTIO_BAD += 1;
        }
        assert(delete_file("sci_textio_out.txt"));
        i += 1;
    }
    assert(TEXTIO_BAD == 0);
}

// npy_case builds the array of a save case.
fn npy_case(r: i64) -> scientific::Arr {
    let rows = array_textio_pins::N_CASES;
    return case_arr(array_textio_pins::VALUES, rows[r + 5] as i64, rows[r + 4] as i64,
                    rows[r] as i64, rows[r + 1] as i64, rows[r + 2] as i64,
                    rows[r + 3] as i64);
}

fn test_array_textio_save() {
    let rows = array_textio_pins::N_CASES;
    let bytes = array_textio_pins::BYTES;
    let spans = array_textio_pins::SPANS;
    let count = rows.len / 8;
    var i: i64 = 0;
    while i < count {
        let r = i * 8;
        let a = npy_case(r);
        let ok, f = scientific::save("sci_textio.npy", a);
        assert(ok && f.kind == scientific::F_NONE);
        var got: u8[>..] = [];
        assert(read_file("sci_textio.npy", got));
        let want = span_text(bytes, spans, rows[r + 6] as i64);
        if !same_text(got, want) {
            print("save case ", i, " holds ", got.len, " bytes, the pin holds ",
                  want.len, "\n");
            TEXTIO_BAD += 1;
        }
        assert(delete_file("sci_textio.npy"));
        i += 1;
    }
    assert(TEXTIO_BAD == 0);
}

fn test_array_textio_load() {
    let rows = array_textio_pins::N_CASES;
    let vals = array_textio_pins::VALUES;
    let count = rows.len / 8;
    var i: i64 = 0;
    while i < count {
        let r = i * 8;
        let a = npy_case(r);
        let ok, f = scientific::save("sci_textio.npy", a);
        assert(ok && f.kind == scientific::F_NONE);
        let b, g = scientific::load("sci_textio.npy");
        var good = g.kind == scientific::F_NONE;
        if good {
            good = check_shape(b, rows[r] as i64, rows[r + 1] as i64,
                               rows[r + 2] as i64, rows[r + 3] as i64);
        }
        var k: i64 = 0;
        while good && k < rows[r + 4] as i64 {
            if !same_val(b.data[k], vals, rows[r + 5] as i64 + k) { good = false; }
            k += 1;
        }
        if !good {
            print("load case ", i, " fault ", g.kind, "\n");
            TEXTIO_BAD += 1;
        }
        assert(delete_file("sci_textio.npy"));
        i += 1;
    }
    assert(TEXTIO_BAD == 0);
}

// holds tells whether a text holds a word.
fn holds(hay: const u8[:], word: const u8[:]) -> bool {
    if word.len == 0 { return true; }
    var i: i64 = 0;
    while i + word.len <= hay.len {
        var k: i64 = 0;
        while k < word.len && hay[i + k] == word[k] { k += 1; }
        if k == word.len { return true; }
        i += 1;
    }
    return false;
}

fn test_array_textio_savez() {
    let a1 = npy_case(0);
    let a2 = npy_case(8);
    let ok, f = scientific::save("sci_textio_one.npy", a1);
    assert(ok && f.kind == scientific::F_NONE);
    let ok2, f2 = scientific::save("sci_textio_two.npy", a2);
    assert(ok2 && f2.kind == scientific::F_NONE);
    var one: u8[>..] = [];
    var two: u8[>..] = [];
    assert(read_file("sci_textio_one.npy", one));
    assert(read_file("sci_textio_two.npy", two));
    let ok3, f3 = scientific::savez("sci_textio.npz", "x", a1, "y", a2);
    assert(ok3 && f3.kind == scientific::F_NONE);
    var z: u8[>..] = [];
    assert(read_file("sci_textio.npz", z));
    if !holds(z, "x.npy") || !holds(z, "y.npy") || !holds(z, one[..]) || !holds(z, two[..]) {
        print("savez misses a member or its bytes\n");
        TEXTIO_BAD += 1;
    }
    let ok4, f4 = scientific::savez("sci_textio.npz", "solo", a1);
    assert(ok4 && f4.kind == scientific::F_NONE);
    var z1: u8[>..] = [];
    assert(read_file("sci_textio.npz", z1));
    if !holds(z1, "solo.npy") || !holds(z1, one[..]) || holds(z1, "y.npy") {
        print("savez of one array holds the wrong members\n");
        TEXTIO_BAD += 1;
    }
    assert(delete_file("sci_textio_one.npy"));
    assert(delete_file("sci_textio_two.npy"));
    assert(delete_file("sci_textio.npz"));
    assert(TEXTIO_BAD == 0);
}

fn test_array_textio_savez_compressed() {
    let a1 = npy_case(0);
    let a2 = npy_case(8);
    let ok, f = scientific::savez_compressed("sci_textio_c.npz", "x", a1, "y", a2);
    assert(ok && f.kind == scientific::F_NONE);
    var z: u8[>..] = [];
    assert(read_file("sci_textio_c.npz", z));
    let ok2, f2 = scientific::save("sci_textio_one.npy", a1);
    assert(ok2 && f2.kind == scientific::F_NONE);
    var one: u8[>..] = [];
    assert(read_file("sci_textio_one.npy", one));
    if !holds(z, "x.npy") || !holds(z, "y.npy") || !holds(z, one[..]) {
        print("savez_compressed misses a member or its bytes\n");
        TEXTIO_BAD += 1;
    }
    assert(delete_file("sci_textio_one.npy"));
    assert(delete_file("sci_textio_c.npz"));
    assert(TEXTIO_BAD == 0);
}

fn test_array_textio_regex() {
    let rc = array_textio_pins::R_CASES;
    let bytes = array_textio_pins::BYTES;
    let spans = array_textio_pins::SPANS;
    let vals = array_textio_pins::VALUES;
    let pat = span_text(bytes, spans, rc[2] as i64);
    let inp = span_text(bytes, spans, rc[3] as i64);
    assert(write_file("sci_textio_in.txt", inp));
    let a, f = scientific::fromregex("sci_textio_in.txt", pat[..]);
    assert(delete_file("sci_textio_in.txt"));
    var good = f.kind == scientific::F_NONE;
    if good { good = check_shape(a, rc[4] as i64, rc[5] as i64, rc[6] as i64, 1); }
    var k: i64 = 0;
    while good && k < rc[0] as i64 {
        if !same_val(a.data[k], vals, rc[1] as i64 + k) { good = false; }
        k += 1;
    }
    if !good {
        print("fromregex fault ", f.kind, " ndim ", a.ndim, "\n");
        TEXTIO_BAD += 1;
    }
    assert(TEXTIO_BAD == 0);
}

fn test_array_textio_faults() {
    let vals = array_textio_pins::VALUES;
    let bytes = array_textio_pins::BYTES;
    let spans = array_textio_pins::SPANS;
    var o = scientific::txt_options();
    // A path with a NUL byte is refused before any read or write.
    var p: u8[>..] = [];
    p.append("sci_textio");
    p.push(0);
    p.append(".txt");
    let small = case_arr(vals, 0, 3, 1, 3, 1, 1);
    let _a, f1 = scientific::loadtxt(p[..], o);
    assert(f1.kind == scientific::F_VALUE);
    let _s, f2 = scientific::savetxt(p[..], small, o);
    assert(f2.kind == scientific::F_VALUE);
    let _w, f3 = scientific::save(p[..], small);
    assert(f3.kind == scientific::F_VALUE);
    // A file that does not read.
    let _b, f4 = scientific::loadtxt("sci_textio_no.txt", o);
    assert(f4.kind == scientific::F_VALUE);
    let _c, f5 = scientific::load("sci_textio_no.npy");
    assert(f5.kind == scientific::F_VALUE);
    // A field that holds no number is a fault of loadtxt.
    var t1: u8[>..] = [];
    t1.append("1,x\n");
    assert(write_file("sci_textio_in.txt", t1));
    let _d, f6 = scientific::loadtxt("sci_textio_in.txt", o);
    assert(f6.kind == scientific::F_VALUE);
    assert(delete_file("sci_textio_in.txt"));
    // A row with another count of columns is a fault.
    var t2: u8[>..] = [];
    t2.append("1,2\n3\n");
    assert(write_file("sci_textio_in.txt", t2));
    var o2 = scientific::txt_options();
    o2.delimiter = ",";
    let _e, f7 = scientific::loadtxt("sci_textio_in.txt", o2);
    assert(f7.kind == scientific::F_SHAPE);
    assert(delete_file("sci_textio_in.txt"));
    // A column of usecols outside the row is a fault.
    var t3: u8[>..] = [];
    t3.append("1\n");
    assert(write_file("sci_textio_in.txt", t3));
    var o3 = scientific::txt_options();
    o3.usecols.push(3);
    let _g, f8 = scientific::loadtxt("sci_textio_in.txt", o3);
    assert(f8.kind == scientific::F_SHAPE);
    // A delimiter of more than one byte is a fault.
    var o4 = scientific::txt_options();
    o4.delimiter = ",,";
    let _h, f9 = scientific::loadtxt("sci_textio_in.txt", o4);
    assert(f9.kind == scientific::F_LIMIT);
    assert(delete_file("sci_textio_in.txt"));
    // A bad format kind is a fault of savetxt.
    var o5 = scientific::save_options();
    o5.format = 'x';
    let _i, f10 = scientific::savetxt("sci_textio_out.txt", small, o5);
    assert(f10.kind == scientific::F_VALUE);
    // An array of three axes is a fault of savetxt.
    let cube = case_arr(vals, 0, 4, 3, 1, 2, 2);
    let _j, f11 = scientific::savetxt("sci_textio_out.txt", cube, o);
    assert(f11.kind == scientific::F_SHAPE);
    // A file without the magic is a fault of load.
    var t4: u8[>..] = [];
    t4.append("no");
    assert(write_file("sci_textio.npy", t4));
    let _k, f12 = scientific::load("sci_textio.npy");
    assert(f12.kind == scientific::F_VALUE);
    // A header that claims more numbers than the bytes hold.
    let cut = span_text(bytes, spans, array_textio_pins::N_CASES[6] as i64);
    assert(write_file("sci_textio.npy", cut[..cut.len - 8]));
    let _m, f13 = scientific::load("sci_textio.npy");
    assert(f13.kind == scientific::F_LIMIT);
    // A file of another type is a fault. The type sits behind 'f' in
    // the header.
    let full = span_text(bytes, spans, array_textio_pins::N_CASES[6] as i64);
    var other: u8[>..] = [];
    var q: i64 = 0;
    while q < full.len {
        if q + 1 < full.len && full[q] == 'f' && full[q + 1] == '8' {
            other.push('f');
            other.push('4');
            q += 2;
        } else {
            other.push(full[q]);
            q += 1;
        }
    }
    assert(write_file("sci_textio.npy", other));
    let _n, f14 = scientific::load("sci_textio.npy");
    assert(f14.kind == scientific::F_VALUE);
    // The Fortran order is a fault. The word False holds the room of
    // True plus a space.
    var fort: u8[>..] = [];
    q = 0;
    while q < full.len {
        if q + 4 < full.len && full[q] == 'F' && full[q + 1] == 'a' && full[q + 2] == 'l'
            && full[q + 3] == 's' && full[q + 4] == 'e' {
            fort.append("True ");
            q += 5;
        } else {
            fort.push(full[q]);
            q += 1;
        }
    }
    assert(write_file("sci_textio.npy", fort));
    let _p, f15 = scientific::load("sci_textio.npy");
    assert(f15.kind == scientific::F_VALUE);
    assert(delete_file("sci_textio.npy"));
    // A pattern that does not compile is a fault.
    var t5: u8[>..] = [];
    t5.append("1.5 2.5\n");
    assert(write_file("sci_textio_in.txt", t5));
    let _r, f16 = scientific::fromregex("sci_textio_in.txt", "[");
    assert(f16.kind == scientific::F_VALUE);
    // A pattern with no group is a fault.
    let _t, f17 = scientific::fromregex("sci_textio_in.txt", "[0-9]+");
    assert(f17.kind == scientific::F_VALUE);
    // A file with no match is a fault.
    var t6: u8[>..] = [];
    t6.append("zz\n");
    assert(write_file("sci_textio_in.txt", t6));
    let _u, f18 = scientific::fromregex("sci_textio_in.txt", "([0-9.]+)");
    assert(f18.kind == scientific::F_SHAPE);
    // A name with a slash is out of an archive.
    let ok, f19 = scientific::savez("sci_textio.npz", "x/y", small);
    assert(!ok && f19.kind == scientific::F_VALUE);
    assert(delete_file("sci_textio_in.txt"));
}
