// The text files of NumPy: savetxt, loadtxt, genfromtxt and fromregex.
//
// NumPy reads and writes the text tables of its arrays. This file does
// the same for the one array type of the package. A file is read as
// bytes, because the package holds no text type, and the numbers are
// read out of the bytes. The reader of NumPy holds more kinds of text;
// the limits of the package README say which ones stay out.
//
// The options of NumPy stand in one struct, because Goose has no named
// arguments. The fields of a read are the delimiter, the comment mark,
// the rows to skip, the rows to hold and the columns to take. The
// fields of a write are the delimiter, the header, the footer, the mark
// of a header line, and the format of a number. The format is a fixed
// set of kinds, never a caller format string, so no byte of a file can
// reach the writer of the text.
//
// A file holds at most TEXT_MAX_BYTES bytes. A read holds at most
// TEXT_MAX_ROWS rows and TEXT_MAX_COLS columns, and the result holds at
// most MAX_ELEMENTS numbers.
namespace scientific;

import os;
import regex;
import std;

const TEXT_MAX_BYTES: i64 = 268435456;
const TEXT_MAX_ROWS: i64 = 16777216;
const TEXT_MAX_COLS: i64 = 1048576;

// TxtOptions holds the options of the text calls. The delimiter of no
// bytes splits at any run of spaces. The comment mark of no bytes turns
// the comments off. A max_rows of -1 holds every row. An empty usecols
// takes every column. The fill holds the value of a missing field of
// genfromtxt. The format of a write is one of the bytes e (the
// scientific form), f (the fixed form) and d (the whole part).
struct TxtOptions {
    delimiter: const u8[:], comments: const u8[:], skiprows: i64, max_rows: i64,
    fill: f64,
    header: const u8[:], footer: const u8[:], mark: const u8[:],
    format: u8, precision: i64, usecols: i64[>..]
}

// txt_options gives the defaults of a read: any space, the comment mark
// of NumPy, no rows skipped, every row, every column and the missing
// value.
fn txt_options() -> TxtOptions {
    return TxtOptions {
        delimiter: "", comments: "#", skiprows: 0, max_rows: -1,
        fill: 0.0 / 0.0,
        header: "", footer: "", mark: "# ", format: 'e', precision: 18,
        usecols: []
    };
}

// save_options gives the defaults of a write: a space between the
// columns, no header and no footer, the mark of NumPy and the
// scientific form with 18 digits after the point.
fn save_options() -> TxtOptions {
    var o = txt_options();
    o.delimiter = " ";
    return o;
}

// txt_read reads the bytes of a file for the text calls. A path with a
// NUL byte is a fault, and the count of bytes is bound.
fn txt_read(path: const u8[:], tag: const u8[:]) -> u8[>..], Fault {
    var k: i64 = 0;
    while k < path.len {
        if path[k] == 0 {
            var none: u8[>..] = [];
            return none, fault(F_VALUE, tag, "the path holds a NUL byte");
        }
        k += 1;
    }
    var bytes: u8[>..] = [];
    if !read_file(path, bytes) {
        return bytes, fault(F_VALUE, tag, "the file does not read");
    }
    if bytes.len > TEXT_MAX_BYTES {
        return bytes, fault(F_LIMIT, tag, "the file holds more bytes than the limit");
    }
    return bytes, no_fault();
}

// txt_eq tells whether a text holds a word, the case of a letter aside.
fn txt_eq(s: const u8[:], word: const u8[:]) -> bool {
    if s.len != word.len { return false; }
    var i: i64 = 0;
    while i < s.len {
        let c = s[i];
        let d = if c >= 'A' && c <= 'Z' { c + 32 } else { c };
        if d != word[i] { return false; }
        i += 1;
    }
    return true;
}

// txt_number reads one field as a number. The flag tells whether the
// field holds one. The words of Python for the missing value and the
// infinities count too.
fn txt_number(field: const u8[:]) -> f64, bool {
    var a: i64 = 0;
    var b = field.len;
    while a < b && (field[a] == ' ' || field[a] == '\t') { a += 1; }
    while b > a && (field[b - 1] == ' ' || field[b - 1] == '\t') { b -= 1; }
    if a == b { return 0.0, false; }
    let t = field[a..b];
    if txt_eq(t, "nan") { return 0.0 / 0.0, true; }
    if txt_eq(t, "inf") || txt_eq(t, "infinity") { return 1.0 / 0.0, true; }
    if txt_eq(t, "+inf") || txt_eq(t, "+infinity") { return 1.0 / 0.0, true; }
    if txt_eq(t, "-inf") || txt_eq(t, "-infinity") { return -1.0 / 0.0, true; }
    var buf: u8[>..] = [];
    buf.append(t);
    return parse_flt(buf[..]);
}

// txt_fields splits one line into fields. The starts and the lengths
// hold the place of each field in the bytes. A delimiter of one byte
// ends a field, and the spaces of a field drop.
fn txt_fields(bytes: const u8[:], lo: i64, hi: i64, delim: const u8[:],
              starts: i64[>..]&, lens: i64[>..]&) {
    starts.clear();
    lens.clear();
    if delim.len == 0 {
        var i = lo;
        while i < hi {
            while i < hi && is_space(bytes[i]) { i += 1; }
            if i >= hi { break; }
            let s = i;
            while i < hi && !is_space(bytes[i]) { i += 1; }
            starts.push(s);
            lens.push(i - s);
        }
        return;
    }
    let d = delim[0];
    var i = lo;
    while true {
        var j = i;
        while j < hi && bytes[j] != d { j += 1; }
        var a = i;
        var b = j;
        while a < b && (bytes[a] == ' ' || bytes[a] == '\t') { a += 1; }
        while b > a && (bytes[b - 1] == ' ' || bytes[b - 1] == '\t') { b -= 1; }
        starts.push(a);
        lens.push(b - a);
        if j >= hi { break; }
        i = j + 1;
    }
}

// txt_cut gives the end of the part of a line before its comment. The
// mark of no bytes cuts nothing.
fn txt_cut(bytes: const u8[:], lo: i64, hi: i64, mark: const u8[:]) -> i64 {
    if mark.len == 0 { return hi; }
    var i = lo;
    while i < hi {
        var k: i64 = 0;
        while k < mark.len && i + k < hi && bytes[i + k] == mark[k] { k += 1; }
        if k == mark.len { return i; }
        i += 1;
    }
    return hi;
}

// txt_take maps the columns of a row to the columns of the usecols. A
// column outside the row is a fault.
fn txt_take(count: i64, o: TxtOptions&, tag: const u8[:]) -> Fault {
    var k: i64 = 0;
    while k < o.usecols.len {
        if o.usecols[k] < 0 || o.usecols[k] >= count {
            return fault(F_SHAPE, tag, "a column of usecols sits outside the row");
        }
        k += 1;
    }
    return no_fault();
}

// txt_table reads the rows of a text file into one array. The gen flag
// chooses the rules of genfromtxt: a missing field or a field of no
// number takes the fill value, and the rows of different lengths pad.
// The rules of loadtxt ask for the same count of columns in every row
// and a number in every field.
fn txt_table(path: const u8[:], o: TxtOptions&, tag: const u8[:], gen: bool) -> Arr, Fault {
    let bytes, fr = txt_read(path, tag);
    if fr.kind != F_NONE { return empty_arr(), fr; }
    if o.delimiter.len > 1 {
        return empty_arr(), fault(F_LIMIT, tag, "the delimiter holds more than one byte");
    }
    var vals: f64[>..] = [];
    var counts: i64[>..] = [];
    var line: i64 = 0;
    var lo: i64 = 0;
    var starts: i64[>..] = [];
    var lens: i64[>..] = [];
    while lo < bytes.len {
        var hi = lo;
        while hi < bytes.len && bytes[hi] != '\n' { hi += 1; }
        line += 1;
        let body_hi = txt_cut(bytes, lo, hi, o.comments);
        // A line of spaces only holds no data row.
        var has_data = false;
        var q = lo;
        while q < body_hi {
            if !is_space(bytes[q]) { has_data = true; break; }
            q += 1;
        }
        var use_line = line > o.skiprows && has_data;
        if use_line {
            txt_fields(bytes, lo, body_hi, o.delimiter, starts, lens);
            if starts.len == 0 { use_line = false; }
        }
        var done = false;
        if use_line {
            let nf = starts.len;
            if nf > TEXT_MAX_COLS {
                return empty_arr(), fault(F_LIMIT, tag, "a row holds more columns than the limit");
            }
            var fc = nf;
            if o.usecols.len > 0 {
                let ft = txt_take(nf, o, tag);
                if ft.kind != F_NONE { return empty_arr(), ft; }
                fc = o.usecols.len;
            }
            counts.push(fc);
            var c: i64 = 0;
            while c < fc {
                let at = if o.usecols.len > 0 { o.usecols[c] } else { c };
                var s = starts[at];
                var e = s + lens[at];
                let v, ok = txt_number(bytes[s..e]);
                if !ok {
                    if gen {
                        vals.push(o.fill);
                    } else {
                        return empty_arr(), fault(F_VALUE, tag, "a field holds no number");
                    }
                } else {
                    vals.push(v);
                }
                c += 1;
            }
            if counts.len > TEXT_MAX_ROWS {
                return empty_arr(), fault(F_LIMIT, tag, "the file holds more rows than the limit");
            }
            if o.max_rows >= 0 && counts.len == o.max_rows { done = true; }
        }
        if done { break; }
        lo = hi + 1;
    }
    if counts.len == 0 {
        return empty_arr(), fault(F_SHAPE, tag, "the file holds no data row");
    }
    // The width of the table. The rows of genfromtxt pad to the widest
    // row with the fill value.
    var width = counts[0];
    var i: i64 = 1;
    while i < counts.len {
        if counts[i] > width { width = counts[i]; }
        i += 1;
    }
    if !gen {
        i = 0;
        while i < counts.len {
            if counts[i] != width {
                return empty_arr(), fault(F_SHAPE, tag, "a row holds a different count of columns");
            }
            i += 1;
        }
    }
    if counts.len * width > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, tag, "the result holds more elements than the limit");
    }
    var d: f64[>..] = [];
    i = 0;
    var at: i64 = 0;
    while i < counts.len {
        var c: i64 = 0;
        while c < width {
            if c < counts[i] {
                d.push(vals[at]);
                at += 1;
            } else {
                d.push(o.fill);
            }
            c += 1;
        }
        i += 1;
    }
    var s: i64[..8] = [];
    if width == 1 {
        s.push(counts.len);
        return Arr { ndim: 1, shape: s, data: d }, no_fault();
    }
    s.push(counts.len);
    s.push(width);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// loadtxt reads a text file of numbers, as NumPy does. A file of one
// column gives a row of numbers.
fn loadtxt(path: const u8[:], o: TxtOptions&) -> Arr, Fault {
    return txt_table(path, o, "loadtxt", false);
}

// genfromtxt reads a text file of numbers and takes the missing value
// for every field that holds none, as NumPy does. The fill of the
// options holds the value of a missing field.
fn genfromtxt(path: const u8[:], o: TxtOptions&) -> Arr, Fault {
    return txt_table(path, o, "genfromtxt", true);
}

// txt_value writes one number in the format of savetxt. The kinds are
// the scientific form with the digits after the point, the fixed form,
// and the whole part toward zero.
fn txt_value(x: f64, o: TxtOptions&, out: u8[>..]&) -> Fault {
    if o.format == 'd' {
        if x != x || x - x != 0.0 {
            return fault(F_VALUE, "savetxt", "the whole format holds no missing value or infinity");
        }
        var t = x;
        if t < 0.0 { t = -t; }
        var n = t as i64;
        if x < 0.0 { n = -n; }
        format_int(out, n, 10, 0, '0');
        return no_fault();
    }
    var ff = float_format();
    ff.unique = false;
    ff.precision = o.precision;
    ff.trim = 'k';
    var s: u8[>..] = [];
    var f = no_fault();
    if o.format == 'e' {
        s, f = format_float_scientific(x, ff);
    } else {
        s, f = format_float_positional(x, ff);
    }
    if f.kind != F_NONE { return f; }
    out.append(s);
    return no_fault();
}

// txt_marked writes a header or a footer with the mark before it and
// before every line inside it, as NumPy marks them.
fn txt_marked(text: const u8[:], mark: const u8[:], out: u8[>..]&) {
    out.append(mark);
    var i: i64 = 0;
    while i < text.len {
        if text[i] == '\n' {
            out.push('\n');
            out.append(mark);
        } else {
            out.push(text[i]);
        }
        i += 1;
    }
}

// savetxt writes an array as a text file, as NumPy does. A row of the
// array is one line of the file. The format of the options is one of
// the kinds e, f and d, never a format string of the caller.
fn savetxt(path: const u8[:], a: Arr&, o: TxtOptions&) -> bool, Fault {
    var k: i64 = 0;
    while k < path.len {
        if path[k] == 0 { return false, fault(F_VALUE, "savetxt", "the path holds a NUL byte"); }
        k += 1;
    }
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, "savetxt");
    if fs.kind != F_NONE { return false, fs; }
    if a.ndim > 2 { return false, fault(F_SHAPE, "savetxt", "the array holds more than two axes"); }
    if o.format != 'e' && o.format != 'f' && o.format != 'd' {
        return false, fault(F_VALUE, "savetxt", "the format is not e, f or d");
    }
    if o.precision < 0 || o.precision > TEXT_LIMIT {
        return false, fault(F_VALUE, "savetxt", "the precision is outside the range");
    }
    if o.delimiter.len > 64 || o.mark.len > 64 {
        return false, fault(F_LIMIT, "savetxt", "the delimiter or the mark is above its limit");
    }
    var out: u8[>..] = [];
    if o.header.len > 0 {
        txt_marked(o.header, o.mark, out);
        out.push('\n');
    }
    let cols = if a.ndim == 1 { 1 } else { a.shape[1] };
    let rows = a.shape[0];
    var i: i64 = 0;
    while i < rows {
        var c: i64 = 0;
        while c < cols {
            if c > 0 { out.append(o.delimiter); }
            let at = if a.ndim == 1 { i } else { i * cols + c };
            let f = txt_value(a.data[at], o, out);
            if f.kind != F_NONE { return false, f; }
            c += 1;
        }
        out.push('\n');
        i += 1;
    }
    if o.footer.len > 0 {
        txt_marked(o.footer, o.mark, out);
        out.push('\n');
    }
    if !write_file(path, out) {
        return false, fault(F_VALUE, "savetxt", "the file does not write");
    }
    return true, no_fault();
}

// fromregex reads a file and takes one row for every match of the
// pattern. Each group of the pattern gives one column, and each group
// reads as a number. A group of no part, or a group that reads as no
// number, takes the missing value. A pattern with no group is a fault.
fn fromregex(path: const u8[:], pattern: const u8[:]) -> Arr, Fault {
    let bytes, fr = txt_read(path, "fromregex");
    if fr.kind != F_NONE { return empty_arr(), fr; }
    var pat: u8[>..] = [];
    pat.append(pattern);
    var err: u8[>..] = [];
    var re = regex::compile(pat[..], err);
    if err.len > 0 {
        return empty_arr(), fault(F_VALUE, "fromregex", "the pattern does not compile");
    }
    if re.groups < 2 {
        return empty_arr(), fault(F_VALUE, "fromregex", "the pattern holds no group");
    }
    let cols = re.groups - 1;
    if cols > TEXT_MAX_COLS {
        return empty_arr(), fault(F_LIMIT, "fromregex", "the pattern holds more groups than the limit");
    }
    var vals: f64[>..] = [];
    regex::each_captures(re, bytes[..]) { caps =>
        var k: i64 = 1;
        while k <= cols {
            let g = regex::group(caps, k);
            var buf: u8[>..] = [];
            buf.append(g);
            let v, ok = parse_flt(buf[..]);
            if ok { vals.push(v); } else { vals.push(0.0 / 0.0); }
            k += 1;
        }
    };
    let rows = vals.len / cols;
    if rows == 0 {
        return empty_arr(), fault(F_SHAPE, "fromregex", "the file holds no match");
    }
    if rows * cols > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "fromregex", "the result holds more elements than the limit");
    }
    var s: i64[..8] = [];
    if cols == 1 {
        s.push(rows);
        return Arr { ndim: 1, shape: s, data: vals }, no_fault();
    }
    s.push(rows);
    s.push(cols);
    return Arr { ndim: 2, shape: s, data: vals }, no_fault();
}
