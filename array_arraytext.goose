// The text of an array: array2string, array_repr and array_str of NumPy.
//
// NumPy prints an array in three steps. It finds the form and the width
// of the numbers from the numbers that show, it writes each number in
// that width, and it lays the numbers out in rows that wrap at the line
// width. This file follows the same steps with the number text of
// array_text_float. The package holds one type of number, the float, so
// the whole number and the bool forms of NumPy do not exist here. The
// options of NumPy stand in one struct, because Goose has no named
// arguments.
//
// An array of more elements than the threshold shows its edge items
// only, with three dots between them. A print holds at most
// PRINT_MAX_ITEMS numbers and PRINT_MAX_BYTES bytes.
namespace scientific;

import std;

const PRINT_MAX_ITEMS: i64 = 4194304;
const PRINT_MAX_BYTES: i64 = 268435456;
const PRINT_MAX_WIDTH: i64 = 4096;

// PrintOptions holds the print options of NumPy. The floatmode is one
// of the bytes f (fixed), u (unique), m (maxprec) and e (maxprec_equal).
// The sign is one of the bytes minus, plus and space.
struct PrintOptions {
    precision: i64, threshold: i64, edgeitems: i64, linewidth: i64,
    suppress: bool, floatmode: u8, sign: u8
}

// print_options gives the defaults of NumPy: precision 8, threshold
// 1000, edgeitems 3, line width 75, no suppress, floatmode maxprec and
// sign minus.
fn print_options() -> PrintOptions {
    return PrintOptions {
        precision: 8, threshold: 1000, edgeitems: 3, linewidth: 75,
        suppress: false, floatmode: 'm', sign: '-'
    };
}

// ApCtx holds the form that the numbers of one print share.
struct ApCtx {
    exp_format: bool, unique: bool, trim: u8, sign: u8,
    precision: i64, min_digits: i64, pad_left: i64, pad_right: i64, exp_size: i64
}

// ap_axis_plan gives the shape of one axis of a print. The first holds
// the count of the leading items, the second the count of the trailing
// items, and the third tells whether the axis shows a summary. A summary
// axis shows its edge items and three dots between them.
fn ap_axis_plan(alen: i64, summary: bool, edge: i64) -> i64, i64, bool {
    if summary && 2 * edge < alen { return edge, edge, true; }
    return 0, alen, false;
}

// ap_pick gives the place in the array of the k-th shown item of an
// axis. The leading items come first, and the trailing items come from
// the end of the axis.
fn ap_pick(alen: i64, summary: bool, edge: i64, k: i64) -> i64 {
    let lead, trail, _ = ap_axis_plan(alen, summary, edge);
    if k < lead { return k; }
    return alen - trail + (k - lead);
}

// ap_stride gives the step in the buffer of one place of an axis.
fn ap_stride(a: Arr&, axis: i64) -> i64 {
    var s: i64 = 1;
    var d: i64 = axis + 1;
    while d < a.ndim { s *= a.shape[d]; d += 1; }
    return s;
}

// ap_count gives the count of the items that one axis shows.
fn ap_count(alen: i64, summary: bool, edge: i64) -> i64 {
    let lead, trail, _ = ap_axis_plan(alen, summary, edge);
    return lead + trail;
}

// ap_gather copies the numbers that show, in the row major order. A
// summary shows the corner numbers of the array, as the corner array of
// NumPy holds.
fn ap_gather(a: Arr&, summary: bool, edge: i64, out: f64[>..]&) {
    var cnt: i64[..8] = [];
    var ctr: i64[..8] = [];
    var d: i64 = 0;
    while d < a.ndim {
        cnt.push(ap_count(a.shape[d], summary, edge));
        ctr.push(0);
        d += 1;
    }
    var total: i64 = 1;
    d = 0;
    while d < a.ndim { total *= cnt[d]; d += 1; }
    var n: i64 = 0;
    while n < total {
        var off: i64 = 0;
        d = 0;
        while d < a.ndim {
            off += ap_pick(a.shape[d], summary, edge, ctr[d]) * ap_stride(a, d);
            d += 1;
        }
        out.push(a.data[off]);
        d = a.ndim - 1;
        while d >= 0 {
            ctr[d] += 1;
            if ctr[d] < cnt[d] { break; }
            ctr[d] = 0;
            d -= 1;
        }
        n += 1;
    }
}

// ap_split gives the widths of a number text. The first holds the count
// of the characters before the point, the second the count after it, and
// the third the count of the digits of the exponent.
fn ap_split(s: const u8[:]) -> i64, i64, i64 {
    var dot: i64 = -1;
    var exp: i64 = -1;
    var k: i64 = 0;
    while k < s.len {
        if s[k] == '.' && dot < 0 { dot = k; }
        if s[k] == 'e' && exp < 0 { exp = k; }
        k += 1;
    }
    if dot < 0 { dot = exp; }
    if dot < 0 { dot = s.len; }
    if exp < 0 { exp = s.len; }
    var digits = s.len - exp - 1;
    if digits > 0 && (s[exp + 1] == '+' || s[exp + 1] == '-') { digits -= 1; }
    return dot, exp - dot - 1, digits;
}

// ap_fill finds the form of the numbers from the numbers that show. The
// rules are the rules of the class FloatingFormat of NumPy.
fn ap_fill(vals: f64[:], o: PrintOptions) -> ApCtx, Fault {
    var c = ApCtx {
        exp_format: false, unique: true, trim: '.', sign: o.sign,
        precision: -1, min_digits: 0, pad_left: 0, pad_right: 0, exp_size: -1
    };
    var prec = o.precision;
    if o.floatmode == 'u' { prec = -1; }
    var nfinite: i64 = 0;
    var nonfinite: i64 = 0;
    var neginf = false;
    var anyneg = false;
    var nnz: i64 = 0;
    var mx: f64 = 0.0;
    var mn: f64 = 0.0;
    var i: i64 = 0;
    while i < vals.len {
        let x = vals[i];
        if x != x {
            nonfinite += 1;
        } else if x - x != 0.0 {
            nonfinite += 1;
            if x < 0.0 { neginf = true; }
        } else {
            nfinite += 1;
            if x < 0.0 || (x == 0.0 && 1.0 / x < 0.0) { anyneg = true; }
            let ax = if x < 0.0 { -x } else { x };
            if ax != 0.0 {
                if nnz == 0 {
                    mx = ax;
                    mn = ax;
                } else {
                    if ax > mx { mx = ax; }
                    if ax < mn { mn = ax; }
                }
                nnz += 1;
            }
        }
        i += 1;
    }
    // The exponent form starts at 1e8, and below 1e-4 or above a ratio
    // of 1000, unless small values are suppressed.
    if nnz > 0 && (mx >= 100000000.0 || (!o.suppress && (mn < 0.0001 || mx / mn > 1000.0))) {
        c.exp_format = true;
    }
    if nfinite > 0 {
        var ff = float_format();
        ff.precision = prec;
        ff.sign = o.sign == '+';
        let unique = o.floatmode != 'f';
        ff.unique = unique;
        if o.floatmode == 'f' { ff.trim = 'k'; } else { ff.trim = '.'; }
        var max_int: i64 = 0;
        var max_frac: i64 = 0;
        var max_exp: i64 = 0;
        i = 0;
        while i < vals.len {
            let x = vals[i];
            if x - x == 0.0 {
                var s: u8[>..] = [];
                var f = no_fault();
                if c.exp_format { s, f = format_float_scientific(x, ff); }
                else { s, f = format_float_positional(x, ff); }
                if f.kind != F_NONE { return c, f; }
                let wi, wf, we = ap_split(s);
                max_int = max_i(max_int, wi);
                max_frac = max_i(max_frac, wf);
                max_exp = max_i(max_exp, we);
            }
            i += 1;
        }
        c.unique = unique;
        c.pad_left = max_int;
        if c.exp_format {
            c.exp_size = max_exp;
            c.trim = 'k';
            c.precision = max_frac;
            c.min_digits = max_frac;
            c.pad_right = c.exp_size + 2 + c.precision;
        } else {
            c.pad_right = max_frac;
            c.exp_size = -1;
            if o.floatmode == 'f' || o.floatmode == 'e' {
                c.precision = max_frac;
                c.min_digits = max_frac;
                c.trim = 'k';
            } else {
                c.precision = prec;
                c.trim = '.';
                c.min_digits = 0;
            }
        }
    }
    if o.sign == ' ' && !anyneg && nfinite > 0 { c.pad_left += 1; }
    if nonfinite > 0 {
        // The missing value and the infinity take the width of the
        // numbers and the width of the point.
        var negw: i64 = 0;
        if o.sign != '-' || neginf { negw = 1; }
        let offset = c.pad_right + 1;
        c.pad_left = max_i(c.pad_left, max_i(3 - offset, 3 + negw - offset));
    }
    return c, no_fault();
}

// ap_elem writes one number in the form of the print.
fn ap_elem(x: f64, c: ApCtx, out: u8[>..]&) -> Fault {
    if x != x || x - x != 0.0 {
        var word: u8[>..] = [];
        if x != x {
            if c.sign == '+' { word.push('+'); }
            word.append("nan");
        } else {
            if x < 0.0 { word.push('-'); }
            else if c.sign == '+' { word.push('+'); }
            word.append("inf");
        }
        push_spaces(out, c.pad_left + c.pad_right + 1 - word.len);
        out.append(word);
        return no_fault();
    }
    var ff = float_format();
    ff.precision = c.precision;
    ff.min_digits = c.min_digits;
    ff.unique = c.unique;
    ff.trim = c.trim;
    ff.sign = c.sign == '+';
    ff.pad_left = c.pad_left;
    var s: u8[>..] = [];
    var f = no_fault();
    if c.exp_format {
        ff.exp_digits = c.exp_size;
        s, f = format_float_scientific(x, ff);
    } else {
        ff.pad_right = c.pad_right;
        s, f = format_float_positional(x, ff);
    }
    if f.kind != F_NONE { return f; }
    out.append(s);
    return no_fault();
}

// ap_rstrip_len gives the length of a text without its trailing spaces.
fn ap_rstrip_len(s: const u8[:]) -> i64 {
    var n = s.len;
    while n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\n' || s[n - 1] == '\r') { n -= 1; }
    return n;
}

// ap_leaf writes the numbers of the last axis. A line that would pass
// the width breaks before the number, as NumPy breaks it. The first line
// has no indent, and the lines after it carry the indent of the level.
fn ap_leaf(a: Arr&, base: i64, c: ApCtx, o: PrintOptions, summary: bool,
           sep: const u8[:], hang: i64, width: i64, out: u8[>..]&) -> Fault {
    let alen = a.shape[a.ndim - 1];
    let lead, trail, dots = ap_axis_plan(alen, summary, o.edgeitems);
    let nslots = lead + trail + (if dots { 1 } else { 0 });
    let sep_len = ap_rstrip_len(sep);
    let elem_width = width - max_i(sep_len, 1);
    var text: u8[>..] = [];
    var words_len: i64 = 0;
    var elem: i64 = 0;
    var slot: i64 = 0;
    while slot < nslots {
        var word: u8[>..] = [];
        if dots && slot == lead {
            word.append("...");
        } else {
            let f = ap_elem(a.data[base + ap_pick(alen, summary, o.edgeitems, elem)], c, word);
            if f.kind != F_NONE { return f; }
            elem += 1;
        }
        // A line that holds only its indent never breaks.
        if words_len > 0 && hang + words_len + word.len > elem_width {
            out.append(text[..ap_rstrip_len(text)]);
            out.push('\n');
            text.clear();
            push_spaces(text, hang);
            words_len = 0;
        }
        text.append(word);
        words_len += word.len;
        if slot < nslots - 1 {
            text.append(sep);
            words_len += sep.len;
        }
        slot += 1;
    }
    out.append(text);
    return no_fault();
}

// ap_level writes one level of the nest of brackets. A level above the
// last axis writes its items one on each line, with a blank line for
// each axis below it.
recursive fn ap_level(a: Arr&, axis: i64, base: i64, c: ApCtx, o: PrintOptions, summary: bool,
                      sep: const u8[:], hang: i64, width: i64, out: u8[>..]&) -> Fault {
    out.push('[');
    let left = a.ndim - axis;
    if left == 1 {
        let f = ap_leaf(a, base, c, o, summary, sep, hang, width, out);
        if f.kind != F_NONE { return f; }
        out.push(']');
        return no_fault();
    }
    let alen = a.shape[axis];
    let lead, trail, dots = ap_axis_plan(alen, summary, o.edgeitems);
    let nslots = lead + trail + (if dots { 1 } else { 0 });
    let stride = ap_stride(a, axis);
    let sep_len = ap_rstrip_len(sep);
    var elem: i64 = 0;
    var slot: i64 = 0;
    while slot < nslots {
        if slot > 0 { push_spaces(out, hang); }
        if dots && slot == lead {
            out.append("...");
        } else {
            let at = base + ap_pick(alen, summary, o.edgeitems, elem) * stride;
            let f = ap_level(a, axis + 1, at, c, o, summary, sep, hang + 1, width - 1, out);
            if f.kind != F_NONE { return f; }
            elem += 1;
        }
        if slot < nslots - 1 {
            out.append(sep[..sep_len]);
            var nl: i64 = 1;
            while nl < left { out.push('\n'); nl += 1; }
        }
        slot += 1;
    }
    out.push(']');
    return no_fault();
}

fn ap_check(o: PrintOptions, tag: const u8[:]) -> Fault {
    if o.precision < 0 { return fault(F_VALUE, tag, "the precision is below zero"); }
    if o.precision > TEXT_LIMIT { return fault(F_LIMIT, tag, "the precision is above the limit of 4096"); }
    if o.threshold < 0 || o.edgeitems < 0 {
        return fault(F_VALUE, tag, "the threshold or the edgeitems is below zero");
    }
    if o.floatmode != 'f' && o.floatmode != 'u' && o.floatmode != 'm' && o.floatmode != 'e' {
        return fault(F_VALUE, tag, "the floatmode is not f, u, m or e");
    }
    if o.sign != '-' && o.sign != '+' && o.sign != ' ' {
        return fault(F_VALUE, tag, "the sign is not a minus, a plus or a space");
    }
    return no_fault();
}

// ap_body writes an array as nested brackets. The separator goes between
// the numbers. The prefix and the suffix lengths set the indent and the
// width of the wrap, as the prefix and the suffix of NumPy do.
fn ap_body(a: Arr&, o: PrintOptions, sep: const u8[:], prefix_len: i64,
           suffix_len: i64, tag: const u8[:]) -> u8[>..], Fault {
    var out: u8[>..] = [];
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fs.kind != F_NONE { return out, fs; }
    let fo = ap_check(o, tag);
    if fo.kind != F_NONE { return out, fo; }
    if sep.len > 64 || prefix_len < 0 || prefix_len > PRINT_MAX_WIDTH
        || suffix_len < 0 || suffix_len > PRINT_MAX_WIDTH {
        return out, fault(F_LIMIT, tag, "the separator, the prefix or the suffix is above its limit");
    }
    let summary = a.data.len > o.threshold;
    var shown: i64 = 1;
    var d: i64 = 0;
    while d < a.ndim {
        shown *= ap_count(a.shape[d], summary, o.edgeitems);
        d += 1;
    }
    if shown > PRINT_MAX_ITEMS {
        return out, fault(F_LIMIT, tag, "the print shows more numbers than the limit");
    }
    var vals: f64[>..] = [];
    ap_gather(a, summary, o.edgeitems, vals);
    let c, fc = ap_fill(vals, o);
    if fc.kind != F_NONE { return out, fc; }
    if shown > PRINT_MAX_BYTES / (c.pad_left + c.pad_right + 2 + sep.len) {
        return out, fault(F_LIMIT, tag, "the text is longer than the limit");
    }
    let f = ap_level(a, 0, 0, c, o, summary, sep, 1 + prefix_len, o.linewidth - suffix_len, out);
    return out, f;
}

// array2string gives the text of an array. The separator goes between
// the numbers. The prefix and the suffix give the indent and the width
// that a text before and after the array takes.
fn array2string(a: Arr&, o: PrintOptions, separator: const u8[:], prefix: const u8[:], suffix: const u8[:]) -> u8[>..], Fault {
    return ap_body(a, o, separator, prefix.len, suffix.len, "array2string");
}

fn array2string(a: Arr&, o: PrintOptions) -> u8[>..], Fault {
    return ap_body(a, o, " ", 0, 0, "array2string");
}

fn array2string(a: Arr&) -> u8[>..], Fault {
    return ap_body(a, print_options(), " ", 0, 0, "array2string");
}

// array_str gives the text of an array as str of NumPy gives it.
fn array_str(a: Arr&, o: PrintOptions) -> u8[>..], Fault {
    return ap_body(a, o, " ", 0, 0, "array_str");
}

fn array_str(a: Arr&) -> u8[>..], Fault {
    return ap_body(a, print_options(), " ", 0, 0, "array_str");
}

// ap_shape_text writes the shape of an array as NumPy writes it.
fn ap_shape_text(a: Arr&, out: u8[>..]&) {
    out.push('(');
    var d: i64 = 0;
    while d < a.ndim {
        if d > 0 { out.append(", "); }
        format_int(out, a.shape[d], 10, 0, '0');
        d += 1;
    }
    if a.ndim == 1 { out.push(','); }
    out.push(')');
}

// array_repr gives the text of an array as repr of NumPy gives it: the
// word array, the numbers, and the shape when the print is a summary.
// The type float64 is the default of NumPy, so it never shows.
fn array_repr(a: Arr&, o: PrintOptions) -> u8[>..], Fault {
    let body, f = ap_body(a, o, ", ", 6, 1, "array_repr");
    var out: u8[>..] = [];
    if f.kind != F_NONE { return out, f; }
    out.append("array(");
    out.append(body);
    if a.data.len <= o.threshold {
        out.push(')');
        return out, no_fault();
    }
    out.push(',');
    var extra: u8[>..] = [];
    extra.append("shape=");
    ap_shape_text(a, extra);
    extra.push(')');
    var last: i64 = out.len;
    while last > 0 && out[last - 1] != '\n' { last -= 1; }
    if out.len - last + extra.len + 1 > o.linewidth {
        out.push('\n');
        push_spaces(out, 6);
    } else {
        out.push(' ');
    }
    out.append(extra);
    return out, no_fault();
}

fn array_repr(a: Arr&) -> u8[>..], Fault {
    return array_repr(a, print_options());
}
