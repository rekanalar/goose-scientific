// The einsum of NumPy with an ellipsis and with many operands.
//
// The rule of letters names the axes of the operands. A letter that
// stands once gives an axis of the result. A letter that stands twice
// in one operand walks the diagonal of that operand. Three dots stand
// for the axes that the rule does not name.
namespace scientific;

// ESet holds the operands of einsum. The dimensions of the operands
// follow each other in one row. The values of the operands follow each
// other in one buffer. The set holds at most eight operands.
struct ESet {
    count: i64,
    nd: i64[..8],
    dm: i64[..64],
    off: i64[..8],
    data: f64[>..]
}

// eset_new gives a set of operands that holds nothing.
fn eset_new() -> ESet {
    return ESet { count: 0, nd: [], dm: [], off: [], data: [] };
}

// eset_add puts one operand into the set. The set grows, so the caller
// keeps the new set in a name of its own. An operand that does not fit
// leaves the set as it was.
fn eset_add(s: ESet&, a) -> ESet {
    if s.count >= 8 || a.ndim > 8 || s.nd.len + a.ndim > 64 { return copy(s); }
    var out = ESet { count: s.count + 1, nd: s.nd, dm: s.dm, off: s.off, data: copy(s.data) };
    var i: i64 = 0;
    while i < a.ndim { out.dm.push(a.shape[i]); i += 1; }
    out.nd.push(a.ndim);
    out.off.push(out.data.len);
    i = 0;
    while i < a.data.len { out.data.push(a.data[i]); i += 1; }
    return out;
}

// ein_label turns a letter into a number of 0 to 51. Another character
// gives -1.
fn ein_label(c: i64) -> i64 {
    if c >= 65 && c <= 90 { return c - 65; }
    if c >= 97 && c <= 122 { return c - 97 + 26; }
    return -1;
}

// ein_find gives the place of a letter in a list of letters, or -1.
fn ein_find(ids: i64[:], l: i64) -> i64 {
    var i: i64 = 0;
    while i < ids.len {
        if ids[i] == l { return i; }
        i += 1;
    }
    return -1;
}

// ein_core sums the products of the operands of a set by a rule of
// letters. The axes of the result are the letters of the rule, or the
// letters that stand once when the rule holds no arrow.
fn ein_core(spec: const u8[:], s: ESet&) -> Arr, Fault {
    var tok: i64[..80] = [];
    var gbeg: i64[..9] = [];
    var gend: i64[..9] = [];
    var outtok: i64[..16] = [];
    var arrow = false;
    var part: i64 = 0;
    gbeg.push(0);
    var i: i64 = 0;
    while i < spec.len {
        let c = spec[i] as i64;
        if c == 32 { i += 1; continue; }
        if c == 44 {
            if arrow {
                return empty_arr(), fault(F_VALUE, "einsum", "the rule holds a comma after the arrow");
            }
            part += 1;
            if part >= 8 {
                return empty_arr(), fault(F_LIMIT, "einsum", "the rule holds more than eight operands");
            }
            gend.push(tok.len);
            gbeg.push(tok.len);
        } else if c == 45 {
            if arrow || i + 1 >= spec.len || spec[i + 1] != 62 {
                return empty_arr(), fault(F_VALUE, "einsum", "the rule holds a broken arrow");
            }
            arrow = true;
            i += 1;
        } else if c == 46 {
            if i + 2 >= spec.len || spec[i + 1] != 46 || spec[i + 2] != 46 {
                return empty_arr(), fault(F_VALUE, "einsum", "the rule holds a broken ellipsis");
            }
            if arrow {
                outtok.push(-1);
            } else {
                if tok.len > gbeg[part] && tok[tok.len - 1] == -1 {
                    return empty_arr(), fault(F_VALUE, "einsum", "the rule holds two ellipses in one operand");
                }
                if tok.len >= 80 {
                    return empty_arr(), fault(F_LIMIT, "einsum", "the rule holds too many letters");
                }
                tok.push(-1);
            }
            i += 2;
        } else {
            let l = ein_label(c);
            if l < 0 {
                return empty_arr(), fault(F_VALUE, "einsum", "the rule holds a character that is no letter");
            }
            if arrow {
                if outtok.len >= 16 {
                    return empty_arr(), fault(F_LIMIT, "einsum", "the rule holds too many letters");
                }
                outtok.push(l);
            } else {
                if tok.len >= 80 {
                    return empty_arr(), fault(F_LIMIT, "einsum", "the rule holds too many letters");
                }
                tok.push(l);
            }
        }
        i += 1;
    }
    gend.push(tok.len);
    let nops = part + 1;
    if nops != s.count {
        return empty_arr(), fault(F_VALUE, "einsum", "the rule holds a different count of operands");
    }
    // The dimensions of every operand and the letter of every axis.
    var opnd: i64[..8] = [];
    var opst: i64[..8] = [];
    var ellc: i64[..8] = [];
    var ellb: i64[..8] = [];
    var oplab: i64[..64] = [];
    var dstart: i64 = 0;
    var k: i64 = 0;
    while k < nops {
        var lb: i64 = 0;
        var nlab: i64 = 0;
        var nel: i64 = 0;
        var p: i64 = gbeg[k];
        while p < gend[k] {
            if tok[p] == -1 { nel = 1; }
            else {
                nlab += 1;
                if nel == 0 { lb += 1; }
            }
            p += 1;
        }
        let nd = s.nd[k];
        if nd > 8 {
            return empty_arr(), fault(F_SHAPE, "einsum", "an operand holds more than eight axes");
        }
        var cel: i64 = 0;
        if nel == 1 {
            cel = nd - nlab;
            if cel < 0 {
                return empty_arr(), fault(F_SHAPE, "einsum", "the letters do not match the dimensions");
            }
        } else if nlab != nd {
            return empty_arr(), fault(F_SHAPE, "einsum", "the letters do not match the dimensions");
        }
        opnd.push(nd);
        opst.push(dstart);
        ellc.push(cel);
        ellb.push(lb);
        var d: i64 = 0;
        while d < nd {
            if d < lb { oplab.push(tok[gbeg[k] + d]); }
            else if d < lb + cel { oplab.push(-1); }
            else { oplab.push(tok[gbeg[k] + lb + 1 + (d - lb - cel)]); }
            d += 1;
        }
        dstart += nd;
        k += 1;
    }
    // The axes of the ellipsis. Every operand puts its axes at the end.
    var rel: i64 = 0;
    k = 0;
    while k < nops {
        if ellc[k] > rel { rel = ellc[k]; }
        k += 1;
    }
    // The letters of the operands and the count of their stands.
    var lids: i64[..24] = [];
    var lsize: i64[..24] = [];
    var lcount: i64[..24] = [];
    k = 0;
    while k < nops {
        var d: i64 = 0;
        while d < opnd[k] {
            let l = oplab[opst[k] + d];
            if l >= 0 {
                var qi = ein_find(lids, l);
                if qi < 0 {
                    lids.push(l);
                    lsize.push(0);
                    lcount.push(0);
                    qi = lids.len - 1;
                }
                lcount[qi] += 1;
            }
            d += 1;
        }
        k += 1;
    }
    let nl = lids.len;
    // The length of every letter. A length of one grows to the length
    // of the other operands.
    k = 0;
    while k < nops {
        var d: i64 = 0;
        while d < opnd[k] {
            let l = oplab[opst[k] + d];
            if l >= 0 {
                let sz = s.dm[opst[k] + d];
                var e: i64 = 0;
                while e < d {
                    if oplab[opst[k] + e] == l && s.dm[opst[k] + e] != sz {
                        return empty_arr(), fault(F_SHAPE, "einsum", "a letter holds two different lengths");
                    }
                    e += 1;
                }
                let qi = ein_find(lids, l);
                if lsize[qi] == 0 { lsize[qi] = sz; }
                else if lsize[qi] != sz {
                    if lsize[qi] == 1 { lsize[qi] = sz; }
                    else if sz != 1 {
                        return empty_arr(), fault(F_SHAPE, "einsum", "a letter holds two different lengths");
                    }
                }
            }
            d += 1;
        }
        k += 1;
    }
    // The length of every axis of the ellipsis.
    var esize: i64[..8] = [];
    var q: i64 = 0;
    while q < rel { esize.push(0); q += 1; }
    k = 0;
    while k < nops {
        let cel = ellc[k];
        if cel > 0 {
            let base = rel - cel;
            var j: i64 = 0;
            while j < cel {
                let sz = s.dm[opst[k] + ellb[k] + j];
                if esize[base + j] == 0 { esize[base + j] = sz; }
                else if esize[base + j] != sz {
                    if esize[base + j] == 1 { esize[base + j] = sz; }
                    else if sz != 1 {
                        return empty_arr(), fault(F_SHAPE, "einsum", "an axis of the ellipsis holds two different lengths");
                    }
                }
                j += 1;
            }
        }
        k += 1;
    }
    // The axis of every dimension of every operand.
    var opax: i64[..64] = [];
    var opbc: i64[..64] = [];
    var opstride: i64[..64] = [];
    var gsize: i64[..32] = [];
    q = 0;
    while q < nl { gsize.push(lsize[q]); q += 1; }
    q = 0;
    while q < rel { gsize.push(esize[q]); q += 1; }
    let naxes = gsize.len;
    k = 0;
    while k < nops {
        var d: i64 = 0;
        while d < opnd[k] {
            let l = oplab[opst[k] + d];
            var ax: i64 = 0;
            if l < 0 {
                ax = nl + rel - ellc[k] + (d - ellb[k]);
            } else {
                ax = ein_find(lids, l);
            }
            opax.push(ax);
            let sz = s.dm[opst[k] + d];
            if sz == 1 && gsize[ax] != 1 { opbc.push(1); } else { opbc.push(0); }
            d += 1;
        }
        // The steps of the axes of the operand.
        var st: i64[..8] = [];
        var acc: i64 = 1;
        d = opnd[k] - 1;
        while d >= 0 { st.push(acc); acc = acc * s.dm[opst[k] + d]; d -= 1; }
        d = 0;
        while d < opnd[k] { opstride.push(st[opnd[k] - 1 - d]); d += 1; }
        k += 1;
    }
    // The axes of the result.
    var oax: i64[..16] = [];
    if arrow {
        q = 0;
        while q < outtok.len {
            let l = outtok[q];
            if l < 0 {
                var j: i64 = 0;
                while j < rel { oax.push(nl + j); j += 1; }
            } else {
                let at = ein_find(lids, l);
                if at < 0 {
                    return empty_arr(), fault(F_VALUE, "einsum", "a letter of the result stands in no operand");
                }
                var seen = false;
                var r: i64 = 0;
                while r < oax.len {
                    if oax[r] == at { seen = true; }
                    r += 1;
                }
                if seen {
                    return empty_arr(), fault(F_VALUE, "einsum", "a letter of the result repeats");
                }
                oax.push(at);
            }
            q += 1;
        }
    } else {
        q = 0;
        while q < rel { oax.push(nl + q); q += 1; }
        var l: i64 = 0;
        while l < 52 {
            let at = ein_find(lids, l);
            if at >= 0 && lcount[at] == 1 { oax.push(at); }
            l += 1;
        }
    }
    let nout = oax.len;
    // The steps of the axes of the result.
    var ostr: i64[..16] = [];
    q = 0;
    while q < nout { ostr.push(0); q += 1; }
    var acc: i64 = 1;
    q = nout - 1;
    while q >= 0 {
        ostr[q] = acc;
        acc = acc * gsize[oax[q]];
        q -= 1;
    }
    let ototal = acc;
    // The loop over every axis.
    var total: i64 = 1;
    q = 0;
    while q < naxes {
        if gsize[q] > 0 && total > MAX_ELEMENTS / gsize[q] {
            return empty_arr(), fault(F_LIMIT, "einsum", "the loop holds more steps than the limit");
        }
        total = total * gsize[q];
        q += 1;
    }
    var d: f64[>..] = [];
    q = 0;
    while q < ototal { d.push(0.0); q += 1; }
    var pos: i64[..32] = [];
    q = 0;
    while q < naxes { pos.push(0); q += 1; }
    var step: i64 = 0;
    while step < total {
        var prod: f64 = 1.0;
        k = 0;
        while k < nops {
            var o: i64 = 0;
            var d2: i64 = 0;
            while d2 < opnd[k] {
                if opbc[opst[k] + d2] == 0 {
                    o = o + pos[opax[opst[k] + d2]] * opstride[opst[k] + d2];
                }
                d2 += 1;
            }
            prod = prod * s.data[s.off[k] + o];
            k += 1;
        }
        var io: i64 = 0;
        q = 0;
        while q < nout {
            io = io + pos[oax[q]] * ostr[q];
            q += 1;
        }
        d[io] = d[io] + prod;
        q = naxes - 1;
        while q >= 0 {
            pos[q] += 1;
            if pos[q] < gsize[q] { break; }
            pos[q] = 0;
            q -= 1;
        }
        step += 1;
    }
    var shape: i64[..8] = [];
    q = 0;
    while q < nout { shape.push(gsize[oax[q]]); q += 1; }
    return Arr { ndim: shape.len, shape: shape, data: d }, no_fault();
}

// einsum sums the products of two arrays by a rule of letters, such as
// "ij,jk->ik". The rule may hold an ellipsis.
fn einsum(spec: const u8[:], a, b) -> Arr, Fault {
    var s = eset_new();
    var s1 = eset_add(s, a);
    var s2 = eset_add(s1, b);
    return ein_core(spec, s2);
}

// einsum1 applies a rule of letters to one array, such as "ii" or
// "ij->ji".
fn einsum1(spec: const u8[:], a) -> Arr, Fault {
    var s = eset_new();
    var s1 = eset_add(s, a);
    return ein_core(spec, s1);
}

// einsum3 sums the products of three arrays by a rule of letters, such
// as "ij,jk,kl->il".
fn einsum3(spec: const u8[:], a, b, c) -> Arr, Fault {
    var s = eset_new();
    var s1 = eset_add(s, a);
    var s2 = eset_add(s1, b);
    var s3 = eset_add(s2, c);
    return ein_core(spec, s3);
}

// einsum_set sums the products of the operands of a set by a rule of
// letters. The set holds any count of operands that einsum_set takes.
fn einsum_set(spec: const u8[:], s: ESet&) -> Arr, Fault {
    return ein_core(spec, s);
}
