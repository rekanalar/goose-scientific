// The path of an einsum: einsum_path gives the order in which the operands
// meet and the count of the operations, as NumPy does.
//
// The mode "greedy" is the default of NumPy. The mode "optimal" tries
// every order of the contractions and keeps the one that costs least. The
// mode "none" gives the one step that takes every operand. The count is
// the "Optimized FLOP count" of NumPy. The rule holds letters only, with
// no ellipsis.
namespace scientific;

// ep_size gives the product of the lengths of the letters of a set.
fn ep_size(mask: i64, dsz: f64[:]) -> f64 {
    var p: f64 = 1.0;
    var l: i64 = 0;
    while l < 52 {
        if ((mask >> l) & 1) != 0 { p = p * dsz[l]; }
        l += 1;
    }
    return p;
}

// ep_flops gives the count of operations of one step.
fn ep_flops(contract: i64, inner: bool, terms: i64, dsz: f64[:]) -> f64 {
    var factor = terms - 1;
    if factor < 1 { factor = 1; }
    if inner { factor += 1; }
    return ep_size(contract, dsz) * (factor as f64);
}

// ep_step contracts the operands that the bits of pos name. It gives the
// letters of the step, the letters of its result, and the letters that it
// sums away.
fn ep_step(sets: i64[:], cnt: i64, pos: i64, outm: i64) -> i64, i64, i64 {
    var contract: i64 = 0;
    var remain = outm;
    var i: i64 = 0;
    while i < cnt {
        if ((pos >> i) & 1) != 0 {
            contract = contract + sets[i] - (contract & sets[i]);
        } else {
            remain = remain + sets[i] - (remain & sets[i]);
        }
        i += 1;
    }
    let res = remain & contract;
    return contract, res, contract - res;
}

// ep_compact moves the operands that the step did not use to the front, and
// puts the result of the step after them. The count of the operands comes
// back.
fn ep_compact(sets: i64[..8]&, cnt: i64, used: i64, res: i64) -> i64 {
    var w: i64 = 0;
    var i: i64 = 0;
    while i < cnt {
        if ((used >> i) & 1) == 0 {
            sets[w] = sets[i];
            w += 1;
        }
        i += 1;
    }
    sets[w] = res;
    return w + 1;
}

// ep_remap gives the place of an operand after a step that took the
// places bx and by. Both tests use the place before the step.
fn ep_remap(v: i64, bx: i64, by: i64) -> i64 {
    var d: i64 = 0;
    if v > bx { d += 1; }
    if v > by { d += 1; }
    return v - d;
}

// ep_score scores one pair of operands against the sieves of NumPy. A
// pair whose result breaks the size limit, or whose cost breaks the cost
// of the plain order, stays out of the pool. The call gives the new
// count of the pairs in the pool.
fn ep_score(cur: i64[:], cnt: i64, x: i64, y: i64, outm: i64, dsz: f64[:],
            limit: f64, naive: f64, spent: f64,
            rem: f64[>..]&, cost: f64[>..]&, px: i64[>..]&, py: i64[>..]&,
            nknown: i64) -> i64 {
    let sc, sr, sm = ep_step(cur, cnt, (1 << x) + (1 << y), outm);
    let nsz = ep_size(sr, dsz);
    if nsz > limit { return nknown; }
    let removed = ep_size(cur[x], dsz) + ep_size(cur[y], dsz) - nsz;
    let c = ep_flops(sc, sm != 0, 2, dsz);
    if spent + c > naive { return nknown; }
    rem.push(removed);
    cost.push(c);
    px.push(x);
    py.push(y);
    return nknown + 1;
}

// ep_greedy fills the path with the greedy rule of NumPy. The rule keeps
// the scores of the pairs of one round while the pair still stands, so a
// pair can win a later round with the score of an earlier one. The path
// holds the places of the operands of every step in pl, and the count of
// the operands of every step in plen.
fn ep_greedy(sets: i64[:], nsets: i64, outm: i64, dsz: f64[:], limit: f64, pl: i64[>..]&, plen: i64[>..]&) {
    // The cost of the contraction of every operand at once. NumPy uses
    // that cost to reject a step that costs more.
    let wc, _, wm = ep_step(sets, nsets, (1 << nsets) - 1, outm);
    let naive = ep_flops(wc, wm != 0, nsets, dsz);
    // The pool of scored pairs: the letters that a pair removes, its
    // count of operations, and the places of its two operands.
    var rem: f64[>..] = [];
    var cost: f64[>..] = [];
    var px: i64[>..] = [];
    var py: i64[>..] = [];
    var cur: i64[..8] = [];
    var k: i64 = 0;
    while k < nsets { cur.push(sets[k]); k += 1; }
    var cnt = nsets;
    var spent: f64 = 0.0;
    var it: i64 = 0;
    while it < nsets - 1 {
        // Score the new pairs and put them in the pool. The first round
        // takes every pair. A later round takes the pairs that hold the
        // newest operand, for the old pairs keep their scores.
        var nrem: f64[>..] = [];
        var ncost: f64[>..] = [];
        var npx: i64[>..] = [];
        var npy: i64[>..] = [];
        var nknown: i64 = 0;
        var i: i64 = 0;
        if it == 0 {
            while i < cnt {
                var j = i + 1;
                while j < cnt {
                    if (cur[i] & cur[j]) != 0 {
                        nknown = ep_score(cur, cnt, i, j, outm, dsz, limit,
                                          naive, spent, nrem, ncost, npx, npy, nknown);
                    }
                    j += 1;
                }
                i += 1;
            }
        } else {
            let newp = cnt - 1;
            while i < newp {
                if (cur[i] & cur[newp]) != 0 {
                    nknown = ep_score(cur, cnt, i, newp, outm, dsz, limit,
                                      naive, spent, nrem, ncost, npx, npy, nknown);
                }
                i += 1;
            }
        }
        if rem.len == 0 && nknown == 0 {
            // No pair with a shared letter fit. Every pair counts now,
            // the outer pairs included.
            i = 0;
            while i < cnt {
                var j = i + 1;
                while j < cnt {
                    nknown = ep_score(cur, cnt, i, j, outm, dsz, limit,
                                      naive, spent, nrem, ncost, npx, npy, nknown);
                    j += 1;
                }
                i += 1;
            }
            if nknown == 0 {
                // No pair fit at all. One step takes every operand.
                var t: i64 = 0;
                while t < cnt { pl.push(t); t += 1; }
                plen.push(cnt);
                return;
            }
        }
        // The old pairs come first in the pool, the new pairs follow.
        var brem: f64[>..] = [];
        var bcost: f64[>..] = [];
        var bpx: i64[>..] = [];
        var bpy: i64[>..] = [];
        i = 0;
        while i < rem.len {
            brem.push(rem[i]);
            bcost.push(cost[i]);
            bpx.push(px[i]);
            bpy.push(py[i]);
            i += 1;
        }
        i = 0;
        while i < nknown {
            brem.push(nrem[i]);
            bcost.push(ncost[i]);
            bpx.push(npx[i]);
            bpy.push(npy[i]);
            i += 1;
        }
        // The pair that removes the most letters wins. The smaller count
        // of operations wins a tie, and the first pair wins a second tie.
        var best: i64 = 0;
        i = 1;
        while i < brem.len {
            if brem[i] > brem[best] || (brem[i] == brem[best] && bcost[i] < bcost[best]) {
                best = i;
            }
            i += 1;
        }
        pl.push(bpx[best]);
        pl.push(bpy[best]);
        plen.push(2);
        spent += bcost[best];
        let bx = bpx[best];
        let by = bpy[best];
        let _, sr, _ = ep_step(cur, cnt, (1 << bx) + (1 << by), outm);
        cnt = ep_compact(cur, cnt, (1 << bx) + (1 << by), sr);
        // The pairs that the step did not touch keep their scores. Their
        // places move, because the result of the step took one slot.
        var krem: f64[>..] = [];
        var kcost: f64[>..] = [];
        var kpx: i64[>..] = [];
        var kpy: i64[>..] = [];
        i = 0;
        while i < brem.len {
            if i != best && bpx[i] != bx && bpx[i] != by && bpy[i] != bx && bpy[i] != by {
                krem.push(brem[i]);
                kcost.push(bcost[i]);
                kpx.push(ep_remap(bpx[i], bx, by));
                kpy.push(ep_remap(bpy[i], bx, by));
            }
            i += 1;
        }
        rem = copy(krem);
        cost = copy(kcost);
        px = copy(kpx);
        py = copy(kpy);
        it += 1;
    }
}

// ep_optimal fills the path with the order that costs least. It tries
// every pair of the operands at every step and keeps the cheapest order.
// The work grows fast, so the call holds at most six operands. It gives
// false when it does not run, and the caller uses the greedy rule then.
fn ep_optimal(sets: i64[:], nsets: i64, outm: i64, dsz: f64[:], limit: f64, pl: i64[>..]&, plen: i64[>..]&) -> bool {
    if nsets > 6 { return false; }
    // The cost, the places and the operands of every order that the
    // search holds. Every order of a step holds the same count of places
    // and the same count of operands, so the counts give the places.
    var cost: f64[>..] = [0.0];
    var pos: i64[>..] = [];
    var rm: i64[>..] = [];
    var i: i64 = 0;
    while i < nsets { rm.push(sets[i]); i += 1; }
    var level: i64 = 0;
    while level < nsets - 1 {
        let rmlen = nsets - level;
        var ncost: f64[>..] = [];
        var npos: i64[>..] = [];
        var nrm: i64[>..] = [];
        var p: i64 = 0;
        while p < cost.len {
            let pc = cost[p];
            let posbase = p * 2 * level;
            let rmbase = p * rmlen;
            var i: i64 = 0;
            while i < rmlen {
                var j = i + 1;
                while j < rmlen {
                    var contract = rm[rmbase + i];
                    contract = contract + rm[rmbase + j] - (contract & rm[rmbase + j]);
                    var remain = outm;
                    var k: i64 = 0;
                    while k < rmlen {
                        if k != i && k != j {
                            remain = remain + rm[rmbase + k] - (remain & rm[rmbase + k]);
                        }
                        k += 1;
                    }
                    let res = remain & contract;
                    if ep_size(res, dsz) <= limit {
                        let step = ep_flops(contract, (contract - res) != 0, 2, dsz);
                        ncost.push(pc + step);
                        var t: i64 = 0;
                        while t < 2 * level { npos.push(pos[posbase + t]); t += 1; }
                        npos.push(i);
                        npos.push(j);
                        k = 0;
                        while k < rmlen {
                            if k != i && k != j { nrm.push(rm[rmbase + k]); }
                            k += 1;
                        }
                        nrm.push(res);
                    }
                    j += 1;
                }
                i += 1;
            }
            p += 1;
        }
        if ncost.len == 0 {
            // No step fits. The order with the least cost takes every
            // operand that is left in one step.
            let best = ep_best(cost);
            var t: i64 = 0;
            while t < 2 * level { pl.push(pos[best * 2 * level + t]); t += 1; }
            t = 0;
            while t < rmlen { pl.push(t); t += 1; }
            var s: i64 = 0;
            while s < level { plen.push(2); s += 1; }
            plen.push(rmlen);
            return true;
        }
        cost = copy(ncost);
        pos = copy(npos);
        rm = copy(nrm);
        level += 1;
    }
    let best = ep_best(cost);
    var t: i64 = 0;
    while t < 2 * level { pl.push(pos[best * 2 * level + t]); t += 1; }
    var s: i64 = 0;
    while s < level { plen.push(2); s += 1; }
    return true;
}

// ep_best gives the place of the order that costs least. The first order
// wins a tie.
fn ep_best(cost: f64[:]) -> i64 {
    var best: i64 = 0;
    var i: i64 = 1;
    while i < cost.len {
        if cost[i] < cost[best] { best = i; }
        i += 1;
    }
    return best;
}

// ep_core parses the rule, checks the operands, finds the path and
// counts its operations.
fn ep_core(spec: const u8[:], mode: const u8[:], s: ESet&) -> Arr, f64, Fault {
    // The set of the operands must agree with the counts of its
    // dimensions. A set that a caller built by hand and broke does not
    // pass the check.
    if s.count < 1 || s.count > 8 || s.nd.len != s.count || s.off.len != s.count {
        return empty_arr(), 0.0, fault(F_SHAPE, "einsum_path", "the set of the operands does not agree with its count");
    }
    var total: i64 = 0;
    var c: i64 = 0;
    while c < s.count {
        if s.nd[c] < 0 { return empty_arr(), 0.0, fault(F_SHAPE, "einsum_path", "the set of the operands holds a negative count of dimensions"); }
        total += s.nd[c];
        c += 1;
    }
    if total > s.dm.len {
        return empty_arr(), 0.0, fault(F_SHAPE, "einsum_path", "the set of the operands holds more dimensions than its buffer");
    }
    if mode != "greedy" && mode != "optimal" && mode != "none" {
        return empty_arr(), 0.0, fault(F_VALUE, "einsum_path", "the mode is not greedy, optimal or none");
    }
    var tok: i64[..80] = [];
    var gend: i64[..9] = [];
    var outtok: i64[..52] = [];
    var arrow = false;
    var i: i64 = 0;
    while i < spec.len {
        let c = spec[i] as i64;
        if c == 32 { i += 1; continue; }
        if c == 44 {
            if arrow { return empty_arr(), 0.0, fault(F_VALUE, "einsum_path", "the rule holds a comma after the arrow"); }
            if gend.len >= 8 { return empty_arr(), 0.0, fault(F_LIMIT, "einsum_path", "the rule holds more than eight operands"); }
            gend.push(tok.len);
        } else if c == 45 {
            if arrow || i + 1 >= spec.len || spec[i + 1] != 62 {
                return empty_arr(), 0.0, fault(F_VALUE, "einsum_path", "the rule holds a broken arrow");
            }
            arrow = true;
            i += 1;
        } else {
            let l = ein_label(c);
            if l < 0 {
                return empty_arr(), 0.0, fault(F_VALUE, "einsum_path", "the rule holds a character that is no letter");
            }
            if arrow {
                if outtok.len >= 52 { return empty_arr(), 0.0, fault(F_LIMIT, "einsum_path", "the rule holds too many letters"); }
                outtok.push(l);
            } else {
                if tok.len >= 80 { return empty_arr(), 0.0, fault(F_LIMIT, "einsum_path", "the rule holds too many letters"); }
                tok.push(l);
            }
        }
        i += 1;
    }
    gend.push(tok.len);
    let n = gend.len;
    // The sets of the operands hold at most eight places.
    if n > 8 {
        return empty_arr(), 0.0, fault(F_LIMIT, "einsum_path", "the rule holds more than eight operands");
    }
    if n != s.count {
        return empty_arr(), 0.0, fault(F_VALUE, "einsum_path", "the rule holds a different count of operands");
    }
    // The sets of letters, the length of every letter, the count of stands.
    var sets: i64[..8] = [];
    var dsz: f64[..52] = [];
    var seen: i64[..52] = [];
    var q: i64 = 0;
    while q < 52 { dsz.push(0.0); seen.push(0); q += 1; }
    var all: i64 = 0;
    var dstart: i64 = 0;
    var from: i64 = 0;
    var k: i64 = 0;
    while k < n {
        if gend[k] - from != s.nd[k] {
            return empty_arr(), 0.0, fault(F_SHAPE, "einsum_path", "the letters do not match the dimensions");
        }
        var m: i64 = 0;
        var d: i64 = 0;
        while d < s.nd[k] {
            let l = tok[from + d];
            let len = s.dm[dstart + d] as f64;
            if ((m >> l) & 1) == 0 { m = m + (1 << l); }
            if ((all >> l) & 1) == 0 { all = all + (1 << l); }
            seen[l] += 1;
            if dsz[l] == 0.0 || dsz[l] == 1.0 { dsz[l] = len; }
            else if len != 1.0 && len != dsz[l] {
                return empty_arr(), 0.0, fault(F_SHAPE, "einsum_path", "a letter holds two different lengths");
            }
            d += 1;
        }
        sets.push(m);
        dstart += s.nd[k];
        from = gend[k];
        k += 1;
    }
    // The letters of the result.
    var outm: i64 = 0;
    if arrow {
        q = 0;
        while q < outtok.len {
            let l = outtok[q];
            if ((all >> l) & 1) == 0 {
                return empty_arr(), 0.0, fault(F_VALUE, "einsum_path", "a letter of the result stands in no operand");
            }
            if ((outm >> l) & 1) != 0 {
                return empty_arr(), 0.0, fault(F_VALUE, "einsum_path", "a letter of the result repeats");
            }
            outm = outm + (1 << l);
            q += 1;
        }
    } else {
        q = 0;
        while q < 52 {
            if seen[q] == 1 { outm = outm + (1 << q); }
            q += 1;
        }
    }
    // The largest operand or the result bounds an intermediate.
    var limit = ep_size(outm, dsz);
    k = 0;
    while k < n {
        let z = ep_size(sets[k], dsz);
        if z > limit { limit = z; }
        k += 1;
    }
    var pl: i64[>..] = [];
    var plen: i64[>..] = [];
    if mode == "none" || n <= 2 || all == outm {
        k = 0;
        while k < n { pl.push(k); k += 1; }
        plen.push(n);
    } else if mode == "optimal" {
        if !ep_optimal(sets, n, outm, dsz, limit, pl, plen) {
            ep_greedy(sets, n, outm, dsz, limit, pl, plen);
        }
    } else {
        ep_greedy(sets, n, outm, dsz, limit, pl, plen);
    }
    // The count of the operations of the path.
    var cur: i64[..8] = [];
    k = 0;
    while k < n { cur.push(sets[k]); k += 1; }
    var cnt = n;
    var total: f64 = 1.0;
    var at: i64 = 0;
    var width: i64 = 0;
    var st: i64 = 0;
    while st < plen.len {
        var bits: i64 = 0;
        var t: i64 = 0;
        while t < plen[st] { bits = bits + (1 << pl[at + t]); t += 1; }
        let sc, sr, sm = ep_step(cur, cnt, bits, outm);
        total = total + ep_flops(sc, sm != 0, plen[st], dsz);
        cnt = ep_compact(cur, cnt, bits, sr);
        if plen[st] > width { width = plen[st]; }
        at += plen[st];
        st += 1;
    }
    var d: f64[>..] = [];
    at = 0;
    st = 0;
    while st < plen.len {
        var t: i64 = 0;
        while t < width {
            if t < plen[st] { d.push(pl[at + t] as f64); } else { d.push(-1.0); }
            t += 1;
        }
        at += plen[st];
        st += 1;
    }
    var shape: i64[..8] = [];
    shape.push(plen.len);
    shape.push(width);
    return Arr { ndim: 2, shape: shape, data: d }, total, no_fault();
}

// einsum_path gives the path of an einsum of two arrays and the count of
// its operations. The path holds one row per step, with the places of the
// operands of the step. A shorter step ends in -1. NumPy gives a list of
// tuples and a text.
fn einsum_path(spec: const u8[:], a, b, mode: const u8[:] = "greedy") -> Arr, f64, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, "einsum_path");
    if fa.kind != F_NONE { return empty_arr(), 0.0, fa; }
    let fb = check_stack(b.ndim, b.shape, b.data.len, 1, "einsum_path");
    if fb.kind != F_NONE { return empty_arr(), 0.0, fb; }
    var s = eset_new();
    var s1 = eset_add(s, a);
    var s2 = eset_add(s1, b);
    return ep_core(spec, mode, s2);
}

// einsum_path1 gives the path of an einsum of one array.
fn einsum_path1(spec: const u8[:], a, mode: const u8[:] = "greedy") -> Arr, f64, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, "einsum_path");
    if fa.kind != F_NONE { return empty_arr(), 0.0, fa; }
    var s = eset_new();
    var s1 = eset_add(s, a);
    return ep_core(spec, mode, s1);
}

// einsum_path3 gives the path of an einsum of three arrays.
fn einsum_path3(spec: const u8[:], a, b, c, mode: const u8[:] = "greedy") -> Arr, f64, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, "einsum_path");
    if fa.kind != F_NONE { return empty_arr(), 0.0, fa; }
    let fb = check_stack(b.ndim, b.shape, b.data.len, 1, "einsum_path");
    if fb.kind != F_NONE { return empty_arr(), 0.0, fb; }
    let fc = check_stack(c.ndim, c.shape, c.data.len, 1, "einsum_path");
    if fc.kind != F_NONE { return empty_arr(), 0.0, fc; }
    var s = eset_new();
    var s1 = eset_add(s, a);
    var s2 = eset_add(s1, b);
    var s3 = eset_add(s2, c);
    return ep_core(spec, mode, s3);
}

// einsum_path_set gives the path of an einsum of the operands of a set.
fn einsum_path_set(spec: const u8[:], s: ESet&, mode: const u8[:] = "greedy") -> Arr, f64, Fault {
    return ep_core(spec, mode, s);
}
