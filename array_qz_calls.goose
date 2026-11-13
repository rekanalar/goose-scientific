// The stages and the public calls qz and ordqz.
//
namespace scientific;

import mathshim;

// qz_finite tells whether every value of a buffer is finite.
fn qz_finite(d: f64[:]) -> bool { return qz_finite_from(d, 0); }

// qz_finite_from does the same for the values from the place from.
fn qz_finite_from(d: f64[:], from: i64) -> bool {
    var i: i64 = from;
    while i < d.len {
        if !(d[i] - d[i] == 0.0) { return false; }
        i += 1;
    }
    return true;
}

// qz_stage_iterate copies one pair of side n, divides each matrix by
// its own power of two and reduces the pair. It runs the iteration.
// The call gives the two scales and a code: 0 on success, 1 when the
// iteration fails and 3 when a value overflows. The pair starts at the
// place ao of ad and at the place bo of bd.
fn qz_stage_iterate(ad: f64[:], ao: i64, bd: f64[:], bo: i64, n: i64, a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&) -> f64, f64, i64 {
    var i: i64 = 0;
    while i < n * n { a.push(ad[ao + i]); b.push(bd[bo + i]); i += 1; }
    let sa = qz_scale_of(a, 0, n * n);
    let sb = qz_scale_of(b, 0, n * n);
    qz_scale_apply(a, 0, n * n, sa, false);
    qz_scale_apply(b, 0, n * n, sb, false);
    let id = qz_identity(n);
    i = 0;
    while i < n * n { q.push(id[i]); z.push(id[i]); i += 1; }
    qz_reduce(a, b, q, z, n);
    if !qz_iterate(a, b, q, z, n, QZ_TURNS) {
        // A value that overflowed turns every test of the iteration
        // false, so the turns run out. That case is an overflow.
        if !(qz_finite(a) && qz_finite(b)) { return sa, sb, 3; }
        return sa, sb, 1;
    }
    return sa, sb, 0;
}

// qz_stage_finish puts the chosen blocks first, when ordered is true,
// and appends the results to the outputs. The flags hold one value for
// each diagonal block. The forms come back to the scale of the pair:
// AA and alpha by sa, BB and beta by sb. The call gives 0 on success,
// 2 when a swap fails and 3 when a value overflows.
fn qz_stage_finish(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, sa: f64, sb: f64, flags: f64[>..]&, ordered: bool, cmode: bool, oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64 {
    let so = oa.len;
    let sl = oal.len;
    let sbe = obe.len;
    if ordered {
        if !qz_partition(a, b, q, z, n, flags) { return 2; }
        qz_eigs(a, b, n, oal, obe);
    }
    if !(qz_finite(a) && qz_finite(b) && qz_finite(q) && qz_finite(z)) { return 3; }
    if cmode {
        qz_to_complex(a, b, q, z, n, oa, ob, oq, oz);
    } else {
        var i: i64 = 0;
        while i < n * n { oa.push(a[i]); ob.push(b[i]); oq.push(q[i]); oz.push(z[i]); i += 1; }
    }
    qz_scale_apply(oa, so, oa.len - so, sa, true);
    qz_scale_apply(ob, so, ob.len - so, sb, true);
    qz_scale_apply(oal, sl, oal.len - sl, sa, true);
    qz_scale_apply(obe, sbe, obe.len - sbe, sb, true);
    // The outputs hold the complex form and alpha and beta, so the
    // check covers what the call emits and not only the work buffers.
    if !(qz_finite_from(oa, so) && qz_finite_from(ob, so) && qz_finite_from(oq, so) && qz_finite_from(oz, so)) { return 3; }
    if !(qz_finite_from(oal, sl) && qz_finite_from(obe, sbe)) { return 3; }
    return 0;
}

// qz_pair decomposes one pair of side n. A rule other than QZ_NONE puts
// the chosen eigenvalues first and appends alpha and beta. The call
// appends the four forms to the outputs, real or complex. It gives 0
// on success, 1 when the iteration fails, 2 when a swap fails and 3
// when a value overflows.
fn qz_pair(ad: f64[:], ao: i64, bd: f64[:], bo: i64, n: i64, rule: i64, cmode: bool, oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64 {
    var a: f64[>..] = [];
    var b: f64[>..] = [];
    var q: f64[>..] = [];
    var z: f64[>..] = [];
    let sa, sb, st = qz_stage_iterate(ad, ao, bd, bo, n, a, b, q, z);
    if st != 0 { return st; }
    var flags: f64[>..] = [];
    if rule != QZ_NONE { qz_flags_of(a, b, n, rule, sa, sb, flags); }
    return qz_stage_finish(a, b, q, z, n, sa, sb, flags, rule != QZ_NONE, cmode, oa, ob, oq, oz, oal, obe);
}

// qz_pair_by decomposes one pair of side n. The block of the caller
// chooses the eigenvalues that come first. The call appends the four
// forms to the outputs, real or complex. It gives 0 on success, 1 when
// the iteration fails, 2 when a swap fails and 3 when a value overflows.
fn qz_pair_by<F: fn(Cplx, f64) -> bool>(ad: f64[:], ao: i64, bd: f64[:], bo: i64, n: i64, cmode: bool, oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64 {
    var a: f64[>..] = [];
    var b: f64[>..] = [];
    var q: f64[>..] = [];
    var z: f64[>..] = [];
    let sa, sb, st = qz_stage_iterate(ad, ao, bd, bo, n, a, b, q, z);
    if st != 0 { return st; }
    var flags: f64[>..] = [];
    qz_flags_by(a, b, n, sa, sb, flags) { al, be => F(al, be) };
    return qz_stage_finish(a, b, q, z, n, sa, sb, flags, true, cmode, oa, ob, oq, oz, oal, obe);
}

// qz_rule gives the code of a rule of sort, or -1 for a name that no
// rule holds.
fn qz_rule(sort: const u8[:]) -> i64 {
    if sort == "lhp" { return QZ_LHP; }
    if sort == "rhp" { return QZ_RHP; }
    if sort == "iuc" { return QZ_IUC; }
    if sort == "ouc" { return QZ_OUC; }
    return -1;
}

// qz_check runs the checks of a call for the whole stack, in the order
// of the shape checks, the rule and the finite values. The width is 1
// for a real operand and 2 for a complex one. The call gives the batch
// shape of the result, the code of the rule and a fault. The slices of
// the broadcast go to ia and ib.
fn qz_check(a, b, width: i64, sort: const u8[:], ordered: bool, tag: const u8[:], ia: i64[>..]&, ib: i64[>..]&) -> i64[>..], i64, Fault {
    var none: i64[>..] = [];
    let fa = cs_check_stack(a.ndim, a.shape, a.data.len, width, 2, tag);
    if fa.kind != F_NONE { return none, QZ_NONE, fa; }
    let fb = cs_check_stack(b.ndim, b.shape, b.data.len, width, 2, tag);
    if fb.kind != F_NONE { return none, QZ_NONE, fb; }
    let m = a.shape[a.ndim - 1];
    if a.shape[a.ndim - 2] != m || b.shape[b.ndim - 1] != m || b.shape[b.ndim - 2] != m {
        return none, QZ_NONE, fault(F_SHAPE, tag, "the matrices are not square or hold different sides");
    }
    if m < 1 { return none, QZ_NONE, fault(F_SHAPE, tag, "the matrices hold no element"); }
    let ba = shape_batch(a.shape, 2);
    let bb = shape_batch(b.shape, 2);
    let cb, fw = cs_batch_walk(ba, bb, MAX_ELEMENTS / (m * m), tag, ia, ib);
    if fw.kind != F_NONE { return none, QZ_NONE, fw; }
    var rule: i64 = QZ_NONE;
    if ordered {
        rule = qz_rule(sort);
        if rule < 0 { return none, QZ_NONE, fault(F_VALUE, tag, "the rule of sort is not lhp, rhp, iuc or ouc"); }
    }
    let ga = carr_finite(a, tag);
    if ga.kind != F_NONE { return none, QZ_NONE, ga; }
    let gb = carr_finite(b, tag);
    if gb.kind != F_NONE { return none, QZ_NONE, gb; }
    return cb, rule, no_fault();
}

// qz_status turns the code of a pair into a fault.
fn qz_status(st: i64, tag: const u8[:]) -> Fault {
    if st == 1 { return fault(F_LIMIT, tag, "the iteration did not converge"); }
    if st == 2 { return fault(F_VALUE, tag, "the reorder failed"); }
    if st == 3 { return fault(F_VALUE, tag, "a value overflowed"); }
    return no_fault();
}

// qz_all runs the decomposition of a real pair for the whole stack.
// The call gives the batch shape of the result and a fault.
fn qz_all(a, b, sort: const u8[:], ordered: bool, cmode: bool, tag: const u8[:], oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64[>..], Fault {
    var none: i64[>..] = [];
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    let cb, rule, fc = qz_check(a, b, 1, sort, ordered, tag, ia, ib);
    if fc.kind != F_NONE { return none, fc; }
    let m = a.shape[a.ndim - 1];
    let count = batch_count(cb);
    var s: i64 = 0;
    while s < count {
        let st = qz_pair(a.data, ia[s] * m * m, b.data, ib[s] * m * m, m, rule, cmode, oa, ob, oq, oz, oal, obe);
        let fs = qz_status(st, tag);
        if fs.kind != F_NONE { return none, fs; }
        s += 1;
    }
    return cb, no_fault();
}

// qz_all_by runs the decomposition of a real pair for the whole stack.
// The block of the caller chooses the eigenvalues that come first. The
// call gives the batch shape of the result and a fault.
fn qz_all_by<F: fn(Cplx, f64) -> bool>(a, b, cmode: bool, tag: const u8[:], oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64[>..], Fault {
    var none: i64[>..] = [];
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    let cb, rule, fc = qz_check(a, b, 1, "", false, tag, ia, ib);
    if fc.kind != F_NONE { return none, fc; }
    let m = a.shape[a.ndim - 1];
    let count = batch_count(cb);
    var s: i64 = 0;
    while s < count {
        let st = qz_pair_by(a.data, ia[s] * m * m, b.data, ib[s] * m * m, m, cmode, oa, ob, oq, oz, oal, obe) { al, be => F(al, be) };
        let fs = qz_status(st, tag);
        if fs.kind != F_NONE { return none, fs; }
        s += 1;
    }
    return cb, no_fault();
}

// qz_vec_shape gives the shape of the batch followed by the side m.
fn qz_vec_shape(cb: i64[:], m: i64) -> i64[>..] {
    var s: i64[>..] = [];
    var i: i64 = 0;
    while i < cb.len { s.push(cb[i]); i += 1; }
    s.push(m);
    return s;
}

// qz gives the generalized Schur form of a pair of real matrices. The
// result holds AA, BB, Q and Z, with A = Q AA Z^T and B = Q BB Z^T.
// The last two dimensions of each operand give the matrix. The
// dimensions before them give the batch, and the batches broadcast.
fn qz(a, b) -> Arr, Arr, Arr, Arr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = qz_all(a, b, "", false, false, "qz", oa, ob, oq, oz, oal, obe);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    return arr_of(copy(oa), sh), arr_of(copy(ob), sh), arr_of(copy(oq), sh), arr_of(copy(oz), sh), no_fault();
}

// ordqz gives the generalized Schur form with the chosen eigenvalues
// first. The rule is lhp, rhp, iuc or ouc, as in SciPy. The result holds
// AA, BB, alpha, beta, Q and Z. The eigenvalues are alpha / beta.
fn ordqz(a, b, sort: const u8[:]) -> Arr, Arr, CArr, Arr, Arr, Arr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = qz_all(a, b, sort, true, false, "ordqz", oa, ob, oq, oz, oal, obe);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), carr_empty(), empty_arr(), empty_arr(), empty_arr(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    let sv = qz_vec_shape(cb, m);
    return arr_of(copy(oa), sh), arr_of(copy(ob), sh), carr_make(sv, copy(oal)), arr_of(copy(obe), sv), arr_of(copy(oq), sh), arr_of(copy(oz), sh), no_fault();
}

// qz_complex gives the triangular complex form of the generalized
// Schur form of a real pair. The matrices AA and BB are upper
// triangular. The diagonal of BB is real and not negative. The
// products Q AA Z^H and Q BB Z^H equal A and B.
fn qz_complex(a, b) -> CArr, CArr, CArr, CArr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = qz_all(a, b, "", false, true, "qz_complex", oa, ob, oq, oz, oal, obe);
    if f.kind != F_NONE { return carr_empty(), carr_empty(), carr_empty(), carr_empty(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    return carr_make(sh, copy(oa)), carr_make(sh, copy(ob)), carr_make(sh, copy(oq)), carr_make(sh, copy(oz)), no_fault();
}

// ordqz_complex gives the triangular complex form with the chosen
// eigenvalues first. The result holds AA, BB, alpha, beta, Q and Z.
fn ordqz_complex(a, b, sort: const u8[:]) -> CArr, CArr, CArr, Arr, CArr, CArr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = qz_all(a, b, sort, true, true, "ordqz_complex", oa, ob, oq, oz, oal, obe);
    if f.kind != F_NONE { return carr_empty(), carr_empty(), carr_empty(), empty_arr(), carr_empty(), carr_empty(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    let sv = qz_vec_shape(cb, m);
    return carr_make(sh, copy(oa)), carr_make(sh, copy(ob)), carr_make(sv, copy(oal)), arr_of(copy(obe), sv), carr_make(sh, copy(oq)), carr_make(sh, copy(oz)), no_fault();
}

// ordqz_by gives the generalized Schur form with the eigenvalues that
// the block of the caller chooses first. The block takes the alpha and
// the beta of a diagonal block and gives true for a chosen block. The
// result holds AA, BB, alpha, beta, Q and Z. The eigenvalues are
// alpha / beta.
fn ordqz_by<F: fn(Cplx, f64) -> bool>(a, b) -> Arr, Arr, CArr, Arr, Arr, Arr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = qz_all_by(a, b, false, "ordqz_by", oa, ob, oq, oz, oal, obe) { al, be => F(al, be) };
    if f.kind != F_NONE { return empty_arr(), empty_arr(), carr_empty(), empty_arr(), empty_arr(), empty_arr(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    let sv = qz_vec_shape(cb, m);
    return arr_of(copy(oa), sh), arr_of(copy(ob), sh), carr_make(sv, copy(oal)), arr_of(copy(obe), sv), arr_of(copy(oq), sh), arr_of(copy(oz), sh), no_fault();
}

// ordqz_complex_by gives the triangular complex form with the
// eigenvalues that the block of the caller chooses first. The result
// holds AA, BB, alpha, beta, Q and Z.
fn ordqz_complex_by<F: fn(Cplx, f64) -> bool>(a, b) -> CArr, CArr, CArr, Arr, CArr, CArr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = qz_all_by(a, b, true, "ordqz_complex_by", oa, ob, oq, oz, oal, obe) { al, be => F(al, be) };
    if f.kind != F_NONE { return carr_empty(), carr_empty(), carr_empty(), empty_arr(), carr_empty(), carr_empty(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    let sv = qz_vec_shape(cb, m);
    return carr_make(sh, copy(oa)), carr_make(sh, copy(ob)), carr_make(sv, copy(oal)), arr_of(copy(obe), sv), carr_make(sh, copy(oq)), carr_make(sh, copy(oz)), no_fault();
}
