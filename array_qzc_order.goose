// The reorder of the complex QZ algorithm and the calls qz_c and ordqz_c.
//
namespace scientific;

import mathshim;

// cq_swap swaps the two eigenvalues at the places j and j + 1 of the
// triangular pair. The method is the one of the LAPACK routine ztgex2:
// two rotations swap the block, and a test of the residual accepts or
// rejects the swap. The call gives false when it rejects.
fn cq_swap(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, j: i64) -> bool {
    let eps = QZ_EPS;
    let smlnum = QZ_SAFMIN / eps;
    var s: f64[>..] = [];
    var t: f64[>..] = [];
    var i: i64 = 0;
    while i < 2 {
        var k: i64 = 0;
        while k < 2 {
            let va = cq_at(a, n, j + i, j + k);
            let vb = cq_at(b, n, j + i, j + k);
            s.push(va.re); s.push(va.im);
            t.push(vb.re); t.push(vb.im);
            k += 1;
        }
        i += 1;
    }
    let sa = cq_fnorm(s);
    let sb = cq_fnorm(t);
    var tha = 20.0 * eps * sa;
    if tha < smlnum { tha = smlnum; }
    var thb = 20.0 * eps * sb;
    if thb < smlnum { thb = smlnum; }
    let s00 = Cplx { re: s[0], im: s[1] };
    let s01 = Cplx { re: s[2], im: s[3] };
    let s10 = Cplx { re: s[4], im: s[5] };
    let s11 = Cplx { re: s[6], im: s[7] };
    let t00 = Cplx { re: t[0], im: t[1] };
    let t01 = Cplx { re: t[2], im: t[3] };
    let t10 = Cplx { re: t[4], im: t[5] };
    let t11 = Cplx { re: t[6], im: t[7] };
    let f = c_sub(c_mul(s11, t00), c_mul(t11, s00));
    let g = c_sub(c_mul(s11, t01), c_mul(t11, s01));
    let sa0 = qz_hypot(s11.re, s11.im) * qz_hypot(t00.re, t00.im);
    let sb0 = qz_hypot(s00.re, s00.im) * qz_hypot(t11.re, t11.im);
    let cz, sz0, r0 = cq_givens(g, f);
    let sz = c_scale(sz0, -1.0);
    // The rotation of the columns on the local block.
    cq_rot2_cols(s, cz, c_conj(sz));
    cq_rot2_cols(t, cz, c_conj(sz));
    let ns00 = Cplx { re: s[0], im: s[1] };
    let ns10 = Cplx { re: s[4], im: s[5] };
    let nt00 = Cplx { re: t[0], im: t[1] };
    let nt10 = Cplx { re: t[4], im: t[5] };
    var gx = nt00;
    var gy = nt10;
    if sa0 >= sb0 { gx = ns00; gy = ns10; }
    let cq2, sq, rq = cq_givens(gx, gy);
    cq_rot2_rows(s, cq2, sq);
    cq_rot2_rows(t, cq2, sq);
    let ws = cq_abs1(Cplx { re: s[4], im: s[5] });
    let wt = cq_abs1(Cplx { re: t[4], im: t[5] });
    if ws > tha || wt > thb { return false; }
    // The strong test turns the rotations back and measures the
    // residual against the original block.
    var rs: f64[>..] = [];
    var rt: f64[>..] = [];
    i = 0;
    while i < 8 { rs.push(s[i]); rt.push(t[i]); i += 1; }
    cq_rot2_cols(rs, cz, c_scale(c_conj(sz), -1.0));
    cq_rot2_cols(rt, cz, c_scale(c_conj(sz), -1.0));
    cq_rot2_rows(rs, cq2, c_scale(sq, -1.0));
    cq_rot2_rows(rt, cq2, c_scale(sq, -1.0));
    i = 0;
    while i < 2 {
        var k: i64 = 0;
        while k < 2 {
            let va = c_sub(Cplx { re: rs[2 * (2 * i + k)], im: rs[2 * (2 * i + k) + 1] }, cq_at(a, n, j + i, j + k));
            let vb = c_sub(Cplx { re: rt[2 * (2 * i + k)], im: rt[2 * (2 * i + k) + 1] }, cq_at(b, n, j + i, j + k));
            rs[2 * (2 * i + k)] = va.re;
            rs[2 * (2 * i + k) + 1] = va.im;
            rt[2 * (2 * i + k)] = vb.re;
            rt[2 * (2 * i + k) + 1] = vb.im;
            k += 1;
        }
        i += 1;
    }
    if cq_fnorm(rs) > tha || cq_fnorm(rt) > thb { return false; }
    // The swap holds. It goes to the pair. The column rotation covers
    // the rows up to the second row of the block, so the second
    // diagonal entry takes its share of the turn.
    cq_cols(a, n, j, j + 1, cz, c_conj(sz), 0, j + 2);
    cq_cols(b, n, j, j + 1, cz, c_conj(sz), 0, j + 2);
    cq_rows(a, n, j, j + 1, cq2, sq, j, n);
    cq_rows(b, n, j, j + 1, cq2, sq, j, n);
    cq_put(a, n, j + 1, j, Cplx { re: 0.0, im: 0.0 });
    cq_put(b, n, j + 1, j, Cplx { re: 0.0, im: 0.0 });
    cq_cols(z, n, j, j + 1, cz, c_conj(sz), 0, n);
    cq_cols(q, n, j, j + 1, cq2, c_conj(sq), 0, n);
    return true;
}

// cq_rot2_rows turns the two rows of each column of the block of two
// rows held in the buffer d of eight values.
fn cq_rot2_rows(d: f64[>..]&, c: f64, s: Cplx) {
    var k: i64 = 0;
    while k < 2 {
        let xr = d[2 * k];
        let xi = d[2 * k + 1];
        let yr = d[2 * k + 4];
        let yi = d[2 * k + 5];
        d[2 * k] = c * xr + s.re * yr - s.im * yi;
        d[2 * k + 1] = c * xi + s.re * yi + s.im * yr;
        d[2 * k + 4] = -(s.re * xr + s.im * xi) + c * yr;
        d[2 * k + 5] = -(s.re * xi - s.im * xr) + c * yi;
        k += 1;
    }
}

// cq_rot2_cols turns the two columns of each row of the block held in
// the buffer d of eight values.
fn cq_rot2_cols(d: f64[>..]&, c: f64, s: Cplx) {
    var k: i64 = 0;
    while k < 2 {
        let xr = d[4 * k];
        let xi = d[4 * k + 1];
        let yr = d[4 * k + 2];
        let yi = d[4 * k + 3];
        d[4 * k] = c * xr + s.re * yr - s.im * yi;
        d[4 * k + 1] = c * xi + s.re * yi + s.im * yr;
        d[4 * k + 2] = -(s.re * xr + s.im * xi) + c * yr;
        d[4 * k + 3] = -(s.re * xi - s.im * xr) + c * yi;
        k += 1;
    }
}

// cq_fnorm gives the Frobenius norm of the block of two rows held in
// the buffer d of eight values.
fn cq_fnorm(d: f64[:]) -> f64 {
    var ss: f64 = 0.0;
    var i: i64 = 0;
    while i < 8 {
        ss += d[i] * d[i];
        i += 1;
    }
    return mathshim::root(ss);
}

// cq_chosen tells whether a rule chooses the eigenvalue alpha / beta.
// The pair holds the values divided by the scales sa and sb, so a rule
// on the modulus compares the true values. An eigenvalue with beta zero
// holds in no half plane and in no unit circle. It lies outside the
// circle when alpha is not zero.
fn cq_chosen(rule: i64, alpha: Cplx, beta: Cplx, sa: f64, sb: f64) -> bool {
    let al = qz_hypot(alpha.re, alpha.im);
    let be = qz_hypot(beta.re, beta.im);
    if rule == QZ_OUC {
        if be == 0.0 { return al != 0.0; }
        return al * sa > be * sb;
    }
    if be == 0.0 { return false; }
    if rule == QZ_LHP { return alpha.re < 0.0; }
    if rule == QZ_RHP { return alpha.re > 0.0; }
    return al * sa < be * sb;
}

// cq_flags_of sets the flag of each diagonal entry once, from a rule.
fn cq_flags_of(a: f64[:], b: f64[:], n: i64, rule: i64, sa: f64, sb: f64, flags: f64[>..]&) {
    var j: i64 = 0;
    while j < n {
        if cq_chosen(rule, cq_at(a, n, j, j), cq_at(b, n, j, j), sa, sb) {
            flags.push(1.0);
        } else {
            flags.push(0.0);
        }
        j += 1;
    }
}

// cq_flags_by sets the flag of each diagonal entry once. The block of
// the caller chooses the entry. It takes the alpha and the beta of the
// entry and gives true for a chosen entry. The values come back to the
// scale of the caller, so the block reads the true alpha and the true
// beta.
fn cq_flags_by<F: fn(Cplx, f64) -> bool>(a: f64[:], b: f64[:], n: i64, sa: f64, sb: f64, flags: f64[>..]&) {
    var j: i64 = 0;
    while j < n {
        let v = cq_at(a, n, j, j);
        let al = Cplx { re: v.re * sa, im: v.im * sa };
        let be = cq_at(b, n, j, j).re * sb;
        if F(al, be) { flags.push(1.0); } else { flags.push(0.0); }
        j += 1;
    }
}

// cq_partition puts the entries that the flags choose before the
// others. The order inside each group stays as it was. A swap moves the
// flags with the entries. The call gives false when a swap fails.
fn cq_partition(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, flags: f64[>..]&) -> bool {
    var again = true;
    while again {
        again = false;
        var k: i64 = 0;
        while k + 1 < n {
            if flags[k] < 0.5 && flags[k + 1] > 0.5 {
                if !cq_swap(a, b, q, z, n, k) { return false; }
                flags[k] = 1.0;
                flags[k + 1] = 0.0;
                again = true;
            }
            k += 1;
        }
    }
    return true;
}

// cq_eigs appends the values alpha, as real and imaginary parts, and
// the values beta of the diagonal.
fn cq_eigs(a: f64[:], b: f64[:], n: i64, al: f64[>..]&, be: f64[>..]&) {
    var j: i64 = 0;
    while j < n {
        let v = cq_at(a, n, j, j);
        al.push(v.re);
        al.push(v.im);
        be.push(cq_at(b, n, j, j).re);
        j += 1;
    }
}

// cq_stage_iterate copies one complex pair of side n, divides each
// matrix by its own power of two and reduces the pair. It runs the
// iteration. The limit gives the turns that one eigenvalue may take.
// The call gives the two scales and a code: 0 on success, 1
// when the iteration does not converge and 3 when a value overflows.
// The pair starts at the place ao of ad and at the place bo of bd, in
// values.
fn cq_stage_iterate(ad: f64[:], ao: i64, bd: f64[:], bo: i64, n: i64, limit: i64, a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&) -> f64, f64, i64 {
    var i: i64 = 0;
    while i < 2 * n * n { a.push(ad[ao + i]); b.push(bd[bo + i]); i += 1; }
    let sa = qz_scale_of(a, 0, 2 * n * n);
    let sb = qz_scale_of(b, 0, 2 * n * n);
    qz_scale_apply(a, 0, 2 * n * n, sa, false);
    qz_scale_apply(b, 0, 2 * n * n, sb, false);
    let id = cq_identity(n);
    i = 0;
    while i < 2 * n * n { q.push(id[i]); z.push(id[i]); i += 1; }
    cq_reduce(a, b, q, z, n);
    if !cq_qz(a, b, q, z, n, limit) {
        if !(qz_finite(a) && qz_finite(b)) { return sa, sb, 3; }
        return sa, sb, 1;
    }
    return sa, sb, 0;
}

// cq_stage_finish puts the chosen entries first, when ordered is true,
// and appends the results to the outputs. The forms come back to the
// scale of the pair: AA and alpha by sa, BB and beta by sb. The call
// gives 0 on success, 2 when a swap fails and 3 when a value overflows.
fn cq_stage_finish(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, sa: f64, sb: f64, flags: f64[>..]&, ordered: bool, oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64 {
    let so = oa.len;
    let sl = oal.len;
    let sbe = obe.len;
    if ordered {
        if !cq_partition(a, b, q, z, n, flags) { return 2; }
        // A swap can leave a phase on the diagonal of b. The phase of
        // every column goes back to real.
        var j: i64 = 0;
        while j < n {
            cq_std_b(a, b, z, n, j);
            j += 1;
        }
        cq_eigs(a, b, n, oal, obe);
    }
    if !(qz_finite(a) && qz_finite(b) && qz_finite(q) && qz_finite(z)) { return 3; }
    var i: i64 = 0;
    while i < 2 * n * n { oa.push(a[i]); ob.push(b[i]); oq.push(q[i]); oz.push(z[i]); i += 1; }
    qz_scale_apply(oa, so, 2 * n * n, sa, true);
    qz_scale_apply(ob, so, 2 * n * n, sb, true);
    qz_scale_apply(oal, sl, oal.len - sl, sa, true);
    qz_scale_apply(obe, sbe, obe.len - sbe, sb, true);
    if !(qz_finite_from(oa, so) && qz_finite_from(ob, so) && qz_finite_from(oq, so) && qz_finite_from(oz, so)) { return 3; }
    if !(qz_finite_from(oal, sl) && qz_finite_from(obe, sbe)) { return 3; }
    return 0;
}

// cq_pair decomposes one complex pair of side n. A rule other than
// QZ_NONE puts the chosen eigenvalues first and appends alpha and beta.
// The limit gives the turns that one eigenvalue may take. It gives 0 on
// success, 1 when the iteration fails, 2 when a swap fails and 3 when a
// value overflows.
fn cq_pair(ad: f64[:], ao: i64, bd: f64[:], bo: i64, n: i64, rule: i64, limit: i64, oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64 {
    var a: f64[>..] = [];
    var b: f64[>..] = [];
    var q: f64[>..] = [];
    var z: f64[>..] = [];
    let sa, sb, st = cq_stage_iterate(ad, ao, bd, bo, n, limit, a, b, q, z);
    if st != 0 { return st; }
    var flags: f64[>..] = [];
    if rule != QZ_NONE { cq_flags_of(a, b, n, rule, sa, sb, flags); }
    return cq_stage_finish(a, b, q, z, n, sa, sb, flags, rule != QZ_NONE, oa, ob, oq, oz, oal, obe);
}

// cq_pair_by decomposes one complex pair of side n. The block of the
// caller chooses the eigenvalues that come first. The limit gives the
// turns that one eigenvalue may take. It gives 0 on success, 1 when the
// iteration fails, 2 when a swap fails and 3 when a value overflows.
fn cq_pair_by<F: fn(Cplx, f64) -> bool>(ad: f64[:], ao: i64, bd: f64[:], bo: i64, n: i64, limit: i64, oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64 {
    var a: f64[>..] = [];
    var b: f64[>..] = [];
    var q: f64[>..] = [];
    var z: f64[>..] = [];
    let sa, sb, st = cq_stage_iterate(ad, ao, bd, bo, n, limit, a, b, q, z);
    if st != 0 { return st; }
    var flags: f64[>..] = [];
    cq_flags_by(a, b, n, sa, sb, flags) { al, be => F(al, be) };
    return cq_stage_finish(a, b, q, z, n, sa, sb, flags, true, oa, ob, oq, oz, oal, obe);
}

// cq_all runs the decomposition of a complex pair for the whole stack.
// The call gives the batch shape of the result and a fault.
fn cq_all(a: CArr, b: CArr, sort: const u8[:], ordered: bool, tag: const u8[:], oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64[>..], Fault {
    var none: i64[>..] = [];
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    let cb, rule, fc = qz_check(a, b, 2, sort, ordered, tag, ia, ib);
    if fc.kind != F_NONE { return none, fc; }
    let m = a.shape[a.ndim - 1];
    let count = batch_count(cb);
    var s: i64 = 0;
    while s < count {
        let st = cq_pair(a.data, ia[s] * 2 * m * m, b.data, ib[s] * 2 * m * m, m, rule, QZ_TURNS, oa, ob, oq, oz, oal, obe);
        let fs = qz_status(st, tag);
        if fs.kind != F_NONE { return none, fs; }
        s += 1;
    }
    return cb, no_fault();
}

// cq_all_by runs the decomposition of a complex pair for the whole
// stack. The block of the caller chooses the eigenvalues that come
// first. The call gives the batch shape of the result and a fault.
fn cq_all_by<F: fn(Cplx, f64) -> bool>(a: CArr, b: CArr, tag: const u8[:], oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64[>..], Fault {
    var none: i64[>..] = [];
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    let cb, rule, fc = qz_check(a, b, 2, "", false, tag, ia, ib);
    if fc.kind != F_NONE { return none, fc; }
    let m = a.shape[a.ndim - 1];
    let count = batch_count(cb);
    var s: i64 = 0;
    while s < count {
        let st = cq_pair_by(a.data, ia[s] * 2 * m * m, b.data, ib[s] * 2 * m * m, m, QZ_TURNS, oa, ob, oq, oz, oal, obe) { al, be => F(al, be) };
        let fs = qz_status(st, tag);
        if fs.kind != F_NONE { return none, fs; }
        s += 1;
    }
    return cb, no_fault();
}

// qz_c gives the generalized Schur form of a pair of complex matrices.
// The result holds AA, BB, Q and Z, with A = Q AA Z^H and B = Q BB Z^H.
// AA and BB are upper triangular and the diagonal of BB is real and
// not negative. The last two dimensions of each operand give the
// matrix. The dimensions before them give the batch, and the batches
// broadcast.
fn qz_c(a: CArr, b: CArr) -> CArr, CArr, CArr, CArr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = cq_all(copy(a), copy(b), "", false, "qz_c", oa, ob, oq, oz, oal, obe);
    if f.kind != F_NONE { return carr_empty(), carr_empty(), carr_empty(), carr_empty(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    return carr_make(sh, copy(oa)), carr_make(sh, copy(ob)), carr_make(sh, copy(oq)), carr_make(sh, copy(oz)), no_fault();
}

// ordqz_c gives the generalized Schur form of a complex pair with the
// chosen eigenvalues first. The rule is lhp, rhp, iuc or ouc. The
// result holds AA, BB, alpha, beta, Q and Z. The eigenvalues are
// alpha / beta.
fn ordqz_c(a: CArr, b: CArr, sort: const u8[:]) -> CArr, CArr, CArr, Arr, CArr, CArr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = cq_all(copy(a), copy(b), sort, true, "ordqz_c", oa, ob, oq, oz, oal, obe);
    if f.kind != F_NONE { return carr_empty(), carr_empty(), carr_empty(), empty_arr(), carr_empty(), carr_empty(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    let sv = qz_vec_shape(cb, m);
    return carr_make(sh, copy(oa)), carr_make(sh, copy(ob)), carr_make(sv, copy(oal)), arr_of(copy(obe), sv), carr_make(sh, copy(oq)), carr_make(sh, copy(oz)), no_fault();
}

// ordqz_c_by gives the generalized Schur form of a complex pair with
// the eigenvalues that the block of the caller chooses first. The block
// takes the alpha and the beta of a diagonal entry and gives true for a
// chosen entry. The result holds AA, BB, alpha, beta, Q and Z. The
// eigenvalues are alpha / beta.
fn ordqz_c_by<F: fn(Cplx, f64) -> bool>(a: CArr, b: CArr) -> CArr, CArr, CArr, Arr, CArr, CArr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = cq_all_by(copy(a), copy(b), "ordqz_c_by", oa, ob, oq, oz, oal, obe) { al, be => F(al, be) };
    if f.kind != F_NONE { return carr_empty(), carr_empty(), carr_empty(), empty_arr(), carr_empty(), carr_empty(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    let sv = qz_vec_shape(cb, m);
    return carr_make(sh, copy(oa)), carr_make(sh, copy(ob)), carr_make(sv, copy(oal)), arr_of(copy(obe), sv), carr_make(sh, copy(oq)), carr_make(sh, copy(oz)), no_fault();
}
