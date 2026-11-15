// The incomplete factorization of a sparse matrix.
//
// spilu factors a square matrix with the rule of the dropped entries.
// The elimination runs in the natural order of the rows and holds no
// pivot. Both permutations of the factor are the identity. A pivot
// of zero is a fault of kind F_VALUE, and a matrix that needs a row
// exchange gives that fault. The complete factorization of the package
// pivots, so a caller with such a matrix uses sparse_lu.
//
// The rule drops the fill-in of the factor below the threshold of the
// drop. The threshold of a row is drop_tol times the absolute value
// of its diagonal. The entries that the matrix itself holds and the
// diagonal of the factor never drop. With a drop_tol of zero the rule
// drops nothing. The factor is then the complete factorization in the
// natural order. The note of SciPy states the same reach for its rule.
// The drop tolerance touches the entries that the elimination generates.
//
// The count of the stored values of the factor has a quota. The quota
// is fill_factor times half of the stored count of the matrix, plus
// two sides. A factor above that count is a fault of kind F_LIMIT.
//
// The order of the factors of SuperLU differs from the order of the
// package, because SuperLU orders the columns and pivots with a
// threshold. The solve of a matrix whose factorization drops nothing
// agrees with SciPy 1.18.1. The agreement holds to a relative
// tolerance of 1e-8, and the tests measure it.
//
// The factors hold the type Splu of the complete factorization, so
// splu_solve solves a right side with them.
namespace scientific;

import mathshim;

// ilu_in_pattern tells whether the original matrix holds the place of
// the row r and the column c. The columns of the rows of t rise, so a
// binary search finds the place.
fn ilu_in_pattern(t, r: i64, c: i64) -> bool {
    var lo = csr_ptr(t, r);
    var hi = csr_ptr(t, r + 1) - 1;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let v = csr_col(t, mid);
        if v == c { return true; }
        if v < c { lo = mid + 1; } else { hi = mid - 1; }
    }
    return false;
}

// spilu gives the incomplete factorization of a square sparse matrix.
// The rule of the drop is basic: the factor loses the fill-in below
// the threshold. A matrix of a forged layout gives a fault before any
// value is read.
fn spilu(s, drop_tol: f64 = 1.0e-4, fill_factor: f64 = 10.0, drop_rule: const u8[:] = "basic") -> Splu, Fault {
    if drop_rule != "basic" {
        return splu_empty(), fault(F_VALUE, "spilu", "the drop rule is not basic");
    }
    if mathshim::is_finite_of(drop_tol) == 0.0 || drop_tol < 0.0 || drop_tol > 1.0 {
        return splu_empty(), fault(F_VALUE, "spilu", "the drop tolerance stands outside zero to one");
    }
    if mathshim::is_finite_of(fill_factor) == 0.0 || fill_factor < 1.0 {
        return splu_empty(), fault(F_VALUE, "spilu", "the fill factor stands below one");
    }
    let bc = bmat_check_block(s, "spilu");
    if bc.kind != F_NONE { return splu_empty(), bc; }
    if s.rows != s.cols {
        return splu_empty(), fault(F_SHAPE, "spilu", "the matrix is not square");
    }
    if s.rows > MAX_SPLU_SIDE {
        return splu_empty(), fault(F_LIMIT, "spilu", "the side is above 16384");
    }
    let t, tf = csr_sorted(s);
    if tf.kind != F_NONE { return splu_empty(), tf; }
    let n = t.rows;
    // The quota of the fill follows the estimate of SuperLU. It counts
    // half of the stored values and two sides.
    var cap: i64 = ((s.nnz as f64 / 2.0 + 2.0 * n as f64) * fill_factor) as i64;
    if cap < n { cap = n; }
    var wc: f64[>..] = [];
    var wv: f64[>..] = [];
    var rstart: i64[>..] = [];
    var rlen: i64[>..] = [];
    var diag: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        rstart.push(wc.len);
        // The diagonal of the row holds the threshold of the drop. A
        // row without a diagonal holds the threshold of zero, so the
        // rule drops nothing of that row.
        var dv: f64 = 0.0;
        var k = csr_ptr(t, i);
        while k < csr_ptr(t, i + 1) {
            let v = csr_val(t, k);
            if mathshim::is_finite_of(v) == 0.0 {
                return splu_empty(), fault(F_VALUE, "spilu", "the matrix holds a value that is not finite");
            }
            if csr_col(t, k) == i { dv = mathshim::abs_of(v); }
            if v != 0.0 { wc.push(csr_col(t, k) as f64); wv.push(v); }
            k += 1;
        }
        diag.push(dv);
        rlen.push(wc.len - rstart[i]);
        i += 1;
    }
    var live: i64 = wc.len;
    var step: i64 = 0;
    while step < n {
        if wc.len - live > live { lu_repack(wc, wv, rstart, rlen); }
        // The pivot is the diagonal of the row step. The order holds no
        // row exchange, so a pivot of zero is a fault.
        let pr = step;
        let pp = lu_find(wc, rstart[pr], rlen[pr], step);
        if pp < 0 {
            return splu_empty(), fault(F_VALUE, "spilu", "the matrix is singular under the rule without a pivot");
        }
        let pivot = wv[pp];
        i = step + 1;
        while i < n {
            let r = i;
            let q = lu_find(wc, rstart[r], rlen[r], step);
            if q >= 0 {
                let m = wv[q] / pivot;
                let ti = drop_tol * diag[r];
                // A place that the matrix holds stays, whatever the
                // multiplier is. A place that the elimination made
                // drops below the threshold, and its update goes away
                // with it.
                let keep = mathshim::abs_of(m) > ti || ilu_in_pattern(t, r, step);
                let rs = rstart[r];
                let re = rs + rlen[r];
                var nstart: i64 = 0;
                if keep {
                    let ps = rstart[pr];
                    let pe = ps + rlen[pr];
                    let room = rlen[r] + (pe - pp - 1);
                    // The quota of the fill is checked on the finished
                    // factor, because the room here overestimates. The
                    // working rows hold their own limit.
                    if wc.len + room > MAX_ELEMENTS {
                        return splu_empty(), fault(F_LIMIT, "spilu", "the working rows are above the limit");
                    }
                    nstart = wc.len;
                    var a: i64 = rs;
                    while a < q {
                        let c = wc[a];
                        let v = wv[a];
                        wc.push(c);
                        wv.push(v);
                        a += 1;
                    }
                    wc.push(step as f64);
                    wv.push(m);
                    a = q + 1;
                    var b: i64 = pp + 1;
                    while a < re || b < pe {
                        var c: f64 = 0.0;
                        var v: f64 = 0.0;
                        if b >= pe || (a < re && wc[a] < wc[b]) {
                            c = wc[a]; v = wv[a]; a += 1;
                        } else if a >= re || wc[b] < wc[a] {
                            c = wc[b]; v = -m * wv[b]; b += 1;
                        } else {
                            c = wc[a]; v = wv[a] - m * wv[b]; a += 1; b += 1;
                        }
                        // The diagonal stays. A fill above the diagonal
                        // drops below the threshold of its row.
                        if v != 0.0 && !(c > r as f64 && !ilu_in_pattern(t, r, c as i64)
                            && mathshim::abs_of(v) <= ti) {
                            wc.push(c); wv.push(v);
                        }
                    }
                    live += (wc.len - nstart) - rlen[r];
                    rstart[r] = nstart;
                    rlen[r] = wc.len - nstart;
                } else {
                    nstart = wc.len;
                    var a: i64 = rs;
                    while a < q {
                        let c = wc[a];
                        let v = wv[a];
                        wc.push(c);
                        wv.push(v);
                        a += 1;
                    }
                    a = q + 1;
                    while a < re {
                        let c = wc[a];
                        let v = wv[a];
                        wc.push(c);
                        wv.push(v);
                        a += 1;
                    }
                    live -= 1;
                    rstart[r] = nstart;
                    rlen[r] = wc.len - nstart;
                }
            }
            i += 1;
        }
        step += 1;
    }
    // The permutations are the identity, so lu_part splits every row at
    // its own place.
    var perm: i64[>..] = [];
    i = 0;
    while i < n { perm.push(i); i += 1; }
    let lp, lidx, lval, nl = lu_part(wc, wv, rstart, rlen, perm, true);
    let up, uidx, uval, nu = lu_part(wc, wv, rstart, rlen, perm, false);
    if nl + nu > cap || splu_len(n, nl, nu) > MAX_ELEMENTS {
        return splu_empty(), fault(F_LIMIT, "spilu", "the factor is above the quota of the fill factor");
    }
    var buf: f64[>..] = [];
    i = 0;
    while i < n { buf.push(i as f64); i += 1; }
    i = 0;
    while i < n { buf.push(i as f64); i += 1; }
    i = 0;
    while i < lp.len { buf.push(lp[i]); i += 1; }
    i = 0;
    while i < lidx.len { buf.push(lidx[i]); i += 1; }
    i = 0;
    while i < lval.len { buf.push(lval[i]); i += 1; }
    i = 0;
    while i < up.len { buf.push(up[i]); i += 1; }
    i = 0;
    while i < uidx.len { buf.push(uidx[i]); i += 1; }
    i = 0;
    while i < uval.len { buf.push(uval[i]); i += 1; }
    // A value that overflowed during the work is not a factor.
    if !qz_finite(buf) {
        return splu_empty(), fault(F_VALUE, "spilu", "a value of the factor is not finite");
    }
    return Splu { n: n, nnz_l: nl, nnz_u: nu, buf: buf }, no_fault();
}
