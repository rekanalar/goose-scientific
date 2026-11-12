// The sparse LU factor of a square matrix.
//
// sparse_lu factors a matrix row by row. It keeps the matrix sparse, and
// the fill-in is the only growth. Each step picks the row with the largest
// absolute value in the column, as the partial pivot of LAPACK does. The
// columns keep their order, so the column permutation is the identity.
// SuperLU orders the columns and pivots with a threshold, so its factors
// can differ. The identity L U = A[perm_r][:, perm_c] and the solution
// are the same.
//
// The buffer of a factor holds, in this order: the row permutation (n
// values), the column permutation (n values), then L and U, each as
// compressed rows (n + 1 row pointers, the columns, the values). L holds
// the strictly lower part and its unit diagonal stays implied. U holds the
// upper part with its diagonal.
namespace scientific;

import mathshim;

// The largest side of a factor. The pivot search passes over the rows at
// each step, so the work grows with the square of the side.
const MAX_SPLU_SIDE: i64 = 16384;

// Splu holds the factor of a square matrix.
struct Splu { n: i64, nnz_l: i64, nnz_u: i64, buf: f64[>..] }

fn splu_empty() -> Splu {
    return Splu { n: 0, nnz_l: 0, nnz_u: 0, buf: [] };
}

// splu_len gives the length of the buffer: the two permutations, then
// the row pointers, the columns and the values of L and of U.
fn splu_len(n: i64, nnz_l: i64, nnz_u: i64) -> i64 {
    return 2 * n + 2 * (n + 1) + 2 * nnz_l + 2 * nnz_u;
}

// splu_count gives the stored values of L (strictly lower) or of U.
fn splu_count(f, lower: bool) -> i64 {
    if lower { return f.nnz_l; }
    return f.nnz_u;
}

// splu_off_ptr gives the start of the row pointers of L or of U.
fn splu_off_ptr(f, lower: bool) -> i64 {
    if lower { return 2 * f.n; }
    return 2 * f.n + (f.n + 1) + 2 * f.nnz_l;
}

// splu_off_col gives the start of the columns of L or of U.
fn splu_off_col(f, lower: bool) -> i64 {
    return splu_off_ptr(f, lower) + f.n + 1;
}

// splu_off_val gives the start of the values of L or of U.
fn splu_off_val(f, lower: bool) -> i64 {
    return splu_off_col(f, lower) + splu_count(f, lower);
}

// splu_check_part checks the layout of L or of U.
fn splu_check_part(f, lower: bool, tag: const u8[:]) -> Fault {
    let n = f.n;
    let cnt = splu_count(f, lower);
    let po = splu_off_ptr(f, lower);
    let co = splu_off_col(f, lower);
    let vo = splu_off_val(f, lower);
    var last: i64 = 0;
    var i: i64 = 0;
    while i <= n {
        let p = f.buf[po + i];
        if !whole_place(p) || p < 0.0 {
            return fault(F_SHAPE, tag, "a row pointer of the factor is not a whole number");
        }
        let q = p as i64;
        if i == 0 {
            if q != 0 { return fault(F_SHAPE, tag, "the first row pointer of the factor is not zero"); }
        } else if q < last || q > cnt {
            return fault(F_SHAPE, tag, "the row pointers of the factor do not rise inside the count");
        }
        last = q;
        i += 1;
    }
    if last != cnt {
        return fault(F_SHAPE, tag, "the last row pointer of the factor differs from its count");
    }
    i = 0;
    while i < n {
        var k = f.buf[po + i] as i64;
        let e = f.buf[po + i + 1] as i64;
        if !lower {
            if k >= e || f.buf[co + k] != i as f64 || !(mathshim::abs_of(f.buf[vo + k]) > 0.0) {
                return fault(F_SHAPE, tag, "a row of U holds no diagonal or a diagonal of zero");
            }
        }
        var prev: i64 = -1;
        while k < e {
            let c = f.buf[co + k];
            if !whole_place(c) || c < 0.0 || c >= n as f64 {
                return fault(F_SHAPE, tag, "a column of the factor lies outside the matrix");
            }
            let ci = c as i64;
            if ci <= prev {
                return fault(F_SHAPE, tag, "the columns of a row of the factor do not rise");
            }
            if lower && ci >= i {
                return fault(F_SHAPE, tag, "a column of L does not lie below its row");
            }
            if !lower && ci < i {
                return fault(F_SHAPE, tag, "a column of U lies below its row");
            }
            if mathshim::is_finite_of(f.buf[vo + k]) == 0.0 {
                return fault(F_SHAPE, tag, "a value of the factor is not finite");
            }
            prev = ci;
            k += 1;
        }
        i += 1;
    }
    return no_fault();
}

// splu_check_perm checks that the n values from the offset are a
// permutation of 0 to n - 1.
fn splu_check_perm(f, off: i64, tag: const u8[:]) -> Fault {
    var seen: i64[>..] = [];
    var i: i64 = 0;
    while i < f.n { seen.push(0); i += 1; }
    i = 0;
    while i < f.n {
        let p = f.buf[off + i];
        if !whole_place(p) || p < 0.0 || p >= f.n as f64 {
            return fault(F_SHAPE, tag, "a permutation of the factor holds a place outside the matrix");
        }
        let q = p as i64;
        if seen[q] != 0 {
            return fault(F_SHAPE, tag, "a permutation of the factor repeats a place");
        }
        seen[q] = 1;
        i += 1;
    }
    return no_fault();
}

// splu_check checks the layout of a factor before any call walks the
// buffer. The side comes first, then the counts, then the length.
fn splu_check(f, tag: const u8[:]) -> Fault {
    if f.n < 1 || f.n > MAX_SPLU_SIDE {
        return fault(F_SHAPE, tag, "the side of the factor stands outside one to the limit");
    }
    if f.nnz_l < 0 || f.nnz_u < f.n || f.nnz_l > MAX_SPARSE_NNZ || f.nnz_u > MAX_SPARSE_NNZ || f.nnz_l + f.nnz_u + f.n > MAX_SPARSE_NNZ {
        return fault(F_SHAPE, tag, "a count of the factor stands outside its range");
    }
    let want = splu_len(f.n, f.nnz_l, f.nnz_u);
    if want > MAX_ELEMENTS || f.buf.len != want {
        return fault(F_SHAPE, tag, "the buffer of the factor does not match its counts");
    }
    let a = splu_check_perm(f, 0, tag);
    if a.kind != F_NONE { return a; }
    let b = splu_check_perm(f, f.n, tag);
    if b.kind != F_NONE { return b; }
    let l = splu_check_part(f, true, tag);
    if l.kind != F_NONE { return l; }
    return splu_check_part(f, false, tag);
}

// splu_nnz gives the stored values of L, with its unit diagonal, plus
// those of U. SciPy counts differently, so the two counts differ. A
// factor with a forged layout gives -1.
fn splu_nnz(f) -> i64 {
    if splu_check(f, "splu_nnz").kind != F_NONE { return -1; }
    return f.nnz_l + f.n + f.nnz_u;
}

// lu_find gives the place of the column c in the sorted columns of a
// working row, or -1.
fn lu_find(wc: f64[:], start: i64, len: i64, c: i64) -> i64 {
    var lo: i64 = 0;
    var hi: i64 = len - 1;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let v = wc[start + mid] as i64;
        if v == c { return start + mid; }
        if v < c { lo = mid + 1; } else { hi = mid - 1; }
    }
    return -1;
}

// lu_repack packs the working rows again and drops the dead space.
fn lu_repack(wc: f64[>..]&, wv: f64[>..]&, rstart: i64[>..]&, rlen: i64[>..]&) {
    var nc: f64[>..] = [];
    var nv: f64[>..] = [];
    var r: i64 = 0;
    while r < rstart.len {
        let s = rstart[r];
        let l = rlen[r];
        rstart[r] = nc.len;
        var e: i64 = 0;
        while e < l {
            nc.push(wc[s + e]);
            nv.push(wv[s + e]);
            e += 1;
        }
        r += 1;
    }
    wc = copy(nc);
    wv = copy(nv);
}

// lu_part gives the compressed rows of L or of U. The row at the step k
// of the order holds L below the column k and U from it. The two parts
// come in one pass each, and the count of the values comes with them.
fn lu_part(wc: f64[:], wv: f64[:], rstart: i64[:], rlen: i64[:], perm: i64[:], lower: bool) -> f64[>..], f64[>..], f64[>..], i64 {
    let n = perm.len;
    var ptr: f64[>..] = [];
    var idx: f64[>..] = [];
    var val: f64[>..] = [];
    ptr.push(0.0);
    var k: i64 = 0;
    while k < n {
        let r = perm[k];
        var e: i64 = 0;
        while e < rlen[r] {
            let c = wc[rstart[r] + e] as i64;
            if (c < k) == lower {
                idx.push(wc[rstart[r] + e]);
                val.push(wv[rstart[r] + e]);
            }
            e += 1;
        }
        ptr.push(idx.len);
        k += 1;
    }
    return ptr, idx, val, idx.len;
}

// sparse_lu factors a square sparse matrix. A matrix that is not square
// is a fault of kind F_SHAPE. A matrix that is singular, or holds a value
// that is not finite, is a fault of kind F_VALUE. A fault gives no factor.
fn sparse_lu(s) -> Splu, Fault {
    return lu_factor_limited(s, MAX_SPARSE_NNZ, MAX_ELEMENTS);
}

// lu_factor_limited is sparse_lu with the two limits as arguments. The
// count of the stored values of L and U stays at max_nnz or below. The
// working rows stay at max_scratch values or below. A caller can lower the
// limits to reach the faults with a small matrix. The limits of the
// package stand, so a limit above them has no effect.
fn lu_factor_limited(s, max_nnz: i64, max_scratch: i64) -> Splu, Fault {
    let bc = bmat_check_block(s, "sparse_lu");
    if bc.kind != F_NONE { return splu_empty(), bc; }
    if s.rows != s.cols {
        return splu_empty(), fault(F_SHAPE, "sparse_lu", "the matrix is not square");
    }
    if s.rows > MAX_SPLU_SIDE {
        return splu_empty(), fault(F_LIMIT, "sparse_lu", "the side is above 16384");
    }
    // The limits of the package stand. A caller can lower them, but it
    // cannot raise them above the limits of the package.
    let nnz_cap = if max_nnz < MAX_SPARSE_NNZ { max_nnz } else { MAX_SPARSE_NNZ };
    let scratch_cap = if max_scratch < MAX_ELEMENTS { max_scratch } else { MAX_ELEMENTS };
    let t, tf = csr_sorted(s);
    if tf.kind != F_NONE { return splu_empty(), tf; }
    let n = t.rows;
    var wc: f64[>..] = [];
    var wv: f64[>..] = [];
    var rstart: i64[>..] = [];
    var rlen: i64[>..] = [];
    var perm: i64[>..] = [];
    var at: i64[>..] = [];
    var i: i64 = 0;
    while i < n {
        at.push(-1);
        rstart.push(wc.len);
        var k = csr_ptr(t, i);
        while k < csr_ptr(t, i + 1) {
            let v = csr_val(t, k);
            if mathshim::is_finite_of(v) == 0.0 {
                return splu_empty(), fault(F_VALUE, "sparse_lu", "the matrix holds a value that is not finite");
            }
            if v != 0.0 {
                // The working rows stay at the scratch limit from the
                // first value, not only from the first merge.
                if wc.len + 1 > scratch_cap {
                    return splu_empty(), fault(F_LIMIT, "sparse_lu", "the working rows are above the limit");
                }
                wc.push(csr_col(t, k) as f64);
                wv.push(v);
            }
            k += 1;
        }
        rlen.push(wc.len - rstart[i]);
        perm.push(i);
        i += 1;
    }
    var live: i64 = wc.len;
    var step: i64 = 0;
    while step < n {
        // A repack only shrinks the buffers, so it needs no limit check.
        if wc.len - live > live { lu_repack(wc, wv, rstart, rlen); }
        // The pivot is the row with the largest absolute value in the column.
        // The place of the column in each row is kept, so that the pass
        // over the rows below does not search a second time.
        var best: i64 = -1;
        var bestv: f64 = 0.0;
        i = step;
        while i < n {
            let r = perm[i];
            let p = lu_find(wc, rstart[r], rlen[r], step);
            at[i] = p;
            if p >= 0 {
                if mathshim::is_finite_of(wv[p]) == 0.0 {
                    return splu_empty(), fault(F_VALUE, "sparse_lu", "a pivot candidate is not finite");
                }
                let a = mathshim::abs_of(wv[p]);
                if a > bestv { bestv = a; best = i; }
            }
            i += 1;
        }
        if best < 0 {
            return splu_empty(), fault(F_VALUE, "sparse_lu", "the matrix is singular");
        }
        let held = perm[step];
        perm[step] = perm[best];
        perm[best] = held;
        // The places of the column move with the rows.
        let heldat = at[step];
        at[step] = at[best];
        at[best] = heldat;
        let pr = perm[step];
        let ps = rstart[pr];
        let pe = ps + rlen[pr];
        let pp = lu_find(wc, ps, rlen[pr], step);
        let pv = wv[pp];
        at[step] = pp;
        i = step + 1;
        while i < n {
            let r = perm[i];
            let q = at[i];
            if q >= 0 {
                let m = wv[q] / pv;
                let rs = rstart[r];
                let re = rs + rlen[r];
                let room = rlen[r] + (pe - pp - 1);
                if live + room > nnz_cap {
                    return splu_empty(), fault(F_LIMIT, "sparse_lu", "the fill-in is above the limit");
                }
                if wc.len + room > scratch_cap {
                    return splu_empty(), fault(F_LIMIT, "sparse_lu", "the working rows are above the limit");
                }
                let nstart = wc.len;
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
                    if v != 0.0 { wc.push(c); wv.push(v); }
                }
                live += (wc.len - nstart) - rlen[r];
                rstart[r] = nstart;
                rlen[r] = wc.len - nstart;
            }
            i += 1;
        }
        step += 1;
    }
    // Build L and U, and check the size, before the buffer grows.
    let lp, lidx, lval, nl = lu_part(wc, wv, rstart, rlen, perm, true);
    let up, uidx, uval, nu = lu_part(wc, wv, rstart, rlen, perm, false);
    if nl + nu + n > nnz_cap || splu_len(n, nl, nu) > scratch_cap {
        return splu_empty(), fault(F_LIMIT, "sparse_lu", "the factor is above the limit");
    }
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < n { buf.push(perm[i] as f64); i += 1; }
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
        return splu_empty(), fault(F_VALUE, "sparse_lu", "a value of the factor is not finite");
    }
    return Splu { n: n, nnz_l: nl, nnz_u: nu, buf: buf }, no_fault();
}

// splu_core solves with the factor for one right side that the caller
// checked. It undoes the row permutation, runs the forward and the back
// substitution and then places the unknowns by the column permutation.
fn splu_core(f, rhs: f64[:]) -> f64[>..] {
    let n = f.n;
    var y: f64[>..] = [];
    var i: i64 = 0;
    while i < n { y.push(rhs[f.buf[i] as i64]); i += 1; }
    let lp = splu_off_ptr(f, true);
    let lc = splu_off_col(f, true);
    let lv = splu_off_val(f, true);
    i = 0;
    while i < n {
        var k = f.buf[lp + i] as i64;
        let e = f.buf[lp + i + 1] as i64;
        var t: f64 = y[i];
        while k < e { t -= f.buf[lv + k] * y[f.buf[lc + k] as i64]; k += 1; }
        y[i] = t;
        i += 1;
    }
    let up = splu_off_ptr(f, false);
    let uc = splu_off_col(f, false);
    let uv = splu_off_val(f, false);
    i = n - 1;
    while i >= 0 {
        var k = f.buf[up + i] as i64;
        let e = f.buf[up + i + 1] as i64;
        let d = f.buf[uv + k];
        var t: f64 = y[i];
        k += 1;
        while k < e { t -= f.buf[uv + k] * y[f.buf[uc + k] as i64]; k += 1; }
        y[i] = t / d;
        i -= 1;
    }
    var x = nd_zeros(n);
    i = 0;
    while i < n { x[f.buf[n + i] as i64] = y[i]; i += 1; }
    return x;
}

// splu_solve solves a x = b with a factor of a. A right side of the wrong
// length is a fault of kind F_SHAPE. A value that is not finite, in the
// right side or in the solution, is a fault of kind F_VALUE.
fn splu_solve(f, b: f64[:]) -> Arr, Fault {
    let c = splu_check(f, "splu_solve");
    if c.kind != F_NONE { return empty_arr(), c; }
    if b.len != f.n {
        return empty_arr(), fault(F_SHAPE, "splu_solve", "the right side does not fit the factor");
    }
    if !qz_finite(b) {
        return empty_arr(), fault(F_VALUE, "splu_solve", "the right side holds a value that is not finite");
    }
    let x = splu_core(f, b);
    if !qz_finite(x) {
        return empty_arr(), fault(F_VALUE, "splu_solve", "the solution is not finite");
    }
    return line_arr(copy(x)), no_fault();
}

// splu_solve_matrix solves a x = b for each column of the matrix b.
fn splu_solve_matrix(f, b) -> Arr, Fault {
    let c = splu_check(f, "splu_solve_matrix");
    if c.kind != F_NONE { return empty_arr(), c; }
    if b.ndim != 2 || b.shape.len != 2 || b.shape[0] != f.n || b.shape[1] < 0 {
        return empty_arr(), fault(F_SHAPE, "splu_solve_matrix", "the right side does not fit the factor");
    }
    let cols = b.shape[1];
    // The count of the elements is checked before the product, so that a
    // large shape cannot overflow the count.
    if cols > MAX_ELEMENTS / f.n {
        return empty_arr(), fault(F_LIMIT, "splu_solve_matrix", "the right side holds more elements than the limit");
    }
    if b.data.len != f.n * cols {
        return empty_arr(), fault(F_SHAPE, "splu_solve_matrix", "the right side does not fit the factor");
    }
    if !qz_finite(b.data) {
        return empty_arr(), fault(F_VALUE, "splu_solve_matrix", "the right side holds a value that is not finite");
    }
    var out = nd_zeros(f.n * cols);
    var i: i64 = 0;
    var j: i64 = 0;
    while j < cols {
        var col: f64[>..] = [];
        i = 0;
        while i < f.n { col.push(b.data[i * cols + j]); i += 1; }
        let x = splu_core(f, col);
        if !qz_finite(x) {
            return empty_arr(), fault(F_VALUE, "splu_solve_matrix", "the solution is not finite");
        }
        i = 0;
        while i < f.n { out[i * cols + j] = x[i]; i += 1; }
        j += 1;
    }
    return mat2(copy(out), f.n, cols), no_fault();
}

// splu_l gives L with its unit diagonal as compressed rows.
fn splu_l(f) -> Csr, Fault {
    let c = splu_check(f, "splu_l");
    if c.kind != F_NONE { return csr_empty(), c; }
    let po = splu_off_ptr(f, true);
    let co = splu_off_col(f, true);
    let vo = splu_off_val(f, true);
    var ptr: f64[>..] = [];
    var idx: f64[>..] = [];
    var val: f64[>..] = [];
    ptr.push(0.0);
    var i: i64 = 0;
    while i < f.n {
        var k = f.buf[po + i] as i64;
        let e = f.buf[po + i + 1] as i64;
        while k < e { idx.push(f.buf[co + k]); val.push(f.buf[vo + k]); k += 1; }
        idx.push(i as f64);
        val.push(1.0);
        ptr.push(idx.len);
        i += 1;
    }
    return csr_from_parts(f.n, f.n, ptr, idx, val), no_fault();
}

// splu_u gives U as compressed rows.
fn splu_u(f) -> Csr, Fault {
    let c = splu_check(f, "splu_u");
    if c.kind != F_NONE { return csr_empty(), c; }
    let po = splu_off_ptr(f, false);
    var buf: f64[>..] = [];
    var i: i64 = 0;
    while i < f.n + 1 + 2 * f.nnz_u { buf.push(f.buf[po + i]); i += 1; }
    return Csr { rows: f.n, cols: f.n, nnz: f.nnz_u, buf: buf }, no_fault();
}

// splu_perm_r gives the row permutation: row i of L U is the row
// perm_r[i] of the matrix.
fn splu_perm_r(f) -> Arr, Fault {
    let c = splu_check(f, "splu_perm_r");
    if c.kind != F_NONE { return empty_arr(), c; }
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < f.n { v.push(f.buf[i]); i += 1; }
    return line_arr(copy(v)), no_fault();
}

// splu_perm_c gives the column permutation. It is the identity.
fn splu_perm_c(f) -> Arr, Fault {
    let c = splu_check(f, "splu_perm_c");
    if c.kind != F_NONE { return empty_arr(), c; }
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < f.n { v.push(f.buf[f.n + i]); i += 1; }
    return line_arr(copy(v)), no_fault();
}
