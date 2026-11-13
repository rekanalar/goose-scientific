// The iteration of lobpcg.
//
// The loop follows the locally optimal block preconditioned conjugate
// gradient method. Each step builds the residual block R and the direction
// block P. It solves the Rayleigh-Ritz problem on the blocks [X, R, P] and
// updates X. A restart drops P. The last step projects the best iterate
// again and solves the problem on it.
namespace scientific;

import mathshim;

// lob_constrain removes from a block of c columns its part in the span of
// the constraints Y. l holds the Cholesky factor of Y^T B Y.
fn lob_constrain(v: f64[:], c: i64, n: i64, y: f64[:], by: f64[:], l: f64[:], sy: i64) -> f64[>..] {
    if sy == 0 { return copy_slice(v); }
    let ybv = lob_tmul(by, n, sy, v, c);
    let t = lob_chol_solve(l, sy, ybv, c);
    return lob_sub(v, lob_mul(y, n, sy, t, c));
}

// lob_residual gives R = AX - BX diag(lam) for blocks of sx columns.
fn lob_residual(ax: f64[:], bx: f64[:], lam: f64[:], n: i64, sx: i64) -> f64[>..] {
    var r: f64[>..] = nd_zeros(n * sx);
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < sx { r[i * sx + j] = ax[i * sx + j] - bx[i * sx + j] * lam[j]; j += 1; }
        i += 1;
    }
    return r;
}

// lob_row_set writes a row of a history. It pads the history with zero rows
// when the row lies past its end.
fn lob_row_set(h: f64[>..]&, row: i64, vals: f64[:], k: i64) {
    while h.len < (row + 1) * k { h.push(0.0); }
    var j: i64 = 0;
    while j < k { h[row * k + j] = vals[j]; j += 1; }
}

// lob_take gives the first rows of a history, padded with zero rows.
fn lob_take(h: f64[:], rows: i64, k: i64) -> f64[>..] {
    return copy_slice(h[0..rows * k]);
}

// lob_mean_abs gives the mean of the absolute values of a list.
fn lob_mean_abs(v: f64[:]) -> f64 {
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < v.len { t += mathshim::abs_of(v[i]); i += 1; }
    return t / v.len;
}

// lob_max_abs gives the largest absolute value of a list.
fn lob_max_abs(v: f64[:]) -> f64 {
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < v.len { if mathshim::abs_of(v[i]) > t { t = mathshim::abs_of(v[i]); } i += 1; }
    return t;
}

// lob_put writes the block blk of br by bc values into the dim by dim
// matrix g at the place (r0, c0). With mirror it writes the transpose into
// the mirrored place too.
fn lob_put(g: f64[>..]&, dim: i64, r0: i64, c0: i64, blk: f64[:], br: i64, bc: i64, mirror: bool) {
    var i: i64 = 0;
    while i < br {
        var j: i64 = 0;
        while j < bc {
            g[(r0 + i) * dim + c0 + j] = blk[i * bc + j];
            if mirror { g[(c0 + j) * dim + r0 + i] = blk[i * bc + j]; }
            j += 1;
        }
        i += 1;
    }
}

// lob_ident gives the identity block of k by k values.
fn lob_ident(k: i64) -> f64[>..] {
    var out: f64[>..] = nd_zeros(k * k);
    var i: i64 = 0;
    while i < k { out[i * k + i] = 1.0; i += 1; }
    return out;
}

// lob_diag gives the diagonal block of the values in lam.
fn lob_diag(lam: f64[:], k: i64) -> f64[>..] {
    var out: f64[>..] = nd_zeros(k * k);
    var i: i64 = 0;
    while i < k { out[i * k + i] = lam[i]; i += 1; }
    return out;
}

// lob_ritz gives the block of n by sx values that the Ritz vectors of the
// rows from row0 to row0 + cnt of the vector block ev form with the block
// v of n by cnt values.
fn lob_ritz(v: f64[:], ev: f64[:], row0: i64, cnt: i64, n: i64, sx: i64) -> f64[>..] {
    return lob_mul(v, n, cnt, ev[row0 * sx..(row0 + cnt) * sx], sx);
}

// lob_iterate runs the iteration. The blocks of A, B and M hold n rows.
fn lob_iterate(a, x0: f64[:], n: i64, sx: i64, b, m, y: f64[:], sy: i64, tol: f64, maxiter: i64, largest: bool,
               rc: i64, want_l: bool, want_r: bool) -> Arr, Arr, i64, Arr, Arr, Fault {
    // The constraints.
    var by: f64[>..] = [];
    var yl: f64[>..] = [];
    if sy > 0 {
        by = lob_apply(b, y, sy);
        let gy = lob_tmul(y, n, sy, by, sy);
        let l, ok = lob_chol(gy, sy);
        if !ok { return empty_arr(), empty_arr(), 0, empty_arr(), empty_arr(), fault(F_VALUE, "lobpcg", "the constraints are linearly dependent"); }
        yl = copy_slice(l);
    }
    // The start block, free of the constraints and B-orthonormal.
    let xc = lob_constrain(x0, sx, n, y, by, yl, sy);
    let bxc = lob_apply(b, xc, sx);
    let xo, bxo, uo, oko = lob_orth(xc, bxc, n, sx);
    if !oko { return empty_arr(), empty_arr(), 0, empty_arr(), empty_arr(), fault(F_VALUE, "lobpcg", "the start vectors are linearly dependent"); }
    var xb: f64[>..] = copy_slice(xo);
    var bx: f64[>..] = copy_slice(bxo);
    var ax: f64[>..] = lob_spmm(a, xb, sx);
    var lam: f64[>..] = [];
    let g0 = lob_tmul(xb, n, sx, ax, sx);
    let lv0, ev0, ok0 = lob_rr(g0, lob_ident(sx), sx, sx, largest);
    if !ok0 { return empty_arr(), empty_arr(), 0, empty_arr(), empty_arr(), fault(F_VALUE, "lobpcg", "the first Rayleigh-Ritz step failed"); }
    lam = copy_slice(lv0);
    let mx = lob_mul(xb, n, sx, ev0, sx);
    let ma = lob_mul(ax, n, sx, ev0, sx);
    let mb = lob_mul(bx, n, sx, ev0, sx);
    xb = copy_slice(mx);
    ax = copy_slice(ma);
    bx = copy_slice(mb);
    var lam_h: f64[>..] = [];
    var res_h: f64[>..] = [];
    if want_l { lob_row_set(lam_h, 0, lam, sx); }
    var mask: bool[>..] = [];
    var q: i64 = 0;
    while q < sx { mask.push(true); q += 1; }
    var p: f64[>..] = [];
    var ap: f64[>..] = [];
    var bp: f64[>..] = [];
    var best_x: f64[>..] = copy_slice(xb);
    var smallest: f64 = 1.7976931348623157e308;
    var best_iter: i64 = maxiter;
    var iter: i64 = -1;
    var restart = true;
    var forced = false;
    var explicit = false;
    let myeps = 1.4901161193847656e-08;
    let factor = mathshim::power(2.0, rc as f64);
    while iter < maxiter {
        iter += 1;
        let r = lob_residual(ax, bx, lam, n, sx);
        let norms = lob_colnorms(r, n, sx);
        if want_r { lob_row_set(res_h, iter, norms, sx); }
        let mean = lob_mean_abs(norms);
        if mean < smallest {
            smallest = mean;
            best_iter = iter;
            best_x = copy_slice(xb);
        } else if mean > factor * smallest {
            forced = true;
            ax = lob_spmm(a, xb, sx);
            bx = lob_apply(b, xb, sx);
        }
        var cur: i64 = 0;
        q = 0;
        while q < sx {
            if !(norms[q] > tol) { mask[q] = false; }
            if mask[q] { cur += 1; }
            q += 1;
        }
        if cur == 0 { break; }
        var ar: f64[>..] = lob_cols(r, n, sx, mask, cur);
        var ap_a: f64[>..] = [];
        var bp_a: f64[>..] = [];
        var p_a: f64[>..] = [];
        if iter > 0 {
            p_a = lob_cols(p, n, sx, mask, cur);
            ap_a = lob_cols(ap, n, sx, mask, cur);
            bp_a = lob_cols(bp, n, sx, mask, cur);
        }
        if m.rows != 0 { let pm = lob_spmm(m, ar, cur); ar = copy_slice(pm); }
        let c1 = lob_constrain(ar, cur, n, y, by, yl, sy);
        // Remove the part of the residuals in X, in the B product.
        let c2 = lob_sub(c1, lob_mul(xb, n, sx, lob_tmul(bx, n, sx, c1, cur), cur));
        ar = copy_slice(c2);
        let bar0 = lob_apply(b, ar, cur);
        let ro, bro, uro, okr = lob_orth(ar, bar0, n, cur);
        if !okr { break; }
        let act_r = copy_slice(ro);
        let act_br = copy_slice(bro);
        let act_ar = lob_spmm(a, act_r, cur);
        if iter > 0 {
            let po, bpo, upo, okp = lob_orth(p_a, bp_a, n, cur);
            if okp {
                let na = lob_mul(ap_a, n, cur, upo, cur);
                p_a = copy_slice(po);
                bp_a = copy_slice(bpo);
                ap_a = copy_slice(na);
                restart = forced;
            } else {
                restart = true;
            }
        }
        if lob_max_abs(norms) > myeps && !explicit { explicit = false; } else { explicit = true; }
        let gxar = lob_tmul(xb, n, sx, act_ar, cur);
        var grar = lob_tmul(act_r, n, cur, act_ar, cur);
        var gxax: f64[>..] = lob_diag(lam, sx);
        var gxbx: f64[>..] = lob_ident(sx);
        var grbr: f64[>..] = lob_ident(cur);
        var gxbr: f64[>..] = nd_zeros(sx * cur);
        if explicit {
            let e1 = lob_symmetrize(grar, cur);
            let e2 = lob_symmetrize(lob_tmul(xb, n, sx, ax, sx), sx);
            let e3 = lob_tmul(xb, n, sx, bx, sx);
            let e4 = lob_tmul(act_r, n, cur, act_br, cur);
            let e5 = lob_tmul(xb, n, sx, act_br, cur);
            grar = copy_slice(e1);
            gxax = copy_slice(e2);
            gxbx = copy_slice(e3);
            grbr = copy_slice(e4);
            gxbr = copy_slice(e5);
        }
        var ev: f64[>..] = [];
            var dim = sx + cur;
        if !restart {
            dim = sx + 2 * cur;
            let gxap = lob_tmul(xb, n, sx, ap_a, cur);
            let grap = lob_tmul(act_r, n, cur, ap_a, cur);
            let h1 = lob_tmul(p_a, n, cur, ap_a, cur);
            let gxbp = lob_tmul(xb, n, sx, bp_a, cur);
            let grbp = lob_tmul(act_r, n, cur, bp_a, cur);
            var gpbp: f64[>..] = lob_ident(cur);
            var gpap: f64[>..] = copy_slice(h1);
            if explicit {
                let h2 = lob_symmetrize(gpap, cur);
                let h3 = lob_tmul(p_a, n, cur, bp_a, cur);
                gpap = copy_slice(h2);
                gpbp = copy_slice(h3);
            }
            var ga: f64[>..] = nd_zeros(dim * dim);
            var gb: f64[>..] = nd_zeros(dim * dim);
            lob_put(ga, dim, 0, 0, gxax, sx, sx, false);
            lob_put(ga, dim, 0, sx, gxar, sx, cur, true);
            lob_put(ga, dim, 0, sx + cur, gxap, sx, cur, true);
            lob_put(ga, dim, sx, sx, grar, cur, cur, false);
            lob_put(ga, dim, sx, sx + cur, grap, cur, cur, true);
            lob_put(ga, dim, sx + cur, sx + cur, gpap, cur, cur, false);
            lob_put(gb, dim, 0, 0, gxbx, sx, sx, false);
            lob_put(gb, dim, 0, sx, gxbr, sx, cur, true);
            lob_put(gb, dim, 0, sx + cur, gxbp, sx, cur, true);
            lob_put(gb, dim, sx, sx, grbr, cur, cur, false);
            lob_put(gb, dim, sx, sx + cur, grbp, cur, cur, true);
            lob_put(gb, dim, sx + cur, sx + cur, gpbp, cur, cur, false);
            let lv, e3, ok3 = lob_rr(ga, gb, dim, sx, largest);
            if ok3 { lam = copy_slice(lv); ev = copy_slice(e3); } else { restart = true; }
        }
        if restart {
            dim = sx + cur;
            var ga2: f64[>..] = nd_zeros(dim * dim);
            var gb2: f64[>..] = nd_zeros(dim * dim);
            lob_put(ga2, dim, 0, 0, gxax, sx, sx, false);
            lob_put(ga2, dim, 0, sx, gxar, sx, cur, true);
            lob_put(ga2, dim, sx, sx, grar, cur, cur, false);
            lob_put(gb2, dim, 0, 0, gxbx, sx, sx, false);
            lob_put(gb2, dim, 0, sx, gxbr, sx, cur, true);
            lob_put(gb2, dim, sx, sx, grbr, cur, cur, false);
            let lv2, e2, ok2 = lob_rr(ga2, gb2, dim, sx, largest);
            if !ok2 { break; }
            lam = copy_slice(lv2);
            ev = copy_slice(e2);
        }
        if want_l { lob_row_set(lam_h, iter + 1, lam, sx); }
        // The Ritz vectors.
        let ex = ev[0..sx * sx];
        let r1 = lob_ritz(act_r, ev, sx, cur, n, sx);
        let r2 = lob_ritz(act_ar, ev, sx, cur, n, sx);
        let r3 = lob_ritz(act_br, ev, sx, cur, n, sx);
        var pp: f64[>..] = copy_slice(r1);
        var app: f64[>..] = copy_slice(r2);
        var bpp: f64[>..] = copy_slice(r3);
        if !restart {
            let r4 = lob_add(pp, lob_ritz(p_a, ev, sx + cur, cur, n, sx));
            let r5 = lob_add(app, lob_ritz(ap_a, ev, sx + cur, cur, n, sx));
            let r6 = lob_add(bpp, lob_ritz(bp_a, ev, sx + cur, cur, n, sx));
            pp = copy_slice(r4);
            app = copy_slice(r5);
            bpp = copy_slice(r6);
        }
        let nx = lob_add(lob_mul(xb, n, sx, ex, sx), pp);
        let na = lob_add(lob_mul(ax, n, sx, ex, sx), app);
        let nb = lob_add(lob_mul(bx, n, sx, ex, sx), bpp);
        xb = copy_slice(nx);
        ax = copy_slice(na);
        bx = copy_slice(nb);
        p = copy_slice(pp);
        ap = copy_slice(app);
        bp = copy_slice(bpp);
    }
    return lob_finish(a, b, n, sx, sy, y, by, yl, largest, lam, xb, ax, bx, best_x, smallest, best_iter, iter, lam_h, res_h, want_l, want_r);
}

// lob_finish projects the best iterate on the constraints again and solves
// the problem on it. It builds the result and the histories.
fn lob_finish(a, b, n: i64, sx: i64, sy: i64, y: f64[:], by: f64[:], yl: f64[:], largest: bool, lam0: f64[:],
              xb: f64[:], ax: f64[:], bx: f64[:], best_x: f64[:], smallest0: f64, best_iter0: i64, iter: i64,
              lam_h0: f64[:], res_h0: f64[:], want_l: bool, want_r: bool) -> Arr, Arr, i64, Arr, Arr, Fault {
    var lam_h: f64[>..] = copy_slice(lam_h0);
    var res_h: f64[>..] = copy_slice(res_h0);
    var smallest = smallest0;
    var best_iter = best_iter0;
    var best: f64[>..] = copy_slice(best_x);
    // The residual of the last iterate decides whether it beats the best one.
    let r = lob_residual(ax, bx, lam0, n, sx);
    let norms = lob_colnorms(r, n, sx);
    if want_l { lob_row_set(lam_h, iter + 1, lam0, sx); }
    if want_r { lob_row_set(res_h, iter + 1, norms, sx); }
    let mean = lob_mean_abs(norms);
    if mean < smallest {
        smallest = mean;
        best_iter = iter + 1;
        best = copy_slice(xb);
    }
    // The final projection and the exact Rayleigh-Ritz step.
    let xp = lob_constrain(best, sx, n, y, by, yl, sy);
    let axp = lob_spmm(a, xp, sx);
    let bxp = lob_apply(b, xp, sx);
    let gax = lob_symmetrize(lob_tmul(xp, n, sx, axp, sx), sx);
    let gbx = lob_symmetrize(lob_tmul(xp, n, sx, bxp, sx), sx);
    let lv, ev, ok = lob_rr(gax, gbx, sx, sx, largest);
    if !ok { return empty_arr(), empty_arr(), 0, empty_arr(), empty_arr(), fault(F_VALUE, "lobpcg", "the last Rayleigh-Ritz step failed"); }
    let mx = lob_mul(xp, n, sx, ev, sx);
    let ma = lob_mul(axp, n, sx, ev, sx);
    let mb = lob_mul(bxp, n, sx, ev, sx);
    let xf = copy_slice(mx);
    let axf = copy_slice(ma);
    let bxf = copy_slice(mb);
    let rf = lob_residual(axf, bxf, lv, n, sx);
    let nf = lob_colnorms(rf, n, sx);
    var lh = empty_arr();
    var rh = empty_arr();
    if want_l {
        lob_row_set(lam_h, best_iter + 1, lv, sx);
        lh = mat2(lob_take(lam_h, best_iter + 2, sx), best_iter + 2, sx);
    }
    if want_r {
        lob_row_set(res_h, best_iter + 1, nf, sx);
        rh = mat2(lob_take(res_h, best_iter + 2, sx), best_iter + 2, sx);
    }
    return line_arr(copy_slice(lv)), mat2(copy_slice(xf), n, sx), iter, lh, rh, no_fault();
}
