// The core of the call gcrotmk and its public calls.
//
namespace scientific;

import mathshim;

// gcrotmk solves the system A x = b by the method GCROT(m,k). The call
// starts from x0, or from zero when x0 is empty. The inner size m is the
// count of Krylov steps of one outer iteration. The call keeps up to k pairs
// of vectors. A k of zero means m. With truncate_oldest set, the call drops
// the oldest pair when it holds k of them. Else it keeps the directions of
// the largest singular values. The flag discard_c leaves the c vectors out
// of the kept pairs.
fn gcrotmk_run<F>(s, b: f64[:], x0: f64[:], rtol: f64, atol: f64, maxiter: i64, m: i64, k: i64,
                  discard_c: bool, truncate_oldest: bool, mp, cu_in, want_pairs: bool) -> Arr, i64, Kept, Fault {
    let f = lg_check(s, b, x0, rtol, atol, maxiter, "gcrotmk");
    if f.kind != F_NONE { return empty_arr(), 0, kept_empty(), f; }
    if m < 1 { return empty_arr(), 0, kept_empty(), fault(F_RANGE, "gcrotmk", "the inner size is below one"); }
    if k < 0 { return empty_arr(), 0, kept_empty(), fault(F_RANGE, "gcrotmk", "the count of kept pairs is below zero"); }
    let n = s.rows;
    let fp = precond_check(mp, n, "gcrotmk");
    if fp.kind != F_NONE { return empty_arr(), 0, kept_empty(), fp; }
    let fq = kept_check(cu_in, n, "gcrotmk");
    if fq.kind != F_NONE { return empty_arr(), 0, kept_empty(), fq; }
    var pmode: i64 = PRE_NONE;
    if precond_present(mp) { pmode = PRE_RIGHT; }
    let cnt = cu_in.count;
    var kk = k;
    if kk == 0 { kk = m; }
    var m_eff = m;
    if m_eff > n { m_eff = n; }
    var k_eff = kk;
    if k_eff > maxiter { k_eff = maxiter; }
    if k_eff > n { k_eff = n; }
    var kst = k_eff;
    if cnt > kst { kst = cnt; }
    let mt = m_eff + k_eff;
    let fs = lg_storage_check(n, mt, 4 * kst, "gcrotmk");
    if fs.kind != F_NONE { return empty_arr(), 0, kept_empty(), fs; }
    let bn = lg_norm(b, 0, n);
    if bn - bn != 0.0 { return empty_arr(), 0, kept_empty(), fault(F_VALUE, "gcrotmk", "the norm of the right side overflows"); }
    if bn == 0.0 { return line_arr(nd_zeros(n)), 0, kept_copy(cu_in), no_fault(); }
    let ld = mt + 1;
    var x: f64[>..] = nd_zeros(n);
    var i: i64 = 0;
    while i < x0.len { x[i] = x0[i]; i += 1; }
    var vs: f64[>..] = nd_zeros(ld * n);
    var zs: f64[>..] = nd_zeros(mt * n);
    var h: f64[>..] = nd_zeros(ld * mt);
    var hr: f64[>..] = nd_zeros(ld * mt);
    var cs: f64[>..] = nd_zeros(mt);
    var sn: f64[>..] = nd_zeros(mt);
    var g: f64[>..] = nd_zeros(ld);
    var w: f64[>..] = nd_zeros(n);
    var y: f64[>..] = nd_zeros(mt);
    var r: f64[>..] = nd_zeros(n);
    var ux: f64[>..] = nd_zeros(n);
    var cx: f64[>..] = nd_zeros(n);
    var tc: f64[>..] = nd_zeros(n);
    var tu: f64[>..] = nd_zeros(n);
    var cc: f64[>..] = nd_zeros(kst * n);
    var uu: f64[>..] = nd_zeros(kst * n);
    var nc_new: f64[>..] = nd_zeros(kst * n);
    var nu_new: f64[>..] = nd_zeros(kst * n);
    var bm: f64[>..] = nd_zeros(mt * kst);
    var no_v: f64[>..] = [];
    // gcrotmk applies the preconditioner on the right, so it needs no scratch.
    var no_pt: f64[>..] = [];
    var tol = rtol * bn;
    if atol > tol { tol = atol; }
    let r0 = lg_residual(s, b, x, r, n);
    if r0 - r0 != 0.0 { return empty_arr(), 0, kept_empty(), fault(F_VALUE, "gcrotmk", "the iterate holds a value that is not finite"); }
    var nc: i64 = 0;
    var jo: i64 = 0;
    if cnt > 0 {
        let nseed, fz = gk_seed(s, cu_in, n, cc, uu, tc, tu, x, r);
        if fz.kind != F_NONE { return empty_arr(), 0, kept_empty(), fz; }
        nc = nseed;
    }
    while jo < maxiter {
        F(x[0..n]);
        var beta = lg_norm(r, 0, n);
        if beta - beta != 0.0 { return empty_arr(), 0, kept_empty(), fault(F_VALUE, "gcrotmk", "the iterate holds a value that is not finite"); }
        if beta <= tol && (jo > 0 || nc > 0) {
            // The update of the residual may drift. The check uses the true residual.
            beta = lg_residual(s, b, x, r, n);
            if beta - beta != 0.0 { return empty_arr(), 0, kept_empty(), fault(F_VALUE, "gcrotmk", "the iterate holds a value that is not finite"); }
        }
        if beta <= tol { return line_arr(copy(x)), 0, gk_pack(cc, uu, nc, n, x, discard_c, want_pairs), no_fault(); }
        var ml = m_eff;
        if k_eff > nc { ml = m_eff + k_eff - nc; }
        i = 0;
        while i < n { vs[i] = r[i] / beta; i += 1; }
        i = 0;
        while i < ld { g[i] = 0.0; i += 1; }
        g[0] = 1.0;
        let used, res, fa = lg_arnoldi(s, n, ml, mt, tol / beta, 0, vs, zs, h, hr, cs, sn, g, w,
                                       no_v, no_v, 0, false, false, cc, nc, bm, kst, mp, pmode, no_pt);
        if fa.kind != F_NONE { return empty_arr(), 0, kept_empty(), fa; }
        lg_solve(hr, g, used, ld, y);
        i = 0;
        while i < used { y[i] = y[i] * beta; i += 1; }
        if !lg_finite(y, used) {
            return empty_arr(), 0, kept_empty(), fault(F_VALUE, "gcrotmk", "the inner solution holds a value that is not finite");
        }
        // ux = (Z - U B) y, and cx = V H y.
        lg_zdir(zs, y, used, n, ux);
        var p: i64 = 0;
        while p < nc {
            var by: f64 = 0.0;
            var c: i64 = 0;
            while c < used { by += bm[c * kst + p] * y[c]; c += 1; }
            i = 0;
            while i < n { ux[i] = ux[i] - by * uu[p * n + i]; i += 1; }
            p += 1;
        }
        lg_hy(h, vs, y, used, ld, n, cx);
        let nxc = lg_norm(cx, 0, n);
        var alpha: f64 = 0.0;
        if nxc > 0.0 { alpha = 1.0 / nxc; }
        // A direction that cannot be normalized is skipped, as SciPy skips it.
        if nxc - nxc == 0.0 && alpha - alpha == 0.0 && alpha != 0.0 {
            i = 0;
            while i < n { cx[i] = cx[i] * alpha; ux[i] = ux[i] * alpha; i += 1; }
            let gamma = lg_dot(cx, 0, r, 0, n);
            i = 0;
            while i < n { r[i] = r[i] - gamma * cx[i]; x[i] = x[i] + gamma * ux[i]; i += 1; }
            if nc >= k_eff {
                if truncate_oldest {
                    let drop = nc - k_eff + 1;
                    kept_drop_oldest(cc, uu, nc, drop, n);
                    nc -= drop;
                } else {
                    // D = B R^-1 by a solve with the triangle of the rotated matrix.
                    var d: f64[>..] = nd_zeros(nc * used);
                    var dmax: f64 = 0.0;
                    i = 0;
                    while i < used {
                        let dd = mathshim::abs_of(hr[i * ld + i]);
                        if dd > dmax { dmax = dd; }
                        i += 1;
                    }
                    var jc: i64 = 0;
                    while jc < used {
                        p = 0;
                        while p < nc {
                            var t = bm[jc * kst + p];
                            var q: i64 = 0;
                            while q < jc { t -= d[p * used + q] * hr[jc * ld + q]; q += 1; }
                            let rd = hr[jc * ld + jc];
                            if rd == 0.0 || mathshim::abs_of(rd) <= KRYLOV_RANK_TOL * dmax {
                                d[p * used + jc] = 0.0;
                            } else {
                                d[p * used + jc] = t / rd;
                            }
                            p += 1;
                        }
                        jc += 1;
                    }
                    let u_svd, sv_svd, vt_svd, fsv = svd(mat2(copy(d), nc, used));
                    if fsv.kind != F_NONE { return empty_arr(), 0, kept_empty(), fsv; }
                    var nnew: i64 = 0;
                    var jd: i64 = 0;
                    while jd < k_eff - 1 {
                        i = 0;
                        while i < n { tc[i] = 0.0; tu[i] = 0.0; i += 1; }
                        p = 0;
                        while p < nc {
                            let wp = u_svd.data[p * nc + jd];
                            i = 0;
                            while i < n {
                                tc[i] += wp * cc[p * n + i];
                                tu[i] += wp * uu[p * n + i];
                                i += 1;
                            }
                            p += 1;
                        }
                        p = 0;
                        while p < nnew {
                            let a = lg_dot(nc_new, p * n, tc, 0, n);
                            i = 0;
                            while i < n {
                                tc[i] = tc[i] - a * nc_new[p * n + i];
                                tu[i] = tu[i] - a * nu_new[p * n + i];
                                i += 1;
                            }
                            p += 1;
                        }
                        let an = lg_norm(tc, 0, n);
                        if an - an == 0.0 && an > 0.0 && 1.0 / an - 1.0 / an == 0.0 {
                            i = 0;
                            while i < n {
                                nc_new[nnew * n + i] = tc[i] / an;
                                nu_new[nnew * n + i] = tu[i] / an;
                                i += 1;
                            }
                            nnew += 1;
                        }
                        jd += 1;
                    }
                    i = 0;
                    while i < nnew * n { cc[i] = nc_new[i]; uu[i] = nu_new[i]; i += 1; }
                    nc = nnew;
                }
            }
            i = 0;
            while i < n { cc[nc * n + i] = cx[i]; uu[nc * n + i] = ux[i]; i += 1; }
            nc += 1;
        }
        jo += 1;
    }
    // The last iteration may have solved the system. The check follows it.
    let rl = lg_residual(s, b, x, r, n);
    if rl - rl != 0.0 { return empty_arr(), 0, kept_empty(), fault(F_VALUE, "gcrotmk", "the iterate holds a value that is not finite"); }
    if rl <= tol { return line_arr(copy(x)), 0, gk_pack(cc, uu, nc, n, x, discard_c, want_pairs), no_fault(); }
    return line_arr(copy(x)), maxiter, gk_pack(cc, uu, nc, n, x, discard_c, want_pairs), no_fault();
}

fn gcrotmk(s, b: f64[:], x0: f64[:] = [], rtol: f64 = 1.0e-5, atol: f64 = 0.0, maxiter: i64 = 1000,
           m: i64 = 20, k: i64 = 0, discard_c: bool = false, truncate_oldest: bool = true,
           precond: Csr = csr_empty()) -> Arr, i64, Fault {
    let x, it, kp, f = gcrotmk_run(s, b, x0, rtol, atol, maxiter, m, k, discard_c, truncate_oldest,
                                   precond, kept_empty(), false) { v => };
    return x, it, f;
}

fn gcrotmk<F>(s, b: f64[:], x0: f64[:] = [], rtol: f64 = 1.0e-5, atol: f64 = 0.0, maxiter: i64 = 1000,
              m: i64 = 20, k: i64 = 0, discard_c: bool = false, truncate_oldest: bool = true,
              precond: Csr = csr_empty()) -> Arr, i64, Fault {
    let x, it, kp, f = gcrotmk_run(s, b, x0, rtol, atol, maxiter, m, k, discard_c, truncate_oldest,
                                   precond, kept_empty(), false) { v => F(v) };
    return x, it, f;
}

// gcrotmk_pairs is gcrotmk with the kept pairs (c, u). The last pair holds
// the solution, with c absent. The call takes the pairs of an earlier solve
// as cu.
fn gcrotmk_pairs(s, b: f64[:], x0: f64[:] = [], rtol: f64 = 1.0e-5, atol: f64 = 0.0, maxiter: i64 = 1000,
                 m: i64 = 20, k: i64 = 0, discard_c: bool = false, truncate_oldest: bool = true,
                 precond: Csr = csr_empty(), cu: Kept = kept_empty()) -> Arr, i64, Kept, Fault {
    return gcrotmk_run(s, b, x0, rtol, atol, maxiter, m, k, discard_c, truncate_oldest,
                       precond, cu, true) { v => };
}
