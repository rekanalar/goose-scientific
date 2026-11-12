// The restarted Krylov solvers lgmres and gcrotmk.
//
// Both calls solve a square system with a matrix of compressed rows. They
// use the matrix only through its product with a vector, so the matrix
// stays sparse. Both give the solution and a count. A count of zero means
// that the residual met the tolerance. A count above zero is the count of
// outer iterations that ran when the iterations ran out.
//
// lgmres is a restarted GMRES that keeps the correction vectors of the
// earlier restarts and adds them to the next Krylov space. The method is
// the LGMRES method of Baker, Jessup and Manteuffel. gcrotmk is the
// GCROT(m,k) method of Hicken and Zingg. It keeps k pairs of vectors across
// the restarts and solves the inner problem with a flexible GMRES.
//
// The package takes no callback, no preconditioner and no list of kept
// vectors. The path of the iteration may differ from SciPy. The solutions
// agree when the tolerance is tight.
namespace scientific;

import mathshim;

// The largest count of outer iterations. A larger count gives a fault of
// kind F_LIMIT.
const MAX_KRYLOV_ITER: i64 = 1000000;

// A diagonal entry below this share of the largest one counts as zero in the
// triangular solve of the inner problem.
const KRYLOV_RANK_TOL: f64 = 1.0e-14;

// lg_norm gives the two-norm of n values that start at off. It divides by
// the largest absolute value first, so no square overflows or underflows.
// A value that is not finite gives that value.
fn lg_norm(v: f64[:], off: i64, n: i64) -> f64 {
    var mx: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        let a = mathshim::abs_of(v[off + i]);
        if a - a != 0.0 { return a; }
        if a > mx { mx = a; }
        i += 1;
    }
    if mx == 0.0 { return 0.0; }
    var t: f64 = 0.0;
    i = 0;
    while i < n {
        let q = v[off + i] / mx;
        t += q * q;
        i += 1;
    }
    return mx * mathshim::root(t);
}

// lg_dot gives the dot product of two runs of n values.
fn lg_dot(a: f64[:], ao: i64, b: f64[:], bo: i64, n: i64) -> f64 {
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < n { t += a[ao + i] * b[bo + i]; i += 1; }
    return t;
}

// lg_finite tells whether n values are all finite.
fn lg_finite(v: f64[:], n: i64) -> bool {
    var i: i64 = 0;
    while i < n {
        let a = v[i];
        if a - a != 0.0 { return false; }
        i += 1;
    }
    return true;
}

// lg_check checks the operands that both calls share. The matrix, the right
// side, the start vector, the tolerances and the count of iterations.
fn lg_check(s, b: f64[:], x0: f64[:], rtol: f64, atol: f64, maxiter: i64, tag: const u8[:]) -> Fault {
    let fm = expm_matrix_check(s, tag);
    if fm.kind != F_NONE { return fm; }
    if s.rows < 1 || b.len != s.rows {
        return fault(F_SHAPE, tag, "the right side does not fit the matrix");
    }
    if x0.len != 0 && x0.len != s.rows {
        return fault(F_SHAPE, tag, "the start vector does not fit the matrix");
    }
    if !lg_finite(b, b.len) {
        return fault(F_VALUE, tag, "the right side holds a value that is not finite");
    }
    if !lg_finite(x0, x0.len) {
        return fault(F_VALUE, tag, "the start vector holds a value that is not finite");
    }
    if !(rtol >= 0.0) || rtol - rtol != 0.0 || !(atol >= 0.0) || atol - atol != 0.0 {
        return fault(F_VALUE, tag, "a tolerance is negative or not finite");
    }
    if maxiter < 1 { return fault(F_RANGE, tag, "the count of iterations is below one"); }
    if maxiter > MAX_KRYLOV_ITER { return fault(F_LIMIT, tag, "the count of iterations stands above the limit"); }
    return no_fault();
}

// lg_storage_check checks the buffers of a call against MAX_ELEMENTS. The
// call holds mt + 1 basis vectors, mt direction vectors, kvec kept vectors
// and 10 work vectors of n values. It also holds four small matrices of
// (mt + 1) by mt values. Each check uses a division, so no product
// overflows. The check runs before any buffer grows.
fn lg_storage_check(n: i64, mt: i64, kvec: i64, tag: const u8[:]) -> Fault {
    let count = 2 * mt + 1 + kvec + 10;
    if count > MAX_ELEMENTS / n {
        return fault(F_LIMIT, tag, "the vectors of the call stand above the element limit");
    }
    if mt + 1 > (MAX_ELEMENTS / 4) / mt {
        return fault(F_LIMIT, tag, "the inner matrices of the call stand above the element limit");
    }
    return no_fault();
}

// lg_arnoldi runs the flexible Arnoldi process with the rotations of
// Givens. The first basis vector stands in vs and has a norm of one. The
// process runs at most ml steps and stops when the residual of the inner
// problem falls below ptol or when the process breaks down.
//
// A step takes a direction z and forms the product w of the matrix with z.
// For lgmres, a direction may be a kept correction vector. The first inner
// steps or the last ones use them, as prepend says. For gcrotmk, nc kept
// vectors c stand in cc, and the step removes their part from w. The part
// goes to the matrix bm with the stride kst.
//
// The call gives the count of the steps and the residual of the inner
// problem. h holds the Hessenberg matrix by columns of mt + 1 values. hr
// holds its rotated copy. g holds the rotated right side.
fn lg_arnoldi(s, n: i64, ml: i64, mt: i64, ptol: f64, inner_m: i64,
              vs: f64[>..]&, zs: f64[>..]&, h: f64[>..]&, hr: f64[>..]&,
              cs: f64[>..]&, sn: f64[>..]&, g: f64[>..]&, w: f64[>..]&,
              ov: f64[:], oav: f64[:], nout: i64, use_av: bool, prepend: bool,
              cc: f64[:], nc: i64, bm: f64[>..]&, kst: i64) -> i64, f64, Fault {
    let ld = mt + 1;
    var j: i64 = 0;
    var used: i64 = 0;
    var res: f64 = 1.0;
    while j < ml {
        var from_outer = false;
        var oi: i64 = 0;
        if nout > 0 {
            if prepend {
                if j < nout { from_outer = true; oi = j; }
            } else {
                if j >= inner_m { from_outer = true; oi = j - inner_m; }
            }
        }
        var i: i64 = 0;
        if from_outer {
            while i < n { zs[j * n + i] = ov[oi * n + i]; i += 1; }
        } else {
            // The first direction after the kept vectors of a prepended
            // run is the first basis vector itself.
            var src: i64 = j;
            if prepend && j == nout { src = 0; }
            while i < n { zs[j * n + i] = vs[src * n + i]; i += 1; }
        }
        if from_outer && use_av {
            i = 0;
            while i < n { w[i] = oav[oi * n + i]; i += 1; }
        } else {
            expm_block_product(s, zs, j * n, n, w, 0);
        }
        let wn = lg_norm(w, 0, n);
        var p: i64 = 0;
        while p < nc {
            let a = lg_dot(cc, p * n, w, 0, n);
            bm[j * kst + p] = a;
            i = 0;
            while i < n { w[i] = w[i] - a * cc[p * n + i]; i += 1; }
            p += 1;
        }
        p = 0;
        while p <= j {
            let a = lg_dot(vs, p * n, w, 0, n);
            h[j * ld + p] = a;
            i = 0;
            while i < n { w[i] = w[i] - a * vs[p * n + i]; i += 1; }
            p += 1;
        }
        let hn = lg_norm(w, 0, n);
        if hn - hn != 0.0 {
            return 0, 0.0, fault(F_VALUE, "krylov", "the iteration produced a value that is not finite");
        }
        h[j * ld + j + 1] = hn;
        let broke = !(hn > KRYLOV_EPS * wn);
        var inv: f64 = 0.0;
        if hn > 0.0 { inv = 1.0 / hn; }
        i = 0;
        if inv - inv == 0.0 && inv != 0.0 {
            while i < n { vs[(j + 1) * n + i] = w[i] * inv; i += 1; }
        } else {
            while i < n { vs[(j + 1) * n + i] = w[i]; i += 1; }
        }
        // The rotations keep the Hessenberg matrix triangular.
        i = 0;
        while i <= j + 1 { hr[j * ld + i] = h[j * ld + i]; i += 1; }
        var k: i64 = 0;
        while k < j {
            let t = cs[k] * hr[j * ld + k] + sn[k] * hr[j * ld + k + 1];
            hr[j * ld + k + 1] = -sn[k] * hr[j * ld + k] + cs[k] * hr[j * ld + k + 1];
            hr[j * ld + k] = t;
            k += 1;
        }
        let a1 = hr[j * ld + j];
        let a2 = hr[j * ld + j + 1];
        let den = mathshim::hypot_of(a1, a2);
        if den > 0.0 {
            cs[j] = a1 / den;
            sn[j] = a2 / den;
        } else {
            cs[j] = 1.0;
            sn[j] = 0.0;
        }
        hr[j * ld + j] = den;
        hr[j * ld + j + 1] = 0.0;
        g[j + 1] = -sn[j] * g[j];
        g[j] = cs[j] * g[j];
        used = j + 1;
        res = mathshim::abs_of(g[j + 1]);
        if res < ptol || broke { break; }
        j += 1;
    }
    let dj = hr[(used - 1) * ld + used - 1];
    if dj - dj != 0.0 {
        return 0, 0.0, fault(F_VALUE, "krylov", "the inner problem holds a value that is not finite");
    }
    return used, res, no_fault();
}

// lg_solve solves the triangular system of the inner problem by back
// substitution. A diagonal entry that is tiny against the largest one gives
// a zero component, as the least-squares solve of SciPy gives for a
// singular system.
fn lg_solve(hr: f64[:], g: f64[:], used: i64, ld: i64, y: f64[>..]&) {
    var dmax: f64 = 0.0;
    var i: i64 = 0;
    while i < used {
        let d = mathshim::abs_of(hr[i * ld + i]);
        if d > dmax { dmax = d; }
        i += 1;
    }
    i = used - 1;
    while i >= 0 {
        var t = g[i];
        var k = i + 1;
        while k < used { t -= hr[k * ld + i] * y[k]; k += 1; }
        let dd = hr[i * ld + i];
        if dd == 0.0 || mathshim::abs_of(dd) <= KRYLOV_RANK_TOL * dmax {
            y[i] = 0.0;
        } else {
            y[i] = t / dd;
        }
        i -= 1;
    }
}

// lg_residual puts b - A x into r and gives its norm.
fn lg_residual(s, b: f64[:], x: f64[:], r: f64[>..]&, n: i64) -> f64 {
    expm_block_product(s, x, 0, n, r, 0);
    var i: i64 = 0;
    while i < n { r[i] = b[i] - r[i]; i += 1; }
    return lg_norm(r, 0, n);
}

// lg_hy puts V (H y) into out. The matrix H holds used + 1 rows and used
// columns. The result is the product of the matrix with the combination of
// the directions that y holds.
fn lg_hy(h: f64[:], vs: f64[:], y: f64[:], used: i64, ld: i64, n: i64, out: f64[>..]&) {
    var i: i64 = 0;
    while i < n { out[i] = 0.0; i += 1; }
    var r: i64 = 0;
    while r <= used {
        var q: f64 = 0.0;
        var c: i64 = 0;
        while c < used { q += h[c * ld + r] * y[c]; c += 1; }
        i = 0;
        while i < n { out[i] += q * vs[r * n + i]; i += 1; }
        r += 1;
    }
}

// lg_zdir puts the combination of the directions that y holds into out.
fn lg_zdir(zs: f64[:], y: f64[:], used: i64, n: i64, out: f64[>..]&) {
    var i: i64 = 0;
    while i < n { out[i] = 0.0; i += 1; }
    var c: i64 = 0;
    while c < used {
        let yc = y[c];
        i = 0;
        while i < n { out[i] += yc * zs[c * n + i]; i += 1; }
        c += 1;
    }
}

// lgmres solves the system A x = b. The call starts from x0, or from zero
// when x0 is empty. The inner size inner_m is the count of Krylov steps of
// one outer iteration. The call keeps the last outer_k correction vectors
// and adds them to the Krylov space of the next iteration. With
// store_outer_av the call keeps the product of the matrix with each
// correction, so the product costs no new call. With prepend_outer_v the
// kept vectors come before the Krylov vectors.
fn lgmres(s, b: f64[:], x0: f64[:] = [], rtol: f64 = 1.0e-5, atol: f64 = 0.0, maxiter: i64 = 1000,
          inner_m: i64 = 30, outer_k: i64 = 3, store_outer_av: bool = true, prepend_outer_v: bool = false) -> Arr, i64, Fault {
    let f = lg_check(s, b, x0, rtol, atol, maxiter, "lgmres");
    if f.kind != F_NONE { return empty_arr(), 0, f; }
    if inner_m < 1 { return empty_arr(), 0, fault(F_RANGE, "lgmres", "the inner size is below one"); }
    if outer_k < 0 { return empty_arr(), 0, fault(F_RANGE, "lgmres", "the count of kept vectors is below zero"); }
    let n = s.rows;
    var m_eff = inner_m;
    if m_eff > n { m_eff = n; }
    var k_eff = outer_k;
    if k_eff > maxiter { k_eff = maxiter; }
    if k_eff > n { k_eff = n; }
    let mt = m_eff + k_eff;
    let fs = lg_storage_check(n, mt, 2 * k_eff, "lgmres");
    if fs.kind != F_NONE { return empty_arr(), 0, fs; }
    let bn = lg_norm(b, 0, n);
    if bn - bn != 0.0 { return empty_arr(), 0, fault(F_VALUE, "lgmres", "the norm of the right side overflows"); }
    if bn == 0.0 { return line_arr(nd_zeros(n)), 0, no_fault(); }
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
    var dx: f64[>..] = nd_zeros(n);
    var ax: f64[>..] = nd_zeros(n);
    var ov: f64[>..] = nd_zeros(k_eff * n);
    var oav: f64[>..] = nd_zeros(k_eff * n);
    var no_cc: f64[>..] = [];
    var no_bm: f64[>..] = [];
    var tol = rtol * bn;
    if atol > tol { tol = atol; }
    var factor: f64 = 1.0;
    var nout: i64 = 0;
    var kout: i64 = 0;
    while kout < maxiter {
        let rn = lg_residual(s, b, x, r, n);
        if rn - rn != 0.0 { return empty_arr(), 0, fault(F_VALUE, "lgmres", "the iterate holds a value that is not finite"); }
        if rn <= tol { return line_arr(copy(x)), 0, no_fault(); }
        i = 0;
        while i < n { vs[i] = r[i] / rn; i += 1; }
        i = 0;
        while i < ld { g[i] = 0.0; i += 1; }
        g[0] = 1.0;
        var ptol = tol / rn;
        if factor < ptol { ptol = factor; }
        let used, res, fa = lg_arnoldi(s, n, m_eff + nout, mt, ptol, m_eff, vs, zs, h, hr, cs, sn, g, w,
                                       ov, oav, nout, store_outer_av, prepend_outer_v, no_cc, 0, no_bm, 1);
        if fa.kind != F_NONE { return empty_arr(), 0, fa; }
        lg_solve(hr, g, used, ld, y);
        i = 0;
        while i < used { y[i] = y[i] * rn; i += 1; }
        if !lg_finite(y, used) {
            return empty_arr(), 0, fault(F_VALUE, "lgmres", "the inner solution holds a value that is not finite");
        }
        if res > ptol {
            factor = 1.5 * factor;
            if factor > 1.0 { factor = 1.0; }
        } else {
            factor = 0.25 * factor;
            if factor < 1.0e-16 { factor = 1.0e-16; }
        }
        lg_zdir(zs, y, used, n, dx);
        let nx = lg_norm(dx, 0, n);
        if nx - nx != 0.0 { return empty_arr(), 0, fault(F_VALUE, "lgmres", "the correction holds a value that is not finite"); }
        if nx > 0.0 && k_eff > 0 {
            if nout == k_eff {
                var slot: i64 = 1;
                while slot < k_eff {
                    i = 0;
                    while i < n {
                        ov[(slot - 1) * n + i] = ov[slot * n + i];
                        oav[(slot - 1) * n + i] = oav[slot * n + i];
                        i += 1;
                    }
                    slot += 1;
                }
                nout -= 1;
            }
            i = 0;
            while i < n { ov[nout * n + i] = dx[i] / nx; i += 1; }
            if store_outer_av {
                lg_hy(h, vs, y, used, ld, n, ax);
                i = 0;
                while i < n { oav[nout * n + i] = ax[i] / nx; i += 1; }
            }
            nout += 1;
        }
        i = 0;
        while i < n { x[i] = x[i] + dx[i]; i += 1; }
        kout += 1;
    }
    // The last iteration may have solved the system. The check follows it.
    let rl = lg_residual(s, b, x, r, n);
    if rl - rl != 0.0 { return empty_arr(), 0, fault(F_VALUE, "lgmres", "the iterate holds a value that is not finite"); }
    if rl <= tol { return line_arr(copy(x)), 0, no_fault(); }
    return line_arr(copy(x)), maxiter, no_fault();
}

// gcrotmk solves the system A x = b by the method GCROT(m,k). The call
// starts from x0, or from zero when x0 is empty. The inner size m is the
// count of Krylov steps of one outer iteration. The call keeps up to k pairs
// of vectors. A k of zero means m. With truncate_oldest set, the call drops
// the oldest pair when it holds k of them. Else it keeps the directions of
// the largest singular values. The flag discard_c has no visible effect,
// because the call keeps no vectors after it returns.
fn gcrotmk(s, b: f64[:], x0: f64[:] = [], rtol: f64 = 1.0e-5, atol: f64 = 0.0, maxiter: i64 = 1000,
           m: i64 = 20, k: i64 = 0, discard_c: bool = false, truncate_oldest: bool = true) -> Arr, i64, Fault {
    let f = lg_check(s, b, x0, rtol, atol, maxiter, "gcrotmk");
    if f.kind != F_NONE { return empty_arr(), 0, f; }
    if m < 1 { return empty_arr(), 0, fault(F_RANGE, "gcrotmk", "the inner size is below one"); }
    if k < 0 { return empty_arr(), 0, fault(F_RANGE, "gcrotmk", "the count of kept pairs is below zero"); }
    let n = s.rows;
    var kk = k;
    if kk == 0 { kk = m; }
    var m_eff = m;
    if m_eff > n { m_eff = n; }
    var k_eff = kk;
    if k_eff > maxiter { k_eff = maxiter; }
    if k_eff > n { k_eff = n; }
    let mt = m_eff + k_eff;
    let fs = lg_storage_check(n, mt, 4 * k_eff, "gcrotmk");
    if fs.kind != F_NONE { return empty_arr(), 0, fs; }
    let bn = lg_norm(b, 0, n);
    if bn - bn != 0.0 { return empty_arr(), 0, fault(F_VALUE, "gcrotmk", "the norm of the right side overflows"); }
    if bn == 0.0 { return line_arr(nd_zeros(n)), 0, no_fault(); }
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
    var cc: f64[>..] = nd_zeros(k_eff * n);
    var uu: f64[>..] = nd_zeros(k_eff * n);
    var nc_new: f64[>..] = nd_zeros(k_eff * n);
    var nu_new: f64[>..] = nd_zeros(k_eff * n);
    var bm: f64[>..] = nd_zeros(mt * k_eff);
    var no_v: f64[>..] = [];
    var tol = rtol * bn;
    if atol > tol { tol = atol; }
    let r0 = lg_residual(s, b, x, r, n);
    if r0 - r0 != 0.0 { return empty_arr(), 0, fault(F_VALUE, "gcrotmk", "the iterate holds a value that is not finite"); }
    var nc: i64 = 0;
    var jo: i64 = 0;
    while jo < maxiter {
        var beta = lg_norm(r, 0, n);
        if beta - beta != 0.0 { return empty_arr(), 0, fault(F_VALUE, "gcrotmk", "the iterate holds a value that is not finite"); }
        if beta <= tol && (jo > 0 || nc > 0) {
            // The update of the residual may drift. The check uses the true residual.
            beta = lg_residual(s, b, x, r, n);
            if beta - beta != 0.0 { return empty_arr(), 0, fault(F_VALUE, "gcrotmk", "the iterate holds a value that is not finite"); }
        }
        if beta <= tol { return line_arr(copy(x)), 0, no_fault(); }
        var ml = m_eff;
        if k_eff > nc { ml = m_eff + k_eff - nc; }
        i = 0;
        while i < n { vs[i] = r[i] / beta; i += 1; }
        i = 0;
        while i < ld { g[i] = 0.0; i += 1; }
        g[0] = 1.0;
        let used, res, fa = lg_arnoldi(s, n, ml, mt, tol / beta, 0, vs, zs, h, hr, cs, sn, g, w,
                                       no_v, no_v, 0, false, false, cc, nc, bm, k_eff);
        if fa.kind != F_NONE { return empty_arr(), 0, fa; }
        lg_solve(hr, g, used, ld, y);
        i = 0;
        while i < used { y[i] = y[i] * beta; i += 1; }
        if !lg_finite(y, used) {
            return empty_arr(), 0, fault(F_VALUE, "gcrotmk", "the inner solution holds a value that is not finite");
        }
        // ux = (Z - U B) y, and cx = V H y.
        lg_zdir(zs, y, used, n, ux);
        var p: i64 = 0;
        while p < nc {
            var by: f64 = 0.0;
            var c: i64 = 0;
            while c < used { by += bm[c * k_eff + p] * y[c]; c += 1; }
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
                    p = 0;
                    while p + drop < nc {
                        i = 0;
                        while i < n {
                            cc[p * n + i] = cc[(p + drop) * n + i];
                            uu[p * n + i] = uu[(p + drop) * n + i];
                            i += 1;
                        }
                        p += 1;
                    }
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
                            var t = bm[jc * k_eff + p];
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
                    if fsv.kind != F_NONE { return empty_arr(), 0, fsv; }
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
    if rl - rl != 0.0 { return empty_arr(), 0, fault(F_VALUE, "gcrotmk", "the iterate holds a value that is not finite"); }
    if rl <= tol { return line_arr(copy(x)), 0, no_fault(); }
    return line_arr(copy(x)), maxiter, no_fault();
}
