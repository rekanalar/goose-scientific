// The helpers of the restarted Krylov solvers: the norms, the Arnoldi step and the solution.
//
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

