// The blocks, the triangles and the slices of sparse matrices, and lsqr, minres and eigsh.
//
namespace scientific;

import mathshim;

// csr_block_diag places two sparse matrices on the diagonal of a larger
// one, as block_diag of SciPy does.
fn csr_block_diag(a, b) -> Csr, Fault {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < a.rows + b.rows {
        var p: i64 = 0;
        var end: i64 = 0;
        if i < a.rows { p = csr_ptr(a, i); end = csr_ptr(a, i + 1); }
        else { p = csr_ptr(b, i - a.rows); end = csr_ptr(b, i - a.rows + 1); }
        while p < end {
            ri.push(i as f64);
            if i < a.rows { ci.push(csr_col(a, p) as f64); v.push(csr_val(a, p)); }
            else { ci.push((a.cols + csr_col(b, p)) as f64); v.push(csr_val(b, p)); }
            p += 1;
        }
        i += 1;
    }
    return csr_from_triplets(a.rows + b.rows, a.cols + b.cols, ri, ci, v);
}

// csr_part keeps the stored values of the lower part (lower true) or
// the upper part, from the diagonal k. The diagonal k holds the values
// with column minus row equal to k.
fn csr_part(a, k: i64, lower: bool) -> Csr, Fault {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < a.rows {
        var p = csr_ptr(a, i);
        while p < csr_ptr(a, i + 1) {
            let j = csr_col(a, p);
            if (lower && j - i <= k) || (!lower && j - i >= k) {
                ri.push(i as f64);
                ci.push(j as f64);
                v.push(csr_val(a, p));
            }
            p += 1;
        }
        i += 1;
    }
    return csr_from_triplets(a.rows, a.cols, ri, ci, v);
}

// csr_tril gives the lower triangle of a sparse matrix up to the
// diagonal k, as tril of SciPy does.
fn csr_tril(a, k: i64 = 0) -> Csr, Fault { return csr_part(a, k, true); }

// csr_triu gives the upper triangle of a sparse matrix from the
// diagonal k, as triu of SciPy does.
fn csr_triu(a, k: i64 = 0) -> Csr, Fault { return csr_part(a, k, false); }

// csr_multiply gives the product of two sparse matrices of one shape,
// value by value, as multiply of SciPy does.
fn csr_multiply(a, b) -> Csr, Fault {
    if a.rows != b.rows || a.cols != b.cols {
        return csr_empty(), fault(F_SHAPE, "csr_multiply", "the matrices differ in shape");
    }
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < a.rows {
        var p = csr_ptr(a, i);
        var q = csr_ptr(b, i);
        let pe = csr_ptr(a, i + 1);
        let qe = csr_ptr(b, i + 1);
        while p < pe && q < qe {
            let ja = csr_col(a, p);
            let jb = csr_col(b, q);
            if ja == jb {
                ri.push(i as f64);
                ci.push(ja as f64);
                v.push(csr_val(a, p) * csr_val(b, q));
                p += 1;
                q += 1;
            } else if ja < jb { p += 1; } else { q += 1; }
        }
        i += 1;
    }
    return csr_from_triplets(a.rows, a.cols, ri, ci, v);
}

// csr_slice gives the block of rows r0 to r1 and columns c0 to c1, with
// the end excluded, as the slice a[r0:r1, c0:c1] of SciPy does.
fn csr_slice(a, r0: i64, r1: i64, c0: i64, c1: i64) -> Csr, Fault {
    if r0 < 0 || c0 < 0 || r1 > a.rows || c1 > a.cols || r0 >= r1 || c0 >= c1 {
        return csr_empty(), fault(F_RANGE, "csr_slice", "the slice stands outside the matrix or holds no element");
    }
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = r0;
    while i < r1 {
        var p = csr_ptr(a, i);
        while p < csr_ptr(a, i + 1) {
            let j = csr_col(a, p);
            if j >= c0 && j < c1 {
                ri.push((i - r0) as f64);
                ci.push((j - c0) as f64);
                v.push(csr_val(a, p));
            }
            p += 1;
        }
        i += 1;
    }
    return csr_from_triplets(r1 - r0, c1 - c0, ri, ci, v);
}

// lsqr solves the least squares problem of a sparse matrix, of any
// shape, by the method of Paige and Saunders. It starts from zero, so
// a system with more unknowns than equations gives the solution of the
// least norm. It stops when the residual norm is not above rtol times
// the norm of b, or when the gradient of the residual is small. It
// gives x and the info: zero for success, or the count of turns when
// the limit of maxiter turns ran out.
fn lsqr(s, b: f64[:], rtol: f64 = 1.0e-8, maxiter: i64 = 0) -> Arr, i64, Fault {
    if b.len != s.rows || b.len < 1 {
        return empty_arr(), 0, fault(F_SHAPE, "lsqr", "the right side does not fit the matrix");
    }
    if !(rtol >= 0.0) {
        return empty_arr(), 0, fault(F_VALUE, "lsqr", "the tolerance holds a negative value");
    }
    let m = s.rows;
    let n = s.cols;
    let st = csr_transpose(s);
    var limit = maxiter;
    if limit < 1 { limit = 2 * n; }
    var u: f64[>..] = [];
    var i: i64 = 0;
    while i < m { u.push(b[i]); i += 1; }
    let bn = krylov_norm(b);
    var x: f64[>..] = [];
    i = 0;
    while i < n { x.push(0.0); i += 1; }
    if bn == 0.0 { return line_arr(copy(x)), 0, no_fault(); }
    i = 0;
    while i < m { u[i] = u[i] / bn; i += 1; }
    let v0, f0 = csr_matvec(st, u);
    if f0.kind != F_NONE { return empty_arr(), 0, f0; }
    var v: f64[>..] = [];
    i = 0;
    while i < n { v.push(v0.data[i]); i += 1; }
    var alpha = krylov_norm(v);
    if alpha == 0.0 { return line_arr(copy(x)), 0, no_fault(); }
    var w: f64[>..] = [];
    i = 0;
    while i < n { v[i] = v[i] / alpha; w.push(v[i]); i += 1; }
    var phibar = bn;
    var rhobar = alpha;
    var turn: i64 = 0;
    while turn < limit {
        let av, f1 = csr_matvec(s, v);
        if f1.kind != F_NONE { return empty_arr(), 0, f1; }
        i = 0;
        while i < m { u[i] = av.data[i] - alpha * u[i]; i += 1; }
        let beta = krylov_norm(u);
        if beta > 0.0 {
            i = 0;
            while i < m { u[i] = u[i] / beta; i += 1; }
            let atu, f2 = csr_matvec(st, u);
            if f2.kind != F_NONE { return empty_arr(), 0, f2; }
            i = 0;
            while i < n { v[i] = atu.data[i] - beta * v[i]; i += 1; }
            alpha = krylov_norm(v);
            if alpha > 0.0 {
                i = 0;
                while i < n { v[i] = v[i] / alpha; i += 1; }
            }
        }
        let rho = mathshim::root(rhobar * rhobar + beta * beta);
        let c = rhobar / rho;
        let sn = beta / rho;
        let theta = sn * alpha;
        rhobar = -c * alpha;
        let phi = c * phibar;
        phibar = sn * phibar;
        i = 0;
        while i < n {
            x[i] = x[i] + (phi / rho) * w[i];
            w[i] = v[i] - (theta / rho) * w[i];
            i += 1;
        }
        turn += 1;
        // The test of the residual, and the test of the gradient.
        let grad = phibar * alpha * mathshim::abs_of(c);
        if phibar <= rtol * bn || grad <= rtol * phibar * (1.0 + alpha) || beta == 0.0 || alpha == 0.0 {
            return line_arr(copy(x)), 0, no_fault();
        }
    }
    return line_arr(copy(x)), turn, no_fault();
}

// minres solves a x = b for a symmetric sparse matrix, which may be
// indefinite, by the method of Paige and Saunders. It starts from zero
// and stops when the estimate of the residual norm is not above rtol
// times the norm of b. It gives x and the info: zero for success, or
// the count of turns when the limit of maxiter turns ran out. The
// matrix must be symmetric. The method does not check that.
fn minres(s, b: f64[:], rtol: f64 = 1.0e-5, maxiter: i64 = 0) -> Arr, i64, Fault {
    if s.rows != s.cols || b.len != s.rows || b.len < 1 {
        return empty_arr(), 0, fault(F_SHAPE, "minres", "the matrix is not square or the right side does not fit");
    }
    if !(rtol >= 0.0) {
        return empty_arr(), 0, fault(F_VALUE, "minres", "the tolerance holds a negative value");
    }
    let n = b.len;
    var limit = maxiter;
    if limit < 1 { limit = 5 * n; }
    var x: f64[>..] = [];
    var y: f64[>..] = [];
    var r1: f64[>..] = [];
    var r2: f64[>..] = [];
    var w: f64[>..] = [];
    var w2: f64[>..] = [];
    var i: i64 = 0;
    while i < n { x.push(0.0); y.push(b[i]); r1.push(b[i]); r2.push(b[i]); w.push(0.0); w2.push(0.0); i += 1; }
    let beta1 = krylov_norm(b);
    if beta1 == 0.0 { return line_arr(copy(x)), 0, no_fault(); }
    var oldb: f64 = 0.0;
    var beta = beta1;
    var dbar: f64 = 0.0;
    var epsln: f64 = 0.0;
    var phibar = beta1;
    var cs: f64 = -1.0;
    var sn: f64 = 0.0;
    var turn: i64 = 0;
    while turn < limit {
        let sc = 1.0 / beta;
        var v: f64[>..] = [];
        i = 0;
        while i < n { v.push(sc * y[i]); i += 1; }
        let av, f1 = csr_matvec(s, v);
        if f1.kind != F_NONE { return empty_arr(), 0, f1; }
        i = 0;
        while i < n { y[i] = av.data[i]; i += 1; }
        if turn >= 1 {
            i = 0;
            while i < n { y[i] = y[i] - (beta / oldb) * r1[i]; i += 1; }
        }
        var alfa: f64 = 0.0;
        i = 0;
        while i < n { alfa += v[i] * y[i]; i += 1; }
        i = 0;
        while i < n { y[i] = y[i] - (alfa / beta) * r2[i]; i += 1; }
        i = 0;
        while i < n { r1[i] = r2[i]; r2[i] = y[i]; i += 1; }
        oldb = beta;
        beta = krylov_norm(y);
        let oldeps = epsln;
        let delta = cs * dbar + sn * alfa;
        let gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;
        let gamma = mathshim::root(gbar * gbar + beta * beta);
        if gamma == 0.0 {
            return empty_arr(), 0, fault(F_VALUE, "minres", "the matrix is singular");
        }
        cs = gbar / gamma;
        sn = beta / gamma;
        let phi = cs * phibar;
        phibar = sn * phibar;
        let denom = 1.0 / gamma;
        i = 0;
        while i < n {
            let w1 = w2[i];
            w2[i] = w[i];
            w[i] = (v[i] - oldeps * w1 - delta * w2[i]) * denom;
            x[i] = x[i] + phi * w[i];
            i += 1;
        }
        turn += 1;
        if mathshim::abs_of(phibar) <= rtol * beta1 || beta == 0.0 { return line_arr(copy(x)), 0, no_fault(); }
    }
    return line_arr(copy(x)), turn, no_fault();
}

// eigsh gives k eigenvalues and eigenvectors of a symmetric sparse
// matrix by the Lanczos method with a full reorthogonalization. The
// choice which is "LA" for the largest values, "SA" for the smallest
// values, "LM" for the largest sizes, and "SM" for the smallest sizes.
// The values come in rising order. The vectors are the columns of the
// second result. The matrix must be symmetric; the method does not
// check that. The method grows the Krylov space until the residuals
// are small. A space above 256 vectors is a fault.
fn eigsh(s, k: i64 = 6, which: const u8[:] = "LM", tol: f64 = 1.0e-10) -> Arr, Arr, Fault {
    if s.rows != s.cols || s.rows < 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "eigsh", "the matrix is not square or holds fewer than two rows");
    }
    let n = s.rows;
    if k < 1 || k >= n {
        return empty_arr(), empty_arr(), fault(F_RANGE, "eigsh", "k must stand from one to below the side");
    }
    if which != "LA" && which != "SA" && which != "LM" && which != "SM" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "eigsh", "which holds an unknown name");
    }
    if !(tol > 0.0) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "eigsh", "the tolerance must be above zero");
    }
    var r = default_rng(1);
    var start: f64[>..] = [];
    var i: i64 = 0;
    while i < n { start.push(random_uniform(r) - 0.5); i += 1; }
    var m = 2 * k + 20;
    if m > n { m = n; }
    while true {
        // The Lanczos steps. V holds the vectors one after another.
        var vs: f64[>..] = [];
        var al: f64[>..] = [];
        var be: f64[>..] = [];
        var nv = krylov_norm(start);
        i = 0;
        while i < n { vs.push(start[i] / nv); i += 1; }
        var steps: i64 = 0;
        var broke = false;
        while steps < m {
            var cur: f64[>..] = [];
            i = 0;
            while i < n { cur.push(vs[steps * n + i]); i += 1; }
            let av, fa = csr_matvec(s, cur);
            if fa.kind != F_NONE { return empty_arr(), empty_arr(), fa; }
            var w: f64[>..] = [];
            i = 0;
            while i < n { w.push(av.data[i]); i += 1; }
            var a: f64 = 0.0;
            i = 0;
            while i < n { a += cur[i] * w[i]; i += 1; }
            al.push(a);
            var pass: i64 = 0;
            while pass < 2 {
                var j: i64 = 0;
                while j <= steps {
                    var d: f64 = 0.0;
                    i = 0;
                    while i < n { d += vs[j * n + i] * w[i]; i += 1; }
                    i = 0;
                    while i < n { w[i] = w[i] - d * vs[j * n + i]; i += 1; }
                    j += 1;
                }
                pass += 1;
            }
            let b = krylov_norm(w);
            steps += 1;
            if steps == m { be.push(b); break; }
            if b <= 1.0e-12 * (1.0 + mathshim::abs_of(a)) { be.push(0.0); broke = true; break; }
            be.push(b);
            i = 0;
            while i < n { vs.push(w[i] / b); i += 1; }
        }
        // The tridiagonal matrix of the steps.
        var td: f64[>..] = [];
        i = 0;
        while i < steps * steps { td.push(0.0); i += 1; }
        i = 0;
        while i < steps {
            td[i * steps + i] = al[i];
            if i + 1 < steps { td[i * steps + i + 1] = be[i]; td[(i + 1) * steps + i] = be[i]; }
            i += 1;
        }
        let tv, tz, tf = eigh(mat2(copy(td), steps, steps));
        if tf.kind != F_NONE { return empty_arr(), empty_arr(), tf; }
        if steps < k {
            return empty_arr(), empty_arr(), fault(F_VALUE, "eigsh", "the Krylov space holds fewer than k vectors");
        }
        // The wanted Ritz values, by index into the rising list.
        var pick: i64[>..] = [];
        var lo: i64 = 0;
        var hi = steps - 1;
        while pick.len < k {
            var take_hi = false;
            if which == "LA" { take_hi = true; }
            else if which == "LM" { take_hi = mathshim::abs_of(tv.data[hi]) >= mathshim::abs_of(tv.data[lo]); }
            else if which == "SM" { take_hi = mathshim::abs_of(tv.data[hi]) < mathshim::abs_of(tv.data[lo]); }
            if take_hi { pick.push(hi); hi -= 1; } else { pick.push(lo); lo += 1; }
        }
        var worst: f64 = 0.0;
        var scale: f64 = 1.0e-300;
        i = 0;
        while i < k {
            let res = mathshim::abs_of(be[steps - 1] * tz.data[(steps - 1) * steps + pick[i]]);
            if res > worst { worst = res; }
            let mag = mathshim::abs_of(tv.data[pick[i]]);
            if mag > scale { scale = mag; }
            i += 1;
        }
        if broke || steps >= n || worst <= tol * scale {
            // Sort the picked indices into rising order of the value.
            var order: i64[>..] = [];
            i = 0;
            while i < k { order.push(pick[i]); i += 1; }
            var x: i64 = 1;
            while x < k {
                var y = x;
                while y > 0 && order[y - 1] > order[y] {
                    let t = order[y];
                    order[y] = order[y - 1];
                    order[y - 1] = t;
                    y -= 1;
                }
                x += 1;
            }
            var vals: f64[>..] = [];
            var vecs: f64[>..] = [];
            i = 0;
            while i < n * k { vecs.push(0.0); i += 1; }
            var c: i64 = 0;
            while c < k {
                vals.push(tv.data[order[c]]);
                var j: i64 = 0;
                while j < steps {
                    let coef = tz.data[j * steps + order[c]];
                    i = 0;
                    while i < n { vecs[i * k + c] = vecs[i * k + c] + coef * vs[j * n + i]; i += 1; }
                    j += 1;
                }
                c += 1;
            }
            return line_arr(copy(vals)), mat2(copy(vecs), n, k), no_fault();
        }
        if m >= 256 || m >= n {
            return empty_arr(), empty_arr(), fault(F_LIMIT, "eigsh", "the method did not converge within 256 vectors");
        }
        m = 2 * m;
        if m > n { m = n; }
        if m > 256 { m = 256; }
    }
    return empty_arr(), empty_arr(), no_fault();
}
