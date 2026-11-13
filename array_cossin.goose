// The cosine-sine decomposition: the checks of the factors and the layout of the blocks.
//
// The cosine-sine decomposition of a real orthogonal matrix. The
// calls follow scipy.linalg.cossin. The matrix X splits at the row p
// and at the column q. It then equals U * CS * Vh, where U and Vh are
// block diagonal orthogonal matrices and CS holds the cosines and the
// sines of the angles.
//
// LAPACK reaches the factors by bidiagonalization. This file reaches
// them by the singular value decomposition of the first block. The
// angles and the matrix CS equal those of SciPy. The factors equal
// them up to the sign of each paired column and row.
namespace scientific;

import mathshim;

// The largest entry of X^T X - I that a matrix may hold and still
// count as orthogonal.
const CX_ORTHO_TOL: f64 = 1.0e-8;

// A sine or a cosine below this value counts as zero.
const CX_ZERO: f64 = 1.0e-13;

// A cosine inside this distance of one belongs to a cluster.
const CX_CLUSTER: f64 = 1.0e-6;

const CX_HALF_PI: f64 = 1.5707963267948966;

// cx_same tells whether two shapes are equal.
fn cx_same(a: i64[:], b: i64[:]) -> bool {
    if a.len != b.len { return false; }
    var i: i64 = 0;
    while i < a.len {
        if a[i] != b[i] { return false; }
        i += 1;
    }
    return true;
}

fn cx_min(a: i64, b: i64) -> i64 {
    if a < b { return a; }
    return b;
}

// cx_orthogonal gives the largest entry of X^T X - I of one matrix of
// side m that starts at the place off.
fn cx_orthogonal(x: f64[:], off: i64, m: i64) -> f64 {
    var worst: f64 = 0.0;
    var i: i64 = 0;
    while i < m {
        var j: i64 = i;
        while j < m {
            var s: f64 = 0.0;
            var k: i64 = 0;
            while k < m { s += x[off + k * m + i] * x[off + k * m + j]; k += 1; }
            if i == j { s -= 1.0; }
            if s < 0.0 { s = -s; }
            if s > worst { worst = s; }
            j += 1;
        }
        i += 1;
    }
    return worst;
}

// cx_add makes the vector v orthogonal to the vectors of the basis.
// It runs the step twice. It then appends the unit vector to the
// basis when the length that remains exceeds minres. The basis holds
// its vectors one after the other, n values each.
fn cx_add(basis: f64[>..]&, n: i64, v: f64[:], minres: f64) -> bool {
    var w: f64[>..] = [];
    var i: i64 = 0;
    while i < n { w.push(v[i]); i += 1; }
    let count = basis.len / n;
    var pass: i64 = 0;
    while pass < 2 {
        var k: i64 = 0;
        while k < count {
            var t: f64 = 0.0;
            i = 0;
            while i < n { t += basis[k * n + i] * w[i]; i += 1; }
            i = 0;
            while i < n { w[i] = w[i] - t * basis[k * n + i]; i += 1; }
            k += 1;
        }
        pass += 1;
    }
    let len = qu_norm(w, 0, n);
    if !(len > minres) { return false; }
    i = 0;
    while i < n { basis.push(w[i] / len); i += 1; }
    return true;
}

// cx_free appends one unit vector that the basis does not span. It
// takes the unit vectors e_k in turn. The cursor holds the next k. A
// vector that fails once fails for good, because the span only grows.
fn cx_free(basis: f64[>..]&, n: i64, cur: i64[>..]&) -> bool {
    let thr = 0.5 / (n * 1.0);
    while cur[0] < n {
        let k = cur[0];
        cur[0] = k + 1;
        var e: f64[>..] = [];
        var i: i64 = 0;
        while i < n { e.push(0.0); i += 1; }
        e[k] = 1.0;
        if cx_add(basis, n, e, thr) { return true; }
    }
    return false;
}

// cx_orth builds the n columns of an orthogonal matrix. The candidates
// hold vectors of n values. A candidate that is marked usable and that
// stays independent keeps its place. The order lists the columns of
// the result: the number of a candidate, or -1 for a free column. A
// free column takes a unit vector that the other columns do not span.
// The result holds its columns one after the other. An empty result
// means that the completion failed.
fn cx_orth(n: i64, cand: f64[:], ok: i64[:], order: i64[:], backward: bool) -> f64[>..] {
    var basis: f64[>..] = [];
    var placed: i64[>..] = [];
    var c: i64 = 0;
    while c < ok.len { placed.push(-1); c += 1; }
    // The candidate that the data fixes best goes first. A candidate
    // with a weak weight goes last, so its error stays in its own
    // column and no later column inherits it.
    var step: i64 = 0;
    while step < ok.len {
        c = step;
        if backward { c = ok.len - 1 - step; }
        if ok[c] == 1 {
            if cx_add(basis, n, cand[c * n..(c + 1) * n], 1.0e-6) { placed[c] = basis.len / n - 1; }
        }
        step += 1;
    }
    // The columns that no candidate fills need free vectors.
    var need: i64 = 0;
    var t: i64 = 0;
    while t < order.len {
        if order[t] < 0 || placed[order[t]] < 0 { need += 1; }
        t += 1;
    }
    let fixed = basis.len / n;
    var cur: i64[>..] = [0];
    var f: i64 = 0;
    while f < need {
        if !cx_free(basis, n, cur) { return []; }
        f += 1;
    }
    var out: f64[>..] = [];
    var next: i64 = fixed;
    t = 0;
    while t < order.len {
        var from: i64 = next;
        if order[t] >= 0 && placed[order[t]] >= 0 {
            from = placed[order[t]];
        } else {
            next += 1;
        }
        var i: i64 = 0;
        while i < n { out.push(basis[from * n + i]); i += 1; }
        t += 1;
    }
    return out;
}

// cx_layout gives the cosine-sine matrix of the angles. The layout and
// the signs follow the source of scipy.linalg.cossin.
fn cx_layout(theta: f64[:], m: i64, p: i64, q: i64, swap: bool) -> f64[>..] {
    let r = cx_min(cx_min(p, q), cx_min(m - p, m - q));
    let n11 = cx_min(p, q) - r;
    let n12 = cx_min(p, m - q) - r;
    let n21 = cx_min(m - p, q) - r;
    let n22 = cx_min(m - p, m - q) - r;
    var cs: f64[>..] = [];
    var i: i64 = 0;
    while i < m * m { cs.push(0.0); i += 1; }
    var s1: f64 = -1.0;
    var s2: f64 = 1.0;
    if swap { s1 = 1.0; s2 = -1.0; }
    i = 0;
    while i < n11 { cs[i * m + i] = 1.0; i += 1; }
    i = 0;
    while i < n12 { cs[(n11 + r + i) * m + (n11 + n21 + n22 + 2 * r + i)] = s1; i += 1; }
    i = 0;
    while i < n21 { cs[(p + n22 + r + i) * m + (n11 + r + i)] = s2; i += 1; }
    i = 0;
    while i < n22 { cs[(p + i) * m + (q + i)] = 1.0; i += 1; }
    i = 0;
    while i < r {
        let c = mathshim::cosine(theta[i]);
        let s = mathshim::sine(theta[i]);
        cs[(n11 + i) * m + (n11 + i)] = c;
        cs[(p + n22 + i) * m + (n11 + r + n21 + n22 + i)] = c;
        cs[(n11 + i) * m + (n11 + n21 + n22 + r + i)] = s1 * s;
        cs[(p + n22 + i) * m + (n11 + i)] = s2 * s;
        i += 1;
    }
    return cs;
}

// cx_svd_right gives the right vectors of a matrix of r rows and c
// columns, and its singular values. The vectors come one after the
// other, so the vector j lies at the index of the component times the
// count of the columns plus the vector.
//
// A wide matrix holds more columns than rows. The one-sided Jacobi
// cannot make more vectors orthogonal than the count of the rows, so it
// wastes its sweeps on a wide matrix. The transpose of a wide matrix is
// tall, and the left vectors of the transpose are the right vectors of
// the matrix.
fn cx_svd_right(d: f64[>..]&, r: i64, c: i64) -> f64[>..], f64[>..], Fault {
    var v: f64[>..] = [];
    var vals: f64[>..] = [];
    if c <= r {
        let sv, svec, f = jacobi_svd(mat2(copy(d), r, c), r, c);
        if f.kind != F_NONE { return v, vals, f; }
        vals = copy(sv.data);
        var i: i64 = 0;
        while i < c * c { v.push(svec.data[i]); i += 1; }
        return v, vals, no_fault();
    }
    // The transpose of the matrix, a tall matrix of c rows.
    var dt: f64[>..] = [];
    var i: i64 = 0;
    while i < c {
        var j: i64 = 0;
        while j < r { dt.push(d[j * c + i]); j += 1; }
        i += 1;
    }
    let sv, svec, f = jacobi_svd(mat2(copy(dt), c, r), c, r);
    if f.kind != F_NONE { return v, vals, f; }
    vals = copy(sv.data);
    // The right vectors of the matrix are the left vectors of its
    // transpose. The product of the transpose and its right vectors
    // gives them.
    var tol: f64 = 0.0;
    if r > 0 { tol = sv.data[0] * 1.0e-12; }
    var valid: i64 = 0;
    while valid < r && sv.data[valid] > tol { valid += 1; }
    var cols: f64[>..] = [];
    i = 0;
    while i < c * c { cols.push(0.0); i += 1; }
    var j: i64 = 0;
    while j < valid {
        var k: i64 = 0;
        while k < c {
            var s: f64 = 0.0;
            var t: i64 = 0;
            while t < r { s += dt[k * r + t] * svec.data[j * r + t]; t += 1; }
            cols[k * c + j] = s / sv.data[j];
            k += 1;
        }
        j += 1;
    }
    // The vectors that the singular values do not reach complete the
    // null space of the matrix.
    let full = complete_basis(cols, c, valid, c);
    i = 0;
    while i < c {
        var k: i64 = 0;
        while k < c { v.push(full[k * c + i]); k += 1; }
        i += 1;
    }
    return v, vals, no_fault();
}

// cx_matrix decomposes one matrix of side m that starts at the place
// off of the buffer x. It appends the angles, U, CS and Vh to the four
// buffers. It gives false when a completion fails.
fn cx_matrix(x: f64[:], off: i64, m: i64, p: i64, q: i64, swap: bool, th: f64[>..]&, uo: f64[>..]&, cso: f64[>..]&, vho: f64[>..]&) -> bool {
    let mp = m - p;
    let mq = m - q;
    let r = cx_min(cx_min(p, q), cx_min(mp, mq));
    let k1 = cx_min(p, q);
    let n11 = k1 - r;
    let n21 = cx_min(mp, q) - r;
    let n22 = cx_min(mp, mq) - r;
    // The two blocks of the first block column.
    var a: f64[>..] = [];
    var b: f64[>..] = [];
    var i: i64 = 0;
    while i < p { var j: i64 = 0; while j < q { a.push(x[off + i * m + j]); j += 1; } i += 1; }
    i = 0;
    while i < mp { var j: i64 = 0; while j < q { b.push(x[off + (p + i) * m + j]); j += 1; } i += 1; }
    // The right vectors of the first block, one after the other.
    let v, vals, fs = cx_svd_right(a, p, q);
    if fs.kind != F_NONE { return false; }
    // A cluster of cosines near one hides its sines. The thin matrix
    // of the cluster holds the sines. Its right vectors rotate the
    // cluster, and the order of the cosines holds.
    var kc: i64 = 0;
    while kc < vals.len && vals[kc] > 1.0 - CX_CLUSTER { kc += 1; }
    if kc > 1 {
        var wk: f64[>..] = [];
        i = 0;
        while i < mp {
            var c: i64 = 0;
            while c < kc {
                var s: f64 = 0.0;
                var k: i64 = 0;
                while k < q { s += b[i * q + k] * v[c * q + k]; k += 1; }
                wk.push(s);
                c += 1;
            }
            i += 1;
        }
        let vecs2, _wv, f2 = cx_svd_right(wk, mp, kc);
        if f2.kind != F_NONE { return false; }
        var rot: f64[>..] = [];
        var t: i64 = 0;
        while t < kc {
            // The new order runs from the smallest sine up.
            let src = kc - 1 - t;
            var k: i64 = 0;
            while k < q {
                var s: f64 = 0.0;
                var c: i64 = 0;
                while c < kc { s += vecs2[src * kc + c] * v[c * q + k]; c += 1; }
                rot.push(s);
                k += 1;
            }
            t += 1;
        }
        i = 0;
        while i < kc * q { v[i] = rot[i]; i += 1; }
    }
    // The first block times a vector, and the second block times it.
    var ua: f64[>..] = [];
    var wb: f64[>..] = [];
    var cj: f64[>..] = [];
    var sj: f64[>..] = [];
    var j: i64 = 0;
    while j < q {
        i = 0;
        while i < p {
            var s: f64 = 0.0;
            var k: i64 = 0;
            while k < q { s += a[i * q + k] * v[j * q + k]; k += 1; }
            ua.push(s);
            i += 1;
        }
        i = 0;
        while i < mp {
            var s: f64 = 0.0;
            var k: i64 = 0;
            while k < q { s += b[i * q + k] * v[j * q + k]; k += 1; }
            wb.push(s);
            i += 1;
        }
        cj.push(qu_norm(ua, j * p, p));
        sj.push(qu_norm(wb, j * mp, mp));
        j += 1;
    }
    // The angles. A sine near zero gives the angle zero and a free
    // column of U2. A cosine near zero gives a free column of U1.
    var theta: f64[>..] = [];
    var ok1: i64[>..] = [];
    var ok2: i64[>..] = [];
    var cand1: f64[>..] = [];
    var cand2: f64[>..] = [];
    j = 0;
    while j < k1 {
        // The candidates of U1 follow the first k1 vectors.
        if cj[j] > CX_ZERO {
            ok1.push(1);
            i = 0;
            while i < p { cand1.push(ua[j * p + i] / cj[j]); i += 1; }
        } else {
            ok1.push(0);
            i = 0;
            while i < p { cand1.push(0.0); i += 1; }
        }
        j += 1;
    }
    j = n11;
    while j < q {
        if sj[j] > CX_ZERO {
            ok2.push(1);
            i = 0;
            while i < mp { cand2.push(wb[j * mp + i] / sj[j]); i += 1; }
        } else {
            ok2.push(0);
            i = 0;
            while i < mp { cand2.push(0.0); i += 1; }
        }
        j += 1;
    }
    i = 0;
    while i < r {
        let jj = n11 + i;
        if sj[jj] <= CX_ZERO { theta.push(0.0); }
        else if cj[jj] <= CX_ZERO { theta.push(CX_HALF_PI); }
        else { theta.push(mathshim::arctangent2_of(sj[jj], cj[jj])); }
        i += 1;
    }
    // The columns of U1: the paired vectors, then the free ones.
    var order1: i64[>..] = [];
    i = 0;
    while i < k1 { order1.push(i); i += 1; }
    i = k1;
    while i < p { order1.push(-1); i += 1; }
    // The columns of U2: the free ones, then the sine slots, then the
    // unit columns.
    var order2: i64[>..] = [];
    i = 0;
    while i < n22 { order2.push(-1); i += 1; }
    i = 0;
    while i < r + n21 { order2.push(i); i += 1; }
    if order2.len != mp || order1.len != p || r + n21 + n11 != q { return false; }
    let u1 = cx_orth(p, cand1, ok1, order1, false);
    let u2 = cx_orth(mp, cand2, ok2, order2, true);
    if u1.len != p * p || u2.len != mp * mp { return false; }
    // U is block diagonal. The columns of each block lie one after the
    // other in u1 and u2, so the entry of the row i and the column c
    // lies at c * n + i.
    var u: f64[>..] = [];
    i = 0;
    while i < m * m { u.push(0.0); i += 1; }
    i = 0;
    while i < p {
        j = 0;
        while j < p { u[i * m + j] = u1[j * p + i]; j += 1; }
        i += 1;
    }
    i = 0;
    while i < mp {
        j = 0;
        while j < mp { u[(p + i) * m + (p + j)] = u2[j * mp + i]; j += 1; }
        i += 1;
    }
    // Vh = CS^T U^T X. The product U^T X comes first.
    var tx: f64[>..] = [];
    i = 0;
    while i < m * m { tx.push(0.0); i += 1; }
    i = 0;
    while i < m {
        var lo: i64 = 0;
        var hi: i64 = p;
        if i >= p { lo = p; hi = m; }
        var k: i64 = lo;
        while k < hi {
            let uk = u[k * m + i];
            if uk != 0.0 {
                j = 0;
                while j < m { tx[i * m + j] += uk * x[off + k * m + j]; j += 1; }
            }
            k += 1;
        }
        i += 1;
    }
    let cs0 = cx_layout(theta, m, p, q, false);
    var vh: f64[>..] = [];
    i = 0;
    while i < m * m { vh.push(0.0); i += 1; }
    var k: i64 = 0;
    while k < m {
        i = 0;
        while i < m {
            let c = cs0[k * m + i];
            if c != 0.0 {
                j = 0;
                while j < m { vh[i * m + j] += c * tx[k * m + j]; j += 1; }
            }
            i += 1;
        }
        k += 1;
    }
    // The blocks off the diagonal of Vh hold rounding only.
    i = 0;
    while i < m {
        j = 0;
        while j < m {
            if (i < q && j >= q) || (i >= q && j < q) { vh[i * m + j] = 0.0; }
            j += 1;
        }
        i += 1;
    }
    // The flag swap moves the negative blocks of CS. The factors then
    // hold the blocks U2 and V2^T with the opposite sign.
    if swap {
        i = 0;
        while i < m {
            j = p;
            while j < m { u[i * m + j] = -u[i * m + j]; j += 1; }
            i += 1;
        }
        i = q;
        while i < m {
            j = 0;
            while j < m { vh[i * m + j] = -vh[i * m + j]; j += 1; }
            i += 1;
        }
    }
    let cs = cx_layout(theta, m, p, q, swap);
    i = 0;
    while i < theta.len { th.push(theta[i]); i += 1; }
    i = 0;
    while i < m * m { uo.push(u[i]); cso.push(cs[i]); vho.push(vh[i]); i += 1; }
    return true;
}

