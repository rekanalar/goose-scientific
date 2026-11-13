// The Jacobi singular value decomposition of a complex matrix and the calls cqr and csvd.
//
namespace scientific;

import mathshim;

// c_jacobi_svd gives the singular values, the left vectors and the right
// vectors of a complex matrix. The buffer vals holds the singular values
// from the largest to the smallest. The buffer left holds the left vectors.
// The buffer vecs holds the right vectors with the transpose and the
// conjugate. The method is the one-sided Jacobi on the columns. A phase step
// makes the inner product of a pair of columns real. The real turn of the
// Jacobi method then makes it zero.
fn c_jacobi_svd(a: f64[:], m: i64, n: i64, vals: f64[>..]&, left: f64[>..]&, vecs: f64[>..]&) {
    // The work buffer holds the matrix.
    var w: f64[>..] = [];
    var i: i64 = 0;
    while i < 2 * m * n {
        w.push(a[i]);
        i += 1;
    }
    // The right vectors start as the unit matrix.
    var vr: f64[>..] = [];
    var vi: f64[>..] = [];
    i = 0;
    while i < 2 * n * n {
        vr.push(0.0);
        vi.push(0.0);
        i += 1;
    }
    i = 0;
    while i < n {
        vr[2 * (i * n + i)] = 1.0;
        i += 1;
    }
    // The method runs at most 60 sweeps. A sweep that changes nothing ends
    // the method.
    var sweep: i64 = 0;
    while sweep < 60 {
        var turned = false;
        var p: i64 = 0;
        while p < n - 1 {
            var q: i64 = p + 1;
            while q < n {
                // The inner product of the two columns and their lengths.
                var alpha: f64 = 0.0;
                var beta: f64 = 0.0;
                var gr: f64 = 0.0;
                var gi: f64 = 0.0;
                var r: i64 = 0;
                while r < m {
                    let fp = 2 * (r * n + p);
                    let fq = 2 * (r * n + q);
                    alpha += w[fp] * w[fp] + w[fp + 1] * w[fp + 1];
                    beta += w[fq] * w[fq] + w[fq + 1] * w[fq + 1];
                    let x, y = c_cmul(w[fp], -w[fp + 1], w[fq], w[fq + 1]);
                    gr += x;
                    gi += y;
                    r += 1;
                }
                let gabs = mathshim::root(gr * gr + gi * gi);
                if gabs > 1.0e-16 * mathshim::root(alpha * beta) && gabs > 0.0 {
                    // The phase step makes the inner product real. The
                    // second column takes the conjugate phase of the product and the right vector
                    // of the column takes the same phase.
                    if gi != 0.0 || gr < 0.0 {
                        let pr = gr / gabs;
                        let pi = -gi / gabs;
                        r = 0;
                        while r < m {
                            let fq = 2 * (r * n + q);
                            let x, y = c_cmul(pr, pi, w[fq], w[fq + 1]);
                            w[fq] = x;
                            w[fq + 1] = y;
                            r += 1;
                        }
                        r = 0;
                        while r < n {
                            let fv = 2 * (r * n + q);
                            let x, y = c_cmul(pr, pi, vr[fv], vi[fv]);
                            vr[fv] = x;
                            vi[fv] = y;
                            r += 1;
                        }
                    }
                    // The turn of the Jacobi method uses the real inner
                    // product.
                    let diff = alpha - beta;
                    let root = mathshim::root(diff * diff + 4.0 * gabs * gabs);
                    if root > 0.0 {
                        var tv: f64 = 0.0;
                        if diff >= 0.0 { tv = -2.0 * gabs / (diff + root); }
                        else { tv = 2.0 * gabs / (root - diff); }
                        let c1 = 1.0 / mathshim::root(tv * tv + 1.0);
                        let sn = tv * c1;
                        // The turn mixes the two columns.
                        r = 0;
                        while r < m {
                            let fp = 2 * (r * n + p);
                            let fq = 2 * (r * n + q);
                            let ap = w[fp];
                            let ai = w[fp + 1];
                            let aq = w[fq];
                            let bq = w[fq + 1];
                            w[fp] = c1 * ap - sn * aq;
                            w[fp + 1] = c1 * ai - sn * bq;
                            w[fq] = sn * ap + c1 * aq;
                            w[fq + 1] = sn * ai + c1 * bq;
                            r += 1;
                        }
                        // The right vectors follow the turn.
                        r = 0;
                        while r < n {
                            let fp = 2 * (r * n + p);
                            let fq = 2 * (r * n + q);
                            let ap = vr[fp];
                            let ai = vi[fp];
                            let aq = vr[fq];
                            let bq = vi[fq];
                            vr[fp] = c1 * ap - sn * aq;
                            vi[fp] = c1 * ai - sn * bq;
                            vr[fq] = sn * ap + c1 * aq;
                            vi[fq] = sn * ai + c1 * bq;
                            r += 1;
                        }
                        turned = true;
                    }
                }
                q += 1;
            }
            p += 1;
        }
        sweep += 1;
        if !turned { break; }
    }
    // The singular values are the lengths of the columns. They come from
    // the largest to the smallest.
    var order: i64[>..] = [];
    var len2: f64[>..] = [];
    i = 0;
    while i < n {
        order.push(i);
        var sum: f64 = 0.0;
        var r: i64 = 0;
        while r < m {
            let f = 2 * (r * n + i);
            sum += w[f] * w[f] + w[f + 1] * w[f + 1];
            r += 1;
        }
        len2.push(sum);
        i += 1;
    }
    // The sort takes the whole length of each column, not one entry.
    i = 1;
    while i < n {
        let key = order[i];
        var j = i - 1;
        while j >= 0 {
            let other = order[j];
            if len2[other] < len2[key] {
                order[j + 1] = other;
                j -= 1;
            } else { break; }
        }
        order[j + 1] = key;
        i += 1;
    }
    // The count of the singular values that hold a value above the
    // tolerance. The others hold no vector.
    var k = m;
    if n < k { k = n; }
    var tol: f64 = 0.0;
    if k > 0 {
        var sum: f64 = 0.0;
        var r: i64 = 0;
        while r < m {
            let f = 2 * (r * n + order[0]);
            sum += w[f] * w[f] + w[f + 1] * w[f + 1];
            r += 1;
        }
        tol = mathshim::root(sum) * 0.000001;
    }
    var valid: i64 = 0;
    while valid < k {
        var sum: f64 = 0.0;
        var r: i64 = 0;
        while r < m {
            let f = 2 * (r * n + order[valid]);
            sum += w[f] * w[f] + w[f + 1] * w[f + 1];
            r += 1;
        }
        if mathshim::root(sum) > tol { valid += 1; } else { break; }
    }
    // The values hold the lengths of the columns.
    i = 0;
    while i < k {
        var sum: f64 = 0.0;
        var r: i64 = 0;
        while r < m {
            let f = 2 * (r * n + order[i]);
            sum += w[f] * w[f] + w[f + 1] * w[f + 1];
            r += 1;
        }
        vals.push(mathshim::root(sum));
        i += 1;
    }
    // The left vectors hold the columns of the work buffer, divided by the
    // singular values.
    var lv: f64[>..] = [];
    i = 0;
    while i < 2 * m * m {
        lv.push(0.0);
        i += 1;
    }
    i = 0;
    while i < valid {
        let col = order[i];
        let sv = vals[i];
        var r: i64 = 0;
        while r < m {
            let f = 2 * (r * n + col);
            lv[2 * (r * m + i)] = w[f] / sv;
            lv[2 * (r * m + i) + 1] = w[f + 1] / sv;
            r += 1;
        }
        i += 1;
    }
    // The right vectors hold the columns of the turn.
    var rv: f64[>..] = [];
    i = 0;
    while i < 2 * n * n {
        rv.push(0.0);
        i += 1;
    }
    i = 0;
    while i < valid {
        let col = order[i];
        var r: i64 = 0;
        while r < n {
            let fv = 2 * (r * n + col);
            rv[2 * (r * n + i)] = vr[fv];
            rv[2 * (r * n + i) + 1] = vi[fv];
            r += 1;
        }
        i += 1;
    }
    // The phase rule. The largest component of each column of the left
    // vectors holds a real value above zero. The same phase comes on the
    // matching column of the right vectors.
    i = 0;
    while i < valid {
        let col = i;
        var best: f64 = -1.0;
        var br: f64 = 0.0;
        var bi: f64 = 0.0;
        var r: i64 = 0;
        while r < m {
            let f = 2 * (r * m + col);
            let size = mathshim::root(lv[f] * lv[f] + lv[f + 1] * lv[f + 1]);
            if size > best {
                best = size;
                br = lv[f];
                bi = lv[f + 1];
            }
            r += 1;
        }
        if best > 0.0 {
            let pr = br / best;
            let pi = -bi / best;
            r = 0;
            while r < m {
                let f = 2 * (r * m + col);
                let x, y = c_cmul(pr, pi, lv[f], lv[f + 1]);
                lv[f] = x;
                lv[f + 1] = y;
                r += 1;
            }
            r = 0;
            while r < n {
                let fv = 2 * (r * n + col);
                let x, y = c_cmul(pr, pi, rv[fv], rv[fv + 1]);
                rv[fv] = x;
                rv[fv + 1] = y;
                r += 1;
            }
        }
        i += 1;
    }
    // The left vectors that lack a value come from the unit vectors that
    // the columns do not hold.
    if m > valid {
        let full = c_complete_basis(lv, m, valid, m);
        i = 0;
        while i < 2 * m * m {
            lv[i] = full[i];
            i += 1;
        }
    }
    // The right vectors that lack a value come from the unit vectors that
    // the columns do not hold.
    if n > valid {
        let full = c_complete_basis(rv, n, valid, n);
        i = 0;
        while i < 2 * n * n {
            rv[i] = full[i];
            i += 1;
        }
    }
    // The left vectors come out.
    i = 0;
    while i < 2 * m * m {
        left.push(lv[i]);
        i += 1;
    }
    // The right vectors come as the transpose with the conjugate.
    i = 0;
    while i < n {
        var r: i64 = 0;
        while r < n {
            let fv = 2 * (r * n + i);
            vecs.push(rv[fv]);
            vecs.push(-rv[fv + 1]);
            r += 1;
        }
        i += 1;
    }
    return;
}

// cqr gives the orthogonal matrix and the upper triangle of a complex matrix.
// The result holds the reduced form, as NumPy gives it by default. The
// diagonal of the upper triangle holds real values. The sign of the diagonal
// follows the rule of LAPACK.
fn cqr(a) -> CArr, CArr, Fault {
    let f = c_factor_check(a, "cqr");
    if f.kind != F_NONE { return carr_empty(), carr_empty(), f; }
    let m = a.shape[0];
    let n = a.shape[1];
    var p = m;
    if n < p { p = n; }
    // The count of the elements of the results must not overflow.
    if m > MAX_ELEMENTS / p || n > MAX_ELEMENTS / p {
        return carr_empty(), carr_empty(), fault(F_LIMIT, "cqr", "a result holds more elements than the limit");
    }
    var q: f64[>..] = [];
    var r: f64[>..] = [];
    let scale = c_scale_of(a.data);
    c_householder_qr(c_scaled(a.data, scale), m, n, q, r);
    // The upper triangle takes the scale back. The orthogonal matrix does not change.
    var ri: i64 = 0;
    while ri < r.len {
        r[ri] = r[ri] * scale;
        ri += 1;
    }
    // The orthogonal matrix holds m rows and p columns.
    var sq: i64[..8] = [];
    sq.push(m);
    sq.push(p);
    // The upper triangle holds p rows and n columns.
    var sr: i64[..8] = [];
    sr.push(p);
    sr.push(n);
    return CArr { ndim: 2, shape: sq, data: q }, CArr { ndim: 2, shape: sr, data: r }, no_fault();
}

// csvd gives the left vectors, the singular values and the right vectors of a
// complex matrix. The singular values come from the largest to the smallest.
// The left vectors hold as many rows and columns as the operand has rows.
// The right vectors hold as many rows and columns as the operand has
// columns. The largest component of each column of the left vectors holds a
// real value above zero.
fn csvd(a) -> CArr, Arr, CArr, Fault {
    let f = c_factor_check(a, "csvd");
    if f.kind != F_NONE { return carr_empty(), empty_arr(), carr_empty(), f; }
    let m = a.shape[0];
    let n = a.shape[1];
    var k = m;
    if n < k { k = n; }
    // The count of the elements of the results must not overflow.
    if m > MAX_ELEMENTS / m || n > MAX_ELEMENTS / n {
        return carr_empty(), empty_arr(), carr_empty(), fault(F_LIMIT, "csvd", "a result holds more elements than the limit");
    }
    var vals: f64[>..] = [];
    var left: f64[>..] = [];
    var vecs: f64[>..] = [];
    let scale = c_scale_of(a.data);
    let work = c_scaled(a.data, scale);
    if m >= n {
        c_jacobi_svd(work, m, n, vals, left, vecs);
    } else {
        c_wide_svd(work, m, n, vals, left, vecs);
    }
    // The singular values take the scale back. The vectors do not change.
    var vi: i64 = 0;
    while vi < vals.len {
        vals[vi] = vals[vi] * scale;
        vi += 1;
    }
    // The left vectors hold m rows and m columns.
    var su: i64[..8] = [];
    su.push(m);
    su.push(m);
    // The values hold the size of the smaller side.
    var ss: i64[..8] = [];
    ss.push(k);
    // The right vectors hold n rows and n columns.
    var sv: i64[..8] = [];
    sv.push(n);
    sv.push(n);
    let out_vals, fv = from_values(vals);
    if fv.kind != F_NONE { return carr_empty(), empty_arr(), carr_empty(), fv; }
    return CArr { ndim: 2, shape: su, data: left }, out_vals, CArr { ndim: 2, shape: sv, data: vecs }, no_fault();
}
