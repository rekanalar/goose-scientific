// The eigenvalues of a sparse matrix and the function of a matrix.
//
// The eigenvalues of a matrix that is not symmetric.
//
// eigs gives the values of a matrix that is not symmetric and the
// vectors of the values. The method builds the Krylov space of the
// matrix with the steps of Arnoldi. The values of the space come from
// the dense solver of the complex matrices. The side of that solver
// bounds the space.
//
// The function of a matrix.
//
// funm_multiply_krylov gives the product of the function of a matrix
// and a vector. The method builds the Krylov space of the matrix with
// the steps of Arnoldi. The function of the small matrix of the space
// comes from the call funm. The call funm accepts a matrix with real
// values only, so the call gives a fault of the kind F_VALUE when the
// matrix of the space holds a pair of complex values.
namespace scientific;

import mathshim;

// The largest count of the columns of the Krylov space. The dense
// solver of the complex matrices holds a side of 256.
const MAX_EIGS_SPACE: i64 = 256;

// The smallest length of the vector of the space. A shorter vector
// stands for a space that the matrix ended.
const EIGS_BREAKDOWN: f64 = 1.0e-14;

// arnoldi builds the Krylov space of a matrix. The basis holds the
// columns of the space, one after another. The matrix of the space
// holds the values of the projection in its upper triangle. The buffer
// of the extra vector holds the next vector of the space when the
// space held its full count of columns, and zeros when the space
// ended. The call gives the count of the columns that the space holds.
fn arnoldi(s, m: i64, v0: f64[:], tag: const u8[:]) -> f64[>..], f64[>..], f64[>..], i64, Fault {
    let n = s.rows;
    var extra: f64[>..] = [];
    var i: i64 = 0;
    while i < n { extra.push(0.0); i += 1; }
    var basis: f64[>..] = [];
    var i: i64 = 0;
    while i < n * m { basis.push(0.0); i += 1; }
    var h: f64[>..] = [];
    i = 0;
    while i < m * m { h.push(0.0); i += 1; }
    // The first vector of the space.
    var v: f64[>..] = [];
    if v0.len == n {
        i = 0;
        while i < n { v.push(v0[i]); i += 1; }
    } else {
        var r = default_rng(1);
        i = 0;
        while i < n { v.push(random_uniform(r) - 0.5); i += 1; }
    }
    let nv = krylov_norm(v);
    if !(nv > 0.0) {
        return basis, h, extra, 0, fault(F_VALUE, tag, "the first vector of the space holds no value");
    }
    i = 0;
    while i < n { v[i] = v[i] / nv; i += 1; }
    var used: i64 = 0;
    var j: i64 = 0;
    while j < m {
        i = 0;
        while i < n { basis[j * n + i] = v[i]; i += 1; }
        let av, fa = csr_matvec(s, v);
        if fa.kind != F_NONE { return basis, h, extra, 0, fa; }
        var w: f64[>..] = [];
        i = 0;
        while i < n { w.push(av.data[i]); i += 1; }
        // The steps of Arnoldi with the Gram and Schmidt rule. The
        // second pass gives the values of the matrix of the space the
        // accuracy that the loss of the norm costs.
        var pass: i64 = 0;
        while pass < 2 {
            var t: i64 = 0;
            while t <= j {
                var d: f64 = 0.0;
                var u: i64 = 0;
                while u < n { d += basis[t * n + u] * w[u]; u += 1; }
                if pass == 0 { h[t * m + j] = d; } else { h[t * m + j] += d; }
                u = 0;
                while u < n { w[u] -= d * basis[t * n + u]; u += 1; }
                t += 1;
            }
            pass += 1;
        }
        let beta = krylov_norm(w);
        used = j + 1;
        if beta > EIGS_BREAKDOWN && used == m {
            i = 0;
            while i < n { extra[i] = w[i] / beta; i += 1; }
            break;
        }
        if used == m { break; }
        if beta <= EIGS_BREAKDOWN { break; }
        h[(j + 1) * m + j] = beta;
        var next: f64[>..] = [];
        i = 0;
        while i < n { next.push(w[i] / beta); i += 1; }
        v = copy(next);
        j += 1;
    }
    return basis, h, extra, used, no_fault();
}

// eigs_score gives the score of a value of the matrix. The names of
// SciPy follow: the largest magnitude, the smallest magnitude, the
// largest real part, the smallest real part, the largest imaginary
// part and the smallest imaginary part.
fn eigs_score(re: f64, im: f64, which: const u8[:]) -> f64 {
    if which == "SM" { return -mathshim::root(re * re + im * im); }
    if which == "LR" { return re; }
    if which == "SR" { return -re; }
    if which == "LI" { return im; }
    if which == "SI" { return -im; }
    return mathshim::root(re * re + im * im);
}

// eigs gives the values of a matrix that is not symmetric and the
// vectors of the values. The values come in the order of the name
// which. The vectors stand in the columns of the second result.
fn eigs(s, k: i64 = 6, which: const u8[:] = "LM", v0: f64[:] = []) -> CArr, CArr, Fault {
    if !csr_valid(s) {
        return carr_empty(), carr_empty(), fault(F_SHAPE, "eigs", "the buffer of the matrix does not agree with its sides");
    }
    if s.rows != s.cols || s.rows < 2 {
        return carr_empty(), carr_empty(), fault(F_SHAPE, "eigs", "the matrix is not square or holds fewer than two rows");
    }
    let n = s.rows;
    if k < 1 || k >= n {
        return carr_empty(), carr_empty(), fault(F_RANGE, "eigs", "the count of the values must stand from one to below the side");
    }
    if which != "LM" && which != "SM" && which != "LR" && which != "SR" && which != "LI" && which != "SI" {
        return carr_empty(), carr_empty(), fault(F_VALUE, "eigs", "the name of the choice holds an unknown name");
    }
    var m = n;
    if m > MAX_EIGS_SPACE { m = MAX_EIGS_SPACE; }
    if k >= m {
        return carr_empty(), carr_empty(), fault(F_RANGE, "eigs", "the count of the values stands above the space");
    }
    let basis, h, extra, used, f0 = arnoldi(s, m, v0, "eigs");
    if f0.kind != F_NONE { return carr_empty(), carr_empty(), f0; }
    if used < k + 1 {
        return carr_empty(), carr_empty(), fault(F_LIMIT, "eigs", "the space ended before the count of the values");
    }
    // The matrix of the space as a complex matrix.
    var hd: f64[>..] = [];
    var i: i64 = 0;
    while i < used * used {
        hd.push(h[(i / used) * m + (i % used)]);
        hd.push(0.0);
        i += 1;
    }
    let hm = carr_make([used, used], copy(hd));
    let w, y, f1 = ceig(hm);
    if f1.kind != F_NONE { return carr_empty(), carr_empty(), f1; }
    // The order of the values follows the name of the choice.
    var ord: i64[>..] = [];
    i = 0;
    while i < used { ord.push(i); i += 1; }
    var a: i64 = 0;
    while a < used {
        var b = a + 1;
        while b < used {
            let sa = eigs_score(w.data[2 * ord[a]], w.data[2 * ord[a] + 1], which);
            let sb = eigs_score(w.data[2 * ord[b]], w.data[2 * ord[b] + 1], which);
            if sb > sa {
                let t = ord[a];
                ord[a] = ord[b];
                ord[b] = t;
            }
            b += 1;
        }
        a += 1;
    }
    // The values and the vectors of the space.
    var values: f64[>..] = [];
    var vectors: f64[>..] = [];
    var j: i64 = 0;
    while j < k {
        let pick = ord[j];
        values.push(w.data[2 * pick]);
        values.push(w.data[2 * pick + 1]);
        var r: i64 = 0;
        while r < n {
            var vr: f64 = 0.0;
            var vi: f64 = 0.0;
            var i2: i64 = 0;
            while i2 < used {
                let yr = y.data[2 * (i2 * used + pick)];
                let yi = y.data[2 * (i2 * used + pick) + 1];
                vr += basis[i2 * n + r] * yr;
                vi += basis[i2 * n + r] * yi;
                i2 += 1;
            }
            vectors.push(vr);
            vectors.push(vi);
            r += 1;
        }
        j += 1;
    }
    return carr_make([k], copy(values)), carr_make([n, k], copy(vectors)), no_fault();
}

// funm_multiply_krylov gives the product of the function of a matrix
// and a vector. The factor t stands for the size of the step. The
// method builds the Krylov space of the matrix with the steps of
// Arnoldi. The function of the small matrix of the space comes from the
// call funm. The call funm accepts a matrix with real values only, so
// the call gives a fault of the kind F_VALUE when the matrix of the
// space holds a pair of complex values.
fn funm_multiply_krylov<F: fn(f64) -> f64>(s, b: f64[:], t: f64 = 1.0) -> Arr, Fault {
    if s.rows != s.cols || s.rows < 1 {
        return empty_arr(), fault(F_SHAPE, "funm_multiply_krylov", "the matrix is not square or holds no row");
    }
    let n = s.rows;
    if b.len != n {
        return empty_arr(), fault(F_SHAPE, "funm_multiply_krylov", "the right side does not agree with the matrix");
    }
    if !(t == t) {
        return empty_arr(), fault(F_VALUE, "funm_multiply_krylov", "the factor of the step holds the missing value");
    }
    var m = n;
    if m > MAX_EIGS_SPACE { m = MAX_EIGS_SPACE; }
    let basis, h, extra, used, f0 = arnoldi(s, m, [], "funm_multiply_krylov");
    if f0.kind != F_NONE { return empty_arr(), f0; }
    // The small matrix of the space with the factor of the step.
    var hd: f64[>..] = [];
    var i: i64 = 0;
    while i < used * used {
        hd.push(t * h[(i / used) * m + (i % used)]);
        i += 1;
    }
    let hm = mat2(copy(hd), used, used);
    let fm, f1 = funm(hm) { x => F(x) };
    if f1.kind != F_NONE { return empty_arr(), f1; }
    // The projection of the right side onto the space.
    var c: f64[>..] = [];
    var j: i64 = 0;
    while j < used {
        var sum: f64 = 0.0;
        var r: i64 = 0;
        while r < n { sum += basis[j * n + r] * b[r]; r += 1; }
        c.push(sum);
        j += 1;
    }
    // The function of the space applied to the projection.
    var fc: f64[>..] = [];
    var i2: i64 = 0;
    while i2 < used {
        var sum: f64 = 0.0;
        var l: i64 = 0;
        while l < used { sum += fm.data[i2 * used + l] * c[l]; l += 1; }
        fc.push(sum);
        i2 += 1;
    }
    // The result is the basis of the space times that vector. The
    // extra term covers the part of the product that the space does
    // not hold.
    var tail: f64 = 0.0;
    if used > 0 { tail = fm.data[(used - 1) * used + (used - 1)] * c[used - 1]; }
    var out: f64[>..] = [];
    var r: i64 = 0;
    while r < n {
        var sum: f64 = extra[r] * tail;
        var i3: i64 = 0;
        while i3 < used {
            sum += basis[i3 * n + r] * fc[i3];
            i3 += 1;
        }
        out.push(sum);
        r += 1;
    }
    return line_arr(copy(out)), no_fault();
}
