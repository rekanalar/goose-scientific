// The dense helpers of lobpcg.
//
// A block holds its columns in one flat buffer of rows by cols values in
// the row-major order. The helpers multiply blocks, factor a small Gram
// matrix, solve the small Rayleigh-Ritz problem and give the dense answer
// of a problem that is too small for the iteration.
namespace scientific;

import mathshim;

// lob_mul gives the product of a block of ar by ac values with a block of
// ac by bc values.
fn lob_mul(a: f64[:], ar: i64, ac: i64, b: f64[:], bc: i64) -> f64[>..] {
    var out: f64[>..] = nd_zeros(ar * bc);
    var i: i64 = 0;
    while i < ar {
        var k: i64 = 0;
        while k < ac {
            let t = a[i * ac + k];
            if t != 0.0 {
                var j: i64 = 0;
                while j < bc { out[i * bc + j] += t * b[k * bc + j]; j += 1; }
            }
            k += 1;
        }
        i += 1;
    }
    return out;
}

// lob_tmul gives the product of the transpose of a block of r by ac values
// with a block of r by bc values. The result holds ac by bc values.
fn lob_tmul(a: f64[:], r: i64, ac: i64, b: f64[:], bc: i64) -> f64[>..] {
    var out: f64[>..] = nd_zeros(ac * bc);
    var i: i64 = 0;
    while i < r {
        var k: i64 = 0;
        while k < ac {
            let t = a[i * ac + k];
            if t != 0.0 {
                var j: i64 = 0;
                while j < bc { out[k * bc + j] += t * b[i * bc + j]; j += 1; }
            }
            k += 1;
        }
        i += 1;
    }
    return out;
}

// lob_spmm gives the product of a sparse matrix with a block of c columns.
fn lob_spmm(s, x: f64[:], c: i64) -> f64[>..] {
    var out: f64[>..] = nd_zeros(s.rows * c);
    var r: i64 = 0;
    while r < s.rows {
        var k = csr_ptr(s, r);
        while k < csr_ptr(s, r + 1) {
            let v = csr_val(s, k);
            let col = csr_col(s, k);
            var j: i64 = 0;
            while j < c { out[r * c + j] += v * x[col * c + j]; j += 1; }
            k += 1;
        }
        r += 1;
    }
    return out;
}

// lob_apply gives the product of an optional matrix with a block. A matrix
// without rows stands for the identity.
fn lob_apply(s, x: f64[:], c: i64) -> f64[>..] {
    if s.rows == 0 { return copy_slice(x); }
    return lob_spmm(s, x, c);
}

// lob_cols gives the columns of a block that the mask keeps.
fn lob_cols(x: f64[:], r: i64, c: i64, mask: bool[:], cnt: i64) -> f64[>..] {
    var out: f64[>..] = nd_zeros(r * cnt);
    var i: i64 = 0;
    while i < r {
        var q: i64 = 0;
        var j: i64 = 0;
        while j < c {
            if mask[j] { out[i * cnt + q] = x[i * c + j]; q += 1; }
            j += 1;
        }
        i += 1;
    }
    return out;
}

// lob_sub gives the block a minus the block b of the same shape.
fn lob_sub(a: f64[:], b: f64[:]) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < a.len { out.push(a[i] - b[i]); i += 1; }
    return out;
}

// lob_add gives the block a plus the block b of the same shape.
fn lob_add(a: f64[:], b: f64[:]) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < a.len { out.push(a[i] + b[i]); i += 1; }
    return out;
}

// lob_colnorms gives the Euclidean norm of each column of a block.
fn lob_colnorms(x: f64[:], r: i64, c: i64) -> f64[>..] {
    var out: f64[>..] = nd_zeros(c);
    var i: i64 = 0;
    while i < r {
        var j: i64 = 0;
        while j < c { out[j] += x[i * c + j] * x[i * c + j]; j += 1; }
        i += 1;
    }
    var j2: i64 = 0;
    while j2 < c { out[j2] = mathshim::root(out[j2]); j2 += 1; }
    return out;
}

// lob_chol gives the lower Cholesky factor of a symmetric k by k matrix.
// The flag is false when a pivot is not positive or not finite.
fn lob_chol(g: f64[:], k: i64) -> f64[>..], bool {
    var l: f64[>..] = nd_zeros(k * k);
    var r: i64 = 0;
    while r < k {
        var c: i64 = 0;
        while c <= r {
            var t = g[r * k + c];
            var q: i64 = 0;
            while q < c { t -= l[r * k + q] * l[c * k + q]; q += 1; }
            if r == c {
                if !(t > 0.0) || t - t != 0.0 { return l, false; }
                l[r * k + r] = mathshim::root(t);
            } else {
                l[r * k + c] = t / l[c * k + c];
            }
            c += 1;
        }
        r += 1;
    }
    return l, true;
}

// lob_chol_solve solves L L^T x = rhs for a block of c columns.
fn lob_chol_solve(l: f64[:], k: i64, rhs: f64[:], c: i64) -> f64[>..] {
    var x: f64[>..] = copy_slice(rhs);
    var j: i64 = 0;
    while j < c {
        var i: i64 = 0;
        while i < k {
            var t = x[i * c + j];
            var q: i64 = 0;
            while q < i { t -= l[i * k + q] * x[q * c + j]; q += 1; }
            x[i * c + j] = t / l[i * k + i];
            i += 1;
        }
        i = k - 1;
        while i >= 0 {
            var t = x[i * c + j];
            var q: i64 = i + 1;
            while q < k { t -= l[q * k + i] * x[q * c + j]; q += 1; }
            x[i * c + j] = t / l[i * k + i];
            i -= 1;
        }
        j += 1;
    }
    return x;
}

// lob_inv_upper gives the inverse of the upper factor L^T of a lower
// factor L. The inverse is upper triangular.
fn lob_inv_upper(l: f64[:], k: i64) -> f64[>..] {
    var inv: f64[>..] = nd_zeros(k * k);
    var c: i64 = 0;
    while c < k {
        var i: i64 = c;
        while i >= 0 {
            var t: f64 = 0.0;
            if i == c { t = 1.0; }
            var q: i64 = i + 1;
            while q <= c { t -= l[q * k + i] * inv[q * k + c]; q += 1; }
            inv[i * k + c] = t / l[i * k + i];
            i -= 1;
        }
        c += 1;
    }
    return inv;
}

// lob_orth B-orthonormalizes a block of r rows and c columns. It takes the
// block and its product with B. It gives the new block, the new product
// and the flag of success. The new block is the block times the inverse of
// the Cholesky factor of V^T B V.
fn lob_orth(v: f64[:], bv: f64[:], r: i64, c: i64) -> f64[>..], f64[>..], f64[>..], bool {
    let g = lob_tmul(v, r, c, bv, c);
    let l, ok = lob_chol(g, c);
    if !ok { return [], [], [], false; }
    let u = lob_inv_upper(l, c);
    let nv = lob_mul(v, r, c, u, c);
    let nbv = lob_mul(bv, r, c, u, c);
    return nv, nbv, u, true;
}

// lob_symmetrize gives the average of a k by k matrix and its transpose.
fn lob_symmetrize(g: f64[:], k: i64) -> f64[>..] {
    var out: f64[>..] = nd_zeros(k * k);
    var i: i64 = 0;
    while i < k {
        var j: i64 = 0;
        while j < k { out[i * k + j] = 0.5 * (g[i * k + j] + g[j * k + i]); j += 1; }
        i += 1;
    }
    return out;
}

// lob_pick gives the places of the sx values that the call keeps, in the
// order of the result. The values come in rising order. The largest values
// come last, so the call reads them from the end.
fn lob_pick(dim: i64, sx: i64, largest: bool) -> i64[>..] {
    var out: i64[>..] = [];
    var i: i64 = 0;
    while i < sx {
        if largest { out.push(dim - 1 - i); } else { out.push(i); }
        i += 1;
    }
    return out;
}

// lob_rr solves the generalized problem of two k by k Gram matrices. It
// gives the sx extreme values and their vectors in the columns of a block
// of k by sx values. The flag is false when the solve fails.
fn lob_rr(ga: f64[:], gb: f64[:], k: i64, sx: i64, largest: bool) -> f64[>..], f64[>..], bool {
    let w, v, f = eigh_pair(mat2(copy_slice(ga), k, k), mat2(copy_slice(gb), k, k));
    if f.kind != F_NONE || w.data.len != k { return [], [], false; }
    var i: i64 = 0;
    while i < k * k { if v.data[i] - v.data[i] != 0.0 { return [], [], false; } i += 1; }
    let ii = lob_pick(k, sx, largest);
    var vals: f64[>..] = [];
    var vecs: f64[>..] = nd_zeros(k * sx);
    var c: i64 = 0;
    while c < sx {
        vals.push(w.data[ii[c]]);
        var r: i64 = 0;
        while r < k { vecs[r * sx + c] = v.data[r * k + ii[c]]; r += 1; }
        c += 1;
    }
    return vals, vecs, true;
}

// lob_dense gives the extreme pairs of a problem that is too small for the
// iteration. It solves the whole dense problem.
fn lob_dense(a, b, sx: i64, largest: bool) -> Arr, Arr, Fault {
    let n = a.rows;
    let da, fa = csr_to_dense(a);
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), fa; }
    var w = empty_arr();
    var v = empty_arr();
    if b.rows == 0 {
        let w1, v1, f1 = eigh(da);
        if f1.kind != F_NONE { return empty_arr(), empty_arr(), f1; }
        w = copy(w1);
        v = copy(v1);
    } else {
        let db, fb = csr_to_dense(b);
        if fb.kind != F_NONE { return empty_arr(), empty_arr(), fb; }
        let w2, v2, f2 = eigh_pair(da, db);
        if f2.kind != F_NONE { return empty_arr(), empty_arr(), fault(F_VALUE, "lobpcg", "the matrix B is not positive definite"); }
        w = copy(w2);
        v = copy(v2);
    }
    let ii = lob_pick(n, sx, largest);
    var vals: f64[>..] = [];
    var vecs: f64[>..] = nd_zeros(n * sx);
    var c: i64 = 0;
    while c < sx {
        vals.push(w.data[ii[c]]);
        var r: i64 = 0;
        while r < n { vecs[r * sx + c] = v.data[r * n + ii[c]]; r += 1; }
        c += 1;
    }
    return line_arr(copy_slice(vals)), mat2(copy_slice(vecs), n, sx), no_fault();
}
