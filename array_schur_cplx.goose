// The conversion of the real Schur form into the complex Schur form,
// and the conversion of the complex diagonal form into the real block
// form. The methods follow scipy.linalg.rsf2csf and
// scipy.linalg.cdf2rdf.
//
// Both calls take a stack of matrices. The last two dimensions of an
// operand give the matrix. The dimensions before them give the batch.
// The batch of the two operands broadcasts.
namespace scientific;

import mathshim;

// The difference below the diagonal that holds no value. The value is
// the machine epsilon of a double.
const SCHUR_EPS: f64 = 2.220446049250313e-16;

// The largest side for the conversions. The work of one matrix grows
// with the square of the side.
const SCHUR_MAX_SIDE: i64 = 512;

// cs_check_stack checks the shape of an operand. The last mdims
// dimensions give the matrix. The dimensions before them give the
// batch. The width gives the count of values for each element.
fn cs_check_stack(ndim: i64, shape: i64[:], len: i64, width: i64, mdims: i64, tag: const u8[:]) -> Fault {
    // The counts come first. The side limit needs a shape that agrees
    // with the count of dimensions.
    if ndim != shape.len {
        return fault(F_SHAPE, tag, "an operand holds a count of dimensions that does not match its shape");
    }
    if ndim < mdims {
        return fault(F_SHAPE, tag, "an operand holds too few dimensions");
    }
    // The side limit comes before the buffer length. A bad side faults
    // even when the buffer is short.
    var i: i64 = ndim - mdims;
    while i < ndim {
        if shape[i] > SCHUR_MAX_SIDE {
            return fault(F_LIMIT, tag, "an operand holds a side that is out of the limits");
        }
        i += 1;
    }
    return check_stack(ndim, shape, len, width, tag);
}

// cs_batch_walk gives the broadcast of two batch shapes. It gives the
// flat index of each operand for every slice of the broadcast. An
// empty batch gives one slice. The broadcast holds at most maxn
// slices. The check runs before any index exists, because the
// broadcast of two valid batches can hold far more slices than
// either batch.
fn cs_batch_walk(sa: i64[:], sb: i64[:], maxn: i64, tag: const u8[:], ia: i64[>..]&, ib: i64[>..]&) -> i64[>..], Fault {
    let cb, f = batch_shape(sa, sb);
    if f.kind != F_NONE { return cb, fault(f.kind, tag, f.detail); }
    var n: i64 = 1;
    var i: i64 = 0;
    while i < cb.len {
        if n > maxn / cb[i] {
            return cb, fault(F_LIMIT, tag, "the result holds more elements than the limit");
        }
        n = n * cb[i];
        i += 1;
    }
    bcast_indices(cb, sa, sb, ia, ib);
    return cb, no_fault();
}

// cs_get gives one element of a complex buffer. The buffer holds the
// real part and the imaginary part of each element side by side.
fn cs_get(d: f64[:], i: i64) -> Cplx {
    return Cplx { re: d[2 * i], im: d[2 * i + 1] };
}

// cs_put writes one element of a complex buffer.
fn cs_put(d: f64[>..]&, i: i64, v: Cplx) {
    d[2 * i] = v.re;
    d[2 * i + 1] = v.im;
}

// cs_zero gives a complex number whose parts hold zero.
fn cs_zero() -> Cplx {
    return Cplx { re: 0.0, im: 0.0 };
}

// cs_roots2 gives the two eigenvalues of a matrix of two rows. The
// first root holds the larger imaginary part. The method follows the
// formula of the characteristic polynomial.
fn cs_roots2(a: Cplx, b: Cplx, c: Cplx, d: Cplx) -> Cplx, Cplx {
    let tr = c_add(a, d);
    let det = c_sub(c_mul(a, d), c_mul(b, c));
    let disc = c_sub(c_mul(tr, tr), c_scale(det, 4.0));
    let s = c_sqrt(disc);
    let r1 = c_scale(c_add(tr, s), 0.5);
    let r2 = c_scale(c_sub(tr, s), 0.5);
    if r1.im >= r2.im { return r1, r2; }
    return r2, r1;
}

// cs_rsf_slice turns one real Schur form into the complex Schur form.
// The buffer t holds the matrix at the place toff. The buffer z holds
// the change of the basis at the place zoff. The side m gives the
// count of rows. The results go into the buffers tc and zc.
fn cs_rsf_slice(t: f64[:], toff: i64, z: f64[:], zoff: i64, m: i64, tc: f64[>..]&, zc: f64[>..]&) {
    // The two matrices become complex. The imaginary parts start at
    // zero.
    var a: f64[>..] = [];
    var b: f64[>..] = [];
    var i: i64 = 0;
    while i < m * m { a.push(t[toff + i]); a.push(0.0); b.push(z[zoff + i]); b.push(0.0); i += 1; }
    var k: i64 = m - 1;
    while k > 0 {
        let a11 = cs_get(a, (k - 1) * m + (k - 1));
        let a12 = cs_get(a, (k - 1) * m + k);
        let a21 = cs_get(a, k * m + (k - 1));
        let a22 = cs_get(a, k * m + k);
        // The turn works only when the part below the diagonal holds a
        // value that is not small.
        if c_abs(a21) > SCHUR_EPS * (c_abs(a11) + c_abs(a22)) {
            // The turn takes the pair of the diagonal of the block onto
            // the diagonal of the complex form.
            var r1 = cs_zero();
            var r2 = cs_zero();
            r1, r2 = cs_roots2(a11, a12, a21, a22);
            let mu = c_sub(r1, a22);
            let rr = mathshim::root(c_abs(mu) * c_abs(mu) + c_abs(a21) * c_abs(a21));
            let c = c_scale(mu, 1.0 / rr);
            let s = c_scale(a21, 1.0 / rr);
            // The turn of the two rows and the two columns.
            var j: i64 = k - 1;
            while j < m {
                let x = cs_get(a, (k - 1) * m + j);
                let y = cs_get(a, k * m + j);
                cs_put(a, (k - 1) * m + j, c_add(c_mul(c_conj(c), x), c_mul(s, y)));
                cs_put(a, k * m + j, c_sub(c_mul(c, y), c_mul(s, x)));
                j += 1;
            }
            // The turn of the two columns of every row above.
            j = 0;
            while j <= k {
                let x = cs_get(a, j * m + (k - 1));
                let y = cs_get(a, j * m + k);
                cs_put(a, j * m + (k - 1), c_add(c_mul(x, c), c_mul(y, c_conj(s))));
                cs_put(a, j * m + k, c_sub(c_mul(y, c_conj(c)), c_mul(x, c_conj(s))));
                j += 1;
            }
            // The turn of the two columns of the change of the basis.
            j = 0;
            while j < m {
                let x = cs_get(b, j * m + (k - 1));
                let y = cs_get(b, j * m + k);
                cs_put(b, j * m + (k - 1), c_add(c_mul(x, c), c_mul(y, c_conj(s))));
                cs_put(b, j * m + k, c_sub(c_mul(y, c_conj(c)), c_mul(x, c_conj(s))));
                j += 1;
            }
        }
        // The part below the diagonal holds no value.
        cs_put(a, k * m + (k - 1), cs_zero());
        k -= 1;
    }
    i = 0;
    while i < 2 * m * m { tc.push(a[i]); zc.push(b[i]); i += 1; }
}

// rsf2csf converts a real Schur form into the complex Schur form. The
// array t holds the real Schur form and the array z holds the change
// of the basis. The last two dimensions of each array give the
// matrix. The dimensions before them give the batch. The batch of the
// two operands broadcasts. The result holds the complex Schur form and
// the complex change of the basis. The method follows
// scipy.linalg.rsf2csf.
fn rsf2csf(t, z) -> CArr, CArr, Fault {
    let ft = cs_check_stack(t.ndim, t.shape, t.data.len, 1, 2, "rsf2csf");
    if ft.kind != F_NONE { return carr_empty(), carr_empty(), ft; }
    let fz = cs_check_stack(z.ndim, z.shape, z.data.len, 1, 2, "rsf2csf");
    if fz.kind != F_NONE { return carr_empty(), carr_empty(), fz; }
    let nt = t.ndim;
    let nz = z.ndim;
    let m = t.shape[nt - 1];
    if t.shape[nt - 2] != m {
        return carr_empty(), carr_empty(), fault(F_SHAPE, "rsf2csf", "the Schur form is not a square matrix");
    }
    if z.shape[nz - 1] != z.shape[nz - 2] {
        return carr_empty(), carr_empty(), fault(F_SHAPE, "rsf2csf", "the change of the basis is not a square matrix");
    }
    if z.shape[nz - 1] != m {
        return carr_empty(), carr_empty(), fault(F_SHAPE, "rsf2csf", "the two matrices hold a different count of rows");
    }
    let f = carr_finite(t, "rsf2csf");
    if f.kind != F_NONE { return carr_empty(), carr_empty(), f; }
    let g = carr_finite(z, "rsf2csf");
    if g.kind != F_NONE { return carr_empty(), carr_empty(), g; }
    // The batch of each operand holds the dimensions before the last
    // two.
    let bt = shape_batch(t.shape, 2);
    let bz = shape_batch(z.shape, 2);
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    let cb, fb = cs_batch_walk(bt, bz, MAX_ELEMENTS / (m * m), "rsf2csf", ia, ib);
    if fb.kind != F_NONE { return carr_empty(), carr_empty(), fb; }
    let n = batch_count(cb);
    // Each slice becomes one complex form.
    var tc: f64[>..] = [];
    var zc: f64[>..] = [];
    var s: i64 = 0;
    while s < n {
        cs_rsf_slice(t.data, ia[s] * m * m, z.data, ib[s] * m * m, m, tc, zc);
        s += 1;
    }
    let sh = shape_matrix(cb, m, m);
    return carr_make(sh, copy(tc)), carr_make(sh, copy(zc)), no_fault();
}

// cs_cdf_pairs gives the places of the values of one set of
// eigenvalues that hold an imaginary part. The places come in the
// order they appear.
fn cs_cdf_pairs(w: f64[:], woff: i64, m: i64) -> i64[>..] {
    var out: i64[>..] = [];
    var i: i64 = 0;
    while i < m {
        if w[woff + 2 * i + 1] != 0.0 { out.push(i); }
        i += 1;
    }
    return out;
}

// cs_cdf_block gives the real block form of one set of eigenvalues.
// The buffer w holds the eigenvalues at the place woff. The side m
// gives the count of rows. The places of the complex values come in
// the pairs list. The result goes into the buffer wr.
fn cs_cdf_block(w: f64[:], woff: i64, m: i64, pairs: i64[:], wr: f64[>..]&) {
    // The block of one set goes into a local buffer. It joins the
    // buffer of the result at the end.
    var blk: f64[>..] = [];
    var i: i64 = 0;
    while i < m * m { blk.push(0.0); i += 1; }
    i = 0;
    while i < m {
        blk[i * m + i] = w[woff + 2 * i];
        i += 1;
    }
    // The complex values pair in the order they appear. The first
    // pairs with the second. The third pairs with the fourth.
    var p: i64 = 0;
    while p + 1 < pairs.len {
        let j = pairs[p];
        let k = pairs[p + 1];
        blk[j * m + k] = w[woff + 2 * j + 1];
        blk[k * m + j] = w[woff + 2 * k + 1];
        p += 2;
    }
    i = 0;
    while i < m * m { wr.push(blk[i]); i += 1; }
}

// cs_cdf_vec gives the real eigenvectors of one set of eigenvalues.
// The buffer v holds the complex eigenvectors at the place voff. The
// side m gives the count of rows. The places of the complex values
// come in the pairs list. The result goes into the buffer vr.
//
// A column outside a pair copies the column of v. The column j of a
// pair holds minus one half of the imaginary part of the difference.
// The column k holds one half of the sum of the real parts.
fn cs_cdf_vec(v: f64[:], voff: i64, m: i64, pairs: i64[:], vr: f64[>..]&) {
    // The role of each column tells whether it takes part in a pair.
    var role: i64[>..] = [];
    var other: i64[>..] = [];
    var i: i64 = 0;
    while i < m { role.push(0); other.push(-1); i += 1; }
    var p: i64 = 0;
    while p + 1 < pairs.len {
        let j = pairs[p];
        let k = pairs[p + 1];
        role[j] = 1;
        role[k] = 2;
        other[j] = k;
        other[k] = j;
        p += 2;
    }
    var r: i64 = 0;
    while r < m {
        var c: i64 = 0;
        while c < m {
            var value: f64 = 0.0;
            if role[c] == 1 {
                let k = other[c];
                value = -0.5 * (v[voff + 2 * (r * m + c) + 1] - v[voff + 2 * (r * m + k) + 1]);
            } else if role[c] == 2 {
                let j = other[c];
                value = 0.5 * (v[voff + 2 * (r * m + j)] + v[voff + 2 * (r * m + c)]);
            } else {
                value = v[voff + 2 * (r * m + c)];
            }
            vr.push(value);
            c += 1;
        }
        r += 1;
    }
}

// cdf2rdf converts a complex diagonal form into a real block form. The
// array w holds the complex eigenvalues and the array v holds the
// complex eigenvectors. The last dimension of w gives the values of
// one set. The last two dimensions of v give the matrix of one set.
// The array v holds one dimension more than the array w. The result
// holds the real block form and the real eigenvectors. The method
// follows scipy.linalg.cdf2rdf.
//
// The block form holds the batch of the eigenvalues. The eigenvectors
// hold the broadcast of the batches of both operands.
fn cdf2rdf(w, v) -> Arr, Arr, Fault {
    let fw = cs_check_stack(w.ndim, w.shape, w.data.len, 2, 1, "cdf2rdf");
    if fw.kind != F_NONE { return empty_arr(), empty_arr(), fw; }
    let fv = cs_check_stack(v.ndim, v.shape, v.data.len, 2, 2, "cdf2rdf");
    if fv.kind != F_NONE { return empty_arr(), empty_arr(), fv; }
    let nw = w.ndim;
    let nv = v.ndim;
    if nv != nw + 1 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "cdf2rdf", "the eigenvectors hold one dimension more than the eigenvalues");
    }
    let m = w.shape[nw - 1];
    if v.shape[nv - 1] != m || v.shape[nv - 2] != m {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "cdf2rdf", "the eigenvalues and the eigenvectors hold a different count");
    }
    let ffw = carr_finite(w, "cdf2rdf");
    if ffw.kind != F_NONE { return empty_arr(), empty_arr(), ffw; }
    let ffv = carr_finite(v, "cdf2rdf");
    if ffv.kind != F_NONE { return empty_arr(), empty_arr(), ffv; }
    // The batch of the eigenvalues holds the dimensions before the last
    // one. The batch of the eigenvectors holds the dimensions before
    // the last two.
    let bw = shape_batch(w.shape, 1);
    let bv = shape_batch(v.shape, 2);
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    let cb, fb = cs_batch_walk(bw, bv, MAX_ELEMENTS / (m * m), "cdf2rdf", ia, ib);
    if fb.kind != F_NONE { return empty_arr(), empty_arr(), fb; }
    // The places of the complex values of every set of eigenvalues.
    // Each set gives its places once. An odd count is a fault.
    let nw_slices = batch_count(bw);
    var places: i64[>..] = [];
    var starts: i64[>..] = [];
    var s: i64 = 0;
    while s < nw_slices {
        starts.push(places.len);
        let found = cs_cdf_pairs(w.data, s * m * 2, m);
        if found.len % 2 != 0 {
            return empty_arr(), empty_arr(), fault(F_VALUE, "cdf2rdf", "the complex eigenvalues hold no pairs");
        }
        var q: i64 = 0;
        while q < found.len { places.push(found[q]); q += 1; }
        s += 1;
    }
    starts.push(places.len);
    // The block form holds one slice for each set of eigenvalues.
    var wr: f64[>..] = [];
    s = 0;
    while s < nw_slices {
        cs_cdf_block(w.data, s * m * 2, m, places[starts[s]..starts[s + 1]], wr);
        s += 1;
    }
    // The eigenvectors hold one slice for each set of the broadcast.
    var vr: f64[>..] = [];
    let n = batch_count(cb);
    s = 0;
    while s < n {
        let at = ia[s];
        cs_cdf_vec(v.data, ib[s] * m * m * 2, m, places[starts[at]..starts[at + 1]], vr);
        s += 1;
    }
    let sw = shape_matrix(bw, m, m);
    let sv = shape_matrix(cb, m, m);
    return arr_of(copy(wr), sw), arr_of(copy(vr), sv), no_fault();
}

// carr_finite checks that every value of a real array holds a finite
// number.
fn carr_finite(a, tag: const u8[:]) -> Fault {
    var i: i64 = 0;
    while i < a.data.len {
        let v = a.data[i];
        if !(v - v == 0.0) {
            return fault(F_VALUE, tag, "an operand holds a value that is not finite");
        }
        i += 1;
    }
    return no_fault();
}
