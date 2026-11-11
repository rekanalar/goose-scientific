// The conversion of the real Schur form into the complex Schur form,
// and the conversion of the complex diagonal form into the real block
// form. The methods follow scipy.linalg.rsf2csf and
// scipy.linalg.cdf2rdf.
namespace scientific;

import mathshim;

// The difference below the diagonal that holds no value. The value is
// the machine epsilon of a double.
const SCHUR_EPS: f64 = 2.220446049250313e-16;

// The largest side for the conversions. The product of the
// eigenvectors grows with the cube of the side.
const SCHUR_MAX_SIDE: i64 = 512;

// cs_check_shape checks that the shape of an array agrees with its
// buffer and with the limit. The buffer holds width values for each
// element.
fn cs_check_shape(ndim: i64, shape: i64[:], len: i64, width: i64, tag: const u8[:]) -> Fault {
    if ndim != shape.len {
        return fault(F_SHAPE, tag, "an operand holds a count of dimensions that does not match its shape");
    }
    var count: i64 = 1;
    var i: i64 = 0;
    while i < shape.len {
        if shape[i] < 1 || shape[i] > SCHUR_MAX_SIDE {
            return fault(F_LIMIT, tag, "an operand holds a side that is out of the limits");
        }
        count = count * shape[i];
        i += 1;
    }
    if len != count * width {
        return fault(F_SHAPE, tag, "an operand holds a buffer that does not match its shape");
    }
    return no_fault();
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

// rsf2csf converts a real Schur form into the complex Schur form. The
// matrix t holds the real Schur form and the matrix z holds the change
// of the basis. The matrix t holds m rows and m columns, and the matrix
// z holds m rows and m columns. The result holds the complex Schur
// form and the complex change of the basis. The method follows
// scipy.linalg.rsf2csf.
fn rsf2csf(t, z) -> CArr, CArr, Fault {
    let ft = cs_check_shape(t.ndim, t.shape, t.data.len, 1, "rsf2csf");
    if ft.kind != F_NONE { return carr_empty(), carr_empty(), ft; }
    let fz = cs_check_shape(z.ndim, z.shape, z.data.len, 1, "rsf2csf");
    if fz.kind != F_NONE { return carr_empty(), carr_empty(), fz; }
    if t.ndim != 2 || t.shape[0] != t.shape[1] {
        return carr_empty(), carr_empty(), fault(F_SHAPE, "rsf2csf", "the Schur form is not a square matrix");
    }
    if z.ndim != 2 || z.shape[0] != z.shape[1] {
        return carr_empty(), carr_empty(), fault(F_SHAPE, "rsf2csf", "the change of the basis is not a square matrix");
    }
    if t.shape[0] != z.shape[0] {
        return carr_empty(), carr_empty(), fault(F_SHAPE, "rsf2csf", "the two matrices hold a different count of rows");
    }
    let m = t.shape[0];
    if m < 1 {
        return carr_empty(), carr_empty(), fault(F_SHAPE, "rsf2csf", "the matrix holds no rows");
    }
    let f = carr_finite(t, "rsf2csf");
    if f.kind != F_NONE { return carr_empty(), carr_empty(), f; }
    let g = carr_finite(z, "rsf2csf");
    if g.kind != F_NONE { return carr_empty(), carr_empty(), g; }
    // The two matrices become complex. The imaginary parts start at
    // zero.
    var tc: f64[>..] = [];
    var i: i64 = 0;
    while i < m * m { tc.push(t.data[i]); tc.push(0.0); i += 1; }
    var zc: f64[>..] = [];
    i = 0;
    while i < m * m { zc.push(z.data[i]); zc.push(0.0); i += 1; }
    var k: i64 = m - 1;
    while k > 0 {
        let a11 = cs_get(tc, (k - 1) * m + (k - 1));
        let a12 = cs_get(tc, (k - 1) * m + k);
        let a21 = cs_get(tc, k * m + (k - 1));
        let a22 = cs_get(tc, k * m + k);
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
                let x = cs_get(tc, (k - 1) * m + j);
                let y = cs_get(tc, k * m + j);
                cs_put(tc, (k - 1) * m + j, c_add(c_mul(c_conj(c), x), c_mul(s, y)));
                cs_put(tc, k * m + j, c_sub(c_mul(c, y), c_mul(s, x)));
                j += 1;
            }
            // The turn of the two columns of every row above.
            j = 0;
            while j <= k {
                let x = cs_get(tc, j * m + (k - 1));
                let y = cs_get(tc, j * m + k);
                cs_put(tc, j * m + (k - 1), c_add(c_mul(x, c), c_mul(y, c_conj(s))));
                cs_put(tc, j * m + k, c_sub(c_mul(y, c_conj(c)), c_mul(x, c_conj(s))));
                j += 1;
            }
            // The turn of the two columns of the change of the basis.
            j = 0;
            while j < m {
                let x = cs_get(zc, j * m + (k - 1));
                let y = cs_get(zc, j * m + k);
                cs_put(zc, j * m + (k - 1), c_add(c_mul(x, c), c_mul(y, c_conj(s))));
                cs_put(zc, j * m + k, c_sub(c_mul(y, c_conj(c)), c_mul(x, c_conj(s))));
                j += 1;
            }
        }
        // The part below the diagonal holds no value.
        cs_put(tc, k * m + (k - 1), cs_zero());
        k -= 1;
    }
    var sh: i64[..8] = [];
    sh.push(m);
    sh.push(m);
    return carr_make(sh, copy(tc)), carr_make(sh, copy(zc)), no_fault();
}

// cdf2rdf converts a complex diagonal form into a real block form. The
// vector w holds the complex eigenvalues and the matrix v holds the
// complex eigenvectors. The vector w holds m values and the matrix v
// holds m rows and m columns. The result holds the real block form of
// the eigenvalues and the real eigenvectors. The method follows
// scipy.linalg.cdf2rdf.
fn cdf2rdf(w, v) -> Arr, Arr, Fault {
    let fw = cs_check_shape(w.ndim, w.shape, w.data.len, 2, "cdf2rdf");
    if fw.kind != F_NONE { return empty_arr(), empty_arr(), fw; }
    let fv = cs_check_shape(v.ndim, v.shape, v.data.len, 2, "cdf2rdf");
    if fv.kind != F_NONE { return empty_arr(), empty_arr(), fv; }
    let ffw = carr_finite(w, "cdf2rdf");
    if ffw.kind != F_NONE { return empty_arr(), empty_arr(), ffw; }
    let ffv = carr_finite(v, "cdf2rdf");
    if ffv.kind != F_NONE { return empty_arr(), empty_arr(), ffv; }
    if w.ndim != 1 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "cdf2rdf", "the eigenvalues are not a vector");
    }
    if v.ndim != 2 || v.shape[0] != v.shape[1] {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "cdf2rdf", "the eigenvectors are not a square matrix");
    }
    let m = w.shape[0];
    if v.shape[0] != m {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "cdf2rdf", "the eigenvalues and the eigenvectors hold a different count");
    }
    if m < 1 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "cdf2rdf", "the eigenvalues hold no values");
    }
    // The count of the values that hold an imaginary part.
    var n_complex: i64 = 0;
    var i: i64 = 0;
    while i < m {
        if cs_get(w.data, i).im != 0.0 { n_complex += 1; }
        i += 1;
    }
    if n_complex % 2 != 0 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "cdf2rdf", "the complex eigenvalues hold no pairs");
    }
    // The form of the eigenvalues holds the pairs in blocks.
    var wr: f64[>..] = [];
    i = 0;
    while i < m * m { wr.push(0.0); i += 1; }
    i = 0;
    while i < m {
        wr[i * m + i] = cs_get(w.data, i).re;
        i += 1;
    }
    // The matrix of the turn holds a quarter and three quarters.
    var u: f64[>..] = [];
    i = 0;
    while i < 2 * m * m { u.push(0.0); i += 1; }
    i = 0;
    while i < m {
        u[2 * (i * m + i)] = 1.0;
        i += 1;
    }
    // The pairs of the complex values give the blocks of the form.
    var j: i64 = 0;
    while j + 1 < m {
        if cs_get(w.data, j).im == 0.0 {
            j += 1;
            continue;
        }
        let k = j + 1;
        let im1 = cs_get(w.data, j).im;
        let im2 = cs_get(w.data, k).im;
        wr[j * m + k] = im1;
        wr[k * m + j] = im2;
        u[2 * (j * m + j)] = 0.0;
        u[2 * (j * m + j) + 1] = 0.5;
        u[2 * (j * m + k)] = 0.5;
        u[2 * (k * m + j) + 1] = -0.5;
        u[2 * (k * m + k)] = 0.5;
        j += 2;
    }
    // The eigenvectors are the product of the matrix of the turn with
    // the given eigenvectors.
    var vr: f64[>..] = [];
    i = 0;
    while i < m {
        var col: i64 = 0;
        while col < m {
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            var p: i64 = 0;
            while p < m {
                let ar = v.data[2 * (i * m + p)];
                let ai = v.data[2 * (i * m + p) + 1];
                let br = u[2 * (p * m + col)];
                let bi = u[2 * (p * m + col) + 1];
                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
                p += 1;
            }
            vr.push(sr);
            col += 1;
        }
        i += 1;
    }
    return mat2(copy(wr), m, m), mat2(copy(vr), m, m), no_fault();
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
