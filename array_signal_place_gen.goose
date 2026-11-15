// The constructions of the gain of a closed loop for the placement of
// the poles.
//
// place_gen builds a gain from the poles themselves. For each pole p
// the call solves the system (A - p I) x = B w, where w holds the
// powers of the pole from the offset up. The solutions form the
// transfer matrix X and the powers form the matrix V, and the gain is
// V X^-1. A pair of conjugate poles gives a gain with no imaginary
// part, because the powers of a pole and the power of its partner are
// conjugates. The system of each pole runs in real arithmetic on the
// block of two by two of a complex number.
namespace scientific;

import mathshim;

// place_cblock builds the real block of a complex matrix. The block
// holds the real part, the negative imaginary part, the imaginary
// part and the real part again. A solve of the block is a solve of the
// complex matrix.
fn place_cblock(zr: f64[:], zi: f64[:], n: i64, out: f64[>..]&) {
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            out[i * 2 * n + j] = zr[i * n + j];
            out[i * 2 * n + j + n] = -zi[i * n + j];
            out[(i + n) * 2 * n + j] = zi[i * n + j];
            out[(i + n) * 2 * n + j + n] = zr[i * n + j];
            j += 1;
        }
        i += 1;
    }
}

// place_gen_ok tells whether the determinant of a block names a matrix
// that a solve can turn. The test compares the logarithms, so no
// power of the scale can overflow.
fn place_gen_ok(dx: f64, scale: f64, side: i64) -> bool {
    if scale <= 0.0 { return false; }
    if dx == 0.0 { return false; }
    let ad = mathshim::abs_of(dx);
    let ld = mathshim::ln(ad);
    let th = mathshim::ln(1.0e-13) + side as f64 * mathshim::ln(scale);
    return ld > th;
}

// place_pow gives the power of a pole at the exponent e. The power is
// one at an exponent of zero.
fn place_pow(re: f64, im: f64, e: i64) -> f64, f64 {
    var r: f64 = 1.0;
    var i: f64 = 0.0;
    var k: i64 = 0;
    while k < e {
        let nr = r * re - i * im;
        i = r * im + i * re;
        r = nr;
        k += 1;
    }
    return r, i;
}

// place_gen builds the gain of the general path. The offset moves the
// powers of the poles that the matrix V holds. A descending order
// runs the powers from the highest one down, because some systems
// hold their reach in the high powers. The caller tries more than one
// set of powers. The result tells whether the construction solved its
// systems and found a matrix X of full size.
fn place_gen(a, b: f64[:], n: i64, m: i64, pr: f64[:], pi: f64[:], off: i64,
             desc: bool, gain: f64[>..]&) -> bool {
    var xr: f64[>..] = [];
    var xi: f64[>..] = [];
    var vr: f64[>..] = [];
    var vi: f64[>..] = [];
    var j: i64 = 0;
    while j < m * n {
        vr.push(0.0);
        vi.push(0.0);
        j += 1;
    }
    j = 0;
    while j < n * n {
        xr.push(0.0);
        xi.push(0.0);
        j += 1;
    }
    j = 0;
    while j < n {
        // The powers of the pole. The order runs up from the offset,
        // or down from the offset plus the count of the inputs.
        var wr: f64[>..] = [];
        var wi: f64[>..] = [];
        var k: i64 = 0;
        while k < m {
            var e = off + k;
            if desc { e = off + m - 1 - k; }
            let pr_pow, pi_pow = place_pow(pr[j], pi[j], e);
            wr.push(pr_pow);
            wi.push(pi_pow);
            vr[k * n + j] = pr_pow;
            vi[k * n + j] = pi_pow;
            k += 1;
        }
        // The right side holds the input matrix on the powers.
        var rr: f64[>..] = [];
        var ri: f64[>..] = [];
        var i2: i64 = 0;
        while i2 < n {
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            k = 0;
            while k < m {
                let bv = b[i2 * m + k];
                sr += bv * wr[k];
                si += bv * wi[k];
                k += 1;
            }
            rr.push(sr);
            ri.push(si);
            i2 += 1;
        }
        // The matrix of the pole: A - p I.
        var zr: f64[>..] = [];
        var zi: f64[>..] = [];
        i2 = 0;
        while i2 < n * n {
            zr.push(0.0);
            zi.push(0.0);
            i2 += 1;
        }
        i2 = 0;
        while i2 < n {
            k = 0;
            while k < n {
                zr[i2 * n + k] = a.data[i2 * n + k];
                k += 1;
            }
            zr[i2 * n + i2] -= pr[j];
            zi[i2 * n + i2] = -pi[j];
            i2 += 1;
        }
        var blk: f64[>..] = [];
        i2 = 0;
        while i2 < 4 * n * n { blk.push(0.0); i2 += 1; }
        place_cblock(zr, zi, n, blk);
        var rhs2: f64[>..] = [];
        i2 = 0;
        while i2 < n { rhs2.push(rr[i2]); i2 += 1; }
        i2 = 0;
        while i2 < n { rhs2.push(ri[i2]); i2 += 1; }
        let x2, fs = solve(mat2(copy(blk), 2 * n, 2 * n), flt_arr(copy(rhs2)));
        if fs.kind != F_NONE { return false; }
        // The column of the solution sits at the place j of the
        // matrices, read by rows.
        i2 = 0;
        while i2 < n {
            xr[j * n + i2] = x2.data[i2];
            xi[j * n + i2] = x2.data[n + i2];
            i2 += 1;
        }
        j += 1;
    }
    // The solve of the gain needs a matrix X of full size, so the
    // determinant of the block decides.
    var xblk: f64[>..] = [];
    j = 0;
    while j < 4 * n * n { xblk.push(0.0); j += 1; }
    place_cblock(xr, xi, n, xblk);
    let dx, fd = det(mat2(copy(xblk), 2 * n, 2 * n));
    if fd.kind != F_NONE { return false; }
    var scale: f64 = 0.0;
    j = 0;
    while j < xblk.len {
        let v = mathshim::abs_of(xblk[j]);
        if v > scale { scale = v; }
        j += 1;
    }
    if !place_gen_ok(dx, scale, 2 * n) { return false; }
    // The gain holds K X = V, so the row of every input solves the
    // system of the transpose of X. The rows of X hold the solution of
    // every pole, so the block of X is the system of the rows.
    j = 0;
    while j < m * n { gain.push(0.0); j += 1; }
    var k: i64 = 0;
    while k < m {
        var rhs: f64[>..] = [];
        j = 0;
        while j < n { rhs.push(vr[k * n + j]); j += 1; }
        j = 0;
        while j < n { rhs.push(vi[k * n + j]); j += 1; }
        let kk, fk = solve(mat2(copy(xblk), 2 * n, 2 * n), flt_arr(copy(rhs)));
        if fk.kind != F_NONE { return false; }
        j = 0;
        while j < n { gain[k * n + j] = kk.data[j]; j += 1; }
        k += 1;
    }
    return true;
}
