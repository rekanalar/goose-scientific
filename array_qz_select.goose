// The selection and the order of the eigenvalues of the QZ algorithm.
//
namespace scientific;

import mathshim;

// qz_flags_of sets the flag of each diagonal block once. A flag is 1 for
// a block that the rule chooses and 0 for the others. The forms a and b
// hold the pair divided by the scales sa and sb.
fn qz_flags_of(a: f64[:], b: f64[:], n: i64, rule: i64, sa: f64, sb: f64, flags: f64[>..]&) {
    var j: i64 = 0;
    while j < n {
        var sz: i64 = 1;
        if j + 1 < n && a[(j + 1) * n + j] != 0.0 { sz = 2; }
        if qz_chosen(a, b, n, j, sz, rule, sa, sb) { flags.push(1.0); } else { flags.push(0.0); }
        j += sz;
    }
}

// qz_flags_by sets the flag of each diagonal block once. The block of
// the caller chooses the block. It takes the alpha and the beta of the
// block, and gives true for a chosen block. The values come back to the
// scale of the caller, so the block reads the true alpha and the true
// beta. A pair of complex eigenvalues is one block. It gets the value
// with the positive imaginary part.
fn qz_flags_by<F: fn(Cplx, f64) -> bool>(a: f64[:], b: f64[:], n: i64, sa: f64, sb: f64, flags: f64[>..]&) {
    var j: i64 = 0;
    while j < n {
        var sz: i64 = 1;
        if j + 1 < n && a[(j + 1) * n + j] != 0.0 { sz = 2; }
        var al = Cplx { re: 0.0, im: 0.0 };
        var be: f64 = 0.0;
        if sz == 2 {
            let re, im, det = qz_roots2(a, b, n, j);
            let bt = mathshim::root(b[j * n + j] * b[(j + 1) * n + j + 1]);
            al = Cplx { re: re * bt * sa, im: im * bt * sa };
            be = bt * sb;
        } else {
            var sg: f64 = 1.0;
            if b[j * n + j] < 0.0 { sg = -1.0; }
            al = Cplx { re: a[j * n + j] * sa * sg, im: 0.0 };
            be = mathshim::abs_of(b[j * n + j]) * sb;
        }
        if F(al, be) { flags.push(1.0); } else { flags.push(0.0); }
        j += sz;
    }
}

// qz_partition puts the blocks that the flags choose before the others.
// The order inside each group stays as it was. A swap moves the flags
// with the blocks, and each swap lowers the count of unchosen blocks
// before a chosen block. The loop ends after at most n * n swaps, for
// any flags. The call gives false when a swap fails.
fn qz_partition(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, flags: f64[>..]&) -> bool {
    var again = true;
    while again {
        again = false;
        var j: i64 = 0;
        var k: i64 = 0;
        var pj: i64 = -1;
        var pk: i64 = -1;
        var psz: i64 = 0;
        while j < n && !again {
            var sz: i64 = 1;
            if j + 1 < n && a[(j + 1) * n + j] != 0.0 { sz = 2; }
            if flags[k] > 0.5 && pj >= 0 && flags[pk] < 0.5 {
                if !qz_swap(a, b, q, z, n, pj, psz, sz) { return false; }
                flags[pk] = 1.0;
                flags[k] = 0.0;
                again = true;
            } else {
                pj = j;
                pk = k;
                psz = sz;
                j += sz;
                k += 1;
            }
        }
    }
    qz_signs(a, b, q, n);
    return true;
}

// qz_eigs appends the eigenvalues of the pair in the form
// alpha / beta. The values alpha come as pairs of the real and the
// imaginary part. A pair of complex eigenvalues gives the value with
// the positive imaginary part first. Both values of the pair hold the
// same beta, the geometric mean of the two entries of b.
fn qz_eigs(a: f64[:], b: f64[:], n: i64, al: f64[>..]&, be: f64[>..]&) {
    var j: i64 = 0;
    while j < n {
        if j + 1 < n && a[(j + 1) * n + j] != 0.0 {
            let re, im, det = qz_roots2(a, b, n, j);
            let beta = mathshim::root(b[j * n + j] * b[(j + 1) * n + j + 1]);
            al.push(re * beta);
            al.push(im * beta);
            be.push(beta);
            al.push(re * beta);
            al.push(-im * beta);
            be.push(beta);
            j += 2;
        } else {
            al.push(a[j * n + j]);
            al.push(0.0);
            be.push(b[j * n + j]);
            j += 1;
        }
    }
}

// qz_cright multiplies the columns j and j + 1 of the complex matrix d
// by the matrix w of two rows from the right, in the rows r0 to r1. The
// matrix w holds four complex values: w00, w01, w10 and w11.
fn qz_cright(d: f64[>..]&, n: i64, j: i64, w: f64[:], r0: i64, r1: i64) {
    var r: i64 = r0;
    while r < r1 {
        let x = Cplx { re: d[2 * (r * n + j)], im: d[2 * (r * n + j) + 1] };
        let y = Cplx { re: d[2 * (r * n + j + 1)], im: d[2 * (r * n + j + 1) + 1] };
        let w00 = Cplx { re: w[0], im: w[1] };
        let w01 = Cplx { re: w[2], im: w[3] };
        let w10 = Cplx { re: w[4], im: w[5] };
        let w11 = Cplx { re: w[6], im: w[7] };
        let nx = c_add(c_mul(x, w00), c_mul(y, w10));
        let ny = c_add(c_mul(x, w01), c_mul(y, w11));
        d[2 * (r * n + j)] = nx.re;
        d[2 * (r * n + j) + 1] = nx.im;
        d[2 * (r * n + j + 1)] = ny.re;
        d[2 * (r * n + j + 1) + 1] = ny.im;
        r += 1;
    }
}

// qz_cleft_h multiplies the rows j and j + 1 of the complex matrix d by
// the conjugate transpose of w from the left, in the columns c0 to c1.
fn qz_cleft_h(d: f64[>..]&, n: i64, j: i64, w: f64[:], c0: i64, c1: i64) {
    var c: i64 = c0;
    while c < c1 {
        let x = Cplx { re: d[2 * (j * n + c)], im: d[2 * (j * n + c) + 1] };
        let y = Cplx { re: d[2 * ((j + 1) * n + c)], im: d[2 * ((j + 1) * n + c) + 1] };
        let h00 = Cplx { re: w[0], im: -w[1] };
        let h01 = Cplx { re: w[4], im: -w[5] };
        let h10 = Cplx { re: w[2], im: -w[3] };
        let h11 = Cplx { re: w[6], im: -w[7] };
        let nx = c_add(c_mul(h00, x), c_mul(h01, y));
        let ny = c_add(c_mul(h10, x), c_mul(h11, y));
        d[2 * (j * n + c)] = nx.re;
        d[2 * (j * n + c) + 1] = nx.im;
        d[2 * ((j + 1) * n + c)] = ny.re;
        d[2 * ((j + 1) * n + c) + 1] = ny.im;
        c += 1;
    }
}

// qz_unitary2 gives the unitary matrix of two rows whose first column
// is the unit vector (x0, x1). Its second column is (-conj x1, conj x0).
fn qz_unitary2(x0: Cplx, x1: Cplx) -> f64[>..] {
    var w: f64[>..] = [];
    w.push(x0.re); w.push(x0.im);
    w.push(-x1.re); w.push(x1.im);
    w.push(x1.re); w.push(x1.im);
    w.push(x0.re); w.push(-x0.im);
    return w;
}

// qz_unit_of turns the vector (x0, x1) into a unit vector.
fn qz_unit_of(x0: Cplx, x1: Cplx) -> Cplx, Cplx {
    let nr = qz_hypot(c_abs(x0), c_abs(x1));
    if nr == 0.0 { return Cplx { re: 1.0, im: 0.0 }, Cplx { re: 0.0, im: 0.0 }; }
    return c_scale(x0, 1.0 / nr), c_scale(x1, 1.0 / nr);
}

// qz_to_complex turns the real form of a pair into the triangular
// complex form. Each block of two rows gets two unitary matrices. The
// first column of the right one is the eigenvector of the block. The
// first column of the left one points along b times that vector. A
// phase on each row then makes the diagonal of b real. The results
// hold the real part and the imaginary part of each value side by side.
// The call appends them to the outputs.
fn qz_to_complex(a: f64[:], b: f64[:], q: f64[:], z: f64[:], n: i64, oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&) {
    var ca: f64[>..] = [];
    var cb: f64[>..] = [];
    var cq: f64[>..] = [];
    var cz: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n {
        ca.push(a[i]); ca.push(0.0);
        cb.push(b[i]); cb.push(0.0);
        cq.push(q[i]); cq.push(0.0);
        cz.push(z[i]); cz.push(0.0);
        i += 1;
    }
    var j: i64 = 0;
    while j < n {
        if j + 1 < n && a[(j + 1) * n + j] != 0.0 {
            let re, im, det = qz_roots2(a, b, n, j);
            let lam = Cplx { re: re, im: im };
            let b00 = Cplx { re: b[j * n + j], im: 0.0 };
            let b11 = Cplx { re: b[(j + 1) * n + j + 1], im: 0.0 };
            let r00 = c_sub(Cplx { re: a[j * n + j], im: 0.0 }, c_mul(lam, b00));
            let r01 = Cplx { re: a[j * n + j + 1], im: 0.0 };
            let r10 = Cplx { re: a[(j + 1) * n + j], im: 0.0 };
            let r11 = c_sub(Cplx { re: a[(j + 1) * n + j + 1], im: 0.0 }, c_mul(lam, b11));
            var x0 = Cplx { re: 0.0, im: 0.0 };
            var x1 = Cplx { re: 0.0, im: 0.0 };
            if c_abs(r00) + c_abs(r01) >= c_abs(r10) + c_abs(r11) {
                x0 = c_scale(r01, -1.0);
                x1 = r00;
            } else {
                x0 = c_scale(r11, -1.0);
                x1 = r10;
            }
            let ux0, ux1 = qz_unit_of(x0, x1);
            let vw = qz_unitary2(ux0, ux1);
            let y0, y1 = qz_unit_of(c_mul(b00, ux0), c_mul(b11, ux1));
            let uw = qz_unitary2(y0, y1);
            qz_cright(ca, n, j, vw, 0, j + 2);
            qz_cright(cb, n, j, vw, 0, j + 2);
            qz_cright(cz, n, j, vw, 0, n);
            qz_cleft_h(ca, n, j, uw, j, n);
            qz_cleft_h(cb, n, j, uw, j, n);
            qz_cright(cq, n, j, uw, 0, n);
            ca[2 * ((j + 1) * n + j)] = 0.0;
            ca[2 * ((j + 1) * n + j) + 1] = 0.0;
            cb[2 * ((j + 1) * n + j)] = 0.0;
            cb[2 * ((j + 1) * n + j) + 1] = 0.0;
            j += 2;
        } else {
            j += 1;
        }
    }
    var k: i64 = 0;
    while k < n {
        let d = Cplx { re: cb[2 * (k * n + k)], im: cb[2 * (k * n + k) + 1] };
        let mag = c_abs(d);
        if mag > 0.0 && d.im != 0.0 {
            let inv = Cplx { re: d.re / mag, im: -d.im / mag };
            let ph = Cplx { re: d.re / mag, im: d.im / mag };
            var c: i64 = 0;
            while c < n {
                let x = c_mul(Cplx { re: ca[2 * (k * n + c)], im: ca[2 * (k * n + c) + 1] }, inv);
                ca[2 * (k * n + c)] = x.re;
                ca[2 * (k * n + c) + 1] = x.im;
                let y = c_mul(Cplx { re: cb[2 * (k * n + c)], im: cb[2 * (k * n + c) + 1] }, inv);
                cb[2 * (k * n + c)] = y.re;
                cb[2 * (k * n + c) + 1] = y.im;
                c += 1;
            }
            var r: i64 = 0;
            while r < n {
                let w = c_mul(Cplx { re: cq[2 * (r * n + k)], im: cq[2 * (r * n + k) + 1] }, ph);
                cq[2 * (r * n + k)] = w.re;
                cq[2 * (r * n + k) + 1] = w.im;
                r += 1;
            }
            cb[2 * (k * n + k)] = mag;
            cb[2 * (k * n + k) + 1] = 0.0;
        }
        k += 1;
    }
    i = 0;
    while i < 2 * n * n {
        oa.push(ca[i]);
        ob.push(cb[i]);
        oq.push(cq[i]);
        oz.push(cz[i]);
        i += 1;
    }
}

