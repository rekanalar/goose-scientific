// The helpers of the complex QZ algorithm: the elements, the rotations and the chase.
//
namespace scientific;

import mathshim;

// The generalized Schur form of a complex pair. The pair A, B of side n
// becomes the upper triangular pair AA, BB with A = Q AA Z^H and
// B = Q BB Z^H. Q and Z are unitary and the diagonal of BB is real and
// not negative. The method is the single shift QZ iteration of the
// LAPACK routine zhgeqz: rotations of the rows reduce the pair to the
// form of a Hessenberg matrix and a triangular one, the iteration
// removes the subdiagonal of A, and a phase on each column makes the
// diagonal of B real. The eigenvalues are alpha / beta with a real
// beta. The swap of two eigenvalues follows the LAPACK routine ztgex2.
//
// Each matrix is a flat buffer of side n. It holds the real part and
// the imaginary part of each element side by side, row by row.

// cq_at gives the element i, j of a complex matrix of side n.
fn cq_at(d: f64[:], n: i64, i: i64, j: i64) -> Cplx {
    return Cplx { re: d[2 * (i * n + j)], im: d[2 * (i * n + j) + 1] };
}

// cq_put writes the element i, j of a complex matrix of side n.
fn cq_put(d: f64[>..]&, n: i64, i: i64, j: i64, v: Cplx) {
    d[2 * (i * n + j)] = v.re;
    d[2 * (i * n + j) + 1] = v.im;
}

// cq_abs1 gives the sum of the absolute real part and imaginary part.
// The iteration uses it as the measure of the size of a value.
fn cq_abs1(v: Cplx) -> f64 {
    return mathshim::abs_of(v.re) + mathshim::abs_of(v.im);
}

// cq_identity gives the complex unit matrix of side n.
fn cq_identity(n: i64) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n {
        d.push(0.0);
        d.push(0.0);
        i += 1;
    }
    i = 0;
    while i < n {
        cq_put(d, n, i, i, Cplx { re: 1.0, im: 0.0 });
        i += 1;
    }
    return d;
}

// cq_givens gives the rotation of the LAPACK routine zlartg. With the
// real cosine c and the complex sine s the matrix
// [[c, s], [-conj(s), c]] turns the vector (f, g) into (r, 0). The
// call gives c, s and r.
fn cq_givens(f: Cplx, g: Cplx) -> f64, Cplx, Cplx {
    let ga = qz_hypot(g.re, g.im);
    if ga == 0.0 { return 1.0, Cplx { re: 0.0, im: 0.0 }, f; }
    let fa = qz_hypot(f.re, f.im);
    if fa == 0.0 {
        return 0.0, Cplx { re: g.re / ga, im: -g.im / ga }, Cplx { re: ga, im: 0.0 };
    }
    let d = qz_hypot(fa, ga);
    let ur = f.re / fa;
    let ui = f.im / fa;
    let s = Cplx { re: (ur * g.re + ui * g.im) / d, im: (ui * g.re - ur * g.im) / d };
    return fa / d, s, Cplx { re: ur * d, im: ui * d };
}

// cq_rows turns the rows i and j of d over the columns c0 to c1 - 1.
// Row i becomes c * row i + s * row j and row j becomes
// -conj(s) * row i + c * row j, as in the LAPACK routine zrot.
fn cq_rows(d: f64[>..]&, n: i64, i: i64, j: i64, c: f64, s: Cplx, c0: i64, c1: i64) {
    var col: i64 = c0;
    while col < c1 {
        let xr = d[2 * (i * n + col)];
        let xi = d[2 * (i * n + col) + 1];
        let yr = d[2 * (j * n + col)];
        let yi = d[2 * (j * n + col) + 1];
        d[2 * (i * n + col)] = c * xr + s.re * yr - s.im * yi;
        d[2 * (i * n + col) + 1] = c * xi + s.re * yi + s.im * yr;
        d[2 * (j * n + col)] = -(s.re * xr + s.im * xi) + c * yr;
        d[2 * (j * n + col) + 1] = -(s.re * xi - s.im * xr) + c * yi;
        col += 1;
    }
}

// cq_cols turns the columns i and j of d over the rows r0 to r1 - 1.
// Column i becomes c * column i + s * column j and column j becomes
// -conj(s) * column i + c * column j.
fn cq_cols(d: f64[>..]&, n: i64, i: i64, j: i64, c: f64, s: Cplx, r0: i64, r1: i64) {
    var row: i64 = r0;
    while row < r1 {
        let xr = d[2 * (row * n + i)];
        let xi = d[2 * (row * n + i) + 1];
        let yr = d[2 * (row * n + j)];
        let yi = d[2 * (row * n + j) + 1];
        d[2 * (row * n + i)] = c * xr + s.re * yr - s.im * yi;
        d[2 * (row * n + i) + 1] = c * xi + s.re * yi + s.im * yr;
        d[2 * (row * n + j)] = -(s.re * xr + s.im * xi) + c * yr;
        d[2 * (row * n + j) + 1] = -(s.re * xi - s.im * xr) + c * yi;
        row += 1;
    }
}

// cq_scale_col multiplies the column j of d over the rows r0 to r1 - 1
// by the value u.
fn cq_scale_col(d: f64[>..]&, n: i64, j: i64, u: Cplx, r0: i64, r1: i64) {
    var row: i64 = r0;
    while row < r1 {
        cq_put(d, n, row, j, c_mul(cq_at(d, n, row, j), u));
        row += 1;
    }
}

// cq_reduce brings the pair to the form of a Hessenberg matrix a and
// an upper triangular matrix b. Rotations of the rows make b
// triangular. Then rotations of the rows clear a under the first
// subdiagonal and rotations of the columns keep b triangular. The
// matrices q and z start as unit matrices and collect the transforms.
fn cq_reduce(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64) {
    var j: i64 = 0;
    while j < n - 1 {
        var i: i64 = n - 1;
        while i > j {
            let g = cq_at(b, n, i, j);
            if cq_abs1(g) != 0.0 {
                let c, s, r = cq_givens(cq_at(b, n, i - 1, j), g);
                cq_rows(a, n, i - 1, i, c, s, 0, n);
                cq_rows(b, n, i - 1, i, c, s, 0, n);
                cq_cols(q, n, i - 1, i, c, c_conj(s), 0, n);
                cq_put(b, n, i, j, Cplx { re: 0.0, im: 0.0 });
            }
            i -= 1;
        }
        j += 1;
    }
    j = 0;
    while j < n - 2 {
        var i: i64 = n - 1;
        while i > j + 1 {
            let c, s, r = cq_givens(cq_at(a, n, i - 1, j), cq_at(a, n, i, j));
            cq_rows(a, n, i - 1, i, c, s, 0, n);
            cq_rows(b, n, i - 1, i, c, s, 0, n);
            cq_cols(q, n, i - 1, i, c, c_conj(s), 0, n);
            cq_put(a, n, i, j, Cplx { re: 0.0, im: 0.0 });
            cq_fix_b(a, b, z, n, i);
            i -= 1;
        }
        j += 1;
    }
}

// cq_fix_b clears the value that a rotation of the rows left below the
// diagonal of b at the place (i, i - 1). A rotation of the columns
// i - 1 and i does it, and z collects the same rotation.
fn cq_fix_b(a: f64[>..]&, b: f64[>..]&, z: f64[>..]&, n: i64, i: i64) {
    let x = cq_at(b, n, i, i - 1);
    if cq_abs1(x) == 0.0 { return; }
    let y = cq_at(b, n, i, i);
    let dx = qz_hypot(x.re, x.im);
    let dy0 = qz_hypot(y.re, y.im);
    let dy = qz_hypot(dx, dy0);
    var c2: f64 = 0.0;
    var sp = Cplx { re: 1.0, im: 0.0 };
    if dy0 != 0.0 {
        c2 = dy0 / dy;
        sp = c_scale(c_div(x, y), -c2);
    }
    cq_cols(a, n, i - 1, i, c2, sp, 0, n);
    cq_cols(b, n, i - 1, i, c2, sp, 0, n);
    cq_cols(z, n, i - 1, i, c2, sp, 0, n);
    cq_put(b, n, i, i - 1, Cplx { re: 0.0, im: 0.0 });
}

// cq_std_b makes the diagonal entry j of b real and not negative. A
// phase on the column j of a, b and z does it. The eigenvalue then has
// a real beta.
fn cq_std_b(a: f64[>..]&, b: f64[>..]&, z: f64[>..]&, n: i64, j: i64) {
    let d = cq_at(b, n, j, j);
    let m = qz_hypot(d.re, d.im);
    if m > QZ_SAFMIN {
        let signbc = Cplx { re: d.re / m, im: -d.im / m };
        cq_put(b, n, j, j, Cplx { re: m, im: 0.0 });
        cq_scale_col(b, n, j, signbc, 0, j);
        cq_scale_col(a, n, j, signbc, 0, j + 1);
        cq_scale_col(z, n, j, signbc, 0, n);
    } else {
        cq_put(b, n, j, j, Cplx { re: 0.0, im: 0.0 });
    }
}

// cq_divr gives a complex value divided by a real number.
fn cq_divr(v: Cplx, r: f64) -> Cplx {
    return Cplx { re: v.re / r, im: v.im / r };
}

// cq_shift gives the Wilkinson shift of the step. It is the eigenvalue
// of the block of two rows at the bottom of a b^-1 that lies nearest to
// its last element.
fn cq_shift(a: f64[:], b: f64[:], n: i64, il: i64) -> Cplx {
    let i1 = il - 1;
    let i2 = il;
    let u12 = c_div(cq_at(b, n, i1, i2), cq_at(b, n, i2, i2));
    let ad11 = c_div(cq_at(a, n, i1, i1), cq_at(b, n, i1, i1));
    let ad21 = c_div(cq_at(a, n, i2, i1), cq_at(b, n, i1, i1));
    let ad12 = c_div(cq_at(a, n, i1, i2), cq_at(b, n, i2, i2));
    let ad22 = c_div(cq_at(a, n, i2, i2), cq_at(b, n, i2, i2));
    let abi22 = c_sub(ad22, c_mul(u12, ad21));
    let abi12 = c_sub(ad12, c_mul(u12, ad11));
    var shift = abi22;
    let ct = c_mul(c_sqrt(abi12), c_sqrt(ad21));
    if cq_abs1(ct) != 0.0 {
        let x = c_scale(c_sub(ad11, shift), 0.5);
        var temp = cq_abs1(ct);
        let temp2 = cq_abs1(x);
        if temp2 > temp { temp = temp2; }
        let xu = cq_divr(x, temp);
        let cu = cq_divr(ct, temp);
        var y = c_scale(c_sqrt(c_add(c_mul(xu, xu), c_mul(cu, cu))), temp);
        if temp2 > 0.0 {
            let xv = cq_divr(x, temp2);
            if xv.re * y.re + xv.im * y.im < 0.0 { y = c_scale(y, -1.0); }
        }
        shift = c_sub(shift, c_mul(ct, c_div(ct, c_add(x, y))));
    }
    return shift;
}

// cq_zero_chase moves a zero of the diagonal of b from the place j to
// the bottom of the window and clears the subdiagonal of a on the way.
fn cq_zero_chase(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, j: i64, il: i64) {
    var jch: i64 = j;
    while jch < il {
        let c, s, r = cq_givens(cq_at(b, n, jch, jch + 1), cq_at(b, n, jch + 1, jch + 1));
        cq_put(b, n, jch, jch + 1, r);
        cq_put(b, n, jch + 1, jch + 1, Cplx { re: 0.0, im: 0.0 });
        if jch < n - 2 {
            cq_rows(b, n, jch, jch + 1, c, s, jch + 2, n);
        }
        cq_rows(a, n, jch, jch + 1, c, s, jch - 1, n);
        cq_cols(q, n, jch, jch + 1, c, c_conj(s), 0, n);
        let c2, s2, r2 = cq_givens(cq_at(a, n, jch + 1, jch), cq_at(a, n, jch + 1, jch - 1));
        cq_put(a, n, jch + 1, jch, r2);
        cq_put(a, n, jch + 1, jch - 1, Cplx { re: 0.0, im: 0.0 });
        cq_cols(a, n, jch, jch - 1, c2, s2, 0, jch + 1);
        cq_cols(b, n, jch, jch - 1, c2, s2, 0, jch);
        cq_cols(z, n, jch, jch - 1, c2, s2, 0, n);
        jch += 1;
    }
}

// cq_block_split splits a block of one rows at the place j where the
// diagonal of b is zero and the subdiagonal of a is small. The zero
// moves up and the rotations follow it.
fn cq_block_split(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, j: i64, il: i64, btol: f64, azr2: bool) -> i64 {
    var az = azr2;
    var jch: i64 = j;
    while jch < il {
        let c, s, r = cq_givens(cq_at(a, n, jch, jch), cq_at(a, n, jch + 1, jch));
        cq_put(a, n, jch, jch, r);
        cq_put(a, n, jch + 1, jch, Cplx { re: 0.0, im: 0.0 });
        cq_rows(a, n, jch, jch + 1, c, s, jch + 1, n);
        cq_rows(b, n, jch, jch + 1, c, s, jch + 1, n);
        cq_cols(q, n, jch, jch + 1, c, c_conj(s), 0, n);
        if az {
            let v = cq_at(a, n, jch, jch - 1);
            cq_put(a, n, jch, jch - 1, c_scale(v, c));
            az = false;
        }
        if cq_abs1(cq_at(b, n, jch + 1, jch + 1)) >= btol {
            return jch + 1;
        }
        cq_put(b, n, jch + 1, jch + 1, Cplx { re: 0.0, im: 0.0 });
        jch += 1;
    }
    return -1;
}

// cq_qz runs the single shift QZ iteration of the LAPACK routine
// zhgeqz on a pair in the form of a Hessenberg matrix a and a
// triangular one b. The matrices q and z collect the transforms. The
// value turns gives the sweeps that one eigenvalue may take. The call
// gives false when the iteration does not converge. After the call the
// pair is triangular and the diagonal of b is real and not negative.
fn cq_qz(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, turns: i64) -> bool {
    let ulp = 4.440892098500626e-16;
    var anorm: f64 = 0.0;
    var bnorm: f64 = 0.0;
    var i: i64 = 0;
    while i < n * n {
        let va = qz_hypot(a[2 * i], a[2 * i + 1]);
        let vb = qz_hypot(b[2 * i], b[2 * i + 1]);
        anorm += va * va;
        bnorm += vb * vb;
        i += 1;
    }
    anorm = mathshim::root(anorm);
    bnorm = mathshim::root(bnorm);
    var atol = anorm * ulp;
    if atol < QZ_SAFMIN { atol = QZ_SAFMIN; }
    var btol = bnorm * ulp;
    if btol < QZ_SAFMIN { btol = QZ_SAFMIN; }
    let maxit = turns * n;
    var il: i64 = n - 1;
    var iter: i64 = 0;
    var es = Cplx { re: 0.0, im: 0.0 };
    var sweeps: i64 = 0;
    while sweeps < maxit {
        sweeps += 1;
        // The place ilast holds the lowest eigenvalue that the iteration
        // has not found yet. A small subdiagonal of a or a zero diagonal
        // of b splits it off.
        if il == 0 {
            cq_std_b(a, b, z, n, 0);
            return true;
        }
        var tol = ulp * (cq_abs1(cq_at(a, n, il, il)) + cq_abs1(cq_at(a, n, il - 1, il - 1)));
        if tol < QZ_SAFMIN { tol = QZ_SAFMIN; }
        var act: i64 = 0;
        var first: i64 = -1;
        if cq_abs1(cq_at(a, n, il, il - 1)) <= tol {
            cq_put(a, n, il, il - 1, Cplx { re: 0.0, im: 0.0 });
            act = 1;
        } else if cq_abs1(cq_at(b, n, il, il)) <= btol {
            cq_put(b, n, il, il, Cplx { re: 0.0, im: 0.0 });
            act = 2;
        } else {
            var j: i64 = il - 1;
            while j >= 0 && act == 0 {
                var azro = false;
                if j == 0 {
                    azro = true;
                } else {
                    var t1 = ulp * (cq_abs1(cq_at(a, n, j, j)) + cq_abs1(cq_at(a, n, j - 1, j - 1)));
                    if t1 < QZ_SAFMIN { t1 = QZ_SAFMIN; }
                    if cq_abs1(cq_at(a, n, j, j - 1)) <= t1 {
                        cq_put(a, n, j, j - 1, Cplx { re: 0.0, im: 0.0 });
                        azro = true;
                    }
                }
                if cq_abs1(cq_at(b, n, j, j)) < btol {
                    cq_put(b, n, j, j, Cplx { re: 0.0, im: 0.0 });
                    var azr2 = false;
                    if !azro {
                        if cq_abs1(cq_at(a, n, j, j - 1)) * cq_abs1(cq_at(a, n, j + 1, j)) <= cq_abs1(cq_at(a, n, j, j)) * atol {
                            azr2 = true;
                        }
                    }
                    if azro || azr2 {
                        let res = cq_block_split(a, b, q, z, n, j, il, btol, azr2);
                        if res < 0 {
                            act = 2;
                        } else if res >= il {
                            act = 1;
                        } else {
                            act = 3;
                            first = res;
                        }
                    } else {
                        cq_zero_chase(a, b, q, z, n, j, il);
                        act = 2;
                    }
                } else if azro {
                    act = 3;
                    first = j;
                }
                j -= 1;
            }
            if act == 0 { return false; }
        }
        if act == 2 {
            // The diagonal of b at the bottom is zero. One rotation of
            // the columns clears the subdiagonal of a.
            let c, s, r = cq_givens(cq_at(a, n, il, il), cq_at(a, n, il, il - 1));
            cq_put(a, n, il, il, r);
            cq_put(a, n, il, il - 1, Cplx { re: 0.0, im: 0.0 });
            cq_cols(a, n, il, il - 1, c, s, 0, il);
            cq_cols(b, n, il, il - 1, c, s, 0, il);
            cq_cols(z, n, il, il - 1, c, s, 0, n);
        }
        if act == 3 {
            // One step of the shifted iteration on the window first to
            // ilast.
            iter += 1;
            var shift = Cplx { re: 0.0, im: 0.0 };
            if iter % 10 != 0 {
                shift = cq_shift(a, b, n, il);
            } else {
                if iter % 20 == 0 && cq_abs1(cq_at(b, n, il, il)) > QZ_SAFMIN {
                    es = c_add(es, c_div(cq_at(a, n, il, il), cq_at(b, n, il, il)));
                } else {
                    es = c_add(es, c_div(cq_at(a, n, il, il - 1), cq_at(b, n, il - 1, il - 1)));
                }
                shift = es;
            }
            // The window starts at the place with two small subdiagonal
            // values in a row.
            var istart: i64 = -1;
            var ctemp = Cplx { re: 0.0, im: 0.0 };
            var j: i64 = il - 1;
            while j > first {
                ctemp = c_sub(cq_at(a, n, j, j), c_mul(shift, cq_at(b, n, j, j)));
                var temp = cq_abs1(ctemp);
                var temp2 = cq_abs1(cq_at(a, n, j + 1, j));
                var tempr = temp;
                if temp2 > tempr { tempr = temp2; }
                if tempr < 1.0 && tempr != 0.0 {
                    temp = temp / tempr;
                    temp2 = temp2 / tempr;
                }
                if cq_abs1(cq_at(a, n, j, j - 1)) * temp2 <= temp * atol {
                    istart = j;
                    break;
                }
                j -= 1;
            }
            if istart < 0 {
                istart = first;
                ctemp = c_sub(cq_at(a, n, istart, istart), c_mul(shift, cq_at(b, n, istart, istart)));
            }
            let c0, s0, r0 = cq_givens(ctemp, cq_at(a, n, istart + 1, istart));
            var c = c0;
            var s = s0;
            j = istart;
            while j < il {
                if j > istart {
                    let cc, ss, rr = cq_givens(cq_at(a, n, j, j - 1), cq_at(a, n, j + 1, j - 1));
                    cq_put(a, n, j, j - 1, rr);
                    cq_put(a, n, j + 1, j - 1, Cplx { re: 0.0, im: 0.0 });
                    c = cc;
                    s = ss;
                }
                cq_rows(a, n, j, j + 1, c, s, j, n);
                cq_rows(b, n, j, j + 1, c, s, j, n);
                cq_cols(q, n, j, j + 1, c, c_conj(s), 0, n);
                let c2, s2, r2 = cq_givens(cq_at(b, n, j + 1, j + 1), cq_at(b, n, j + 1, j));
                cq_put(b, n, j + 1, j + 1, r2);
                cq_put(b, n, j + 1, j, Cplx { re: 0.0, im: 0.0 });
                var r1 = j + 3;
                if r1 > il + 1 { r1 = il + 1; }
                cq_cols(a, n, j + 1, j, c2, s2, 0, r1);
                cq_cols(b, n, j + 1, j, c2, s2, 0, j + 1);
                cq_cols(z, n, j + 1, j, c2, s2, 0, n);
                j += 1;
            }
        }
        if act != 3 {
            // The eigenvalue at the place ilast is found.
            cq_std_b(a, b, z, n, il);
            il -= 1;
            iter = 0;
            es = Cplx { re: 0.0, im: 0.0 };
            if il < 0 { return true; }
        }
    }
    return false;
}

