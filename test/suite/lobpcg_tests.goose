// The tests of the group lobpcg. Run them with SCIENTIFIC_SUITES=lobpcg.
//
// The pins come from converged runs of SciPy. The tests compare values,
// the orthonormality of the vectors and the residuals. They never compare
// a vector element for element, because the sign of a vector is free.
namespace lobpcg_tests;

import scientific from "../..";
import common;
import lobpcg_pins;

// lt_matrix builds a sparse matrix from the pinned entries.
fn lt_matrix(n: i64, ri: f64[:], ci: f64[:], va: f64[:]) -> scientific::Csr {
    let s, f = scientific::csr_from_triplets(n, n, ri, ci, va);
    assert(f.kind == scientific::F_NONE);
    return s;
}

// lt_gram gives the largest error of V^T B V against the identity. The
// matrix b may be absent, and then the identity serves as B.
fn lt_gram(v: scientific::Arr&, n: i64, b: scientific::Csr&) -> f64 {
    let k = v.shape[1];
    var worst: f64 = 0.0;
    var j: i64 = 0;
    while j < k {
        var j2: i64 = 0;
        while j2 < k {
            var dot: f64 = 0.0;
            var i: i64 = 0;
            while i < n {
                var bv = v.data[i * k + j2];
                if b.rows != 0 {
                    bv = 0.0;
                    var p = scientific::csr_ptr(b, i);
                    while p < scientific::csr_ptr(b, i + 1) {
                        bv += scientific::csr_val(b, p) * v.data[scientific::csr_col(b, p) * k + j2];
                        p += 1;
                    }
                }
                dot += v.data[i * k + j] * bv;
                i += 1;
            }
            var want: f64 = 0.0;
            if j == j2 { want = 1.0; }
            let er = common::mathshim_abs(dot - want);
            if er > worst { worst = er; }
            j2 += 1;
        }
        j += 1;
    }
    return worst;
}

// lt_residual gives the largest residual norm of A v = w B v over the columns.
fn lt_residual(a: scientific::Csr&, b: scientific::Csr&, w: scientific::Arr&, v: scientific::Arr&, n: i64) -> f64 {
    let k = v.shape[1];
    var worst: f64 = 0.0;
    var j: i64 = 0;
    while j < k {
        var sum: f64 = 0.0;
        var i: i64 = 0;
        while i < n {
            var av: f64 = 0.0;
            var p = scientific::csr_ptr(a, i);
            while p < scientific::csr_ptr(a, i + 1) {
                av += scientific::csr_val(a, p) * v.data[scientific::csr_col(a, p) * k + j];
                p += 1;
            }
            var bv = v.data[i * k + j];
            if b.rows != 0 {
                bv = 0.0;
                var q = scientific::csr_ptr(b, i);
                while q < scientific::csr_ptr(b, i + 1) {
                    bv += scientific::csr_val(b, q) * v.data[scientific::csr_col(b, q) * k + j];
                    q += 1;
                }
            }
            let d = av - w.data[j] * bv;
            sum += d * d;
            i += 1;
        }
        let nr = common::mathshim_sqrt(sum);
        if nr > worst { worst = nr; }
        j += 1;
    }
    return worst;
}

// lt_values checks the values of a result against the pins to a tolerance.
fn lt_values(w: scientific::Arr&, want: f64[:], tol: f64) {
    assert(w.data.len == want.len);
    var i: i64 = 0;
    while i < want.len { assert(common::close(w.data[i], want[i], tol)); i += 1; }
}

// The standard problem gives the smallest and the largest values.
fn test_lobpcg_standard() {
    let n: i64 = 120;
    let a = lt_matrix(n, lobpcg_pins::CHAIN_RI, lobpcg_pins::CHAIN_CI, lobpcg_pins::CHAIN_VA);
    let none = scientific::csr_empty();
    let x = common::mat(lobpcg_pins::CHAIN_X, n, 3);
    let w, v, it, lh, rh, f = scientific::lobpcg(a, x, scientific::csr_empty(), scientific::csr_empty(), scientific::empty_arr(), 1.0e-10, 500, false);
    assert(f.kind == scientific::F_NONE);
    lt_values(w, lobpcg_pins::W_SMALL, 1.0e-8);
    lt_values(w, lobpcg_pins::W_EXACT, 1.0e-8);
    assert(lt_gram(v, n, none) <= 1.0e-10);
    assert(lt_residual(a, none, w, v, n) <= 1.0e-8);
    assert(it > 0 && it <= 500);
    let x2 = common::mat(lobpcg_pins::CHAIN_X, n, 3);
    let w2, v2, it2, lh2, rh2, f2 = scientific::lobpcg(a, x2, scientific::csr_empty(), scientific::csr_empty(), scientific::empty_arr(), 1.0e-10, 500, true);
    assert(f2.kind == scientific::F_NONE);
    lt_values(w2, lobpcg_pins::W_LARGE, 1.0e-8);
    assert(w2.data[0] > w2.data[1] && w2.data[1] > w2.data[2]);
    assert(lt_gram(v2, n, none) <= 1.0e-10);
}

// The default tolerance gives values near the exact ones.
fn test_lobpcg_default_tolerance() {
    let n: i64 = 120;
    let a = lt_matrix(n, lobpcg_pins::CHAIN_RI, lobpcg_pins::CHAIN_CI, lobpcg_pins::CHAIN_VA);
    let x = common::mat(lobpcg_pins::CHAIN_X, n, 3);
    let w, v, it, lh, rh, f = scientific::lobpcg(a, x, scientific::csr_empty(), scientific::csr_empty(), scientific::empty_arr(), 0.0, 500, false);
    assert(f.kind == scientific::F_NONE);
    lt_values(w, lobpcg_pins::W_EXACT, 1.0e-5);
}

// A problem that is small against the block size takes the dense path.
fn test_lobpcg_dense() {
    let a = lt_matrix(6, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let x = scientific::mat2(scientific::copy_slice([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]), 6, 3);
    let w, v, it, lh, rh, f = scientific::lobpcg(a, x, scientific::csr_empty(), scientific::csr_empty(), scientific::empty_arr(), 0.0, 20, false);
    assert(f.kind == scientific::F_NONE && it == 0 && lh.data.len == 0);
    assert(common::close(w.data[0], 1.0, 1.0e-12) && common::close(w.data[1], 2.0, 1.0e-12) && common::close(w.data[2], 3.0, 1.0e-12));
    let x2 = scientific::mat2(scientific::copy_slice([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]), 6, 3);
    let w2, v2, it2, lh2, rh2, f2 = scientific::lobpcg(a, x2, scientific::csr_empty(), scientific::csr_empty(), scientific::empty_arr(), 0.0, 20, true);
    assert(f2.kind == scientific::F_NONE);
    assert(common::close(w2.data[0], 6.0, 1.0e-12) && common::close(w2.data[2], 4.0, 1.0e-12));
}

// lt_ytv gives the largest absolute value of Y^T B V.
fn lt_ytv(y: f64[:], ny: i64, v: scientific::Arr&, n: i64) -> f64 {
    let k = v.shape[1];
    var worst: f64 = 0.0;
    var c: i64 = 0;
    while c < ny {
        var j: i64 = 0;
        while j < k {
            var dot: f64 = 0.0;
            var i: i64 = 0;
            while i < n { dot += y[i * ny + c] * v.data[i * k + j]; i += 1; }
            if common::mathshim_abs(dot) > worst { worst = common::mathshim_abs(dot); }
            j += 1;
        }
        c += 1;
    }
    return worst;
}

// The generalized problem with a diagonal B.
fn test_lobpcg_generalized() {
    let n: i64 = 120;
    let a = lt_matrix(n, lobpcg_pins::CHAIN_RI, lobpcg_pins::CHAIN_CI, lobpcg_pins::CHAIN_VA);
    let b = lt_matrix(n, lobpcg_pins::BDIAG_RI, lobpcg_pins::BDIAG_CI, lobpcg_pins::BDIAG_VA);
    let x = common::mat(lobpcg_pins::CHAIN_X, n, 3);
    let w, v, it, lh, rh, f = scientific::lobpcg(a, x, copy(b), scientific::csr_empty(), scientific::empty_arr(), 1.0e-10, 500, false);
    assert(f.kind == scientific::F_NONE);
    lt_values(w, lobpcg_pins::W_GEN, 1.0e-8);
    assert(lt_gram(v, n, b) <= 1.0e-10);
    assert(lt_residual(a, b, w, v, n) <= 1.0e-8);
}

// The preconditioner leaves the values as they are.
fn test_lobpcg_preconditioner() {
    let n: i64 = 120;
    let a = lt_matrix(n, lobpcg_pins::CHAIN_RI, lobpcg_pins::CHAIN_CI, lobpcg_pins::CHAIN_VA);
    let m = lt_matrix(n, lobpcg_pins::M_RI, lobpcg_pins::M_CI, lobpcg_pins::M_VA);
    let none = scientific::csr_empty();
    let x = common::mat(lobpcg_pins::CHAIN_X, n, 3);
    let w, v, it, lh, rh, f = scientific::lobpcg(a, x, scientific::csr_empty(), copy(m), scientific::empty_arr(), 1.0e-10, 500, false);
    assert(f.kind == scientific::F_NONE);
    lt_values(w, lobpcg_pins::W_PRE, 1.0e-8);
    assert(lt_gram(v, n, none) <= 1.0e-10);
}

// The constraints keep the vectors out of the span of Y.
fn test_lobpcg_constraints() {
    let n: i64 = 60;
    let d = lt_matrix(n, lobpcg_pins::DIAG_RI, lobpcg_pins::DIAG_CI, lobpcg_pins::DIAG_VA);
    let x = common::mat(lobpcg_pins::DIAG_X, n, 3);
    let none = scientific::csr_empty();
    let y = scientific::mat2(scientific::copy_slice(lobpcg_pins::Y), n, 2);
    let w, v, it, lh, rh, f = scientific::lobpcg(d, x, scientific::csr_empty(), scientific::csr_empty(), copy(y), 1.0e-10, 500, false);
    assert(f.kind == scientific::F_NONE);
    lt_values(w, lobpcg_pins::W_CON, 1.0e-8);
    assert(lt_ytv(lobpcg_pins::Y, 2, v, n) <= 1.0e-8);
    assert(lt_gram(v, n, none) <= 1.0e-10);
    // The dense path takes no constraints.
    let ds = lt_matrix(8, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let xs = scientific::mat2(scientific::copy_slice([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]), 8, 2);
    let w2, v2, it2, lh2, rh2, f2 = scientific::lobpcg(ds, xs, scientific::csr_empty(), scientific::csr_empty(), scientific::mat2(scientific::copy_slice([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 8, 1), 0.0, 20, false);
    assert(f2.kind == scientific::F_SHAPE);
}

// The histories hold one row of k values per iteration. The last row of the
// value history equals the result.
fn test_lobpcg_histories() {
    let n: i64 = 120;
    let a = lt_matrix(n, lobpcg_pins::CHAIN_RI, lobpcg_pins::CHAIN_CI, lobpcg_pins::CHAIN_VA);
    let x = common::mat(lobpcg_pins::CHAIN_X, n, 3);
    let w, v, it, lh, rh, f = scientific::lobpcg(a, x, scientific::csr_empty(), scientific::csr_empty(), scientific::empty_arr(), 1.0e-8, 500, false, 20, true, true);
    assert(f.kind == scientific::F_NONE);
    assert(lh.ndim == 2 && lh.shape[1] == 3 && rh.ndim == 2 && rh.shape[1] == 3);
    assert(lh.shape[0] == rh.shape[0] && lh.shape[0] >= 2);
    let last = lh.shape[0] - 1;
    var j: i64 = 0;
    while j < 3 {
        assert(common::close(lh.data[last * 3 + j], w.data[j], 1.0e-12));
        assert(rh.data[last * 3 + j] <= 1.0e-6);
        j += 1;
    }
    // Without the flags the histories stay empty.
    let x2 = common::mat(lobpcg_pins::CHAIN_X, n, 3);
    let w2, v2, it2, lh2, rh2, f2 = scientific::lobpcg(a, x2, scientific::csr_empty(), scientific::csr_empty(), scientific::empty_arr(), 1.0e-8, 500, false);
    assert(lh2.data.len == 0 && rh2.data.len == 0);
    // A small restart factor still gives the values.
    let x3 = common::mat(lobpcg_pins::CHAIN_X, n, 3);
    let w3, v3, it3, lh3, rh3, f3 = scientific::lobpcg(a, x3, scientific::csr_empty(), scientific::csr_empty(), scientific::empty_arr(), 1.0e-8, 500, false, 1);
    assert(f3.kind == scientific::F_NONE);
    lt_values(w3, lobpcg_pins::W_EXACT, 1.0e-6);
}

// lt_fault runs lobpcg on the chain matrix of side 120 with one change and
// gives the kind of the fault.
fn lt_fault(rows: i64, cols: i64, maxiter: i64, tol: f64, rc: i64) -> i32 {
    let n: i64 = 120;
    let a = lt_matrix(n, lobpcg_pins::CHAIN_RI, lobpcg_pins::CHAIN_CI, lobpcg_pins::CHAIN_VA);
    let x = scientific::mat2(scientific::copy_slice(lobpcg_pins::CHAIN_X[0..rows * cols]), rows, cols);
    let w, v, it, lh, rh, f = scientific::lobpcg(a, x, scientific::csr_empty(), scientific::csr_empty(), scientific::empty_arr(), tol, maxiter, false, rc);
    return f.kind;
}

// Each hostile input gives its fault and no pair.
fn test_lobpcg_faults() {
    assert(lt_fault(120, 3, 20, 0.0, 20) == scientific::F_NONE);
    assert(lt_fault(100, 3, 20, 0.0, 20) == scientific::F_SHAPE);
    assert(lt_fault(120, 3, 0, 0.0, 20) == scientific::F_RANGE);
    assert(lt_fault(120, 3, 2000000, 0.0, 20) == scientific::F_LIMIT);
    assert(lt_fault(120, 3, 20, 0.0 / 0.0, 20) == scientific::F_VALUE);
    assert(lt_fault(120, 3, 20, 0.0, 0) == scientific::F_RANGE);
    assert(lt_fault(120, 3, 20, 0.0, 61) == scientific::F_RANGE);
    // A start block with a value that is not finite.
    let n: i64 = 120;
    let a = lt_matrix(n, lobpcg_pins::CHAIN_RI, lobpcg_pins::CHAIN_CI, lobpcg_pins::CHAIN_VA);
    var bad: f64[>..] = scientific::copy_slice(lobpcg_pins::CHAIN_X);
    bad[5] = 1.0 / 0.0;
    let w, v, it, lh, rh, f = scientific::lobpcg(a, scientific::mat2(copy(bad), n, 3));
    assert(f.kind == scientific::F_VALUE && w.data.len == 0);
    // A start block of dependent columns.
    var dep: f64[>..] = [];
    var i: i64 = 0;
    while i < n { dep.push(1.0); dep.push(1.0); dep.push(2.0); i += 1; }
    let w2, v2, it2, lh2, rh2, f2 = scientific::lobpcg(a, scientific::mat2(copy(dep), n, 3));
    assert(f2.kind == scientific::F_VALUE);
    // A matrix that is not symmetric.
    let ns = lt_matrix(6, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0]);
    var xs: f64[>..] = [];
    i = 0;
    while i < 6 { xs.push(((i * 7 + 3) % 5) as f64 + 1.0); i += 1; }
    let w3, v3, it3, lh3, rh3, f3 = scientific::lobpcg(ns, scientific::mat2(copy(xs), 6, 1));
    assert(f3.kind == scientific::F_VALUE);
    // A B that is not positive definite.
    let a4 = lt_matrix(60, lobpcg_pins::DIAG_RI, lobpcg_pins::DIAG_CI, lobpcg_pins::DIAG_VA);
    var neg: f64[>..] = [];
    var ri: f64[>..] = [];
    i = 0;
    while i < 60 { ri.push(i as f64); neg.push(1.0); i += 1; }
    neg[7] = -1.0;
    let bneg, fb = scientific::csr_from_triplets(60, 60, ri, ri, neg);
    assert(fb.kind == scientific::F_NONE);
    let w4, v4, it4, lh4, rh4, f4 = scientific::lobpcg(a4, scientific::mat2(scientific::copy_slice(lobpcg_pins::DIAG_X), 60, 3), copy(bneg));
    assert(f4.kind == scientific::F_VALUE);
}

// lt_diag builds the diagonal matrix of side n with the value v on the
// diagonal and, when extra is set, one more entry at (0, n - 1).
fn lt_diag(n: i64, v: f64, extra: f64) -> scientific::Csr {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var va: f64[>..] = [];
    var i: i64 = 0;
    while i < n { ri.push(i as f64); ci.push(i as f64); va.push(v); i += 1; }
    if extra != 0.0 { ri.push(0.0); ci.push((n - 1) as f64); va.push(extra); }
    let s, f = scientific::csr_from_triplets(n, n, ri, ci, va);
    assert(f.kind == scientific::F_NONE);
    return s;
}

// lt_ones gives a block of r rows and c columns that holds ones and one
// larger value on the diagonal, so the columns are independent.
fn lt_ones(r: i64, c: i64) -> scientific::Arr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < r {
        var j: i64 = 0;
        while j < c { d.push(if i == j { 2.0 } else { 0.1 * ((i + j) % 3) as f64 }); j += 1; }
        i += 1;
    }
    return scientific::mat2(copy(d), r, c);
}

// The shape rows of the fault table: k above n, ny plus k above n, and a b,
// m or y of the wrong size.
fn test_lobpcg_fault_shapes() {
    let a6 = lt_diag(6, 1.0, 0.0);
    let w, v, it, lh, rh, f = scientific::lobpcg(a6, lt_ones(6, 7));
    assert(f.kind == scientific::F_SHAPE);
    let a6b = lt_diag(6, 1.0, 0.0);
    let w2, v2, it2, lh2, rh2, f2 = scientific::lobpcg(a6b, lt_ones(6, 2), scientific::csr_empty(), scientific::csr_empty(), lt_ones(6, 5));
    assert(f2.kind == scientific::F_SHAPE);
    let a20 = lt_diag(20, 1.0, 0.0);
    let w3, v3, it3, lh3, rh3, f3 = scientific::lobpcg(a20, lt_ones(20, 2), lt_diag(5, 1.0, 0.0));
    assert(f3.kind == scientific::F_SHAPE);
    let a20b = lt_diag(20, 1.0, 0.0);
    let w4, v4, it4, lh4, rh4, f4 = scientific::lobpcg(a20b, lt_ones(20, 2), scientific::csr_empty(), lt_diag(5, 1.0, 0.0));
    assert(f4.kind == scientific::F_SHAPE);
    let a20c = lt_diag(20, 1.0, 0.0);
    let w5, v5, it5, lh5, rh5, f5 = scientific::lobpcg(a20c, lt_ones(20, 2), scientific::csr_empty(), scientific::csr_empty(), lt_ones(5, 1));
    assert(f5.kind == scientific::F_SHAPE);
}

// A value that is not finite in any input faults, and so does an asymmetric
// b. The dense path above its side limit faults with the limit kind.
fn test_lobpcg_fault_values() {
    let nan = 0.0 / 0.0;
    let wa, va, ita, lha, rha, fa = scientific::lobpcg(lt_diag(20, nan, 0.0), lt_ones(20, 2));
    assert(fa.kind == scientific::F_VALUE);
    let wb, vb, itb, lhb, rhb, fb = scientific::lobpcg(lt_diag(20, 1.0, 0.0), lt_ones(20, 2), lt_diag(20, nan, 0.0));
    assert(fb.kind == scientific::F_VALUE);
    let wm, vm, itm, lhm, rhm, fm = scientific::lobpcg(lt_diag(20, 1.0, 0.0), lt_ones(20, 2), scientific::csr_empty(), lt_diag(20, nan, 0.0));
    assert(fm.kind == scientific::F_VALUE);
    var yb: f64[>..] = [];
    var i: i64 = 0;
    while i < 20 { yb.push(if i == 3 { nan } else { 1.0 }); i += 1; }
    let wy, vy, ity, lhy, rhy, fy = scientific::lobpcg(lt_diag(20, 1.0, 0.0), lt_ones(20, 2), scientific::csr_empty(), scientific::csr_empty(), scientific::mat2(copy(yb), 20, 1));
    assert(fy.kind == scientific::F_VALUE);
    // The matrix b holds an entry without its mirror.
    let wn, vn, itn, lhn, rhn, fn_ = scientific::lobpcg(lt_diag(20, 1.0, 0.0), lt_ones(20, 2), lt_diag(20, 1.0, 0.5));
    assert(fn_.kind == scientific::F_VALUE);
    // The dense path above the side limit.
    let big: i64 = scientific::BD_MAX_SIDE + 1;
    let k: i64 = big / 5 + 1;
    let wl, vl, itl, lhl, rhl, fl = scientific::lobpcg(lt_diag(big, 1.0, 0.0), lt_ones(big, k));
    assert(fl.kind == scientific::F_LIMIT);
}

// A finite matrix whose arithmetic overflows gives a fault, not NaN.
fn test_lobpcg_overflow() {
    let a, f = scientific::csr_from_triplets(2, 2, [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0e308, 1.0e308, 1.0e308, -1.0e308]);
    assert(f.kind == scientific::F_NONE);
    let w, v, it, lh, rh, g = scientific::lobpcg(a, common::mat([1.0, 0.5], 2, 1));
    assert(g.kind == scientific::F_VALUE && w.data.len == 0);
}

// A chain of side 2000 with three pairs finishes within the time limit.
fn test_lobpcg_large() {
    let n: i64 = 2000;
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var va: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        if i > 0 { ri.push(i as f64); ci.push((i - 1) as f64); va.push(-1.0); }
        ri.push(i as f64); ci.push(i as f64); va.push(2.0);
        if i < n - 1 { ri.push(i as f64); ci.push((i + 1) as f64); va.push(-1.0); }
        i += 1;
    }
    let a = lt_matrix(n, ri, ci, va);
    var x: f64[>..] = [];
    var st: i64 = 48271;
    i = 0;
    while i < n * 3 {
        st = (st * 48271) % 2147483647;
        x.push(((st % 1000) as f64) / 500.0 - 1.0);
        i += 1;
    }
    let w, v, it, lh, rh, f = scientific::lobpcg(a, scientific::mat2(copy(x), n, 3), scientific::csr_empty(), scientific::csr_empty(), scientific::empty_arr(), 1.0e-3, 200, false);
    assert(f.kind == scientific::F_NONE);
    assert(w.data.len == 3 && w.data[0] > 0.0 && w.data[0] < w.data[1]);
}
