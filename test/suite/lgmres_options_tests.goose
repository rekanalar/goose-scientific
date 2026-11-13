// The tests of the group lgmres_options. Run them with SCIENTIFIC_SUITES=lgmres_options.
//
// The callback, the preconditioner and the kept pairs of lgmres and gcrotmk.
// The pins come from the generated file of the group.
namespace lgmres_options_tests;

import scientific from "../..";
import common;
import lgmres_options_pins;

fn lo_dense(d: f64[:], n: i64) -> scientific::Csr {
    let c, f = scientific::csr_from_dense(common::mat(scientific::copy_slice(d), n, n));
    assert(f.kind == scientific::F_NONE);
    return c;
}

// lo_diag gives the diagonal matrix of the values d.
fn lo_diag(d: f64[:]) -> scientific::Csr {
    let n = d.len;
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var i: i64 = 0;
    while i < n { ri.push(i as f64); ci.push(i as f64); i += 1; }
    let c, f = scientific::csr_from_triplets(n, n, ri, ci, d);
    assert(f.kind == scientific::F_NONE);
    return c;
}

fn lo_kept(p: f64[:], count: i64, n: i64) -> scientific::Kept {
    return scientific::Kept { count: count, n: n, buf: scientific::copy_slice(p) };
}

// lo_near tells whether n values of a and b agree to tol.
fn lo_near(a: f64[:], b: f64[:], tol: f64) -> bool {
    if a.len != b.len { return false; }
    var i: i64 = 0;
    while i < a.len {
        if !(common::mathshim_abs(a[i] - b[i]) <= tol) { return false; }
        i += 1;
    }
    return true;
}

// lo_lg runs lgmres with a block that keeps the vectors it receives.
fn lo_lg(a: scientific::Csr&, b: f64[:], inner_m: i64, store: bool, prepend: bool, mp: scientific::Csr&,
         cb: f64[>..]&, count: i64&) -> scientific::Arr, scientific::Fault {
    let x, it, f = scientific::lgmres(a, b, [], 1.0e-10, 0.0, 1000, inner_m, 3, store, prepend, copy(mp)) { v =>
        count += 1;
        var i: i64 = 0;
        while i < v.len { cb.push(v[i]); i += 1; }
    };
    return x, f;
}

fn lo_gk(a: scientific::Csr&, b: f64[:], mp: scientific::Csr&, cb: f64[>..]&, count: i64&) -> scientific::Arr, scientific::Fault {
    let x, it, f = scientific::gcrotmk(a, b, [], 1.0e-10, 0.0, 1000, 20, 0, false, true, copy(mp)) { v =>
        count += 1;
        var i: i64 = 0;
        while i < v.len { cb.push(v[i]); i += 1; }
    };
    return x, f;
}

// The block gets the solution at the start of each outer iteration.
fn test_lgmres_options_callback() {
    let a1 = lo_dense(lgmres_options_pins::A1, 6);
    let a2 = lo_dense(lgmres_options_pins::A2, 8);
    let none = scientific::csr_empty();
    var cb: f64[>..] = [];
    var count: i64 = 0;
    let x, f = lo_lg(a1, lgmres_options_pins::B1, 30, true, false, none, cb, count);
    assert(f.kind == scientific::F_NONE && count == lgmres_options_pins::LG1_NCB[0] as i64);
    assert(lo_near(cb, lgmres_options_pins::LG1_CB, 1.0e-6) && lo_near(x.data, lgmres_options_pins::LG1_X, 1.0e-8));
    // The first vector is the start vector.
    assert(cb[0] == 0.0 && cb[5] == 0.0);
    var cb2: f64[>..] = [];
    var count2: i64 = 0;
    let y, f2 = lo_gk(a1, lgmres_options_pins::B1, none, cb2, count2);
    assert(f2.kind == scientific::F_NONE && count2 == lgmres_options_pins::GK1_NCB[0] as i64);
    assert(lo_near(cb2, lgmres_options_pins::GK1_CB, 1.0e-6) && lo_near(y.data, lgmres_options_pins::GK1_X, 1.0e-8));
    var cb3: f64[>..] = [];
    var count3: i64 = 0;
    let z, f3 = lo_lg(a2, lgmres_options_pins::B2, 30, true, false, none, cb3, count3);
    assert(f3.kind == scientific::F_NONE && count3 == lgmres_options_pins::LG2_NCB[0] as i64);
    var cb4: f64[>..] = [];
    var count4: i64 = 0;
    let w, f4 = lo_gk(a2, lgmres_options_pins::B2, none, cb4, count4);
    assert(f4.kind == scientific::F_NONE && count4 == lgmres_options_pins::GK2_NCB[0] as i64);
    // A call with no block gives the same solution.
    let p, pc, pf = scientific::lgmres(a1, lgmres_options_pins::B1, [], 1.0e-10);
    assert(pf.kind == scientific::F_NONE && lo_near(p.data, x.data, 0.0));
    let q, qc, qf = scientific::gcrotmk(a1, lgmres_options_pins::B1, [], 1.0e-10);
    assert(qf.kind == scientific::F_NONE && lo_near(q.data, y.data, 0.0));
}

// A start vector that solves the system still calls the block once.
fn test_lgmres_options_callback_once() {
    let a1 = lo_dense(lgmres_options_pins::A1, 6);
    var count: i64 = 0;
    let x, it, f = scientific::lgmres(a1, lgmres_options_pins::B1, lgmres_options_pins::LG1_X, 1.0e-6) { v => count += 1; };
    assert(f.kind == scientific::F_NONE && it == 0 && count == 1);
    var count2: i64 = 0;
    let y, it2, f2 = scientific::gcrotmk(a1, lgmres_options_pins::B1, lgmres_options_pins::GK1_X, 1.0e-6) { v => count2 += 1; };
    assert(f2.kind == scientific::F_NONE && it2 == 0 && count2 == 1);
}

// A diagonal preconditioner gives the solution of SciPy.
fn test_lgmres_options_precond() {
    let a2 = lo_dense(lgmres_options_pins::A2, 8);
    let mp = lo_diag(lgmres_options_pins::MD2);
    var cb: f64[>..] = [];
    var count: i64 = 0;
    let x, f = lo_lg(a2, lgmres_options_pins::B2, 30, true, false, mp, cb, count);
    assert(f.kind == scientific::F_NONE && lo_near(x.data, lgmres_options_pins::LG2_M_X, 1.0e-8));
    assert(count == lgmres_options_pins::LG2_M_NCB[0] as i64 && count <= lgmres_options_pins::LG2_NCB[0] as i64);
    var cb2: f64[>..] = [];
    var count2: i64 = 0;
    let y, f2 = lo_gk(a2, lgmres_options_pins::B2, mp, cb2, count2);
    assert(f2.kind == scientific::F_NONE && lo_near(y.data, lgmres_options_pins::GK2_M_X, 1.0e-8));
    assert(count2 == lgmres_options_pins::GK2_M_NCB[0] as i64 && count2 <= lgmres_options_pins::GK2_NCB[0] as i64);
    // The identity gives the result of the call with no preconditioner.
    let id = lo_diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let u, uc, uf = scientific::lgmres(a2, lgmres_options_pins::B2, [], 1.0e-10, 0.0, 1000, 30, 3, true, false, copy(id));
    let u0, uc0, uf0 = scientific::lgmres(a2, lgmres_options_pins::B2, [], 1.0e-10);
    assert(uf.kind == scientific::F_NONE && uf0.kind == scientific::F_NONE && lo_near(u.data, u0.data, 1.0e-12));
    let g, gc, gf = scientific::gcrotmk(a2, lgmres_options_pins::B2, [], 1.0e-10, 0.0, 1000, 20, 0, false, true, copy(id));
    let g0, gc0, gf0 = scientific::gcrotmk(a2, lgmres_options_pins::B2, [], 1.0e-10);
    assert(gf.kind == scientific::F_NONE && gf0.kind == scientific::F_NONE && lo_near(g.data, g0.data, 1.0e-12));
}

// A preconditioner on a third system gives the solution of SciPy.
fn test_lgmres_options_precond_third() {
    let a3 = lo_dense(lgmres_options_pins::A3, 12);
    let mp = lo_diag(lgmres_options_pins::MD3);
    var cb: f64[>..] = [];
    var count: i64 = 0;
    let x, f = lo_lg(a3, lgmres_options_pins::B3, 30, true, false, mp, cb, count);
    assert(f.kind == scientific::F_NONE && lo_near(x.data, lgmres_options_pins::LG3_M_X, 1.0e-8));
    assert(count == lgmres_options_pins::LG3_M_NCB[0] as i64);
    var cb2: f64[>..] = [];
    var count2: i64 = 0;
    let y, f2 = lo_gk(a3, lgmres_options_pins::B3, mp, cb2, count2);
    assert(f2.kind == scientific::F_NONE && lo_near(y.data, lgmres_options_pins::GK3_M_X, 1.0e-8));
    assert(count2 == lgmres_options_pins::GK3_M_NCB[0] as i64);
    var cb3: f64[>..] = [];
    var count3: i64 = 0;
    let none = scientific::csr_empty();
    let z, f3 = lo_lg(a3, lgmres_options_pins::B3, 30, true, false, none, cb3, count3);
    assert(f3.kind == scientific::F_NONE && lo_near(z.data, lgmres_options_pins::LG3_X, 1.0e-8));
    var cb4: f64[>..] = [];
    var count4: i64 = 0;
    let w, f4 = lo_gk(a3, lgmres_options_pins::B3, none, cb4, count4);
    assert(f4.kind == scientific::F_NONE && lo_near(w.data, lgmres_options_pins::GK3_X, 1.0e-8));
}

// A preconditioner of the wrong side, shape or value gives a fault.
fn test_lgmres_options_precond_faults() {
    let a1 = lo_dense(lgmres_options_pins::A1, 6);
    let side = lo_diag([1.0, 1.0, 1.0]);
    let x, it, f = scientific::lgmres(a1, lgmres_options_pins::B1, [], 1.0e-10, 0.0, 1000, 30, 3, true, false, copy(side));
    assert(f.kind == scientific::F_SHAPE);
    let y, it2, f2 = scientific::gcrotmk(a1, lgmres_options_pins::B1, [], 1.0e-10, 0.0, 1000, 20, 0, false, true, copy(side));
    assert(f2.kind == scientific::F_SHAPE);
    let wide, fw = scientific::csr_from_triplets(6, 7, [0.0], [6.0], [1.0]);
    assert(fw.kind == scientific::F_NONE);
    let x3, it3, f3 = scientific::lgmres(a1, lgmres_options_pins::B1, [], 1.0e-10, 0.0, 1000, 30, 3, true, false, copy(wide));
    assert(f3.kind == scientific::F_SHAPE);
    var bad = lo_diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    bad.buf[bad.rows + 1 + bad.nnz] = 1.0 / 0.0;
    let x4, it4, f4 = scientific::lgmres(a1, lgmres_options_pins::B1, [], 1.0e-10, 0.0, 1000, 30, 3, true, false, copy(bad));
    assert(f4.kind == scientific::F_VALUE);
    let y4, it5, f5 = scientific::gcrotmk(a1, lgmres_options_pins::B1, [], 1.0e-10, 0.0, 1000, 20, 0, false, true, copy(bad));
    assert(f5.kind == scientific::F_VALUE);
    // A preconditioner that maps the residual to zero gives a fault.
    let zero = lo_diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let x6, it6, f6 = scientific::lgmres(a1, lgmres_options_pins::B1, [], 1.0e-10, 0.0, 1000, 30, 3, true, false, copy(zero));
    assert(f6.kind == scientific::F_VALUE);
}

// lo_pairs_lg runs lgmres_pairs and gives the pairs.
fn lo_pairs_lg(b: f64[:], inner_m: i64, store: bool, prepend: bool, ov: scientific::Kept&) -> scientific::Arr, scientific::Kept {
    let a1 = lo_dense(lgmres_options_pins::A1, 6);
    let x, it, kp, f = scientific::lgmres_pairs(a1, b, [], 1.0e-10, 0.0, 1000, inner_m, 3, store, prepend,
                                                scientific::csr_empty(), copy(ov));
    assert(f.kind == scientific::F_NONE);
    return x, kp;
}

fn lo_pairs_gk(b: f64[:], m: i64, k: i64, discard: bool, oldest: bool, cu: scientific::Kept&) -> scientific::Arr, scientific::Kept {
    let a1 = lo_dense(lgmres_options_pins::A1, 6);
    let x, it, kp, f = scientific::gcrotmk_pairs(a1, b, [], 1.0e-10, 0.0, 1000, m, k, discard, oldest,
                                                 scientific::csr_empty(), copy(cu));
    assert(f.kind == scientific::F_NONE);
    return x, kp;
}

// The pairs of lgmres match SciPy, with and without the products.
fn test_lgmres_options_pairs_lgmres() {
    let none = scientific::kept_empty();
    let x, kp = lo_pairs_lg(lgmres_options_pins::B1, 30, true, false, none);
    assert(kp.count == lgmres_options_pins::LG1_NP[0] as i64 && kp.n == 6 && lo_near(kp.buf, lgmres_options_pins::LG1_P, 1.0e-8));
    let x2, kp2 = lo_pairs_lg(lgmres_options_pins::B1, 30, false, false, none);
    assert(kp2.count == lgmres_options_pins::LG1_NOAV_NP[0] as i64 && lo_near(kp2.buf, lgmres_options_pins::LG1_NOAV_P, 1.0e-8));
    let x3, kp3 = lo_pairs_lg(lgmres_options_pins::B1, 3, true, true, none);
    assert(kp3.count == lgmres_options_pins::LG1_PRE_NP[0] as i64 && lo_near(x3.data, lgmres_options_pins::LG1_PRE_X, 1.0e-8));
    // A second solve that takes the pairs matches SciPy and needs no more
    // outer turns than the first solve.
    let x4, kp4 = lo_pairs_lg(lgmres_options_pins::B1B, 30, true, false, kp);
    assert(lo_near(x4.data, lgmres_options_pins::LG1_2_X, 1.0e-8));
    assert(lgmres_options_pins::LG1_2_IT[0] <= lgmres_options_pins::LG1_IT[0]);
}

// The pairs of gcrotmk match SciPy: the last pair holds the solution.
fn test_lgmres_options_pairs_gcrotmk() {
    let none = scientific::kept_empty();
    let x, kp = lo_pairs_gk(lgmres_options_pins::B1, 20, 0, false, true, none);
    assert(kp.count == lgmres_options_pins::GK1_NP[0] as i64 && lo_near(kp.buf, lgmres_options_pins::GK1_P, 1.0e-8));
    let x2, kp2 = lo_pairs_gk(lgmres_options_pins::B1, 20, 0, true, true, none);
    assert(kp2.count == lgmres_options_pins::GK1_DISC_NP[0] as i64 && lo_near(kp2.buf, lgmres_options_pins::GK1_DISC_P, 1.0e-8));
    let x3, kp3 = lo_pairs_gk(lgmres_options_pins::B1, 3, 2, false, true, none);
    assert(kp3.count == lgmres_options_pins::GK1_M3_NP[0] as i64 && lo_near(kp3.buf, lgmres_options_pins::GK1_M3_P, 1.0e-8));
    let x4, kp4 = lo_pairs_gk(lgmres_options_pins::B1, 3, 2, false, false, none);
    assert(kp4.count == lgmres_options_pins::GK1_SMALL_NP[0] as i64 && lo_near(x4.data, lgmres_options_pins::GK1_SMALL_X, 1.0e-8));
    let x5, kp5 = lo_pairs_gk(lgmres_options_pins::B1B, 20, 0, false, true, kp);
    assert(lo_near(x5.data, lgmres_options_pins::GK1_2_X, 1.0e-8));
    assert(lgmres_options_pins::GK1_2_IT[0] <= lgmres_options_pins::GK1_IT[0]);
}

// lo_many_pairs gives five pairs that do not match any matrix.
fn lo_many_pairs() -> scientific::Kept {
    var buf: f64[>..] = [];
    var p: i64 = 0;
    while p < 5 {
        buf.push(1.0);
        var i: i64 = 0;
        while i < 6 { buf.push(0.1); i += 1; }
        var j: i64 = 0;
        while j < 6 { buf.push(0.2); j += 1; }
        p += 1;
    }
    return scientific::Kept { count: 5, n: 6, buf: copy(buf) };
}

// More pairs than the call keeps give a result and no fault.
fn test_lgmres_options_pairs_many() {
    let a1 = lo_dense(lgmres_options_pins::A1, 6);
    let kp = lo_many_pairs();
    // The call keeps outer_k pairs, so the older ones go.
    let x, it, out, f = scientific::lgmres_pairs(a1, lgmres_options_pins::B1, [], 1.0e-10, 0.0, 1000, 30, 3,
                                                 true, false, scientific::csr_empty(), copy(kp));
    assert(f.kind == scientific::F_NONE && out.count == 3 && out.n == 6);
    assert(out.buf.len == 3 * 13);
}

// A pair of the wrong size, a value that is not finite and too many pairs give faults.
fn test_lgmres_options_pairs_faults() {
    let a1 = lo_dense(lgmres_options_pins::A1, 6);
    let none = scientific::csr_empty();
    let short = scientific::Kept { count: 1, n: 5, buf: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] };
    let x, it, kp, f = scientific::lgmres_pairs(a1, lgmres_options_pins::B1, [], 1.0e-10, 0.0, 1000, 30, 3, true, false, copy(none), copy(short));
    assert(f.kind == scientific::F_SHAPE);
    let y, it2, kp2, f2 = scientific::gcrotmk_pairs(a1, lgmres_options_pins::B1, [], 1.0e-10, 0.0, 1000, 20, 0, false, true, copy(none), copy(short));
    assert(f2.kind == scientific::F_SHAPE);
    var bad = lo_kept(lgmres_options_pins::GK1_P, lgmres_options_pins::GK1_NP[0] as i64, 6);
    bad.buf[3] = 1.0 / 0.0;
    let x3, it3, kp3, f3 = scientific::gcrotmk_pairs(a1, lgmres_options_pins::B1, [], 1.0e-10, 0.0, 1000, 20, 0, false, true, copy(none), copy(bad));
    assert(f3.kind == scientific::F_VALUE);
    var flag = lo_kept(lgmres_options_pins::GK1_P, lgmres_options_pins::GK1_NP[0] as i64, 6);
    flag.buf[0] = 2.0;
    let x4, it4, kp4, f4 = scientific::gcrotmk_pairs(a1, lgmres_options_pins::B1, [], 1.0e-10, 0.0, 1000, 20, 0, false, true, copy(none), copy(flag));
    assert(f4.kind == scientific::F_VALUE);
    var many: f64[>..] = [];
    var i: i64 = 0;
    while i < 65 * 13 { many.push(0.0); i += 1; }
    let big = scientific::Kept { count: 65, n: 6, buf: copy(many) };
    let x5, it5, kp5, f5 = scientific::lgmres_pairs(a1, lgmres_options_pins::B1, [], 1.0e-10, 0.0, 1000, 30, 3, true, false, copy(none), copy(big));
    assert(f5.kind == scientific::F_LIMIT);
    // Pairs that do not come from a solve of this matrix give a fault.
    let odd = lo_many_pairs();
    let x6, it6, kp6, f6 = scientific::gcrotmk_pairs(a1, lgmres_options_pins::B1, [], 1.0e-10, 0.0, 1000, 20, 0, false, true, copy(none), copy(odd));
    assert(f6.kind == scientific::F_VALUE);
}
