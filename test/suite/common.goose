// The helpers that the tests of more than one group share.
namespace common;

import scientific from "../..";
import os;

fn empty_row() -> f64[>..] { var r: f64[>..] = []; return r; }

// all_close tells whether every tap of a is within tol of v.
fn all_close(a: scientific::Arr&, v: f64[:], tol: f64) -> bool {
    if a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        var d = a.data[i] - v[i];
        if d < 0.0 { d = -d; }
        if d > tol { return false; }
        i += 1;
    }
    return true;
}

// close tells whether two numbers agree to a small tolerance.
fn mathshim_abs(x: f64) -> f64 {
    if x < 0.0 { return -x; }
    return x;
}

// pin_splrep checks one smoothing spline against the knots, the
// coefficients and the residual that SciPy gives.
fn pin_splrep(x: f64[:], y: f64[:], k: i64, s: f64, kn: f64[:], co: f64[:], fp: f64) {
    let b, f = scientific::splrep(x, y, k, s);
    assert(f.kind == scientific::F_NONE);
    let bk = scientific::bs_knots_of(b);
    let bc = scientific::bs_coefs_of(b);
    assert(bk.len == kn.len && bc.len == co.len);
    var q: i64 = 0;
    while q < kn.len { assert(common::mathshim_abs(bk[q] - kn[q]) < 1.0e-9); q += 1; }
    q = 0;
    while q < co.len { assert(common::mathshim_abs(bc[q] - co[q]) < 1.0e-8 * (1.0 + common::mathshim_abs(co[q]))); q += 1; }
    var w: f64[>..] = [];
    q = 0;
    while q < x.len { w.push(1.0); q += 1; }
    let r, fr = scientific::splrep_residual(x, y, w, b);
    assert(fr.kind == scientific::F_NONE);
    assert(common::mathshim_abs(r - fp) < 1.0e-8);
}

fn gmat(r: i64, c: i64, v: f64[:]) -> scientific::Arr {
    var sh: i64[..8] = [];
    sh.push(r);
    sh.push(c);
    return scientific::Arr { ndim: 2, shape: sh, data: scientific::copy_slice(v) };
}

// sm_check compares a special matrix with the values of SciPy.
fn sm_check(a, f, r: i64, c: i64, v: f64[:]) {
    assert(f.kind == scientific::F_NONE);
    assert(a.ndim == 2 && a.shape[0] == r && a.shape[1] == c);
    var i: i64 = 0;
    while i < v.len {
        assert(common::mathshim_abs(a.data[i] - v[i]) <= 1.0e-12 * (1.0 + common::mathshim_abs(v[i])));
        i += 1;
    }
}

// mchk and vchk compare a result with the values of SciPy.
fn mchk(a, f, r: i64, c: i64, v: f64[:]) {
    assert(f.kind == scientific::F_NONE);
    if c == 1 && a.ndim == 1 {
        assert(a.shape[0] == r);
    } else {
        assert(a.ndim == 2 && a.shape[0] == r && a.shape[1] == c);
    }
    var i: i64 = 0;
    while i < v.len {
        assert(common::mathshim_abs(a.data[i] - v[i]) <= 1.0e-9 * (1.0 + common::mathshim_abs(v[i])));
        i += 1;
    }
}

fn vchk(a, f, v: f64[:]) {
    assert(f.kind == scientific::F_NONE);
    assert(a.data.len == v.len);
    var i: i64 = 0;
    while i < v.len {
        assert(common::mathshim_abs(a.data[i] - v[i]) <= 1.0e-8 * (1.0 + common::mathshim_abs(v[i])));
        i += 1;
    }
}

fn grid_close(a, v: f64[:], tol: f64) -> bool {
    if a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        let x = a.data[i];
        if v[i] != v[i] {
            if x == x { return false; }
        } else if common::mathshim_abs(x - v[i]) > tol { return false; }
        i += 1;
    }
    return true;
}

fn rob_jac(y: f64[:], mode: i64, bad: i64) -> f64[>..] {
    var o: f64[>..] = [];
    if mode == 0 {
        o.push(-0.04 * y[0] + 1.0e4 * y[1] * y[2]);
        o.push(0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1]);
        o.push(3.0e7 * y[1] * y[1]);
    } else if bad == 1 {
        o = [1.0, 2.0];
    } else if bad == 2 {
        o = [0.0 / 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    } else {
        o = [-0.04, 1.0e4 * y[2], 1.0e4 * y[1], 0.04, -1.0e4 * y[2] - 6.0e7 * y[1], -1.0e4 * y[1], 0.0, 6.0e7 * y[1], 0.0];
    }
    return o;
}

fn close(a: f64, b: f64) -> bool {
    let d = a - b;
    if d < 0.0 { return -d < 0.0000001; }
    return d < 0.0000001;
}

// close tells whether a is within tol of b, relative to one plus b.
fn close(a: f64, b: f64, tol: f64) -> bool {
    var d = a - b;
    if d < 0.0 { d = -d; }
    var m = b;
    if m < 0.0 { m = -m; }
    return d <= tol * (1.0 + m);
}

fn make23() -> scientific::Arr {
    let a, f = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert(f.kind == scientific::F_NONE);
    return a;
}

// dquad is the derivative of the square. The solvers that need a
// derivative take it as a named function.
fn dquad(x: f64) -> f64 { return 2.0 * x; }

// dflat is the derivative of a function that never changes.
fn dflat(x: f64) -> f64 { return 0.0; }

// dsecond is the second derivative of the square of x and two.
fn dsecond(x: f64) -> f64 { return 2.0; }

// The stacks hold the values of NumPy.
fn line_sum(line: scientific::Arr&) -> f64 {
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < line.data.len { t += line.data[i]; i += 1; }
    return t;
}

fn line_reverse(line: scientific::Arr&) -> scientific::Arr {
    var d: f64[>..] = [];
    var i: i64 = line.data.len - 1;
    while i >= 0 { d.push(line.data[i]); i -= 1; }
    var s: i64[..8] = [];
    s.push(d.len);
    return scientific::Arr { ndim: 1, shape: s, data: d };
}

fn line_bounds(line: scientific::Arr&) -> scientific::Arr {
    var lo = line.data[0];
    var hi = line.data[0];
    var i: i64 = 1;
    while i < line.data.len {
        if line.data[i] < lo { lo = line.data[i]; }
        if line.data[i] > hi { hi = line.data[i]; }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(2);
    var d: f64[>..] = [];
    d.push(lo);
    d.push(hi);
    return scientific::Arr { ndim: 1, shape: s, data: d };
}

fn close_row(a, at: i64, v: f64[:]) -> bool {
    var i: i64 = 0;
    while i < v.len {
        let d = a.data[at + i] - v[i];
        if d > 0.000000001 || d < -0.000000001 { return false; }
        i += 1;
    }
    return true;
}

fn close_all(a, v: f64[:]) -> bool {
    if a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        let d = a.data[i] - v[i];
        if d > 0.000000001 || d < -0.000000001 { return false; }
        i += 1;
    }
    return true;
}

fn mat(d: f64[:], rows: i64, cols: i64) -> scientific::Arr {
    var c: f64[>..] = [];
    var i: i64 = 0;
    while i < d.len { c.push(d[i]); i += 1; }
    var s: i64[..8] = [];
    s.push(rows);
    s.push(cols);
    return scientific::Arr { ndim: 2, shape: s, data: c };
}

fn av(x: f64[:]) -> scientific::Arr {
    let a, f = scientific::from_values(x);
    assert(f.kind == scientific::F_NONE);
    return a;
}

// near tells whether two numbers agree to a relative tolerance.
fn near(a: f64, b: f64) -> bool {
    var d = a - b;
    if d < 0.0 { d = -d; }
    var s = b;
    if s < 0.0 { s = -s; }
    return d <= 0.000000001 * (1.0 + s);
}

// same tells whether an array holds the shape and the values.
fn same_vals(a: scientific::Arr&, shape: i64[:], v: f64[:]) -> bool {
    if a.ndim != shape.len || a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < shape.len { if a.shape[i] != shape[i] { return false; } i += 1; }
    i = 0;
    while i < v.len { if a.data[i] != v[i] { return false; } i += 1; }
    return true;
}

fn seq(n: i64) -> scientific::Arr {
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < n { v.push(i); i += 1; }
    let a, f = scientific::from_values(v);
    assert(f.kind == scientific::F_NONE);
    return a;
}

fn all_near(a: scientific::Arr&, v: f64[:]) -> bool {
    if a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len { if !common::near(a.data[i], v[i]) { return false; } i += 1; }
    return true;
}

fn win_is(name: const u8[:], n: i64, fftbins: bool, p: f64, v: f64[:]) -> bool {
    let w, f = scientific::get_window(name, n, fftbins, p);
    assert(f.kind == scientific::F_NONE);
    return common::all_near(w, v);
}

fn mx(r: i64, c: i64, v: f64[:]) -> scientific::Arr {
    let a, f = scientific::from_rows(r, c, v);
    assert(f.kind == scientific::F_NONE);
    return a;
}

fn mathshim_sin(x: f64) -> f64 { return mathshim::sine(x); }

fn mathshim_exp(x: f64) -> f64 { return mathshim::power_e(x); }

fn mathshim_sqrt(x: f64) -> f64 { return mathshim::root(x); }

fn mathshim_cos(x: f64) -> f64 { return mathshim::cosine(x); }

fn scientific_abs(x: f64) -> f64 { return common::mathshim_abs(x); }

// The complex array type and its operations, pinned against NumPy.
// all_near_c tells whether the parts of a complex array agree to two lists.
fn all_near_c(a: scientific::CArr&, vr: f64[:], vi: f64[:]) -> bool {
    var i: i64 = 0;
    while i < vr.len {
        if !common::near(a.data[2 * i], vr[i]) { return false; }
        if !common::near(a.data[2 * i + 1], vi[i]) { return false; }
        i += 1;
    }
    return true;
}

fn dbl_lo1(x: f64) -> f64 { return 0.5 * x; }

fn dbl_hi1(x: f64) -> f64 { return 1.0 + x; }

fn dbl_zero(x: f64) -> f64 { return 0.0; }

fn dbl_sq(x: f64) -> f64 { return x * x; }

fn dbl_m1(x: f64) -> f64 { return -1.0; }

fn dbl_p1(x: f64) -> f64 { return 1.0; }

fn dbl_inf(x: f64) -> f64 { return 1.0 / 0.0; }

// dual_annealing and basinhopping find the global minimum of a
// function with many local minima. The test uses the function of
// Rastrigin, whose global minimum is zero at the origin.
fn rastrigin(v: f64[:]) -> f64 {
    return 20.0 + v[0] * v[0] - 10.0 * common::mathshim_cos(6.283185307179586 * v[0]) + v[1] * v[1] - 10.0 * common::mathshim_cos(6.283185307179586 * v[1]);
}

// welch_ex is pinned against SciPy welch.
fn welch_probe(p: scientific::Arr&, n: i64, vals: f64[:]) -> bool {
    if p.data.len != n { return false; }
    let idx = [0, 1, 2, 3, n / 2, n - 1];
    var i: i64 = 0;
    while i < 6 { if !common::near(p.data[idx[i]], vals[i]) { return false; } i += 1; }
    return true;
}

fn welch_probe_f(p: scientific::Arr&, n: i64, vals: f64[:]) -> bool {
    if p.data.len != n { return false; }
    let idx = [0, 1, 2, 3, n / 2, n - 1];
    var i: i64 = 0;
    while i < 6 { if !common::near(p.data[idx[i]] + 1.0, vals[i] + 1.0) { return false; } i += 1; }
    return true;
}

// lfilter on arrays of N dimensions, pinned against SciPy.
fn lf_close(a: scientific::Arr&, stride: i64, vals: f64[:]) -> bool {
    var i: i64 = 0;
    var j: i64 = 0;
    while i < a.data.len {
        if j >= vals.len || !common::near(a.data[i] + 1.0, vals[j] + 1.0) { return false; }
        i += stride;
        j += 1;
    }
    return j == vals.len;
}

// The windows of scipy.signal.windows are pinned against SciPy.
fn win_close(a: scientific::Arr&, e: f64[:]) -> bool {
    if a.data.len != e.len { return false; }
    var i: i64 = 0;
    while i < e.len {
        if common::mathshim_abs(a.data[i] - e[i]) > 1.0e-10 { return false; }
        i += 1;
    }
    return true;
}

// The method Radau, pinned against SciPy 1.18.1. The count of steps
// equals the count of SciPy.
// head_close compares the first values of an array with a list.
fn head_close(a: scientific::Arr&, v: f64[:], tol: f64) -> bool {
    if a.data.len < v.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        if common::mathshim_abs(a.data[i] - v[i]) > tol * (1.0 + common::mathshim_abs(v[i])) { return false; }
        i += 1;
    }
    return true;
}

// cov_close tells whether two values of a covariance matrix agree. The
// comparison uses a scale of 1e-3, because the small values of the
// matrix lose their digits in the reference library.
fn cov_close(a: f64, b: f64) -> bool {
    var d = a - b;
    if d < 0.0 { d = -d; }
    return d <= 0.001 * 0.001;
}

// loose tells whether two numbers agree to a relative 1e-5.
fn loose(a: f64, b: f64) -> bool {
    var d = a - b;
    if d < 0.0 { d = -d; }
    var s = b;
    if s < 0.0 { s = -s; }
    return d <= 0.00001 * (s + 0.00001);
}

fn all_loose(a: scientific::Arr&, v: f64[:]) -> bool {
    if a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len { if !common::loose(a.data[i], v[i]) { return false; } i += 1; }
    return true;
}

fn tq_zero(x: f64) -> f64 { return 0.0; }

fn tq_one(x: f64) -> f64 { return 1.0; }

fn tq_half_x(x: f64) -> f64 { return 0.5 * x; }

fn tq_one_x(x: f64) -> f64 { return 1.0 + x; }

fn tq_m1(x: f64) -> f64 { return -1.0; }

fn tq_p1(x: f64) -> f64 { return 1.0; }

fn tq_zero_xy(x: f64, y: f64) -> f64 { return 0.0; }

fn tq_one_xy(x: f64, y: f64) -> f64 { return 1.0; }

fn tq_half_y(x: f64, y: f64) -> f64 { return 0.5 * y; }

fn tq_y_half(x: f64, y: f64) -> f64 { return y + 0.5; }

fn tq_m1_xy(x: f64, y: f64) -> f64 { return -1.0; }

fn tq_p1_xy(x: f64, y: f64) -> f64 { return 1.0; }

// The trigonometric and hyperbolic functions of a matrix are pinned against SciPy.
fn tm_mat(v: f64[:]) -> scientific::Arr {
    let m, f = scientific::from_rows(2, 2, v);
    return m;
}

fn sf_sos() -> scientific::Arr {
    let s, f = scientific::from_rows(2, 6, [0.01856301062689717, 0.03712602125379434, 0.01856301062689717, 1.0, -0.6727409111915273, 0.14453519983312102, 1.0, 2.0, 1.0, 1.0, -0.8976579400366445, 0.5271869046315972]);
    return s;
}

fn rp_signal() -> f64[>..] {
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < 20 { v.push(common::mathshim_sin(i * 0.3) + 0.1 * i); i += 1; }
    return copy(v);
}

fn dc_signal() -> f64[>..] {
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < 60 { v.push(common::mathshim_sin(i * 0.3) + 0.1 * i); i += 1; }
    return copy(v);
}

fn sp_signal() -> f64[>..] {
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < 40 { v.push(common::mathshim_sin(i * 0.7) + 0.05 * i); i += 1; }
    return copy(v);
}

fn eg_matrix() -> scientific::Csr {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < 30 {
        ri.push(i); ci.push(i); v.push(2.0 + 0.1 * i);
        if i + 1 < 30 {
            ri.push(i); ci.push(i + 1); v.push(-1.0);
            ri.push(i + 1); ci.push(i); v.push(-1.0);
        }
        i += 1;
    }
    let s, f = scientific::csr_from_triplets(30, 30, copy(ri), copy(ci), copy(v));
    assert(f.kind == scientific::F_NONE);
    return s;
}

// maxdiff gives the largest absolute difference of the array and the
// list of the caller.
fn maxdiff(a: scientific::Arr&, v: f64[:]) -> f64 {
    var w: f64 = 0.0;
    var i: i64 = 0;
    while i < v.len {
        var d = a.data[i] - v[i];
        if d < 0.0 { d = -d; }
        if d > w { w = d; }
        i += 1;
    }
    return w;
}

// rel_close tells whether the values agree to a relative tolerance.
fn rel_close(a: scientific::Arr&, v: f64[:], tol: f64) -> bool {
    if a.data.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        var d = a.data[i] - v[i];
        if d < 0.0 { d = -d; }
        var s = v[i];
        if s < 0.0 { s = -s; }
        if d > tol * (1.0 + s) { return false; }
        i += 1;
    }
    return true;
}

// res_match tells whether each expected pole and residue has a partner in
// the result. Equal poles keep their order, and poles of one size may
// come in either order.
fn res_match(rr: scientific::Arr&, ri: scientific::Arr&, pr: scientific::Arr&, pi: scientific::Arr&, er: f64[:], ei: f64[:], ep: f64[:], eq: f64[:], tol: f64) -> bool {
    let n = er.len;
    if rr.data.len != n || ri.data.len != n || pr.data.len != n || pi.data.len != n { return false; }
    var used: i64[>..] = [];
    var i: i64 = 0;
    while i < n { used.push(0); i += 1; }
    i = 0;
    while i < n {
        var found = false;
        var j: i64 = 0;
        while j < n && !found {
            if used[j] == 0 && common::mathshim_abs(pr.data[j] - ep[i]) < 1.0e-6 && common::mathshim_abs(pi.data[j] - eq[i]) < 1.0e-6 {
                used[j] = 1;
                found = true;
                if common::mathshim_abs(rr.data[j] - er[i]) > tol * (1.0 + common::mathshim_abs(er[i])) { return false; }
                if common::mathshim_abs(ri.data[j] - ei[i]) > tol * (1.0 + common::mathshim_abs(ei[i])) { return false; }
            }
            j += 1;
        }
        if !found { return false; }
        i += 1;
    }
    return true;
}

// tri_check checks the eigenvectors of a symmetric tridiagonal
// matrix. Each column must stay an eigenvector and the columns must
// be orthonormal.
fn tri_check(d: f64[:], e: f64[:], w, v) {
    let n = d.len;
    let nc = v.shape[1];
    var j: i64 = 0;
    while j < nc {
        var i: i64 = 0;
        while i < n {
            var acc = d[i] * v.data[i * nc + j];
            if i > 0 { acc += e[i - 1] * v.data[(i - 1) * nc + j]; }
            if i < n - 1 { acc += e[i] * v.data[(i + 1) * nc + j]; }
            let t = acc - w.data[j] * v.data[i * nc + j];
            assert(common::mathshim_abs(t) < 1.0e-8);
            i += 1;
        }
        j += 1;
    }
    j = 0;
    while j < nc {
        var j2: i64 = 0;
        while j2 < nc {
            var dot: f64 = 0.0;
            var i: i64 = 0;
            while i < n {
                dot += v.data[i * nc + j] * v.data[i * nc + j2];
                i += 1;
            }
            var want: f64 = 0.0;
            if j == j2 { want = 1.0; }
            assert(common::mathshim_abs(dot - want) < 1.0e-8);
            j2 += 1;
        }
        j += 1;
    }
}

// vchk12 compares an array with the values of SciPy. The values must
// agree to 1.0e-12.
fn vchk12(a, f, v: f64[:]) {
    assert(f.kind == scientific::F_NONE);
    assert(a.data.len == v.len);
    var i: i64 = 0;
    while i < v.len {
        assert(common::mathshim_abs(a.data[i] - v[i]) <= 1.0e-12 * (1.0 + common::mathshim_abs(v[i])));
        i += 1;
    }
}

// achk compares two arrays of the same length.
fn achk(a, b) {
    assert(a.data.len == b.data.len);
    var i: i64 = 0;
    while i < a.data.len {
        assert(common::mathshim_abs(a.data[i] - b.data[i]) <= 1.0e-8 * (1.0 + common::mathshim_abs(b.data[i])));
        i += 1;
    }
}

// ct_orth builds an orthogonal matrix of side m from reflections. A
// small generator of integers gives the vectors.
fn ct_orth(m: i64, seed: i64) -> scientific::Arr {
    var q: f64[>..] = [];
    var i: i64 = 0;
    while i < m * m { q.push(0.0); i += 1; }
    i = 0;
    while i < m { q[i * m + i] = 1.0; i += 1; }
    var s: i64 = seed;
    var k: i64 = 0;
    while k < m {
        var v: f64[>..] = [];
        var nv: f64 = 0.0;
        i = 0;
        while i < m {
            s = (s * 1103515245 + 12345) % 2147483648;
            let x = (s as f64) / 2147483648.0 - 0.5;
            v.push(x);
            nv += x * x;
            i += 1;
        }
        i = 0;
        while i < m {
            var t: f64 = 0.0;
            var j: i64 = 0;
            while j < m { t += q[i * m + j] * v[j]; j += 1; }
            j = 0;
            while j < m { q[i * m + j] = q[i * m + j] - 2.0 * t * v[j] / nv; j += 1; }
            i += 1;
        }
        k += 1;
    }
    return scientific::Arr { ndim: 2, shape: [m, m], data: q };
}

// ct_close checks that two buffers agree to a tolerance.
fn ct_close(a: f64[:], b: f64[:], tol: f64) {
    assert(a.len == b.len);
    var i: i64 = 0;
    while i < a.len {
        if !(common::mathshim_abs(a[i] - b[i]) <= tol) { print("ct_close: index ", i, " got ", a[i], " want ", b[i], "\n"); }
        assert(common::mathshim_abs(a[i] - b[i]) <= tol);
        i += 1;
    }
}

// ct_check runs the decomposition of one matrix at one split and checks
// the product, the orthogonality of the factors and their block shape.
fn ct_check(x, p: i64, q: i64, tol: f64) {
    let m = x.shape[0];
    let u, cs, vh, f = scientific::cossin(x, p, q);
    assert(f.kind == scientific::F_NONE);
    let uc, f1 = scientific::matmul(u, cs);
    assert(f1.kind == scientific::F_NONE);
    let prod, f2 = scientific::matmul(uc, vh);
    assert(f2.kind == scientific::F_NONE);
    common::ct_close(prod.data, x.data, tol);
    var i: i64 = 0;
    while i < m {
        var j: i64 = 0;
        while j < m {
            var a: f64 = 0.0;
            var b: f64 = 0.0;
            var k: i64 = 0;
            while k < m {
                a += u.data[k * m + i] * u.data[k * m + j];
                b += vh.data[i * m + k] * vh.data[j * m + k];
                k += 1;
            }
            var id: f64 = 0.0;
            if i == j { id = 1.0; }
            assert(common::mathshim_abs(a - id) <= tol);
            assert(common::mathshim_abs(b - id) <= tol);
            // The factors hold zeros off their blocks.
            if (i < p && j >= p) || (i >= p && j < p) { assert(u.data[i * m + j] == 0.0); }
            if (i < q && j >= q) || (i >= q && j < q) { assert(vh.data[i * m + j] == 0.0); }
            j += 1;
        }
        i += 1;
    }
}

// ct_sign checks that a block equals a pinned block up to the sign of
// each pair. The columns of the left block pair with the rows of the
// right one, so the two signs must agree.
fn ct_sign(u: f64[:], vh: f64[:], pu: f64[:], pv: f64[:], n: i64) {
    var c: i64 = 0;
    while c < n {
        var dp: f64 = 0.0;
        var dm: f64 = 0.0;
        var ep: f64 = 0.0;
        var em: f64 = 0.0;
        var i: i64 = 0;
        while i < n {
            dp = common::max_abs(dp, u[i * n + c] - pu[i * n + c]);
            dm = common::max_abs(dm, u[i * n + c] + pu[i * n + c]);
            ep = common::max_abs(ep, vh[c * n + i] - pv[c * n + i]);
            em = common::max_abs(em, vh[c * n + i] + pv[c * n + i]);
            i += 1;
        }
        assert((dp <= 1.0e-12 && ep <= 1.0e-12) || (dm <= 1.0e-12 && em <= 1.0e-12));
        c += 1;
    }
}

fn max_abs(a: f64, d: f64) -> f64 {
    var v = d;
    if v < 0.0 { v = -v; }
    if v > a { return v; }
    return a;
}

// ct_small checks a matrix with an angle of eps at the split 2 and 2.
// The angles of SciPy there are 0 and eps.
fn ct_small(g, eps: f64) {
    common::ct_check(g, 2, 2, 1.0e-12);
    common::ct_check(g, 1, 2, 1.0e-12);
    let g1, g2, gt, g4, g5, gf = scientific::cossin_separate(g, 2, 2);
    assert(gf.kind == scientific::F_NONE);
    assert(gt.data[0] == 0.0);
    assert(common::mathshim_abs(gt.data[1] - eps) <= 1.0e-15 * eps + 1.0e-18);
}

// qt_pair builds a pair of side n from a small integer generator. The
// kind 1 makes the second matrix singular.
fn qt_pair(n: i64, seed: i64, kind: i64) -> scientific::Arr, scientific::Arr {
    var a: f64[>..] = [];
    var b: f64[>..] = [];
    var s: i64 = seed;
    var i: i64 = 0;
    while i < n * n {
        s = (s * 1103515245 + 12345) % 2147483648;
        a.push((s as f64) / 1073741824.0 - 1.0);
        s = (s * 1103515245 + 12345) % 2147483648;
        b.push((s as f64) / 1073741824.0 - 1.0);
        i += 1;
    }
    if kind == 1 {
        i = 0;
        while i < n { b[i * n + 1] = 0.0; i += 1; }
    }
    return scientific::Arr { ndim: 2, shape: [n, n], data: a }, scientific::Arr { ndim: 2, shape: [n, n], data: b };
}

// qt_recon gives the largest entry of Q X Z^T - A for matrices of side
// n that start at the place off of their buffers.
fn qt_recon(q: f64[:], x: f64[:], z: f64[:], a: f64[:], off: i64, n: i64) -> f64 {
    var worst: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            var s: f64 = 0.0;
            var k: i64 = 0;
            while k < n {
                var l: i64 = 0;
                while l < n {
                    s += q[off + i * n + k] * x[off + k * n + l] * z[off + j * n + l];
                    l += 1;
                }
                k += 1;
            }
            let d = common::mathshim_abs(s - a[off + i * n + j]);
            if d > worst { worst = d; }
            j += 1;
        }
        i += 1;
    }
    return worst;
}

// qt_orth gives the largest entry of F^T F - I.
fn qt_orth(f: f64[:], off: i64, n: i64) -> f64 {
    var worst: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            var s: f64 = 0.0;
            var k: i64 = 0;
            while k < n { s += f[off + k * n + i] * f[off + k * n + j]; k += 1; }
            if i == j { s -= 1.0; }
            let d = common::mathshim_abs(s);
            if d > worst { worst = d; }
            j += 1;
        }
        i += 1;
    }
    return worst;
}

// qt_shape gives the largest entry that breaks the form: b under the
// diagonal, a under the first subdiagonal, and two subdiagonal entries
// in a row.
fn qt_shape(aa: f64[:], bb: f64[:], off: i64, n: i64) -> f64 {
    var worst: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < i {
            let d = common::mathshim_abs(bb[off + i * n + j]);
            if d > worst { worst = d; }
            if j < i - 1 {
                let e = common::mathshim_abs(aa[off + i * n + j]);
                if e > worst { worst = e; }
            }
            j += 1;
        }
        if i > 1 && aa[off + i * n + i - 1] != 0.0 && aa[off + (i - 1) * n + i - 2] != 0.0 { worst = 1.0; }
        i += 1;
    }
    return worst;
}

// qt_real_case checks one real decomposition of a pair.
fn qt_real_case(a: scientific::Arr&, b: scientific::Arr&, n: i64) {
    let aa, bb, q, z, f = scientific::qz(a, b);
    assert(f.kind == scientific::F_NONE);
    assert(common::qt_recon(q.data, aa.data, z.data, a.data, 0, n) < 1.0e-12);
    assert(common::qt_recon(q.data, bb.data, z.data, b.data, 0, n) < 1.0e-12);
    assert(common::qt_orth(q.data, 0, n) < 1.0e-12);
    assert(common::qt_orth(z.data, 0, n) < 1.0e-12);
    assert(common::qt_shape(aa.data, bb.data, 0, n) < 1.0e-12);
    var i: i64 = 0;
    while i < n { assert(bb.data[i * n + i] >= 0.0); i += 1; }
}

// qt_sorted sorts a small buffer in place.
fn qt_sorted(v: f64[>..]&) {
    var i: i64 = 1;
    while i < v.len {
        let x = v[i];
        var j = i - 1;
        while j >= 0 && v[j] > x { v[j + 1] = v[j]; j -= 1; }
        v[j + 1] = x;
        i += 1;
    }
}

// qt_chosen tells whether a rule chooses the eigenvalue alpha / beta.
fn qt_chosen(rule: const u8[:], are: f64, aim: f64, be: f64) -> bool {
    if rule == "ouc" {
        if be == 0.0 { return are != 0.0 || aim != 0.0; }
        return (are * are + aim * aim) > be * be;
    }
    if be == 0.0 { return false; }
    if rule == "lhp" { return are / be < 0.0; }
    if rule == "rhp" { return are / be > 0.0; }
    return (are * are + aim * aim) < be * be;
}

// qt_ord checks one ordered decomposition. The products and the forms
// hold, and no unchosen eigenvalue sits before a chosen one.
fn qt_ord(a: scientific::Arr&, b: scientific::Arr&, n: i64, rule: const u8[:]) {
    let aa, bb, al, be, q, z, f = scientific::ordqz(a, b, rule);
    assert(f.kind == scientific::F_NONE);
    assert(common::qt_recon(q.data, aa.data, z.data, a.data, 0, n) < 1.0e-12);
    assert(common::qt_recon(q.data, bb.data, z.data, b.data, 0, n) < 1.0e-12);
    assert(common::qt_orth(q.data, 0, n) < 1.0e-12);
    assert(common::qt_orth(z.data, 0, n) < 1.0e-12);
    assert(common::qt_shape(aa.data, bb.data, 0, n) < 1.0e-12);
    assert(al.data.len == 2 * n && be.data.len == n);
    var seen_unchosen = false;
    var i: i64 = 0;
    while i < n {
        let c = common::qt_chosen(rule, al.data[2 * i], al.data[2 * i + 1], be.data[i]);
        if c { assert(!seen_unchosen); } else { seen_unchosen = true; }
        i += 1;
    }
}
