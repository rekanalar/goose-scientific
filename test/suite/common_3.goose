// The helpers that the tests of more than one group share.
namespace common_3;

import scientific from "../..";
import os;
import common;
import common_2;

// check_eig tests the properties of the eigenpairs of a complex matrix. Each
// pair satisfies the eigen equation. Each vector has the length one and a real
// largest component above zero.
fn check_eig(a: scientific::CArr&, w: scientific::CArr&, v: scientific::CArr&) {
    let n = a.shape[0];
    var norm: f64 = 0.0;
    var i: i64 = 0;
    while i < a.data.len {
        var x = a.data[i];
        if x < 0.0 { x = -x; }
        if x > norm { norm = x; }
        i += 1;
    }
    var col: i64 = 0;
    while col < n {
        var len2: f64 = 0.0;
        var best: f64 = -1.0;
        var br: f64 = 0.0;
        var bi: f64 = 0.0;
        var r: i64 = 0;
        while r < n {
            // The row r of A v minus lambda v.
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            var k: i64 = 0;
            while k < n {
                let ar = a.data[2 * (r * n + k)];
                let ai = a.data[2 * (r * n + k) + 1];
                let xr = v.data[2 * (k * n + col)];
                let xi = v.data[2 * (k * n + col) + 1];
                sr += ar * xr - ai * xi;
                si += ar * xi + ai * xr;
                k += 1;
            }
            let vr = v.data[2 * (r * n + col)];
            let vi = v.data[2 * (r * n + col) + 1];
            let lr = w.data[2 * col];
            let li = w.data[2 * col + 1];
            var er = sr - (lr * vr - li * vi);
            var ei = si - (lr * vi + li * vr);
            if er < 0.0 { er = -er; }
            if ei < 0.0 { ei = -ei; }
            assert(er <= 1.0e-10 * (1.0 + norm * (n as f64)));
            assert(ei <= 1.0e-10 * (1.0 + norm * (n as f64)));
            len2 += vr * vr + vi * vi;
            if vr * vr + vi * vi > best {
                best = vr * vr + vi * vi;
                br = vr;
                bi = vi;
            }
            r += 1;
        }
        assert(common::close(len2, 1.0, 1.0e-12));
        var abi = bi;
        if abi < 0.0 { abi = -abi; }
        assert(br > 0.0 && abi < 1.0e-12);
        col += 1;
    }
    return;
}

// random_hermitian gives a Hermitian matrix of normal complex values.
fn random_hermitian(rng: scientific::Rng&, n: i64) -> scientific::CArr {
    let m = common_2::random_complex(rng, n, n);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            d.push(m.data[2 * (i * n + j)] + m.data[2 * (j * n + i)]);
            d.push(m.data[2 * (i * n + j) + 1] - m.data[2 * (j * n + i) + 1]);
            j += 1;
        }
        i += 1;
    }
    let h, f = scientific::complex_array(scientific::from_rows(n, n, common_3::c_parts(d, 0)), scientific::from_rows(n, n, common_3::c_parts(d, 1)));
    assert(f.kind == scientific::F_NONE);
    return h;
}

// c_parts gives the real parts, when which is zero, or the imaginary parts of
// an interleaved buffer.
fn c_parts(d: f64[:], which: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < d.len / 2 {
        out.push(d[2 * i + which]);
        i += 1;
    }
    return out;
}

// check_eigh tests the properties of the result of ceigh for a Hermitian matrix.
fn check_eigh(a: scientific::CArr&) {
    let n = a.shape[0];
    let w, v, f = scientific::ceigh(a);
    assert(f.kind == scientific::F_NONE);
    assert(w.data.len == n);
    var i: i64 = 1;
    while i < n {
        assert(w.data[i - 1] <= w.data[i]);
        i += 1;
    }
    assert(common_2::gram_dev(v, true, n) < 1.0e-12);
    var wc: f64[>..] = [];
    i = 0;
    while i < n {
        wc.push(w.data[i]);
        wc.push(0.0);
        i += 1;
    }
    var shape: i64[..8] = [];
    shape.push(n);
    let wcarr = scientific::CArr { ndim: 1, shape: shape, data: copy(wc) };
    common_3::check_eig(a, wcarr, v);
    let vals, fv = scientific::ceigvalsh(a);
    assert(fv.kind == scientific::F_NONE);
    i = 0;
    while i < n {
        assert(vals.data[i] == w.data[i]);
        i += 1;
    }
    return;
}

// check_general_pin compares ceig with the values of NumPy. The values come
// sorted. The vectors are compared when withv holds.
fn check_general_pin(n: i64, are: f64[:], aim: f64[:], wre: f64[:], wim: f64[:], vre: f64[:], vim: f64[:], withv: bool) {
    let a, f0 = scientific::complex_array(scientific::from_rows(n, n, are), scientific::from_rows(n, n, aim));
    assert(f0.kind == scientific::F_NONE);
    let w, v, f = scientific::ceig(a);
    assert(f.kind == scientific::F_NONE);
    assert(w.data.len == 2 * n && v.shape[0] == n && v.shape[1] == n);
    var ws: f64[>..] = [];
    var vs: f64[>..] = [];
    common_2::sort_eig(w, v, n, ws, vs);
    var i: i64 = 0;
    while i < n {
        assert(common::close(ws[2 * i], wre[i], 1.0e-12) && common::close(ws[2 * i + 1], wim[i], 1.0e-12));
        i += 1;
    }
    if withv {
        i = 0;
        while i < n * n {
            assert(common::close(vs[2 * i], vre[i], 1.0e-12) && common::close(vs[2 * i + 1], vim[i], 1.0e-12));
            i += 1;
        }
    }
    common_3::check_eig(a, w, v);
    let w2, f2 = scientific::ceigvals(a);
    assert(f2.kind == scientific::F_NONE);
    assert(common_2::cmat_err(w, w2) == 0.0);
    return;
}

// A complex matrix of side n from two lists of parts.
fn gm(n: i64, re: f64[:], im: f64[:]) -> scientific::CArr {
    let c, f = scientific::complex_array(common::mx(n, n, re), common::mx(n, n, im));
    assert(f.kind == scientific::F_NONE);
    return c;
}

fn gclose(a: f64, b: f64, tol: f64) -> bool {
    var d = a - b;
    if d < 0.0 { d = -d; }
    var s = b;
    if s < 0.0 { s = -s; }
    return d <= tol * (1.0 + s);
}

// gnorm gives the largest absolute part of a complex buffer.
fn gnorm(d: f64[:]) -> f64 {
    var top: f64 = 0.0;
    var i: i64 = 0;
    while i < d.len {
        var x = d[i];
        if x < 0.0 { x = -x; }
        if x > top { top = x; }
        i += 1;
    }
    return top;
}

// pair_residual gives the largest error of the eigen equation of a pair,
// relative to the sizes of the terms. It also asserts the length one and the
// real largest component of each vector.
fn pair_residual(a: scientific::CArr&, b: scientific::CArr&, al: scientific::CArr&, be: scientific::CArr&, v: scientific::CArr&) -> f64 {
    let n = a.shape[0];
    let na = common_3::gnorm(a.data);
    let nb = common_3::gnorm(b.data);
    var worst: f64 = 0.0;
    var k: i64 = 0;
    while k < n {
        let alr = al.data[2 * k];
        let ali = al.data[2 * k + 1];
        let ber = be.data[2 * k];
        var len: f64 = 0.0;
        var best: f64 = -1.0;
        var at: i64 = 0;
        var i: i64 = 0;
        while i < n {
            let pr = v.data[2 * (i * n + k)];
            let pi = v.data[2 * (i * n + k) + 1];
            len += pr * pr + pi * pi;
            if pr * pr + pi * pi > best {
                best = pr * pr + pi * pi;
                at = i;
            }
            i += 1;
        }
        assert(common_3::gclose(len, 1.0, 1.0e-12));
        assert(v.data[2 * (at * n + k) + 1] == 0.0);
        assert(v.data[2 * (at * n + k)] > 0.0);
        i = 0;
        while i < n {
            var ar: f64 = 0.0;
            var ai: f64 = 0.0;
            var br: f64 = 0.0;
            var bi: f64 = 0.0;
            var j: i64 = 0;
            while j < n {
                let vr = v.data[2 * (j * n + k)];
                let vi = v.data[2 * (j * n + k) + 1];
                ar += a.data[2 * (i * n + j)] * vr - a.data[2 * (i * n + j) + 1] * vi;
                ai += a.data[2 * (i * n + j)] * vi + a.data[2 * (i * n + j) + 1] * vr;
                br += b.data[2 * (i * n + j)] * vr - b.data[2 * (i * n + j) + 1] * vi;
                bi += b.data[2 * (i * n + j)] * vi + b.data[2 * (i * n + j) + 1] * vr;
                j += 1;
            }
            let rr = ber * ar - (alr * br - ali * bi);
            let ri = ber * ai - (alr * bi + ali * br);
            var e = rr;
            if e < 0.0 { e = -e; }
            if e > worst { worst = e; }
            e = ri;
            if e < 0.0 { e = -e; }
            if e > worst { worst = e; }
            i += 1;
        }
        // The terms have the size of beta |A| + |alpha| |B| at most.
        var scale = ber * na + common_3::gnorm([alr, ali]) * nb;
        if scale == 0.0 { scale = 1.0; }
        worst = worst / scale;
        k += 1;
    }
    return worst;
}

// pair_unit tells that every column of a matrix of vectors is finite and has the
// length one.
fn pair_unit(v: scientific::CArr&) -> bool {
    let n = v.shape[0];
    var k: i64 = 0;
    while k < n {
        var len: f64 = 0.0;
        var i: i64 = 0;
        while i < n {
            let p = v.data[2 * (i * n + k)];
            let q = v.data[2 * (i * n + k) + 1];
            len += p * p + q * q;
            i += 1;
        }
        if !(len > 0.999999999999 && len < 1.000000000001) { return false; }
        k += 1;
    }
    return true;
}

// pair_find gives the index of the eigenvalue near the pinned value, or -1.
fn pair_find(al: scientific::CArr&, be: scientific::CArr&, n: i64, wr: f64, wi: f64) -> i64 {
    var k: i64 = 0;
    while k < n {
        let b = be.data[2 * k];
        if b != 0.0 {
            if common_3::gclose(al.data[2 * k] / b, wr, 1.0e-9) && common_3::gclose(al.data[2 * k + 1] / b, wi, 1.0e-9) { return k; }
        }
        k += 1;
    }
    return -1;
}

fn random_definite(rng: scientific::Rng&, n: i64) -> scientific::CArr {
    let m = common_2::random_complex(rng, n, n);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            var k: i64 = 0;
            while k < n {
                // The entry of M M^H at (i, j).
                let ar = m.data[2 * (i * n + k)];
                let ai = m.data[2 * (i * n + k) + 1];
                let br = m.data[2 * (j * n + k)];
                let bi = -m.data[2 * (j * n + k) + 1];
                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
                k += 1;
            }
            if i == j { sr += 1.0; }
            d.push(sr);
            d.push(si);
            j += 1;
        }
        i += 1;
    }
    let h, f = scientific::complex_array(common::mx(n, n, common_3::c_parts(d, 0)), common::mx(n, n, common_3::c_parts(d, 1)));
    assert(f.kind == scientific::F_NONE);
    return h;
}

// check_definite checks the eigen equation A v = w B v and the B-orthonormality
// of the vectors of a Hermitian pair.
fn check_definite(a: scientific::CArr&, b: scientific::CArr&, w: scientific::Arr&, v: scientific::CArr&) {
    let n = a.shape[0];
    let na = common_3::gnorm(a.data);
    let nb = common_3::gnorm(b.data);
    var k: i64 = 0;
    while k < n {
        var best: f64 = -1.0;
        var at: i64 = 0;
        var i: i64 = 0;
        while i < n {
            let p = v.data[2 * (i * n + k)];
            let q = v.data[2 * (i * n + k) + 1];
            if p * p + q * q > best {
                best = p * p + q * q;
                at = i;
            }
            i += 1;
        }
        assert(v.data[2 * (at * n + k) + 1] == 0.0);
        assert(v.data[2 * (at * n + k)] > 0.0);
        var l: i64 = 0;
        while l < n {
            // The entry (k, l) of V^H B V, and the residual of the column.
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            i = 0;
            while i < n {
                var br: f64 = 0.0;
                var bi: f64 = 0.0;
                var j: i64 = 0;
                while j < n {
                    let vr = v.data[2 * (j * n + l)];
                    let vi = v.data[2 * (j * n + l) + 1];
                    br += b.data[2 * (i * n + j)] * vr - b.data[2 * (i * n + j) + 1] * vi;
                    bi += b.data[2 * (i * n + j)] * vi + b.data[2 * (i * n + j) + 1] * vr;
                    j += 1;
                }
                let cr = v.data[2 * (i * n + k)];
                let ci = -v.data[2 * (i * n + k) + 1];
                sr += cr * br - ci * bi;
                si += cr * bi + ci * br;
                i += 1;
            }
            var target: f64 = 0.0;
            if k == l { target = 1.0; }
            assert(common_3::gclose(sr, target, 1.0e-10));
            assert(common_3::gclose(si, 0.0, 1.0e-10));
            l += 1;
        }
        var worst: f64 = 0.0;
        i = 0;
        while i < n {
            var ar: f64 = 0.0;
            var ai: f64 = 0.0;
            var br: f64 = 0.0;
            var bi: f64 = 0.0;
            var j: i64 = 0;
            while j < n {
                let vr = v.data[2 * (j * n + k)];
                let vi = v.data[2 * (j * n + k) + 1];
                ar += a.data[2 * (i * n + j)] * vr - a.data[2 * (i * n + j) + 1] * vi;
                ai += a.data[2 * (i * n + j)] * vi + a.data[2 * (i * n + j) + 1] * vr;
                br += b.data[2 * (i * n + j)] * vr - b.data[2 * (i * n + j) + 1] * vi;
                bi += b.data[2 * (i * n + j)] * vi + b.data[2 * (i * n + j) + 1] * vr;
                j += 1;
            }
            var e = ar - w.data[k] * br;
            if e < 0.0 { e = -e; }
            if e > worst { worst = e; }
            e = ai - w.data[k] * bi;
            if e < 0.0 { e = -e; }
            if e > worst { worst = e; }
            i += 1;
        }
        var wk = w.data[k];
        if wk < 0.0 { wk = -wk; }
        assert(worst < 1.0e-10 * (na + wk * nb + 1.0));
        k += 1;
    }
    return;
}

// nd_same tells that an array holds the expected values. A NaN matches a NaN.
fn nd_same(a: scientific::Arr&, want: f64[:]) -> bool {
    if a.data.len != want.len { return false; }
    var i: i64 = 0;
    while i < want.len {
        let x = a.data[i];
        let y = want[i];
        if y != y {
            if x == x { return false; }
        } else if !common_3::gclose(x, y, 1.0e-12) {
            return false;
        }
        i += 1;
    }
    return true;
}

fn nd_line(v: f64[:]) -> scientific::Arr {
    let a, f = scientific::from_values(v);
    assert(f.kind == scientific::F_NONE);
    return a;
}

// The inputs of the pinned filter tests. Printed by test/gen_ndimage.py.
fn ni_arr(k: i64) -> scientific::Arr {
    if k == 1 { return common_3::ni_nd([9], [-0.284, -0.941, -0.051, -0.025, -0.076, -0.712, -0.985, -0.777, 0.236]); }
    if k == 2 { return common_3::ni_nd([5, 6], [-0.29, -0.707, -0.843, 0.722, 0.089, 0.684, 0.264, -1.744, 0.083, 0.43, 0.314, 0.396, -0.047, -0.982, -1.931, 0.802, 1.014, 0.384, 0.27, 0.327, 0.71, 1.732, -0.781, 0.263, -0.877, 0.248, -0.041, 1.765, 0.191, -1.532]); }
    return common_3::ni_nd([3, 4, 5], [-0.133, 0.39, 0.118, 0.947, -1.544, 0.385, 0.157, -0.891, -0.563, -0.448, 0.057, -2.031, 0.099, -1.645, -0.096, -1.88, 0.998, 0.408, 0.279, 0.963, 0.535, 1.126, 0.615, -0.248, -0.721, 1.623, -0.442, 0.991, 1.457, -0.014, -0.119, -1.219, -1.526, 1.197, -0.931, -0.115, -0.697, -0.087, 0.203, 0.267, -0.71, -0.981, 1.344, -0.687, 1.059, -0.148, 1.379, -1.725, -0.267, -0.022, 0.804, -0.075, -0.727, 0.847, 0.607, 0.538, 0.854, -1.632, 0.749, -0.721]);
}

// ni_nd builds an array of a shape from row-major values.
fn ni_nd(shape: i64[:], v: h64[:]) -> scientific::Arr {
    let a, f = scientific::from_values(v);
    assert(f.kind == scientific::F_NONE);
    let b, g = scientific::reshape(a, shape);
    assert(g.kind == scientific::F_NONE);
    return b;
}

fn mo_arr(k: i64) -> scientific::Arr {
    if k == 1 { return common_3::ni_nd([12], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]); }
    if k == 2 { return common_3::ni_nd([6, 7], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]); }
    return common_3::ni_nd([3, 4, 5], [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
}

fn gsum(w: f64[:]) -> f64 {
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < w.len { s += w[i]; i += 1; }
    return s;
}

fn grange(w: f64[:]) -> f64 {
    var lo = w[0];
    var hi = w[0];
    var i: i64 = 1;
    while i < w.len {
        if w[i] < lo { lo = w[i]; }
        if w[i] > hi { hi = w[i]; }
        i += 1;
    }
    return hi - lo;
}

fn gwt(w: f64[:]) -> f64 {
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < w.len { s += w[i] * ((i + 1) as f64); i += 1; }
    return s;
}

fn gsecond(a: scientific::Arr, axis: i64, mode: const u8[:], cval: f64) -> scientific::Arr {
    let d, g = scientific::correlate1d(a, [1.0, -2.0, 1.0], axis, mode, cval, 0);
    assert(g.kind == scientific::F_NONE);
    return d;
}

fn gfirst(a: scientific::Arr, axis: i64, mode: const u8[:], cval: f64) -> scientific::Arr {
    let d, g = scientific::correlate1d(a, [-1.0, 0.0, 1.0], axis, mode, cval, 0);
    assert(g.kind == scientific::F_NONE);
    return d;
}

fn gshort(a: scientific::Arr, axis: i64, mode: const u8[:], cval: f64) -> scientific::Arr {
    let e = scientific::empty_arr();
    return e;
}

// tr_same compares the data of an array with the values of SciPy. The
// recursive spline filter and the cardinal weights agree to 1.0e-10.
fn tr_same(a: scientific::Arr&, want: f64[:]) -> bool {
    if a.data.len != want.len { return false; }
    var i: i64 = 0;
    while i < want.len {
        if !common_3::gclose(a.data[i], want[i], 1.0e-10) { return false; }
        i += 1;
    }
    return true;
}

fn tr_arr(k: i64) -> scientific::Arr {
    if k == 1 { return common_3::ni_nd([7], [-1.086, -0.324, -0.07, 0.026, -0.449, -0.552, 0.354]); }
    if k == 2 { return common_3::ni_nd([4, 5], [-0.079, 1.088, 0.341, -0.171, 0.109, 1.046, 0.696, -1.126, -0.221, 1.645, 0.263, -2.842, -0.602, 0.641, 1.635, 0.948, -1.511, -0.224, 0.448, 0.79]); }
    return common_3::ni_nd([3, 2, 4], [-1.034, -1.298, 0.573, 2.07, -0.687, 0.205, -0.41, 0.117, -1.643, -1.181, -1.723, 0.127, 2.037, 0.123, -0.106, -0.622, -0.829, 2.315, 0.562, 0.152, 1.15, -1.934, -0.5, 0.466]);
}

// ms_same compares the data of an array with the values of SciPy. A NaN
// in the pin needs a NaN in the result.
fn ms_same(a: scientific::Arr&, want: f64[:]) -> bool {
    if a.data.len != want.len { return false; }
    var i: i64 = 0;
    while i < want.len {
        let x = a.data[i];
        let y = want[i];
        if y != y {
            if x == x { return false; }
        } else if !common_3::gclose(x, y, 1.0e-12) {
            return false;
        }
        i += 1;
    }
    return true;
}

fn msrange(w: f64[:]) -> f64 {
    var lo = w[0];
    var hi = w[0];
    var i: i64 = 1;
    while i < w.len {
        if w[i] < lo { lo = w[i]; }
        if w[i] > hi { hi = w[i]; }
        i += 1;
    }
    return hi - lo;
}

// dia_same compares every place of a matrix of diagonals with the values
// of SciPy.
fn dia_same(s: scientific::Dia&, rows: i64, cols: i64, want: f64[:]) -> bool {
    var i: i64 = 0;
    while i < rows {
        var j: i64 = 0;
        while j < cols {
            let x, f = scientific::dia_get(s, i, j);
            if f.kind != scientific::F_NONE { return false; }
            if x != want[i * cols + j] { return false; }
            j += 1;
        }
        i += 1;
    }
    return true;
}

// bsr_same checks every place of a block matrix against the dense values.
fn bsr_same(s: scientific::Bsr&, rows: i64, cols: i64, want: f64[:]) -> bool {
    var i: i64 = 0;
    while i < rows {
        var j: i64 = 0;
        while j < cols {
            let x, f = scientific::bsr_get(s, i, j);
            if f.kind != scientific::F_NONE { return false; }
            if x != want[i * cols + j] { return false; }
            j += 1;
        }
        i += 1;
    }
    return true;
}

// The pins of the sparse matrices of keys, measured from SciPy.
fn dok_is(d: scientific::Dok&, r: f64[:], c: f64[:], v: f64[:]) -> bool {
    let rr, cc, vv, f = scientific::dok_keys(d);
    if f.kind != scientific::F_NONE || rr.data.len != r.len { return false; }
    var i: i64 = 0;
    while i < r.len {
        if rr.data[i] != r[i] || cc.data[i] != c[i] || !common::near(vv.data[i], v[i]) { return false; }
        i += 1;
    }
    return true;
}

fn dok_pin_p1() {
    var d, f0 = scientific::dok_new(3, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 2.0, 1.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 0.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 1.0, 3.0, 3.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 2.0, 2.0).kind == scientific::F_NONE);
    assert(common_3::dok_is(d, [2.0, 0.0, 1.0, 0.0], [1.0, 0.0, 3.0, 2.0], [5.0, 1.0, 3.0, 2.0]));
    assert(scientific::dok_nnz(d) == 4);
    let dn, fd = scientific::dok_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 5.0, 0.0, 0.0]));
    let t, ft = scientific::dok_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_3::dok_is(t, [1.0, 0.0, 3.0, 2.0], [2.0, 0.0, 1.0, 0.0], [5.0, 1.0, 3.0, 2.0]));
    let co, fc = scientific::dok_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 4);
    let cs, fs = scientific::dok_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 4);
    let mv, fm = scientific::dok_matvec(d, [-2.0, 1.0, 4.0, 0.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [6.0, 0.0, 5.0]));
    let sm, fz = scientific::dok_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 11.0));
    let dg, fg = scientific::dok_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [1.0, 0.0, 0.0]));
    let sd, fk = scientific::dok_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_3::dok_is(sd, [2.0, 0.0, 1.0, 0.0], [1.0, 0.0, 3.0, 2.0], [-7.5, -1.5, -4.5, -3.0]));
}

fn dok_pin_p2() {
    var d, f0 = scientific::dok_new(3, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 2.0, 1.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 0.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 1.0, 3.0, 3.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 2.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 2.0, 1.0, 6.0).kind == scientific::F_NONE);
    assert(common_3::dok_is(d, [2.0, 0.0, 1.0, 0.0], [1.0, 0.0, 3.0, 2.0], [6.0, 1.0, 3.0, 2.0]));
    assert(scientific::dok_nnz(d) == 4);
    let dn, fd = scientific::dok_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 6.0, 0.0, 0.0]));
    let t, ft = scientific::dok_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_3::dok_is(t, [1.0, 0.0, 3.0, 2.0], [2.0, 0.0, 1.0, 0.0], [6.0, 1.0, 3.0, 2.0]));
    let co, fc = scientific::dok_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 4);
    let cs, fs = scientific::dok_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 4);
    let mv, fm = scientific::dok_matvec(d, [-2.0, 1.0, 4.0, 0.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [6.0, 0.0, 6.0]));
    let sm, fz = scientific::dok_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 12.0));
    let dg, fg = scientific::dok_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [1.0, 0.0, 0.0]));
    let sd, fk = scientific::dok_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_3::dok_is(sd, [2.0, 0.0, 1.0, 0.0], [1.0, 0.0, 3.0, 2.0], [-9.0, -1.5, -4.5, -3.0]));
}

fn dok_pin_p3() {
    var d, f0 = scientific::dok_new(3, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 2.0, 1.0, 6.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 0.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 1.0, 3.0, 3.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 2.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(common_3::dok_is(d, [2.0, 1.0, 0.0], [1.0, 3.0, 2.0], [6.0, 3.0, 2.0]));
    assert(scientific::dok_nnz(d) == 3);
    let dn, fd = scientific::dok_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 6.0, 0.0, 0.0]));
    let t, ft = scientific::dok_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_3::dok_is(t, [1.0, 3.0, 2.0], [2.0, 1.0, 0.0], [6.0, 3.0, 2.0]));
    let co, fc = scientific::dok_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 3);
    let cs, fs = scientific::dok_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 3);
    let mv, fm = scientific::dok_matvec(d, [-2.0, 1.0, 4.0, 0.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [8.0, 0.0, 6.0]));
    let sm, fz = scientific::dok_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 11.0));
    let dg, fg = scientific::dok_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [0.0, 0.0, 0.0]));
    let sd, fk = scientific::dok_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_3::dok_is(sd, [2.0, 1.0, 0.0], [1.0, 3.0, 2.0], [-9.0, -4.5, -3.0]));
}

fn dok_pin_p4() {
    var d, f0 = scientific::dok_new(3, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 2.0, 1.0, 6.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 0.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 1.0, 3.0, 3.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 2.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 0.0, 9.0).kind == scientific::F_NONE);
    assert(common_3::dok_is(d, [2.0, 1.0, 0.0, 0.0], [1.0, 3.0, 2.0, 0.0], [6.0, 3.0, 2.0, 9.0]));
    assert(scientific::dok_nnz(d) == 4);
    let dn, fd = scientific::dok_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [9.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 6.0, 0.0, 0.0]));
    let t, ft = scientific::dok_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_3::dok_is(t, [1.0, 3.0, 2.0, 0.0], [2.0, 1.0, 0.0, 0.0], [6.0, 3.0, 2.0, 9.0]));
    let co, fc = scientific::dok_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 4);
    let cs, fs = scientific::dok_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 4);
    let mv, fm = scientific::dok_matvec(d, [-2.0, 1.0, 4.0, 0.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [-10.0, 0.0, 6.0]));
    let sm, fz = scientific::dok_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 20.0));
    let dg, fg = scientific::dok_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [9.0, 0.0, 0.0]));
    let sd, fk = scientific::dok_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_3::dok_is(sd, [2.0, 1.0, 0.0, 0.0], [1.0, 3.0, 2.0, 0.0], [-9.0, -4.5, -3.0, -13.5]));
}

fn dok_pin_p5() {
    var d, f0 = scientific::dok_new(3, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 2.0, 1.0, 6.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 0.0, 9.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 1.0, 3.0, 3.0).kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 0.0, 2.0, 2.0).kind == scientific::F_NONE);
    let p6, g6 = scientific::dok_pop(d, 1.0, 3.0);
    assert(g6.kind == scientific::F_NONE && p6 == 3.0);
    let p8, g8 = scientific::dok_pop(d, 2.0, 2.0);
    assert(g8.kind == scientific::F_NONE && p8 == 0.0);
    assert(common_3::dok_is(d, [2.0, 0.0, 0.0], [1.0, 0.0, 2.0], [6.0, 9.0, 2.0]));
    assert(scientific::dok_nnz(d) == 3);
    let dn, fd = scientific::dok_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [9.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0]));
    let t, ft = scientific::dok_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_3::dok_is(t, [1.0, 0.0, 2.0], [2.0, 0.0, 0.0], [6.0, 9.0, 2.0]));
    let co, fc = scientific::dok_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 3);
    let cs, fs = scientific::dok_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 3);
    let mv, fm = scientific::dok_matvec(d, [-2.0, 1.0, 4.0, 0.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [-10.0, 0.0, 6.0]));
    let sm, fz = scientific::dok_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 17.0));
    let dg, fg = scientific::dok_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [9.0, 0.0, 0.0]));
    let sd, fk = scientific::dok_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_3::dok_is(sd, [2.0, 0.0, 0.0], [1.0, 0.0, 2.0], [-9.0, -13.5, -3.0]));
}

fn dok_pin_p10_0() {
    var d, f0 = scientific::dok_new(3, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 1.0, 1.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::dok_setdiag(d, 0, [2.0, 2.0, 2.0]).kind == scientific::F_NONE);
    assert(common_3::dok_is(d, [1.0, 0.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 2.0]));
    assert(scientific::dok_nnz(d) == 3);
    let dn, fd = scientific::dok_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]));
    let t, ft = scientific::dok_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_3::dok_is(t, [1.0, 0.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 2.0]));
    let co, fc = scientific::dok_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 3);
    let cs, fs = scientific::dok_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 3);
    let mv, fm = scientific::dok_matvec(d, [-2.0, 1.0, 4.0, 0.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [-4.0, 2.0, 8.0]));
    let sm, fz = scientific::dok_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 6.0));
    let dg, fg = scientific::dok_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [2.0, 2.0, 2.0]));
    let sd, fk = scientific::dok_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_3::dok_is(sd, [1.0, 0.0, 2.0], [1.0, 0.0, 2.0], [-3.0, -3.0, -3.0]));
}

fn dok_pin_p10_1() {
    var d, f0 = scientific::dok_new(3, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 1.0, 1.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::dok_setdiag(d, 0, [1.0, 0.0, 3.0]).kind == scientific::F_NONE);
    assert(common_3::dok_is(d, [0.0, 2.0], [0.0, 2.0], [1.0, 3.0]));
    assert(scientific::dok_nnz(d) == 2);
    let dn, fd = scientific::dok_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0]));
    let t, ft = scientific::dok_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_3::dok_is(t, [0.0, 2.0], [0.0, 2.0], [1.0, 3.0]));
    let co, fc = scientific::dok_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 2);
    let cs, fs = scientific::dok_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 2);
    let mv, fm = scientific::dok_matvec(d, [-2.0, 1.0, 4.0, 0.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [-2.0, 0.0, 12.0]));
    let sm, fz = scientific::dok_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 4.0));
    let dg, fg = scientific::dok_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [1.0, 0.0, 3.0]));
    let sd, fk = scientific::dok_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_3::dok_is(sd, [0.0, 2.0], [0.0, 2.0], [-1.5, -4.5]));
}

fn dok_pin_p10_2() {
    var d, f0 = scientific::dok_new(3, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 1.0, 1.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::dok_setdiag(d, 1, [4.0, 5.0, 6.0]).kind == scientific::F_NONE);
    assert(common_3::dok_is(d, [1.0, 0.0, 1.0, 2.0], [1.0, 1.0, 2.0, 3.0], [5.0, 4.0, 5.0, 6.0]));
    assert(scientific::dok_nnz(d) == 4);
    let dn, fd = scientific::dok_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [0.0, 4.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 6.0]));
    let t, ft = scientific::dok_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_3::dok_is(t, [1.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 2.0], [5.0, 4.0, 5.0, 6.0]));
    let co, fc = scientific::dok_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 4);
    let cs, fs = scientific::dok_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 4);
    let mv, fm = scientific::dok_matvec(d, [-2.0, 1.0, 4.0, 0.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [4.0, 25.0, 0.0]));
    let sm, fz = scientific::dok_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 20.0));
    let dg, fg = scientific::dok_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [0.0, 5.0, 0.0]));
    let sd, fk = scientific::dok_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_3::dok_is(sd, [1.0, 0.0, 1.0, 2.0], [1.0, 1.0, 2.0, 3.0], [-7.5, -6.0, -7.5, -9.0]));
}

fn dok_pin_p10_3() {
    var d, f0 = scientific::dok_new(3, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 1.0, 1.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::dok_setdiag(d, -1, [7.0, 8.0]).kind == scientific::F_NONE);
    assert(common_3::dok_is(d, [1.0, 1.0, 2.0], [1.0, 0.0, 1.0], [5.0, 7.0, 8.0]));
    assert(scientific::dok_nnz(d) == 3);
    let dn, fd = scientific::dok_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 7.0, 5.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0]));
    let t, ft = scientific::dok_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_3::dok_is(t, [1.0, 0.0, 1.0], [1.0, 1.0, 2.0], [5.0, 7.0, 8.0]));
    let co, fc = scientific::dok_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 3);
    let cs, fs = scientific::dok_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 3);
    let mv, fm = scientific::dok_matvec(d, [-2.0, 1.0, 4.0, 0.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [0.0, -9.0, 8.0]));
    let sm, fz = scientific::dok_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 20.0));
    let dg, fg = scientific::dok_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [0.0, 5.0, 0.0]));
    let sd, fk = scientific::dok_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_3::dok_is(sd, [1.0, 1.0, 2.0], [1.0, 0.0, 1.0], [-7.5, -10.5, -12.0]));
}

fn dok_pin_p10_4() {
    var d, f0 = scientific::dok_new(3, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::dok_set(d, 1.0, 1.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::dok_setdiag(d, 1, [4.0, 5.0]).kind == scientific::F_NONE);
    assert(common_3::dok_is(d, [1.0, 0.0, 1.0], [1.0, 1.0, 2.0], [5.0, 4.0, 5.0]));
    assert(scientific::dok_nnz(d) == 3);
    let dn, fd = scientific::dok_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [0.0, 4.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    let t, ft = scientific::dok_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_3::dok_is(t, [1.0, 1.0, 2.0], [1.0, 0.0, 1.0], [5.0, 4.0, 5.0]));
    let co, fc = scientific::dok_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 3);
    let cs, fs = scientific::dok_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 3);
    let mv, fm = scientific::dok_matvec(d, [-2.0, 1.0, 4.0, 0.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [4.0, 25.0, 0.0]));
    let sm, fz = scientific::dok_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 14.0));
    let dg, fg = scientific::dok_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [0.0, 5.0, 0.0]));
    let sd, fk = scientific::dok_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_3::dok_is(sd, [1.0, 0.0, 1.0], [1.0, 1.0, 2.0], [-7.5, -6.0, -7.5]));
}
