// The helpers that the tests of more than one group share.
namespace common_2;

import scientific from "../..";
import os;
import common;

// qt_group checks the eigenvalues of the first group of a rule on the
// pinned pair. The values are compared as sets.
fn qt_group(rule: const u8[:], first: f64[:], rest: f64[:]) {
    let a = scientific::Arr { ndim: 2, shape: [4, 4], data: [2.0, 5.0, 8.0, 7.0, 5.0, 2.0, 2.0, 8.0, 7.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 8.0] };
    let b = scientific::Arr { ndim: 2, shape: [4, 4], data: [0.0, 6.0, 0.0, 0.0, 5.0, 0.0, 2.0, 1.0, 5.0, 2.0, 6.0, 6.0, 4.0, 7.0, 7.0, 7.0] };
    let aa, bb, al, be, q, z, f = scientific::ordqz(a, b, rule);
    assert(f.kind == scientific::F_NONE);
    var g1: f64[>..] = [];
    var g2: f64[>..] = [];
    var i: i64 = 0;
    while i < 4 {
        let e = al.data[2 * i] / be.data[i];
        assert(common::mathshim_abs(al.data[2 * i + 1]) < 1.0e-12);
        if i < first.len { g1.push(e); } else { g2.push(e); }
        i += 1;
    }
    common::qt_sorted(g1);
    common::qt_sorted(g2);
    var fs: f64[>..] = [];
    var rs: f64[>..] = [];
    i = 0;
    while i < first.len { fs.push(first[i]); i += 1; }
    i = 0;
    while i < rest.len { rs.push(rest[i]); i += 1; }
    common::qt_sorted(fs);
    common::qt_sorted(rs);
    common::ct_close(g1, fs, 1.0e-10);
    common::ct_close(g2, rs, 1.0e-10);
}

// qt_finite_all tells whether every value of a buffer is finite.
fn qt_finite_all(d: f64[:]) -> bool {
    var i: i64 = 0;
    while i < d.len {
        if !(d[i] - d[i] == 0.0) { return false; }
        i += 1;
    }
    return true;
}

// qt_crecon gives the largest entry of Q X Z^H - A for complex buffers.
fn qt_crecon(q: f64[:], x: f64[:], z: f64[:], a: f64[:], n: i64) -> f64 {
    var worst: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            var k: i64 = 0;
            while k < n {
                var l: i64 = 0;
                while l < n {
                    let qr = q[2 * (i * n + k)];
                    let qi = q[2 * (i * n + k) + 1];
                    let xr = x[2 * (k * n + l)];
                    let xi = x[2 * (k * n + l) + 1];
                    let zr = z[2 * (j * n + l)];
                    let zi = -z[2 * (j * n + l) + 1];
                    let pr = qr * xr - qi * xi;
                    let pi = qr * xi + qi * xr;
                    sr += pr * zr - pi * zi;
                    si += pr * zi + pi * zr;
                    l += 1;
                }
                k += 1;
            }
            let d1 = common::mathshim_abs(sr - a[2 * (i * n + j)]);
            let d2 = common::mathshim_abs(si - a[2 * (i * n + j) + 1]);
            if d1 > worst { worst = d1; }
            if d2 > worst { worst = d2; }
            j += 1;
        }
        i += 1;
    }
    return worst;
}

// qt_complex_case checks one triangular complex decomposition.
fn qt_complex_case(a: scientific::Arr&, b: scientific::Arr&, n: i64) {
    let aa, bb, q, z, f = scientific::qz_complex(a, b);
    assert(f.kind == scientific::F_NONE);
    var ar: f64[>..] = [];
    var br: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { ar.push(a.data[i]); ar.push(0.0); br.push(b.data[i]); br.push(0.0); i += 1; }
    assert(common_2::qt_crecon(q.data, aa.data, z.data, ar, n) < 1.0e-12);
    assert(common_2::qt_crecon(q.data, bb.data, z.data, br, n) < 1.0e-12);
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < i {
            assert(common::mathshim_abs(aa.data[2 * (i * n + j)]) < 1.0e-12 && common::mathshim_abs(aa.data[2 * (i * n + j) + 1]) < 1.0e-12);
            assert(common::mathshim_abs(bb.data[2 * (i * n + j)]) < 1.0e-12 && common::mathshim_abs(bb.data[2 * (i * n + j) + 1]) < 1.0e-12);
            j += 1;
        }
        assert(bb.data[2 * (i * n + i)] >= 0.0 && bb.data[2 * (i * n + i) + 1] == 0.0);
        i += 1;
    }
}

// qt_cpair gives the data of a random complex pair of side n. The
// values come from a fixed seed, so the pair never changes.
fn qt_cpair(n: i64, seed: i64) -> f64[>..], f64[>..] {
    var a: f64[>..] = [];
    var b: f64[>..] = [];
    var s: i64 = seed;
    var i: i64 = 0;
    while i < n * n {
        s = (s * 1103515245 + 12345) % 2147483648;
        a.push((s as f64) / 1073741824.0 - 1.0);
        s = (s * 1103515245 + 12345) % 2147483648;
        a.push((s as f64) / 536870912.0 - 0.5);
        s = (s * 1103515245 + 12345) % 2147483648;
        b.push((s as f64) / 1073741824.0 - 1.0);
        s = (s * 1103515245 + 12345) % 2147483648;
        b.push((s as f64) / 536870912.0 - 0.5);
        i += 1;
    }
    return a, b;
}

// qt_corth gives the largest entry of F^H F - I for a complex buffer.
fn qt_corth(f: f64[:], n: i64) -> f64 {
    var worst: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            var k: i64 = 0;
            while k < n {
                let ar = f[2 * (i * n + k)];
                let ai = f[2 * (i * n + k) + 1];
                let br = f[2 * (j * n + k)];
                let bi = f[2 * (j * n + k) + 1];
                sr += ar * br + ai * bi;
                si += ar * bi - ai * br;
                k += 1;
            }
            if i == j { sr -= 1.0; }
            let d = common::mathshim_abs(sr) + common::mathshim_abs(si);
            if d > worst { worst = d; }
            j += 1;
        }
        i += 1;
    }
    return worst;
}

// qt_cshape gives the largest entry that breaks the triangular form of
// a complex pair.
fn qt_cshape(aa: f64[:], bb: f64[:], n: i64) -> f64 {
    var worst: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < i {
            let d = common::mathshim_abs(aa[2 * (i * n + j)]) + common::mathshim_abs(aa[2 * (i * n + j) + 1]);
            if d > worst { worst = d; }
            let e = common::mathshim_abs(bb[2 * (i * n + j)]) + common::mathshim_abs(bb[2 * (i * n + j) + 1]);
            if e > worst { worst = e; }
            j += 1;
        }
        i += 1;
    }
    return worst;
}

// qt_csorteigs sorts the eigenvalues of a complex buffer by the real
// part and then by the imaginary part. Each eigenvalue holds the real
// part and the imaginary part.
fn qt_csorteigs(v: f64[>..]&) {
    var i: i64 = 1;
    while i * 2 < v.len {
        let xr = v[2 * i];
        let xi = v[2 * i + 1];
        var j = i - 1;
        while j >= 0 && (v[2 * j] > xr || (v[2 * j] == xr && v[2 * j + 1] > xi)) {
            v[2 * j + 2] = v[2 * j];
            v[2 * j + 3] = v[2 * j + 1];
            j -= 1;
        }
        v[2 * j + 2] = xr;
        v[2 * j + 3] = xi;
        i += 1;
    }
}

// qt_ceigs_close compares two buffers of eigenvalues. Each eigenvalue
// holds the real part and the imaginary part.
fn qt_ceigs_close(got: f64[:], want: f64[:], tol: f64) {
    assert(got.len == want.len);
    var i: i64 = 0;
    while i < got.len {
        assert(common::mathshim_abs(got[i] - want[i]) < tol);
        i += 1;
    }
}

// qt_pinned_pair gives the pinned complex pair. Its values come from
// the reference library.
fn qt_pinned_pair() -> f64[>..], f64[>..] {
    var a: f64[>..] = [2.0, 1.0, 5.0, -1.0, 8.0, 0.0, 7.0, 2.0,
                       5.0, 0.0, 2.0, 3.0, 2.0, 0.0, 8.0, -1.0,
                       7.0, -2.0, 5.0, 0.0, 6.0, 1.0, 6.0, 0.0,
                       5.0, 0.0, 4.0, 1.0, 4.0, 0.0, 8.0, -2.0];
    var b: f64[>..] = [0.0, 1.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       5.0, 0.0, 1.0, 0.0, 2.0, -1.0, 1.0, 0.0,
                       5.0, 1.0, 2.0, 0.0, 6.0, 0.0, 6.0, -1.0,
                       4.0, 0.0, 7.0, 0.0, 0.0, 7.0, 7.0, 0.0];
    return a, b;
}

// qt_rule_name gives the name of a rule by its place.
fn qt_rule_name(r: i64) -> const u8[:] {
    if r == 0 { return "lhp"; }
    if r == 1 { return "rhp"; }
    if r == 2 { return "iuc"; }
    return "ouc";
}

// qt_rule_count gives the count of the first group of a rule on the
// pinned pair.
fn qt_rule_count(r: i64) -> i64 {
    if r == 0 { return 1; }
    if r == 1 { return 3; }
    if r == 2 { return 2; }
    return 2;
}

// qt_eigs_close compares two buffers of eigenvalues. Each eigenvalue
// holds the real part and the imaginary part.
fn qt_eigs_close(got: scientific::CArr&, want: scientific::CArr&, be: scientific::Arr&, wob: scientific::Arr&, tol: f64) {
    assert(got.data.len == want.data.len);
    assert(be.data.len == wob.data.len);
    var i: i64 = 0;
    while i < be.data.len {
        let gr = got.data[2 * i] / be.data[i];
        let gi = got.data[2 * i + 1] / be.data[i];
        let wr = want.data[2 * i] / wob.data[i];
        let wi = want.data[2 * i + 1] / wob.data[i];
        assert(common::mathshim_abs(gr - wr) < tol && common::mathshim_abs(gi - wi) < tol);
        i += 1;
    }
}

// qt_maxabs gives the largest absolute value of a buffer.
fn qt_maxabs(d: f64[:]) -> f64 {
    var m: f64 = 0.0;
    var i: i64 = 0;
    while i < d.len {
        if common::mathshim_abs(d[i]) > m { m = common::mathshim_abs(d[i]); }
        i += 1;
    }
    return m;
}

// qt_pinned_scaled gives the pinned pair with each matrix times a factor.
fn qt_pinned_scaled(fa: f64, fb: f64) -> scientific::Arr, scientific::Arr {
    var ad: f64[>..] = [2.0, 5.0, 8.0, 7.0, 5.0, 2.0, 2.0, 8.0, 7.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 8.0];
    var bd: f64[>..] = [0.0, 6.0, 0.0, 0.0, 5.0, 0.0, 2.0, 1.0, 5.0, 2.0, 6.0, 6.0, 4.0, 7.0, 7.0, 7.0];
    var i: i64 = 0;
    while i < 16 { ad[i] = ad[i] * fa; bd[i] = bd[i] * fb; i += 1; }
    return scientific::Arr { ndim: 2, shape: [4, 4], data: copy(ad) }, scientific::Arr { ndim: 2, shape: [4, 4], data: copy(bd) };
}

// qt_scaled_ok checks that the eigenvalues alpha / beta of the pinned pair
// equal the pinned values times fac, as a set, to a relative 1.0e-9.
fn qt_scaled_ok(al: f64[:], be: f64[:], fac: f64) {
    var got: f64[>..] = [];
    var i: i64 = 0;
    while i < 4 { assert(common::mathshim_abs(al[2 * i + 1]) < 1.0e-9 * common::mathshim_abs(al[2 * i])); got.push(al[2 * i] / be[i]); i += 1; }
    common::qt_sorted(got);
    let want = [-15.928848409502654, -0.6126906030083017, 0.19514833489844835, 1.543360374582167];
    i = 0;
    while i < 4 {
        assert(common::mathshim_abs(got[i] - want[i] * fac) <= 1.0e-9 * common::mathshim_abs(want[i] * fac));
        i += 1;
    }
}

// qt_scaled_case decomposes the pinned pair with each matrix times a
// factor. It runs qz, the four rules of ordqz, qz_complex and
// ordqz_complex. The products hold to 1.0e-12 of the largest entry of
// each matrix. The eigenvalues equal the pinned values times fac, when
// check is true.
fn qt_scaled_case(fa: f64, fb: f64, fac: f64, check: bool) {
    let a, b = common_2::qt_pinned_scaled(fa, fb);
    let ma = common_2::qt_maxabs(a.data);
    let mb = common_2::qt_maxabs(b.data);
    let aa, bb, q, z, f = scientific::qz(a, b);
    assert(f.kind == scientific::F_NONE);
    assert(common::qt_recon(q.data, aa.data, z.data, a.data, 0, 4) <= 1.0e-12 * ma);
    assert(common::qt_recon(q.data, bb.data, z.data, b.data, 0, 4) <= 1.0e-12 * mb);
    assert(common::qt_orth(q.data, 0, 4) < 1.0e-12 && common::qt_orth(z.data, 0, 4) < 1.0e-12);
    let rules = ["lhp", "rhp", "iuc", "ouc"];
    var r: i64 = 0;
    while r < 4 {
        let xa, xb, al, be, xq, xz, xf = scientific::ordqz(a, b, rules[r]);
        assert(xf.kind == scientific::F_NONE);
        assert(common::qt_recon(xq.data, xa.data, xz.data, a.data, 0, 4) <= 1.0e-12 * ma);
        assert(common::qt_recon(xq.data, xb.data, xz.data, b.data, 0, 4) <= 1.0e-12 * mb);
        assert(common_2::qt_finite_all(al.data) && common_2::qt_finite_all(be.data));
        if check { common_2::qt_scaled_ok(al.data, be.data, fac); }
        if r < 2 {
            // The sign rules hold at every scale.
            var seen = false;
            var i: i64 = 0;
            while i < 4 {
                let neg = al.data[2 * i] < 0.0;
                let chosen = (r == 0 && neg) || (r == 1 && !neg);
                if chosen { assert(!seen); } else { seen = true; }
                i += 1;
            }
        }
        r += 1;
    }
    let ca, cb, cq, cz, cf = scientific::qz_complex(a, b);
    assert(cf.kind == scientific::F_NONE);
    var ar: f64[>..] = [];
    var br: f64[>..] = [];
    var i2: i64 = 0;
    while i2 < 16 { ar.push(a.data[i2]); ar.push(0.0); br.push(b.data[i2]); br.push(0.0); i2 += 1; }
    assert(common_2::qt_crecon(cq.data, ca.data, cz.data, ar, 4) <= 1.0e-12 * ma);
    assert(common_2::qt_crecon(cq.data, cb.data, cz.data, br, 4) <= 1.0e-12 * mb);
    let oa, ob, oal, obe, oq, oz, of = scientific::ordqz_complex(a, b, "lhp");
    assert(of.kind == scientific::F_NONE);
    assert(common_2::qt_crecon(oq.data, oa.data, oz.data, ar, 4) <= 1.0e-12 * ma);
    assert(common_2::qt_finite_all(oal.data) && common_2::qt_finite_all(obe.data));
}

// qt_pad_mode gives the name of a mode of the pad tests.
fn qt_pad_mode(i: i64) -> const u8[:] {
    if i == 0 { return "constant"; }
    if i == 1 { return "linear_ramp"; }
    if i == 2 { return "maximum"; }
    if i == 3 { return "minimum"; }
    if i == 4 { return "mean"; }
    if i == 5 { return "median"; }
    return "edge";
}

// qt_pad_edge_value gives the value that a statistic mode takes from
// the first n values of a line (the start) or from the last n values.
// The test sorts a copy and reads the middle, so it shares no code with
// the package.
fn qt_pad_edge_value(line: f64[:], n: i64, atstart: bool, mode: const u8[:]) -> f64 {
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        if atstart { v.push(line[i]); } else { v.push(line[line.len - n + i]); }
        i += 1;
    }
    if mode == "mean" {
        var t: f64 = 0.0;
        i = 0;
        while i < n { t += v[i]; i += 1; }
        return t / n;
    }
    var a = 0;
    while a < n {
        var b = a + 1;
        while b < n {
            if v[b] < v[a] { let t = v[a]; v[a] = v[b]; v[b] = t; }
            b += 1;
        }
        a += 1;
    }
    if mode == "maximum" { return v[n - 1]; }
    if mode == "minimum" { return v[0]; }
    if n % 2 == 1 { return v[n / 2]; }
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// qt_pad_line pads one line by the rules of NumPy, written out by
// position.
fn qt_pad_line(line: f64[:], before: i64, after: i64, mode: const u8[:], lb: i64, la: i64, eb: f64, ea: f64, cb: f64, ca: f64) -> f64[>..] {
    var out: f64[>..] = [];
    let w = line.len;
    var k: i64 = 0;
    while k < before {
        // The place k counts from the outer edge.
        if mode == "constant" { out.push(cb); }
        else if mode == "empty" { out.push(0.0); }
        else if mode == "edge" { out.push(line[0]); }
        else if mode == "linear_ramp" { out.push(eb + (line[0] - eb) * k / before); }
        else {
            var n = lb;
            if n < 1 || n > w { n = w; }
            out.push(common_2::qt_pad_edge_value(line, n, true, mode));
        }
        k += 1;
    }
    var i: i64 = 0;
    while i < w { out.push(line[i]); i += 1; }
    k = 0;
    while k < after {
        // The place k counts from the line outward. The ramp holds its
        // values from the end value inward, so the place k reads the
        // ramp from the other end.
        if mode == "constant" { out.push(ca); }
        else if mode == "empty" { out.push(0.0); }
        else if mode == "edge" { out.push(line[w - 1]); }
        else if mode == "linear_ramp" { out.push(ea + (line[w - 1] - ea) * (after - 1 - k) / after); }
        else {
            var n = la;
            if n < 1 || n > w { n = w; }
            out.push(common_2::qt_pad_edge_value(line, n, false, mode));
        }
        k += 1;
    }
    return out;
}

// qt_pad2 pads the matrix x of r rows and c columns. The axis 0 comes
// first, then the axis 1. The lists hold one pair for each axis.
fn qt_pad2(x: f64[:], r: i64, c: i64, b: i64[:], a: i64[:], mode: const u8[:], sl: i64[:], ev: f64[:], cv: f64[:]) -> f64[>..] {
    let r2 = r + b[0] + a[0];
    var mid: f64[>..] = [];
    var i: i64 = 0;
    while i < r2 * c { mid.push(0.0); i += 1; }
    var j: i64 = 0;
    while j < c {
        var col: f64[>..] = [];
        i = 0;
        while i < r { col.push(x[i * c + j]); i += 1; }
        let p = common_2::qt_pad_line(col, b[0], a[0], mode, sl[0], sl[1], ev[0], ev[1], cv[0], cv[1]);
        i = 0;
        while i < r2 { mid[i * c + j] = p[i]; i += 1; }
        j += 1;
    }
    var out: f64[>..] = [];
    i = 0;
    while i < r2 {
        var row: f64[>..] = [];
        j = 0;
        while j < c { row.push(mid[i * c + j]); j += 1; }
        let p = common_2::qt_pad_line(row, b[1], a[1], mode, sl[2], sl[3], ev[2], ev[3], cv[2], cv[3]);
        j = 0;
        while j < p.len { out.push(p[j]); j += 1; }
        i += 1;
    }
    return out;
}

// qt_sketch_rebuild builds the sketch of x (n rows, d columns) with a
// second generator of the same seed. It draws all hashes, then all
// signs, as the call documents.
fn qt_sketch_rebuild(x: f64[:], n: i64, d: i64, s: i64, seed: i64) -> f64[>..] {
    var g = scientific::default_rng(seed);
    var h: i64[>..] = [];
    var i: i64 = 0;
    while i < n {
        let v, f = scientific::integers(g, 0, s);
        h.push(v);
        i += 1;
    }
    var out: f64[>..] = [];
    i = 0;
    while i < s * d { out.push(0.0); i += 1; }
    i = 0;
    while i < n {
        let b, f = scientific::integers(g, 0, 2);
        var sg: f64 = -1.0;
        if b == 1 { sg = 1.0; }
        var j: i64 = 0;
        while j < d { out[h[i] * d + j] += sg * x[i * d + j]; j += 1; }
        i += 1;
    }
    return out;
}

// qt_norm gives the Euclidean norm of a buffer.
fn qt_norm(v: f64[:]) -> f64 {
    var t: f64 = 0.0;
    var i: i64 = 0;
    while i < v.len { t += v[i] * v[i]; i += 1; }
    return common::mathshim_sqrt(t);
}

// qt_mm gives the product of a matrix of ar rows and ac columns with a
// matrix of ac rows and bc columns.
fn qt_mm(a: f64[:], ar: i64, ac: i64, b: f64[:], bc: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < ar {
        var j: i64 = 0;
        while j < bc {
            var t: f64 = 0.0;
            var k: i64 = 0;
            while k < ac { t += a[i * ac + k] * b[k * bc + j]; k += 1; }
            out.push(t);
            j += 1;
        }
        i += 1;
    }
    return out;
}

// qt_tr gives the transpose of a matrix of r rows and c columns.
fn qt_tr(a: f64[:], r: i64, c: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var j: i64 = 0;
    while j < c {
        var i: i64 = 0;
        while i < r { out.push(a[i * c + j]); i += 1; }
        j += 1;
    }
    return out;
}

// qt_inv gives the inverse of a square matrix by Gauss-Jordan.
fn qt_inv(a: f64[:], n: i64) -> f64[>..] {
    var w: f64[>..] = [];
    var inv: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { w.push(a[i]); inv.push(0.0); i += 1; }
    i = 0;
    while i < n { inv[i * n + i] = 1.0; i += 1; }
    var c: i64 = 0;
    while c < n {
        var p = c;
        var r: i64 = c + 1;
        while r < n {
            if common::mathshim_abs(w[r * n + c]) > common::mathshim_abs(w[p * n + c]) { p = r; }
            r += 1;
        }
        var k: i64 = 0;
        while k < n {
            let t1 = w[c * n + k]; w[c * n + k] = w[p * n + k]; w[p * n + k] = t1;
            let t2 = inv[c * n + k]; inv[c * n + k] = inv[p * n + k]; inv[p * n + k] = t2;
            k += 1;
        }
        let d = w[c * n + c];
        k = 0;
        while k < n { w[c * n + k] = w[c * n + k] / d; inv[c * n + k] = inv[c * n + k] / d; k += 1; }
        r = 0;
        while r < n {
            if r != c {
                let f = w[r * n + c];
                k = 0;
                while k < n {
                    w[r * n + k] -= f * w[c * n + k];
                    inv[r * n + k] -= f * inv[c * n + k];
                    k += 1;
                }
            }
            r += 1;
        }
        c += 1;
    }
    return inv;
}

// qt_dare_resid gives the largest absolute residual of the discrete
// equation at x, relative to the largest term. The equation is
// a^T x a - e^T x e - (a^T x b + s) (r + b^T x b)^-1 (b^T x a + s^T) + q.
// The matrices e and s are the identity and zero when they hold no value.
fn qt_dare_resid(a: f64[:], b: f64[:], q: f64[:], r: f64[:], x: f64[:], e: f64[:], s: f64[:], n: i64, m: i64) -> f64 {
    let at = common_2::qt_tr(a, n, n);
    let xa = common_2::qt_mm(x, n, n, a, n);
    let t1 = common_2::qt_mm(at, n, n, xa, n);
    var t2: f64[>..] = [];
    if e.len > 0 { t2 = common_2::qt_mm(common_2::qt_tr(e, n, n), n, n, common_2::qt_mm(x, n, n, e, n), n); } else { var k: i64 = 0; while k < n * n { t2.push(x[k]); k += 1; } }
    var cross = common_2::qt_mm(at, n, n, common_2::qt_mm(x, n, n, b, m), m);
    if s.len > 0 { var i: i64 = 0; while i < n * m { cross[i] += s[i]; i += 1; } }
    var mid = common_2::qt_mm(common_2::qt_tr(b, n, m), m, n, common_2::qt_mm(x, n, n, b, m), m);
    var i: i64 = 0;
    while i < m * m { mid[i] += r[i]; i += 1; }
    let t3 = common_2::qt_mm(common_2::qt_mm(cross, n, m, common_2::qt_inv(mid, m), m), n, m, common_2::qt_tr(cross, n, m), n);
    var worst: f64 = 0.0;
    var scale: f64 = 1.0;
    i = 0;
    while i < n * n {
        var xe: f64 = x[i];
        if e.len > 0 { xe = t2[i]; }
        let v = t1[i] - xe - t3[i] + q[i];
        if common::mathshim_abs(v) > worst { worst = common::mathshim_abs(v); }
        if common::mathshim_abs(t1[i]) > scale { scale = common::mathshim_abs(t1[i]); }
        if common::mathshim_abs(t3[i]) > scale { scale = common::mathshim_abs(t3[i]); }
        i += 1;
    }
    return worst / scale;
}

// all_close_c tells whether every element of a complex array agrees
// with the values of the two lists, relative to one plus the value.
fn all_close_c(a: scientific::CArr&, vr: f64[:], vi: f64[:], tol: f64) -> bool {
    var i: i64 = 0;
    while i < vr.len {
        if !common::close(a.data[2 * i], vr[i], tol) { return false; }
        if !common::close(a.data[2 * i + 1], vi[i], tol) { return false; }
        i += 1;
    }
    return true;
}

// cmat_err gives the largest difference between two complex arrays.
fn cmat_err(x: scientific::CArr&, y: scientific::CArr&) -> f64 {
    var worst: f64 = 0.0;
    var i: i64 = 0;
    while i < x.data.len {
        var d = x.data[i] - y.data[i];
        if d < 0.0 { d = -d; }
        if d > worst { worst = d; }
        i += 1;
    }
    return worst;
}

// gram_dev gives the largest deviation of the inner products of the
// first count vectors from the unit matrix. The vectors are the columns
// when by_cols holds, else the rows.
fn gram_dev(q: scientific::CArr&, by_cols: bool, count: i64) -> f64 {
    let rows = q.shape[0];
    let cols = q.shape[1];
    var len = cols;
    if by_cols { len = rows; }
    var worst: f64 = 0.0;
    var a: i64 = 0;
    while a < count {
        var b: i64 = 0;
        while b < count {
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            var k: i64 = 0;
            while k < len {
                var ia = a * cols + k;
                var ib = b * cols + k;
                if by_cols {
                    ia = k * cols + a;
                    ib = k * cols + b;
                }
                // The inner product of the vector a with the conjugate of b.
                sr += q.data[2 * ia] * q.data[2 * ib] + q.data[2 * ia + 1] * q.data[2 * ib + 1];
                si += q.data[2 * ia + 1] * q.data[2 * ib] - q.data[2 * ia] * q.data[2 * ib + 1];
                k += 1;
            }
            var want: f64 = 0.0;
            if a == b { want = 1.0; }
            var dr = sr - want;
            if dr < 0.0 { dr = -dr; }
            var di = si;
            if di < 0.0 { di = -di; }
            if dr > worst { worst = dr; }
            if di > worst { worst = di; }
            b += 1;
        }
        a += 1;
    }
    return worst;
}

// check_qr tests the properties of the two factors of a complex matrix.
fn check_qr(a: scientific::CArr&) {
    let m = a.shape[0];
    let n = a.shape[1];
    var p = m;
    if n < p { p = n; }
    let q, r, f = scientific::cqr(a);
    assert(f.kind == scientific::F_NONE);
    assert(q.shape[0] == m && q.shape[1] == p);
    assert(r.shape[0] == p && r.shape[1] == n);
    let qr, pf = scientific::cdot(q, r);
    assert(pf.kind == scientific::F_NONE);
    assert(common_2::cmat_err(qr, a) < 1.0e-12);
    assert(common_2::gram_dev(q, true, p) < 1.0e-12);
    // The upper triangle holds zero below the diagonal and a real diagonal.
    var i: i64 = 0;
    while i < p {
        var j: i64 = 0;
        while j < n {
            if j <= i {
                if j < i {
                    assert(r.data[2 * (i * n + j)] == 0.0 && r.data[2 * (i * n + j) + 1] == 0.0);
                } else {
                    assert(r.data[2 * (i * n + j) + 1] == 0.0);
                }
            }
            j += 1;
        }
        i += 1;
    }
    return;
}

// check_svd tests the properties of the three results of a complex matrix.
fn check_svd(a: scientific::CArr&) {
    let m = a.shape[0];
    let n = a.shape[1];
    var k = m;
    if n < k { k = n; }
    let u, s, vh, f = scientific::csvd(a);
    assert(f.kind == scientific::F_NONE);
    assert(u.shape[0] == m && u.shape[1] == m);
    assert(vh.shape[0] == n && vh.shape[1] == n);
    assert(s.data.len == k);
    var i: i64 = 0;
    while i < k {
        assert(s.data[i] >= 0.0);
        if i > 0 { assert(s.data[i - 1] >= s.data[i]); }
        i += 1;
    }
    assert(common_2::gram_dev(u, true, m) < 1.0e-12);
    assert(common_2::gram_dev(vh, false, n) < 1.0e-12);
    // The matrix equals the first k columns of U times S times the first k rows of Vh.
    var worst: f64 = 0.0;
    i = 0;
    while i < m {
        var j: i64 = 0;
        while j < n {
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            var t: i64 = 0;
            while t < k {
                let ur = u.data[2 * (i * m + t)] * s.data[t];
                let ui = u.data[2 * (i * m + t) + 1] * s.data[t];
                let vr = vh.data[2 * (t * n + j)];
                let vi = vh.data[2 * (t * n + j) + 1];
                sr += ur * vr - ui * vi;
                si += ur * vi + ui * vr;
                t += 1;
            }
            var dr = sr - a.data[2 * (i * n + j)];
            var di = si - a.data[2 * (i * n + j) + 1];
            if dr < 0.0 { dr = -dr; }
            if di < 0.0 { di = -di; }
            if dr > worst { worst = dr; }
            if di > worst { worst = di; }
            j += 1;
        }
        i += 1;
    }
    assert(worst < 1.0e-12);
    // The largest component of each column of U is real and above zero.
    var c: i64 = 0;
    while c < k {
        if s.data[c] > 1.0e-6 {
            var best: f64 = -1.0;
            var br: f64 = 0.0;
            var bi: f64 = 0.0;
            var r: i64 = 0;
            while r < m {
                let xr = u.data[2 * (r * m + c)];
                let xi = u.data[2 * (r * m + c) + 1];
                if xr * xr + xi * xi > best + 1.0e-9 {
                    best = xr * xr + xi * xi;
                    br = xr;
                    bi = xi;
                }
                r += 1;
            }
            assert(br > 0.0);
            var abi = bi;
            if abi < 0.0 { abi = -abi; }
            assert(abi < 1.0e-12);
        }
        c += 1;
    }
    return;
}

// random_complex gives a matrix of normal complex values.
fn random_complex(rng: scientific::Rng&, m: i64, n: i64) -> scientific::CArr {
    let re, f0 = scientific::randn(rng, [m, n]);
    let im, f1 = scientific::randn(rng, [m, n]);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);
    let a, f2 = scientific::complex_array(re, im);
    assert(f2.kind == scientific::F_NONE);
    return a;
}

// sort_eig orders the eigenvalues of a general matrix by the real part and then
// by the imaginary part, as the generator of the pinned values does. The
// vectors follow. Real parts that differ by less than 1e-9 count as equal.
fn sort_eig(w: scientific::CArr&, v: scientific::CArr&, n: i64, ws: f64[>..]&, vs: f64[>..]&) {
    var order: i64[>..] = [];
    var top: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        order.push(i);
        var mr = w.data[2 * i];
        if mr < 0.0 { mr = -mr; }
        var mi = w.data[2 * i + 1];
        if mi < 0.0 { mi = -mi; }
        if mr > top { top = mr; }
        if mi > top { top = mi; }
        i += 1;
    }
    // The tolerance is relative to the largest eigenvalue.
    let tol = 1.0e-9 * top;
    i = 1;
    while i < n {
        let key = order[i];
        var j = i - 1;
        while j >= 0 {
            let dr = w.data[2 * order[j]] - w.data[2 * key];
            var before = false;
            if dr > tol { before = true; }
            if dr <= tol && dr >= -tol && w.data[2 * order[j] + 1] > w.data[2 * key + 1] { before = true; }
            if before {
                order[j + 1] = order[j];
                j -= 1;
            } else { break; }
        }
        order[j + 1] = key;
        i += 1;
    }
    i = 0;
    while i < n {
        ws.push(w.data[2 * order[i]]);
        ws.push(w.data[2 * order[i] + 1]);
        i += 1;
    }
    var r: i64 = 0;
    while r < n {
        var c: i64 = 0;
        while c < n {
            vs.push(v.data[2 * (r * n + order[c])]);
            vs.push(v.data[2 * (r * n + order[c]) + 1]);
            c += 1;
        }
        r += 1;
    }
    return;
}
