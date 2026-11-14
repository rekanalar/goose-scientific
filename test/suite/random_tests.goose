// The tests of the group random. Run them with SCIENTIFIC_SUITES=random.
namespace random_tests;

import scientific from "../..";
import os;
import common;
import common_2;

fn test_random() {
    // The numbers of one seed are pinned. The same seed gives the
    // same numbers on every run.
    var r = scientific::default_rng(42);
    assert(scientific::uniform(r, 2.0, 5.0) == 4.22469463631547);
    assert(scientific::random_uniform(r) == 0.1599103928769201);
    let a, fa = scientific::integers(r, 1, 7);
    assert(fa.kind == scientific::F_NONE && a == 1);
    // The pair of Box and Muller gives one number and holds one.
    let n1, f1 = scientific::normal(r, 0.0, 1.0);
    let n2, f2 = scientific::normal(r, 0.0, 1.0);
    assert(f1.kind == scientific::F_NONE && f2.kind == scientific::F_NONE);
    assert(n1 == 0.892470392690831 && n2 == 0.21741001724982148);
    // A spread below zero is a fault.
    let n3, f3 = scientific::normal(r, 0.0, -1.0);
    assert(f3.kind == scientific::F_VALUE);
    // The exponential law takes the inverse of the law.
    let e, f4 = scientific::exponential(r, 2.0);
    assert(f4.kind == scientific::F_NONE && e == 4.053365406960153);
    let e2, f5 = scientific::exponential(r, -1.0);
    assert(f5.kind == scientific::F_VALUE);
    // The method of Knuth holds a rate of 30 and below.
    let p, f6 = scientific::poisson(r, 5.0);
    assert(f6.kind == scientific::F_NONE && p == 5);
    let p0, f7 = scientific::poisson(r, 0.0);
    assert(f7.kind == scientific::F_NONE && p0 == 0);
    let pbig, f8 = scientific::poisson(r, 31.0);
    assert(f8.kind == scientific::F_VALUE);
    // The order of a count holds every whole number of the count.
    let pc, f9 = scientific::permutation(r, 5);
    assert(f9.kind == scientific::F_NONE);
    assert(pc.data[0] == 1.0 && pc.data[1] == 0.0 && pc.data[2] == 2.0);
    assert(pc.data[3] == 4.0 && pc.data[4] == 3.0);
    // A range that holds no number is a fault.
    let a2, f10 = scientific::integers(r, 5, 5);
    assert(f10.kind == scientific::F_VALUE);
    // The picks of an array.
    var r2 = scientific::default_rng(7);
    let src = scientific::from_values([10.0, 20.0, 30.0, 40.0]);
    let c1, fc1 = scientific::choice(r2, src, 3, true);
    assert(fc1.kind == scientific::F_NONE);
    assert(c1.data[0] == 40.0 && c1.data[1] == 10.0 && c1.data[2] == 30.0);
    // A pick without a repeat holds no element twice.
    let c2, fc2 = scientific::choice(r2, src, 3, false);
    assert(fc2.kind == scientific::F_NONE);
    assert(c2.data[0] == 10.0 && c2.data[1] == 30.0 && c2.data[2] == 20.0);
    let c3, fc3 = scientific::choice(r2, src, 5, false);
    assert(fc3.kind == scientific::F_VALUE);
    // A shuffled copy leaves the array of the caller as it was.
    var r3 = scientific::default_rng(9);
    let src2 = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0]);
    let sh, fsh = scientific::shuffled(r3, src2);
    assert(fsh.kind == scientific::F_NONE);
    assert(sh.data[0] == 2.0 && sh.data[1] == 5.0 && sh.data[2] == 1.0);
    assert(sh.data[3] == 3.0 && sh.data[4] == 4.0);
    var same = true;
    var i: i64 = 0;
    while i < 5 {
        if src2.data[i] != i + 1 as f64 { same = false; }
        i += 1;
    }
    assert(same);
    // The rows of a matrix move as a whole.
    var r4 = scientific::default_rng(7);
    let m = scientific::from_rows(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let fr = scientific::shuffle_rows(r4, m);
    assert(fr.kind == scientific::F_NONE);
    assert(m.data[0] == 3.0 && m.data[1] == 4.0 && m.data[2] == 5.0);
    assert(m.data[3] == 6.0 && m.data[4] == 1.0 && m.data[5] == 2.0);
    // The same seed gives the same numbers.
    var ra = scientific::default_rng(11);
    var rb = scientific::default_rng(11);
    var i2: i64 = 0;
    while i2 < 4 {
        let ua = scientific::uniform(ra, 0.0, 1.0);
        let ub = scientific::uniform(rb, 0.0, 1.0);
        assert(ua == ub);
        i2 += 1;
    }
}

fn test_generators() {
    // The names of NumPy call the generator of the package. A name
    // gives the same numbers as the function under it.
    let r = scientific::seed(42);
    let r2 = scientific::default_rng(42);
    assert(scientific::random_sample(r) == scientific::random_uniform(r2));
    assert(scientific::randint(r, 1, 10) == scientific::integers(r2, 1, 10));
    let r3 = scientific::seed(7);
    let r4 = scientific::seed(7);
    var d: f64[>..] = [3.0, 1.0, 4.0, 1.5];
    var q = scientific::Arr { ndim: 1, shape: [4], data: copy(d) };
    let f = scientific::shuffle(r3, q);
    assert(f.kind == scientific::F_NONE);
    var d2: f64[>..] = [3.0, 1.0, 4.0, 1.5];
    scientific::mix(d2, r4.g);
    var i: i64 = 0;
    while i < 4 { assert(q.data[i] == d2[i]); i += 1; }
    // The shuffled copy of the plain values holds the same order.
    var d3: f64[>..] = [3.0, 1.0, 4.0, 1.5];
    let r5 = scientific::seed(7);
    let plain = scientific::Arr { ndim: 1, shape: [4], data: copy(d3) };
    let c, fc = scientific::shuffled(r5, plain);
    assert(fc.kind == scientific::F_NONE);
    i = 0;
    while i < 4 { assert(c.data[i] == d2[i]); i += 1; }
    let r6 = scientific::seed(1);
    let u, f2 = scientific::rand(r6, [2, 3]);
    assert(f2.kind == scientific::F_NONE && u.ndim == 2 && u.shape[0] == 2 && u.shape[1] == 3);
    assert(u.data[0] >= 0.0 && u.data[0] < 1.0 && u.data[5] >= 0.0 && u.data[5] < 1.0);
    let r7 = scientific::seed(1);
    let u2, f3 = scientific::rand(r7, [2, 3]);
    i = 0;
    while i < 6 { assert(u.data[i] == u2.data[i]); i += 1; }
    let n, f4 = scientific::randn(r6, [2, 3]);
    assert(f4.kind == scientific::F_NONE && n.data.len == 6);
    // The state round trips through the word.
    let r8 = scientific::seed(5);
    var out: u8[>..] = [];
    let g = scientific::bytes(r8, 4, out);
    assert(g.kind == scientific::F_NONE && out.len == 4);
    let st = scientific::get_state(r8);
    let r9 = scientific::set_state(st);
    assert(scientific::random_uniform(r9) == scientific::random_uniform(r8));
    // The faults.
    var bad: u8[>..] = [];
    let r10 = scientific::seed(3);
    let g2 = scientific::bytes(r10, 0, bad);
    assert(g2.kind == scientific::F_RANGE);
    let r11 = scientific::seed(3);
    let e1, g3 = scientific::rand(r11, []);
    assert(g3.kind == scientific::F_VALUE);
    let e2, g4 = scientific::randn(r11, [2, 0]);
    assert(g4.kind == scientific::F_VALUE);
    var m: f64[>..] = [1.0, 2.0, 3.0, 4.0];
    let q2 = scientific::Arr { ndim: 2, shape: [2, 2], data: copy(m) };
    let r12 = scientific::seed(3);
    let g5 = scientific::shuffle(r12, q2);
    assert(g5.kind == scientific::F_SHAPE);
    let badint = scientific::randint(r11, 5, 5);
}

// test_pad_empty checks that the mode empty gives zero in the pad.
fn test_pad_empty() {
    var g = scientific::default_rng(94);
    var t: i64 = 0;
    while t < 20 {
        let r, f1 = scientific::integers(g, 1, 4);
        let c, f2 = scientific::integers(g, 1, 5);
        var d: f64[>..] = [];
        var i: i64 = 0;
        while i < r * c { d.push(1.0 + scientific::random_uniform(g)); i += 1; }
        let x = common::mx(r, c, d);
        let b0, f3 = scientific::integers(g, 0, 3);
        let a0, f4 = scientific::integers(g, 0, 3);
        let b1, f5 = scientific::integers(g, 0, 3);
        let a1, f6 = scientific::integers(g, 0, 3);
        let p, f = scientific::pad_modes(x, [b0, b1], [a0, a1], "empty");
        assert(f.kind == scientific::F_NONE);
        assert(p.shape[0] == r + b0 + a0 && p.shape[1] == c + b1 + a1);
        let w = c + b1 + a1;
        var ok = true;
        var y: i64 = 0;
        while y < p.shape[0] {
            var z: i64 = 0;
            while z < w {
                let inside = y >= b0 && y < b0 + r && z >= b1 && z < b1 + c;
                if inside { if p.data[y * w + z] != x.data[(y - b0) * c + (z - b1)] { ok = false; } }
                else if p.data[y * w + z] != 0.0 { ok = false; }
                z += 1;
            }
            y += 1;
        }
        assert(ok);
        t += 1;
    }
    return;
}

// test_pad_axes pins the lists of values for each axis against NumPy and
// compares random cases with the line by line construction.
fn test_pad_axes() {
    let a = common::mx(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let sl = [1, 2, 2, 3];
    let r1, f1 = scientific::pad_axes(a, [1, 2], [1, 1], "maximum", sl);
    assert(f1.kind == scientific::F_NONE && r1.shape[0] == 4 && r1.shape[1] == 6);
    assert(common::all_near(r1, [2.0, 2.0, 1.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.0, 2.0, 3.0, 3.0, 5.0, 5.0, 4.0, 5.0, 6.0, 6.0, 5.0, 5.0, 4.0, 5.0, 6.0, 6.0]));
    let r2, f2 = scientific::pad_axes(a, [1, 2], [1, 1], "mean", sl);
    assert(f2.kind == scientific::F_NONE);
    assert(common::all_near(r2, [1.5, 1.5, 1.0, 2.0, 3.0, 2.0, 1.5, 1.5, 1.0, 2.0, 3.0, 2.0, 4.5, 4.5, 4.0, 5.0, 6.0, 5.0, 3.0, 3.0, 2.5, 3.5, 4.5, 3.5]));
    let r3, f3 = scientific::pad_axes(a, [1, 2], [1, 1], "minimum", sl);
    assert(f3.kind == scientific::F_NONE);
    assert(common::all_near(r3, [1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 4.0, 4.0, 4.0, 5.0, 6.0, 4.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0]));
    let r4, f4 = scientific::pad_axes(a, [1, 2], [1, 1], "median", sl);
    assert(f4.kind == scientific::F_NONE);
    assert(common::all_near(r4, [1.5, 1.5, 1.0, 2.0, 3.0, 2.0, 1.5, 1.5, 1.0, 2.0, 3.0, 2.0, 4.5, 4.5, 4.0, 5.0, 6.0, 5.0, 3.0, 3.0, 2.5, 3.5, 4.5, 3.5]));
    let r5, f5 = scientific::pad_axes(a, [1, 2], [1, 1], "linear_ramp", [], [5.0, 6.0, 7.0, 8.0]);
    assert(f5.kind == scientific::F_NONE);
    assert(common::all_near(r5, [7.0, 6.0, 5.0, 5.0, 5.0, 8.0, 7.0, 4.0, 1.0, 2.0, 3.0, 8.0, 7.0, 5.5, 4.0, 5.0, 6.0, 8.0, 7.0, 6.5, 6.0, 6.0, 6.0, 8.0]));
    let r6, f6 = scientific::pad_axes(a, [1, 1], [1, 2], "constant", [], [], [1.0, 2.0, 3.0, 4.0]);
    assert(f6.kind == scientific::F_NONE);
    assert(common::all_near(r6, [3.0, 1.0, 1.0, 1.0, 4.0, 4.0, 3.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 4.0, 5.0, 6.0, 4.0, 4.0, 3.0, 2.0, 2.0, 2.0, 4.0, 4.0]));
    // A constant of NaN passes into the pad. An end value of infinity gives
    // NaN in the ramp, as NumPy does.
    let r7, f7 = scientific::pad_axes(a, [1, 1], [1, 1], "constant", [], [], [0.0 / 0.0]);
    assert(f7.kind == scientific::F_NONE && r7.data[0] != r7.data[0] && r7.data[5] != r7.data[5] && r7.data[6] == 1.0);
    let inf = 1.0 / 0.0;
    let r8, f8 = scientific::pad_axes(a, [1, 0], [0, 0], "linear_ramp", [], [inf]);
    assert(f8.kind == scientific::F_NONE && r8.shape[0] == 3 && r8.data[0] != r8.data[0]);

    // The one value and the pair forms equal the call of pad_modes.
    let r9, f9 = scientific::pad_axes(a, [1, 2], [2, 1], "linear_ramp", [], [4.0]);
    let r10, f10 = scientific::pad_modes(a, [1, 2], [2, 1], "linear_ramp", 4.0);
    assert(f9.kind == scientific::F_NONE && f10.kind == scientific::F_NONE && common::all_near(r9, r10.data));
    let r11, f11 = scientific::pad_axes(a, [1, 2], [2, 1], "linear_ramp", [], [4.0, 4.0]);
    let r12, f12 = scientific::pad_axes(a, [1, 2], [2, 1], "linear_ramp", [], [4.0, 4.0, 4.0, 4.0]);
    assert(f11.kind == scientific::F_NONE && f12.kind == scientific::F_NONE && common::all_near(r11, r10.data) && common::all_near(r12, r10.data));
    let r13, f13 = scientific::pad_axes(a, [1, 2], [2, 1], "mean", [2]);
    let r14, f14 = scientific::pad_modes(a, [1, 2], [2, 1], "mean", 0.0, 2);
    assert(f13.kind == scientific::F_NONE && f14.kind == scientific::F_NONE && common::all_near(r13, r14.data));
    // A pair of the vector form: the line of one axis.
    let v = common::av([1.0, 2.0, 3.0]);
    let r15, f15 = scientific::pad_axes(v, [2], [1], "constant", [], [], [5.0, 6.0]);
    assert(f15.kind == scientific::F_NONE && common::all_near(r15, [5.0, 5.0, 1.0, 2.0, 3.0, 6.0]));
    // A width of one pads with its one element in the mirror modes.
    let one = common::av([7.0]);
    let r16, f16 = scientific::pad_axes(one, [2], [2], "reflect");
    assert(f16.kind == scientific::F_NONE && common::all_near(r16, [7.0, 7.0, 7.0, 7.0, 7.0]));
    // A length of one with the median gives the nearest value.
    let r17, f17 = scientific::pad_axes(v, [1], [1], "median", [1]);
    assert(f17.kind == scientific::F_NONE && common::all_near(r17, [1.0, 1.0, 2.0, 3.0, 3.0]));
    // A count of zero on every side copies the input in every mode.
    var m: i64 = 0;
    while m < 7 {
        let rz, fz = scientific::pad_axes(a, [0, 0], [0, 0], common_2::qt_pad_mode(m), [1], [3.0], [4.0]);
        assert(fz.kind == scientific::F_NONE && common::all_near(rz, a.data));
        m += 1;
    }

    // Random inputs against the line by line construction.
    var g = scientific::default_rng(294);
    var t: i64 = 0;
    while t < 50 {
        var d: f64[>..] = [];
        var i: i64 = 0;
        while i < 12 { d.push(10.0 * scientific::random_uniform(g) - 5.0); i += 1; }
        let x = common::mx(3, 4, d);
        var cnt: i64[>..] = [];
        var lens: i64[>..] = [];
        var ends: f64[>..] = [];
        var cons: f64[>..] = [];
        i = 0;
        while i < 4 {
            let c0, ff = scientific::integers(g, 0, 4);
            cnt.push(c0);
            let l0, fl = scientific::integers(g, 1, 6);
            lens.push(l0);
            ends.push(10.0 * scientific::random_uniform(g) - 5.0);
            cons.push(10.0 * scientific::random_uniform(g) - 5.0);
            i += 1;
        }
        let mi, fm = scientific::integers(g, 0, 7);
        let mode = common_2::qt_pad_mode(mi);
        let b = [cnt[0], cnt[2]];
        let af = [cnt[1], cnt[3]];
        let p, f = scientific::pad_axes(x, b, af, mode, lens, ends, cons);
        assert(f.kind == scientific::F_NONE);
        let want = common_2::qt_pad2(x.data, 3, 4, b, af, mode, lens, ends, cons);
        assert(p.shape[0] == 3 + cnt[0] + cnt[1] && p.shape[1] == 4 + cnt[2] + cnt[3]);
        assert(common::all_near(p, want));
        t += 1;
    }
    return;
}

// test_sketch checks the structure of the sketch and its norms.
fn test_sketch() {
    var g = scientific::default_rng(77);
    var t: i64 = 0;
    while t < 20 {
        let n, f1 = scientific::integers(g, 1, 30);
        let d, f2 = scientific::integers(g, 1, 6);
        let s, f3 = scientific::integers(g, 1, 12);
        var dat: f64[>..] = [];
        var i: i64 = 0;
        while i < n * d { dat.push(10.0 * scientific::random_uniform(g) - 5.0); i += 1; }
        let x = common::mx(n, d, dat);
        let y, f = scientific::clarkson_woodruff_transform(x, s, 5 + t);
        assert(f.kind == scientific::F_NONE && y.ndim == 2 && y.shape[0] == s && y.shape[1] == d);
        let want = common_2::qt_sketch_rebuild(x.data, n, d, s, 5 + t);
        assert(common::all_near(y, want));
        let y2, f4 = scientific::clarkson_woodruff_transform(x, s, 5 + t);
        assert(f4.kind == scientific::F_NONE && common::all_near(y2, y.data));
        t += 1;
    }
    // Another seed gives another sketch.
    var big: f64[>..] = [];
    var i: i64 = 0;
    while i < 40 * 3 { big.push(1.0 + i); i += 1; }
    let bx = common::mx(40, 3, big);
    let u1, g1 = scientific::clarkson_woodruff_transform(bx, 8, 1);
    let u2, g2 = scientific::clarkson_woodruff_transform(bx, 8, 2);
    assert(g1.kind == scientific::F_NONE && g2.kind == scientific::F_NONE && !common::all_near(u1, u2.data));
    // The norm ratio of a sketch with many rows stays near one.
    var rows: f64[>..] = [];
    i = 0;
    while i < 400 * 5 { rows.push(scientific::random_uniform(g) - 0.5); i += 1; }
    let a = common::mx(400, 5, rows);
    let sa, fs = scientific::clarkson_woodruff_transform(a, 100, 3);
    assert(fs.kind == scientific::F_NONE);
    var k: i64 = 0;
    while k < 20 {
        var xv: f64[>..] = [];
        i = 0;
        while i < 5 { xv.push(scientific::random_uniform(g) - 0.5); i += 1; }
        var ax: f64[>..] = [];
        i = 0;
        while i < 400 {
            var q: f64 = 0.0;
            var j: i64 = 0;
            while j < 5 { q += a.data[i * 5 + j] * xv[j]; j += 1; }
            ax.push(q);
            i += 1;
        }
        var sx: f64[>..] = [];
        i = 0;
        while i < 100 {
            var q: f64 = 0.0;
            var j: i64 = 0;
            while j < 5 { q += sa.data[i * 5 + j] * xv[j]; j += 1; }
            sx.push(q);
            i += 1;
        }
        let ratio = common_2::qt_norm(sx) / common_2::qt_norm(ax);
        assert(ratio > 0.5 && ratio < 1.5);
        k += 1;
    }
    // A size of one sums all rows with their signs.
    let one, fo = scientific::clarkson_woodruff_transform(bx, 1, 4);
    assert(fo.kind == scientific::F_NONE && one.shape[0] == 1 && common::all_near(one, common_2::qt_sketch_rebuild(bx.data, 40, 3, 1, 4)));
    // A size above the count of rows is valid. A matrix of one row too.
    let many, fm = scientific::clarkson_woodruff_transform(common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]), 9, 4);
    assert(fm.kind == scientific::F_NONE && many.shape[0] == 9 && many.data.len == 18);
    let row1, fr = scientific::clarkson_woodruff_transform(common::mx(1, 3, [1.0, -2.0, 3.0]), 4, 4);
    assert(fr.kind == scientific::F_NONE && row1.data.len == 12);
    var nz: i64 = 0;
    i = 0;
    while i < 12 { if row1.data[i] != 0.0 { nz += 1; } i += 1; }
    assert(nz == 3);
    // A zero matrix gives a zero sketch.
    let zr, fz = scientific::clarkson_woodruff_transform(common::mx(3, 2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 2, 1);
    assert(fz.kind == scientific::F_NONE && common_2::qt_maxabs(zr.data) == 0.0);
    return;
}

// test_dare_random compares the call with a residual test for 20 random
// pairs of side 2 to 5 with a nonsingular r.
fn test_dare_random() {
    var g = scientific::default_rng(2941);
    var t: i64 = 0;
    while t < 20 {
        let n, f1 = scientific::integers(g, 2, 6);
        let m, f2 = scientific::integers(g, 1, 4);
        var ad: f64[>..] = [];
        var i: i64 = 0;
        while i < n * n { ad.push(0.6 * (scientific::random_uniform(g) - 0.5)); i += 1; }
        var bd: f64[>..] = [];
        i = 0;
        while i < n * m { bd.push(scientific::random_uniform(g) - 0.5); i += 1; }
        // q and r are symmetric and positive definite: c^T c + I.
        var cq: f64[>..] = [];
        i = 0;
        while i < n * n { cq.push(scientific::random_uniform(g) - 0.5); i += 1; }
        var qd = common_2::qt_mm(common_2::qt_tr(cq, n, n), n, n, cq, n);
        i = 0;
        while i < n { qd[i * n + i] += 1.0; i += 1; }
        var cr: f64[>..] = [];
        i = 0;
        while i < m * m { cr.push(scientific::random_uniform(g) - 0.5); i += 1; }
        var rd = common_2::qt_mm(common_2::qt_tr(cr, m, m), m, m, cr, m);
        i = 0;
        while i < m { rd[i * m + i] += 1.0; i += 1; }
        let a = common::mx(n, n, ad);
        let b = common::mx(n, m, bd);
        let q = common::mx(n, n, qd);
        let r = common::mx(m, m, rd);
        let x, f = scientific::solve_discrete_are(a, b, q, r);
        assert(f.kind == scientific::F_NONE);
        assert(common_2::qt_dare_resid(a.data, b.data, q.data, r.data, x.data, [], [], n, m) < 1.0e-9);
        // The same equation with the weight r split in the pencil route.
        let xp, fp = scientific::solve_discrete_are_pencil(a, b, q, r);
        assert(fp.kind == scientific::F_NONE);
        assert(common_2::qt_maxabs([0.0]) == 0.0);
        var diff: f64 = 0.0;
        i = 0;
        while i < n * n { diff = common::mathshim_abs(x.data[i] - xp.data[i]); if diff > 1.0e-8 * (1.0 + common::mathshim_abs(x.data[i])) { assert(false); } i += 1; }
        t += 1;
    }
    return;
}

// The security of the calls of the group. A parameter that is not a
// number and a parameter that is infinite give a fault. A rate so
// small that the limit of the Poisson law rounds to one gives a count
// of zero and not a count below zero.
fn test_calls_refuse_a_bad_parameter() {
    let nan = 0.0 / 0.0;
    let inf = 1.0 / 0.0;
    let v = scientific::F_VALUE;
    var r = scientific::default_rng(13);
    // The mean and the spread of the normal law are finite numbers.
    let n1, f1 = scientific::normal(r, nan, 1.0);
    assert(f1.kind == v && n1 == 0.0);
    let n2, f2 = scientific::normal(r, 0.0, nan);
    assert(f2.kind == v && n2 == 0.0);
    let n3, f3 = scientific::normal(r, inf, 1.0);
    assert(f3.kind == v && n3 == 0.0);
    let n4, f4 = scientific::normal(r, 0.0, inf);
    assert(f4.kind == v && n4 == 0.0);
    // The scale of the exponential law is a finite number.
    let e1, f5 = scientific::exponential(r, nan);
    assert(f5.kind == v && e1 == 0.0);
    let e2, f6 = scientific::exponential(r, inf);
    assert(f6.kind == v && e2 == 0.0);
    // The rate of the Poisson law is a finite number.
    let p1, f7 = scientific::poisson(r, nan);
    assert(f7.kind == v && p1 == 0);
    let p2, f8 = scientific::poisson(r, inf);
    assert(f8.kind == v && p2 == 0);
    // A rate so small that the limit of the law rounds to one gives a
    // count of zero.
    let p3, f9 = scientific::poisson(r, 1.0e-17);
    assert(f9.kind == scientific::F_NONE && p3 == 0);
    return;
}
