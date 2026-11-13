// The tests of the group cossin. Run them with SCIENTIFIC_SUITES=cossin.
namespace cossin_tests;

import scientific from "../..";
import os;
import common;
import svd_wide_tests;

fn test_cossin() {
    svd_wide_tests::test_svd_wide_rank();
    let x = scientific::Arr { ndim: 2, shape: [4, 4], data: [-0.05047916295569932, 0.12468065699085004, -0.44557229035821955, 0.8850830028559764, 0.21506477395034093, -0.370571330318633, -0.8313138415261359, -0.35403577367072037, 0.2825411849039954, 0.9024201593252053, -0.23212987916874345, -0.22786850171452355, 0.9334717328050309, -0.1810234210307588, 0.23769381596729314, 0.19840034007833807] };
    // The angles and the cosine-sine matrix of the five splits.
    let th22 = [1.1054722590375083, 1.5527252251051293];
    let cs22 = [0.44871243664417465, 0.0, -0.8936761993031075, 0.0, 0.0, 0.01807011814179432, 0.0, -0.9998367220853321, 0.8936761993031075, 0.0, 0.44871243664417465, 0.0, 0.0, 0.9998367220853321, 0.0, 0.01807011814179432];
    let u1, u2, th, v1h, v2h, fs = scientific::cossin_separate(x, 2, 2);
    assert(fs.kind == scientific::F_NONE);
    common::ct_close(th.data, th22, 1.0e-12);
    assert(u1.shape[0] == 2 && u2.shape[0] == 2 && v1h.shape[0] == 2 && v2h.shape[0] == 2);
    let u, cs, vh, f = scientific::cossin(x, 2, 2);
    assert(f.kind == scientific::F_NONE);
    common::ct_close(cs.data, cs22, 1.0e-12);
    common::ct_sign(u1.data, v1h.data, [-0.2972966118507211, 0.954785171953399, 0.954785171953399, 0.2972966118507211], [0.4910671586038948, -0.8711217169493005, 0.8711217169493005, 0.4910671586038948], 2);
    common::ct_check(x, 2, 2, 1.0e-12);

    let th13 = [1.0866723154419118];
    let cs13 = [0.46543321546216226, 0.0, 0.0, -0.8850830028559765, 0.8850830028559765, 0.0, 0.0, 0.46543321546216226, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let a1, a2, ta, a3, a4, fa = scientific::cossin_separate(x, 1, 3);
    assert(fa.kind == scientific::F_NONE);
    common::ct_close(ta.data, th13, 1.0e-12);
    assert(u1.data.len == 4 && a1.data.len == 1 && a2.data.len == 9 && a3.data.len == 9 && a4.data.len == 1);
    let ub, csb, vb, fb = scientific::cossin(x, 1, 3);
    common::ct_close(csb.data, cs13, 1.0e-12);
    common::ct_check(x, 1, 3, 1.0e-12);

    let th31 = [1.2039740016666491];
    let cs31 = [0.35865097804686513, -0.9334717328050309, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.9334717328050309, 0.35865097804686513, 0.0, 0.0];
    let b1, b2, tb, b3, b4, fc = scientific::cossin_separate(x, 3, 1);
    common::ct_close(tb.data, th31, 1.0e-12);
    let uc, csc, vc, fd = scientific::cossin(x, 3, 1);
    common::ct_close(csc.data, cs31, 1.0e-12);
    common::ct_check(x, 3, 1, 1.0e-12);

    let th21 = [1.3480494255974935];
    let cs21 = [0.2209094902601952, 0.0, -0.9752943130732286, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.9752943130732286, 0.0, 0.2209094902601952, 0.0];
    let c1, c2, tc, c3, c4, fe = scientific::cossin_separate(x, 2, 1);
    common::ct_close(tc.data, th21, 1.0e-12);
    let ud, csd, vd, ff = scientific::cossin(x, 2, 1);
    common::ct_close(csd.data, cs21, 1.0e-12);
    common::ct_check(x, 2, 1, 1.0e-12);

    let th11 = [1.520295701174285];
    let cs11 = [0.050479162955699236, 0.0, 0.0, -0.9987251143869829, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.9987251143869829, 0.0, 0.0, 0.050479162955699236];
    let d1, d2, td, d3, d4, fg = scientific::cossin_separate(x, 1, 1);
    common::ct_close(td.data, th11, 1.0e-12);
    let ue, cse, ve, fh = scientific::cossin(x, 1, 1);
    common::ct_close(cse.data, cs11, 1.0e-12);
    common::ct_check(x, 1, 1, 1.0e-12);

    // The cosines and the sines of the angles fill the matrix cs.
    assert(common::mathshim_abs(cs.data[0] - mathshim::cosine(th.data[0])) <= 1.0e-12);
    assert(common::mathshim_abs(cs.data[10] - mathshim::cosine(th.data[0])) <= 1.0e-12);
    assert(common::mathshim_abs(cs.data[8] - mathshim::sine(th.data[0])) <= 1.0e-12);
    assert(common::mathshim_abs(cs.data[5] - mathshim::cosine(th.data[1])) <= 1.0e-12);

    // 50 orthogonal matrices of side 2 to 12, every split.
    var n: i64 = 0;
    while n < 50 {
        let m = 2 + n % 11;
        let r = common::ct_orth(m, 17 + n * 31);
        var p: i64 = 1;
        while p < m {
            var q: i64 = 1;
            while q < m {
                common::ct_check(r, p, q, 1.0e-12);
                q += 1;
            }
            p += 1;
        }
        n += 1;
    }

    // The identity and the flip give no NaN and angles of 0 or pi/2.
    var side: i64 = 2;
    while side <= 6 {
        var eye: f64[>..] = [];
        var flip: f64[>..] = [];
        var i: i64 = 0;
        while i < side * side { eye.push(0.0); flip.push(0.0); i += 1; }
        i = 0;
        while i < side { eye[i * side + i] = 1.0; flip[i * side + (side - 1 - i)] = 1.0; i += 1; }
        let ei = scientific::Arr { ndim: 2, shape: [side, side], data: copy(eye) };
        let fl = scientific::Arr { ndim: 2, shape: [side, side], data: copy(flip) };
        var p: i64 = 1;
        while p < side {
            var q: i64 = 1;
            while q < side {
                common::ct_check(ei, p, q, 1.0e-12);
                common::ct_check(fl, p, q, 1.0e-12);
                let e1, e2, et, e3, e4, ef = scientific::cossin_separate(ei, p, q);
                assert(ef.kind == scientific::F_NONE);
                var k: i64 = 0;
                while k < et.data.len { assert(et.data[k] == 0.0); k += 1; }
                // The flip holds zero blocks. Its angles are pi/2.
                let f1, f2, ft, f3, f4, ff = scientific::cossin_separate(fl, p, q);
                assert(ff.kind == scientific::F_NONE);
                k = 0;
                while k < ft.data.len { assert(common::mathshim_abs(ft.data[k] - 1.5707963267948966) <= 1.0e-12); k += 1; }
                q += 1;
            }
            p += 1;
        }
        side += 1;
    }
    let one = scientific::Arr { ndim: 2, shape: [1, 1], data: [1.0] };
    let o1, o2, o3, fone = scientific::cossin(one, 1, 1);
    assert(fone.kind == scientific::F_RANGE);

    // A rotation of a small angle keeps the structure. The angles of
    // SciPy are pinned for the split 2 and 2.
    let g3 = scientific::Arr { ndim: 2, shape: [4, 4], data: [0.9987982165618562, 0.04900214398473614, -0.00093452177547949, 0.00019785551791107154, -0.04900244645598382, 0.9987986146963844, -0.0002894254255818241, 6.127670748575393e-05, 0.0008449216131918317, 0.0003078057240830517, 0.9703490150303187, 0.24170639294978183, -0.00041102512286209883, -0.00014973683189492437, -0.24170608583911316, 0.970349409597164] };
    let g6 = scientific::Arr { ndim: 2, shape: [4, 4], data: [-0.9299338175358779, -0.36772692994004547, -7.981513136639488e-08, -1.0493541076019276e-07, 0.3677269299399894, -0.929933817535381, -6.001074864257612e-07, -7.889797900510357e-07, 2.4097357824007834e-07, -9.665250503532252e-07, 0.6731568963107278, 0.7394996909724276, 2.1311884784668855e-08, -8.548020353543597e-08, -0.7394996909728507, 0.6731568963109941] };
    let g9 = scientific::Arr { ndim: 2, shape: [4, 4], data: [-0.8440308210717271, -0.5362946700098613, -3.610114766230826e-11, 1.4297369823744022e-12, 0.5362946700098616, -0.8440308210717271, -9.985643267895333e-10, 3.954678562148264e-11, 4.950577298145533e-10, -8.451148349156985e-10, 0.9866554045556312, 0.16282233465088172, -1.0196918109049084e-10, 1.7407195656971198e-10, -0.1628223346508817, 0.9866554045556312] };
    common::ct_small(g3, 1.0e-3);
    common::ct_small(g6, 1.0e-6);
    common::ct_small(g9, 1.0e-9);
    let gs1, gs2, gst, gs4, gs5, gsf = scientific::cossin_separate(g9, 1, 2);
    assert(common::mathshim_abs(gst.data[0] - 3.61294479677116e-11) <= 1.0e-15);

    // The faults. A lying count of dimensions, a short buffer, a side
    // above the limit, values that are not finite, a matrix that is
    // not square, a split outside the range and a matrix that is not
    // orthogonal.
    let l1, l2, l3, lf1 = scientific::cossin(scientific::Arr { ndim: 3, shape: [4, 4], data: copy(x.data) }, 2, 2);
    assert(lf1.kind == scientific::F_SHAPE);
    let l4, l5, l6, lf2 = scientific::cossin(scientific::Arr { ndim: 2, shape: [4, 4], data: [1.0, 0.0] }, 2, 2);
    assert(lf2.kind == scientific::F_SHAPE);
    let l7, l8, l9, lf3 = scientific::cossin(scientific::Arr { ndim: 2, shape: [513, 513], data: [1.0] }, 2, 2);
    assert(lf3.kind == scientific::F_LIMIT);
    var nan = copy(x.data);
    nan[3] = 0.0 / 0.0;
    let n1, n2, n3, nf = scientific::cossin(scientific::Arr { ndim: 2, shape: [4, 4], data: copy(nan) }, 2, 2);
    assert(nf.kind == scientific::F_VALUE);
    var inf = copy(x.data);
    inf[3] = 1.0 / 0.0;
    let i1, i2, i3, inf1 = scientific::cossin(scientific::Arr { ndim: 2, shape: [4, 4], data: copy(inf) }, 2, 2);
    assert(inf1.kind == scientific::F_VALUE);
    let s1, s2, s3, sf = scientific::cossin(scientific::Arr { ndim: 2, shape: [2, 3], data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] }, 1, 1);
    assert(sf.kind == scientific::F_SHAPE);
    let r1, r2, r3, rf0 = scientific::cossin(x, 0, 2);
    assert(rf0.kind == scientific::F_RANGE);
    let r4, r5, r6, rf1 = scientific::cossin(x, 2, 4);
    assert(rf1.kind == scientific::F_RANGE);
    let k1, k2, k3, kf = scientific::cossin(scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0, 1.0, 1.0, 1.0] }, 1, 1);
    assert(kf.kind == scientific::F_VALUE);
    // The separate form takes one matrix.
    let st = scientific::Arr { ndim: 3, shape: [2, 4, 4], data: [-0.05047916295569932, 0.12468065699085004, -0.44557229035821955, 0.8850830028559764, 0.21506477395034093, -0.370571330318633, -0.8313138415261359, -0.35403577367072037, 0.2825411849039954, 0.9024201593252053, -0.23212987916874345, -0.22786850171452355, 0.9334717328050309, -0.1810234210307588, 0.23769381596729314, 0.19840034007833807, -0.27198917008034473, 0.9141756388831458, 0.21754656299593678, 0.20731204875803041, -0.7125528205804887, -0.09777396404483507, -0.6666080748855563, 0.1958121659527865, -0.28693367853545837, 0.1277520658468022, 0.009106225010121496, -0.9493500673935035, 0.5796200767432966, 0.3720249492969867, -0.7128984809158261, -0.13196120507397927] };
    let z1, z2, z3, z4, z5, zf = scientific::cossin_separate(st, 2, 2);
    assert(zf.kind == scientific::F_SHAPE);

    // A stack gives the results of each matrix.
    let us, css, vs, sfault = scientific::cossin(st, 2, 2);
    assert(sfault.kind == scientific::F_NONE);
    assert(css.ndim == 3 && css.shape[0] == 2 && css.shape[1] == 4 && css.shape[2] == 4);
    common::ct_close(css.data, [0.44871243664417465, 0.0, -0.8936761993031075, 0.0, 0.0, 0.01807011814179432, 0.0, -0.9998367220853321, 0.8936761993031075, 0.0, 0.44871243664417465, 0.0, 0.0, 0.9998367220853321, 0.0, 0.01807011814179432, 0.9673434995441561, 0.0, -0.25346903931183634, 0.0, 0.0, 0.7008801833554142, 0.0, -0.713279025753373, 0.25346903931183634, 0.0, 0.9673434995441561, 0.0, 0.0, 0.713279025753373, 0.0, 0.7008801833554142], 1.0e-12);
    common::ct_close(us.data[0..16], u.data, 1.0e-12);

    // The four blocks give the same results as the joined matrix.
    let x11 = scientific::Arr { ndim: 2, shape: [2, 2], data: [x.data[0], x.data[1], x.data[4], x.data[5]] };
    let x12 = scientific::Arr { ndim: 2, shape: [2, 2], data: [x.data[2], x.data[3], x.data[6], x.data[7]] };
    let x21 = scientific::Arr { ndim: 2, shape: [2, 2], data: [x.data[8], x.data[9], x.data[12], x.data[13]] };
    let x22 = scientific::Arr { ndim: 2, shape: [2, 2], data: [x.data[10], x.data[11], x.data[14], x.data[15]] };
    let bu, bc, bv, bf = scientific::cossin_blocks(x11, x12, x21, x22);
    assert(bf.kind == scientific::F_NONE);
    common::ct_close(bu.data, u.data, 1.0e-12);
    common::ct_close(bc.data, cs.data, 1.0e-12);
    common::ct_close(bv.data, vh.data, 1.0e-12);
    let m1, m2, m3, mf = scientific::cossin_blocks(x11, scientific::Arr { ndim: 2, shape: [2, 3], data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] }, x21, x22);
    assert(mf.kind == scientific::F_SHAPE);
    let b1s = scientific::Arr { ndim: 3, shape: [1, 2, 2], data: copy(x11.data) };
    let m4, m5, m6, mf2 = scientific::cossin_blocks(b1s, x12, x21, x22);
    assert(mf2.kind == scientific::F_SHAPE);
    let r7, r8, r9, mf3 = scientific::cossin_blocks(x11, x12, x21, scientific::Arr { ndim: 2, shape: [3, 3], data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] });
    assert(mf3.kind == scientific::F_SHAPE);

    // The flags. A skipped factor holds no element. The swapped sign
    // gives the matrix of SciPy and still reconstructs.
    let nu, ncs, nv, nf2 = scientific::cossin(x, 2, 2, false, false, true);
    assert(nf2.kind == scientific::F_NONE && nu.data.len == 0 && nv.data.len == 16);
    let nu2, ncs2, nv2, nf3 = scientific::cossin(x, 2, 2, false, true, false);
    assert(nu2.data.len == 16 && nv2.data.len == 0);
    let wu, wcs, wv, wf = scientific::cossin(x, 2, 2, true);
    assert(wf.kind == scientific::F_NONE);
    common::ct_close(wcs.data, [0.4487124366441745, 0.0, 0.8936761993031076, 0.0, 0.0, 0.01807011814179432, 0.0, 0.9998367220853321, -0.8936761993031076, 0.0, 0.4487124366441745, 0.0, 0.0, -0.9998367220853321, 0.0, 0.01807011814179432], 1.0e-12);
    let wuc, wf1 = scientific::matmul(wu, wcs);
    let wprod, wf2 = scientific::matmul(wuc, wv);
    common::ct_close(wprod.data, x.data, 1.0e-12);

    // A matrix of side 512 shows the cube law of the work. The
    // tolerance of the product grows with the side.
    let big = common::ct_orth(512, 7);
    let gu, gcs, gvh, gf5 = scientific::cossin(big, 256, 256);
    assert(gf5.kind == scientific::F_NONE);
    let guc, gf6 = scientific::matmul(gu, gcs);
    let gprod, gf7 = scientific::matmul(guc, gvh);
    common::ct_close(gprod.data, big.data, 1.0e-9);
}
