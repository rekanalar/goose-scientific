// The tests of the group fftn. Run them with SCIENTIFIC_SUITES=fftn.
namespace fftn_tests;

import scientific from "../..";
import os;

fn test_fftn() {
    // NumPy 2.5.3: fftshift([0,1,2]) is [2,0,1] and ifftshift is [1,2,0].
    let v, fv = scientific::from_values([0.0, 1.0, 2.0]);
    assert(fv.kind == scientific::F_NONE);
    let s, fs = scientific::fftshift(v);
    assert(fs.kind == scientific::F_NONE && s.data[0] == 2.0 && s.data[1] == 0.0 && s.data[2] == 1.0);
    let t, ft = scientific::ifftshift(v);
    assert(ft.kind == scientific::F_NONE && t.data[0] == 1.0 && t.data[1] == 2.0 && t.data[2] == 0.0);
    // The shift of a matrix moves both axes. For [[0,1,2],[3,4,5]],
    // NumPy gives [[5,3,4],[2,0,1]].
    let m, fm = scientific::from_rows(2, 3, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert(fm.kind == scientific::F_NONE);
    let sm, fsm = scientific::fftshift(m);
    assert(fsm.kind == scientific::F_NONE && sm.data[0] == 5.0 && sm.data[1] == 3.0 && sm.data[2] == 4.0 && sm.data[3] == 2.0);
    let tm, ftm = scientific::ifftshift(sm);
    assert(ftm.kind == scientific::F_NONE && tm.data[0] == 0.0 && tm.data[5] == 5.0);
    // The transform of many axes: fftn of [[1,2],[3,4]] is [[10,-2],[-4,0]].
    let a, fa = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let z, fz = scientific::from_rows(2, 2, [0.0, 0.0, 0.0, 0.0]);
    assert(fa.kind == scientific::F_NONE && fz.kind == scientific::F_NONE);
    let fr, fi, ff = scientific::fftn(a, z);
    assert(ff.kind == scientific::F_NONE && fr.data[0] == 10.0 && fr.data[1] == -2.0 && fr.data[2] == -4.0 && fr.data[3] == 0.0);
    assert(fi.data[0] == 0.0 && fi.data[3] == 0.0);
    // The inverse gives the input back.
    let br, bi, fb = scientific::ifftn(fr, fi);
    assert(fb.kind == scientific::F_NONE);
    var k: i64 = 0;
    while k < 4 {
        var d = br.data[k] - a.data[k];
        if d < 0.0 { d = -d; }
        assert(d < 1.0e-12 && bi.data[k] < 1.0e-12 && bi.data[k] > -1.0e-12);
        k += 1;
    }
    let g2r, g2i, f2 = scientific::ifft2(fr, fi);
    assert(f2.kind == scientific::F_NONE && g2r.data[3] > 3.999999999 && g2r.data[3] < 4.000000001);
    let c2r, c2i, fc2 = scientific::fft2c(a, z);
    assert(fc2.kind == scientific::F_NONE && c2r.data[0] == 10.0 && c2i.data[1] == 0.0);
    // A transform of three axes, and the shapes that disagree.
    let cube, fcube = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert(fcube.kind == scientific::F_NONE);
    let c3, fc3 = scientific::reshape(cube, [2, 2, 2]);
    let z3, fz3 = scientific::zeros([2, 2, 2]);
    assert(fc3.kind == scientific::F_NONE && fz3.kind == scientific::F_NONE);
    let r3, i3, f3 = scientific::fftn(c3, z3);
    assert(f3.kind == scientific::F_NONE && r3.data[0] == 36.0 && r3.data[7] == 0.0);
    let _x, _y, bad = scientific::fftn(a, z3);
    assert(bad.kind == scientific::F_SHAPE);
    let _x2, _y2, bad2 = scientific::ifft2(c3, z3);
    assert(bad2.kind == scientific::F_SHAPE);
    // The frequencies of the real transform: NumPy 2.5.3 gives
    // rfftfreq(8, 1) = [0, 0.125, 0.25, 0.375, 0.5].
    let q, fq = scientific::rfftfreq(8, 1.0);
    assert(fq.kind == scientific::F_NONE && q.data.len == 5 && q.data[1] == 0.125 && q.data[4] == 0.5);
    let q7, fq7 = scientific::rfftfreq(7, 0.5);
    assert(fq7.kind == scientific::F_NONE && q7.data.len == 4);
    let _q0, bq = scientific::rfftfreq(0, 1.0);
    assert(bq.kind == scientific::F_VALUE);
    let _q1, bq2 = scientific::rfftfreq(4, 0.0);
    assert(bq2.kind == scientific::F_VALUE);
}
