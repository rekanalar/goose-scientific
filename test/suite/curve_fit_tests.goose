// The tests of the group curve_fit. Run them with SCIENTIFIC_SUITES=curve_fit.
namespace curve_fit_tests;

import scientific from "../..";
import os;
import common;

// curve_fit is pinned against SciPy.
fn test_curve_fit() {
    let x = [0.0, 0.36363636363636365, 0.7272727272727273, 1.0909090909090908, 1.4545454545454546, 1.8181818181818183, 2.1818181818181817, 2.5454545454545454, 2.909090909090909, 3.272727272727273, 3.6363636363636367, 4.0];
    let y = [3.02, 2.0282500972972257, 1.4812573462907253, 1.1453847417432672, 0.8573363330894807, 0.7651937511001847, 0.6365962742142248, 0.6113734634230907, 0.5169530833077661, 0.5454988590422815, 0.5521264402226302, 0.49379141105190194];
    // Bounds on the parameters.
    let bx = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let by = [1.0, 2.7, 7.4, 20.1, 54.6, 148.4];
    let pb, cb, fb = scientific::curve_fit(bx, by, [1.0, 0.5], 200, [], [0.0, 0.0], [1.5, 0.9]) { t, q => q[0] * common::mathshim_exp(q[1] * t) };
    assert(fb.kind == scientific::F_NONE && common::mathshim_abs(pb.data[0] - 1.5) < 1.0e-6 && common::mathshim_abs(pb.data[1] - 0.9) < 1.0e-6);
    assert(common::mathshim_abs(cb.data[0] - 0.47668620363771635) < 1.0e-2 && common::mathshim_abs(cb.data[3] - 0.009094064870893112) < 1.0e-3);
    let pc, cc, fc = scientific::curve_fit(bx, by, [1.0, 0.5], 200, [], [0.5, 0.0], [1.0e308 * 10.0, 0.95]) { t, q => q[0] * common::mathshim_exp(q[1] * t) };
    assert(fc.kind == scientific::F_NONE && common::mathshim_abs(pc.data[0] - 1.273035098317418) < 1.0e-5 && common::mathshim_abs(pc.data[1] - 0.95) < 1.0e-8);
    let pd, cd, fd = scientific::curve_fit(bx, by, [1.0, 0.5], 200, [], [2.0, 0.0], [3.0, 1.0]) { t, q => q[0] * common::mathshim_exp(q[1] * t) };
    assert(fd.kind == scientific::F_VALUE);
    let pe, ce, fe = scientific::curve_fit(bx, by, [1.0, 0.5], 200, [], [0.0], [3.0, 1.0]) { t, q => q[0] * common::mathshim_exp(q[1] * t) };
    assert(fe.kind == scientific::F_SHAPE);
    let p, c, f = scientific::curve_fit(x, y, [1.0, 1.0, 0.0]) { t, q => q[0] * common::mathshim_exp(-q[1] * t) + q[2] };
    assert(f.kind == scientific::F_NONE);
    assert(common::loose(p.data[0], 2.504842778143317) && common::loose(p.data[1], 1.2966838932208153) && common::loose(p.data[2], 0.501188851174762));
    assert(common::loose(c.data[0], 0.0008144738293533199) && common::loose(c.data[4], 0.001227646248841025) && common::loose(c.data[8], 0.00022564752813311048));
    assert(common::loose(c.data[1], 0.0001032144836216253) && common::loose(c.data[2], -0.00014630932777603877));
    let y2 = [-1.98, -0.9390909090909092, 0.19181818181818167, 1.3127272727272725, 2.3436363636363633, 3.484545454545455, 4.535454545454545, 5.656363636363636, 6.687272727272727, 7.828181818181818, 8.93909090909091, 9.98];
    let p2, c2, f2 = scientific::curve_fit(x, y2, [1.0, 0.0]) { t, q => q[0] * t + q[1] };
    assert(f2.kind == scientific::F_NONE);
    assert(common::loose(p2.data[0], 2.997884615327012) && common::loose(p2.data[1], -1.992435897275306));
    assert(common::loose(c2.data[0], 4.00973865517021e-05) && common::loose(c2.data[1], -8.019477395462219e-05) && common::loose(c2.data[3], 0.00022357331117456666));
    // The values of sigma scale the residuals.
    let sig = [0.05, 0.06, 0.05, 0.07, 0.05, 0.06, 0.05, 0.07, 0.05, 0.06, 0.05, 0.07];
    let ps, cs, fs = scientific::curve_fit(x, y, [1.0, 1.0, 0.0], 200, sig) { t, q => q[0] * common::mathshim_exp(-q[1] * t) + q[2] };
    assert(fs.kind == scientific::F_NONE);
    assert(common::loose(ps.data[0], 2.5105340700115826) && common::loose(ps.data[1], 1.3067584455977996) && common::loose(ps.data[2], 0.5014645542629256));
    // The covariance holds a scale. The comparison uses it, because the
    // small values of the matrix lose their digits in SciPy.
    assert(common::cov_close(cs.data[0], 0.0006939681062620468) && common::cov_close(cs.data[4], 0.001161032323389619) && common::cov_close(cs.data[8], 0.00022211805651708264));
    assert(common::cov_close(cs.data[1], 1.4921191905565632e-05) && common::cov_close(cs.data[2], -0.000159433318239479));
    // One value of sigma serves every point.
    let pv, cv, fv = scientific::curve_fit(x, y, [1.0, 1.0, 0.0], 200, [0.1]) { t, q => q[0] * common::mathshim_exp(-q[1] * t) + q[2] };
    assert(fv.kind == scientific::F_NONE);
    assert(common::loose(pv.data[0], 2.5048427779253335) && common::loose(pv.data[1], 1.2966838921487769) && common::loose(pv.data[2], 0.5011888506950971));
    assert(common::cov_close(cv.data[0], 0.0008144738275679136) && common::cov_close(cv.data[4], 0.0012276462627711755) && common::cov_close(cv.data[8], 0.00022564753169769866));
    // A matrix of sigma holds the covariance.
    let mat: f64[>..] = [];
    var mr: i64 = 0;
    while mr < 12 {
        var mc: i64 = 0;
        while mc < 12 {
            if mr == mc { mat.push(sig[mr] * sig[mr]); } else { mat.push(0.0); }
            mc += 1;
        }
        mr += 1;
    }
    let pm, cm, fm = scientific::curve_fit(x, y, [1.0, 1.0, 0.0], 200, mat) { t, q => q[0] * common::mathshim_exp(-q[1] * t) + q[2] };
    assert(fm.kind == scientific::F_NONE);
    assert(common::loose(pm.data[0], 2.510534070187927) && common::loose(pm.data[1], 1.3067584452752734) && common::loose(pm.data[2], 0.5014645540838876));
    assert(common::cov_close(cm.data[0], 0.0006939680930908152) && common::cov_close(cm.data[4], 0.00116103238092085) && common::cov_close(cm.data[8], 0.00022211806304844883));
    // A wrong count of sigma, a value of zero and a matrix that is not
    // positive definite are faults.
    let b7, b8, g7 = scientific::curve_fit(x, y, [1.0, 1.0, 0.0], 200, [0.1, 0.2]) { t, q => q[0] * common::mathshim_exp(-q[1] * t) + q[2] };
    assert(g7.kind == scientific::F_SHAPE);
    let b9, b10, g8 = scientific::curve_fit(x, y, [1.0, 1.0, 0.0], 200, [0.0]) { t, q => q[0] * common::mathshim_exp(-q[1] * t) + q[2] };
    assert(g8.kind == scientific::F_VALUE);
    let bad: f64[>..] = [];
    var br: i64 = 0;
    while br < 12 {
        var bc: i64 = 0;
        while bc < 12 {
            if br == bc { bad.push(-1.0); } else { bad.push(0.0); }
            bc += 1;
        }
        br += 1;
    }
    let b11, b12, g9 = scientific::curve_fit(x, y, [1.0, 1.0, 0.0], 200, bad) { t, q => q[0] * common::mathshim_exp(-q[1] * t) + q[2] };
    assert(g9.kind == scientific::F_VALUE);
    // Too few points and data of unequal length are faults.
    let b1, b2, g1 = scientific::curve_fit([1.0, 2.0], [1.0, 2.0], [1.0, 0.0]) { t, q => q[0] * t + q[1] };
    assert(g1.kind == scientific::F_VALUE);
    let b3, b4, g2 = scientific::curve_fit([1.0, 2.0, 3.0], [1.0, 2.0], [1.0]) { t, q => q[0] * t };
    assert(g2.kind == scientific::F_SHAPE);
    return;
}
