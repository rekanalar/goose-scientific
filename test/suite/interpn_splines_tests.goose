// The tests of the group interpn_splines. Run them with SCIENTIFIC_SUITES=interpn_splines.
namespace interpn_splines_tests;

import scientific from "../..";
import os;
import common;

// interpn with the spline methods and with extrapolation. The methods cubic and quintic
// are pinned against the legacy methods of SciPy, which fold one axis at a time.
fn test_interpn_splines() {
    let g = [0.0, 0.7, 1.5, 2.4, 4.0, 5.5, -1.0, 0.2, 1.1, 2.0, 3.3, 4.0, 5.5];
    let v0 = common::av([0.3, 0.012, 0.36300000000000004, 1.2, 3.2669999999999995, 4.8, 9.075, 1.0142176872376911, 0.642217687237691, 0.930217687237691, 1.704217687237691, 3.6802176872376906, 5.164217687237691, 9.334217687237691, 1.4474949866040543, 0.9794949866040543, 1.1954949866040545, 1.8974949866040542, 3.7694949866040535, 5.197494986604054, 9.247494986604053, 1.2154631805511509, 0.6394631805511509, 0.774463180551151, 1.3954631805511508, 3.1504631805511507, 4.5154631805511505, 8.43046318055115, -0.05680249530792819, -0.8248024953079283, -0.8338024953079282, -0.3568024953079283, 1.190197504692071, 2.4431975046920713, 6.118197504692071, 0.1444596744296081, -0.8035403255703919, -0.947540325570392, -0.6055403255703921, 0.7464596744296075, 1.8944596744296076, 5.344459674429608]);
    let v, vf = scientific::reshape(v0, [6, 7]);
    assert(vf.kind == scientific::F_NONE);
    let pts = common::mx(5, 2, [0.3, 0.5, 1.0, 2.5, 4.9, 3.7, 2.2, -0.5, 0.0, 5.5]);
    let ext = common::mx(4, 2, [-0.5, 0.5, 6.5, 3.0, 2.0, 6.5, -1.0, -2.0]);
    let in_linear, fi_linear = scientific::interpn(g, [6, 7], v, pts, "linear");
    assert(fi_linear.kind == scientific::F_NONE && common::all_near(in_linear, [0.390093294530439, 2.521696674500077, 1.6039548065345932, 1.0370258041184626, 9.075]));
    let ex_linear, fe_linear = scientific::interpn(g, [6, 7], v, ext, "linear", false, 0.0 / 0.0, true);
    assert(fe_linear.kind == scientific::F_NONE && common::all_near(ex_linear, [-0.306155490884065, 0.16863445425463208, 11.443588427685773, -0.5803109817681302]));
    let in_nearest, fi_nearest = scientific::interpn(g, [6, 7], v, pts, "nearest");
    assert(fi_nearest.kind == scientific::F_NONE && common::all_near(in_nearest, [0.012, 1.704217687237691, 1.8944596744296076, 1.2154631805511509, 9.075]));
    let ex_nearest, fe_nearest = scientific::interpn(g, [6, 7], v, ext, "nearest", false, 0.0 / 0.0, true);
    assert(fe_nearest.kind == scientific::F_NONE && common::all_near(ex_nearest, [0.012, 0.7464596744296075, 8.43046318055115, 0.3]));
    let in_slinear, fi_slinear = scientific::interpn(g, [6, 7], v, pts, "slinear");
    assert(fi_slinear.kind == scientific::F_NONE && common::all_near(in_slinear, [0.39009329453043895, 2.521696674500077, 1.6039548065345928, 1.0370258041184626, 9.075]));
    let ex_slinear, fe_slinear = scientific::interpn(g, [6, 7], v, ext, "slinear", false, 0.0 / 0.0, true);
    assert(fe_slinear.kind == scientific::F_NONE && common::all_near(ex_slinear, [-0.306155490884065, 0.16863445425463186, 11.443588427685771, -0.580310981768131]));
    let in_cubic, fi_cubic = scientific::interpn(g, [6, 7], v, pts, "cubic");
    assert(fi_cubic.kind == scientific::F_NONE && common::all_near(in_cubic, [0.3601378668491843, 2.4644145438874174, 1.226847818416172, 0.9901070923899632, 9.075]));
    let ex_cubic, fe_cubic = scientific::interpn(g, [6, 7], v, ext, "cubic", false, 0.0 / 0.0, true);
    assert(fe_cubic.kind == scientific::F_NONE && common::all_near(ex_cubic, [-0.43286976485756157, 2.1888610637244117, 12.27985974354317, -0.04220496744887711]));
    let in_quintic, fi_quintic = scientific::interpn(g, [6, 7], v, pts, "quintic");
    assert(fi_quintic.kind == scientific::F_NONE && common::all_near(in_quintic, [0.3497547726283384, 2.46917224288565, 1.2687607840782, 0.9911859361128745, 9.075]));
    let ex_quintic, fe_quintic = scientific::interpn(g, [6, 7], v, ext, "quintic", false, 0.0 / 0.0, true);
    assert(fe_quintic.kind == scientific::F_NONE && common::all_near(ex_quintic, [-0.28457086698537554, 1.3769166842529352, 12.281152123659014, 0.6320075452968477]));
    let in_pchip, fi_pchip = scientific::interpn(g, [6, 7], v, pts, "pchip");
    assert(fi_pchip.kind == scientific::F_NONE && common::all_near(in_pchip, [0.3774395182316613, 2.431191642057183, 1.4558408967902194, 0.9462779416130317, 9.074999999999998]));
    let ex_pchip, fe_pchip = scientific::interpn(g, [6, 7], v, ext, "pchip", false, 0.0 / 0.0, true);
    assert(fe_pchip.kind == scientific::F_NONE && common::all_near(ex_pchip, [-0.5062516107030002, 0.6494846343457823, 12.321037586196036, 0.18500419002241833]));
    // Faults: a grid with too few points for the degree, and an unknown name.
    let t1, ft1 = scientific::interpn([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], [3, 3], common::mx(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]), common::mx(1, 2, [0.5, 0.5]), "cubic");
    assert(ft1.kind == scientific::F_SHAPE);
    var big: f64[>..] = [];
    var bi: i64 = 0;
    while bi < 513 { big.push(bi * 1.0); bi += 1; }
    var bv: f64[>..] = [];
    bi = 0;
    while bi < 513 { bv.push(0.0); bi += 1; }
    let bva = common::av(bv);
    let t4, ft4 = scientific::interpn(big, [513], bva, common::mx(1, 1, [1.5]), "cubic");
    assert(ft4.kind == scientific::F_LIMIT);
    let t2, ft2 = scientific::interpn(g, [6, 7], v, pts, "other");
    assert(ft2.kind == scientific::F_VALUE);
    let t3, ft3 = scientific::interpn(g, [6, 7], v, ext, "cubic");
    assert(ft3.kind == scientific::F_RANGE);
}
