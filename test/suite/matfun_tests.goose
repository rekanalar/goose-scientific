// The tests of the group matfun. Run them with SCIENTIFIC_SUITES=matfun.
namespace matfun_tests;

import scientific from "../..";
import os;
import common;

fn test_matfun() {
    let eg, egf = scientific::from_rows(3, 3, [4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0]);
    let ew, ev, evf = scientific::eigh(eg);
    assert(evf.kind == scientific::F_NONE && common::near(ew.data[0], 1.7078872783917407));
    var ec: i64 = 0;
    while ec < 3 {
        var er: i64 = 0;
        while er < 3 {
            var av: f64 = 0.0;
            var ek: i64 = 0;
            while ek < 3 { av += eg.data[er * 3 + ek] * ev.data[ek * 3 + ec]; ek += 1; }
            assert(common::mathshim_abs(av - ew.data[ec] * ev.data[er * 3 + ec]) < 1e-12);
            er += 1;
        }
        ec += 1;
    }
    let sq1, sq1f = scientific::sqrtm(common::mx(3, 3, [4.0, 1.0, 0.5, 2.0, 5.0, 1.0, 0.3, 1.0, 3.0]));
    assert(sq1f.kind == scientific::F_NONE && common::all_near(sq1, [1.9705736458839043, 0.2326189015635945, 0.12052654846139223, 0.4770456766300537, 2.197288610696068, 0.24105309692278393, 0.04870018207110656, 0.25286097042564826, 1.7126521582427425]));
    let sq2, sq2f = scientific::sqrtm(common::mx(3, 3, [4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0]));
    assert(sq2f.kind == scientific::F_NONE && common::all_near(sq2, [1.9813751385926213, 0.27199847235457564, -0.01301503760473606, 0.27199847235457564, 1.7028691474356779, 0.16202931138676027, -0.013015037604736044, 0.1620293113867601, 1.4048405998716291]));
    let lg1, lg1f = scientific::logm(common::mx(3, 3, [4.0, 1.0, 0.5, 2.0, 5.0, 1.0, 0.3, 1.0, 3.0]));
    assert(lg1f.kind == scientific::F_NONE && common::all_near(lg1, [1.3294065180596149, 0.21844838449997658, 0.11804265288679479, 0.4615884587830102, 1.5355090576680641, 0.23608530577358888, 0.02144221216596351, 0.2607769955566456, 1.058047369738802]));
    let lg2, lg2f = scientific::logm(common::mx(3, 3, [4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0]));
    assert(lg2f.kind == scientific::F_NONE && common::all_near(lg2, [1.3462209177519089, 0.2998680085203079, -0.03151209665712595, 0.2998680085203081, 1.030596860903038, 0.21295819757440607, -0.031512096657125926, 0.21295819757440604, 0.6677046590684773]));
    let fp1, fp1f = scientific::fractional_matrix_power(common::mx(3, 3, [4.0, 1.0, 0.5, 2.0, 5.0, 1.0, 0.3, 1.0, 3.0]), 0.3);
    assert(fp1f.kind == scientific::F_NONE && common::all_near(fp1, [1.4972472821626006, 0.10303621366502853, 0.05424803410513154, 0.21371622369338172, 1.5964615976459684, 0.10849606821026034, 0.01726122773643093, 0.1161398645735845, 1.377831504861876]));
    let fp2, fp2f = scientific::fractional_matrix_power(common::mx(3, 3, [4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0]), -1.5);
    assert(fp2f.kind == scientific::F_NONE && common::all_near(fp2, [0.14969850863100492, -0.08248914960039823, 0.02784907813664865, -0.0824891496003982, 0.24611219729972747, -0.09694273107349637, 0.027849078136648702, -0.09694273107349645, 0.38429950317342276]));
    let pu, pp, pf = scientific::polar(common::mx(3, 3, [4.0, 1.0, 0.5, 2.0, 5.0, 1.0, 0.3, 1.0, 3.0]));
    assert(pf.kind == scientific::F_NONE && common::all_near(pu, [0.992059084201441, -0.11589029290017523, 0.04886935097694872, 0.11660154520113564, 0.9931068338620442, -0.011953919575263015, -0.047147143180727956, 0.017557236343341512, 0.9987336433413461]) && common::all_near(pp, [4.187295284253818, 1.5279196670263924, 0.4711896577596735, 1.5279196670263924, 4.867201112753388, 0.9878333964419811, 0.47118965775967353, 0.9878333964419813, 3.008681685937249]));
    let cl, clf = scientific::cho_factor(common::mx(3, 3, [4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0]));
    let cs, csf = scientific::cho_solve(cl, common::av([1.0, 2.0, 3.0]));
    assert(clf.kind == scientific::F_NONE && csf.kind == scientific::F_NONE && common::all_near(cs, [0.15476190476190477, 0.38095238095238093, 1.4047619047619047]));
    let sy, syf = scientific::solve_sylvester(common::mx(2, 2, [3.0, 1.0, 0.0, 2.0]), common::mx(3, 3, [1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 2.0]), common::mx(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert(syf.kind == scientific::F_NONE && common::all_near(sy, [-0.04283447911158113, 0.24008461131676362, 0.2842411422527766, 0.8870967741935487, 0.6451612903225802, 1.3387096774193552]));
    let cl, clf = scientific::solve_continuous_lyapunov(common::mx(3, 3, [-4.0, -1.0, -0.5, -2.0, -5.0, -1.0, -0.3, -1.0, -3.0]), common::mx(3, 3, [1.0, 0.5, 0.0, 0.5, 2.0, 1.0, 0.0, 1.0, 3.0]));
    assert(clf.kind == scientific::F_NONE && common::all_near(cl, [-0.1289495773548676, -0.008766389020677931, 0.04912939688029589, -0.00876638902067795, -0.18593391332990858, -0.05279765530910223, 0.04912939688029592, -0.052797655309102264, -0.48731372125166217]));
    let dl, dlf = scientific::solve_discrete_lyapunov(common::mx(3, 3, [0.5, 0.1, 0.0, 0.2, 0.3, 0.1, 0.0, 0.1, 0.4]), common::mx(3, 3, [1.0, 0.5, 0.0, 0.5, 2.0, 1.0, 0.0, 1.0, 3.0]));
    assert(dlf.kind == scientific::F_NONE && common::all_near(dl, [1.486984365210881, 0.8998682629247193, 0.15999637619937518, 0.8998682629247193, 2.525144761568877, 1.4438060049393848, 0.15999637619937518, 1.443806004939385, 3.7389951523938567]));
    let bw, bv, bf = scientific::eig_banded(common::mx(2, 5, [0.0, 0.5, 0.7, 0.9, 0.2, 4.0, 3.0, 2.0, 1.0, 5.0]));
    assert(bf.kind == scientific::F_NONE && common::all_near(bw, [0.41060867538102946, 2.095896048454904, 3.2338111188766776, 4.2489152019055245, 5.010768955381866]));
    let bw2, bf2 = scientific::eigvals_banded(common::mx(2, 5, [0.0, 0.5, 0.7, 0.9, 0.2, 4.0, 3.0, 2.0, 1.0, 5.0]));
    assert(bf2.kind == scientific::F_NONE && common::all_near(bw2, [0.41060867538102863, 2.0958960484549043, 3.233811118876677, 4.2489152019055245, 5.010768955381865]));
    var bo: f64 = 0.0;
    var bi: i64 = 0;
    while bi < 25 { bo += common::scientific_abs(bv.data[bi]); bi += 1; }
    assert(common::near(bo, 7.978538605903487));
    let lw, lv, lf = scientific::eig_banded(common::mx(2, 5, [4.0, 3.0, 2.0, 1.0, 5.0, 0.5, 0.7, 0.9, 0.2, 0.0]), true);
    assert(lf.kind == scientific::F_NONE && common::all_near(lw, [0.41060867538102946, 2.095896048454904, 3.2338111188766776, 4.2489152019055245, 5.010768955381866]));
    let tw, tf = scientific::eigvals_banded(common::mx(3, 5, [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.5, 0.6, 0.7, 0.8, 4.0, 3.0, 2.0, 1.0, 5.0]));
    assert(tf.kind == scientific::F_NONE && common::all_near(tw, [-0.34285674122535553, -0.07369672660208344, 3.698430363453611, 4.441065369269095, 7.277057735104733]));
    // the select of the banded eigenvalues
    let sab = common::mx(2, 5, [0.0, 0.5, 0.7, 0.9, 0.2, 4.0, 3.0, 2.0, 1.0, 5.0]);
    let sv1, sf1 = scientific::eigvals_banded(sab, false, "v", 1.0, 4.0);
    assert(sf1.kind == scientific::F_NONE && common::all_near(sv1, [2.0958960484549043, 3.2338111188766776]));
    let si1, si1f = scientific::eigvals_banded(sab, false, "i", 0.0, 0.0, 1, 3);
    assert(si1f.kind == scientific::F_NONE && common::all_near(si1, [2.095896048454904, 3.233811118876678, 4.248915201905524]));
    let sw2, sv2, sf2 = scientific::eig_banded(sab, false, "i", 0.0, 0.0, 0, 1);
    assert(sf2.kind == scientific::F_NONE && common::all_near(sw2, [0.4106086753810293, 2.0958960484549043]) && sv2.shape[0] == 5 && sv2.shape[1] == 2);
    assert(common::near(common::scientific_abs(sv2.data[3 * 2]), 0.8301897815947307) && common::near(common::scientific_abs(sv2.data[3 * 2 + 1]), 0.5091050040794023));
    let sx1, sx1f = scientific::eigvals_banded(sab, false, "v", 4.0, 1.0);
    assert(sx1f.kind == scientific::F_RANGE && sx1.data.len == 0);
    let sx2, sx2f = scientific::eigvals_banded(sab, false, "i", 0.0, 0.0, 2, 9);
    assert(sx2f.kind == scientific::F_RANGE && sx2.data.len == 0);
    let sx3, sx3f = scientific::eigvals_banded(sab, false, "q");
    assert(sx3f.kind == scientific::F_VALUE && sx3.data.len == 0);
    let n1, n1f = scientific::sqrtm(common::mx(2, 2, [-1.0, 0.0, 0.0, -1.0]));
    assert(n1f.kind == scientific::F_VALUE);
    let n2, n2f = scientific::logm(common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]));
    assert(n2f.kind == scientific::F_VALUE);
    let n3, n3f = scientific::sqrtm(common::mx(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert(n3f.kind == scientific::F_SHAPE);
    let n4, n4f = scientific::solve_sylvester(common::mx(2, 2, [1.0, 0.0, 0.0, 1.0]), common::mx(2, 2, [-1.0, 0.0, 0.0, -1.0]), common::mx(2, 2, [1.0, 1.0, 1.0, 1.0]));
    assert(n4f.kind == scientific::F_VALUE);
}

fn test_matrix_trig() {
    let b = common::tm_mat([2.0, 1.0, 0.5, 3.0]);
    let s, f1 = scientific::sinm(b);
    assert(f1.kind == scientific::F_NONE && common::all_loose(s, [0.7400706, -0.70468966, -0.35234483, 0.03538094]));
    let c, f2 = scientific::cosm(b);
    assert(f2.kind == scientific::F_NONE && common::all_loose(c, [-0.25581893, -0.52641889, -0.26320944, -0.78223782]));
    let t, f3 = scientific::tanm(b);
    assert(f3.kind == scientific::F_NONE && common::all_loose(t, [-12.4184748, 9.25806596, 4.62903298, -3.16040885]));
    let sh, f4 = scientific::sinhm(b);
    assert(f4.kind == scientific::F_NONE && common::all_loose(sh, [5.00038102, 6.92808946, 3.46404473, 11.92847048]));
    let ch, f5 = scientific::coshm(b);
    assert(f5.kind == scientific::F_NONE && common::all_loose(ch, [5.16158917, 6.83535212, 3.41767606, 11.99694129]));
    let th, f6 = scientific::tanhm(b);
    assert(f6.kind == scientific::F_NONE && common::all_loose(th, [0.94162814, 0.04098791, 0.02049395, 0.98261605]));
    let sg, f7 = scientific::signm(common::tm_mat([2.0, 1.0, 1.0, -3.0]));
    assert(f7.kind == scientific::F_NONE && common::all_loose(sg, [0.92847669, 0.37139068, 0.37139068, -0.92847669]));
    let bad, f8 = scientific::signm(common::tm_mat([0.0, 1.0, -1.0, 0.0]));
    assert(f8.kind == scientific::F_VALUE);
    let nsq, f9 = scientific::sinm(scientific::line_arr([1.0, 2.0, 3.0]));
    assert(f9.kind == scientific::F_SHAPE);
    return;
}
