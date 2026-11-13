// The tests of the group griddata_columns. Run them with SCIENTIFIC_SUITES=griddata_columns.
namespace griddata_columns_tests;

import scientific from "../..";
import os;
import common;
import common_3;

// test_griddata_columns pins the values of SciPy for values of several columns.
fn test_griddata_columns() {
    let p = common::mx(5, 2, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5]);
    let v = common::mx(5, 2, [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0]);
    let xi = common::mx(4, 2, [0.25, 0.25, 0.75, 0.5, 2.0, 2.0, 0.5, 0.1]);
    let r1, f1 = scientific::griddata(p, v, xi, "linear", -1.0);
    assert(f1.kind == scientific::F_NONE && r1.ndim == 2 && r1.shape[0] == 4 && r1.shape[1] == 2);
    assert(common_3::nd_same(r1, [3.0, 30.0, 4.0, 40.0, -1.0, -1.0, 2.2, 22.0]));
    let r2, f2 = scientific::griddata(p, v, xi, "nearest");
    assert(f2.kind == scientific::F_NONE && common_3::nd_same(r2, [1.0, 10.0, 5.0, 50.0, 4.0, 40.0, 5.0, 50.0]));
    let r3, f3 = scientific::griddata(p, v, xi, "linear", 0.0 / 0.0, true);
    assert(f3.kind == scientific::F_NONE && common_3::nd_same(r3, [3.0, 30.0, 4.0, 40.0, 0.0/0.0, 0.0/0.0, 2.1999999999999997, 22.0]));
    // Values of one dimension keep a result of one dimension.
    let col = common_3::nd_line([1.0, 2.0, 3.0, 4.0, 5.0]);
    let r4, f4 = scientific::griddata(p, col, xi, "linear", -1.0);
    assert(f4.kind == scientific::F_NONE && r4.ndim == 1 && common_3::nd_same(r4, [3.0, 4.0, -1.0, 2.2]));
    // The random set of twelve points of a plane.
    let rp = common::mx(12, 2, [3.2200116949815207, 0.8079407897364937, 2.061302244168568, 0.2858013800881416, 0.2157228095266257, 0.38336888078551823, 1.6338928216799946, 0.045275193902445166, 0.19503084290867223, 0.9991761150650714, 2.609476446351951, 0.23451020166982395, 1.7397902089005681, 0.9741861932592554, 3.590710432434195, 0.844231037608741, 1.5696186573391127, 0.4930230187317426, 2.706757407324264, 0.06080271295805606, 2.2223844676828937, 0.27145160453010153, 3.5186046933396886, 0.06421443731219101]);
    let rv = common::mx(12, 2, [0.679181533021365, 0.8700885023275033, 0.2273185251609081, 0.895448239414126, 0.872195468024335, 0.01851721767021075, 0.7074955673371773, 0.0011996835868286482, 0.5033639655536645, 0.4366670521756527, 0.2032528361145648, 0.32494264457560607, 0.8062153310270436, 0.31645208740449016, 0.1490385835335558, 0.6985119903183108, 0.4485441007603762, 0.798939490963134, 0.23551645730617143, 0.3197846543182863, 0.7998795260549534, 0.5070681389233913, 0.506385001421571, 0.2361941283958635]);
    let rq = common::mx(6, 2, [0.05814512145525308, 0.9332239002254337, 0.3433031643473199, 0.84492680147361, 1.471522955646067, 0.9510229811957994, 1.5977010463152257, 0.9364420618420031, 2.2246391021355882, 0.24013532830277584, 2.965686680111251, 0.6743897148867628]);
    let r5, f5 = scientific::griddata(rp, rv, rq, "linear", -1.0);
    assert(f5.kind == scientific::F_NONE && common_3::nd_same(r5, [-1.0, -1.0, 0.558019354503607, 0.40633638745699363, 0.7351894388633744, 0.36399813680049836, 0.7515229474645407, 0.3663383338754342, 0.748294303684064, 0.46004769132399315, 0.5675979613190607, 0.6949705616658863]));
    let r6, f6 = scientific::griddata(rp, rv, rq, "linear", -1.0, true);
    assert(f6.kind == scientific::F_NONE && common_3::nd_same(r6, [-1.0, -1.0, 0.558019354503607, 0.4063363874569936, 0.7351894388633745, 0.3639981368004982, 0.7515229474645407, 0.36633833387543413, 0.748294303684064, 0.4600476913239931, 0.7075342078612104, 0.7805171616377014]));
    let r7, f7 = scientific::griddata(rp, rv, rq, "nearest");
    assert(f7.kind == scientific::F_NONE && common_3::nd_same(r7, [0.5033639655536645, 0.4366670521756527, 0.5033639655536645, 0.4366670521756527, 0.8062153310270436, 0.31645208740449016, 0.8062153310270436, 0.31645208740449016, 0.7998795260549534, 0.5070681389233913, 0.679181533021365, 0.8700885023275033]));
    // One dimension with two columns.
    let lp = common_3::nd_line([3.0, 1.0, 2.0, 5.0]);
    let lv = common::mx(4, 2, [30.0, 3.0, 10.0, 1.0, 20.0, 2.0, 50.0, 5.5]);
    let lq = common_3::nd_line([1.5, 2.5, 4.0, 6.0, 0.5]);
    let r8, f8 = scientific::griddata(lp, lv, lq, "linear", -1.0);
    assert(f8.kind == scientific::F_NONE && common_3::nd_same(r8, [15.0, 1.5, 25.0, 2.5, 40.0, 4.25, -1.0, -1.0, -1.0, -1.0]));
    let r9, f9 = scientific::griddata(lp, lv, lq, "nearest");
    assert(f9.kind == scientific::F_NONE && common_3::nd_same(r9, [10.0, 1.0, 20.0, 2.0, 30.0, 3.0, 50.0, 5.5, 10.0, 1.0]));
    // Each column equals the call with that column alone.
    let c0 = common_3::nd_line([1.0, 2.0, 3.0, 4.0, 5.0]);
    let c1 = common_3::nd_line([10.0, 20.0, 30.0, 40.0, 50.0]);
    let s0, g0 = scientific::griddata(p, c0, xi, "linear", -1.0);
    let s1, g1 = scientific::griddata(p, c1, xi, "linear", -1.0);
    var i: i64 = 0;
    while i < 4 {
        assert(r1.data[2 * i] == s0.data[i] && r1.data[2 * i + 1] == s1.data[i]);
        i += 1;
    }
    // Hostile values.
    let short = common::mx(4, 2, [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]);
    let h1, e1 = scientific::griddata(p, short, xi);
    assert(e1.kind == scientific::F_SHAPE);
    let nan = 0.0 / 0.0;
    let bad = common::mx(5, 2, [1.0, 10.0, 2.0, nan, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0]);
    let h2, e2 = scientific::griddata(p, bad, xi);
    assert(e2.kind == scientific::F_VALUE);
    return;
}
