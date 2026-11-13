// The tests of the group cheby. Run them with SCIENTIFIC_SUITES=cheby.
namespace cheby_tests;

import scientific from "../..";
import os;
import common;

// cheby1_sos is pinned against SciPy.
fn test_cheby1() {
    let a, fa = scientific::cheby1_sos(3, 1.0, [0.3], "lowpass");
    assert(fa.kind == scientific::F_NONE && common::all_near(a, [0.034384966351816024, 0.06876993270363205, 0.034384966351816024, 1.0, -0.5977088859557689, 0.0, 1.0, 1.0, 0.0, 1.0, -0.9826960523706986, 0.6664788186594006]));
    let b, fb = scientific::cheby1_sos(4, 0.5, [0.25], "lowpass");
    assert(fb.kind == scientific::F_NONE && common::all_near(b, [0.005619419204229541, 0.011238838408459082, 0.005619419204229541, 1.0, -1.329949932426935, 0.5031986921167156, 1.0, 2.0, 1.0, 1.0, -1.231461224536315, 0.7811808980112693]));
    let c, fc = scientific::cheby1_sos(4, 2.0, [0.4], "highpass");
    assert(fc.kind == scientific::F_NONE && common::all_near(c, [0.0909132735975814, -0.1818265471951628, 0.0909132735975814, 1.0, 0.5482374733732417, 0.3414066130550003, 1.0, -2.0, 1.0, 1.0, -0.4982271986803331, 0.8105470523665526]));
    let d, fd = scientific::cheby1_sos(2, 1.0, [0.2, 0.5], "bandpass");
    assert(fd.kind == scientific::F_NONE && common::all_near(d, [0.13822540868747046, 0.2764508173749409, 0.13822540868747046, 1.0, -0.05304216112661112, 0.557304021729915, 1.0, -2.0, 1.0, 1.0, -1.3601282270267474, 0.7067397185454787]));
    let e, fe = scientific::cheby1_sos(3, 3.0, [0.3, 0.6], "bandstop");
    assert(fe.kind == scientific::F_NONE && common::all_near(e, [0.24788111204717675, -0.08704122383840084, 0.24788111204717675, 1.0, -0.1297511163775456, -0.2609742924400598, 1.0, -0.3511410091698925, 1.0, 1.0, 0.6356323433677218, 0.8595451504415792, 1.0, -0.3511410091698925, 1.0, 1.0, -1.1494556109789573, 0.8803918167017183]));
    let g1, h1 = scientific::cheby1_sos(3, 0.0, [0.3], "lowpass");
    assert(h1.kind == scientific::F_VALUE);
    let g2, h2 = scientific::cheby1_sos(3, 1.0, [1.2], "lowpass");
    assert(h2.kind == scientific::F_RANGE);
    return;
}

// cheby2_sos is pinned against SciPy.
fn test_cheby2() {
    let r0, f0 = scientific::cheby2_sos(3, 30.0, [0.3], "lowpass");
    assert(f0.kind == scientific::F_NONE && common::all_near(r0, [0.0397692799675111, 0.0397692799675111, 0.0, 1.0, -0.5711000156523628, 0.0, 1.0, -0.971426219672632, 0.9999999999999998, 1.0, -1.4598918274880655, 0.6506385755873847]));
    let r1, f1 = scientific::cheby2_sos(4, 40.0, [0.25], "lowpass");
    assert(f1.kind == scientific::F_NONE && common::all_near(r1, [0.014946994058911952, 0.0023618813591870865, 0.01494699405891195, 1.0, -1.286192073852133, 0.4319468029476599, 1.0, -1.3305297855692517, 1.0000000000000004, 1.0, -1.6126739149245188, 0.7608292714568752]));
    let r2, f2 = scientific::cheby2_sos(4, 25.0, [0.4], "highpass");
    assert(f2.kind == scientific::F_NONE && common::all_near(r2, [0.20438409640140692, -0.3501043833528033, 0.20438409640140692, 1.0, -0.25303233631047123, 0.08647805683020464, 1.0, -0.7575554262491747, 0.9999999999999997, 1.0, -0.0035478386759487837, 0.558689528742404]));
    let r3, f3 = scientific::cheby2_sos(2, 30.0, [0.2, 0.5], "bandpass");
    assert(f3.kind == scientific::F_NONE && common::all_near(r3, [0.04021131306265627, 0.01583844789390953, 0.04021131306265626, 1.0, -0.779000714965027, 0.8294533123017738, 1.0, -1.7354143450159694, 1.0, 1.0, -1.0789880679024455, 0.8457146813671075]));
    let r4, f4 = scientific::cheby2_sos(3, 35.0, [0.3, 0.6], "bandstop");
    assert(f4.kind == scientific::F_NONE && common::all_near(r4, [0.146084034035982, -0.05129609513500364, 0.146084034035982, 1.0, -0.1613013579732175, -0.08127302843641923, 1.0, 0.5030145806808963, 1.0, 1.0, 1.085659506469457, 0.570049073882224, 1.0, -1.0908398786270708, 1.0, 1.0, -1.3826014621234592, 0.6551034176147763]));
    let e1, g1 = scientific::cheby2_sos(3, -1.0, [0.3], "lowpass");
    assert(g1.kind == scientific::F_VALUE);
    let e2, g2 = scientific::cheby2_sos(3, 30.0, [0.0], "lowpass");
    assert(g2.kind == scientific::F_RANGE);
    return;
}
