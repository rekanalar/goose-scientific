// The tests of the group quad. Run them with SCIENTIFIC_SUITES=quad.
namespace quad_tests;

import scientific from "../..";
import os;
import common;

fn test_quad() {
    let q1, q1e, q1f = scientific::quad(0.0, 3.0, 1e-12, 1e-12, 200) { x => common::mathshim_sin(x) };
    assert(q1f.kind == scientific::F_NONE && common::near(q1, 1.9899924966004452));
    let q2, q2e, q2f = scientific::quad(-1.0, 2.5, 1e-12, 1e-12, 200) { x => x * x * x - 2.0 * x };
    assert(q2f.kind == scientific::F_NONE && common::near(q2, 4.265625));
    let q3, q3e, q3f = scientific::quad(-(1.0 / 0.0), (1.0 / 0.0), 1e-12, 1e-12, 200) { x => common::mathshim_exp(-x * x) };
    assert(q3f.kind == scientific::F_NONE && common::near(q3, 1.772453850905516));
    let q4, q4e, q4f = scientific::quad(0.0, (1.0 / 0.0), 1e-12, 1e-12, 200) { x => common::mathshim_exp(-x) };
    assert(q4f.kind == scientific::F_NONE && common::near(q4, 1.0000000000000002));
    let q5, q5e, q5f = scientific::quad(-(1.0 / 0.0), 1.0, 1e-12, 1e-12, 200) { x => 1.0 / (1.0 + x * x) };
    assert(q5f.kind == scientific::F_NONE && common::near(q5, 2.356194490192345));
    let q6, q6e, q6f = scientific::quad(0.0, 4.0, 1e-12, 1e-12, 200) { x => common::mathshim_sqrt(x) };
    assert(q6f.kind == scientific::F_NONE && common::near(q6, 5.333333333333335));
    let q7, q7e, q7f = scientific::quad(3.0, 0.0, 1e-12, 1e-12, 200) { x => common::mathshim_sin(x) };
    assert(q7f.kind == scientific::F_NONE && common::near(q7, -1.9899924966004452));
    let fq, fqe = scientific::fixed_quad(0.0, 2.0, 5) { x => common::mathshim_exp(x) };
    assert(fqe.kind == scientific::F_NONE && common::near(fq, 6.389056096688674));
    // Events and dense output.
    let et, ey, ee, ei, eyv, ef = scientific::solve_ivp_events(0.0, 10.0, [10.0, 0.0], 1, [true], [-1.0], 1.0e-8, 1.0e-10) { t, y => [y[1], -9.81, y[0]] };
    assert(ef.kind == scientific::F_NONE && ee.data.len == 1 && common::mathshim_abs(ee.data[0] - 1.42784312) < 1.0e-7);
    assert(common::mathshim_abs(et.data[et.data.len - 1] - 1.4278431229270645) < 1.0e-7 && common::mathshim_abs(eyv.data[1] + 14.007141035914504) < 1.0e-6 && ei.data[0] == 0.0);
    let ot, oy, oe, oi, oyv, of = scientific::solve_ivp_events(0.0, 10.0, [1.0, 0.0], 1, [false], [0.0], 1.0e-8, 1.0e-10) { t, y => [y[1], -y[0], y[0]] };
    assert(of.kind == scientific::F_NONE && oe.data.len == 3 && common::mathshim_abs(oe.data[0] - 1.5707963264325224) < 1.0e-7 && common::mathshim_abs(oe.data[1] - 4.712388979391328) < 1.0e-7 && common::mathshim_abs(oe.data[2] - 7.85398163235004) < 1.0e-7);
    assert(ot.data[ot.data.len - 1] == 10.0);
    let ob, obb, obe, obi, obv, obf = scientific::solve_ivp_events(0.0, 10.0, [1.0, 0.0], 2, [false], [0.0]) { t, y => [y[1], -y[0], y[0], y[1]] };
    assert(obf.kind == scientific::F_SHAPE);
    let dt, dy, dd, df = scientific::solve_ivp_dense(0.0, 6.0, [1.0, 0.0], 1.0e-8, 1.0e-10) { t, y => [y[1], -y[0]] };
    assert(df.kind == scientific::F_NONE);
    let dv, dvf = scientific::dense_eval(dd, [0.5, 1.7, 5.2]);
    assert(dvf.kind == scientific::F_NONE && common::all_close(dv, [0.8775825612156084, -0.47942553943018557, -0.1288444943415755, -0.9916648084454521, 0.46851667032694333, 0.883454649492061], 1.0e-7));
    let t_rk45, y_rk45, e_rk45 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "RK45") { t, y => [y[1], -y[0]] };
    assert(e_rk45.kind == scientific::F_NONE && common::all_near(t_rk45, [0.0, 0.000999000999000999, 0.010989010989010988, 0.11088911088911087, 0.7601505581783583, 1.6934656227081029, 2.6818921292145386, 3.662498620296199, 4.577872200228697, 5.0]) && common::all_near(y_rk45, [1.0, 0.0, 0.9999995009985435, -0.0009990008328333421, 0.9999396214263466, -0.010988789821184265, 0.9938580997401778, -0.11066199453652523, 0.7247097637593363, -0.6890377126475217, -0.12260764766130006, -0.992423097760466, -0.8964129826445695, -0.4431387530172439, -0.8668812844862507, 0.49842806151422436, -0.1330055785521448, 0.9910483076876293, 0.28470530202789335, 0.9585445205284323]));
    let t_rk23, y_rk23, e_rk23 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "RK23") { t, y => [y[1], -y[0]] };
    assert(e_rk23.kind == scientific::F_NONE && common::all_near(t_rk23, [0.0, 0.000999000999000999, 0.010989010989010988, 0.09501646775614085, 0.2631021017304901, 0.49917840208969655, 0.7929707310047646, 1.126751987501435, 1.3999644810686591, 1.6015049189722004, 1.8030453568757416, 2.028858031827757, 2.314302675177404, 2.6442877009293686, 2.927512396909703, 3.144063408764921, 3.360614420620139, 3.5818272726373714, 3.8635031091628056, 4.191437175612456, 4.478844465694581, 4.701414776242469, 4.923985086790357, 5.0]) && common::all_near(y_rk23, [1.0, 0.0, 0.999999500998502, -0.0009990008328333339, 0.9999396210113036, -0.010988789819940078, 0.9954872539421785, -0.09487350436657084, 0.9655528300346052, -0.2600724810893409, 0.877820072249416, -0.4786521408210731, 0.7013346669277642, -0.7121817553737404, 0.4289678535496003, -0.9022553607215416, 0.16951701028118504, -0.9843222226689229, -0.030963499382748477, -0.99826459975658, -0.23016332551647012, -0.9717924191750676, -0.44187325017090273, -0.8954838148545725, -0.6760115511939843, -0.7345848805754808, -0.8772088761745034, -0.47556509459531565, -0.9749167172987311, -0.21136544132431817, -0.9974713425177815, 0.0030596324978935663, -0.9734260739361247, 0.21730309822942753, -0.9019305259435636, 0.4255643433843891, -0.7478642959762679, 0.65937450075388, -0.49529560412761336, 0.8647741342190143, -0.22971849702806812, 0.9694494775598642, -0.010039438925150662, 0.9961437919370207, 0.21009074719856316, 0.9736865664941141, 0.28342718737507855, 0.9549188059367262]));
    let t_b, y_b, e_b = scientific::solve_ivp(1.0, -1.0, [3.0], 1e-8, 1e-10) { t, y => [-2.0 * y[0]] };
    assert(e_b.kind == scientific::F_NONE && common::all_near(t_b, [1.0, 0.9924163715019694, 0.9452288909139589, 0.896981996408003, 0.8487089585227905, 0.8004376140514565, 0.7521684987542352, 0.7039014265324454, 0.6556362101002151, 0.6073726788671658, 0.5591106779130687, 0.5108500665476304, 0.46259071700822413, 0.4143325132786344, 0.3660753500048519, 0.3178191315092954, 0.26956377091857076, 0.22130918934202373, 0.17305531513896355, 0.12480208325590475, 0.07654943462509203, 0.028297315609388474, -0.01995432250344239, -0.06820552394119547, -0.11645632886113524, -0.16470677373454357, -0.21295689167722867, -0.2612067127571098, -0.3094562642802536, -0.3577055710382285, -0.40595465554309856, -0.4542035382382169, -0.5024522376848576, -0.5507007707400633, -0.5989491527073207, -0.6471973974830382, -0.6954455176825663, -0.7436935247701385, -0.7919414291462445, -0.8401892402612638, -0.8884369666904695, -0.9366846162281081, -0.9849321959426126, -1.0]) && common::all_near(y_b, [3.0, 3.045848590731626, 3.3473015273313025, 3.6863923475153633, 4.060046278395815, 4.471558749174671, 4.924758766904875, 5.42386920767848, 5.973540915615807, 6.578895883708401, 7.245574976988688, 7.979790495716302, 8.788384063269335, 9.678890378181594, 10.659607424632133, 11.739673795515595, 12.929153848016647, 14.239131486346215, 15.681813444640502, 17.270643032602173, 19.020425403826767, 20.947465514463808, 23.069720057615154, 25.40696478827133, 27.980978799267604, 30.815747465454802, 33.93768594564593, 37.37588532627237, 41.16238370058473, 45.332464707976754, 49.92498631696867, 54.98274291465527, 60.55286407628073, 66.68725373221251, 73.44307382238468, 80.8832769474832, 89.07719297811569, 98.10117509189514, 108.03931125016106, 118.98420775588633, 131.03785218216046, 144.31256372395978, 158.9320398150422, 163.7944513686982]));
    let bq1, bq2, bqf = scientific::quad(0.0, 1.0, -1.0) { x => x };
    assert(bqf.kind == scientific::F_VALUE);
    let hq1, hq2, hqf = scientific::quad(0.0, 1.0, 1e-300, 1e-300, 3) { x => common::mathshim_sin(1.0 / x) };
    assert(hqf.kind == scientific::F_LIMIT);
    let bi1, bi2, bif = scientific::solve_ivp(0.0, 1.0, [1.0], 0.001, 1e-6, "LSODA") { t, y => [y[0]] };
    assert(bif.kind == scientific::F_VALUE);
    let bi3, bi4, bi5 = scientific::solve_ivp(0.0, 1.0, [1.0, 2.0]) { t, y => [y[0]] };
    assert(bi5.kind == scientific::F_SHAPE);
}

// solve_ivp takes the times t_eval and gives the values from the dense
// output of the method, as SciPy does.
// The method of Dormand and Prince of the order eight, pinned against
// SciPy 1.18.1.
fn test_dop853() {
    // The steps of the method. The times and the states match SciPy.
    let e1 = [0.0, 0.000999000999000999, 0.010989010989010988, 0.11088911088911087, 1.1098901098901095, 3.577580383341898, 5.0];
    let e2 = [1.0, 0.0, 0.9999995009985435, -0.0009990008328333421, 0.9999396214263468, -0.010988789821184265, 0.9938581000167407, -0.11066199451986314, 0.44475998750894596, -0.8956497674212576, -0.9063114415145316, 0.42208333978208246, 0.2834614005249253, 0.9587505956092199];
    let t1, y1, f1 = scientific::solve_ivp(0.0, 5.0, [1.0, 0.0], 0.001, 1e-6, "DOP853") { t, v => [v[1], -v[0]] };
    assert(f1.kind == scientific::F_NONE && t1.data.len == 7 && y1.data.len == 14);
    assert(common::all_near(t1, e1) && common::all_near(y1, e2));
    // The values between the steps, from the dense output.
    let e3 = [0.8775825615538217, -0.479425538086932, -0.12884449379636104, -0.9916648039614262, 0.46851666612604054, 0.883454646570579];
    let t2, y2, d2, f2 = scientific::solve_ivp_dense(0.0, 6.0, [1.0, 0.0], 1.0e-8, 1.0e-10, "DOP853") { t, v => [v[1], -v[0]] };
    assert(f2.kind == scientific::F_NONE);
    let dv, dvf = scientific::dense_eval(d2, [0.5, 1.7, 5.2]);
    assert(dvf.kind == scientific::F_NONE && common::all_near(dv, e3));
    // The values at the times of t_eval.
    let e4 = [1.0, 0.0, 0.9553364760057397, -0.29552021042511084, 0.5403013851324846, -0.8414706099471434, 0.020795124339923032, -0.9997845908803505, -0.989992409386347, -0.141120163362502];
    let t3, y3, f3 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1.0e-6, 1.0e-9, "DOP853", 1.0e308, [0.0, 0.3, 1.0, 1.55, 3.0]) { t, v => [v[1], -v[0]] };
    assert(f3.kind == scientific::F_NONE && common::all_near(y3, e4));
    // The time and the state of an event.
    let et, ey, eet, eei, eey, ef = scientific::solve_ivp_events(0.0, 10.0, [10.0, 0.0], 1, [true], [-1.0], 1.0e-8, 1.0e-10, "DOP853") { t, v => [v[1], -9.81, v[0]] };
    assert(ef.kind == scientific::F_NONE && eet.data.len == 1);
    assert(common::mathshim_abs(eet.data[0] - 1.4278431229270638) < 1.0e-9);
    assert(common::mathshim_abs(eey.data[1] + 14.00714103591449) < 1.0e-9);
}

// The method BDF, pinned against SciPy 1.18.1. The count of steps equals
// the count of SciPy and the values agree to the tolerance of the solver.
fn test_bdf() {
    // A stiff pair, the oscillator of Van der Pol with a large damping.
    let t1, y1, f1 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 0.001, 1e-6, "BDF") { t, v => [v[1], 1000.0 * (1.0 - v[0] * v[0]) * v[1] - v[0]] };
    assert(f1.kind == scientific::F_NONE && t1.data.len == 142);
    assert(common::mathshim_abs(t1.data[141] - 3.0) < 1.0e-12 && common::mathshim_abs(t1.data[3] - 0.0014270485380032654) < 1.0e-9);
    // The system of Robertson, with three components.
    let t2, y2, f2 = scientific::solve_ivp(0.0, 40.0, [1.0, 0.0, 0.0], 1e-6, 1e-9, "BDF") { t, v => [-0.04 * v[0] + 1.0e4 * v[1] * v[2], 0.04 * v[0] - 1.0e4 * v[1] * v[2] - 3.0e7 * v[1] * v[1], 3.0e7 * v[1] * v[1]] };
    assert(f2.kind == scientific::F_NONE && t2.data.len == 130);
    assert(common::mathshim_abs(y2.data[3 * 129] - 0.7158277520683286) < 1.0e-8);
    // The values at the times of t_eval, from the dense output.
    let e3 = [1.0, 0.0, -2.001004843907636, 0.00665546563554308, -1.9976659519432036, 0.006679561733094654, -1.99297673523183, 0.0067058161092881, -1.9848951527017453, 0.006751652306705219];
    let t3, y3, f3 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1e-4, 1e-7, "BDF", 1.0e308, [0.0, 0.5, 1.0, 1.7, 2.9]) { t, v => [v[1], 100.0 * (1.0 - v[0] * v[0]) * v[1] - v[0]] };
    assert(f3.kind == scientific::F_NONE && common::all_close(y3, e3, 1.0e-8));
    // The time and the state of an event.
    let et, ey, eet, eei, eey, ef = scientific::solve_ivp_events(0.0, 3.0, [1.0, 0.0], 1, [true], [0.0], 1e-6, 1e-9, "BDF") { t, v => [v[1], 100.0 * (1.0 - v[0] * v[0]) * v[1] - v[0], v[0] - 0.5] };
    assert(ef.kind == scientific::F_NONE && et.data.len == 104);
    assert(common::mathshim_abs(eet.data[0] - 0.4154166021038258) < 1.0e-8 && common::mathshim_abs(eey.data[1] + 21.22445517558779) < 1.0e-5);
}

fn test_radau() {
    let t1, y1, f1 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 0.001, 1e-6, "Radau") { t, v => [v[1], 1000.0 * (1.0 - v[0] * v[0]) * v[1] - v[0]] };
    assert(f1.kind == scientific::F_NONE && t1.data.len == 68);
    assert(common::mathshim_abs(t1.data[67] - 3.0) < 1.0e-12 && common::mathshim_abs(t1.data[3] - 0.034402725378283136) < 1.0e-4);
    let t2, y2, f2 = scientific::solve_ivp(0.0, 40.0, [1.0, 0.0, 0.0], 1e-6, 1e-9, "Radau") { t, v => [-0.04 * v[0] + 1.0e4 * v[1] * v[2], 0.04 * v[0] - 1.0e4 * v[1] * v[2] - 3.0e7 * v[1] * v[1], 3.0e7 * v[1] * v[1]] };
    assert(f2.kind == scientific::F_NONE && t2.data.len == 67);
    assert(common::mathshim_abs(y2.data[3 * 66] - 0.715827074754968) < 1.0e-6);
    let e3 = [1.0, 0.0, -2.000925847183714, 0.006655492040691016, -1.9975905328886725, 0.006679958122998222, -1.9929053915140043, 0.006706216921244254, -1.9848305494473333, 0.006752021194119795];
    let t3, y3, f3 = scientific::solve_ivp(0.0, 3.0, [1.0, 0.0], 1e-4, 1e-7, "Radau", 1.0e308, [0.0, 0.5, 1.0, 1.7, 2.9]) { t, v => [v[1], 100.0 * (1.0 - v[0] * v[0]) * v[1] - v[0]] };
    assert(f3.kind == scientific::F_NONE && common::all_close(y3, e3, 1.0e-8));
    let et, ey, eet, eei, eey, ef = scientific::solve_ivp_events(0.0, 3.0, [1.0, 0.0], 1, [true], [0.0], 1e-6, 1e-9, "Radau") { t, v => [v[1], 100.0 * (1.0 - v[0] * v[0]) * v[1] - v[0], v[0] - 0.5] };
    assert(ef.kind == scientific::F_NONE && et.data.len == 89);
    assert(common::mathshim_abs(eet.data[0] - 0.41542002) < 1.0e-7 && common::mathshim_abs(eey.data[1] + 21.22437489) < 1.0e-5);
}
