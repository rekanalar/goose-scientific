// The tests of the group qrupdate. Run them with SCIENTIFIC_SUITES=qrupdate.
namespace qrupdate_tests;

import scientific from "../..";
import os;
import common;

fn test_qrupdate() {
    // The reduced form with a change of rank one.
    let a, fa = scientific::from_rows(4, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 1.0, 0.0, 2.0]);
    assert(fa.kind == scientific::F_NONE);
    let q, r, fq = scientific::qr(a);
    assert(fq.kind == scientific::F_NONE);
    let u1, fu1 = scientific::from_rows(4, 1, [1.0, 2.0, 3.0, 4.0]);
    let v1, fv1 = scientific::from_rows(3, 1, [2.0, -1.0, 1.0]);
    let q1, r1, f1 = scientific::qr_update(q, r, u1, v1);
    common::vchk(q1, f1, [-0.1669244652223973, 0.07467949309668298, 0.9828208985798269, -0.44513190725972585, 0.2510194574411082, -0.07293755093489812, -0.7233393492970545, 0.4273594217855335, -0.16841158108265333, -0.5007733956671914, -0.8653185135589834, -0.01951241490953357]);
    common::vchk(r1, f1, [-17.97220075561143, -3.1159233508180812, -16.63680503383225, 0.0, 6.425809028583608, 2.6706350337736326, 0.0, 0.0, 1.041358143308427]);
    // The reduced form with a change of rank two.
    let u2, fu2 = scientific::from_rows(4, 2, [1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0]);
    let v2, fv2 = scientific::from_rows(3, 2, [2.0, 1.0, -1.0, 1.0, 1.0, 0.0]);
    let q2, r2, f2 = scientific::qr_update(q, r, u2, v2);
    common::vchk(q2, f2, [-0.15833394419077804, 0.06382770743837601, 0.743355836881709, -0.47500183257233364, 0.3587396695441567, -0.5984809859100798, -0.6861137581600374, 0.3880910970523152, 0.29184792309239205, -0.5277798139692595, -0.8465324409892635, -0.06377616376555018]);
    common::vchk(r2, f2, [-18.947295321496412, -3.9055706233725225, -16.519508177237824, 0.0, 5.978839193844368, 3.091217801851273, 0.0, 0.0, 1.5969414778539894]);
    // The full form with a change of rank one.
    let b, fb = scientific::from_rows(4, 4, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0, 2.0, 1.0, 0.0, 1.0]);
    assert(fb.kind == scientific::F_NONE);
    let qb, rb, fqb = scientific::qr(b);
    assert(fqb.kind == scientific::F_NONE);
    let ub, fub = scientific::from_rows(4, 1, [1.0, -2.0, 0.5, 3.0]);
    let vb, fvb = scientific::from_rows(4, 1, [2.0, -1.0, 1.0, 0.5]);
    let q3, r3, f3 = scientific::qr_update(qb, rb, ub, vb);
    common::vchk(q3, f3, [-0.22742941307367098, 0.05197127229248562, -0.8182889316907134, -0.5253361525706526, -0.07580980435789027, -0.7048603804668386, 0.36409143506759156, -0.6040381978995892, -0.7580980435789034, -0.40764966704418526, -0.13097905907074012, 0.4918877833058545, -0.6064784348631225, 0.5781804042539042, 0.4250707438389939, -0.3423538971808746]);
    common::vchk(r3, f3, [-13.19090595827292, -6.822882392210128, -12.584427523409795, -13.115096153915028, 0.0, -10.615944416869795, -6.677496438454854, -8.656059015652382, 0.0, 0.0, -1.814724558292168, -1.8064558202248937, 0.0, 0.0, 0.0, -0.9306516860146785]);
    // The full form with a change of rank two.
    let ub2, fub2 = scientific::from_rows(4, 2, [1.0, 0.0, -2.0, 1.0, 0.5, 0.0, 3.0, 1.0]);
    let vb2, fvb2 = scientific::from_rows(4, 2, [2.0, 1.0, -1.0, 1.0, 1.0, 0.0, 0.5, 0.0]);
    let q4, r4, f4 = scientific::qr_update(qb, rb, ub2, vb2);
    common::vchk(q4, f4, [0.2153874475853214, -0.06124960325034845, 0.5750477232987226, -0.7868779126734404, 0.1435916317235475, 0.7388534817286145, -0.5227651183626076, -0.40024193540882536, 0.7179581586177382, 0.37280270324818454, 0.38208305910483475, 0.4467289590682071, 0.6461623427559642, -0.5579983540208919, -0.5000492804688117, -0.14513022020489988]);
    common::vchk(r4, f4, [13.92838827718412, 7.682152297209796, 12.492471959948645, 13.102736394773721, 0.0, 10.688055767185828, 6.43530772418131, 8.44099109045995, 0.0, 0.0, 2.962255698785856, 2.740836258273034, 0.0, 0.0, 0.0, -0.7863109977507678]);
    // A wide matrix takes the full form.
    let c, fc = scientific::from_rows(3, 5, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 2.0, 0.0, 1.0, 1.0, 3.0]);
    assert(fc.kind == scientific::F_NONE);
    let qc, rc, fqc = scientific::qr(c);
    assert(fqc.kind == scientific::F_NONE);
    let uc, fuc = scientific::from_rows(3, 1, [1.0, -2.0, 0.5]);
    let vc, fvc = scientific::from_rows(5, 1, [2.0, -1.0, 1.0, 0.5, 1.0]);
    let q5, r5, f5 = scientific::qr_update(qc, rc, uc, vc);
    common::vchk(q5, f5, [0.6396021490668317, 0.20583888637689168, 0.7406344872906016, 0.4264014327112208, -0.8966680255870075, -0.11903054260027514, 0.6396021490668313, 0.3919397973477799, -0.6612807922237514]);
    common::vchk(r5, f5, [4.69041575982343, 4.157413968934403, 6.0762204161348965, 7.515325251535267, 9.913833310535887, 0.0, -8.060143242580068, -3.9687429119928086, -6.65381249490233, -5.4631896213044895, 0.0, 0.0, 1.2564335052251274, 1.4349793191255407, 1.0580492675580029]);
    let uc2, fuc2 = scientific::from_rows(3, 2, [1.0, 0.0, -2.0, 1.0, 0.5, 0.0]);
    let vc2, fvc2 = scientific::from_rows(5, 2, [2.0, 1.0, -1.0, 1.0, 1.0, 0.0, 0.5, 0.0, 1.0, 1.0]);
    let q6, r6, f6 = scientific::qr_update(qc, rc, uc2, vc2);
    common::vchk(q6, f6, [0.577350269189626, 0.31128640318234513, 0.754829412424069, 0.5773502691896254, -0.8093446482740976, -0.10783277320343837, 0.5773502691896257, 0.4980582450917523, -0.6469966392206308]);
    common::vchk(r6, f6, [5.196152422706633, 6.0621778264910695, 6.639528095680693, 8.515916470546978, 11.258330249197702, 0.0, -8.031189202104507, -2.8638349092775757, -5.260740213781634, -4.482524205825772, 0.0, 0.0, 1.4018260516446992, 1.617491598051576, 1.1861605052378232]);
    // A rank above the size of the factors faults.
    let u3, fu3 = scientific::from_rows(4, 4, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    let v3, fv3 = scientific::from_rows(3, 4, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let g1, g2, fg1 = scientific::qr_update(q, r, u3, v3);
    assert(fg1.kind == scientific::F_VALUE);
    // A wrong row count and a wrong column count fault.
    let u4, fu4 = scientific::from_rows(3, 1, [1.0, 2.0, 3.0]);
    let g3, g4, fg2 = scientific::qr_update(q, r, u4, v1);
    assert(fg2.kind == scientific::F_SHAPE);
    let v4, fv4 = scientific::from_rows(4, 1, [1.0, 2.0, 3.0, 4.0]);
    let g5, g6, fg3 = scientific::qr_update(q, r, u1, v4);
    assert(fg3.kind == scientific::F_SHAPE);
    let g7, g8, fg4 = scientific::qr_update(q, r, u2, v1);
    assert(fg4.kind == scientific::F_SHAPE);
    let g9, g10, fg5 = scientific::qr_update(r, r, u1, v1);
    assert(fg5.kind == scientific::F_SHAPE);
    // A vector of one column gives the same result.
    let u5, fu5 = scientific::from_rows(4, 1, [1.0, 2.0, 3.0, 4.0]);
    let v5, fv5 = scientific::from_rows(3, 1, [2.0, -1.0, 1.0]);
    let u6, fu6 = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    let v6, fv6 = scientific::from_values([2.0, -1.0, 1.0]);
    let q8, r8, f8 = scientific::qr_update(q, r, u6, v6);
    common::vchk(q8, f8, [-0.1669244652223973, 0.07467949309668298, 0.9828208985798269, -0.44513190725972585, 0.2510194574411082, -0.07293755093489812, -0.7233393492970545, 0.4273594217855335, -0.16841158108265333, -0.5007733956671914, -0.8653185135589834, -0.01951241490953357]);
    common::vchk(r8, f8, [-17.97220075561143, -3.1159233508180812, -16.63680503383225, 0.0, 6.425809028583608, 2.6706350337736326, 0.0, 0.0, 1.041358143308427]);
    // A matrix with one row has no turn.
    let d, fd = scientific::from_rows(1, 3, [1.0, 2.0, 3.0]);
    let qd, rd, fqd = scientific::qr(d);
    let ud,fud = scientific::from_rows(1, 1, [2.0]);
    let vd,fvd = scientific::from_rows(3, 1, [1.0, -1.0, 0.5]);
    let q9, r9, f9 = scientific::qr_update(qd, rd, ud, vd);
    common::vchk(q9, f9, [1.0]);
    common::vchk(r9, f9, [3.0, 0.0, 4.0]);
    // The full form needs a row for each column of the change.
    let qf2, fqf2 = scientific::from_rows(6, 6, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    let rf2, frf2 = scientific::from_rows(6, 4, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let u7, fu7 = scientific::from_rows(6, 3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    let v7, fv7 = scientific::from_rows(4, 3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    let g11, g12, fg6 = scientific::qr_update(qf2, rf2, u7, v7);
    assert(fg6.kind == scientific::F_VALUE);
    // A vector of no length leaves the factors as they are.
    let uz, fuz = scientific::from_values([0.0, 0.0, 0.0, 0.0]);
    let vz, fvz = scientific::from_values([2.0, -1.0, 1.0]);
    let qz, rz, fz = scientific::qr_update(q, r, uz, vz);
    common::vchk(qz, fz, [-0.12216944435630528, 0.5642764926868793, 0.7889662677447253, -0.48867777742522095, 0.23175641663925378, -0.017339917972411356, -0.8551861104941366, -0.10076365940837123, -0.18640411820342428, -0.12216944435630524, -0.7859565433852946, 0.5852222315688907]);
    common::vchk(rz, fz, [-8.18535277187245, -9.529216659791809, -12.094774991274218, 0.0, 1.4812257933030573, 0.5038182970418574, 0.0, 0.0, 1.569262576503246]);
    // A shape that lies about its buffer is a fault.
    let lb = scientific::Arr { ndim: 2, shape: [3, 3], data: [1.0] };
    let g13, g14, fg7 = scientific::qr_update(lb, r, u1, v1);
    assert(fg7.kind == scientific::F_SHAPE);
}
