// The tests of the group riccati. Run them with SCIENTIFIC_SUITES=riccati.
namespace riccati_tests;

import scientific from "../..";
import os;
import common;

fn test_riccati() {
    // The stabilizing solution of both equations, plain and with the
    // matrices e and s, pinned against SciPy.
    let a, fa = scientific::from_rows(3, 3, [0.5, 0.2, 0.0, 0.1, 0.3, 0.4, 0.0, 0.2, 0.6]);
    let b, fb = scientific::from_rows(3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let q, fq = scientific::from_rows(3, 3, [2.0, 0.5, 0.0, 0.5, 1.0, 0.2, 0.0, 0.2, 3.0]);
    let r, fr = scientific::from_rows(2, 2, [2.0, 0.3, 0.3, 1.0]);
    let e, fe = scientific::from_rows(3, 3, [2.0, 0.1, 0.0, 0.1, 1.5, 0.2, 0.0, 0.2, 1.0]);
    let s, fs = scientific::from_rows(3, 2, [0.1, 0.0, 0.0, 0.2, 0.1, 0.1]);
    let ae, fae = scientific::from_rows(3, 3, [0.5, 0.2, 0.0, 0.1, 0.3, 0.4, 0.0, 0.2, 0.6]);
    let none = scientific::empty_arr();
    var x = scientific::empty_arr();
    var f = scientific::no_fault();
    x, f = scientific::solve_continuous_are(a, b, q, r);
    assert(f.kind == scientific::F_NONE);
    assert(common::all_close(x, [36.10175101039144, 27.00783575900568, -26.977219258740384, 27.00783575900568, 22.216181563972352, -21.24180192837101, -26.977219258740384, -21.24180192837101, 23.133022011530684], 1.0e-10 * 150.0));
    x, f = scientific::solve_continuous_are(a, b, q, r, e, none);
    assert(f.kind == scientific::F_NONE);
    assert(common::all_close(x, [149.43978345126962, 135.14518893434243, -139.98044807624223, 135.14518893434243, 123.65340802716335, -127.61304194058968, -139.98044807624223, -127.61304194058968, 133.95277026319263], 1.0e-10 * 150.0));
    x, f = scientific::solve_continuous_are(a, b, q, r, none, s);
    assert(f.kind == scientific::F_NONE);
    assert(common::all_close(x, [35.45705706757999, 26.53556468296663, -26.512527148759137, 26.53556468296663, 21.548017823342732, -20.842915961880816, -26.512527148759137, -20.842915961880816, 22.599826036561307], 1.0e-10 * 150.0));
    x, f = scientific::solve_continuous_are(a, b, q, r, e, s);
    assert(f.kind == scientific::F_NONE);
    assert(common::all_close(x, [141.52090729656769, 127.68501615346085, -132.4950321099197, 127.68501615346085, 116.4066495993227, -120.49865675494677, -132.4950321099197, -120.49865675494677, 126.7077716147099], 1.0e-10 * 150.0));
    x, f = scientific::solve_discrete_are(a, b, q, r);
    assert(f.kind == scientific::F_NONE);
    assert(common::all_close(x, [2.4492532557881295, 0.6971897907251743, -0.09099889531657793, 0.6971897907251743, 1.125857832649779, 0.24678662628249465, -0.09099889531657793, 0.24678662628249465, 3.2552461316199737], 1.0e-10));
    x, f = scientific::solve_discrete_are(a, b, q, r, e, none);
    assert(f.kind == scientific::F_NONE);
    assert(common::all_close(x, [0.5198365600332961, 0.13384815156438246, -0.030933420972429605, 0.13384815156438246, 0.4867613658753309, -0.3621468421540943, -0.030933420972429605, -0.3621468421540943, 3.3619282847066483], 1.0e-10));
    x, f = scientific::solve_discrete_are(a, b, q, r, none, s);
    assert(f.kind == scientific::F_NONE);
    assert(common::all_close(x, [2.4034794875346654, 0.7089889236196282, -0.11279505354235805, 0.7089889236196282, 1.06289568797544, 0.13976263689949922, -0.11279505354235805, 0.13976263689949922, 3.1588667886217174], 1.0e-10));
    x, f = scientific::solve_discrete_are(a, b, q, r, e, s);
    assert(f.kind == scientific::F_NONE);
    assert(common::all_close(x, [0.5158022081391355, 0.1382846048695044, -0.03764377861585397, 0.1382846048695044, 0.47031772421061624, -0.40670750620095997, -0.03764377861585397, -0.40670750620095997, 3.282550432047606], 1.0e-10));
    // The doc examples of SciPy.
    let a1, f1 = scientific::from_rows(2, 2, [4.0, 3.0, -4.5, -3.5]);
    let b1, f2 = scientific::from_rows(2, 1, [1.0, -1.0]);
    let q1, f3 = scientific::from_rows(2, 2, [9.0, 6.0, 6.0, 4.0]);
    let r1, f4 = scientific::from_rows(1, 1, [1.0]);
    x, f = scientific::solve_continuous_are(a1, b1, q1, r1);
    assert(f.kind == scientific::F_NONE);
    assert(common::all_close(x, [21.72792206135783, 14.485281374238562, 14.485281374238562, 9.656854249492381], 1.0e-9));
    let a2, f5 = scientific::from_rows(2, 2, [0.0, 1.0, 0.0, -1.0]);
    let b2, f6 = scientific::from_rows(2, 2, [1.0, 0.0, 2.0, 1.0]);
    let q2, f7 = scientific::from_rows(2, 2, [-4.0, -4.0, -4.0, 7.0]);
    let r2, f8 = scientific::from_rows(2, 2, [9.0, 3.0, 3.0, 1.0]);
    // SciPy accepts this singular r for the discrete equation, and so
    // does the package.
    x, f = scientific::solve_discrete_are(a2, b2, q2, r2);
    assert(f.kind == scientific::F_NONE);
    assert(common::all_close(x, [-4.0, -4.0, -4.0, 7.0], 1.0e-8));
    // Faults: a shape that does not fit, a matrix that is not
    // symmetric, a singular r, a value that is not finite.
    let bad, fbad = scientific::from_rows(2, 3, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    x, f = scientific::solve_continuous_are(a, bad, q, r);
    assert(f.kind == scientific::F_SHAPE);
    let qn, fqn = scientific::from_rows(3, 3, [2.0, 0.5, 0.0, 0.0, 1.0, 0.2, 0.0, 0.2, 3.0]);
    x, f = scientific::solve_discrete_are(a, b, qn, r);
    assert(f.kind == scientific::F_VALUE);
    let rs, frs = scientific::from_rows(2, 2, [1.0, 1.0, 1.0, 1.0]);
    x, f = scientific::solve_continuous_are(a, b, q, rs);
    assert(f.kind == scientific::F_VALUE);
    let inf = 1.0 / 0.0;
    let ai, fai = scientific::from_rows(3, 3, [0.5, inf, 0.0, 0.1, 0.3, 0.4, 0.0, 0.2, 0.6]);
    x, f = scientific::solve_discrete_are(ai, b, q, r);
    assert(f.kind == scientific::F_VALUE);
    // Hostile operands: a buffer shorter than the shape, a count of
    // dimensions that lies, a side beyond the limit, a NaN.
    let lb = scientific::Arr { ndim: 2, shape: [3, 3], data: [1.0] };
    x, f = scientific::solve_continuous_are(lb, b, q, r);
    assert(f.kind == scientific::F_SHAPE);
    let lie = scientific::Arr { ndim: 3, shape: [3, 3], data: [1.0] };
    x, f = scientific::solve_discrete_are(a, b, lie, r);
    assert(f.kind == scientific::F_SHAPE);
    let big = scientific::Arr { ndim: 2, shape: [4294967296, 4294967296], data: [1.0] };
    x, f = scientific::solve_continuous_are(big, b, q, r);
    assert(f.kind == scientific::F_LIMIT);
    let nan = 0.0 / 0.0;
    let an, fan = scientific::from_rows(3, 3, [0.5, nan, 0.0, 0.1, 0.3, 0.4, 0.0, 0.2, 0.6]);
    x, f = scientific::solve_continuous_are(an, b, q, r);
    assert(f.kind == scientific::F_VALUE);
    let sb, fsb = scientific::from_rows(2, 2, [1.0, 0.0, 0.0, 1.0]);
    x, f = scientific::solve_continuous_are(a, b, q, r, e, sb);
    assert(f.kind == scientific::F_SHAPE);
    // A pair with no stabilizing solution: a is zero, b is zero.
    let zz, fzz = scientific::from_rows(1, 1, [0.0]);
    let one, fone = scientific::from_rows(1, 1, [1.0]);
    x, f = scientific::solve_continuous_are(zz, zz, one, one);
    assert(f.kind == scientific::F_VALUE);
    let es2, fes2 = scientific::from_rows(3, 3, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    x, f = scientific::solve_discrete_are(a, b, q, r, es2, none);
    assert(f.kind == scientific::F_VALUE);
}
