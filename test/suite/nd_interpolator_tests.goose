// The tests of the group nd_interpolator. Run them with SCIENTIFIC_SUITES=nd_interpolator.
namespace nd_interpolator_tests;

import scientific from "../..";
import os;
import common;
import common_3;

// test_nd_interpolator checks the object form against the pinned values.
fn test_nd_interpolator() {
    let p = common::mx(5, 2, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5]);
    let col = common_3::nd_line([1.0, 2.0, 3.0, 4.0, 5.0]);
    let xi = common::mx(4, 2, [0.25, 0.25, 0.75, 0.5, 2.0, 2.0, 0.5, 0.1]);
    let it, f = scientific::nd_interpolator(p, col, "linear", -1.0);
    assert(f.kind == scientific::F_NONE);
    let r1, f1 = scientific::nd_interpolate(it, xi);
    assert(f1.kind == scientific::F_NONE && common_3::nd_same(r1, [3.0, 4.0, -1.0, 2.2]));
    // The interpolator serves a second call with the same result.
    let r2, f2 = scientific::nd_interpolate(it, xi);
    assert(f2.kind == scientific::F_NONE && common_3::nd_same(r2, [3.0, 4.0, -1.0, 2.2]));
    let half = common::mx(2, 2, [0.25, 0.25, 0.75, 0.5]);
    let r3, f3 = scientific::nd_interpolate(it, half);
    assert(f3.kind == scientific::F_NONE && common_3::nd_same(r3, [3.0, 4.0]));
    // The method nearest in three dimensions.
    let p3 = common::mx(8, 3, [0.6842052123845983, 0.46382436008336725, 0.2218885591644547, 0.640940141481311, 0.10708949613728647, 0.6922222746448299, 0.6353867998907108, 0.3765125255876922, 0.7985233458061055, 0.19402547600704456, 0.3904589137977216, 0.797933886097021, 0.3804753706433236, 0.7132578641244521, 0.6125178041653119, 0.9410009752426018, 0.9916767169901963, 0.7236762546507963, 0.8088438090346699, 0.15286502314567285, 0.7128902625665684, 0.8476243802545339, 0.40122574799263144, 0.5532500397935409]);
    let v3 = common_3::nd_line([0.4794877517608621, 0.9585229997993715, 0.3172778402767157, 0.40208452693776586, 0.0009197891108657652, 0.42018453469313044, 0.6314361061322901, 0.9349610941866503]);
    let q3 = common::mx(4, 3, [0.9236765847577346, 0.32734981332409785, 0.9888610256190192, 0.1876749402345188, 0.8232520180360315, 0.15725948708404458, 0.40509865286115954, 0.07347331180727756, 0.8580389311980173, 0.8287980709765224, 0.13978929854271627, 0.5271064765570825]);
    let it3, g3 = scientific::nd_interpolator(p3, v3, "nearest");
    assert(g3.kind == scientific::F_NONE);
    let r4, f4 = scientific::nd_interpolate(it3, q3);
    assert(f4.kind == scientific::F_NONE && common_3::nd_same(r4, [0.6314361061322901, 0.0009197891108657652, 0.9585229997993715, 0.6314361061322901]));
    // The method linear stays at one or two dimensions.
    let l5, g5 = scientific::nd_interpolator(p3, v3, "linear");
    assert(g5.kind == scientific::F_LIMIT);
    let l6, g6 = scientific::nd_interpolator(p, col, "cubic");
    assert(g6.kind == scientific::F_VALUE);
    // A point to find of another dimension.
    let one = common_3::nd_line([0.5, 0.5]);
    let r7, f7 = scientific::nd_interpolate(it, one);
    assert(f7.kind == scientific::F_SHAPE);
    let r8, f8 = scientific::nd_interpolate(it3, xi);
    assert(f8.kind == scientific::F_SHAPE);
    let nan = 0.0 / 0.0;
    let badq = common::mx(1, 2, [nan, 0.5]);
    let r9, f9 = scientific::nd_interpolate(it, badq);
    assert(f9.kind == scientific::F_VALUE);
    // A shape that lies about its size gives a fault, not a wrapped product.
    let lie = scientific::Arr { ndim: 2, shape: [4611686018427387904, 2], data: [] };
    let r11, f11 = scientific::nd_interpolate(it, lie);
    assert(f11.kind == scientific::F_LIMIT);
    let lie2 = scientific::Arr { ndim: 2, shape: [4, 4611686018427387904], data: [] };
    let l12, g12 = scientific::nd_interpolator(lie2, col, "nearest");
    assert(g12.kind == scientific::F_LIMIT || g12.kind == scientific::F_SHAPE);
    // Points that have no triangulation.
    let line = common::mx(3, 2, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    let l10, g10 = scientific::nd_interpolator(line, common_3::nd_line([1.0, 2.0, 3.0]), "linear");
    assert(g10.kind != scientific::F_NONE);
    // Twenty random sets give the same result from the object and from griddata.
    var rng = scientific::default_rng(300);
    var round: i64 = 0;
    while round < 20 {
        let rp, fr = scientific::randn(rng, [15, 2]);
        let rv, fv = scientific::randn(rng, [15, 3]);
        let rq, fq = scientific::randn(rng, [7, 2]);
        assert(fr.kind == scientific::F_NONE && fv.kind == scientific::F_NONE && fq.kind == scientific::F_NONE);
        let ob, fo = scientific::nd_interpolator(rp, rv, "linear", -9.0);
        assert(fo.kind == scientific::F_NONE);
        let a1, e1 = scientific::nd_interpolate(ob, rq);
        let a2, e2 = scientific::griddata(rp, rv, rq, "linear", -9.0);
        assert(e1.kind == scientific::F_NONE && e2.kind == scientific::F_NONE);
        assert(common_3::nd_same(a1, a2.data));
        round += 1;
    }
    return;
}
