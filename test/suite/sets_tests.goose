// The tests of the group sets. Run them with SCIENTIFIC_SUITES=sets.
namespace sets_tests;

import scientific from "../..";
import os;

// The sets of numbers hold the values of NumPy.
fn test_sets() {
    let a, fa = scientific::from_values([3.0, 1.0, 2.0, 2.0]);
    let b, fb = scientific::from_values([2.0, 3.0, 4.0]);
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    // The union holds each value once, in the order of its value.
    let u, fu = scientific::union1d(a, b);
    assert(fu.kind == scientific::F_NONE && u.data.len == 4);
    assert(u.data[0] == 1.0 && u.data[1] == 2.0 && u.data[2] == 3.0 && u.data[3] == 4.0);
    // The common values of two rows.
    let i2, fi2 = scientific::intersect1d(a, b);
    assert(fi2.kind == scientific::F_NONE && i2.data.len == 2);
    assert(i2.data[0] == 2.0 && i2.data[1] == 3.0);
    // The values of the first row that the second holds no value of.
    let s, fs = scientific::setdiff1d(a, b);
    assert(fs.kind == scientific::F_NONE && s.data.len == 1 && s.data[0] == 1.0);
    // The membership of a flat row.
    let q, fq = scientific::in1d(a, b);
    assert(fq.kind == scientific::F_NONE && q.data.len == 4);
    assert(q.data[0] == 1.0 && q.data[1] == 0.0 && q.data[2] == 1.0 && q.data[3] == 1.0);
    // The membership with the answer turned around.
    let qi, fqi = scientific::in1d(a, b, true);
    assert(fqi.kind == scientific::F_NONE && qi.data[0] == 0.0 && qi.data[3] == 0.0);
    // The membership keeps the shape of the array.
    let m, fm = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let e, fe = scientific::isin(m, b);
    assert(fe.kind == scientific::F_NONE && e.ndim == 2 && e.data.len == 4);
    assert(e.data[0] == 0.0 && e.data[1] == 1.0 && e.data[2] == 1.0 && e.data[3] == 1.0);
    // A NaN holds no value, not even another NaN.
    let nan = 0.0 / 0.0;
    let na, fna = scientific::from_values([1.0, nan, nan, 2.0]);
    let nb, fnb = scientific::from_values([nan, 2.0, 3.0]);
    assert(fna.kind == scientific::F_NONE && fnb.kind == scientific::F_NONE);
    let nu, fnu = scientific::union1d(na, nb);
    assert(fnu.kind == scientific::F_NONE && nu.data.len == 4);
    assert(nu.data[0] == 1.0 && nu.data[1] == 2.0 && nu.data[2] == 3.0);
    assert(nu.data[3] != nu.data[3]);
    let ni, fni = scientific::intersect1d(na, nb);
    assert(fni.kind == scientific::F_NONE && ni.data.len == 1 && ni.data[0] == 2.0);
    let ns, fns = scientific::setdiff1d(na, nb);
    assert(fns.kind == scientific::F_NONE && ns.data.len == 2);
    assert(ns.data[0] == 1.0 && ns.data[1] != ns.data[1]);
    let nq, fnq = scientific::in1d(na, nb);
    assert(fnq.kind == scientific::F_NONE && nq.data[0] == 0.0);
    assert(nq.data[1] == 0.0 && nq.data[2] == 0.0 && nq.data[3] == 1.0);
}
