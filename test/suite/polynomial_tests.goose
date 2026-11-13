// The tests of the group polynomial. Run them with SCIENTIFIC_SUITES=polynomial.
namespace polynomial_tests;

import scientific from "../..";
import os;
import common;

fn test_polynomial() {
    // The values come from numpy.polynomial.Polynomial.
    let p = scientific::polynomial([1.0, 2.0, 3.0]);
    assert(scientific::polynomial_call(p, 2.0) == 17.0);
    assert(scientific::polynomial_degree(p) == 2);
    assert(scientific::polynomial_degree(scientific::polynomial([1.0, 2.0, 0.0])) == 1);
    let q = scientific::polynomial([1.0, 1.0]);
    let a = scientific::polynomial_add(p, q);
    let b = scientific::polynomial_sub(p, q);
    let m = scientific::polynomial_mul(p, q);
    assert(a.coe[0] == 2.0 && a.coe[2] == 3.0);
    assert(b.coe[0] == 0.0 && b.coe[2] == 3.0);
    assert(m.coe.len == 4 && m.coe[1] == 3.0 && m.coe[3] == 3.0);
    let quo, rem, f = scientific::polynomial_divmod(scientific::polynomial([1.0, 3.0, 5.0, 3.0]), q);
    assert(f.kind == scientific::F_NONE);
    assert(quo.coe.len == 3 && common::close(quo.coe[2], 3.0) && rem.coe.len == 1 && rem.coe[0] == 0.0);
    let d1, f1 = scientific::polynomial_der(p, 1);
    let d2, f2 = scientific::polynomial_der(p, 2);
    assert(f1.kind == scientific::F_NONE && d1.coe.len == 2 && d1.coe[1] == 6.0);
    assert(d2.coe.len == 1 && d2.coe[0] == 6.0);
    let ig = scientific::polynomial_integ(p);
    assert(ig.coe.len == 4 && ig.coe[0] == 0.0 && common::close(ig.coe[3], 1.0));
    let rp, rpi, fr = scientific::polynomial_roots(scientific::polynomial([-1.0, 0.0, 1.0]));
    assert(fr.kind == scientific::F_NONE);
    assert((common::close(rp.data[0], -1.0) && common::close(rp.data[1], 1.0)) || (common::close(rp.data[0], 1.0) && common::close(rp.data[1], -1.0)));
    assert(common::close(rpi.data[0], 0.0) && common::close(rpi.data[1], 0.0));
    let frp = scientific::polynomial_fromroots([1.0, -1.0]);
    assert(common::close(frp.coe[0], -1.0) && frp.coe[1] == 0.0 && common::close(frp.coe[2], 1.0));
    let pw, f3 = scientific::polynomial_pow(q, 2);
    let pw3, f4 = scientific::polynomial_pow(scientific::polynomial([0.0, 1.0]), 3);
    assert(f3.kind == scientific::F_NONE && pw.coe.len == 3 && common::close(pw.coe[1], 2.0));
    assert(pw3.coe.len == 4 && common::close(pw3.coe[3], 1.0));
    let bad, f5 = scientific::polynomial_pow(p, -1);
    assert(f5.kind == scientific::F_RANGE);
    let cd, f6 = scientific::polynomial_cutdeg(scientific::polynomial([1.0, 2.0, 3.0, 4.0]), 1);
    assert(f6.kind == scientific::F_NONE && cd.coe.len == 2 && cd.coe[1] == 2.0);
    let fp, f7 = scientific::polynomial_fit([0.0, 1.0, 2.0], [1.0, 3.0, 7.0], 2);
    assert(f7.kind == scientific::F_NONE);
    assert(common::close(fp.coe[0], 1.0) && common::close(fp.coe[1], 1.0) && common::close(fp.coe[2], 1.0));
    let xs, ys, f8 = scientific::polynomial_linspace(scientific::polynomial([0.0, 1.0]), -1.0, 1.0, 5);
    assert(f8.kind == scientific::F_NONE && xs.data.len == 5);
    assert(common::close(xs.data[1], -0.5) && common::close(ys.data[1], -0.5) && common::close(ys.data[4], 1.0));
    let noq, nor, f9 = scientific::polynomial_divmod(p, scientific::polynomial([0.0]));
    assert(f9.kind == scientific::F_VALUE);
    let noc, f10 = scientific::polynomial_cutdeg(p, -1);
    assert(f10.kind == scientific::F_RANGE);
}
