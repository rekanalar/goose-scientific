// The tests of the group qrtools. Run them with SCIENTIFIC_SUITES=qrtools.
namespace qrtools_tests;

import scientific from "../..";
import os;
import common;

fn test_qrtools() {
    let a, fa = scientific::from_rows(4, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 1.0, 0.0, 2.0]);
    let c1, fc1 = scientific::from_rows(3, 2, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let ql, rl, fl = scientific::qr_multiply(a, c1, "left");
    common::vchk(ql, fl, [4.28441805635266, 5.5154913724279595, 0.39415316138886203, 0.11989188263048334, -0.9471437916304399, -2.0894976797363722, 0.768975839504974, 0.4460720833322649]);
    common::vchk(rl, fl, [-8.18535277187245, -9.529216659791809, -12.094774991274218, 0.0, 1.4812257933030573, 0.5038182970418574, 0.0, 0.0, 1.569262576503246]);
    let c2, fc2 = scientific::from_rows(2, 4, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let qr2, rr2, fr = scientific::qr_multiply(a, c2);
    common::vchk(qr2, fr, [-2.5655583314824097, -2.327640532333373, 1.365518540327412, -8.91836943801028, -2.6903897062035056, 6.0472963928785335]);
    // A wrong row count, a wrong column count and a bad mode fault.
    let b1, b2, bf1 = scientific::qr_multiply(a, c2, "left");
    assert(bf1.kind == scientific::F_SHAPE);
    let b3, b4, bf2 = scientific::qr_multiply(a, c1);
    assert(bf2.kind == scientific::F_SHAPE);
    let b5, b6, bf3 = scientific::qr_multiply(a, c1, "up");
    assert(bf3.kind == scientific::F_VALUE);
    // The DFT matrix with the three scales.
    let d3, fd = scientific::dft(3, "n");
    assert(fd.kind == scientific::F_NONE && d3.data.len == 18);
    let want = [0.3333333333333333, 0.0, 0.3333333333333333, 0.0, 0.3333333333333333, 0.0, 0.3333333333333333, 0.0, -0.16666666666666657, -0.28867513459481287, -0.16666666666666677, 0.28867513459481275, 0.3333333333333333, 0.0, -0.1666666666666668, 0.28867513459481275, -0.16666666666666635, -0.28867513459481303];
    var i: i64 = 0;
    while i < 18 { assert(common::mathshim_abs(d3.data[i] - want[i]) < 1.0e-12); i += 1; }
    let d4, fd4 = scientific::dft(4, "sqrtn");
    assert(fd4.kind == scientific::F_NONE && common::mathshim_abs(d4.data[2 * 5 + 1] + 0.5) < 1.0e-12);
    let d5, fd5 = scientific::dft(2);
    assert(fd5.kind == scientific::F_NONE && common::mathshim_abs(d5.data[6] + 1.0) < 1.0e-12);
    let e1, fe1 = scientific::dft(0);
    assert(fe1.kind == scientific::F_VALUE);
    let e2, fe2 = scientific::dft(3, "x");
    assert(fe2.kind == scientific::F_VALUE);
    let e3, fe3 = scientific::dft(5000);
    assert(fe3.kind == scientific::F_LIMIT);
}
