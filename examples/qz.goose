// The generalized Schur form of a pair of matrices.
import scientific from "..";
import os;

fn main() {
    let a = scientific::Arr { ndim: 2, shape: [4, 4], data: [2.0, 5.0, 8.0, 7.0, 5.0, 2.0, 2.0, 8.0, 7.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 8.0] };
    let b = scientific::Arr { ndim: 2, shape: [4, 4], data: [0.0, 6.0, 0.0, 0.0, 5.0, 0.0, 2.0, 1.0, 5.0, 2.0, 6.0, 6.0, 4.0, 7.0, 7.0, 7.0] };

    // The eigenvalues of the pair come as alpha / beta. The rule lhp
    // puts the eigenvalues with a negative real part first.
    let aa, bb, alpha, beta, q, z, f = scientific::ordqz(a, b, "lhp");
    if f.kind != scientific::F_NONE {
        print("fault: ", f.detail, "\n");
        return;
    }
    var i: i64 = 0;
    while i < 4 {
        print("eigenvalue ", i, ": ", alpha.data[2 * i] / beta.data[i], "\n");
        i += 1;
    }

    // The product Q AA Z^T gives A back.
    var worst: f64 = 0.0;
    var r: i64 = 0;
    while r < 4 {
        var c: i64 = 0;
        while c < 4 {
            var s: f64 = 0.0;
            var k: i64 = 0;
            while k < 4 {
                var l: i64 = 0;
                while l < 4 {
                    s += q.data[r * 4 + k] * aa.data[k * 4 + l] * z.data[c * 4 + l];
                    l += 1;
                }
                k += 1;
            }
            var d = s - a.data[r * 4 + c];
            if d < 0.0 { d = -d; }
            if d > worst { worst = d; }
            c += 1;
        }
        r += 1;
    }
    print("largest error of Q AA Z^T - A: ", worst, "\n");

    // The triangular complex form of a pair with complex eigenvalues.
    let rot = scientific::Arr { ndim: 2, shape: [2, 2], data: [0.0, 1.0, -1.0, 0.0] };
    let eye = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0, 0.0, 0.0, 1.0] };
    let ca, cb, cq, cz, cf = scientific::qz_complex(rot, eye);
    print("complex form, fault kind ", cf.kind, ", AA[0][0] = ", ca.data[0], " + ", ca.data[1], "i\n");

    // A value that is not finite is a fault.
    let bad = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0, 0.0 / 0.0, 3.0, 4.0] };
    let x1, x2, x3, x4, xf = scientific::qz(bad, eye);
    print("a NaN gives the fault: ", xf.detail, "\n");

    // A complex pair gives the eigenvalues of the pair.
    let pa = scientific::CArr { ndim: 2, shape: [4, 4], data: [2.0, 1.0, 5.0, -1.0, 8.0, 0.0, 7.0, 2.0, 5.0, 0.0, 2.0, 3.0, 2.0, 0.0, 8.0, -1.0, 7.0, -2.0, 5.0, 0.0, 6.0, 1.0, 6.0, 0.0, 5.0, 0.0, 4.0, 1.0, 4.0, 0.0, 8.0, -2.0] };
    let pb = scientific::CArr { ndim: 2, shape: [4, 4], data: [0.0, 1.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 2.0, -1.0, 1.0, 0.0, 5.0, 1.0, 2.0, 0.0, 6.0, 0.0, 6.0, -1.0, 4.0, 0.0, 7.0, 0.0, 0.0, 7.0, 7.0, 0.0] };
    let caa, cbb, cal, cbe, cq, cz, cf2 = scientific::ordqz_c(copy(pa), copy(pb), "lhp");
    if cf2.kind != scientific::F_NONE {
        print("fault: ", cf2.detail, "\n");
        return;
    }
    i = 0;
    while i < 4 {
        print("complex eigenvalue ", i, ": ", cal.data[2 * i] / cbe.data[i], " ", cal.data[2 * i + 1] / cbe.data[i], "i\n");
        i += 1;
    }

    // A block of the caller chooses the eigenvalues that come first. It
    // takes the alpha and the beta of a block.
    let ba, bb2, bal, bbe, bq, bz, bf = scientific::ordqz_by(a, b) { al, be => al.re / be < 0.0 };
    if bf.kind != scientific::F_NONE {
        print("fault: ", bf.detail, "\n");
        return;
    }
    print("first eigenvalue of the block: ", bal.data[0] / bbe.data[0], "\n");

    // A pair at a scale near the limit of the number format keeps its
    // eigenvalues.
    let sa = scientific::Arr { ndim: 2, shape: [4, 4], data: [2.0e300, 5.0e300, 8.0e300, 7.0e300, 5.0e300, 2.0e300, 2.0e300, 8.0e300, 7.0e300, 5.0e300, 6.0e300, 6.0e300, 5.0e300, 4.0e300, 4.0e300, 8.0e300] };
    let sb = scientific::Arr { ndim: 2, shape: [4, 4], data: [0.0, 6.0e300, 0.0, 0.0, 5.0e300, 0.0, 2.0e300, 1.0e300, 5.0e300, 2.0e300, 6.0e300, 6.0e300, 4.0e300, 7.0e300, 7.0e300, 7.0e300] };
    let sqa, sqb, sqq, sqz, sqf = scientific::qz(sa, sb);
    if sqf.kind != scientific::F_NONE {
        print("fault: ", sqf.detail, "\n");
        return;
    }
    var sworst: f64 = 0.0;
    r = 0;
    while r < 4 {
        var c: i64 = 0;
        while c < 4 {
            var s: f64 = 0.0;
            var k: i64 = 0;
            while k < 4 {
                var l: i64 = 0;
                while l < 4 {
                    s += sqq.data[r * 4 + k] * sqa.data[k * 4 + l] * sqz.data[c * 4 + l];
                    l += 1;
                }
                k += 1;
            }
            var d = s - sa.data[r * 4 + c];
            if d < 0.0 { d = -d; }
            if d > sworst { sworst = d; }
            c += 1;
        }
        r += 1;
    }
    print("largest relative error at the scale 1e300: ", sworst / 8.0e300, "\n");
}
