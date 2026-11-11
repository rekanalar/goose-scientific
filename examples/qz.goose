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
}
