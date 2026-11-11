import scientific from "..";

// The Schur conversions and the bandwidth of a stack of matrices.
fn main() {
    // Two real Schur forms of three rows. The second one holds a
    // complex pair.
    let t = scientific::Arr { ndim: 3, shape: [2, 3, 3], data: [2.6589670819169946, 1.4244045756721777, -1.9293343948770278, 0.0, -0.32948354095849702, -0.49063703822841376, 0.0, 1.3117892147840797, -0.32948354095849702, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.0, -1.0, 4.0] };
    let z = scientific::Arr { ndim: 3, shape: [2, 3, 3], data: [0.72711590900990863, -0.60156188058076243, 0.33079564491817354, 0.52839427726760835, 0.79801891979128614, 0.28976765072380067, 0.43829435613014156, 0.035904143037891308, -0.89811410739236175, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
    let tc, zc, f = scientific::rsf2csf(t, z);
    if f.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f.kind), "\n"); return; }
    print("complex Schur form ", tc.ndim, "x", tc.shape[0], "x", tc.shape[1], "x", tc.shape[2], "\n");
    print("first row ", tc.data[0], " ", tc.data[1], "\n");
    print("second row ", tc.data[2], " ", tc.data[3], "\n");
    // The eigenvalues 2.5 with the imaginary part 0.866.
    print("second slice ", tc.data[24], " ", tc.data[25], "\n");

    // The eigenvalues 1+1j, 2 and 1-1j. The two complex values pair
    // with each other. The real value between them does not break the
    // pair.
    let w = scientific::CArr { ndim: 1, shape: [3], data: [1.0, 1.0, 2.0, 0.0, 1.0, -1.0] };
    let v = scientific::CArr { ndim: 2, shape: [3, 3], data: [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0, 9.0, 1.0] };
    let wr, vr, fw = scientific::cdf2rdf(w, v);
    if fw.kind != scientific::F_NONE { print("fault ", scientific::kind_name(fw.kind), "\n"); return; }
    print("block form\n");
    print(wr.data[0], " ", wr.data[1], " ", wr.data[2], "\n");
    print(wr.data[3], " ", wr.data[4], " ", wr.data[5], "\n");
    print(wr.data[6], " ", wr.data[7], " ", wr.data[8], "\n");

    // The counts of the diagonals of two matrices.
    let a = scientific::Arr { ndim: 3, shape: [2, 3, 4], data: [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0] };
    let lo, hi, fl = scientific::bandwidth_stack(a);
    if fl.kind != scientific::F_NONE { print("fault ", scientific::kind_name(fl.kind), "\n"); return; }
    print("lower ", lo.data[0], " ", lo.data[1], " upper ", hi.data[0], " ", hi.data[1], "\n");
    return;
}
