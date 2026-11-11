import scientific from "..";

// The cosine-sine decomposition of an orthogonal matrix.
fn main() {
    // A matrix of four rows. It is orthogonal.
    let x = scientific::Arr { ndim: 2, shape: [4, 4], data: [-0.05047916295569932, 0.12468065699085004, -0.44557229035821955, 0.8850830028559764, 0.21506477395034093, -0.370571330318633, -0.8313138415261359, -0.35403577367072037, 0.2825411849039954, 0.9024201593252053, -0.23212987916874345, -0.22786850171452355, 0.9334717328050309, -0.1810234210307588, 0.23769381596729314, 0.19840034007833807] };

    // The angles of the split of two rows and two columns.
    let u1, u2, theta, v1h, v2h, f = scientific::cossin_separate(x, 2, 2);
    if f.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f.kind), "\n"); return; }
    print("angles ", theta.data[0], " ", theta.data[1], "\n");
    print("left block of side ", u1.shape[0], " right block of side ", v2h.shape[0], "\n");

    // The three factors multiply back to the matrix.
    let u, cs, vh, g = scientific::cossin(x, 2, 2);
    if g.kind != scientific::F_NONE { print("fault ", scientific::kind_name(g.kind), "\n"); return; }
    let uc, g1 = scientific::matmul(u, cs);
    let prod, g2 = scientific::matmul(uc, vh);
    var err: f64 = 0.0;
    var i: i64 = 0;
    while i < 16 {
        var d = prod.data[i] - x.data[i];
        if d < 0.0 { d = -d; }
        if d > err { err = d; }
        i += 1;
    }
    print("product error ", err, "\n");

    // A matrix that is not orthogonal gives a fault.
    let bad = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0, 1.0, 1.0, 1.0] };
    let bu, bcs, bvh, bf = scientific::cossin(bad, 1, 1);
    print("fault ", scientific::kind_name(bf.kind), "\n");
    return;
}
