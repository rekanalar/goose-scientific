import scientific from "..";

// A CountSketch shrinks a tall matrix to a few rows. The norm of a
// vector stays near its value in expectation.
fn main() {
    var data: f64[>..] = [];
    var i: i64 = 0;
    while i < 200 * 3 { data.push(((i * 7) % 11) as f64 - 5.0); i += 1; }
    let a, fa = scientific::from_rows(200, 3, data);
    assert(fa.kind == scientific::F_NONE);
    let sk, f = scientific::clarkson_woodruff_transform(a, 40, 7);
    if f.kind != scientific::F_NONE {
        print("fault: ", f.detail, "\n");
        return;
    }
    print("sketch shape: ", sk.shape[0], " by ", sk.shape[1], "\n");
    var n0: f64 = 0.0;
    var n1: f64 = 0.0;
    i = 0;
    while i < 200 { n0 += a.data[i * 3] * a.data[i * 3]; i += 1; }
    i = 0;
    while i < 40 { n1 += sk.data[i * 3] * sk.data[i * 3]; i += 1; }
    print("squared norm of column 0, input and sketch: ", n0, " ", n1, "\n");

    // A pad with one pair of counts per axis and its own end values.
    let m, fm = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let p, fp = scientific::pad_axes(m, [1, 0], [1, 2], "linear_ramp", [], [0.0, 10.0, 0.0, 20.0]);
    print("padded shape: ", p.shape[0], " by ", p.shape[1], " fault ", fp.kind, "\n");
    // The mode empty gives zero in the pad.
    let z, fz = scientific::pad_axes(m, [1, 1], [1, 1], "empty");
    print("empty pad corner: ", z.data[0], "\n");

    // A singular weight r in the discrete Riccati equation.
    let a2, f1 = scientific::from_rows(2, 2, [0.9, 0.5, 0.0, 0.7]);
    let b2, f2 = scientific::from_rows(2, 2, [1.0, 0.0, 0.0, 1.0]);
    let r2, f3 = scientific::from_rows(2, 2, [1.0, 1.0, 1.0, 1.0]);
    let x, fx = scientific::solve_discrete_are(a2, b2, b2, r2);
    print("x00 = ", x.data[0], " fault ", fx.kind, "\n");
}
