import scientific from "..";

// The integrators of the package: the methods of the solver of the
// equations, the solver of the boundary value problems, the cubature
// of a box and the rules of the sphere.
fn main() {
    // The method RK23 takes the steps of SciPy.
    let t, y, f = scientific::solve_ivp(0.0, 1.0, [1.0, 0.0], 1.0e-8, 1.0e-10, "RK23", 1.0e308, [0.0, 0.5, 1.0], 0.0) { tt, yy =>
        var out: f64[>..] = [];
        out.push(yy[1]);
        out.push(0.0 - yy[0]);
        out
    };
    assert(f.kind == scientific::F_NONE);
    print("RK23: ", y.data[4], " ", y.data[5], " fault ", f.kind, "\n");
    // The method LSODA runs the Adams method and the BDF method.
    let t2, y2, f2 = scientific::solve_ivp(0.0, 2.0, [1.0, 0.0], 1.0e-8, 1.0e-10, "LSODA", 1.0e308, [0.0, 1.0, 2.0], 0.0) { tt, yy =>
        var out: f64[>..] = [];
        out.push(998.0 * yy[0] + 1998.0 * yy[1]);
        out.push(-999.0 * yy[0] - 1999.0 * yy[1]);
        out
    };
    assert(f2.kind == scientific::F_NONE);
    print("LSODA: ", y2.data[4], " ", y2.data[5], " fault ", f2.kind, "\n");
    // The solver of the boundary value problems refines the mesh.
    let guess, fg = scientific::from_rows(5, 2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert(fg.kind == scientific::F_NONE);
    let bx, by, bp, br, bf = scientific::solve_bvp([0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345, 3.141592653589793], copy(guess), [], 1.0e-6, 1000, bvp_bc) { xx, yy, pp =>
        var out: f64[>..] = [];
        out.push(yy[1]);
        out.push(0.0 - yy[0]);
        out
    };
    assert(bf.kind == scientific::F_NONE);
    print("bvp: nodes ", bx.data.len, " value ", by.data[0], " fault ", bf.kind, "\n");
    // The cubature integrates over a box.
    let v, e, cf = scientific::cubature([0.0, 0.0], [1.0, 1.0], 1000) { x => x[0] * x[1] };
    assert(cf.kind == scientific::F_NONE);
    print("cubature: ", v, " error ", e, " fault ", cf.kind, "\n");
    // The rules of the sphere give the points and the weights.
    let px, pw, pf = scientific::lebedev_rule(9);
    assert(pf.kind == scientific::F_NONE);
    var sum: f64 = 0.0;
    var i: i64 = 0;
    while i < pw.data.len { sum += pw.data[i]; i += 1; }
    print("lebedev: points ", px.shape[1], " sum of the weights ", sum, " fault ", pf.kind, "\n");
}

// bvp_bc gives the conditions of the oscillator at the two ends.
fn bvp_bc(ya: f64[:], yb: f64[:], p: f64[:]) -> f64[>..] {
    var out: f64[>..] = [];
    out.push(ya[0] - 1.0);
    out.push(yb[0] + 1.0);
    return out;
}
