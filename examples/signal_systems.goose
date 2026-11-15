// The systems of a signal: the forms of a system, the walk of an
// input through a system, the gain of a closed loop, the forms of a
// cascade, the design by the least squares and the unit impulse.
import std;
import scientific from "..";

fn show(label: const u8[:], a: scientific::Arr&) {
    print(label);
    var i: i64 = 0;
    while i < a.data.len { print(" ", a.data[i]); i += 1; }
    print("\n");
}

fn show_list(label: const u8[:], v: f64[>..]&) {
    print(label);
    var i: i64 = 0;
    while i < v.len { print(" ", v[i]); i += 1; }
    print("\n");
}

fn main() {
    // lti puts the transfer function in the form of SciPy: the
    // denominator starts at one and the small coefficients of the
    // numerator leave.
    let sys, f1 = scientific::lti([1.0e-15, 1.0, 2.0], [1.0, 3.0, 2.0]);
    assert(f1.kind == scientific::F_NONE);
    // The transfer function holds its coefficients in one buffer:
    // the numerator first, then the denominator.
    var nv: f64[>..] = [];
    var dv: f64[>..] = [];
    var ti: i64 = 0;
    while ti < sys.num_len { nv.push(sys.buf[ti]); ti += 1; }
    while ti < sys.num_len + sys.den_len { dv.push(sys.buf[ti]); ti += 1; }
    show_list("numerator:", nv);
    show_list("denominator:", dv);
    print("dt: ", sys.dt, "\n");

    // dlsim walks an input through a discrete state space system.
    let am = scientific::from_rows(2, 2, [0.0, 1.0, -2.0, -3.0]);
    let bm = scientific::from_rows(2, 1, [0.0, 1.0]);
    let cm = scientific::from_rows(1, 2, [1.0, 0.0]);
    let dm = scientific::from_rows(1, 1, [0.0]);
    let dsys, f1b = scientific::state_space(am, bm, cm, dm, 0.1);
    assert(f1b.kind == scientific::F_NONE);
    let tout, yout, xout, f2 = scientific::dlsim(dsys, [1.0, 0.0, 1.0, 0.0, 1.0]);
    assert(f2.kind == scientific::F_NONE);
    show("times:", tout);
    show("outputs:", yout);

    // place_poles gives the gain that puts the poles of the closed
    // loop on the places the caller asks for.
    let pa = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let g, cr, ci, rr, ri, _x, rtol, iters, f3 =
        scientific::place_poles(pa, [1.0, 0.0, 0.0, 1.0], [-1.0, -2.0], [0.0, 0.0]);
    assert(f3.kind == scientific::F_NONE);
    show("gain:", g);
    show("poles of the closed loop:", cr);
    show("imaginary parts:", ci);

    // sos2tf joins the sections of a cascade into one transfer
    // function.
    let sos = scientific::from_rows(2, 6,
        [1.0, 2.0, 0.0, 1.0, 0.0, 0.0,
         2.0, -1.0, 0.5, 1.0, -0.9, 0.2]);
    let fb, fa, f4 = scientific::sos2tf(sos);
    assert(f4.kind == scientific::F_NONE);
    show("numerator of the cascade:", fb);
    show("denominator of the cascade:", fa);

    // firls gives the taps of the filter of the least squares error.
    let taps, f5 = scientific::firls(11, [0.0, 0.2, 0.3, 0.5], [1.0, 1.0, 0.0, 0.0]);
    assert(f5.kind == scientific::F_NONE);
    show("taps:", taps);

    // unit_impulse puts one hit in an array of zeros.
    let hit, f6 = scientific::unit_impulse([2, 3], [1], false);
    assert(f6.kind == scientific::F_NONE);
    show("unit impulse:", hit);

    // The states of the walk hold one row for each time.
    print("states: ", xout.shape[0], " rows of ", xout.shape[1], "\n");
    print("rtol: ", rtol, " iterations: ", iters, "\n");
    print("requested poles: ", rr.data[0], " + ", ri.data[0], "j\n");
}
