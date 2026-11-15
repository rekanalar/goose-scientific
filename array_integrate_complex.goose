// The solver of the systems with the complex values.
//
// The call follows complex_ode of SciPy. The system of n complex
// values runs as the system of 2n real values, where the real part of
// every value pairs with its imaginary part. The block of the caller
// takes the time and a list of the complex values and gives a list of
// the derivatives.
//
namespace scientific;

import mathshim;

// complex_ode integrates the system z' = F(t, z) of complex values
// from t0 to t1. The start y0 holds one row per value. The block takes
// the time and a list of the complex values and gives the list of the
// derivatives. The method is RK45, RK23, DOP853, BDF, Radau or LSODA.
// The result holds the values at the times of t_eval. An empty t_eval
// gives the value at t1, as the call integrate of SciPy gives it.
fn complex_ode<F: fn(f64, Cplx[>..]) -> Cplx[>..]>(t0: f64, t1: f64, y0: CArr, rtol: f64 = 0.001, atol: f64 = 1.0e-6, method: const u8[:] = "RK45", max_step: f64 = 1.0e308, t_eval: f64[:] = []) -> Arr, CArr, Fault {
    if y0.ndim != 1 || y0.data.len < 2 {
        return empty_arr(), carr_empty(), fault(F_SHAPE, "complex_ode", "the start is not a list of values");
    }
    let n = y0.data.len / 2;
    var yint: f64[>..] = [];
    var i: i64 = 0;
    while i < y0.data.len {
        if !(y0.data[i] - y0.data[i] == 0.0) {
            return empty_arr(), carr_empty(), fault(F_VALUE, "complex_ode", "the start holds a value that is not finite");
        }
        yint.push(y0.data[i]);
        i += 1;
    }
    // The system of the real values pairs the real part of every
    // value with its imaginary part.
    // An empty t_eval gives the value at t1, as SciPy gives it.
    var te: f64[>..] = [];
    if t_eval.len == 0 { te.push(t1); } else { te = copy_slice(t_eval); }
    let t, y, f = solve_ivp(t0, t1, yint, rtol, atol, method, max_step, te) { tt, yy =>
        var z: Cplx[>..] = [];
        var q: i64 = 0;
        while q < n { z.push(Cplx { re: yy[2 * q], im: yy[2 * q + 1] }); q += 1; }
        let d = F(tt, copy(z));
        var out: f64[>..] = [];
        if d.len == n {
            q = 0;
            while q < n { out.push(d[q].re); out.push(d[q].im); q += 1; }
        }
        out
    };
    if f.kind != F_NONE {
        return empty_arr(), carr_empty(), fault(f.kind, "complex_ode", f.detail);
    }
    if y.data.len != yint.len * t.data.len {
        return empty_arr(), carr_empty(), fault(F_SHAPE, "complex_ode", "the function holds a different count of values");
    }
    // The rows of the answer hold the interleaved parts of the values.
    var shape: i64[..8] = [];
    shape.push(t.data.len);
    shape.push(n);
    return t, carr_make(shape, copy(y.data)), no_fault();
}
