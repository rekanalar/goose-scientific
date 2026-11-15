// The discretization of a system: the state space form at a step of
// time.
//
// The call turns the continuous system of the matrices A, B and C and
// the direct part D into the system of one step of time dt. The
// methods follow scipy.signal.cont2discrete. The zero order hold and
// the first order hold use the exponential of a matrix. The methods of
// the bilinear family solve a linear system. The state space form of
// this package holds one input and one output.
namespace scientific;

import mathshim;

// cont2discrete gives the discrete system of the continuous system a,
// b, c and d at the step dt. The method is "zoh", "foh", "impulse",
// "gbt", "bilinear", "tustin", "euler", "forward_diff" or
// "backward_diff". The method "gbt" takes the alpha of the caller in
// the range of zero to one. The matrices of the form are those of the
// state space calls: a is the square matrix of the states, b is the
// column of the input, c is the row of the output and d is the direct
// part.
fn cont2discrete(a: Arr, b: Arr, c: Arr, d: f64, dt: f64, method: const u8[:] = "zoh", alpha: f64 = 0.5) -> Arr, Arr, Arr, f64, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_SHAPE, "cont2discrete", "the matrix of the states is not square");
    }
    let n = a.shape[0];
    if b.ndim != 2 || b.shape[0] != n || b.shape[1] != 1 || c.ndim != 2 || c.shape[0] != 1 || c.shape[1] != n {
        return empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_SHAPE, "cont2discrete", "the input or the output does not match the matrix of the states");
    }
    // The matrices are read by their shape, so the buffer of each
    // matrix must agree with its shape.
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, "cont2discrete");
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fa; }
    let fb = check_stack(b.ndim, b.shape, b.data.len, 1, "cont2discrete");
    if fb.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fb; }
    let fc = check_stack(c.ndim, c.shape, c.data.len, 1, "cont2discrete");
    if fc.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fc; }
    if !(dt > 0.0) {
        return empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_VALUE, "cont2discrete", "the step holds a value of zero or below");
    }
    if !all_finite(a.data) || !all_finite(b.data) || !all_finite(c.data) || !all_finite([d, dt]) {
        return empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_VALUE, "cont2discrete", "a value is not finite");
    }
    if method == "gbt" {
        if !(alpha >= 0.0 && alpha <= 1.0) {
            return empty_arr(), empty_arr(), empty_arr(), 0.0,
                fault(F_VALUE, "cont2discrete", "the alpha holds a value outside zero to one");
        }
    } else if method == "bilinear" || method == "tustin" {
        let ad, bd, cd, dd, f = cont2discrete_gbt(a, b, c, d, dt, 0.5);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, f; }
        return ad, bd, cd, dd, no_fault();
    } else if method == "euler" || method == "forward_diff" {
        let ad, bd, cd, dd, f = cont2discrete_gbt(a, b, c, d, dt, 0.0);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, f; }
        return ad, bd, cd, dd, no_fault();
    } else if method == "backward_diff" {
        let ad, bd, cd, dd, f = cont2discrete_gbt(a, b, c, d, dt, 1.0);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, f; }
        return ad, bd, cd, dd, no_fault();
    } else if method == "zoh" {
        return cont2discrete_zoh(a, b, c, d, dt);
    } else if method == "impulse" {
        return cont2discrete_impulse(a, b, c, d, dt);
    } else if method == "foh" {
        return cont2discrete_foh(a, b, c, d, dt);
    } else {
        return empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_VALUE, "cont2discrete", "the method holds an unknown name");
    }
    return cont2discrete_gbt(a, b, c, d, dt, alpha);
}

// cont2discrete_gbt gives the discrete system of the method of the
// generalized bilinear transform with the weight alpha.
fn cont2discrete_gbt(a: Arr&, b: Arr&, c: Arr&, d: f64, dt: f64, alpha: f64) -> Arr, Arr, Arr, f64, Fault {
    let n = a.shape[0];
    let eye_m, fe = eye(n);
    if fe.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fe; }
    // ima = I - alpha dt A; the other side = I + (1 - alpha) dt A.
    let ada, fa = scale(a, alpha * dt);
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fa; }
    let ima, fi = sub(eye_m, ada);
    if fi.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fi; }
    let aod, fo = scale(a, (1.0 - alpha) * dt);
    if fo.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fo; }
    let other, fn2 = add(eye_m, aod);
    if fn2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fn2; }
    let ad, fad = solve(ima, other);
    if fad.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fad; }
    let bd_in, fb1 = scale(b, dt);
    if fb1.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fb1; }
    let bd, fbd = solve(ima, bd_in);
    if fbd.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fbd; }
    let ct, fct = transpose(c);
    if fct.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fct; }
    let it, fit = transpose(ima);
    if fit.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fit; }
    let cd_t, fcd = solve(it, ct);
    if fcd.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fcd; }
    let cd, fcd2 = transpose(cd_t);
    if fcd2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fcd2; }
    let cb, fcb = matmul(c, bd);
    if fcb.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fcb; }
    let dd = d + alpha * cb.data[0];
    return ad, bd, cd, dd, no_fault();
}

// cont2discrete_zoh gives the discrete system of the zero order hold.
fn cont2discrete_zoh(a: Arr&, b: Arr&, c: Arr&, d: f64, dt: f64) -> Arr, Arr, Arr, f64, Fault {
    let n = a.shape[0];
    let size = n + 1;
    var em: f64[>..] = [];
    var i: i64 = 0;
    while i < size * size { em.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            em[i * size + j] = a.data[i * n + j] * dt;
            j += 1;
        }
        em[i * size + n] = b.data[i] * dt;
        i += 1;
    }
    let emm = mat2(copy(em), size, size);
    let ms, f = expm(emm);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, f; }
    let rows, fr = slice_axis(ms, 0, 0, n, 1);
    if fr.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fr; }
    let ad, fa = slice_axis(rows, 1, 0, n, 1);
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fa; }
    let bd, fb = slice_axis(rows, 1, n, size, 1);
    if fb.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fb; }
    return ad, bd, copy(c), d, no_fault();
}

// cont2discrete_impulse gives the discrete system of the method of the
// impulse. The direct part must hold zero.
fn cont2discrete_impulse(a: Arr&, b: Arr&, c: Arr&, d: f64, dt: f64) -> Arr, Arr, Arr, f64, Fault {
    if d != 0.0 {
        return empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_VALUE, "cont2discrete", "the impulse method needs a direct part of zero");
    }
    let ada, fa = scale(a, dt);
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fa; }
    let ad, fe = expm(ada);
    if fe.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fe; }
    let adb, fadb = matmul(ad, b);
    if fadb.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fadb; }
    let bd, fb = scale(adb, dt);
    if fb.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fb; }
    let cb, fcb = matmul(c, b);
    if fcb.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fcb; }
    return ad, bd, copy(c), cb.data[0] * dt, no_fault();
}

// cont2discrete_foh gives the discrete system of the first order hold.
fn cont2discrete_foh(a: Arr&, b: Arr&, c: Arr&, d: f64, dt: f64) -> Arr, Arr, Arr, f64, Fault {
    let n = a.shape[0];
    let size = n + 2;
    var em: f64[>..] = [];
    var i: i64 = 0;
    while i < size * size { em.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            em[i * size + j] = a.data[i * n + j] * dt;
            j += 1;
        }
        em[i * size + n] = b.data[i] * dt;
        i += 1;
    }
    em[n * size + n + 1] = 1.0;
    let emm = mat2(copy(em), size, size);
    let ms, f = expm(emm);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, f; }
    let rows, fr = slice_axis(ms, 0, 0, n, 1);
    if fr.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fr; }
    let ad, fa = slice_axis(rows, 1, 0, n, 1);
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fa; }
    let m12, f12 = slice_axis(rows, 1, n, n + 1, 1);
    if f12.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, f12; }
    let m13, f13 = slice_axis(rows, 1, n + 1, size, 1);
    if f13.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, f13; }
    let bd, fbd = cont2discrete_bcol(ad, m12, m13);
    if fbd.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fbd; }
    let c13, fc13 = matmul(c, m13);
    if fc13.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, fc13; }
    return ad, bd, copy(c), d + c13.data[0], no_fault();
}

// cont2discrete_bcol gives the column of the input of the first order
// hold: bd = m12 - m13 + ad m13.
fn cont2discrete_bcol(ad, m12, m13) -> Arr, Fault {
    let t, f1 = sub(m12, m13);
    if f1.kind != F_NONE { return empty_arr(), f1; }
    let p, f2 = matmul(ad, m13);
    if f2.kind != F_NONE { return empty_arr(), f2; }
    return add(t, p);
}
