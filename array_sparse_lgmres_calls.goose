// The public calls lgmres, with and without a callback, and the call that
// gives the kept pairs.
namespace scientific;

import mathshim;

// lgmres solves the system A x = b. The call starts from x0, or from zero
// when x0 is empty. The inner size inner_m is the count of Krylov steps of
// one outer iteration. The call keeps the last outer_k correction vectors
// and adds them to the Krylov space of the next iteration. With
// store_outer_av the call keeps the product of the matrix with each
// correction, so the product costs no new call. With prepend_outer_v the
// kept vectors come before the Krylov vectors.
// With a preconditioner precond, lgmres solves the system on the left, as
// SciPy does with M. A trailing block gets the solution at the start of each
// outer iteration, before the stop check.
fn lgmres(s, b: f64[:], x0: f64[:] = [], rtol: f64 = 1.0e-5, atol: f64 = 0.0, maxiter: i64 = 1000,
          inner_m: i64 = 30, outer_k: i64 = 3, store_outer_av: bool = true, prepend_outer_v: bool = false,
          precond: Csr = csr_empty()) -> Arr, i64, Fault {
    let x, it, kp, f = lgmres_run(s, b, x0, rtol, atol, maxiter, inner_m, outer_k, store_outer_av,
                                  prepend_outer_v, precond, kept_empty(), false) { v => };
    return x, it, f;
}

fn lgmres<F>(s, b: f64[:], x0: f64[:] = [], rtol: f64 = 1.0e-5, atol: f64 = 0.0, maxiter: i64 = 1000,
             inner_m: i64 = 30, outer_k: i64 = 3, store_outer_av: bool = true, prepend_outer_v: bool = false,
             precond: Csr = csr_empty()) -> Arr, i64, Fault {
    let x, it, kp, f = lgmres_run(s, b, x0, rtol, atol, maxiter, inner_m, outer_k, store_outer_av,
                                  prepend_outer_v, precond, kept_empty(), false) { v => F(v) };
    return x, it, f;
}

// lgmres_pairs is lgmres with the kept pairs (dx, ax). The call takes the
// pairs of an earlier solve as outer_v and gives the pairs that stay.
fn lgmres_pairs(s, b: f64[:], x0: f64[:] = [], rtol: f64 = 1.0e-5, atol: f64 = 0.0, maxiter: i64 = 1000,
                inner_m: i64 = 30, outer_k: i64 = 3, store_outer_av: bool = true, prepend_outer_v: bool = false,
                precond: Csr = csr_empty(), outer_v: Kept = kept_empty()) -> Arr, i64, Kept, Fault {
    return lgmres_run(s, b, x0, rtol, atol, maxiter, inner_m, outer_k, store_outer_av,
                      prepend_outer_v, precond, outer_v, true) { v => };
}
