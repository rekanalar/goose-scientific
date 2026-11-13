// The call lobpcg and its checks.
//
// lobpcg finds the extreme eigenpairs of a sparse symmetric matrix with the
// locally optimal block preconditioned conjugate gradient method. The matrix
// B of a generalized problem, the preconditioner M and the constraint block
// Y are optional. A matrix without rows stands for an absent B or M.
namespace scientific;

import mathshim;

// The largest restart factor. The factor 2^60 stays finite.
const LOBPCG_MAX_RESTART: i64 = 60;

// lob_symmetric tells whether a sparse matrix equals its transpose. Each
// stored value must meet its mirror to a relative 1e-12.
fn lob_symmetric(s) -> bool {
    var r: i64 = 0;
    while r < s.rows {
        var k = csr_ptr(s, r);
        while k < csr_ptr(s, r + 1) {
            let v = csr_val(s, k);
            let mv, f = csr_get(s, csr_col(s, k), r);
            if f.kind != F_NONE { return false; }
            if mathshim::abs_of(v - mv) > 1.0e-12 * (mathshim::abs_of(v) + mathshim::abs_of(mv)) { return false; }
            k += 1;
        }
        r += 1;
    }
    return true;
}

// lob_positive_diagonal tells whether every diagonal value of a matrix is
// positive. A positive definite matrix has a positive diagonal.
fn lob_positive_diagonal(s) -> bool {
    var i: i64 = 0;
    while i < s.rows {
        let d, f = csr_get(s, i, i);
        if f.kind != F_NONE || !(d > 0.0) { return false; }
        i += 1;
    }
    return true;
}

// lob_check_optional checks a matrix B or M that may stand absent.
fn lob_check_optional(s, n: i64, name: const u8[:]) -> Fault {
    if s.rows == 0 { return no_fault(); }
    if s.rows != n || s.cols != n {
        return fault(F_SHAPE, "lobpcg", name);
    }
    return no_fault();
}

// lob_check checks every input of lobpcg. It gives the side n, the count k
// of the columns and the count ny of the constraints.
fn lob_check(a, x, b, m, y, tol: f64, maxiter: i64, rc: i64, hist: bool) -> i64, i64, i64, Fault {
    let fa = csr_check(a.rows, a.cols, "lobpcg");
    if fa.kind != F_NONE { return 0, 0, 0, fa; }
    if a.rows != a.cols { return 0, 0, 0, fault(F_SHAPE, "lobpcg", "the matrix A is not square"); }
    let n = a.rows;
    let fx = check_stack(x.ndim, x.shape, x.data.len, 1, "lobpcg");
    if fx.kind != F_NONE { return 0, 0, 0, fx; }
    if x.ndim != 2 || x.shape[0] != n { return 0, 0, 0, fault(F_SHAPE, "lobpcg", "the start block does not hold one row per row of A"); }
    let k = x.shape[1];
    if k < 1 || k > n { return 0, 0, 0, fault(F_SHAPE, "lobpcg", "the count of start vectors stands outside one to n"); }
    var ny: i64 = 0;
    if y.data.len > 0 {
        let fy = check_stack(y.ndim, y.shape, y.data.len, 1, "lobpcg");
        if fy.kind != F_NONE { return 0, 0, 0, fy; }
        if y.ndim != 2 || y.shape[0] != n { return 0, 0, 0, fault(F_SHAPE, "lobpcg", "the constraints do not hold one row per row of A"); }
        ny = y.shape[1];
        if ny + k > n { return 0, 0, 0, fault(F_SHAPE, "lobpcg", "the constraints and the start vectors outnumber the rows"); }
    }
    let fb = lob_check_optional(b, n, "the matrix B does not fit the matrix A");
    if fb.kind != F_NONE { return 0, 0, 0, fb; }
    let fm = lob_check_optional(m, n, "the preconditioner M does not fit the matrix A");
    if fm.kind != F_NONE { return 0, 0, 0, fm; }
    let fla = expm_matrix_check(a, "lobpcg");
    if fla.kind != F_NONE { return 0, 0, 0, fla; }
    if b.rows != 0 {
        let flb = expm_matrix_check(b, "lobpcg");
        if flb.kind != F_NONE { return 0, 0, 0, flb; }
    }
    if m.rows != 0 {
        let flm = expm_matrix_check(m, "lobpcg");
        if flm.kind != F_NONE { return 0, 0, 0, flm; }
    }
    if !lg_finite(x.data, x.data.len) || !lg_finite(y.data, y.data.len) {
        return 0, 0, 0, fault(F_VALUE, "lobpcg", "an input block holds a value that is not finite");
    }
    if !(tol >= 0.0) || tol - tol != 0.0 {
        return 0, 0, 0, fault(F_VALUE, "lobpcg", "the tolerance is negative or not finite");
    }
    if maxiter < 1 { return 0, 0, 0, fault(F_RANGE, "lobpcg", "the count of iterations is below one"); }
    if maxiter > MAX_KRYLOV_ITER { return 0, 0, 0, fault(F_LIMIT, "lobpcg", "the count of iterations stands above the limit"); }
    if rc < 1 || rc > LOBPCG_MAX_RESTART { return 0, 0, 0, fault(F_RANGE, "lobpcg", "the restart control stands outside one to 60"); }
    if n > MAX_ELEMENTS / (3 * k + ny) { return 0, 0, 0, fault(F_LIMIT, "lobpcg", "the work blocks hold more values than the limit"); }
    if hist && (maxiter + 3) > MAX_ELEMENTS / k { return 0, 0, 0, fault(F_LIMIT, "lobpcg", "a history holds more values than the limit"); }
    if !lob_symmetric(a) { return 0, 0, 0, fault(F_VALUE, "lobpcg", "the matrix A is not symmetric"); }
    if b.rows != 0 && !lob_symmetric(b) { return 0, 0, 0, fault(F_VALUE, "lobpcg", "the matrix B is not symmetric"); }
    if b.rows != 0 && !lob_positive_diagonal(b) { return 0, 0, 0, fault(F_VALUE, "lobpcg", "the matrix B has a diagonal value that is not positive"); }
    return n, k, ny, no_fault();
}

// lobpcg gives the k extreme eigenpairs of the symmetric matrix a. The
// start block x holds n rows and k columns. The values fall when largest
// is true and rise otherwise. A tol of zero gives sqrt(eps) * n. The call
// runs at most maxiter iterations, and niter tells how many ran. A caller
// passes copy(s) for a matrix b, m or y that the caller keeps. The call
// gives the values, the vectors in the columns of an n by k block, the
// count of iterations, the history of the values, the history of the
// residual norms and a fault. A history holds one row of k values per
// iteration. It is empty unless the caller asks for it.
// lob_result_finite tells whether the values and the vectors are finite. A
// finite input can still overflow in the arithmetic, and NumPy and SciPy
// then return NaN with no error.
fn lob_result_finite(w: Arr&, v: Arr&) -> bool {
    return lg_finite(w.data, w.data.len) && lg_finite(v.data, v.data.len);
}

fn lob_overflow() -> Fault {
    return fault(F_VALUE, "lobpcg", "the arithmetic overflowed: the result is not finite");
}

fn lobpcg(a, x, b: Csr = csr_empty(), m: Csr = csr_empty(), y: Arr = empty_arr(), tol: f64 = 0.0, maxiter: i64 = 20,
          largest: bool = true, restart_control: i64 = 20, want_lambda_history: bool = false,
          want_residual_history: bool = false) -> Arr, Arr, i64, Arr, Arr, Fault {
    let n, k, ny, f = lob_check(a, x, b, m, y, tol, maxiter, restart_control, want_lambda_history || want_residual_history);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), 0, empty_arr(), empty_arr(), f; }
    if n - ny < 5 * k {
        if ny > 0 { return empty_arr(), empty_arr(), 0, empty_arr(), empty_arr(), fault(F_SHAPE, "lobpcg", "the dense path takes no constraints"); }
        if n > BD_MAX_SIDE { return empty_arr(), empty_arr(), 0, empty_arr(), empty_arr(), fault(F_LIMIT, "lobpcg", "the dense path stands above the limit of the side"); }
        let w, v, fd = lob_dense(a, b, k, largest);
        if fd.kind != F_NONE { return empty_arr(), empty_arr(), 0, empty_arr(), empty_arr(), fd; }
        if !lob_result_finite(w, v) { return empty_arr(), empty_arr(), 0, empty_arr(), empty_arr(), lob_overflow(); }
        return w, v, 0, empty_arr(), empty_arr(), no_fault();
    }
    var t = tol;
    if t == 0.0 { t = 1.4901161193847656e-08 * (n as f64); }
    let w, v, it, lh, rh, fi = lob_iterate(a, x.data, n, k, b, m, y.data, ny, t, maxiter, largest, restart_control, want_lambda_history, want_residual_history);
    if fi.kind != F_NONE { return empty_arr(), empty_arr(), 0, empty_arr(), empty_arr(), fi; }
    if !lob_result_finite(w, v) { return empty_arr(), empty_arr(), 0, empty_arr(), empty_arr(), lob_overflow(); }
    return w, v, it, lh, rh, no_fault();
}
