// The preconditioner of the restarted solvers lgmres and gcrotmk.
//
// The preconditioner is a sparse matrix. lgmres applies it on the left and
// gcrotmk on the right, as SciPy does with M. An empty matrix means no
// preconditioner.
namespace scientific;

import mathshim;

// The sides of a preconditioner. With PRE_LEFT the product of the matrix
// becomes the product of the preconditioner with it. With PRE_RIGHT the
// direction becomes the product of the preconditioner with the basis
// vector. With PRE_NONE the call applies no preconditioner.
const PRE_NONE: i64 = 0;
const PRE_LEFT: i64 = 1;
const PRE_RIGHT: i64 = 2;

// precond_present tells whether the call applies a preconditioner. An empty
// matrix means none.
fn precond_present(mp) -> bool {
    return !(mp.rows == 0 && mp.cols == 0 && mp.nnz == 0);
}

// precond_check checks the preconditioner against the side n of the system.
// An empty matrix means no preconditioner.
fn precond_check(mp, n: i64, tag: const u8[:]) -> Fault {
    if !precond_present(mp) { return no_fault(); }
    let f = expm_matrix_check(mp, tag);
    if f.kind != F_NONE { return f; }
    if mp.rows != n { return fault(F_SHAPE, tag, "the preconditioner does not fit the matrix"); }
    return no_fault();
}

// precond_apply gives out = M v for the n values of v that start at off. The
// product goes to the n values of out that start at poff.
fn precond_apply(mp, v: f64[:], off: i64, n: i64, out: f64[>..]&, poff: i64) {
    expm_block_product(mp, v, off, n, out, poff);
}
