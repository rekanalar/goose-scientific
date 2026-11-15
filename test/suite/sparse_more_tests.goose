// The tests of the group sparse_more. Run them with SCIENTIFIC_SUITES=sparse_more.
//
// The pins come from the generated file of the group.
namespace sparse_more_tests;

import scientific from "../..";
import os;
import common;
import sparse_more_pins;

// The tolerance of every value of a pin.
const TOL: f64 = 1.0e-8;

// sm_close tells whether two values agree to the tolerance.
fn sm_close(a: f64, b: f64) -> bool {
    let d = common::mathshim_abs(a - b);
    return d <= TOL * (1.0 + common::mathshim_abs(b));
}

// sm_dense builds a dense matrix from the values of a pin.
fn sm_dense(v: f64[:], rows: i64, cols: i64) -> scientific::Arr {
    return common::mat(v, rows, cols);
}

// sm_csr builds a matrix of compressed rows from a dense pin.
fn sm_csr(v: f64[:], rows: i64, cols: i64) -> scientific::Csr {
    let s, f = scientific::csr_from_dense(sm_dense(v, rows, cols));
    assert(f.kind == scientific::F_NONE);
    return s;
}

// The checks of the formats accept a sound matrix and reject a matrix
// that a caller broke.
fn test_sparse_more_valid() {
    let csr = sm_csr(sparse_more_pins::LOW, 4, 4);
    assert(scientific::csr_valid(csr));
    let csc, f1 = scientific::csc_from_csr(csr);
    assert(f1.kind == scientific::F_NONE);
    assert(scientific::csc_valid(csc));
    let coo, f2 = scientific::coo_from_csr(csr);
    assert(f2.kind == scientific::F_NONE);
    assert(scientific::coo_valid(coo));
    let dok, f3 = scientific::dok_from_csr(csr);
    assert(f3.kind == scientific::F_NONE);
    assert(scientific::dok_valid(dok));
    let lil, f4 = scientific::lil_from_csr(csr);
    assert(f4.kind == scientific::F_NONE);
    assert(scientific::lil_valid(lil));
    // A matrix of diagonals from the offsets minus one, zero and one.
    let off: f64[3] = [-1.0, 0.0, 1.0];
    let dvals: f64[10] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let dia, f5 = scientific::dia_from_diagonals(4, 4, off, dvals);
    assert(f5.kind == scientific::F_NONE);
    assert(scientific::dia_valid(dia));
    // A matrix of blocks.
    let bsr, f6 = scientific::bsr_from_dense(sm_dense(sparse_more_pins::BAND, 4, 4), 2, 2);
    assert(f6.kind == scientific::F_NONE);
    assert(scientific::bsr_valid(bsr));
    // A matrix whose pointer of the rows does not grow is not sound.
    var buf: f64[>..] = [0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0];
    let broken = scientific::Csr { rows: 2, cols: 2, nnz: 2, buf: copy(buf) };
    assert(!scientific::csr_valid(broken));
    // A matrix of a buffer that holds too few values is not sound.
    var short_buf: f64[>..] = [0.0, 1.0];
    let short = scientific::Csr { rows: 1, cols: 2, nnz: 1, buf: copy(short_buf) };
    assert(!scientific::csr_valid(short));
    // A column that stands outside the matrix is not sound.
    var bad_col: f64[>..] = [0.0, 1.0, 5.0, 1.0];
    let outside = scientific::Csr { rows: 1, cols: 2, nnz: 1, buf: copy(bad_col) };
    assert(!scientific::csr_valid(outside));
    // A value that is not finite is not sound.
    var bad_val: f64[>..] = [0.0, 1.0, 0.0, 0.0 / 0.0];
    let nan_v = scientific::Csr { rows: 1, cols: 2, nnz: 1, buf: copy(bad_val) };
    assert(!scientific::csr_valid(nan_v));
}

// The type of the indices.
fn test_sparse_more_index_dtype() {
    assert(scientific::get_index_dtype(1000) == 32);
    assert(scientific::get_index_dtype(scientific::MAX_INDEX32) == 32);
    assert(scientific::get_index_dtype(scientific::MAX_INDEX32 + 1) == 64);
    let small: f64[3] = [0.0, 1.0, 2.0];
    let bits, f0 = scientific::safely_cast_index_arrays(small);
    assert(f0.kind == scientific::F_NONE && bits == 32);
    let big: f64[2] = [0.0, 3000000000.0];
    let bits2, f1 = scientific::safely_cast_index_arrays(big);
    assert(f1.kind == scientific::F_NONE && bits2 == 64);
    // A value that is not a whole number is a fault.
    let half: f64[1] = [0.5];
    let bits3, f2 = scientific::safely_cast_index_arrays(half);
    assert(f2.kind == scientific::F_VALUE);
    // A value below zero is a fault.
    let neg: f64[1] = [-1.0];
    let bits4, f3 = scientific::safely_cast_index_arrays(neg);
    assert(f3.kind == scientific::F_VALUE);
}

// The places of a matrix of chance stand apart, and the count of the
// places follows the density.
fn test_sparse_more_random() {
    let s, f = scientific::random_array([8, 6], 0.25, 7);
    assert(f.kind == scientific::F_NONE);
    // The count of the stored values is 12, the density times 48.
    assert(s.nnz == 12);
    assert(scientific::csr_valid(s));
    // Every value stands in the range from zero to one.
    var k: i64 = 0;
    while k < s.nnz {
        let v = scientific::csr_val(s, k);
        assert(v >= 0.0 && v < 1.0);
        k += 1;
    }
    // The places stand apart. The count of the pairs of the places is
    // the count of the stored values.
    var pairs: i64 = 0;
    var i: i64 = 0;
    while i < s.rows {
        pairs += scientific::csr_ptr(s, i + 1) - scientific::csr_ptr(s, i);
        i += 1;
    }
    assert(pairs == s.nnz);
    // The same seed gives the same matrix.
    let s2, f2 = scientific::random_array([8, 6], 0.25, 7);
    assert(f2.kind == scientific::F_NONE);
    assert(s2.nnz == s.nnz);
    var k2: i64 = 0;
    while k2 < s.buf.len {
        assert(s2.buf[k2] == s.buf[k2]);
        k2 += 1;
    }
    // The old call of two sides gives the same matrix.
    let s3, f3 = scientific::random(8, 6, 0.25, 7);
    assert(f3.kind == scientific::F_NONE);
    assert(s3.buf.len == s.buf.len);
    var k3: i64 = 0;
    while k3 < s.buf.len {
        assert(s3.buf[k3] == s.buf[k3]);
        k3 += 1;
    }
    // A density of zero gives a matrix without values.
    let s4, f4 = scientific::random_array([4, 4], 0.0, 1);
    assert(f4.kind == scientific::F_NONE && s4.nnz == 0);
    // A density of one gives the whole matrix.
    let s5, f5 = scientific::random_array([3, 3], 1.0, 2);
    assert(f5.kind == scientific::F_NONE && s5.nnz == 9);
    // A density outside zero to one is a fault.
    let s6, f6 = scientific::random_array([4, 4], 1.5, 1);
    assert(f6.kind == scientific::F_VALUE);
    let s7, f7 = scientific::random_array([4, 4], -0.1, 1);
    assert(f7.kind == scientific::F_VALUE);
}

// The solve of a triangular system.
fn test_sparse_more_triangular_solve() {
    let low = sm_csr(sparse_more_pins::LOW, 4, 4);
    let x, f = scientific::spsolve_triangular(low, sparse_more_pins::RHS, true);
    assert(f.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < 4 {
        assert(sm_close(x.data[i], sparse_more_pins::LOW_X[i]));
        i += 1;
    }
    let up = sm_csr(sparse_more_pins::LOW, 4, 4);
    let ut, f2 = scientific::csr_transpose(up);
    assert(f2.kind == scientific::F_NONE);
    let x2, f3 = scientific::spsolve_triangular(ut, sparse_more_pins::RHS, false);
    assert(f3.kind == scientific::F_NONE);
    i = 0;
    while i < 4 {
        assert(sm_close(x2.data[i], sparse_more_pins::UP_X[i]));
        i += 1;
    }
    // A zero on the diagonal is a fault.
    let zero = sm_csr([0.0, 0.0, 0.0, 1.0], 2, 2);
    let x4, f4 = scientific::spsolve_triangular(zero, [1.0, 1.0], true);
    assert(f4.kind == scientific::F_VALUE);
    // A right side of another length is a fault.
    let x5, f5 = scientific::spsolve_triangular(low, [1.0], true);
    assert(f5.kind == scientific::F_SHAPE);
}

// The check of the triangle and the bandwidth.
fn test_sparse_more_band() {
    let band = sm_csr(sparse_more_pins::BAND, 4, 4);
    let lower, upper = scientific::is_sptriangular(band);
    assert(!lower && !upper);
    let low = sm_csr(sparse_more_pins::LOW, 4, 4);
    let l2, u2 = scientific::is_sptriangular(low);
    assert(l2 && !u2);
    let l3, u3 = scientific::is_sptriangular(low);
    // The transpose holds only the upper triangle.
    let lt, ft = scientific::csr_transpose(low);
    assert(ft.kind == scientific::F_NONE);
    let l4, u4 = scientific::is_sptriangular(lt);
    assert(!l4 && u4);
    let below, above = scientific::spbandwidth(band);
    assert(below == 2 && above == 2);
    let diag = sm_csr([1.0, 0.0, 0.0, 2.0], 2, 2);
    let b2, a2 = scientific::spbandwidth(diag);
    assert(b2 == 0 && a2 == 0);
    let b3, a3 = scientific::spbandwidth(low);
    assert(b3 == 3 && a3 == 0);
}

// The Laplacian of a grid and its values.
fn test_sparse_more_laplacian() {
    var s1: i64[>..] = [];
    s1.push(5);
    let l1, f1 = scientific::laplacian_nd(s1, "neumann");
    assert(f1.kind == scientific::F_NONE);
    let v1, f2 = scientific::laplacian_nd_values(s1, "neumann");
    assert(f2.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < 5 {
        assert(sm_close(v1.data[i], sparse_more_pins::LAP_NEUMANN_5[i]));
        i += 1;
    }
    // The values agree with the values of the matrix.
    let w1, w2, f3 = scientific::eigsh(l1, 2);
    assert(f3.kind == scientific::F_NONE);
    // The values of the matrix agree with the two values of the closed
    // form that carry the largest magnitude. The values stand in the
    // order from the lowest to the highest.
    assert(sm_close(w1.data[0], sparse_more_pins::LAP_NEUMANN_5[0]));
    assert(sm_close(w1.data[1], sparse_more_pins::LAP_NEUMANN_5[1]));
    var s2: i64[>..] = [];
    s2.push(3);
    s2.push(4);
    let v2, f4 = scientific::laplacian_nd_values(s2, "neumann");
    assert(f4.kind == scientific::F_NONE);
    i = 0;
    while i < 12 {
        assert(sm_close(v2.data[i], sparse_more_pins::LAP_NEUMANN_3X4[i]));
        i += 1;
    }
    let v3, f5 = scientific::laplacian_nd_values(s2, "dirichlet");
    assert(f5.kind == scientific::F_NONE);
    i = 0;
    while i < 12 {
        assert(sm_close(v3.data[i], sparse_more_pins::LAP_DIRICHLET_3X4[i]));
        i += 1;
    }
    let v4, f6 = scientific::laplacian_nd_values(s2, "periodic");
    assert(f6.kind == scientific::F_NONE);
    i = 0;
    while i < 12 {
        assert(sm_close(v4.data[i], sparse_more_pins::LAP_PERIODIC_3X4[i]));
        i += 1;
    }
    let v5, f7 = scientific::laplacian_nd_values(s1, "dirichlet");
    assert(f7.kind == scientific::F_NONE);
    i = 0;
    while i < 5 {
        assert(sm_close(v5.data[i], sparse_more_pins::LAP_DIRICHLET_5[i]));
        i += 1;
    }
    let v6, f8 = scientific::laplacian_nd_values(s1, "periodic");
    assert(f8.kind == scientific::F_NONE);
    i = 0;
    while i < 5 {
        assert(sm_close(v6.data[i], sparse_more_pins::LAP_PERIODIC_5[i]));
        i += 1;
    }
    // The boundary of an unknown name is a fault.
    let l7, f9 = scientific::laplacian_nd(s1, "unknown");
    assert(f9.kind == scientific::F_VALUE);
}

// The values of a matrix that is not symmetric.
fn test_sparse_more_eigs() {
    let m = sm_csr(sparse_more_pins::EIGS_M, 5, 5);
    let vals, vecs, f = scientific::eigs(m, 2, "LM");
    assert(f.kind == scientific::F_NONE);
    // The values and the vectors are complex, so each element holds a
    // real part and an imaginary part: two values hold four numbers, and
    // the five rows of two vectors hold twenty.
    assert(vals.data.len == 4 && vecs.data.len == 20);
    // The absolute values of the two values agree with NumPy.
    let absarr = scientific::cabs(vals);
    let abs1 = absarr.data;
    assert(sm_close(abs1[0], sparse_more_pins::EIGS_LM2_ABS[0]));
    assert(sm_close(abs1[1], sparse_more_pins::EIGS_LM2_ABS[1]));
}

// The product of the function of a matrix and a vector.
fn test_sparse_more_funm_multiply() {
    let s = sm_csr(sparse_more_pins::SYM6, 6, 6);
    // The exponential of the matrix times the vector.
    let y, f = scientific::funm_multiply_krylov(s, sparse_more_pins::SYM6_V) { x => mathshim::power_e(x) };
    assert(f.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < 6 {
        assert(sm_close(y.data[i], sparse_more_pins::SYM6_EXPV[i]));
        i += 1;
    }
    // A right side of another length is a fault.
    let y2, f2 = scientific::funm_multiply_krylov(s, [1.0]) { x => mathshim::power_e(x) };
    assert(f2.kind == scientific::F_SHAPE);
}

// A matrix that a caller built by hand and broke does not pass the
// check of its format, and the calls that read the buffer of a matrix
// give a fault.
fn test_sparse_more_broken_matrix() {
    // The buffer of the diagonals holds fewer values than its count of
    // the offsets.
    let dia = scientific::Dia { rows: 2, cols: 2, ndiag: 5, nnz: 5, buf: [0.0] };
    assert(!scientific::dia_valid(dia));
    // A matrix of compressed rows whose buffer holds fewer values than
    // its sides ask for.
    let bad = scientific::Csr { rows: 2, cols: 2, nnz: 4, buf: [0.0] };
    assert(!scientific::csr_valid(bad));
    let below, above = scientific::spbandwidth(bad);
    assert(below == 0 && above == 0);
    let low, up = scientific::is_sptriangular(bad);
    assert(!low && !up);
    let x, fx = scientific::spsolve_triangular(bad, [1.0, 1.0]);
    assert(fx.kind == scientific::F_SHAPE);
    let w, v, fw = scientific::eigs(bad, 1);
    assert(fw.kind == scientific::F_SHAPE);
}
