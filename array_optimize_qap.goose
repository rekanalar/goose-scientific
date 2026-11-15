// The quadratic assignment of SciPy quadratic_assignment.
//
// The problem holds two square matrices A and B and asks for the
// permutation P of the rows that gives the least value of the trace
// of the product of the transpose of A, P, B and the transpose of P.
// The problem is NP hard, so both methods of SciPy give an
// approximation.
//
// The method faq is the algorithm of Frank and Wolfe of the paper of
// Vogelstein and Conroy. It walks a doubly stochastic matrix from the
// barycenter of the polytope of Birkhoff, it solves the linear
// assignment on the gradient for the direction, and it takes the
// whole step or the best point of the span between the two matrices.
// The method 2opt is the local search that swaps two rows at a time
// while a swap lowers the value.
namespace scientific;

import mathshim;

// The largest side of the matrices of the method faq. One iteration
// holds products of the side three times and a linear assignment.
const QAP_MAX_FAQ: i64 = 256;

// The largest side of the method 2opt. One pass holds a test of every
// pair, and each test holds the whole value.
const QAP_MAX_2OPT: i64 = 64;

// qap_score gives the value of a permutation. It holds the sum of the
// products of A and the matrix B with its rows and columns in the
// order of perm, as the helper _calc_score of SciPy holds it.
fn qap_score(a, b, perm: f64[:]) -> f64 {
    let n = perm.len;
    var s: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        let pi = perm[i] as i64;
        var j: i64 = 0;
        while j < n {
            let pj = perm[j] as i64;
            s += a.data[i * n + j] * b.data[pi * n + pj];
            j += 1;
        }
        i += 1;
    }
    return s;
}

// qap_check_args holds the checks that the two methods share.
fn qap_check_args(a, b, tag: const u8[:]) -> i64, Fault {
    let fsa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fsa.kind != F_NONE { return 0, fsa; }
    let fsb = check_stack(b.ndim, b.shape, b.data.len, 1, tag);
    if fsb.kind != F_NONE { return 0, fsb; }
    if a.ndim != 2 || a.shape.len != 2 || a.shape[0] != a.shape[1] {
        return 0, fault(F_SHAPE, tag, "the first matrix is not square");
    }
    if b.ndim != 2 || b.shape.len != 2 || b.shape[0] != b.shape[1] {
        return 0, fault(F_SHAPE, tag, "the second matrix is not square");
    }
    if a.shape[0] != b.shape[0] {
        return 0, fault(F_SHAPE, tag, "the matrices do not hold the same side");
    }
    let n = a.shape[0];
    if n < 1 {
        return 0, fault(F_VALUE, tag, "the matrices hold no node");
    }
    if !all_finite(a.data) || !all_finite(b.data) {
        return 0, fault(F_VALUE, tag, "a value of the matrices is not finite");
    }
    return n, no_fault();
}

// quadratic_assignment gives the permutation of the rows of B that
// gives the least value of the problem of the assignment of two
// matrices. The method faq runs the algorithm of Frank and Wolfe from
// the barycenter, and the method 2opt runs the local search of the
// swaps from a random start. The seed fixes the start of the method
// 2opt, so equal seeds give equal permutations. The answer holds the
// column of each row.
fn quadratic_assignment(a, b, method: const u8[:] = "faq", seed: i64 = 1) -> Arr, Fault {
    let n, g = qap_check_args(a, b, "quadratic_assignment");
    if g.kind != F_NONE { return empty_arr(), g; }
    if method != "faq" && method != "2opt" {
        return empty_arr(), fault(F_VALUE, "quadratic_assignment", "the method holds none of faq and 2opt");
    }
    var perm: f64[>..] = [];
    if method == "faq" {
        if n > QAP_MAX_FAQ {
            return empty_arr(), fault(F_LIMIT, "quadratic_assignment", "the side holds a value above the limit of the method faq");
        }
        perm = qap_faq(a, b, n);
    } else {
        if n > QAP_MAX_2OPT {
            return empty_arr(), fault(F_LIMIT, "quadratic_assignment", "the side holds a value above the limit of the method 2opt");
        }
        perm = qap_2opt(a, b, n, seed);
    }
    return vec1(copy(perm)), no_fault();
}

// qap_grad gives the gradient of the value at a doubly stochastic
// matrix P. It forms the products A P and transpose(A) P first. It
// then multiplies each product with B in a single pass.
fn qap_grad(a, b, p: f64[:], n: i64) -> f64[>..] {
    var t1: f64[>..] = [];
    var t2: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { t1.push(0.0); t2.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var l: i64 = 0;
        while l < n {
            var s1: f64 = 0.0;
            var s2: f64 = 0.0;
            var k: i64 = 0;
            while k < n {
                s1 += a.data[i * n + k] * p[k * n + l];
                s2 += a.data[k * n + i] * p[k * n + l];
                k += 1;
            }
            t1[i * n + l] = s1;
            t2[i * n + l] = s2;
            l += 1;
        }
        i += 1;
    }
    var out: f64[>..] = [];
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            var t: f64 = 0.0;
            var l: i64 = 0;
            while l < n {
                t += t1[i * n + l] * b.data[j * n + l];
                t += t2[i * n + l] * b.data[l * n + j];
                l += 1;
            }
            out.push(t);
            j += 1;
        }
        i += 1;
    }
    return out;
}

// qap_quad gives the count of the products of the two matrices with a
// point between P and Q. It gives the two coefficients of the value
// as a form of the second order in the step.
fn qap_quad(a, b, p: f64[:], q: f64[:], n: i64) -> f64, f64 {
    var s00: f64 = 0.0;
    var s01: f64 = 0.0;
    var s10: f64 = 0.0;
    var s11: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        var k: i64 = 0;
        while k < n {
            var ap: f64 = 0.0;
            var aq: f64 = 0.0;
            var l: i64 = 0;
            while l < n {
                ap += a.data[i * n + l] * p[l * n + k];
                aq += a.data[i * n + l] * q[l * n + k];
                l += 1;
            }
            var pb: f64 = 0.0;
            var qb: f64 = 0.0;
            l = 0;
            while l < n {
                pb += p[i * n + l] * b.data[l * n + k];
                qb += q[i * n + l] * b.data[l * n + k];
                l += 1;
            }
            s00 += ap * pb;
            s01 += ap * qb;
            s10 += aq * pb;
            s11 += aq * qb;
            k += 1;
        }
        i += 1;
    }
    let c2 = s00 - s01 - s10 + s11;
    let c1 = s01 + s10 - 2.0 * s11;
    return c2, c1;
}

// qap_faq holds the algorithm of Frank and Wolfe. The start is the
// barycenter of the polytope of Birkhoff, as SciPy holds it with the
// setting P0 of barycenter. The loop takes the direction from the
// linear assignment on the gradient, and it takes the best point of
// the span between the point and the direction.
fn qap_faq(a, b, n: i64) -> f64[>..] {
    let inv_n = 1.0 / (n as f64);
    var p: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { p.push(inv_n); i += 1; }
    var iter: i64 = 0;
    while iter < 30 {
        iter += 1;
        let grad = qap_grad(a, b, p, n);
        // The direction holds the permutation of the linear
        // assignment on the gradient.
        let _, cols, fg = linear_sum_assignment(mat2(copy(grad), n, n));
        if fg.kind != F_NONE { break; }
        var q: f64[>..] = [];
        i = 0;
        while i < n * n { q.push(0.0); i += 1; }
        i = 0;
        while i < n { q[i * n + (cols.data[i] as i64)] = 1.0; i += 1; }
        // The best point of the span. The value is a form of the
        // second order in the step. The constant of the form does not
        // move the choice, so the ends hold the form of c1 and of c2
        // plus c1. The step of zero takes the direction and the step
        // of one keeps the point.
        let c2, c1 = qap_quad(a, b, p, q, n);
        var alpha: f64 = 1.0;
        if c2 > 0.0 {
            let t = -c1 / (2.0 * c2);
            if t >= 0.0 && t <= 1.0 { alpha = t; }
            else if c1 + c2 >= 0.0 { alpha = 0.0; }
        } else if c1 + c2 >= 0.0 { alpha = 0.0; }
        var diff: f64 = 0.0;
        i = 0;
        while i < n * n {
            let old = p[i];
            p[i] = alpha * p[i] + (1.0 - alpha) * q[i];
            let d = p[i] - old;
            diff += d * d;
            i += 1;
        }
        if mathshim::root(diff) / mathshim::root(n as f64) < 0.03 { break; }
    }
    // The projection onto the permutations. The linear assignment on
    // the point takes the largest values, as SciPy holds it.
    let _, cols, fg = linear_sum_assignment(mat2(copy(p), n, n), true);
    var out: f64[>..] = [];
    if fg.kind == F_NONE {
        i = 0;
        while i < n { out.push(cols.data[i]); i += 1; }
    } else {
        i = 0;
        while i < n { out.push(i as f64); i += 1; }
    }
    return out;
}

// qap_2opt holds the local search of the swaps. It starts from a
// random permutation, and it walks the pairs of rows in order. A swap
// that lowers the value stays, and the walk starts again. A full walk
// with no improving swap ends the search.
fn qap_2opt(a, b, n: i64, sd: i64) -> f64[>..] {
    var r = default_rng(sd);
    let pm, fp = permutation(r, n);
    var perm: f64[>..] = [];
    var i: i64 = 0;
    if fp.kind == F_NONE {
        i = 0;
        while i < n { perm.push(pm.data[i]); i += 1; }
    } else {
        i = 0;
        while i < n { perm.push(i as f64); i += 1; }
    }
    var best = qap_score(a, b, perm);
    var done = false;
    var gcount: i64 = 0;
    while !done {
        gcount += 1;
        if gcount > 1000000 { break; }
        var swapped = false;
        i = 0;
        while i < n && !swapped {
            var j: i64 = i + 1;
            while j < n {
                let t = perm[i];
                perm[i] = perm[j];
                perm[j] = t;
                let s = qap_score(a, b, perm);
                if s < best {
                    best = s;
                    swapped = true;
                    break;
                }
                perm[j] = perm[i];
                perm[i] = t;
                j += 1;
            }
            i += 1;
        }
        if !swapped { done = true; }
    }
    return perm;
}
