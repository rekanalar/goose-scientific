// The random laws of a vector.
//
// The calls dirichlet and multivariate_normal follow
// numpy.random.Generator. Each call takes the generator by reference and
// gives one row and a fault. The stream of the generator differs from the
// stream of NumPy. The law is the same: the mean and the covariance
// agree.
//
// NumPy only warns about a covariance that is not symmetric and not
// positive semidefinite, and it still draws a number. This package
// refuses such a covariance with F_VALUE, because the number of such a
// draw follows no law.
namespace scientific;

import std;
import mathshim;

// The largest side of the covariance matrix. The eigen decomposition
// of the matrix costs the fourth power of the side in the worst case.
const MVN_DIM_MAX: i64 = 64;
// The relative size by which a covariance may leave the symmetric form
// and by which an eigenvalue may fall below zero. NumPy uses the same
// value as its tolerance.
const MVN_TOL: f64 = 1.0e-8;

// dirichlet gives one point of the simplex from the row of the
// concentrations. Each value of the point comes from a gamma number of
// the concentration. The call divides the numbers by their sum. It
// works with the logarithms, so a small concentration cannot give zero
// over zero. A concentration at or below zero, or not finite, gives
// F_VALUE. A concentration outside 1.0e-300 to 1.0e100 gives F_LIMIT.
fn dirichlet(r: Rng&, alpha) -> Arr, Fault {
    let fs = check_stack(alpha.ndim, alpha.shape, alpha.data.len, 1, "dirichlet");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    if alpha.ndim != 1 {
        return empty_arr(), fault(F_SHAPE, "dirichlet", "the concentrations hold a count of dimensions that is not one");
    }
    var i: i64 = 0;
    while i < alpha.data.len {
        let fp = dist_check_pos(alpha.data[i], "dirichlet", "a concentration is not above zero");
        if fp.kind != F_NONE { return empty_arr(), fp; }
        let fl = dist_check_shape(alpha.data[i], "dirichlet");
        if fl.kind != F_NONE { return empty_arr(), fl; }
        i += 1;
    }
    var logs: f64[>..] = [];
    var top: f64 = 0.0;
    i = 0;
    while i < alpha.data.len {
        let lg, fg = gamma_log(r, alpha.data[i], "dirichlet");
        if fg.kind != F_NONE { return empty_arr(), fg; }
        logs.push(lg);
        if i == 0 || lg > top { top = lg; }
        i += 1;
    }
    // The largest term becomes one, so the sum is at least one.
    var total: f64 = 0.0;
    i = 0;
    while i < logs.len {
        logs[i] = mathshim::power_e(logs[i] - top);
        total += logs[i];
        i += 1;
    }
    if total <= 0.0 {
        return empty_arr(), fault(F_VALUE, "dirichlet", "the sum of the terms stands at or below zero");
    }
    i = 0;
    while i < logs.len { logs[i] = logs[i] / total; i += 1; }
    var s: i64[..8] = [];
    s.push(logs.len);
    return Arr { ndim: 1, shape: s, data: logs }, no_fault();
}

// mvn_check checks the mean and the covariance. The mean holds one
// dimension. The covariance holds as many rows and columns as the mean
// holds values. Every value is finite. The covariance is symmetric and
// positive semidefinite, each inside the tolerance.
fn mvn_check(mean, cov) -> Fault {
    let fm = check_stack(mean.ndim, mean.shape, mean.data.len, 1, "multivariate_normal");
    if fm.kind != F_NONE { return fm; }
    let fc = check_stack(cov.ndim, cov.shape, cov.data.len, 1, "multivariate_normal");
    if fc.kind != F_NONE { return fc; }
    if mean.ndim != 1 || cov.ndim != 2 || cov.shape[0] != mean.data.len || cov.shape[1] != mean.data.len {
        return fault(F_SHAPE, "multivariate_normal", "the mean and the covariance do not have the same length");
    }
    if mean.data.len > MVN_DIM_MAX {
        return fault(F_LIMIT, "multivariate_normal", "the mean holds more than 64 values");
    }
    if !all_finite(mean.data) || !all_finite(cov.data) {
        return fault(F_VALUE, "multivariate_normal", "the mean or the covariance holds a value that is not finite");
    }
    let n = mean.data.len;
    let scale = max_abs(cov.data);
    var row: i64 = 0;
    while row < n {
        var col: i64 = row + 1;
        while col < n {
            let gap = mathshim::abs_of(cov.data[row * n + col] - cov.data[col * n + row]);
            if gap > MVN_TOL * scale {
                return fault(F_VALUE, "multivariate_normal", "the covariance is not symmetric");
            }
            col += 1;
        }
        row += 1;
    }
    return no_fault();
}

// multivariate_normal gives one point of the normal law of a vector. The
// point is the mean plus the product of the root of the covariance and
// a row of standard normal numbers. The root comes from the eigen
// decomposition of the covariance. An eigenvalue that falls below zero
// by no more than the tolerance counts as zero. A covariance that is
// not symmetric, or that holds an eigenvalue below the tolerance, gives
// F_VALUE. A singular covariance gives a point on a line or a plane.
fn multivariate_normal(r: Rng&, mean, cov) -> Arr, Fault {
    let fc = mvn_check(mean, cov);
    if fc.kind != F_NONE { return empty_arr(), fc; }
    let n = mean.data.len;
    let vals, vecs, fe = eigh(cov);
    if fe.kind != F_NONE { return empty_arr(), fe; }
    let scale = max_abs(vals.data);
    var root: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        if vals.data[i] < -MVN_TOL * scale {
            return empty_arr(), fault(F_VALUE, "multivariate_normal", "the covariance is not positive semidefinite");
        }
        var v = vals.data[i];
        if v < 0.0 { v = 0.0; }
        root.push(mathshim::root(v));
        i += 1;
    }
    var out = copy(mean.data);
    var j: i64 = 0;
    while j < n {
        let z, fz = normal(r, 0.0, 1.0);
        if fz.kind != F_NONE { return empty_arr(), fz; }
        let w = z * root[j];
        i = 0;
        while i < n {
            out[i] += w * vecs.data[i * n + j];
            i += 1;
        }
        j += 1;
    }
    if !all_finite(out) {
        return empty_arr(), fault(F_VALUE, "multivariate_normal", "the result is not finite");
    }
    var s: i64[..8] = [];
    s.push(out.len);
    return Arr { ndim: 1, shape: s, data: out }, no_fault();
}
