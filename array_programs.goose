// The root finders of one equation and the assignment of rows to columns.
//
// The programs of optimization: linear programs, the least squares of
// non-negative values, the assignment of rows to columns, the
// numerical gradient and two more root finders.
namespace scientific;

import mathshim;

// The largest count of rows or columns that a program takes.
const MAX_PROGRAM_SIDE: i64 = 2048;

fn prog_is_inf(v: f64) -> bool {
    return v - v != 0.0 && v == v;
}

// approx_fprime gives the gradient of a function by the forward
// difference with the step eps.
fn approx_fprime<F: fn(f64[:]) -> f64>(x: f64[:], eps: f64 = 1.4901161193847656e-8) -> Arr, Fault {
    if x.len < 1 || x.len > 4096 {
        return empty_arr(), fault(F_SHAPE, "approx_fprime", "the point holds no value or too many");
    }
    if !(eps > 0.0) {
        return empty_arr(), fault(F_VALUE, "approx_fprime", "the step holds a value of zero or below");
    }
    var p: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len { p.push(x[i]); i += 1; }
    let f0 = F(copy(p));
    var g: f64[>..] = [];
    i = 0;
    while i < x.len {
        p[i] = x[i] + eps;
        g.push((F(copy(p)) - f0) / eps);
        p[i] = x[i];
        i += 1;
    }
    return line_arr(copy(g)), no_fault();
}

// prog_bracket checks that the function changes sign over [a, b].
fn prog_bracket(fa: f64, fb: f64, tag: const u8[:]) -> Fault {
    if !(fa * fb <= 0.0) {
        return fault(F_VALUE, tag, "the function holds the same sign at both ends");
    }
    return no_fault();
}

// bisect finds a root of a function in [a, b] by the cut of the
// interval into halves. The function must change sign over the
// interval.
fn bisect<F: fn(f64) -> f64>(a0: f64, b0: f64, xtol: f64 = 2.0e-12, turns: i64 = 100) -> f64, Fault {
    if !(xtol > 0.0) || turns < 1 {
        return 0.0, fault(F_VALUE, "bisect", "the tolerance or the count of turns holds a value out of range");
    }
    var a = a0;
    var b = b0;
    var fa = F(a);
    let fb = F(b);
    let g = prog_bracket(fa, fb, "bisect");
    if g.kind != F_NONE { return 0.0, g; }
    if fa == 0.0 { return a, no_fault(); }
    if fb == 0.0 { return b, no_fault(); }
    var turn: i64 = 0;
    while turn < turns {
        let m = 0.5 * (a + b);
        let fm = F(m);
        if fm == 0.0 || 0.5 * mathshim::abs_of(b - a) < xtol { return m, no_fault(); }
        if fa * fm < 0.0 { b = m; } else { a = m; fa = fm; }
        turn += 1;
    }
    return 0.0, fault(F_LIMIT, "bisect", "the count of turns ran out before the root");
}

// ridder finds a root of a function in [a, b] by the method of
// Ridders. The function must change sign over the interval.
fn ridder<F: fn(f64) -> f64>(a0: f64, b0: f64, xtol: f64 = 2.0e-12, turns: i64 = 100) -> f64, Fault {
    if !(xtol > 0.0) || turns < 1 {
        return 0.0, fault(F_VALUE, "ridder", "the tolerance or the count of turns holds a value out of range");
    }
    var a = a0;
    var b = b0;
    var fa = F(a);
    var fb = F(b);
    let g = prog_bracket(fa, fb, "ridder");
    if g.kind != F_NONE { return 0.0, g; }
    if fa == 0.0 { return a, no_fault(); }
    if fb == 0.0 { return b, no_fault(); }
    var turn: i64 = 0;
    while turn < turns {
        let m = 0.5 * (a + b);
        let fm = F(m);
        var s = mathshim::root(fm * fm - fa * fb);
        if s == 0.0 { return m, no_fault(); }
        var sg: f64 = 1.0;
        if fa < fb { sg = -1.0; }
        let x = m + (m - a) * sg * fm / s;
        let fx = F(x);
        if fx == 0.0 { return x, no_fault(); }
        if fm * fx < 0.0 { a = m; fa = fm; b = x; fb = fx; }
        else if fa * fx < 0.0 { b = x; fb = fx; }
        else { a = x; fa = fx; }
        if mathshim::abs_of(b - a) < xtol { return 0.5 * (a + b), no_fault(); }
        turn += 1;
    }
    return 0.0, fault(F_LIMIT, "ridder", "the count of turns ran out before the root");
}

// linear_sum_assignment pairs each row of a cost matrix with one
// column so that the total cost is least. It gives the rows and the
// columns of the pairs, by row. The matrix may hold more columns than
// rows or more rows than columns. The cost must be finite. With
// maximize true it takes the pairs of the largest total.
fn linear_sum_assignment(cost, maximize: bool = false) -> Arr, Arr, Fault {
    if cost.ndim != 2 || cost.shape[0] < 1 || cost.shape[1] < 1 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "linear_sum_assignment", "the cost is not a matrix");
    }
    if cost.shape[0] > MAX_PROGRAM_SIDE || cost.shape[1] > MAX_PROGRAM_SIDE {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "linear_sum_assignment", "a side of the matrix is above the limit");
    }
    var i: i64 = 0;
    while i < cost.data.len {
        if !(cost.data[i] - cost.data[i] == 0.0) {
            return empty_arr(), empty_arr(), fault(F_VALUE, "linear_sum_assignment", "the cost holds a value that is not finite");
        }
        i += 1;
    }
    let nr = cost.shape[0];
    let nc = cost.shape[1];
    var tr = false;
    var n = nr;
    var m = nc;
    if nr > nc { tr = true; n = nc; m = nr; }
    // The matrix of n rows and m columns, with n not above m.
    var w: f64[>..] = [];
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < m {
            var v = cost.data[i * nc + j];
            if tr { v = cost.data[j * nc + i]; }
            if maximize { v = -v; }
            w.push(v);
            j += 1;
        }
        i += 1;
    }
    // The method of the potentials, one row at a time.
    var u: f64[>..] = [];
    var vv: f64[>..] = [];
    var p: i64[>..] = [];
    var way: i64[>..] = [];
    i = 0;
    while i <= n { u.push(0.0); i += 1; }
    i = 0;
    while i <= m { vv.push(0.0); p.push(0); way.push(0); i += 1; }
    i = 1;
    while i <= n {
        p[0] = i;
        var j0: i64 = 0;
        var minv: f64[>..] = [];
        var used: bool[>..] = [];
        var k: i64 = 0;
        while k <= m { minv.push(1.0e308); used.push(false); k += 1; }
        var more = true;
        while more {
            used[j0] = true;
            let i0 = p[j0];
            var delta: f64 = 1.0e308;
            var j1: i64 = 0;
            var j: i64 = 1;
            while j <= m {
                if !used[j] {
                    let cur = w[(i0 - 1) * m + (j - 1)] - u[i0] - vv[j];
                    if cur < minv[j] { minv[j] = cur; way[j] = j0; }
                    if minv[j] < delta { delta = minv[j]; j1 = j; }
                }
                j += 1;
            }
            j = 0;
            while j <= m {
                if used[j] { u[p[j]] += delta; vv[j] -= delta; }
                else { minv[j] -= delta; }
                j += 1;
            }
            j0 = j1;
            more = p[j0] != 0;
        }
        var run = true;
        while run {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            run = j0 != 0;
        }
        i += 1;
    }
    // The column of each row of the small side.
    var colof: i64[>..] = [];
    i = 0;
    while i < n { colof.push(0); i += 1; }
    var j: i64 = 1;
    while j <= m { if p[j] != 0 { colof[p[j] - 1] = j - 1; } j += 1; }
    var ro: f64[>..] = [];
    var co: f64[>..] = [];
    if !tr {
        i = 0;
        while i < n { ro.push(i); co.push(colof[i]); i += 1; }
    } else {
        // The rows of the large side, in order.
        var at: i64[>..] = [];
        i = 0;
        while i < nr { at.push(-1); i += 1; }
        i = 0;
        while i < n { at[colof[i]] = i; i += 1; }
        i = 0;
        while i < nr {
            if at[i] >= 0 { ro.push(i); co.push(at[i]); }
            i += 1;
        }
    }
    return line_arr(copy(ro)), line_arr(copy(co)), no_fault();
}

