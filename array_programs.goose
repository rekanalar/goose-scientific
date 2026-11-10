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

// nnls solves the least squares of a x = b with x not below zero, by
// the method of Lawson and Hanson. It gives x and the norm of the
// residual.
fn nnls(a, b: f64[:], turns: i64 = 0) -> Arr, f64, Fault {
    if a.ndim != 2 || a.shape[0] != b.len || b.len < 1 || a.shape[1] < 1 {
        return empty_arr(), 0.0, fault(F_SHAPE, "nnls", "the matrix and the vector do not fit");
    }
    if a.shape[0] > MAX_PROGRAM_SIDE || a.shape[1] > MAX_PROGRAM_SIDE {
        return empty_arr(), 0.0, fault(F_LIMIT, "nnls", "a side of the matrix is above the limit");
    }
    let m = a.shape[0];
    let n = a.shape[1];
    var limit = turns;
    if limit < 1 { limit = 3 * n; }
    var x: f64[>..] = [];
    var pas: bool[>..] = [];
    var i: i64 = 0;
    while i < n { x.push(0.0); pas.push(false); i += 1; }
    var outer: i64 = 0;
    while outer < limit + n {
        // The gradient w = A'(b - A x).
        var r: f64[>..] = [];
        i = 0;
        while i < m {
            var s = b[i];
            var j: i64 = 0;
            while j < n { s -= a.data[i * n + j] * x[j]; j += 1; }
            r.push(s);
            i += 1;
        }
        var best: i64 = -1;
        var bw: f64 = 1.0e-12;
        var j: i64 = 0;
        while j < n {
            if !pas[j] {
                var wj: f64 = 0.0;
                i = 0;
                while i < m { wj += a.data[i * n + j] * r[i]; i += 1; }
                if wj > bw { bw = wj; best = j; }
            }
            j += 1;
        }
        if best < 0 { break; }
        pas[best] = true;
        var inner: i64 = 0;
        var go = true;
        while go && inner < 3 * n + 10 {
            inner += 1;
            // The free solution on the passive set.
            var idx: i64[>..] = [];
            j = 0;
            while j < n { if pas[j] { idx.push(j); } j += 1; }
            let k = idx.len;
            var g: f64[>..] = [];
            var h: f64[>..] = [];
            var p: i64 = 0;
            while p < k {
                var q: i64 = 0;
                while q < k {
                    var s: f64 = 0.0;
                    i = 0;
                    while i < m { s += a.data[i * n + idx[p]] * a.data[i * n + idx[q]]; i += 1; }
                    g.push(s);
                    q += 1;
                }
                var s2: f64 = 0.0;
                i = 0;
                while i < m { s2 += a.data[i * n + idx[p]] * b[i]; i += 1; }
                h.push(s2);
                p += 1;
            }
            let ga = mat2(copy(g), k, k);
            let hb = line_arr(copy(h));
            let z, fz = solve(ga, hb);
            if fz.kind != F_NONE {
                return empty_arr(), 0.0, fault(F_VALUE, "nnls", "the passive columns hold a singular system");
            }
            var allpos = true;
            p = 0;
            while p < k { if z.data[p] <= 0.0 { allpos = false; } p += 1; }
            if allpos {
                p = 0;
                while p < k { x[idx[p]] = z.data[p]; p += 1; }
                go = false;
            } else {
                var alpha: f64 = 1.0e308;
                p = 0;
                while p < k {
                    if z.data[p] <= 0.0 {
                        let t = x[idx[p]] / (x[idx[p]] - z.data[p]);
                        if t < alpha { alpha = t; }
                    }
                    p += 1;
                }
                p = 0;
                while p < k {
                    x[idx[p]] = x[idx[p]] + alpha * (z.data[p] - x[idx[p]]);
                    p += 1;
                }
                p = 0;
                while p < k {
                    if mathshim::abs_of(x[idx[p]]) < 1.0e-14 { x[idx[p]] = 0.0; pas[idx[p]] = false; }
                    p += 1;
                }
            }
        }
        outer += 1;
    }
    var rn: f64 = 0.0;
    i = 0;
    while i < m {
        var s = b[i];
        var j: i64 = 0;
        while j < n { s -= a.data[i * n + j] * x[j]; j += 1; }
        rn += s * s;
        i += 1;
    }
    return line_arr(copy(x)), mathshim::root(rn), no_fault();
}

// simplex_pivot pivots the tableau on the row r and the column c.
fn simplex_pivot(t: f64[>..]&, w: i64, rows: i64, r: i64, c: i64) {
    let pv = t[r * w + c];
    var j: i64 = 0;
    while j < w { t[r * w + j] = t[r * w + j] / pv; j += 1; }
    var i: i64 = 0;
    while i < rows {
        if i != r {
            let f = t[i * w + c];
            if f != 0.0 {
                j = 0;
                while j < w { t[i * w + j] = t[i * w + j] - f * t[r * w + j]; j += 1; }
            }
        }
        i += 1;
    }
}

// simplex_run runs the simplex rule of Bland on a tableau. The last
// row holds the reduced costs. The columns from the index allowed on
// do not enter. It gives 0 for an optimum, 1 for an unbounded problem
// and 2 for a count of turns that ran out.
fn simplex_run(t: f64[>..]&, w: i64, m: i64, allowed: i64, basis: i64[>..]&) -> i64 {
    var turn: i64 = 0;
    while turn < 200000 {
        var c: i64 = -1;
        var j: i64 = 0;
        while j < allowed { if t[m * w + j] < -1.0e-10 { c = j; break; } j += 1; }
        if c < 0 { return 0; }
        var r: i64 = -1;
        var best: f64 = 0.0;
        var i: i64 = 0;
        while i < m {
            let a = t[i * w + c];
            if a > 1.0e-10 {
                let q = t[i * w + w - 1] / a;
                if r < 0 || q < best - 1.0e-12 || (q < best + 1.0e-12 && basis[i] < basis[r]) { r = i; best = q; }
            }
            i += 1;
        }
        if r < 0 { return 1; }
        simplex_pivot(t, w, m + 1, r, c);
        basis[r] = c;
        turn += 1;
    }
    return 2;
}

// linprog minimizes c x under a_ub x <= b_ub, a_eq x = b_eq and the
// bounds lb <= x <= ub. The matrices are flat lists by row, with one
// row per element of the right side and one column per element of c.
// An empty list means no row. The lists lb and ub hold one bound per
// variable, or no value for the defaults zero and infinity. A bound of
// minus infinity or infinity is free. The method is the two phase
// simplex with the rule of Bland. It gives x and the value. A problem
// with no point, or with no least value, is a fault.
fn linprog(c: f64[:], a_ub: f64[:], b_ub: f64[:], a_eq: f64[:], b_eq: f64[:], lb: f64[:] = [], ub: f64[:] = []) -> Arr, f64, Fault {
    let nv = c.len;
    if nv < 1 || nv > MAX_PROGRAM_SIDE {
        return empty_arr(), 0.0, fault(F_SHAPE, "linprog", "the cost holds no value or too many");
    }
    if a_ub.len != b_ub.len * nv || a_eq.len != b_eq.len * nv {
        return empty_arr(), 0.0, fault(F_SHAPE, "linprog", "a matrix does not fit its right side");
    }
    if (lb.len != 0 && lb.len != nv) || (ub.len != 0 && ub.len != nv) {
        return empty_arr(), 0.0, fault(F_SHAPE, "linprog", "the bounds do not hold one value per variable");
    }
    if b_ub.len + b_eq.len + nv > MAX_PROGRAM_SIDE {
        return empty_arr(), 0.0, fault(F_LIMIT, "linprog", "the program holds more rows than the limit");
    }
    var i: i64 = 0;
    while i < nv {
        if !(c[i] - c[i] == 0.0) { return empty_arr(), 0.0, fault(F_VALUE, "linprog", "the cost holds a value that is not finite"); }
        i += 1;
    }
    // The map of each variable: kind, first column, offset, sign.
    var kind: i64[>..] = [];
    var col: i64[>..] = [];
    var off: f64[>..] = [];
    var ncol: i64 = 0;
    var nbound: i64 = 0;
    i = 0;
    while i < nv {
        var lo: f64 = 0.0;
        var hi: f64 = 1.0 / 0.0;
        if lb.len > 0 { lo = lb[i]; }
        if ub.len > 0 { hi = ub[i]; }
        if lo != lo || hi != hi || lo > hi || (prog_is_inf(lo) && lo > 0.0) || (prog_is_inf(hi) && hi < 0.0) {
            return empty_arr(), 0.0, fault(F_VALUE, "linprog", "a bound holds a value out of range");
        }
        col.push(ncol);
        if !prog_is_inf(lo) {
            kind.push(0); off.push(lo); ncol += 1;
            if !prog_is_inf(hi) { nbound += 1; }
        } else if !prog_is_inf(hi) {
            kind.push(1); off.push(hi); ncol += 1;
        } else {
            kind.push(2); off.push(0.0); ncol += 2;
        }
        i += 1;
    }
    let mub = b_ub.len;
    let meq = b_eq.len;
    let m = mub + meq + nbound;
    let nslack = mub + nbound;
    let ntot = ncol + nslack;
    let w = ntot + m + 1;
    // The rows of the tableau, with the artificial columns after the others.
    var t: f64[>..] = [];
    i = 0;
    while i < (m + 1) * w { t.push(0.0); i += 1; }
    var row: i64 = 0;
    var pass: i64 = 0;
    while pass < 3 {
        var count = mub;
        if pass == 1 { count = meq; }
        if pass == 2 { count = 0; }
        var k: i64 = 0;
        while k < count {
            var rhs: f64 = 0.0;
            if pass == 0 { rhs = b_ub[k]; } else { rhs = b_eq[k]; }
            var v: i64 = 0;
            while v < nv {
                var a: f64 = 0.0;
                if pass == 0 { a = a_ub[k * nv + v]; } else { a = a_eq[k * nv + v]; }
                if kind[v] == 0 { t[row * w + col[v]] = a; rhs -= a * off[v]; }
                else if kind[v] == 1 { t[row * w + col[v]] = -a; rhs -= a * off[v]; }
                else { t[row * w + col[v]] = a; t[row * w + col[v] + 1] = -a; }
                v += 1;
            }
            if pass == 0 { t[row * w + ncol + k] = 1.0; }
            t[row * w + w - 1] = rhs;
            row += 1;
            k += 1;
        }
        pass += 1;
    }
    var bi: i64 = 0;
    i = 0;
    while i < nv {
        if kind[i] == 0 {
            var hi: f64 = 1.0 / 0.0;
            if ub.len > 0 { hi = ub[i]; }
            if !prog_is_inf(hi) {
                t[row * w + col[i]] = 1.0;
                t[row * w + ncol + mub + bi] = 1.0;
                t[row * w + w - 1] = hi - off[i];
                row += 1;
                bi += 1;
            }
        }
        i += 1;
    }
    // Keep the right sides above zero, then add the artificial columns.
    var basis: i64[>..] = [];
    i = 0;
    while i < m {
        if t[i * w + w - 1] < 0.0 {
            var j: i64 = 0;
            while j < w { t[i * w + j] = -t[i * w + j]; j += 1; }
        }
        t[i * w + ntot + i] = 1.0;
        basis.push(ntot + i);
        i += 1;
    }
    // The cost row of the first phase.
    var j: i64 = 0;
    while j < ntot {
        var s: f64 = 0.0;
        i = 0;
        while i < m { s += t[i * w + j]; i += 1; }
        t[m * w + j] = -s;
        j += 1;
    }
    var s0: f64 = 0.0;
    i = 0;
    while i < m { s0 += t[i * w + w - 1]; i += 1; }
    t[m * w + w - 1] = -s0;
    var rc = simplex_run(t, w, m, ntot, basis);
    if rc == 2 { return empty_arr(), 0.0, fault(F_LIMIT, "linprog", "the simplex ran out of turns"); }
    var scale: f64 = 1.0;
    i = 0;
    while i < m { let q = mathshim::abs_of(t[i * w + w - 1]); if q > scale { scale = q; } i += 1; }
    if -t[m * w + w - 1] > 1.0e-9 * scale {
        return empty_arr(), 0.0, fault(F_VALUE, "linprog", "the problem holds no feasible point");
    }
    // Push the artificial variables out of the basis.
    i = 0;
    while i < m {
        if basis[i] >= ntot {
            j = 0;
            while j < ntot {
                if mathshim::abs_of(t[i * w + j]) > 1.0e-9 {
                    simplex_pivot(t, w, m + 1, i, j);
                    basis[i] = j;
                    break;
                }
                j += 1;
            }
        }
        i += 1;
    }
    // The cost row of the second phase.
    var cost: f64[>..] = [];
    j = 0;
    while j < w { cost.push(0.0); j += 1; }
    i = 0;
    while i < nv {
        if kind[i] == 0 { cost[col[i]] = c[i]; }
        else if kind[i] == 1 { cost[col[i]] = -c[i]; }
        else { cost[col[i]] = c[i]; cost[col[i] + 1] = -c[i]; }
        i += 1;
    }
    j = 0;
    while j < w { t[m * w + j] = cost[j]; j += 1; }
    i = 0;
    while i < m {
        let cb = cost[basis[i]];
        if basis[i] < ntot && cb != 0.0 {
            j = 0;
            while j < w { t[m * w + j] = t[m * w + j] - cb * t[i * w + j]; j += 1; }
        }
        i += 1;
    }
    rc = simplex_run(t, w, m, ntot, basis);
    if rc == 1 { return empty_arr(), 0.0, fault(F_VALUE, "linprog", "the problem holds no least value"); }
    if rc == 2 { return empty_arr(), 0.0, fault(F_LIMIT, "linprog", "the simplex ran out of turns"); }
    var y: f64[>..] = [];
    j = 0;
    while j < ntot { y.push(0.0); j += 1; }
    i = 0;
    while i < m { if basis[i] < ntot { y[basis[i]] = t[i * w + w - 1]; } i += 1; }
    var x: f64[>..] = [];
    var fun: f64 = 0.0;
    i = 0;
    while i < nv {
        var xv: f64 = 0.0;
        if kind[i] == 0 { xv = off[i] + y[col[i]]; }
        else if kind[i] == 1 { xv = off[i] - y[col[i]]; }
        else { xv = y[col[i]] - y[col[i] + 1]; }
        x.push(xv);
        fun += c[i] * xv;
        i += 1;
    }
    return line_arr(copy(x)), fun, no_fault();
}
