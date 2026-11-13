// Non-negative least squares and the linear and integer programs.
//
namespace scientific;

import mathshim;

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

// milp minimizes c x under the rows of linprog and the bounds, where
// the variables flagged in integer take whole values. The method is
// branch and bound over the relaxations of linprog. It takes the most
// fractional variable first, and it explores the deeper node first. A
// problem with no point is a fault, and so is a relaxation with no
// least value. A search of more than 20000 nodes is a limit fault.
fn milp(c: f64[:], integer: bool[:], a_ub: f64[:], b_ub: f64[:], a_eq: f64[:], b_eq: f64[:], lb: f64[:] = [], ub: f64[:] = []) -> Arr, f64, Fault {
    let nv = c.len;
    if integer.len != nv {
        return empty_arr(), 0.0, fault(F_SHAPE, "milp", "the flags do not hold one value per variable");
    }
    if (lb.len != 0 && lb.len != nv) || (ub.len != 0 && ub.len != nv) {
        return empty_arr(), 0.0, fault(F_SHAPE, "milp", "the bounds do not hold one value per variable");
    }
    var nlb: f64[>..] = [];
    var nub: f64[>..] = [];
    var i: i64 = 0;
    while i < nv {
        if lb.len > 0 { nlb.push(lb[i]); } else { nlb.push(0.0); }
        if ub.len > 0 { nub.push(ub[i]); } else { nub.push(1.0 / 0.0); }
        i += 1;
    }
    // The stack of the nodes holds the bounds of each node.
    var slb: f64[>..<] = copy_slice(nlb);
    var sub: f64[>..<] = copy_slice(nub);
    var best: f64[>..] = [];
    var bestv: f64 = 1.0 / 0.0;
    var found = false;
    var nodes: i64 = 0;
    while slb.len > 0 {
        nodes += 1;
        if nodes > 20000 {
            return empty_arr(), 0.0, fault(F_LIMIT, "milp", "the search holds more nodes than the limit");
        }
        var clb: f64[>..] = [];
        var cub: f64[>..] = [];
        i = 0;
        while i < nv {
            clb.push(slb[slb.len - nv + i]);
            cub.push(sub[sub.len - nv + i]);
            i += 1;
        }
        i = 0;
        while i < nv { slb.pop(); sub.pop(); i += 1; }
        var skip = false;
        i = 0;
        while i < nv { if clb[i] > cub[i] { skip = true; } i += 1; }
        if skip { continue; }
        let x, v, f = linprog(c, a_ub, b_ub, a_eq, b_eq, clb, cub);
        if f.kind != F_NONE {
            if f.kind == F_VALUE && f.detail == "the problem holds no feasible point" { continue; }
            return empty_arr(), 0.0, fault(f.kind, "milp", f.detail);
        }
        if found && v >= bestv - 0.000000001 { continue; }
        // The most fractional variable.
        var pick: i64 = -1;
        var worst: f64 = 0.000001;
        i = 0;
        while i < nv {
            if integer[i] {
                var fr = x.data[i] - mathshim::floor_of(x.data[i]);
                if fr > 0.5 { fr = 1.0 - fr; }
                if fr > worst { worst = fr; pick = i; }
            }
            i += 1;
        }
        if pick < 0 {
            best = copy_slice(x.data);
            i = 0;
            while i < nv { if integer[i] { best[i] = mathshim::round_of(best[i]); } i += 1; }
            bestv = v;
            found = true;
            continue;
        }
        let fl = mathshim::floor_of(x.data[pick]);
        // The node of the upper part is pushed first, so the lower
        // part runs first.
        i = 0;
        while i < nv {
            slb.push(clb[i]);
            if i == pick { sub.push(cub[i]); } else { sub.push(cub[i]); }
            i += 1;
        }
        slb[slb.len - nv + pick] = fl + 1.0;
        i = 0;
        while i < nv { slb.push(clb[i]); sub.push(cub[i]); i += 1; }
        sub[sub.len - nv + pick] = fl;
    }
    if !found {
        return empty_arr(), 0.0, fault(F_VALUE, "milp", "the problem holds no feasible point");
    }
    var s: i64[..8] = [];
    s.push(nv);
    return Arr { ndim: 1, shape: s, data: best }, bestv, no_fault();
}
