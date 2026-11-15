// The solver of the boundary value problems.
//
// The call follows solve_bvp of SciPy. The method is the collocation
// of the fourth order: the answer is a cubic spline whose derivative
// matches the right side at the nodes and at the middle of every
// interval. The nonlinear system goes through a damped Newton method
// with the criterion of the affine change. The mesh refines where the
// estimate of the residual is large. The method comes from the papers
// of Kierzenka and Shampine, 2001, and of Ascher, Mattheij and
// Russell, 1995.
//
namespace scientific;

import mathshim;

// The count of the unknowns that the dense solver takes. A system
// above the limit gives a fault.
const BVP_MAX_N: i64 = 2048;

// The base of the steps of the forward differences.
const BVP_EPS: f64 = 2.220446049250313e-16;

// bvp_assemble builds the matrix of the collocation system in
// row-major order. The rows of the collocation come first and the
// rows of the boundary conditions after them. The buffers dy and dym
// hold the derivatives of the right side over the values at the nodes
// and at the middle points, dp and dpm the ones over the parameters,
// and dya, dyb and dpc the ones of the boundary conditions.
fn bvp_assemble(h: f64[:], n: i64, k: i64, m: i64, dy: f64[:], dym: f64[:], dp: f64[:], dpm: f64[:], dya: f64[:], dyb: f64[:], dpc: f64[:]) -> f64[>..] {
    let big = m * n;
    let big_n = big + k;
    // The rows hold the residuals of the intervals and the residuals
    // of the conditions. The columns hold the values of the nodes and
    // the parameters.
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < big_n * big_n { out.push(0.0); i += 1; }
    var q: i64 = 0;
    while q < m - 1 {
        var j: i64 = 0;
        while j < n {
            // The row of the residual of the component j in the
            // interval q.
            let row = q * n + j;
            var t: i64 = 0;
            while t < n {
                // The derivative over the values of the left node. The
                // middle point of the interval depends on the values of
                // the two nodes, so the derivative of the function
                // there enters the row.
                var s0: f64 = 0.0;
                var s2: i64 = 0;
                while s2 < n {
                    var dm = dym[(s2 * (m - 1) + q) * n + j];
                    var mk = 0.0;
                    if s2 == t { mk = 0.5; }
                    mk += h[q] / 8.0 * dy[(t * m + q) * n + s2];
                    s0 += dm * mk;
                    s2 += 1;
                }
                var a0: f64 = 0.0;
                if t == j { a0 = -1.0; }
                a0 -= h[q] / 6.0 * (dy[(t * m + q) * n + j] + 4.0 * s0);
                out[row * big_n + q * n + t] = a0;
                // The derivative over the values of the right node.
                var s1: f64 = 0.0;
                s2 = 0;
                while s2 < n {
                    var dm = dym[(s2 * (m - 1) + q) * n + j];
                    var mk = 0.0;
                    if s2 == t { mk = 0.5; }
                    mk -= h[q] / 8.0 * dy[(t * m + q + 1) * n + s2];
                    s1 += dm * mk;
                    s2 += 1;
                }
                var b0: f64 = 0.0;
                if t == j { b0 = 1.0; }
                b0 -= h[q] / 6.0 * (dy[(t * m + q + 1) * n + j] + 4.0 * s1);
                out[row * big_n + (q + 1) * n + t] = b0;
                t += 1;
            }
            if k > 0 {
                // The derivative over the parameters. The middle point
                // does not depend on the parameters, so the derivative
                // of the function there enters the row as it is.
                var c: i64 = 0;
                while c < k {
                    out[row * big_n + big + c] = -h[q] / 6.0 * (dp[(c * m + q) * n + j]
                        + dp[(c * m + q + 1) * n + j]
                        + 4.0 * dpm[(c * (m - 1) + q) * n + j]);
                    c += 1;
                }
            }
            j += 1;
        }
        q += 1;
    }
    // The rows of the boundary conditions. They follow the rows of the
    // intervals, so the system holds one row for each value of the
    // answer.
    let nbc = n + k;
    let brow = (m - 1) * n;
    var r: i64 = 0;
    while r < nbc {
        var c: i64 = 0;
        while c < n {
            out[(brow + r) * big_n + c] = dya[r * n + c];
            out[(brow + r) * big_n + (m - 1) * n + c] = dyb[r * n + c];
            c += 1;
        }
        c = 0;
        while c < k {
            out[(brow + r) * big_n + big + c] = dpc[r * k + c];
            c += 1;
        }
        r += 1;
    }
    return out;
}

// solve_bvp solves the system y' = F(x, y, p) with the conditions
// B(y(a), y(b), p) = 0. The block F gives the n derivatives at one
// place and the block B gives the n plus k residuals of the
// conditions at the two ends. The mesh x rises strictly. The guess y
// holds one row of n values per node. The parameters p holds the
// values that the call finds with the answer. It gives the mesh, the
// answer with one row per node, the parameters and the estimate of
// the residual of every interval.
fn solve_bvp<B: fn(f64[:], f64[:], f64[:]) -> f64[:], F: fn(f64, f64[:], f64[:]) -> f64[:]>(x: f64[:], y: Arr, p: f64[:] = [], tol: f64 = 1.0e-3, max_nodes: i64 = 1000) -> Arr, Arr, Arr, Arr, Fault {
    let fa = check_stack(y.ndim, y.shape, y.data.len, 1, "solve_bvp");
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fa; }
    if x.len < 2 || y.ndim != 2 || y.shape[0] != x.len {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "solve_bvp", "the mesh or the guess holds the wrong count of nodes");
    }
    let m0 = x.len;
    let n = y.shape[1];
    if n < 1 || m0 * n + p.len > BVP_MAX_N {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_LIMIT, "solve_bvp", "the system holds more unknowns than the limit");
    }
    var m = m0;
    var q: i64 = 0;
    while q < m0 {
        if !(x[q] - x[q] == 0.0) || (q > 0 && x[q] <= x[q - 1]) {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "solve_bvp", "the mesh does not rise strictly or holds a value that is not finite");
        }
        q += 1;
    }
    if !all_finite(y.data) {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "solve_bvp", "the guess holds a value that is not finite");
    }
    var tol2 = tol;
    if tol2 < 100.0 * BVP_EPS { tol2 = 100.0 * BVP_EPS; }
    if !(tol2 > 0.0) || max_nodes < m0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_RANGE, "solve_bvp", "the tolerance or the count of nodes holds a value out of range");
    }
    let bc_tol = tol2;
    var xc: f64[>..] = [];
    var yc: f64[>..] = [];
    q = 0;
    while q < m0 { xc.push(x[q]); q += 1; }
    q = 0;
    while q < m0 * n { yc.push(y.data[q]); q += 1; }
    var pc: f64[>..] = [];
    q = 0;
    while q < p.len { pc.push(p[q]); q += 1; }
    let k = p.len;
    var h: f64[>..] = [];
    var fr: f64[>..] = [];
    var fm: f64[>..] = [];
    var cm: f64[>..] = [];
    var ymid: f64[>..] = [];
    var xmid: f64[>..] = [];
    var b0: f64[>..] = [];
    var ya: f64[>..] = [];
    var yb: f64[>..] = [];
    var run = true;
    var iteration: i64 = 0;
    while run {
        run = false;
        h = [];
        q = 0;
        while q < m - 1 { h.push(xc[q + 1] - xc[q]); q += 1; }
        xmid = [];
        q = 0;
        while q < m - 1 { xmid.push(xc[q] + 0.5 * h[q]); q += 1; }
        let fv, fe = bvp_f_evals(xc, yc, pc, n, F);
        if fe.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fe; }
        fr = copy(fv);
        ymid = bvp_middle(yc, fr, h, m, n);
        let fmv, fme = bvp_f_evals(xmid, ymid, pc, n, F);
        if fme.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fme; }
        fm = copy(fmv);
        cm = bvp_col_res(yc, fr, fm, h, m, n);
        ya = [];
        yb = [];
        q = 0;
        while q < n { ya.push(yc[q]); yb.push(yc[(m - 1) * n + q]); q += 1; }
        b0 = B(ya, yb, pc);
        if b0.len != n + k {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "solve_bvp", "the boundary conditions do not hold one value per condition");
        }
        if !all_finite(b0) {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "solve_bvp", "the boundary conditions hold a value that is not finite");
        }
        // The Newton method with the criterion of the affine change.
        let big_n = m * n + k;
        var jmat = empty_arr();
        var tol_r: f64[>..] = [];
        q = 0;
        while q < m - 1 { tol_r.push(2.0 / 3.0 * h[q] * 0.05 * tol2); q += 1; }
        var njev: i64 = 0;
        var recompute = true;
        var step: f64[>..] = [];
        var cost: f64 = 0.0;
        var it: i64 = 0;
        while it < 8 {
            if recompute {
                // The derivatives by the forward differences.
                let dy, j1 = bvp_fun_jac(0, xc, yc, pc, n, fr, F);
                if j1.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), j1; }
                let dym, j2 = bvp_fun_jac(0, xmid, ymid, pc, n, fm, F);
                if j2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), j2; }
                var dp: f64[>..] = [];
                var dpm: f64[>..] = [];
                var dya: f64[>..] = [];
                var dyb: f64[>..] = [];
                var dpc: f64[>..] = [];
                if k > 0 {
                    let dpv, j3 = bvp_fun_jac(1, xc, yc, pc, n, fr, F);
                    if j3.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), j3; }
                    let dpmv, j4 = bvp_fun_jac(1, xmid, ymid, pc, n, fm, F);
                    if j4.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), j4; }
                    dp = copy(dpv);
                    dpm = copy(dpmv);
                }
                let jyav, j5 = bvp_bc_jac(0, ya, yb, pc, b0, B);
                if j5.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), j5; }
                let jybv, j6 = bvp_bc_jac(1, ya, yb, pc, b0, B);
                if j6.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), j6; }
                dya = copy(jyav);
                dyb = copy(jybv);
                if k > 0 {
                    let dpcv, j7 = bvp_bc_jac(2, ya, yb, pc, b0, B);
                    if j7.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), j7; }
                    dpc = copy(dpcv);
                }
                let jm = bvp_assemble(h, n, k, m, dy, dym, dp, dpm, dya, dyb, dpc);
                jmat = mat2(copy(jm), big_n, big_n);
                var rv: f64[>..] = [];
                q = 0;
                while q < cm.len { rv.push(cm[q]); q += 1; }
                q = 0;
                while q < b0.len { rv.push(b0[q]); q += 1; }
                let sv, slf = solve(jmat, line_arr(copy(rv)));
                if slf.kind != F_NONE {
                    return empty_arr(), empty_arr(), empty_arr(), empty_arr(), slf;
                }
                step = [];
                q = 0;
                while q < big_n { step.push(sv.data[q]); q += 1; }
                cost = 0.0;
                q = 0;
                while q < big_n { cost += step[q] * step[q]; q += 1; }
                njev += 1;
            }
            // The line search over the size of the Newton step.
            var alpha: f64 = 1.0;
            var y2: f64[>..] = [];
            var p2: f64[>..] = [];
            var cm2: f64[>..] = [];
            var fm2: f64[>..] = [];
            var mid2: f64[>..] = [];
            var f2v: f64[>..] = [];
            var b2: f64[>..] = [];
            var step2: f64[>..] = [];
            var cost2: f64 = 0.0;
            var trial: i64 = 0;
            while trial <= 4 {
                y2 = [];
                q = 0;
                while q < m * n { y2.push(yc[q] - alpha * step[q]); q += 1; }
                p2 = [];
                q = 0;
                while q < k { p2.push(pc[q] - alpha * step[m * n + q]); q += 1; }
                let f2a, f2b = bvp_f_evals(xc, y2, p2, n, F);
                if f2b.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), f2b; }
                f2v = copy(f2a);
                mid2 = bvp_middle(y2, f2v, h, m, n);
                let fm2a, fm2b = bvp_f_evals(xmid, mid2, p2, n, F);
                if fm2b.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fm2b; }
                fm2 = copy(fm2a);
                cm2 = bvp_col_res(y2, f2v, fm2, h, m, n);
                q = 0;
                while q < n { ya[q] = y2[q]; yb[q] = y2[(m - 1) * n + q]; q += 1; }
                b2 = B(ya, yb, p2);
                if !all_finite(b2) {
                    return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "solve_bvp", "the boundary conditions hold a value that is not finite");
                }
                var rv2: f64[>..] = [];
                q = 0;
                while q < cm2.len { rv2.push(cm2[q]); q += 1; }
                q = 0;
                while q < b2.len { rv2.push(b2[q]); q += 1; }
                let sv2, slf2 = solve(jmat, line_arr(copy(rv2)));
                if slf2.kind != F_NONE {
                    return empty_arr(), empty_arr(), empty_arr(), empty_arr(), slf2;
                }
                step2 = [];
                q = 0;
                while q < big_n { step2.push(sv2.data[q]); q += 1; }
                cost2 = 0.0;
                q = 0;
                while q < big_n { cost2 += step2[q] * step2[q]; q += 1; }
                if cost2 < (1.0 - 2.0 * alpha * 0.2) * cost { break; }
                if trial < 4 { alpha *= 0.5; }
                trial += 1;
            }
            yc = copy(y2);
            pc = copy(p2);
            cm = copy(cm2);
            fm = copy(fm2);
            ymid = copy(mid2);
            fr = copy(f2v);
            b0 = copy(b2);
            it += 1;
            if njev == 4 { break; }
            // The test of the end of the Newton run. The residual of
            // every interval and the residual of the conditions must
            // lie under the tolerance.
            var done = true;
            q = 0;
            while q < cm.len {
                if mathshim::abs_of(cm[q]) >= tol_r[q / n] * (1.0 + mathshim::abs_of(fm[q])) { done = false; break; }
                q += 1;
            }
            if done {
                q = 0;
                while q < b0.len {
                    if mathshim::abs_of(b0[q]) >= bc_tol { done = false; break; }
                    q += 1;
                }
            }
            if done { break; }
            if alpha == 1.0 {
                step = copy(step2);
                cost = cost2;
                recompute = false;
            } else {
                recompute = true;
            }
        }
        iteration += 1;
        // The estimate of the residual of every interval.
        let rms = bvp_rms(xc, h, yc, fr, fm, cm, m, n, pc, F);
        // The test of the end of the run. The estimate of the residual
        // of every interval and the residual of the conditions must lie
        // under the tolerance. The run stops when they do.
        var ok = true;
        q = 0;
        while q < rms.len {
            if rms[q] > tol2 { ok = false; break; }
            q += 1;
        }
        if ok {
            q = 0;
            while q < b0.len {
                if mathshim::abs_of(b0[q]) >= bc_tol { ok = false; break; }
                q += 1;
            }
        }
        if ok { break; }
        if iteration >= 100 {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "solve_bvp", "the conditions at the ends stay above the tolerance after the count of the turns");
        }
        var added: i64 = 0;
        var ins: i64[>..] = [];
        q = 0;
        while q < m - 1 {
            if rms[q] >= 100.0 * tol2 { ins.push(2); added += 2; }
            else if rms[q] > tol2 { ins.push(1); added += 1; }
            else { ins.push(0); }
            q += 1;
        }
        if m + added > max_nodes {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_LIMIT, "solve_bvp", "the mesh reached the largest count of nodes before the residual fell below the tolerance");
        }
        {
            // The mesh gains the middle points of the marked
            // intervals, and the answer comes from the spline.
            var nx: f64[>..] = [];
            var ny: f64[>..] = [];
            q = 0;
            while q < m - 1 {
                nx.push(xc[q]);
                bvp_spline_eval(q, xc[q], xc, yc, fr, n, false, ny);
                if ins[q] == 1 {
                    let mm = xc[q] + 0.5 * h[q];
                    nx.push(mm);
                    bvp_spline_eval(q, mm, xc, yc, fr, n, false, ny);
                } else if ins[q] == 2 {
                    let m1 = xc[q] + h[q] / 3.0;
                    let m2 = xc[q] + 2.0 * h[q] / 3.0;
                    nx.push(m1);
                    bvp_spline_eval(q, m1, xc, yc, fr, n, false, ny);
                    nx.push(m2);
                    bvp_spline_eval(q, m2, xc, yc, fr, n, false, ny);
                }
                q += 1;
            }
            nx.push(xc[m - 1]);
            bvp_spline_eval(m - 2, xc[m - 1], xc, yc, fr, n, false, ny);
            xc = copy(nx);
            yc = copy(ny);
            m = xc.len;
            run = true;
        }
    }
    let rms = bvp_rms(xc, h, yc, fr, fm, cm, m, n, pc, F);
    var po: f64[>..] = [];
    q = 0;
    while q < k { po.push(pc[q]); q += 1; }
    return line_arr(copy(xc)), mat2(copy(yc), m, n), line_arr(copy(po)), line_arr(copy(rms)), no_fault();
}
