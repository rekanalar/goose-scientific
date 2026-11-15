// The helpers of the solver of the boundary value problems.
//
// The collocation, the derivatives by the forward differences and
// the estimate of the residual of the solver of the boundary value
// problems. The call that holds them lives in array_integrate_bvp.
//
namespace scientific;

import mathshim;

// bvp_f_evals gives the values of the right side at every point. The
// result holds one row of n values per point.
fn bvp_f_evals<F: fn(f64, f64[:], f64[:]) -> f64[:]>(xq: f64[:], y: f64[:], p: f64[:], n: i64) -> f64[>..], Fault {
    var out: f64[>..] = [];
    var q: i64 = 0;
    while q < xq.len {
        let fv = F(xq[q], y[q * n..q * n + n], p);
        if fv.len != n {
            return out, fault(F_SHAPE, "solve_bvp", "the right side does not give one row of values per point");
        }
        var j: i64 = 0;
        while j < n {
            if !(fv[j] - fv[j] == 0.0) {
                return out, fault(F_VALUE, "solve_bvp", "the right side holds a value that is not finite");
            }
            out.push(fv[j]);
            j += 1;
        }
        q += 1;
    }
    return out, no_fault();
}

// bvp_middle gives the values of the spline at the middle of every
// interval. The result holds one row per interval.
fn bvp_middle(y: f64[:], f: f64[:], h: f64[:], m: i64, n: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var q: i64 = 0;
    while q < m - 1 {
        var j: i64 = 0;
        while j < n {
            out.push(0.5 * (y[q * n + j] + y[(q + 1) * n + j]) - 0.125 * h[q] * (f[(q + 1) * n + j] - f[q * n + j]));
            j += 1;
        }
        q += 1;
    }
    return out;
}

// bvp_col_res gives the residuals of the collocation at the middle of
// every interval. The result holds one row per interval.
fn bvp_col_res(y: f64[:], f: f64[:], fm: f64[:], h: f64[:], m: i64, n: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var q: i64 = 0;
    while q < m - 1 {
        var j: i64 = 0;
        while j < n {
            out.push(y[(q + 1) * n + j] - y[q * n + j] - h[q] / 6.0 * (f[q * n + j] + f[(q + 1) * n + j] + 4.0 * fm[q * n + j]));
            j += 1;
        }
        q += 1;
    }
    return out;
}

// bvp_fun_jac gives the derivatives of the right side by the forward
// differences. The mode zero perturbs the values and the mode one
// perturbs the parameters. Both modes hold, for every perturbed
// value, one row of n derivatives per point: the derivative of the
// row j of the function over the perturbed value i at the point q
// stands at the place (i * points + q) * n + j.
fn bvp_fun_jac<F: fn(f64, f64[:], f64[:]) -> f64[:]>(mode: i64, xq: f64[:], y: f64[:], p: f64[:], n: i64, f0: f64[:]) -> f64[>..], Fault {
    let eps = mathshim::root(BVP_EPS);
    let points = xq.len;
    var out: f64[>..] = [];
    let w: i64 = if mode == 0 { n } else { p.len };
    var i: i64 = 0;
    while i < w {
        var q: i64 = 0;
        while q < points {
            var ynew: f64[>..] = [];
            var pnew: f64[>..] = [];
            var j: i64 = 0;
            while j < n { ynew.push(y[q * n + j]); j += 1; }
            j = 0;
            while j < p.len { pnew.push(p[j]); j += 1; }
            var hi: f64 = 0.0;
            if mode == 0 {
                let h0 = eps * (1.0 + mathshim::abs_of(ynew[i]));
                ynew[i] += h0;
                hi = ynew[i] - y[q * n + i];
            } else {
                let h0 = eps * (1.0 + mathshim::abs_of(pnew[i]));
                pnew[i] += h0;
                hi = pnew[i] - p[i];
            }
            let fv = F(xq[q], copy(ynew), copy(pnew));
            if fv.len != n {
                return out, fault(F_SHAPE, "solve_bvp", "the right side does not give one row of values per point");
            }
            var j2: i64 = 0;
            while j2 < n {
                out.push((fv[j2] - f0[q * n + j2]) / hi);
                j2 += 1;
            }
            q += 1;
        }
        i += 1;
    }
    return out, no_fault();
}

// bvp_bc_jac gives the derivatives of the boundary conditions by the
// forward differences. The mode zero perturbs the values at the left
// end, the mode one the values at the right end and the mode two the
// parameters. The result holds one row per condition and one column
// per perturbed value.
fn bvp_bc_jac<B: fn(f64[:], f64[:], f64[:]) -> f64[:]>(mode: i64, ya: f64[:], yb: f64[:], p: f64[:], bc0: f64[:]) -> f64[>..], Fault {
    let n = ya.len;
    let k = p.len;
    let eps = mathshim::root(BVP_EPS);
    var out: f64[>..] = [];
    let w: i64 = if mode == 2 { k } else { n };
    var i: i64 = 0;
    while i < w {
        var pa: f64[>..] = [];
        var pb: f64[>..] = [];
        var pp: f64[>..] = [];
        var j: i64 = 0;
        while j < n { pa.push(ya[j]); pb.push(yb[j]); j += 1; }
        j = 0;
        while j < k { pp.push(p[j]); j += 1; }
        var hi: f64 = 0.0;
        if mode == 0 {
            let h0 = eps * (1.0 + mathshim::abs_of(pa[i]));
            pa[i] += h0;
            hi = pa[i] - ya[i];
        } else if mode == 1 {
            let h0 = eps * (1.0 + mathshim::abs_of(pb[i]));
            pb[i] += h0;
            hi = pb[i] - yb[i];
        } else {
            let h0 = eps * (1.0 + mathshim::abs_of(pp[i]));
            pp[i] += h0;
            hi = pp[i] - p[i];
        }
        let bv = B(pa, pb, pp);
        if bv.len != n + k {
            return out, fault(F_SHAPE, "solve_bvp", "the boundary conditions do not hold one value per condition");
        }
        var r: i64 = 0;
        while r < n + k {
            out.push((bv[r] - bc0[r]) / hi);
            r += 1;
        }
        i += 1;
    }
    // The result holds one row per condition and one column per
    // perturbed value: the place of the condition r over the value i
    // is r * w + i.
    var mat: f64[>..] = [];
    var c: i64 = 0;
    while c < n + k {
        var i2: i64 = 0;
        while i2 < w {
            mat.push(out[i2 * (n + k) + c]);
            i2 += 1;
        }
        c += 1;
    }
    return mat, no_fault();
}

// bvp_spline_eval gives the values of the spline of the answer, or
// the values of its derivative, at the place xx of the interval q.
// The result goes into out.
fn bvp_spline_eval(q: i64, xx: f64, x: f64[:], y: f64[:], f: f64[:], n: i64, der: bool, out: f64[>..]&) {
    let h = x[q + 1] - x[q];
    let d = xx - x[q];
    var j: i64 = 0;
    while j < n {
        let y0 = y[q * n + j];
        let y1 = y[(q + 1) * n + j];
        let f0 = f[q * n + j];
        let f1 = f[(q + 1) * n + j];
        let slope = (y1 - y0) / h;
        let t = (f0 + f1 - 2.0 * slope) / h;
        let c0 = t / h;
        let c1 = (slope - f0) / h - t;
        if der {
            out.push(3.0 * c0 * d * d + 2.0 * c1 * d + f0);
        } else {
            out.push(((c0 * d + c1) * d + f0) * d + y0);
        }
        j += 1;
    }
}

// bvp_rms gives the estimate of the residual of every interval. The
// estimate uses the rule of Lobatto of five points on the relative
// residual, as SciPy does.
fn bvp_rms<G: fn(f64, f64[:], f64[:]) -> f64[:]>(x: f64[:], h: f64[:], y: f64[:], f: f64[:], fm: f64[:], cm: f64[:], m: i64, n: i64, p: f64[:]) -> f64[>..] {
    var out: f64[>..] = [];
    var q: i64 = 0;
    while q < m - 1 {
        let s = 0.5 * h[q] * mathshim::root(3.0 / 7.0);
        let xm = x[q] + 0.5 * h[q];
        let x1 = xm + s;
        let x2 = xm - s;
        var v1: f64[>..] = [];
        var v2: f64[>..] = [];
        var d1: f64[>..] = [];
        var d2: f64[>..] = [];
        bvp_spline_eval(q, x1, x, y, f, n, false, v1);
        bvp_spline_eval(q, x2, x, y, f, n, false, v2);
        bvp_spline_eval(q, x1, x, y, f, n, true, d1);
        bvp_spline_eval(q, x2, x, y, f, n, true, d2);
        // The values of the right side at the two points of the rule.
        let f1 = G(x1, v1[..], p);
        let f2 = G(x2, v2[..], p);
        var r1: f64 = 0.0;
        var r2: f64 = 0.0;
        var rm: f64 = 0.0;
        var j: i64 = 0;
        while j < n {
            let w1 = 1.0 + mathshim::abs_of(fm[q * n + j]);
            let g = 1.5 * cm[q * n + j] / h[q];
            rm += (g / w1) * (g / w1);
            let t1 = (d1[j] - f1[j]) / (1.0 + mathshim::abs_of(f1[j]));
            let t2 = (d2[j] - f2[j]) / (1.0 + mathshim::abs_of(f2[j]));
            r1 += t1 * t1;
            r2 += t2 * t2;
            j += 1;
        }
        out.push(mathshim::root(0.5 * (32.0 / 45.0 * rm + 49.0 / 90.0 * (r1 + r2))));
        q += 1;
    }
    return out;
}
