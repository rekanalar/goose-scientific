// The main loop of the ODE solver ivp_run.
//
namespace scientific;

import mathshim;

// ivp_run integrates the system y' = f(t, y) from t0 to t1. The block
// gives n values of the derivative, then nev values of the event
// functions. It gives the times and the states, the times, the numbers
// and the states of the events, and the dense output of the steps.
fn ivp_run<F: fn(f64, f64[:]) -> f64[:]>(t0: f64, t1: f64, y0: f64[:], rtol: f64, atol: f64, method: const u8[:], max_step: f64, t_eval: f64[:], nev: i64, terminal: bool[:], direction: f64[:], want_dense: bool, first_step: f64, use_jac: bool) -> Arr, Arr, Arr, Arr, Arr, Dense, Fault {
    if method != "RK45" && method != "RK23" && method != "DOP853" && method != "BDF" && method != "Radau" {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_VALUE, "solve_ivp", "the method holds an unknown name");
    }
    if !(t0 - t0 == 0.0) || !(t1 - t1 == 0.0) || t0 == t1 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_VALUE, "solve_ivp", "the times are not finite and different");
    }
    if y0.len < 1 || y0.len > 4096 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_SHAPE, "solve_ivp", "the start holds no value or too many");
    }
    if !(rtol > 0.0) || !(atol >= 0.0) || !(max_step > 0.0) {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_VALUE, "solve_ivp", "a tolerance or the largest step holds a value out of range");
    }
    if !(first_step >= 0.0) || first_step > mathshim::abs_of(t1 - t0) {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_VALUE, "solve_ivp", "the first step is negative or exceeds the span");
    }
    var tol = rtol;
    if tol < 2.220446049250313e-14 { tol = 2.220446049250313e-14; }
    let n = y0.len;
    var dir: f64 = 1.0;
    if t1 < t0 { dir = -1.0; }
    // The times of t_eval must lie in the span and must be sorted.
    var qi: i64 = 0;
    while qi < t_eval.len {
        let te = t_eval[qi];
        if !(te - te == 0.0) || dir * (te - t0) < 0.0 || dir * (t1 - te) < 0.0 {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_VALUE, "solve_ivp", "a time of t_eval is not finite or lies outside the span");
        }
        if qi > 0 && dir * (te - t_eval[qi - 1]) < 0.0 {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_VALUE, "solve_ivp", "the times of t_eval do not follow the direction");
        }
        qi += 1;
    }
    var hist_t: f64[>..] = [];
    var hist_h: f64[>..] = [];
    var hist_y: f64[>..] = [];
    var hist_k: f64[>..] = [];
    // The kind of the method. The number zero is RK45, the number one
    // is RK23 and the number two is DOP853.
    var kind: i32 = 1;
    if method == "RK45" { kind = 0; }
    if method == "DOP853" { kind = 2; }
    if method == "BDF" { kind = 3; }
    if method == "Radau" { kind = 4; }
    var order: f64 = 3.0;
    if kind == 0 { order = 5.0; }
    if kind == 2 { order = 8.0; }
    if kind == 3 { order = 2.0; }
    if kind == 4 { order = 4.0; }
    // The exponent of the control is minus one over the order of the
    // error estimate plus one.
    var cc: f64[>..] = [];
    var aa: f64[>..] = [];
    var bb: f64[>..] = [];
    var ee: f64[>..] = [];
    var e3: f64[>..] = [];
    var ns: i64 = 0;
    if kind < 3 { ns = ivp_tableau(kind, cc, aa, bb, ee, e3); }
    // The count of the rows of the stages that the dense output holds.
    // The method of the order eight holds three more rows for the
    // interpolation.
    var rows: i64 = ns + 1;
    if kind == 2 { rows = 16; }
    if kind == 3 { rows = 9; }
    if kind == 4 { rows = 3; }
    var y: f64[>..] = [];
    var i: i64 = 0;
    while i < n { y.push(y0[i]); i += 1; }
    var t = t0;
    let f00 = F(t, copy(y));
    if f00.len != n + nev {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
    }
    var f: f64[>..] = [];
    i = 0;
    while i < n { f.push(f00[i]); i += 1; }
    // The first step.
    var gold: f64[>..] = [];
    i = 0;
    while i < nev { gold.push(f00[n + i]); i += 1; }
    var ev_t: f64[>..] = [];
    var ev_i: f64[>..] = [];
    var ev_y: f64[>..] = [];
    var stop = false;
    var scale: f64[>..] = [];
    var q1: f64[>..] = [];
    var q2: f64[>..] = [];
    i = 0;
    while i < n { scale.push(atol + mathshim::abs_of(y[i]) * tol); q1.push(y[i] / scale[i]); q2.push(f[i] / scale[i]); i += 1; }
    let d0 = ivp_rms(q1);
    let d1 = ivp_rms(q2);
    var h0: f64 = 0.000001;
    if !(d0 < 0.00001 || d1 < 0.00001) { h0 = 0.01 * d0 / d1; }
    // The step of the trial does not leave the span, as SciPy does.
    let span = mathshim::abs_of(t1 - t0);
    if h0 > span { h0 = span; }
    var y1: f64[>..] = [];
    i = 0;
    while i < n { y1.push(y[i] + h0 * dir * f[i]); i += 1; }
    let f1 = F(t + h0 * dir, copy(y1));
    if f1.len != n + nev {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
    }
    var q3: f64[>..] = [];
    i = 0;
    while i < n { q3.push((f1[i] - f[i]) / scale[i]); i += 1; }
    let d2 = ivp_rms(q3) / h0;
    var h1: f64 = 0.0;
    if d1 <= 1.0e-15 && d2 <= 1.0e-15 {
        h1 = h0 * 0.001;
        if h1 < 0.000001 { h1 = 0.000001; }
    } else {
        var big = d1;
        if d2 > big { big = d2; }
        h1 = mathshim::power(0.01 / big, 1.0 / order);
    }
    var h_abs = 100.0 * h0;
    if h1 < h_abs { h_abs = h1; }
    if span < h_abs { h_abs = span; }
    if max_step < h_abs { h_abs = max_step; }
    // A first step from the caller replaces the chosen one.
    if first_step > 0.0 { h_abs = first_step; }
    // The state of the method BDF.
    var bdm: f64[>..] = [];
    var bfst: f64[>..] = [h_abs];
    var bist: i64[>..] = [1, 0, 0];
    var bjm: f64[>..] = [];
    var bjfac: f64[>..] = [];
    if use_jac { bjfac.push(-1.0); }
    var blum: f64[>..] = [];
    var bpiv: i64[>..] = [];
    var btn: f64[>..] = [0.0];
    // The state of the method Radau.
    var rfst: f64[>..] = [h_abs, -1.0, -1.0, 0.0, 1.0];
    var rist: i64[>..] = [0, 0, 1];
    var rrq: f64[>..] = [];
    var rryo: f64[>..] = [];
    var rlcr: f64[>..] = [];
    var rlci: f64[>..] = [];
    var rpivc: i64[>..] = [];
    if kind == 4 {
        i = 0;
        while i < 3 * n { rrq.push(0.0); i += 1; }
        i = 0;
        while i < n { rryo.push(0.0); bpiv.push(0); rpivc.push(0); i += 1; }
        i = 0;
        while i < n * n { bjm.push(0.0); blum.push(0.0); rlcr.push(0.0); rlci.push(0.0); i += 1; }
        let jf0 = bdf_jac(t, y, n, nev, atol, bjfac, bjm) { tt, yy => F(tt, copy(yy)) };
        if jf0.kind != F_NONE {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), jf0;
        }
    }
    if kind == 3 {
        i = 0;
        while i < 8 * n { bdm.push(0.0); i += 1; }
        i = 0;
        while i < n { bdm[i] = y[i]; bdm[n + i] = f[i] * h_abs * dir; i += 1; }
        i = 0;
        while i < n * n { bjm.push(0.0); blum.push(0.0); i += 1; }
        i = 0;
        while i < n { bpiv.push(0); i += 1; }
        let jf0 = bdf_jac(t, y, n, nev, atol, bjfac, bjm) { tt, yy => F(tt, copy(yy)) };
        if jf0.kind != F_NONE {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), jf0;
        }
    }
    var ts: f64[>..] = [t];
    var ys: f64[>..] = [];
    i = 0;
    while i < n { ys.push(y[i]); i += 1; }
    var rejected = false;
    var ntol = 10.0 * 2.220446049250313e-16 / rtol;
    var ntol2 = mathshim::root(rtol);
    if ntol2 > 0.03 { ntol2 = 0.03; }
    if ntol2 > ntol { ntol = ntol2; }
    var steps: i64 = 0;
    while dir * (t1 - t) > 0.0 && !stop {
        steps += 1;
        if steps > 1000000 {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_LIMIT, "solve_ivp", "the count of steps reached the limit");
        }
        var min_step = 10.0 * mathshim::abs_of(t) * 2.220446049250313e-16;
        if min_step < 5.0e-323 { min_step = 5.0e-323; }
        if h_abs < min_step { h_abs = min_step; }
        if h_abs > max_step { h_abs = max_step; }
        var accepted = false;
        var kacc: f64[>..] = [];
        var fnew: f64[>..] = [];
        var gnew: f64[>..] = [];
        var ynew: f64[>..] = [];
        var tnew: f64 = t;
        if kind == 3 {
            let bf = bdf_step(t, t1, dir, n, nev, atol, rtol, ntol, max_step, bdm, bfst, bist, bjm, bjfac, blum, bpiv, ynew, btn) { tt, yy => F(tt, copy(yy)) };
            if bf.kind != F_NONE {
                return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), bf;
            }
            tnew = btn[0];
            let fb = F(tnew, copy(ynew));
            if fb.len != n + nev {
                return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
            }
            i = 0;
            while i < n { fnew.push(fb[i]); i += 1; }
            i = 0;
            while i < nev { gnew.push(fb[n + i]); i += 1; }
            accepted = true;
        }
        if kind == 4 {
            var fcur: f64[>..] = [];
            i = 0;
            while i < n { fcur.push(f[i]); i += 1; }
            var fbn: f64[>..] = [];
            let rf = rad_step(t, t1, dir, n, nev, atol, rtol, ntol, max_step, y, fcur, rfst, rist, rrq, rryo, bjm, bjfac, blum, bpiv, rlcr, rlci, rpivc, ynew, fbn, btn) { tt, yy => F(tt, copy(yy)) };
            if rf.kind != F_NONE {
                return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), rf;
            }
            tnew = btn[0];
            i = 0;
            while i < n { fnew.push(fbn[i]); i += 1; }
            i = 0;
            while i < nev { gnew.push(fbn[n + i]); i += 1; }
            accepted = true;
        }
        while !accepted {
            if h_abs < min_step {
                return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_VALUE, "solve_ivp", "the step fell below the smallest step");
            }
            var h = h_abs * dir;
            tnew = t + h;
            if dir * (tnew - t1) > 0.0 { tnew = t1; }
            h = tnew - t;
            h_abs = mathshim::abs_of(h);
            // The stages. K holds one row of n values per stage.
            var k: f64[>..] = [];
            i = 0;
            while i < n { k.push(f[i]); i += 1; }
            var s: i64 = 1;
            while s < ns {
                var ytmp: f64[>..] = [];
                i = 0;
                while i < n {
                    var acc: f64 = 0.0;
                    var j: i64 = 0;
                    while j < s { acc += k[j * n + i] * aa[s * ns + j]; j += 1; }
                    ytmp.push(y[i] + h * acc);
                    i += 1;
                }
                let ks = F(t + cc[s] * h, copy(ytmp));
                if ks.len != n + nev {
                    return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
                }
                i = 0;
                while i < n { k.push(ks[i]); i += 1; }
                s += 1;
            }
            ynew = [];
            i = 0;
            while i < n {
                var acc: f64 = 0.0;
                var j: i64 = 0;
                while j < ns { acc += k[j * n + i] * bb[j]; j += 1; }
                ynew.push(y[i] + h * acc);
                i += 1;
            }
            let fn1 = F(tnew, copy(ynew));
            if fn1.len != n + nev {
                return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
            }
            fnew = [];
            gnew = [];
            i = 0;
            while i < n { fnew.push(fn1[i]); i += 1; }
            i = 0;
            while i < nev { gnew.push(fn1[n + i]); i += 1; }
            var err: f64 = 0.0;
            if kind == 2 {
                // The method of the order eight holds two error vectors.
                // The estimate of the error takes the fifth order and
                // the third order together, as SciPy does.
                var s5: f64 = 0.0;
                var s3: f64 = 0.0;
                i = 0;
                while i < n {
                    var a5: f64 = 0.0;
                    var a3: f64 = 0.0;
                    var j: i64 = 0;
                    while j < ns { a5 += k[j * n + i] * ee[j]; a3 += k[j * n + i] * e3[j]; j += 1; }
                    a5 += fnew[i] * ee[ns];
                    a3 += fnew[i] * e3[ns];
                    var big = mathshim::abs_of(y[i]);
                    if mathshim::abs_of(ynew[i]) > big { big = mathshim::abs_of(ynew[i]); }
                    let sc = atol + big * tol;
                    let d5 = a5 / sc;
                    let d3 = a3 / sc;
                    s5 += d5 * d5;
                    s3 += d3 * d3;
                    i += 1;
                }
                if !(s5 == 0.0 && s3 == 0.0) {
                    err = mathshim::abs_of(h) * s5 / mathshim::root((s5 + 0.01 * s3) * n);
                }
            } else {
                var en: f64[>..] = [];
                i = 0;
                while i < n {
                    var acc: f64 = 0.0;
                    var j: i64 = 0;
                    while j < ns { acc += k[j * n + i] * ee[j]; j += 1; }
                    acc += fnew[i] * ee[ns];
                    en.push(h * acc);
                    i += 1;
                }
                var sq: f64[>..] = [];
                i = 0;
                while i < n {
                    var big = mathshim::abs_of(y[i]);
                    if mathshim::abs_of(ynew[i]) > big { big = mathshim::abs_of(ynew[i]); }
                    sq.push(en[i] / (atol + big * tol));
                    i += 1;
                }
                err = ivp_rms(sq);
            }
            if err < 1.0 {
                kacc = copy_slice(k);
                var factor: f64 = 10.0;
                if err != 0.0 {
                    factor = 0.9 * mathshim::power(err, -1.0 / order);
                    if factor > 10.0 { factor = 10.0; }
                }
                if rejected && factor > 1.0 { factor = 1.0; }
                h_abs = h_abs * factor;
                accepted = true;
                rejected = false;
            } else {
                var factor = 0.9 * mathshim::power(err, -1.0 / order);
                if factor < 0.2 { factor = 0.2; }
                h_abs = h_abs * factor;
                rejected = true;
            }
        }
        // The rows of the stages that the interpolation of the step
        // uses. The method of the order eight holds three more rows.
        var kall: f64[>..] = [];
        var kex: f64[>..] = [];
        if kind == 2 && (t_eval.len > 0 || want_dense || nev > 0) {
            let hh = tnew - t;
            var exa: f64[>..] = [];
            var exc: f64[>..] = [];
            dop853_extra(exa, exc);
            var s: i64 = 13;
            while s < 16 {
                var base: i64 = 0;
                if s == 14 { base = 13; }
                if s == 15 { base = 27; }
                var ytmp: f64[>..] = [];
                i = 0;
                while i < n {
                    var acc: f64 = 0.0;
                    var j: i64 = 0;
                    while j < s {
                        var kv: f64 = 0.0;
                        if j < ns { kv = kacc[j * n + i]; }
                        else if j == ns { kv = fnew[i]; }
                        else { kv = kex[(j - ns - 1) * n + i]; }
                        acc += kv * exa[base + j];
                        j += 1;
                    }
                    ytmp.push(y[i] + hh * acc);
                    i += 1;
                }
                let ks = F(t + exc[s - 13] * hh, copy(ytmp));
                if ks.len != n + nev {
                    return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), dense_none(), fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
                }
                i = 0;
                while i < n { kex.push(ks[i]); i += 1; }
                s += 1;
            }
        }
        if kind == 4 {
            i = 0;
            while i < 3 * n { kall.push(rrq[i]); i += 1; }
        } else if kind == 3 {
            // The rows of the differences, the end of the step, the step
            // and the order, as the polynomial of the step needs them.
            i = 0;
            while i < 6 * n { kall.push(bdm[i]); i += 1; }
            var hd: i64 = 0;
            while hd < 3 {
                var val: f64 = btn[0];
                if hd == 1 { val = bfst[0] * dir; }
                if hd == 2 { val = bist[0]; }
                i = 0;
                while i < n { if i == 0 { kall.push(val); } else { kall.push(0.0); } i += 1; }
                hd += 1;
            }
        } else {
            i = 0;
            while i < ns * n { kall.push(kacc[i]); i += 1; }
            i = 0;
            while i < n { kall.push(fnew[i]); i += 1; }
            i = 0;
            while i < kex.len { kall.push(kex[i]); i += 1; }
        }
        if t_eval.len > 0 || want_dense {
            hist_t.push(t);
            hist_h.push(tnew - t);
            i = 0;
            while i < n { hist_y.push(y[i]); i += 1; }
            i = 0;
            while i < kall.len { hist_k.push(kall[i]); i += 1; }
        }
        if nev > 0 {
            // The events of the step. A change of sign of an event
            // function in the direction that the caller asks for.
            var hit_t: f64[>..] = [];
            var hit_i: f64[>..] = [];
            var e: i64 = 0;
            while e < nev {
                let g0 = gold[e];
                let g1 = gnew[e];
                let up = g0 <= 0.0 && g1 >= 0.0;
                let down = g0 >= 0.0 && g1 <= 0.0;
                var want = false;
                if direction[e] > 0.0 { want = up; }
                else if direction[e] < 0.0 { want = down; }
                else { want = up || down; }
                if want {
                    // The root by bisection of the polynomial of the step.
                    var lo: f64 = 0.0;
                    var hi: f64 = 1.0;
                    var glo = g0;
                    var it: i64 = 0;
                    while it < 200 {
                        let mid = 0.5 * (lo + hi);
                        if mid == lo || mid == hi { break; }
                        let ym = ivp_point_kind(kind, ns, n, y, ynew, tnew - t, kall, mid);
                        let gm = F(t + mid * (tnew - t), copy(ym));
                        let gv = gm[n + e];
                        if (glo <= 0.0 && gv >= 0.0) || (glo >= 0.0 && gv <= 0.0) { hi = mid; } else { lo = mid; glo = gv; }
                        it += 1;
                    }
                    hit_t.push(t + hi * (tnew - t));
                    hit_i.push(e);
                }
                e += 1;
            }
            // The hits in the order of the direction of time.
            var order: f64[>..] = [];
            var taken: f64[>..] = [];
            i = 0;
            while i < hit_t.len { taken.push(0.0); i += 1; }
            var c: i64 = 0;
            while c < hit_t.len {
                var pick: i64 = -1;
                i = 0;
                while i < hit_t.len {
                    if taken[i] == 0.0 && (pick < 0 || dir * hit_t[i] < dir * hit_t[pick]) { pick = i; }
                    i += 1;
                }
                taken[pick] = 1.0;
                order.push(pick);
                c += 1;
            }
            var oi: i64 = 0;
            while oi < order.len && !stop {
                let h = order[oi] as i64;
                let th = hit_t[h];
                let eh = hit_i[h] as i64;
                let fr = (th - t) / (tnew - t);
                let yh = ivp_point_kind(kind, ns, n, y, ynew, tnew - t, kall, fr);
                ev_t.push(th);
                ev_i.push(eh);
                i = 0;
                while i < n { ev_y.push(yh[i]); i += 1; }
                if terminal[eh] {
                    stop = true;
                    tnew = th;
                    ynew = copy_slice(yh);
                    fnew = [];
                    var fz = F(th, copy(yh));
                    i = 0;
                    while i < n { fnew.push(fz[i]); i += 1; }
                }
                oi += 1;
            }
            gold = copy_slice(gnew);
        }
        t = tnew;
        y = copy(ynew);
        f = copy(fnew);
        ts.push(t);
        i = 0;
        while i < n { ys.push(y[i]); i += 1; }
    }
    var dn = dense_none();
    if want_dense {
        var buf: f64[>..] = [];
        var q: i64 = 0;
        while q < hist_t.len { buf.push(hist_t[q]); q += 1; }
        q = 0;
        while q < hist_h.len { buf.push(hist_h[q]); q += 1; }
        q = 0;
        while q < hist_y.len { buf.push(hist_y[q]); q += 1; }
        q = 0;
        while q < hist_k.len { buf.push(hist_k[q]); q += 1; }
        dn = Dense { kind: kind, ns: ns, n: n, steps: hist_t.len, buf: copy(buf) };
    }
    var evt = empty_arr();
    var evi = empty_arr();
    var evy = empty_arr();
    if ev_t.len > 0 {
        evt = line_arr(copy(ev_t));
        evi = line_arr(copy(ev_i));
        evy = mat2(copy(ev_y), ev_t.len, n);
    }
    if t_eval.len > 0 {
        // The times after a terminal event drop out.
        var kept: f64[>..] = [];
        var qq: i64 = 0;
        while qq < t_eval.len {
            if dir * (t_eval[qq] - t) <= 0.0 { kept.push(t_eval[qq]); }
            qq += 1;
        }
        let rt, ry, rf = ivp_dense(kept, hist_t, hist_h, hist_y, hist_k, kind, ns, n);
        return rt, ry, evt, evi, evy, dn, rf;
    }
    return line_arr(copy(ts)), mat2(copy(ys), ts.len, n), evt, evi, evy, dn, no_fault();
}

