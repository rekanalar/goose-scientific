// The equiripple filter design of Parks and McClellan, like scipy.signal.remez.
//
namespace scientific;

import mathshim;

// The limits of remez. The grid holds at most MAX_REMEZ_GRID points, so
// the cost of one pass stays bounded.
const MAX_REMEZ_GRID: i64 = 262144;
const MAX_REMEZ_TAPS: i64 = 4096;
const MAX_REMEZ_PASSES: i64 = 1000;
const MAX_REMEZ_DENSITY: i64 = 1024;
const REMEZ_PI: f64 = 3.141592653589793;

// rz_lagrange gives the coefficient of the barycentric form for the
// node k of n nodes. The routine splits the product in m strides to
// keep the product in range. The arrays of remez hold the index one
// first, so slot zero stays unused.
fn rz_lagrange(k: i64, n: i64, m: i64, x: f64[:]) -> f64 {
    var r: f64 = 1.0;
    let q = x[k];
    var l: i64 = 1;
    while l <= m {
        var j = l;
        while j <= n {
            if j != k { r = r * 2.0 * (q - x[j]); }
            j += m;
        }
        l += 1;
    }
    return 1.0 / r;
}

// rz_eval gives the response at the grid point k by the barycentric
// form over the n extremal nodes.
fn rz_eval(k: i64, n: i64, grid: f64[:], x: f64[:], y: f64[:], ad: f64[:]) -> f64 {
    var d: f64 = 0.0;
    var p: f64 = 0.0;
    let xf = mathshim::cosine(2.0 * REMEZ_PI * grid[k]);
    var j: i64 = 1;
    while j <= n {
        let c = ad[j] / (xf - x[j]);
        d += c;
        p += c * y[j];
        j += 1;
    }
    return p / d;
}

// rz_err gives the weighted error at the grid point l.
fn rz_err(l: i64, nz: i64, grid: f64[:], x: f64[:], y: f64[:], ad: f64[:], des: f64[:], wt: f64[:]) -> f64 {
    return (rz_eval(l, nz, grid, x, y, ad) - des[l]) * wt[l];
}

// rz_exchange runs the exchange of the extremal set. The set iext holds
// nfcns + 1 grid indexes and one slot more. The arrays x, y and ad keep
// the nodes, the values and the weights of the last pass. The result is
// 0 when the exchange ends, -1 when the deviation stops rising, and -2
// when the count of steps passes the bound. The block of each label of
// the original routine is one branch of the loop. The variable pc names
// the next label, and 999 leaves the loop.
fn rz_exchange(des: f64[:], grid: f64[:], wt: f64[:], ngrid: i64, iext: i64[>..]&, nfcns: i64, itrmax: i64, x: f64[>..]&, y: f64[>..]&, ad: f64[>..]&) -> i64 {
    let nz = nfcns + 1;
    let nzz = nfcns + 2;
    var devl: f64 = -1.0;
    var dev: f64 = 0.0;
    var niter: i64 = 0;
    var j: i64 = 0;
    var k1: i64 = 0;
    var knz: i64 = 0;
    var klow: i64 = 0;
    var kup: i64 = 0;
    var l: i64 = 0;
    var luck: i64 = 0;
    var jchnge: i64 = 0;
    var nu: f64 = 0.0;
    var nut: f64 = 0.0;
    var nut1: f64 = 0.0;
    var ynz: f64 = 0.0;
    var comp: f64 = 0.0;
    var y1: f64 = 0.0;
    var err: f64 = 0.0;
    var pc: i64 = 100;
    var steps: i64 = 0;
    let bound = (itrmax + 1) * 16 * (ngrid + nz + 16);
    while pc != 999 {
        steps += 1;
        if steps > bound { return -2; }
        if pc == 100 {
            iext[nzz] = ngrid + 1;
            niter += 1;
            if niter > itrmax { pc = 999; }
            else {
                j = 1;
                while j <= nz { x[j] = mathshim::cosine(2.0 * REMEZ_PI * grid[iext[j]]); j += 1; }
                let jet = (nfcns - 1) / 15 + 1;
                j = 1;
                while j <= nz { ad[j] = rz_lagrange(j, nz, jet, x); j += 1; }
                var dnum: f64 = 0.0;
                var dden: f64 = 0.0;
                var kf: f64 = 1.0;
                j = 1;
                while j <= nz {
                    l = iext[j];
                    dnum += ad[j] * des[l];
                    dden += kf * ad[j] / wt[l];
                    kf = -kf;
                    j += 1;
                }
                dev = dnum / dden;
                nu = 1.0;
                if dev > 0.0 { nu = -1.0; }
                dev = -nu * dev;
                kf = nu;
                j = 1;
                while j <= nz {
                    l = iext[j];
                    y[j] = des[l] + kf * dev / wt[l];
                    kf = -kf;
                    j += 1;
                }
                if dev <= devl { return -1; }
                devl = dev;
                jchnge = 0;
                k1 = iext[1];
                knz = iext[nz];
                klow = 0;
                nut = -nu;
                j = 1;
                pc = 200;
            }
        } else if pc == 200 {
            if j == nzz { ynz = comp; }
            if j >= nzz { pc = 300; }
            else {
                kup = iext[j + 1];
                l = iext[j] + 1;
                nut = -nut;
                if j == 2 { y1 = comp; }
                comp = dev;
                if l >= kup { pc = 220; }
                else {
                    err = rz_err(l, nz, grid, x, y, ad, des, wt);
                    if nut * err - comp <= 0.0 { pc = 220; }
                    else { comp = nut * err; pc = 210; }
                }
            }
        } else if pc == 210 {
            l += 1;
            if l >= kup { pc = 215; }
            else {
                err = rz_err(l, nz, grid, x, y, ad, des, wt);
                if nut * err - comp <= 0.0 { pc = 215; }
                else { comp = nut * err; }
            }
        } else if pc == 215 {
            iext[j] = l - 1;
            j += 1;
            klow = l - 1;
            jchnge += 1;
            pc = 200;
        } else if pc == 220 {
            l -= 1;
            pc = 225;
        } else if pc == 225 {
            l -= 1;
            if l <= klow { pc = 250; }
            else {
                err = rz_err(l, nz, grid, x, y, ad, des, wt);
                if nut * err - comp > 0.0 { pc = 230; }
                else if jchnge > 0 { pc = 260; }
            }
        } else if pc == 230 {
            comp = nut * err;
            pc = 235;
        } else if pc == 235 {
            l -= 1;
            if l <= klow { pc = 240; }
            else {
                err = rz_err(l, nz, grid, x, y, ad, des, wt);
                if nut * err - comp <= 0.0 { pc = 240; }
                else { comp = nut * err; }
            }
        } else if pc == 240 {
            klow = iext[j];
            iext[j] = l + 1;
            j += 1;
            jchnge += 1;
            pc = 200;
        } else if pc == 250 {
            l = iext[j] + 1;
            if jchnge > 0 { pc = 215; } else { pc = 255; }
        } else if pc == 255 {
            l += 1;
            if l >= kup { pc = 260; }
            else {
                err = rz_err(l, nz, grid, x, y, ad, des, wt);
                if nut * err - comp > 0.0 { comp = nut * err; pc = 210; }
            }
        } else if pc == 260 {
            klow = iext[j];
            j += 1;
            pc = 200;
        } else if pc == 300 {
            if j > nzz { pc = 320; }
            else {
                if k1 > iext[1] { k1 = iext[1]; }
                if knz < iext[nz] { knz = iext[nz]; }
                nut1 = nut;
                nut = -nu;
                l = 0;
                kup = k1;
                comp = ynz * 1.00001;
                luck = 1;
                pc = 310;
            }
        } else if pc == 310 {
            l += 1;
            if l >= kup { pc = 315; }
            else {
                err = rz_err(l, nz, grid, x, y, ad, des, wt);
                if nut * err - comp > 0.0 { comp = nut * err; j = nzz; pc = 210; }
            }
        } else if pc == 315 {
            luck = 6;
            pc = 325;
        } else if pc == 320 {
            if luck > 9 { pc = 350; }
            else {
                if comp > y1 { y1 = comp; }
                k1 = iext[nzz];
                pc = 325;
            }
        } else if pc == 325 {
            l = ngrid + 1;
            klow = knz;
            nut = -nut1;
            comp = y1 * 1.00001;
            pc = 330;
        } else if pc == 330 {
            l -= 1;
            if l <= klow { pc = 340; }
            else {
                err = rz_err(l, nz, grid, x, y, ad, des, wt);
                if nut * err - comp > 0.0 { j = nzz; comp = nut * err; luck += 10; pc = 235; }
            }
        } else if pc == 340 {
            if luck == 6 { pc = 370; }
            else {
                j = 1;
                while j <= nfcns { iext[nzz - j] = iext[nz - j]; j += 1; }
                iext[1] = k1;
                pc = 100;
            }
        } else if pc == 350 {
            let kn = iext[nzz];
            j = 1;
            while j <= nfcns { iext[j] = iext[j + 1]; j += 1; }
            iext[nz] = kn;
            pc = 100;
        } else {
            // the label 370: the loop ends when no value changed
            if jchnge > 0 { pc = 100; } else { pc = 999; }
        }
    }
    return 0;
}

// rz_coeffs reads the cosine coefficients of the best fit by the inverse
// transform at the nfcns points. It writes them into alpha, with the
// index one first. The grid slot one changes during the call and comes
// back.
fn rz_coeffs(grid: f64[>..]&, ngrid: i64, edge: f64[:], nbands: i64, nfcns: i64, x: f64[>..]&, y: f64[:], ad: f64[:], alpha: f64[>..]&) {
    let nz = nfcns + 1;
    let nzz = nfcns + 2;
    let nm1 = nfcns - 1;
    let tp = 2.0 * REMEZ_PI;
    let gtemp = grid[1];
    x[nzz] = -2.0;
    let cn = 2 * nfcns - 1;
    let delf = 1.0 / cn;
    var l: i64 = 1;
    var kkk: i64 = 0;
    var aa: f64 = 0.0;
    var bb: f64 = 0.0;
    var a: f64[>..] = [];
    var p: f64[>..] = [];
    var q: f64[>..] = [];
    var i: i64 = 0;
    while i < nfcns + 4 { a.push(0.0); p.push(0.0); q.push(0.0); i += 1; }
    if edge[1] == 0.0 && edge[2 * nbands] == 0.5 { kkk = 1; }
    if nfcns <= 3 { kkk = 1; }
    if kkk != 1 {
        let dtemp = mathshim::cosine(tp * grid[1]);
        let dnum = mathshim::cosine(tp * grid[ngrid]);
        aa = 2.0 / (dtemp - dnum);
        bb = -(dtemp + dnum) / (dtemp - dnum);
    }
    var j: i64 = 1;
    while j <= nfcns {
        var ft = (j - 1) * delf;
        var xt = mathshim::cosine(tp * ft);
        if kkk != 1 {
            xt = (xt - bb) / aa;
            ft = mathshim::arccosine_of(xt) / tp;
        }
        var st: i64 = 410;
        while st != 425 {
            if st == 410 {
                let xe = x[l];
                if xt > xe { st = 420; }
                else if xe - xt < 1.0e-6 { st = 415; }
                else { l += 1; }
            } else if st == 415 {
                a[j] = y[l];
                st = 425;
            } else {
                let xe = x[l];
                if xt - xe < 1.0e-6 { st = 415; }
                else {
                    grid[1] = ft;
                    a[j] = rz_eval(1, nz, grid, x, y, ad);
                    st = 425;
                }
            }
        }
        if l > 1 { l -= 1; }
        j += 1;
    }
    grid[1] = gtemp;
    let dden = tp / cn;
    j = 1;
    while j <= nfcns {
        var dtemp: f64 = 0.0;
        let dnum = (j - 1) * dden;
        var k: i64 = 1;
        while k <= nm1 { dtemp += a[k + 1] * mathshim::cosine(dnum * k); k += 1; }
        alpha[j] = 2.0 * dtemp + a[1];
        j += 1;
    }
    j = 2;
    while j <= nfcns { alpha[j] = alpha[j] * 2.0 / cn; j += 1; }
    alpha[1] = alpha[1] / cn;
    if kkk != 1 {
        p[1] = 2.0 * alpha[nfcns] * bb + alpha[nm1];
        p[2] = 2.0 * aa * alpha[nfcns];
        q[1] = alpha[nfcns - 2] - alpha[nfcns];
        j = 2;
        while j <= nm1 {
            if j >= nm1 { aa = aa * 0.5; bb = bb * 0.5; }
            p[j + 1] = 0.0;
            var k: i64 = 1;
            while k <= j { a[k] = p[k]; p[k] = 2.0 * bb * a[k]; k += 1; }
            p[2] += a[1] * 2.0 * aa;
            k = 1;
            while k <= j - 1 { p[k] += q[k] + aa * a[k + 1]; k += 1; }
            k = 3;
            while k <= j + 1 { p[k] += aa * a[k - 1]; k += 1; }
            if j != nm1 {
                k = 1;
                while k <= j { q[k] = -a[k]; k += 1; }
                q[1] += alpha[nfcns - 1 - j];
            }
            j += 1;
        }
        j = 1;
        while j <= nfcns { alpha[j] = p[j]; j += 1; }
    }
    if nfcns <= 3 {
        alpha[nfcns + 1] = 0.0;
        alpha[nfcns + 2] = 0.0;
    }
}

// remez designs the taps of the optimal filter of Parks and McClellan.
// The bands hold two edges for each value of desired, in the units of
// fs, and the edges rise. The weight holds one value for each band, or
// it stays empty for a weight of one. The kind is bandpass,
// differentiator or hilbert. The exchange stops after maxiter passes.
// It gives a fault when the deviation stops rising.
fn remez(numtaps: i64, bands: f64[:], desired: f64[:], weight: f64[:] = [], kind: const u8[:] = "bandpass", maxiter: i64 = 25, grid_density: i64 = 16, fs: f64 = 1.0) -> Arr, Fault {
    var jtype: i64 = 0;
    if kind == "bandpass" { jtype = 1; }
    else if kind == "differentiator" { jtype = 2; }
    else if kind == "hilbert" { jtype = 3; }
    else { return empty_arr(), fault(F_VALUE, "remez", "the kind is not bandpass, differentiator or hilbert"); }
    if numtaps < 2 {
        return empty_arr(), fault(F_RANGE, "remez", "the count of taps is below two");
    }
    if numtaps > MAX_REMEZ_TAPS {
        return empty_arr(), fault(F_LIMIT, "remez", "the count of taps is above 4096");
    }
    if maxiter < 1 || maxiter > MAX_REMEZ_PASSES {
        return empty_arr(), fault(F_RANGE, "remez", "maxiter stands outside one to 1000");
    }
    if grid_density < 1 || grid_density > MAX_REMEZ_DENSITY {
        return empty_arr(), fault(F_RANGE, "remez", "grid_density stands outside one to 1024");
    }
    if !(fs > 0.0) || fs - fs != 0.0 {
        return empty_arr(), fault(F_VALUE, "remez", "fs is not a finite value above zero");
    }
    let nbands = desired.len;
    if nbands == 0 || bands.len != 2 * nbands || (weight.len != 0 && weight.len != nbands) {
        return empty_arr(), fault(F_SHAPE, "remez", "bands holds two edges for each value of desired, and weight one value for each band");
    }
    // The arrays keep the index one first, as the routine does.
    var edge: f64[>..] = [0.0];
    var fx: f64[>..] = [0.0];
    var wtx: f64[>..] = [0.0];
    var old: f64 = 0.0;
    var i: i64 = 0;
    while i < bands.len {
        let b = bands[i];
        if !(b >= old) || b - b != 0.0 {
            return empty_arr(), fault(F_VALUE, "remez", "the bands do not rise from zero");
        }
        if b * 2.0 > fs {
            return empty_arr(), fault(F_RANGE, "remez", "a band edge lies above half of fs");
        }
        old = b;
        edge.push(b / fs);
        i += 1;
    }
    i = 0;
    while i < nbands {
        let d = desired[i];
        if d - d != 0.0 {
            return empty_arr(), fault(F_VALUE, "remez", "desired holds a value that is not finite");
        }
        fx.push(d);
        var w: f64 = 1.0;
        if weight.len != 0 { w = weight[i]; }
        if !(w > 0.0) || w - w != 0.0 {
            return empty_arr(), fault(F_VALUE, "remez", "a weight is not a finite value above zero");
        }
        wtx.push(w);
        i += 1;
    }
    let neg = jtype != 1;
    let nodd = numtaps % 2 == 1;
    var nfcns = numtaps / 2;
    if nodd && !neg { nfcns += 1; }
    let dimsize = (numtaps + 1) / 2 + 2;
    let wrksize = grid_density * dimsize;
    if wrksize > MAX_REMEZ_GRID {
        return empty_arr(), fault(F_LIMIT, "remez", "the grid would hold more points than the limit, so reduce grid_density or the count of taps");
    }
    // The dense grid and the desired response and the weight on it.
    var des: f64[>..] = [0.0];
    var wt: f64[>..] = [0.0];
    let delf = 0.5 / (grid_density * nfcns * 1.0);
    var first = edge[1];
    if neg && first < delf { first = delf; }
    // The grid g keeps the index one first, so slot zero stays unused.
    var g: f64[>..] = [0.0, first];
    var j: i64 = 1;
    var l: i64 = 1;
    var lband: i64 = 1;
    des.push(0.0);
    wt.push(0.0);
    while true {
        let fup = edge[l + 1];
        var more = true;
        while more {
            let temp = g[j];
            var dv = fx[lband];
            var wv = wtx[lband];
            if jtype == 2 {
                dv = fx[lband] * temp;
                if fx[lband] >= 0.0001 { wv = wtx[lband] / temp; }
            }
            des[j] = dv;
            wt[j] = wv;
            j += 1;
            if j > wrksize {
                return empty_arr(), fault(F_LIMIT, "remez", "the grid holds too many points, so reduce grid_density or the count of taps");
            }
            g.push(temp + delf);
            des.push(0.0);
            wt.push(0.0);
            more = g[j] <= fup;
        }
        g[j - 1] = fup;
        var d2 = fx[lband];
        var w2 = wtx[lband];
        if jtype == 2 {
            d2 = fx[lband] * fup;
            if fx[lband] >= 0.0001 { w2 = wtx[lband] / fup; }
        }
        des[j - 1] = d2;
        wt[j - 1] = w2;
        lband += 1;
        l += 2;
        if lband > nbands { break; }
        g[j] = edge[l];
    }
    var ngrid = j - 1;
    if neg == nodd {
        if g[ngrid] > 0.5 - delf { ngrid -= 1; }
    }
    if ngrid < nfcns + 1 {
        return empty_arr(), fault(F_RANGE, "remez", "the bands are too close for the grid to hold the extremal set");
    }
    // The equivalent problem of the cosine series.
    j = 1;
    while j <= ngrid {
        var change: f64 = 1.0;
        if !neg { if !nodd { change = mathshim::cosine(REMEZ_PI * g[j]); } }
        else if !nodd { change = mathshim::sine(REMEZ_PI * g[j]); }
        else { change = mathshim::sine(2.0 * REMEZ_PI * g[j]); }
        des[j] = des[j] / change;
        wt[j] = wt[j] * change;
        j += 1;
    }
    var iext: i64[>..] = [];
    i = 0;
    while i < nfcns + 4 { iext.push(0); i += 1; }
    let temp = (ngrid - 1) * 1.0 / nfcns;
    j = 1;
    while j <= nfcns { iext[j] = ((j - 1) * temp) as i64 + 1; j += 1; }
    iext[nfcns + 1] = ngrid;
    var x: f64[>..] = [];
    var y: f64[>..] = [];
    var ad: f64[>..] = [];
    var alpha: f64[>..] = [];
    i = 0;
    while i < nfcns + 4 { x.push(0.0); y.push(0.0); ad.push(0.0); alpha.push(0.0); i += 1; }
    let st = rz_exchange(des, g, wt, ngrid, iext, nfcns, maxiter, x, y, ad);
    if st == -1 {
        return empty_arr(), fault(F_VALUE, "remez", "the exchange failed to converge, so reduce the width of the transition bands");
    }
    if st == -2 {
        return empty_arr(), fault(F_LIMIT, "remez", "the exchange passed the bound of its steps");
    }
    rz_coeffs(g, ngrid, edge, nbands, nfcns, x, y, ad, alpha);
    // The taps.
    let nz = nfcns + 1;
    let nm1 = nfcns - 1;
    var h: f64[>..] = [];
    i = 0;
    while i < numtaps + 2 { h.push(0.0); i += 1; }
    if !neg {
        if nodd {
            j = 1;
            while j <= nm1 { h[j] = 0.5 * alpha[nz - j]; j += 1; }
            h[nfcns] = alpha[1];
        } else {
            h[1] = 0.25 * alpha[nfcns];
            j = 2;
            while j <= nm1 { h[j] = 0.25 * (alpha[nz - j] + alpha[nfcns + 2 - j]); j += 1; }
            h[nfcns] = 0.5 * alpha[1] + 0.25 * alpha[2];
        }
    } else {
        if nodd {
            h[1] = 0.25 * alpha[nfcns];
            h[2] = 0.25 * alpha[nm1];
            j = 3;
            while j <= nm1 { h[j] = 0.25 * (alpha[nz - j] - alpha[nfcns + 3 - j]); j += 1; }
            h[nfcns] = 0.5 * alpha[1] - 0.25 * alpha[3];
            h[nz] = 0.0;
        } else {
            h[1] = 0.25 * alpha[nfcns];
            j = 2;
            while j <= nm1 { h[j] = 0.25 * (alpha[nz - j] - alpha[nfcns + 2 - j]); j += 1; }
            h[nfcns] = 0.5 * alpha[1] - 0.25 * alpha[2];
        }
    }
    j = 1;
    while j <= nfcns {
        let k = numtaps + 1 - j;
        if !neg { h[k] = h[j]; } else { h[k] = -h[j]; }
        j += 1;
    }
    if neg && nodd { h[nz] = 0.0; }
    var out: f64[>..] = [];
    j = 1;
    while j <= numtaps { out.push(h[j]); j += 1; }
    return line_arr(copy(out)), no_fault();
}
