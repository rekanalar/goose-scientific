// The least squares solver LSMR.
//
// The method follows the updates of SciPy, so the solution and the
// estimates match. It starts from zero and takes no start vector.
namespace scientific;

// Lsmr holds the result of the solver: the solution, the reason to stop,
// the count of the turns and the estimates of the method.
struct Lsmr { istop: i64, itn: i64, normr: f64, normar: f64, norma: f64, conda: f64, normx: f64, x: Arr }

fn lsmr_sign(a: f64) -> f64 {
    if a < 0.0 { return -1.0; }
    if a > 0.0 { return 1.0; }
    return 0.0;
}

// lsmr_ortho gives the rotation c, s, r of a and b, as the symmetric
// rotation of SciPy gives it.
fn lsmr_ortho(a: f64, b: f64) -> f64, f64, f64 {
    if b == 0.0 { return lsmr_sign(a), 0.0, mathshim::abs_of(a); }
    if a == 0.0 { return 0.0, lsmr_sign(b), mathshim::abs_of(b); }
    if mathshim::abs_of(b) > mathshim::abs_of(a) {
        let tau = a / b;
        let s = lsmr_sign(b) / mathshim::root(1.0 + tau * tau);
        return s * tau, s, b / s;
    }
    let tau = b / a;
    let c = lsmr_sign(a) / mathshim::root(1.0 + tau * tau);
    return c, c * tau, a / c;
}

// lsmr_empty gives the result of a failed call.
fn lsmr_empty() -> Lsmr {
    return Lsmr { istop: 0, itn: 0, normr: 0.0, normar: 0.0, norma: 0.0, conda: 1.0, normx: 0.0, x: empty_arr() };
}

// lsmr solves the least squares problem of a sparse matrix of any shape
// by the method LSMR of Fong and Saunders, from a start of zero. It
// minimizes the norm of the residual with the damping term. The matrix
// of the problem gets the rows of damp times the unit matrix. The
// result holds the reason to stop in istop, as SciPy gives it: zero for
// a zero solution, one to six for the tests of the method, and seven
// when maxiter ran out. The default limit is the lesser side.
fn lsmr(s, b: f64[:], damp: f64 = 0.0, atol: f64 = 1.0e-6, btol: f64 = 1.0e-6, conlim: f64 = 1.0e8, maxiter: i64 = 0) -> Lsmr, Fault {
    if b.len != s.rows || b.len < 1 {
        return lsmr_empty(), fault(F_SHAPE, "lsmr", "the right side does not fit the matrix");
    }
    if !(atol >= 0.0) || !(btol >= 0.0) || !(damp >= 0.0) || atol - atol != 0.0 || btol - btol != 0.0 || damp - damp != 0.0 {
        return lsmr_empty(), fault(F_VALUE, "lsmr", "a tolerance or the damping is negative or not finite");
    }
    if conlim != conlim {
        return lsmr_empty(), fault(F_VALUE, "lsmr", "the limit of the condition number is not a number");
    }
    if !krylov_norm_ok(b) {
        return lsmr_empty(), fault(F_VALUE, "lsmr", "the right side holds a value that is not finite, too large or too small");
    }
    let normb = krylov_norm(b);
    let m = s.rows;
    let n = s.cols;
    var limit = maxiter;
    if limit < 1 { limit = m; if n < m { limit = n; } }
    let st = csr_transpose(s);
    var x: f64[>..] = nd_zeros(n);
    var u: f64[>..] = copy_slice(b);
    var beta = normb;
    var v: f64[>..] = nd_zeros(n);
    var alpha: f64 = 0.0;
    var i: i64 = 0;
    if beta > 0.0 {
        while i < m { u[i] = u[i] / beta; i += 1; }
        let w, fw = csr_matvec(st, u);
        if fw.kind != F_NONE { return lsmr_empty(), fw; }
        i = 0;
        while i < n { v[i] = w.data[i]; i += 1; }
        alpha = krylov_norm(v);
    }
    if alpha > 0.0 {
        i = 0;
        while i < n { v[i] = v[i] / alpha; i += 1; }
    }
    var zetabar = alpha * beta;
    var alphabar = alpha;
    var rho: f64 = 1.0;
    var rhobar: f64 = 1.0;
    var cbar: f64 = 1.0;
    var sbar: f64 = 0.0;
    var h: f64[>..] = copy(v);
    var hbar: f64[>..] = nd_zeros(n);
    var betadd = beta;
    var betad: f64 = 0.0;
    var rhodold: f64 = 1.0;
    var tautildeold: f64 = 0.0;
    var thetatilde: f64 = 0.0;
    var zeta: f64 = 0.0;
    var d: f64 = 0.0;
    var norma2 = alpha * alpha;
    var maxrbar: f64 = 0.0;
    var minrbar: f64 = 1.0e100;
    var norma = mathshim::root(norma2);
    var conda: f64 = 1.0;
    var normx: f64 = 0.0;
    var istop: i64 = 0;
    var ctol: f64 = 0.0;
    if conlim > 0.0 { ctol = 1.0 / conlim; }
    var normr = beta;
    var normar = alpha * beta;
    var itn: i64 = 0;
    if normar == 0.0 || normb == 0.0 {
        return Lsmr { istop: 0, itn: 0, normr: normr, normar: normar, norma: norma, conda: conda, normx: normx, x: line_arr(copy(x)) }, no_fault();
    }
    while itn < limit {
        itn += 1;
        let av, fa = csr_matvec(s, v);
        if fa.kind != F_NONE { return lsmr_empty(), fa; }
        i = 0;
        while i < m { u[i] = av.data[i] - alpha * u[i]; i += 1; }
        beta = krylov_norm(u);
        if beta > 0.0 {
            i = 0;
            while i < m { u[i] = u[i] / beta; i += 1; }
            let atu, ft = csr_matvec(st, u);
            if ft.kind != F_NONE { return lsmr_empty(), ft; }
            i = 0;
            while i < n { v[i] = atu.data[i] - beta * v[i]; i += 1; }
            alpha = krylov_norm(v);
            if alpha > 0.0 {
                i = 0;
                while i < n { v[i] = v[i] / alpha; i += 1; }
            }
        }
        let chat, shat, alphahat = lsmr_ortho(alphabar, damp);
        let rhoold = rho;
        let c, sn, rn = lsmr_ortho(alphahat, beta);
        rho = rn;
        let thetanew = sn * alpha;
        alphabar = c * alpha;
        let rhobarold = rhobar;
        let zetaold = zeta;
        let thetabar = sbar * rho;
        let rhotemp = cbar * rho;
        let cb, sb, rb = lsmr_ortho(cbar * rho, thetanew);
        cbar = cb; sbar = sb; rhobar = rb;
        zeta = cbar * zetabar;
        zetabar = -sbar * zetabar;
        let fh = -(thetabar * rho / (rhoold * rhobarold));
        let fx = zeta / (rho * rhobar);
        let fv = -(thetanew / rho);
        i = 0;
        while i < n {
            hbar[i] = fh * hbar[i] + h[i];
            x[i] = x[i] + fx * hbar[i];
            h[i] = fv * h[i] + v[i];
            i += 1;
        }
        let betaacute = chat * betadd;
        let betacheck = -shat * betadd;
        let betahat = c * betaacute;
        betadd = -sn * betaacute;
        let thetatildeold = thetatilde;
        let ctildeold, stildeold, rhotildeold = lsmr_ortho(rhodold, thetabar);
        thetatilde = stildeold * rhobar;
        rhodold = ctildeold * rhobar;
        betad = -stildeold * betad + ctildeold * betahat;
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold;
        let taud = (zeta - thetatilde * tautildeold) / rhodold;
        d = d + betacheck * betacheck;
        normr = mathshim::root(d + (betad - taud) * (betad - taud) + betadd * betadd);
        norma2 = norma2 + beta * beta;
        norma = mathshim::root(norma2);
        norma2 = norma2 + alpha * alpha;
        if rhobarold > maxrbar { maxrbar = rhobarold; }
        if itn > 1 && rhobarold < minrbar { minrbar = rhobarold; }
        var hi = maxrbar;
        if rhotemp > hi { hi = rhotemp; }
        var lo = minrbar;
        if rhotemp < lo { lo = rhotemp; }
        conda = hi / lo;
        normar = mathshim::abs_of(zetabar);
        normx = krylov_norm(x);
        let test1 = normr / normb;
        var test2 = 1.0 / 0.0;
        if norma * normr != 0.0 { test2 = normar / (norma * normr); }
        let test3 = 1.0 / conda;
        let t1 = test1 / (1.0 + norma * normx / normb);
        let rtol = btol + atol * norma * normx / normb;
        if itn >= limit { istop = 7; }
        if 1.0 + test3 <= 1.0 { istop = 6; }
        if 1.0 + test2 <= 1.0 { istop = 5; }
        if 1.0 + t1 <= 1.0 { istop = 4; }
        if test3 <= ctol { istop = 3; }
        if test2 <= atol { istop = 2; }
        if test1 <= rtol { istop = 1; }
        if istop > 0 { break; }
    }
    return Lsmr { istop: istop, itn: itn, normr: normr, normar: normar, norma: norma, conda: conda, normx: normx, x: line_arr(copy(x)) }, no_fault();
}
