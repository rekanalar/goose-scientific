// The public calls of the orthogonal polynomial bases.
//
// Each call fixes the family and hands the work to the core. A series holds
// its coefficients from the lowest degree to the highest, as NumPy holds
// them. The names follow numpy.polynomial.
namespace scientific;

// legval gives the value of a Legendre series at x.
fn legval(x: f64, c: f64[:]) -> f64, Fault { return pb_val(PB_LEG, x, c, "legval"); }

// leg2poly gives the monomial coefficients of a Legendre series.
fn leg2poly(c: f64[:]) -> Arr, Fault { return pb_convert(PB_LEG, c, true, "leg2poly"); }

// poly2leg gives the Legendre series of monomial coefficients.
fn poly2leg(p: f64[:]) -> Arr, Fault { return pb_convert(PB_LEG, p, false, "poly2leg"); }

// legder gives the derivative of order m of a Legendre series.
fn legder(c: f64[:], m: i64 = 1, scl: f64 = 1.0) -> Arr, Fault { return pb_der(PB_LEG, c, m, scl, "legder"); }

// legint gives the integral of order m of a Legendre series.
fn legint(c: f64[:], m: i64 = 1, k: f64[:] = [], lbnd: f64 = 0.0, scl: f64 = 1.0) -> Arr, Fault {
    return pb_int(PB_LEG, c, m, k, lbnd, scl, "legint");
}

// legroots gives the roots of a Legendre series: the real parts, then the
// imaginary parts.
fn legroots(c: f64[:]) -> Arr, Arr, Fault { return pb_roots(PB_LEG, c, "legroots"); }

// chebval gives the value of a Chebyshev series at x.
fn chebval(x: f64, c: f64[:]) -> f64, Fault { return pb_val(PB_CHEB, x, c, "chebval"); }

// cheb2poly gives the monomial coefficients of a Chebyshev series.
fn cheb2poly(c: f64[:]) -> Arr, Fault { return pb_convert(PB_CHEB, c, true, "cheb2poly"); }

// poly2cheb gives the Chebyshev series of monomial coefficients.
fn poly2cheb(p: f64[:]) -> Arr, Fault { return pb_convert(PB_CHEB, p, false, "poly2cheb"); }

// chebder gives the derivative of order m of a Chebyshev series.
fn chebder(c: f64[:], m: i64 = 1, scl: f64 = 1.0) -> Arr, Fault { return pb_der(PB_CHEB, c, m, scl, "chebder"); }

// chebint gives the integral of order m of a Chebyshev series.
fn chebint(c: f64[:], m: i64 = 1, k: f64[:] = [], lbnd: f64 = 0.0, scl: f64 = 1.0) -> Arr, Fault {
    return pb_int(PB_CHEB, c, m, k, lbnd, scl, "chebint");
}

// chebroots gives the roots of a Chebyshev series: the real parts, then the
// imaginary parts.
fn chebroots(c: f64[:]) -> Arr, Arr, Fault { return pb_roots(PB_CHEB, c, "chebroots"); }

// hermval gives the value of a Hermite series at x.
fn hermval(x: f64, c: f64[:]) -> f64, Fault { return pb_val(PB_HERM, x, c, "hermval"); }

// herm2poly gives the monomial coefficients of a Hermite series.
fn herm2poly(c: f64[:]) -> Arr, Fault { return pb_convert(PB_HERM, c, true, "herm2poly"); }

// poly2herm gives the Hermite series of monomial coefficients.
fn poly2herm(p: f64[:]) -> Arr, Fault { return pb_convert(PB_HERM, p, false, "poly2herm"); }

// hermder gives the derivative of order m of a Hermite series.
fn hermder(c: f64[:], m: i64 = 1, scl: f64 = 1.0) -> Arr, Fault { return pb_der(PB_HERM, c, m, scl, "hermder"); }

// hermint gives the integral of order m of a Hermite series.
fn hermint(c: f64[:], m: i64 = 1, k: f64[:] = [], lbnd: f64 = 0.0, scl: f64 = 1.0) -> Arr, Fault {
    return pb_int(PB_HERM, c, m, k, lbnd, scl, "hermint");
}

// hermroots gives the roots of a Hermite series: the real parts, then the
// imaginary parts.
fn hermroots(c: f64[:]) -> Arr, Arr, Fault { return pb_roots(PB_HERM, c, "hermroots"); }

// hermeval gives the value of a HermiteE series at x.
fn hermeval(x: f64, c: f64[:]) -> f64, Fault { return pb_val(PB_HERME, x, c, "hermeval"); }

// herme2poly gives the monomial coefficients of a HermiteE series.
fn herme2poly(c: f64[:]) -> Arr, Fault { return pb_convert(PB_HERME, c, true, "herme2poly"); }

// poly2herme gives the HermiteE series of monomial coefficients.
fn poly2herme(p: f64[:]) -> Arr, Fault { return pb_convert(PB_HERME, p, false, "poly2herme"); }

// hermeder gives the derivative of order m of a HermiteE series.
fn hermeder(c: f64[:], m: i64 = 1, scl: f64 = 1.0) -> Arr, Fault { return pb_der(PB_HERME, c, m, scl, "hermeder"); }

// hermeint gives the integral of order m of a HermiteE series.
fn hermeint(c: f64[:], m: i64 = 1, k: f64[:] = [], lbnd: f64 = 0.0, scl: f64 = 1.0) -> Arr, Fault {
    return pb_int(PB_HERME, c, m, k, lbnd, scl, "hermeint");
}

// hermeroots gives the roots of a HermiteE series: the real parts, then the
// imaginary parts.
fn hermeroots(c: f64[:]) -> Arr, Arr, Fault { return pb_roots(PB_HERME, c, "hermeroots"); }

// lagval gives the value of a Laguerre series at x.
fn lagval(x: f64, c: f64[:]) -> f64, Fault { return pb_val(PB_LAG, x, c, "lagval"); }

// lag2poly gives the monomial coefficients of a Laguerre series.
fn lag2poly(c: f64[:]) -> Arr, Fault { return pb_convert(PB_LAG, c, true, "lag2poly"); }

// poly2lag gives the Laguerre series of monomial coefficients.
fn poly2lag(p: f64[:]) -> Arr, Fault { return pb_convert(PB_LAG, p, false, "poly2lag"); }

// lagder gives the derivative of order m of a Laguerre series.
fn lagder(c: f64[:], m: i64 = 1, scl: f64 = 1.0) -> Arr, Fault { return pb_der(PB_LAG, c, m, scl, "lagder"); }

// lagint gives the integral of order m of a Laguerre series.
fn lagint(c: f64[:], m: i64 = 1, k: f64[:] = [], lbnd: f64 = 0.0, scl: f64 = 1.0) -> Arr, Fault {
    return pb_int(PB_LAG, c, m, k, lbnd, scl, "lagint");
}

// lagroots gives the roots of a Laguerre series: the real parts, then the
// imaginary parts.
fn lagroots(c: f64[:]) -> Arr, Arr, Fault { return pb_roots(PB_LAG, c, "lagroots"); }
