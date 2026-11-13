// The scale of a tridiagonal matrix.
//
// The solver of the tridiagonal matrix holds absolute limits: a width at
// which the bisection stops, a floor of the pivot and a threshold of the
// residual. They suit a matrix whose largest entry lies near one. These
// helpers scale a matrix by an exact power of two, so that its largest
// entry lies in the range 0.5 to 1. A power of two changes no digit of a
// value. The values scale back by the inverse power. A vector does not
// change by a scale.
namespace scientific;

// bd_pow2 gives two to the power k, for k from minus 1000 to 1000. The
// loop multiplies by two or one half, and each step is exact.
fn bd_pow2(k: i64) -> f64 {
    var p: f64 = 1.0;
    var i: i64 = 0;
    if k >= 0 {
        while i < k { p = p * 2.0; i += 1; }
    } else {
        while i > k { p = p * 0.5; i -= 1; }
    }
    return p;
}

// bd_mul2 gives x times two to the power k. The power splits in two
// halves, so that each factor stays finite for any k from minus 2000 to
// 2000.
fn bd_mul2(x: f64, k: i64) -> f64 {
    let k1 = k / 2;
    return x * bd_pow2(k1) * bd_pow2(k - k1);
}

// bd_exp gives the power k such that the largest absolute entry of the
// diagonal d and of the off diagonal e, times two to the power k, lies
// in the range 0.5 to 1. A matrix of zeros gives zero. The entries are
// finite, and the loops run at most 1100 steps.
fn bd_exp(d: f64[:], e: f64[:], n: i64) -> i64 {
    var m: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        let v = mathshim::abs_of(d[i]);
        if v > m { m = v; }
        i += 1;
    }
    i = 0;
    while i < n - 1 {
        let v = mathshim::abs_of(e[i]);
        if v > m { m = v; }
        i += 1;
    }
    if m == 0.0 { return 0; }
    var k: i64 = 0;
    // 2^512 and 2^-512.
    while m < 7.458340731200207e-155 { m = m * 1.3407807929942597e154; k += 512; }
    while m >= 1.3407807929942597e154 { m = m * 7.458340731200207e-155; k -= 512; }
    while m < 0.5 { m = m * 2.0; k += 1; }
    while m >= 1.0 { m = m * 0.5; k -= 1; }
    return k;
}

// bd_scaled gives a copy of the first n values of v times two to the
// power k.
fn bd_scaled(v: f64[:], n: i64, k: i64) -> f64[>..] {
    let k1 = k / 2;
    let f1 = bd_pow2(k1);
    let f2 = bd_pow2(k - k1);
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n { out.push(v[i] * f1 * f2); i += 1; }
    return out;
}
