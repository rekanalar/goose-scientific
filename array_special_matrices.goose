// The special matrices and the structure tests of scipy.linalg:
// companion, hadamard, pascal, invpascal, invhilbert, leslie, fiedler,
// fiedler_companion, helmert, convolution_matrix, khatri_rao,
// bandwidth, issymmetric and ishermitian. Every size is bounded, so
// that an input cannot ask for a result of unbounded size.
namespace scientific;

import mathshim;

// The largest side of a special matrix. The result holds at most
// MAX_ELEMENTS elements.
const MAX_SIDE: i64 = 4096;

fn sm_side_fault(tag: const u8[:], n: i64) -> Fault {
    if n < 1 { return fault(F_VALUE, tag, "the size is below one"); }
    if n > MAX_SIDE { return fault(F_LIMIT, tag, "the size is above the limit"); }
    return no_fault();
}

fn sm_zeros(r: i64, c: i64) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < r * c { d.push(0.0); i += 1; }
    return d;
}

// companion gives the companion matrix of the polynomial with the
// coefficients a, highest power first. The first row holds the
// negated coefficients divided by the first one. The sub-diagonal
// holds ones. The first coefficient must not be zero.
fn companion(a: f64[:]) -> Arr, Fault {
    if a.len < 2 { return empty_arr(), fault(F_SHAPE, "companion", "the coefficients hold fewer than two values"); }
    if a.len - 1 > MAX_SIDE { return empty_arr(), fault(F_LIMIT, "companion", "the degree is above the limit"); }
    if a[0] == 0.0 || !(a[0] - a[0] == 0.0) { return empty_arr(), fault(F_VALUE, "companion", "the first coefficient is zero or not finite"); }
    let n = a.len - 1;
    var d = sm_zeros(n, n);
    var j: i64 = 0;
    while j < n { d[j] = -a[j + 1] / a[0]; j += 1; }
    j = 1;
    while j < n { d[j * n + j - 1] = 1.0; j += 1; }
    return mat2(copy(d), n, n), no_fault();
}

// fiedler_companion gives the companion matrix of Fiedler, the sparse
// form with the coefficients spread over the rows.
fn fiedler_companion(a: f64[:]) -> Arr, Fault {
    if a.len < 2 { return empty_arr(), fault(F_SHAPE, "fiedler_companion", "the coefficients hold fewer than two values"); }
    if a.len - 1 > MAX_SIDE { return empty_arr(), fault(F_LIMIT, "fiedler_companion", "the degree is above the limit"); }
    if a[0] == 0.0 || !(a[0] - a[0] == 0.0) { return empty_arr(), fault(F_VALUE, "fiedler_companion", "the first coefficient is zero or not finite"); }
    let n = a.len - 1;
    var d = sm_zeros(n, n);
    var k: i64 = 1;
    while k <= n {
        let v = -a[k] / a[0];
        // An odd coefficient sits on the row k - 1 and the even one on k - 2.
        if k == 1 {
            d[0] = v;
            if n > 1 { d[n] = 1.0; }
        } else if k % 2 == 1 {
            d[(k - 1) * n + k - 2] = v;
            if k < n { d[k * n + k - 2] = 1.0; }
        } else {
            d[(k - 2) * n + k - 1] = v;
            if k < n { d[(k - 2) * n + k] = 1.0; }
        }
        k += 1;
    }
    return mat2(copy(d), n, n), no_fault();
}

// hadamard gives the Hadamard matrix of the order n, a power of two,
// by the construction of Sylvester.
fn hadamard(n: i64) -> Arr, Fault {
    if n < 1 || n > MAX_SIDE || (n & (n - 1)) != 0 {
        return empty_arr(), fault(F_VALUE, "hadamard", "the order is not a power of two inside the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            // The sign is the parity of the bits that i and j share.
            var b = i & j;
            var s: i64 = 0;
            while b > 0 { s += b & 1; b = b >> 1; }
            if s % 2 == 0 { d.push(1.0); } else { d.push(-1.0); }
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d), n, n), no_fault();
}

// sm_binomials gives the table of the binomials by the rule of Pascal,
// n rows of n columns, with the lower triangle filled.
fn sm_binomials(n: i64) -> f64[>..] {
    var t = sm_zeros(n, n);
    var i: i64 = 0;
    while i < n {
        t[i * n] = 1.0;
        var j: i64 = 1;
        while j <= i {
            t[i * n + j] = t[(i - 1) * n + j - 1];
            if j < i { t[i * n + j] = t[i * n + j] + t[(i - 1) * n + j]; }
            j += 1;
        }
        i += 1;
    }
    return t;
}

// pascal gives the Pascal matrix of the order n. The kind is
// "symmetric", "lower" or "upper".
fn pascal(n: i64, kind: const u8[:] = "symmetric") -> Arr, Fault {
    let fs = sm_side_fault("pascal", n);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    if kind != "symmetric" && kind != "lower" && kind != "upper" {
        return empty_arr(), fault(F_VALUE, "pascal", "the kind is not symmetric, lower or upper");
    }
    if kind == "symmetric" {
        if n > 1000 { return empty_arr(), fault(F_LIMIT, "pascal", "the symmetric form is limited to the order 1000"); }
        var d = sm_zeros(n, n);
        var i: i64 = 0;
        while i < n {
            var j: i64 = 0;
            while j < n {
                if i == 0 || j == 0 { d[i * n + j] = 1.0; }
                else { d[i * n + j] = d[(i - 1) * n + j] + d[i * n + j - 1]; }
                j += 1;
            }
            i += 1;
        }
        return mat2(copy(d), n, n), no_fault();
    }
    let t = sm_binomials(n);
    var d = sm_zeros(n, n);
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j <= i {
            if kind == "lower" { d[i * n + j] = t[i * n + j]; } else { d[j * n + i] = t[i * n + j]; }
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d), n, n), no_fault();
}

// invpascal gives the inverse of the Pascal matrix. The entries are
// integers, summed in 64-bit integers. The order is limited to 30,
// where the entries still fit.
fn invpascal(n: i64, kind: const u8[:] = "symmetric") -> Arr, Fault {
    let fs = sm_side_fault("invpascal", n);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    if kind != "symmetric" && kind != "lower" && kind != "upper" {
        return empty_arr(), fault(F_VALUE, "invpascal", "the kind is not symmetric, lower or upper");
    }
    if n > 30 { return empty_arr(), fault(F_LIMIT, "invpascal", "the order is above 30"); }
    let t = sm_binomials(n);
    // The inverse of the lower matrix has the entries (-1)^(i-j) C(i, j).
    var li: i64[>..] = [];
    var i: i64 = 0;
    while i < n * n {
        li.push(0);
        i += 1;
    }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j <= i {
            var v = t[i * n + j] as i64;
            if (i - j) % 2 == 1 { v = -v; }
            li[i * n + j] = v;
            j += 1;
        }
        i += 1;
    }
    var d: f64[>..] = [];
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            var v: i64 = 0;
            if kind == "lower" { v = li[i * n + j]; }
            else if kind == "upper" { v = li[j * n + i]; }
            else {
                // The sum over k of li[k][i] li[k][j].
                var k: i64 = 0;
                while k < n { v += li[k * n + i] * li[k * n + j]; k += 1; }
            }
            d.push(v * 1.0);
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d), n, n), no_fault();
}

// sm_choose gives the binomial n over k in floating point.
fn sm_choose(n: i64, k: i64) -> f64 {
    if k < 0 || k > n { return 0.0; }
    var kk = k;
    if n - k < kk { kk = n - k; }
    var r: f64 = 1.0;
    var i: i64 = 1;
    while i <= kk {
        r = r * ((n - kk + i) * 1.0) / (i * 1.0);
        i += 1;
    }
    return r;
}

// invhilbert gives the inverse of the Hilbert matrix of the order n by
// the closed form, in floating point. The order is limited to 150.
fn invhilbert(n: i64) -> Arr, Fault {
    let fs = sm_side_fault("invhilbert", n);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    if n > 150 { return empty_arr(), fault(F_LIMIT, "invhilbert", "the order is above 150"); }
    var d = sm_zeros(n, n);
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j <= i {
            let s = i + j;
            var v = (s + 1) * 1.0 * sm_choose(n + i, n - j - 1) * sm_choose(n + j, n - i - 1) * sm_choose(s, i) * sm_choose(s, i);
            if s % 2 == 1 { v = -v; }
            d[i * n + j] = v;
            d[j * n + i] = v;
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d), n, n), no_fault();
}

// leslie gives the Leslie matrix of the fecundities f and the survival
// rates s. The first holds n values, the second n minus one.
fn leslie(f: f64[:], s: f64[:]) -> Arr, Fault {
    let n = f.len;
    if n < 1 || n > MAX_SIDE { return empty_arr(), fault(F_SHAPE, "leslie", "the fecundities hold no value or too many"); }
    if s.len != n - 1 { return empty_arr(), fault(F_SHAPE, "leslie", "the survival rates must hold one value fewer than the fecundities"); }
    var d = sm_zeros(n, n);
    var j: i64 = 0;
    while j < n { d[j] = f[j]; j += 1; }
    j = 0;
    while j < n - 1 { d[(j + 1) * n + j] = s[j]; j += 1; }
    return mat2(copy(d), n, n), no_fault();
}

// fiedler gives the symmetric matrix of the absolute differences of
// the values.
fn fiedler(a: f64[:]) -> Arr, Fault {
    let n = a.len;
    if n < 1 || n > MAX_SIDE { return empty_arr(), fault(F_SHAPE, "fiedler", "the values hold no value or too many"); }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            var v = a[i] - a[j];
            if v < 0.0 { v = -v; }
            d.push(v);
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d), n, n), no_fault();
}

// helmert gives the Helmert matrix of the order n. The flag full adds
// the first row of equal entries. Without it the matrix holds n minus
// one rows.
fn helmert(n: i64, full: bool = false) -> Arr, Fault {
    let fs = sm_side_fault("helmert", n);
    if fs.kind != F_NONE { return empty_arr(), fs; }
    var d: f64[>..] = [];
    if full {
        var j: i64 = 0;
        while j < n { d.push(1.0 / mathshim::root(n * 1.0)); j += 1; }
    }
    var k: i64 = 1;
    while k < n {
        let w = 1.0 / mathshim::root((k * (k + 1)) * 1.0);
        var j: i64 = 0;
        while j < n {
            if j < k { d.push(w); }
            else if j == k { d.push(-(k * 1.0) * w); }
            else { d.push(0.0); }
            j += 1;
        }
        k += 1;
    }
    var rows = n - 1;
    if full { rows = n; }
    if rows == 0 { return empty_arr(), no_fault(); }
    return mat2(copy(d), rows, n), no_fault();
}

// convolution_matrix gives the matrix that applies the convolution
// with the kernel a to a vector of n values. The mode is "full",
// "valid" or "same", as in numpy.convolve.
fn convolution_matrix(a: f64[:], n: i64, mode: const u8[:] = "full") -> Arr, Fault {
    let m = a.len;
    if m < 1 { return empty_arr(), fault(F_SHAPE, "convolution_matrix", "the kernel holds no value"); }
    if n < 1 { return empty_arr(), fault(F_VALUE, "convolution_matrix", "the size is below one"); }
    if mode != "full" && mode != "valid" && mode != "same" {
        return empty_arr(), fault(F_VALUE, "convolution_matrix", "the mode is not full, valid or same");
    }
    if (m + n) * n > MAX_ELEMENTS { return empty_arr(), fault(F_LIMIT, "convolution_matrix", "the result holds more elements than the limit"); }
    var tb: i64 = 0;
    var te: i64 = 0;
    var trim = m;
    if n < trim { trim = n; }
    trim -= 1;
    if mode == "same" { tb = trim / 2; te = trim - tb; }
    if mode == "valid" { tb = trim; te = trim; }
    let rows = m + n - 1 - tb - te;
    var d: f64[>..] = [];
    var i: i64 = tb;
    while i < m + n - 1 - te {
        var j: i64 = 0;
        while j < n {
            let k = i - j;
            if k >= 0 && k < m { d.push(a[k]); } else { d.push(0.0); }
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d), rows, n), no_fault();
}

// khatri_rao gives the column-wise Kronecker product of two matrices
// with the same count of columns.
fn khatri_rao(a, b) -> Arr, Fault {
    if a.ndim != 2 || b.ndim != 2 || a.shape[1] != b.shape[1] {
        return empty_arr(), fault(F_SHAPE, "khatri_rao", "the operands are not two matrices with the same count of columns");
    }
    let ra = a.shape[0];
    let rb = b.shape[0];
    let c = a.shape[1];
    if ra * rb * c > MAX_ELEMENTS { return empty_arr(), fault(F_LIMIT, "khatri_rao", "the result holds more elements than the limit"); }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < ra {
        var j: i64 = 0;
        while j < rb {
            var k: i64 = 0;
            while k < c { d.push(a.data[i * c + k] * b.data[j * c + k]); k += 1; }
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d), ra * rb, c), no_fault();
}

// bw_scan gives the largest distance below and above the main diagonal
// of the values that are not zero. The matrix of r rows and c columns
// starts at the place off of the buffer d.
fn bw_scan(d: f64[:], off: i64, r: i64, c: i64) -> i64, i64 {
    var lo: i64 = 0;
    var hi: i64 = 0;
    var i: i64 = 0;
    while i < r {
        var j: i64 = 0;
        while j < c {
            if d[off + i * c + j] != 0.0 {
                if i - j > lo { lo = i - j; }
                if j - i > hi { hi = j - i; }
            }
            j += 1;
        }
        i += 1;
    }
    return lo, hi;
}

// bandwidth gives the count of the non-zero diagonals below and above
// the main one. A matrix of zeros gives zero and zero.
fn bandwidth(a) -> i64, i64, Fault {
    if a.ndim != 2 { return 0, 0, fault(F_SHAPE, "bandwidth", "the operand is not a matrix"); }
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "bandwidth");
    if f.kind != F_NONE { return 0, 0, f; }
    let lo, hi = bw_scan(a.data, 0, a.shape[0], a.shape[1]);
    return lo, hi, no_fault();
}

// bandwidth_stack gives the count of the non-zero diagonals below and
// above the main one for every matrix of a stack. The last two
// dimensions of the array give the matrix. The dimensions before them
// give the batch. The result holds two arrays of the batch shape. A
// matrix without a batch gives the shape [1].
fn bandwidth_stack(a) -> Arr, Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "bandwidth_stack");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    if a.ndim < 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "bandwidth_stack", "the operand holds fewer than two dimensions");
    }
    let r = a.shape[a.ndim - 2];
    let c = a.shape[a.ndim - 1];
    var batch = shape_batch(a.shape, 2);
    let n = batch_count(batch);
    var lo: f64[>..] = [];
    var hi: f64[>..] = [];
    var s: i64 = 0;
    while s < n {
        let l, h = bw_scan(a.data, s * r * c, r, c);
        lo.push(l as f64);
        hi.push(h as f64);
        s += 1;
    }
    // A matrix without a batch gives the shape [1].
    if batch.len == 0 { batch.push(1); }
    return arr_of(copy(lo), batch), arr_of(copy(hi), batch), no_fault();
}

// issymmetric tells whether a square matrix equals its transpose. A
// tolerance atol or rtol of zero asks for exact equality. A tolerance
// above zero accepts |a - aT| up to atol + rtol * |aT|, as numpy.allclose.
// A matrix with NaN is not symmetric.
fn issymmetric(a, atol: f64 = 0.0, rtol: f64 = 0.0) -> bool, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return false, fault(F_SHAPE, "issymmetric", "the operand is not a square matrix");
    }
    if !(atol >= 0.0) || !(rtol >= 0.0) { return false, fault(F_VALUE, "issymmetric", "a tolerance is negative or NaN"); }
    let n = a.shape[0];
    var i: i64 = 0;
    while i < n {
        var j: i64 = i;
        while j < n {
            let x = a.data[i * n + j];
            let y = a.data[j * n + i];
            if x != x || y != y { return false, no_fault(); }
            var df = x - y;
            if df < 0.0 { df = -df; }
            var ay = y;
            if ay < 0.0 { ay = -ay; }
            if df > atol + rtol * ay { return false, no_fault(); }
            j += 1;
        }
        i += 1;
    }
    return true, no_fault();
}

// ishermitian is issymmetric for a real matrix.
fn ishermitian(a, atol: f64 = 0.0, rtol: f64 = 0.0) -> bool, Fault {
    return issymmetric(a, atol, rtol);
}
