// The balance of a matrix, like matrix_balance of scipy.linalg. The
// method follows LAPACK dgebal: a permutation moves the eigenvalues
// apart, and the rows and the columns are scaled by powers of two.
namespace scientific;

import mathshim;
import array_shape;

// The largest side of a matrix for the balance.
const BAL_MAX_SIDE: i64 = 4096;

// mb_norm gives the norm of a part of a column or of a row. The
// scaling keeps the result free of overflow and of underflow.
fn mb_norm(v: f64[:]) -> f64 {
    var scale: f64 = 0.0;
    var sum: f64 = 1.0;
    var i: i64 = 0;
    while i < v.len {
        let a = mathshim::abs_of(v[i]);
        if a != 0.0 {
            if scale < a {
                let r = scale / a;
                sum = 1.0 + sum * r * r;
                scale = a;
            } else {
                let r = a / scale;
                sum += r * r;
            }
        }
        i += 1;
    }
    if scale == 0.0 { return 0.0; }
    return scale * mathshim::root(sum);
}

// mb_col gives the part of a column. The buffer holds the matrix in the
// row-major order.
fn mb_col(a: f64[:], n: i64, j: i64, from: i64, to: i64) -> f64[>..] {
    var v: f64[>..] = [];
    var i: i64 = from;
    while i <= to { v.push(a[i * n + j]); i += 1; }
    return v;
}

// mb_row gives the part of a row. The buffer holds the matrix in the
// row-major order.
fn mb_row(a: f64[:], n: i64, i: i64, from: i64, to: i64) -> f64[>..] {
    var v: f64[>..] = [];
    var j: i64 = from;
    while j <= to { v.push(a[i * n + j]); j += 1; }
    return v;
}

// mb_max gives the largest absolute value of a part of a column or of a
// row.
fn mb_max(v: f64[:]) -> f64 {
    var m: f64 = 0.0;
    var i: i64 = 0;
    while i < v.len {
        let a = mathshim::abs_of(v[i]);
        if a > m { m = a; }
        i += 1;
    }
    return m;
}

// mb_balance gives the balanced matrix and the factors of the change.
// The factors hold the permutation and the scales of LAPACK dgebal. The
// matrix holds the elements in the row-major order.
fn mb_balance(a: f64[:], n: i64, permute: bool, scale: bool) -> f64[>..], f64[>..], i64, i64 {
    var m: f64[>..] = [];
    var i: i64 = 0;
    while i < a.len { m.push(a[i]); i += 1; }
    var ps: f64[>..] = [];
    i = 0;
    while i < n { ps.push(1.0); i += 1; }
    // The range of the block that holds the eigenvalues. The bounds
    // count from zero.
    var k: i64 = 0;
    var l: i64 = n - 1;
    if permute {
        // The rows that hold one eigenvalue move to the end.
        var again = true;
        while again {
            again = false;
            var r: i64 = l;
            while r >= 0 {
                var canswap = true;
                var j: i64 = 0;
                while j <= l {
                    if j != r && m[r * n + j] != 0.0 { canswap = false; break; }
                    j += 1;
                }
                if canswap {
                    ps[l] = r as f64;
                    if r != l {
                        // The columns r and l swap for the rows of the
                        // block.
                        var i2: i64 = 0;
                        while i2 <= l {
                            let t = m[i2 * n + r];
                            m[i2 * n + r] = m[i2 * n + l];
                            m[i2 * n + l] = t;
                            i2 += 1;
                        }
                        // The rows r and l swap for the columns of the
                        // matrix.
                        var j2: i64 = k;
                        while j2 < n {
                            let t = m[r * n + j2];
                            m[r * n + j2] = m[l * n + j2];
                            m[l * n + j2] = t;
                            j2 += 1;
                        }
                    }
                    again = true;
                    if l == 0 { return m, ps, 0, 0; }
                    l -= 1;
                }
                r -= 1;
            }
        }
        // The columns that hold one eigenvalue move to the front.
        again = true;
        while again {
            again = false;
            var c: i64 = k;
            while c <= l {
                var canswap = true;
                var i2: i64 = k;
                while i2 <= l {
                    if i2 != c && m[i2 * n + c] != 0.0 { canswap = false; break; }
                    i2 += 1;
                }
                if canswap {
                    ps[k] = c as f64;
                    if c != k {
                        var i3: i64 = 0;
                        while i3 <= l {
                            let t = m[i3 * n + c];
                            m[i3 * n + c] = m[i3 * n + k];
                            m[i3 * n + k] = t;
                            i3 += 1;
                        }
                        var j3: i64 = k;
                        while j3 < n {
                            let t = m[c * n + j3];
                            m[c * n + j3] = m[k * n + j3];
                            m[k * n + j3] = t;
                            j3 += 1;
                        }
                    }
                    again = true;
                    k += 1;
                }
                c += 1;
            }
        }
    }
    i = k;
    while i <= l { ps[i] = 1.0; i += 1; }
    if !scale { return m, ps, k, l; }
    // The smallest and the largest value of the scaling.
    let sfmin1 = 2.2250738585072014e-308 / 2.220446049250313e-16;
    let sfmax1 = 1.0 / sfmin1;
    let sfmin2 = sfmin1 * 2.0;
    let sfmax2 = 1.0 / sfmin2;
    var again = true;
    while again {
        again = false;
        var i2: i64 = k;
        while i2 <= l {
            var c = mb_norm(mb_col(m, n, i2, k, l));
            var r = mb_norm(mb_row(m, n, i2, k, l));
            var ca = mb_max(mb_col(m, n, i2, 0, l));
            var ra = mb_max(mb_row(m, n, i2, k, n - 1));
            if c == 0.0 || r == 0.0 { i2 += 1; continue; }
            var g = r / 2.0;
            var f = 1.0;
            let s = c + r;
            while c < g && mb_max3(f, c, ca) < sfmax2 && mb_min3(r, g, ra) > sfmin2 {
                f = f * 2.0;
                c = c * 2.0;
                ca = ca * 2.0;
                r = r / 2.0;
                g = g / 2.0;
                ra = ra / 2.0;
            }
            g = c / 2.0;
            while g >= r && mb_max2(r, ra) < sfmax2 && mb_min4(f, c, g, ca) > sfmin2 {
                f = f / 2.0;
                c = c / 2.0;
                g = g / 2.0;
                ca = ca / 2.0;
                r = r * 2.0;
                ra = ra * 2.0;
            }
            if (c + r) >= 0.95 * s { i2 += 1; continue; }
            if f < 1.0 && ps[i2] < 1.0 {
                if f * ps[i2] <= sfmin1 { i2 += 1; continue; }
            }
            if f > 1.0 && ps[i2] > 1.0 {
                if ps[i2] >= sfmax1 / f { i2 += 1; continue; }
            }
            g = 1.0 / f;
            ps[i2] = ps[i2] * f;
            again = true;
            // The row i2 scales by one over f, and the column i2 scales
            // by f.
            var j: i64 = k;
            while j < n { m[i2 * n + j] = m[i2 * n + j] * g; j += 1; }
            var i3: i64 = 0;
            while i3 <= l { m[i3 * n + i2] = m[i3 * n + i2] * f; i3 += 1; }
            i2 += 1;
        }
    }
    return m, ps, k, l;
}

// mb_max3 gives the largest of three values.
fn mb_max3(a: f64, b: f64, c: f64) -> f64 {
    var m = a;
    if b > m { m = b; }
    if c > m { m = c; }
    return m;
}

// mb_max2 gives the largest of two values.
fn mb_max2(a: f64, b: f64) -> f64 {
    if b > a { return b; }
    return a;
}

// mb_min3 gives the smallest of three values.
fn mb_min3(a: f64, b: f64, c: f64) -> f64 {
    var m = a;
    if b < m { m = b; }
    if c < m { m = c; }
    return m;
}

// mb_min4 gives the smallest of four values.
fn mb_min4(a: f64, b: f64, c: f64, d: f64) -> f64 {
    var m = mb_min3(a, b, c);
    if d < m { m = d; }
    return m;
}

// mb_perm gives the scales and the permutation of the balance. The
// scales hold one value for each row of the matrix. The permutation
// holds the row of each scale.
fn mb_perm(ps: f64[:], k: i64, l: i64, n: i64) -> f64[>..], f64[>..] {
    var scaling: f64[>..] = [];
    var i: i64 = 0;
    while i < n { scaling.push(1.0); i += 1; }
    i = k;
    while i <= l { scaling[i] = ps[i]; i += 1; }
    var perm: f64[>..] = [];
    i = 0;
    while i < n { perm.push(i as f64); i += 1; }
    // The rows that move to the end.
    i = n - 1;
    while i > l {
        let p = ps[i] as i64;
        if p != i {
            let t = perm[p];
            perm[p] = perm[i];
            perm[i] = t;
        }
        i -= 1;
    }
    // The columns that move to the front.
    i = 0;
    while i < k {
        let p = ps[i] as i64;
        if p != i {
            let t = perm[p];
            perm[p] = perm[i];
            perm[i] = t;
        }
        i += 1;
    }
    return scaling, perm;
}

// matrix_balance balances a square matrix. The result holds the
// balanced matrix and the matrix of the change. The balanced matrix
// holds the same eigenvalues as the matrix of the caller.
fn matrix_balance(a) -> Arr, Arr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] || a.shape[0] < 1 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "matrix_balance", "the operand is not a square matrix");
    }
    let n = a.shape[0];
    if n > BAL_MAX_SIDE {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "matrix_balance", "the side is above the limit");
    }
    var i: i64 = 0;
    while i < a.data.len {
        if !(a.data[i] - a.data[i] == 0.0) {
            return empty_arr(), empty_arr(), fault(F_VALUE, "matrix_balance", "the matrix holds a value that is not finite");
        }
        i += 1;
    }
    let m, ps, k, l = mb_balance(a.data, n, true, true);
    let scaling, perm = mb_perm(ps, k, l, n);
    // The matrix of the change holds the scales on its diagonal.
    var t: f64[>..] = [];
    i = 0;
    while i < n * n { t.push(0.0); i += 1; }
    // The inverse of the permutation gives the row of each scale.
    var iperm: i64[>..] = [];
    i = 0;
    while i < n { iperm.push(0); i += 1; }
    i = 0;
    while i < n {
        let p = perm[i] as i64;
        iperm[p] = i;
        i += 1;
    }
    i = 0;
    while i < n {
        let j = iperm[i];
        t[i * n + j] = scaling[j];
        i += 1;
    }
    return mat2(copy(m), n, n), mat2(copy(t), n, n), no_fault();
}

// matrix_balance_separate balances a square matrix and gives the
// balanced matrix, the scales and the permutation apart. The scales
// hold the powers of two, and the permutation holds the row of each
// scale.
fn matrix_balance_separate(a) -> Arr, Arr, Arr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] || a.shape[0] < 1 {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "matrix_balance_separate", "the operand is not a square matrix");
    }
    let n = a.shape[0];
    if n > BAL_MAX_SIDE {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_LIMIT, "matrix_balance_separate", "the side is above the limit");
    }
    var i: i64 = 0;
    while i < a.data.len {
        if !(a.data[i] - a.data[i] == 0.0) {
            return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "matrix_balance_separate", "the matrix holds a value that is not finite");
        }
        i += 1;
    }
    let m, ps, k, l = mb_balance(a.data, n, true, true);
    let scaling, perm = mb_perm(ps, k, l, n);
    var sd: f64[>..] = [];
    i = 0;
    while i < n { sd.push(scaling[i]); i += 1; }
    return mat2(copy(m), n, n), line_arr(copy(sd)), line_arr(copy(perm)), no_fault();
}
