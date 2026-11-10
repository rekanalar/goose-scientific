// The solving of a system of linear equations.
//
// The method is a Gaussian elimination with a partial pivot. The pivot
// row is the row with the largest value in the column.
namespace scientific;

// lu_core splits a square matrix into an upper and a lower part.
// The pivot row of each column is in pivots. A zero pivot is a fault.
fn lu_core(a, m: i64, n: i64, lower: f64[>..]&, upper: f64[>..]&, pivots: i64[>..]&, sign: i64&) -> Fault {
    var i: i64 = 0;
    while i < m * n { lower.push(0.0); i += 1; }
    i = 0;
    while i < m * n { upper.push(0.0); i += 1; }
    i = 0;
    while i < m * n { lower[i] = a.data[i]; i += 1; }
    var perm: i64[>..] = [];
    i = 0;
    while i < m { perm.push(i); i += 1; }
    var s: i64 = 1;
    var col: i64 = 0;
    while col < n {
        // The row with the largest value in the column becomes the
        // pivot row.
        var best = col;
        var best_val = mathshim::root(lower[perm[col] * n + col] * lower[perm[col] * n + col]);
        var r: i64 = col + 1;
        while r < m {
            let v = lower[perm[r] * n + col];
            let a = mathshim::root(v * v);
            if a > best_val { best = r; best_val = a; }
            r += 1;
        }
        if best_val == 0.0 {
            return fault(F_VALUE, "lu", "the matrix holds no pivot");
        }
        if best != col {
            // The rows trade their places.
            var t: i64 = perm[col];
            perm[col] = perm[best];
            perm[best] = t;
            s = -s;
        }
        // The row below the pivot takes its part.
        r = col + 1;
        while r < m {
            let f = lower[perm[r] * n + col] / lower[perm[col] * n + col];
            lower[perm[r] * n + col] = f;
            var c: i64 = col + 1;
            while c < n {
                lower[perm[r] * n + c] = lower[perm[r] * n + c] - f * lower[perm[col] * n + c];
                c += 1;
            }
            r += 1;
        }
        col += 1;
    }
    // The upper part holds the result of the elimination.
    i = 0;
    while i < m {
        var c: i64 = 0;
        while c < n {
            upper[i * n + c] = lower[perm[i] * n + c];
            c += 1;
        }
        i += 1;
    }
    i = 0;
    while i < m { pivots.push(perm[i]); i += 1; }
    sign = s;
    return no_fault();
}

// solve gives the vector or the matrix that solves a system of linear
// equations. The right side holds one column or a matrix.
fn solve(a, b) -> Arr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return empty_arr(), fault(F_SHAPE, "solve", "the matrix is not square");
    }
    let m = a.shape[0];
    let n = a.shape[1];
    if b.ndim == 2 && b.shape[0] == n {
        let f, piv, g = lu_factor(a);
        if g.kind != F_NONE { return empty_arr(), g; }
        return lu_solve(f, piv, b, false);
    }
    if b.ndim != 1 || b.data.len != n {
        return empty_arr(), fault(F_SHAPE, "solve", "the right side does not agree with the matrix");
    }
    var lower: f64[>..] = [];
    var upper: f64[>..] = [];
    var pivots: i64[>..] = [];
    var sign: i64 = 1;
    let f = lu_core(a, m, n, lower, upper, pivots, sign);
    if f.kind != F_NONE { return empty_arr(), f; }
    // The right side follows the trade of the rows.
    var y: f64[>..] = [];
    var i: i64 = 0;
    while i < m { y.push(b.data[pivots[i]]); i += 1; }
    // The forward substitution.
    i = 1;
    while i < m {
        var c: i64 = 0;
        while c < i {
            y[i] = y[i] - lower[i * n + c] * y[c];
            c += 1;
        }
        i += 1;
    }
    // The back substitution.
    i = m - 1;
    while i >= 0 {
        var c: i64 = i + 1;
        while c < m {
            y[i] = y[i] - upper[i * n + c] * y[c];
            c += 1;
        }
        y[i] = y[i] / upper[i * n + i];
        i -= 1;
    }
    var s: i64[..8] = [];
    s.push(m);
    return Arr { ndim: 1, shape: s, data: y }, no_fault();
}

// inv gives the inverse of a matrix.
fn inv(a) -> Arr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return empty_arr(), fault(F_SHAPE, "inv", "the matrix is not square");
    }
    let m = a.shape[0];
    let n = a.shape[1];
    // The inverse holds the columns of the solution of the systems
    // with the unit vectors.
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < m * n { d.push(0.0); i += 1; }
    var col: i64 = 0;
    while col < n {
        var b: f64[>..] = [];
        i = 0;
        while i < m {
            if i == col { b.push(1.0); } else { b.push(0.0); }
            i += 1;
        }
        let x, f = solve(a, Arr { ndim: 1, shape: [], data: copy(b) });
        if f.kind != F_NONE { return empty_arr(), f; }
        i = 0;
        while i < m {
            d[i * n + col] = x.data[i];
            i += 1;
        }
        col += 1;
    }
    var s: i64[..8] = [];
    s.push(m);
    s.push(n);
    return Arr { ndim: 2, shape: s, data: d }, no_fault();
}

// det gives the determinant of a matrix.
fn det(a) -> f64, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return 0.0, fault(F_SHAPE, "det", "the matrix is not square");
    }
    let m = a.shape[0];
    let n = a.shape[1];
    var lower: f64[>..] = [];
    var upper: f64[>..] = [];
    var pivots: i64[>..] = [];
    var sign: i64 = 1;
    let f = lu_core(a, m, n, lower, upper, pivots, sign);
    if f.kind != F_NONE { return 0.0, f; }
    var t: f64 = sign as f64;
    var i: i64 = 0;
    while i < m {
        t *= upper[i * n + i];
        i += 1;
    }
    return t, no_fault();
}

// slogdet gives the sign and the logarithm of the determinant. The
// sign is zero when the determinant is zero.
fn slogdet(a) -> f64, f64, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return 0.0, 0.0, fault(F_SHAPE, "slogdet", "the matrix is not square");
    }
    let m = a.shape[0];
    let n = a.shape[1];
    var lower: f64[>..] = [];
    var upper: f64[>..] = [];
    var pivots: i64[>..] = [];
    var sign: i64 = 1;
    let f = lu_core(a, m, n, lower, upper, pivots, sign);
    if f.kind != F_NONE {
        // A matrix without a pivot has a determinant of zero.
        return 0.0, 0.0, no_fault();
    }
    var lg: f64 = 0.0;
    var sgn: f64 = 1.0;
    var i: i64 = 0;
    while i < m {
        let d = upper[i * n + i];
        if d < 0.0 { sgn = -sgn; }
        let ad = d;
        if ad < 0.0 { lg += mathshim::ln(-ad); } else { lg += mathshim::ln(ad); }
        i += 1;
    }
    if sign < 0 { sgn = -sgn; }
    return sgn, lg, no_fault();
}

// matrix_power raises a matrix to a whole power. A power below zero
// gives the inverse of the matrix.
fn matrix_power(a, power: i64) -> Arr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return empty_arr(), fault(F_SHAPE, "matrix_power", "the matrix is not square");
    }
    if power < 0 {
        let invm, f = inv(a);
        if f.kind != F_NONE { return empty_arr(), f; }
        return positive_power(invm, -power);
    }
    return positive_power(a, power);
}

// positive_power raises a matrix to a power above zero. The power is
// found by repeated squaring.
fn positive_power(a, power: i64) -> Arr, Fault {
    let n = a.shape[0];
    var s: i64[..8] = [];
    s.push(n);
    s.push(n);
    if power == 0 {
        var d: f64[>..] = [];
        var i: i64 = 0;
        while i < n * n { d.push(0.0); i += 1; }
        i = 0;
        while i < n { d[i * n + i] = 1.0; i += 1; }
        return Arr { ndim: 2, shape: s, data: d }, no_fault();
    }
    var result: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { result.push(0.0); i += 1; }
    i = 0;
    while i < n { result[i * n + i] = 1.0; i += 1; }
    var base = copy(a.data);
    var k = power;
    while k > 0 {
        if k % 2 == 1 {
            let r = Arr { ndim: 2, shape: s, data: copy(result) };
            let p, f = matmul(r, Arr { ndim: 2, shape: s, data: copy(base) });
            if f.kind != F_NONE { return empty_arr(), f; }
            result = copy(p.data);
        }
        let b = Arr { ndim: 2, shape: s, data: copy(base) };
        let p2, f2 = matmul(b, b);
        if f2.kind != F_NONE { return empty_arr(), f2; }
        base = copy(p2.data);
        k /= 2;
    }
    return Arr { ndim: 2, shape: s, data: result }, no_fault();
}
