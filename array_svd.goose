// The singular values, the least squares and the rank.
//
// The singular values come from the one-sided Jacobi method. The
// method works on the columns of the matrix.
namespace scientific;

import mathshim;

// jacobi_svd finds the singular values and the right vectors of a
// matrix. The right vectors are the columns of the second result. The
// singular values come in the order of their size, from the largest to
// the smallest.
fn jacobi_svd(a, m: i64, n: i64) -> Arr, Arr, Fault {
    // The columns of the matrix. The buffer holds the columns one
    // after the other, so the column c lies at the index of the row
    // times the count of the columns plus the column.
    var w: f64[>..] = [];
    var i: i64 = 0;
    while i < n * m { w.push(0.0); i += 1; }
    var r0: i64 = 0;
    while r0 < m {
        var c0: i64 = 0;
        while c0 < n {
            w[c0 * m + r0] = a.data[r0 * n + c0];
            c0 += 1;
        }
        r0 += 1;
    }
    // The right vectors. They start as the unit matrix.
    var v: f64[>..] = [];
    i = 0;
    while i < n * n { v.push(0.0); i += 1; }
    i = 0;
    while i < n { v[i * n + i] = 1.0; i += 1; }
    // The turn gives the two columns the same length.
    var sweep: i64 = 0;
    while sweep < 60 {
        var turned = false;
        var p: i64 = 0;
        while p < n - 1 {
            var q: i64 = p + 1;
            while q < n {
                // The inner product of the two columns and their
                // lengths.
                var alpha: f64 = 0.0;
                var beta: f64 = 0.0;
                var gamma: f64 = 0.0;
                var r: i64 = 0;
                while r < m {
                    alpha += w[p * m + r] * w[p * m + r];
                    beta += w[q * m + r] * w[q * m + r];
                    gamma += w[p * m + r] * w[q * m + r];
                    r += 1;
                }
                var gabs = gamma;
                if gabs < 0.0 { gabs = -gabs; }
                if gabs > 1.0e-16 * mathshim::root(alpha * beta) && gamma != 0.0 {
                    // The turn runs when the two columns hold a real
                    // part of their product. The angle comes from
                    // the two lengths and their product.
                    let diff = alpha - beta;
                    let root = mathshim::root(diff * diff + 4.0 * gamma * gamma);
                    // Two columns that shrank to the size of the
                    // underflow give a root of zero. The turn would
                    // divide zero by zero, so those columns stay.
                    if root == 0.0 { q += 1; continue; }
                    var t: f64 = 0.0;
                    // The two forms are equal by algebra. These ones
                    // hold no subtraction of two close numbers.
                    if diff >= 0.0 { t = -2.0 * gamma / (diff + root); }
                    else { t = 2.0 * gamma / (root - diff); }
                    let c = 1.0 / mathshim::root(t * t + 1.0);
                    let sn = t * c;
                    // The turn mixes the two columns.
                    r = 0;
                    while r < m {
                        let wp = w[p * m + r];
                        let wq = w[q * m + r];
                        w[p * m + r] = c * wp - sn * wq;
                        w[q * m + r] = sn * wp + c * wq;
                        r += 1;
                    }
                    // The right vectors follow the turn.
                    r = 0;
                    while r < n {
                        let vp = v[p * n + r];
                        let vq = v[q * n + r];
                        v[p * n + r] = c * vp - sn * vq;
                        v[q * n + r] = sn * vp + c * vq;
                        r += 1;
                    }
                    turned = true;
                }
                q += 1;
            }
            p += 1;
        }
        sweep += 1;
        if !turned { break; }
    }
    // The singular values are the lengths of the columns. They come in
    // the order of their size, from the largest to the smallest.
    var order: i64[>..] = [];
    var len2: f64[>..] = [];
    i = 0;
    while i < n {
        order.push(i);
        var sum: f64 = 0.0;
        var rr: i64 = 0;
        while rr < m {
            sum += w[i * m + rr] * w[i * m + rr];
            rr += 1;
        }
        len2.push(sum);
        i += 1;
    }
    // The sort takes the whole length of each column, not one entry.
    i = 1;
    while i < n {
        let key = order[i];
        var j = i - 1;
        while j >= 0 {
            let other = order[j];
            if len2[other] < len2[key] {
                order[j + 1] = other;
                j -= 1;
            } else { break; }
        }
        order[j + 1] = key;
        i += 1;
    }
    var vals: f64[>..] = [];
    var vecs: f64[>..] = [];
    i = 0;
    while i < n {
        let col = order[i];
        var length: f64 = 0.0;
        var r: i64 = 0;
        while r < m {
            length += w[col * m + r] * w[col * m + r];
            r += 1;
        }
        vals.push(mathshim::root(length));
        var c: i64 = 0;
        while c < n {
            vecs.push(v[col * n + c]);
            c += 1;
        }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    var sv: i64[..8] = [];
    sv.push(n);
    sv.push(n);
    return Arr { ndim: 1, shape: s, data: vals }, Arr { ndim: 2, shape: sv, data: vecs }, no_fault();
}

// complete_basis gives the columns that a set of columns lacks. The
// result holds the columns of the matrix and the new ones. The matrix
// holds the columns in its rows, so the column c lies at the index of
// the row times the count plus the column.
fn complete_basis(cols: f64[:], m: i64, count: i64, want: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < m * want { out.push(0.0); i += 1; }
    i = 0;
    while i < count {
        var r: i64 = 0;
        while r < m {
            out[r * want + i] = cols[r * m + i];
            r += 1;
        }
        i += 1;
    }
    var c = count;
    while c < want {
        // The new column comes from the unit vector that no column
        // holds. The part of a unit vector that the columns hold has
        // the squared length of the sum of the squares of its entries.
        // One pass over the columns gives that length for every unit
        // vector. Without the pass, a search for a vector that the
        // columns do not hold costs the square of the count of the
        // columns for each candidate.
        var k: i64 = 0;
        var found = false;
        while k < m && !found {
            var held: f64 = 0.0;
            var j: i64 = 0;
            while j < c {
                let t = out[k * want + j];
                held += t * t;
                j += 1;
            }
            if 1.0 - held > 1.0e-12 {
                var v: f64[>..] = [];
                var r: i64 = 0;
                while r < m {
                    if r == k { v.push(1.0); } else { v.push(0.0); }
                    r += 1;
                }
                // The vector takes the part that no column holds.
                j = 0;
                while j < c {
                    var t: f64 = 0.0;
                    r = 0;
                    while r < m {
                        t += v[r] * out[r * want + j];
                        r += 1;
                    }
                    r = 0;
                    while r < m {
                        v[r] = v[r] - t * out[r * want + j];
                        r += 1;
                    }
                    j += 1;
                }
                // The vector must hold a length.
                var length: f64 = 0.0;
                r = 0;
                while r < m {
                    length += v[r] * v[r];
                    r += 1;
                }
                length = mathshim::root(length);
                if length > 0.000001 {
                    r = 0;
                    while r < m {
                        out[r * want + c] = v[r] / length;
                        r += 1;
                    }
                    found = true;
                }
            }
            k += 1;
        }
        c += 1;
    }
    return out;
}

// svd gives the left vectors, the singular values and the right
// vectors of a matrix. The left vectors hold as many rows and columns
// as the matrix has rows. The right vectors hold as many rows and
// columns as the matrix has columns. The right vectors are the
// transpose of the vectors, as NumPy gives them.
fn svd(a) -> Arr, Arr, Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "svd", "the operand holds a count of dimensions that is not two");
    }
    let m = a.shape[0];
    let n = a.shape[1];
    let vals, vecs, f = jacobi_svd(a, m, n);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
    // The count of the singular values.
    var k = m;
    if n < k { k = n; }
    // The count of the singular values that hold a value above the
    // tolerance. The others hold no vector.
    var tol: f64 = 0.0;
    if k > 0 { tol = vals.data[0] * 0.000001; }
    var valid: i64 = 0;
    while valid < k && vals.data[valid] > tol { valid += 1; }
    // The right vectors hold their columns.
    var right: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { right.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var c: i64 = 0;
        while c < valid {
            right[i * n + c] = vecs.data[c * n + i];
            c += 1;
        }
        i += 1;
    }
    // The right vectors of the singular values that hold no value come
    // from the unit vectors that no other column holds.
    if n > valid {
        let full_right = complete_basis(right, n, valid, n);
        right = copy(full_right);
    }
    // The left vectors come from the product of the matrix and the
    // right vectors, divided by the singular values.
    var left: f64[>..] = [];
    i = 0;
    while i < m * m { left.push(0.0); i += 1; }
    var c: i64 = 0;
    while c < valid {
        let sv = vals.data[c];
        var r: i64 = 0;
        while r < m {
            var t: f64 = 0.0;
            var q: i64 = 0;
            while q < n {
                t += a.data[r * n + q] * right[q * n + c];
                q += 1;
            }
            left[r * m + c] = t / sv;
            r += 1;
        }
        c += 1;
    }
    if m > valid {
        let full_left = complete_basis(left, m, valid, m);
        left = copy(full_left);
    }
    // The singular values hold only the count of the smaller
    // dimension.
    var out_vals: f64[>..] = [];
    i = 0;
    while i < k { out_vals.push(vals.data[i]); i += 1; }
    // The right vectors come as the transpose of the vectors.
    var transposed: f64[>..] = [];
    i = 0;
    while i < n * n { transposed.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            transposed[i * n + j] = right[j * n + i];
            j += 1;
        }
        i += 1;
    }
    var su: i64[..8] = [];
    su.push(m);
    su.push(m);
    var ss: i64[..8] = [];
    ss.push(k);
    var sv: i64[..8] = [];
    sv.push(n);
    sv.push(n);
    return Arr { ndim: 2, shape: su, data: left }, Arr { ndim: 1, shape: ss, data: out_vals }, Arr { ndim: 2, shape: sv, data: transposed }, no_fault();
}

// svdvals gives the singular values of a matrix.
fn svdvals(a) -> Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "svdvals", "the operand holds a count of dimensions that is not two");
    }
    let m = a.shape[0];
    let n = a.shape[1];
    let vals, vecs, f = jacobi_svd(a, m, n);
    if f.kind != F_NONE { return empty_arr(), f; }
    // The values hold only the count of the smaller dimension.
    var k = m;
    if n < k { k = n; }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < k { out.push(vals.data[i]); i += 1; }
    var s: i64[..8] = [];
    s.push(k);
    return Arr { ndim: 1, shape: s, data: out }, no_fault();
}

// lstsq gives the vector that fits the system of linear equations in
// the sense of the least squares.
fn lstsq(a, b) -> Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "lstsq", "the matrix holds a count of dimensions that is not two");
    }
    let m = a.shape[0];
    let n = a.shape[1];
    if b.ndim != 1 || b.data.len != m {
        return empty_arr(), fault(F_SHAPE, "lstsq", "the right side does not agree with the matrix");
    }
    let vals, vecs, f = jacobi_svd(a, m, n);
    if f.kind != F_NONE { return empty_arr(), f; }
    // The tolerance of the smallest singular value.
    var tol: f64 = 0.0;
    if vals.data.len > 0 { tol = vals.data[0] * 0.000001; }
    // The solution holds the sum over the values above the tolerance.
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n { d.push(0.0); i += 1; }
    var c: i64 = 0;
    while c < n {
        let sv = vals.data[c];
        if sv > tol {
            // The left vector comes from the matrix and the right
            // vector, divided by the value.
            var left: f64[>..] = [];
            var r: i64 = 0;
            while r < m {
                var t: f64 = 0.0;
                var q: i64 = 0;
                while q < n {
                    t += a.data[r * n + q] * vecs.data[c * n + q];
                    q += 1;
                }
                left.push(t / sv);
                r += 1;
            }
            // The product of the left vector and the right side.
            var t: f64 = 0.0;
            r = 0;
            while r < m {
                t += left[r] * b.data[r];
                r += 1;
            }
            let factor = t / sv;
            var k: i64 = 0;
            while k < n {
                d[k] += factor * vecs.data[c * n + k];
                k += 1;
            }
        }
        c += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// pinv gives the pseudo-inverse of a matrix. It sums the outer
// products of the right and left vectors, each divided by its
// singular value. A singular value at or under the cutoff counts as
// zero. The default cutoff is the larger side times the machine
// epsilon, times the largest singular value, as NumPy 2 gives it. A
// negative rtol takes that default.
fn pinv(a, rtol: f64 = -1.0) -> Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "pinv", "the operand holds a count of dimensions that is not two");
    }
    let m = a.shape[0];
    let n = a.shape[1];
    let u, sv, vt, f = svd(a);
    if f.kind != F_NONE { return empty_arr(), f; }
    var k = m;
    if n < k { k = n; }
    var cut = rtol;
    if cut < 0.0 {
        var big = m;
        if n > big { big = n; }
        cut = big as f64 * 2.220446049250313e-16;
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n * m { d.push(0.0); i += 1; }
    if k > 0 {
        let limit = cut * sv.data[0];
        var t: i64 = 0;
        while t < k {
            if sv.data[t] > limit {
                var r: i64 = 0;
                while r < n {
                    var c: i64 = 0;
                    while c < m {
                        d[r * m + c] += vt.data[t * n + r] * u.data[c * m + t] / sv.data[t];
                        c += 1;
                    }
                    r += 1;
                }
            }
            t += 1;
        }
    }
    var sh: i64[..8] = [];
    sh.push(n);
    sh.push(m);
    return Arr { ndim: 2, shape: sh, data: d }, no_fault();
}

// matrix_rank gives the count of the singular values that hold a
// value above the tolerance. The tolerance is the larger side times
// the machine epsilon times the largest singular value, as NumPy
// gives it.
fn matrix_rank(a) -> i64, Fault {
    let vals, f = svdvals(a);
    if f.kind != F_NONE { return 0, f; }
    if vals.data.len == 0 { return 0, no_fault(); }
    var big = a.shape[0];
    if a.shape[1] > big { big = a.shape[1]; }
    let tol = vals.data[0] * big as f64 * 2.220446049250313e-16;
    var count: i64 = 0;
    var i: i64 = 0;
    while i < vals.data.len {
        if vals.data[i] > tol { count += 1; }
        i += 1;
    }
    return count, no_fault();
}

// cond gives the ratio of the largest singular value to the smallest
// one.
fn cond(a) -> f64, Fault {
    let vals, f = svdvals(a);
    if f.kind != F_NONE { return 0.0, f; }
    if vals.data.len == 0 { return 0.0, fault(F_VALUE, "cond", "the matrix holds no value"); }
    let smallest = vals.data[vals.data.len - 1];
    if smallest == 0.0 { return 0.0, fault(F_VALUE, "cond", "the matrix holds a zero value"); }
    return vals.data[0] / smallest, no_fault();
}
