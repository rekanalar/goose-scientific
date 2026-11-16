// The rational approximation of scipy.interpolate.AAA as the call
// aaa_interpolate. The algorithm is the greedy method of Nakatsukasa,
// Sete and Trefethen 2018 in the form that SciPy 1.18.1 uses: the
// greedy choice of the support points, the weights of the least
// squares problem through the singular vectors, and the barycentric
// evaluation. The values of the package are real, so the weights stay
// real.
namespace scientific;

import mathshim;

// The count of terms that aaa_interpolate takes. The buffers of the
// iteration hold the square of it times the count of the sample.
const MAX_AAA_TERMS: i64 = 1000;

// aaa_interpolate follows scipy.interpolate.AAA: the rational
// approximation of the values y at the sample x, evaluated at the
// points xs. The sample holds distinct finite values after the call
// drops the samples whose value is not finite. The rtol below zero
// takes the default of the reference, which is the power of 0.75 of
// the machine epsilon. The max_terms of -1 takes the default of 100.
// The call holds no cleanup of the Froissart doublets, as the
// reference holds.
fn aaa_interpolate(x: f64[:], y: f64[:], xs: f64[:], rtol: f64 = -1.0,
                   max_terms: i64 = 100) -> Arr, Fault {
    if x.len != y.len || x.len < 1 {
        return empty_arr(), fault(F_SHAPE, "aaa_interpolate", "the sample and the values need equal counts of at least one");
    }
    if xs.len < 1 {
        return empty_arr(), fault(F_SHAPE, "aaa_interpolate", "the list of points holds no value");
    }
    if max_terms < 1 {
        return empty_arr(), fault(F_RANGE, "aaa_interpolate", "the count of terms stands below one");
    }
    if max_terms > MAX_AAA_TERMS {
        return empty_arr(), fault(F_LIMIT, "aaa_interpolate", "the count of terms stands above the limit");
    }
    return aaa_run(x, y, xs, rtol, max_terms);
}

// aaa_run holds the work of aaa_interpolate after the check of the
// count of terms.
fn aaa_run(x: f64[:], y: f64[:], xs: f64[:], rtol: f64, max_terms: i64) -> Arr, Fault {
    var i: i64 = 0;
    while i < x.len {
        if !(x[i] - x[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "aaa_interpolate", "a sample is not finite");
        }
        i += 1;
    }
    // The samples whose value is not finite drop, as the reference
    // drops them. A sample that repeats an earlier sample keeps its
    // first place only. The sort of the sample holds the samples in
    // the rising order of the values.
    var ord = dl_sort_order(x);
    var zu: f64[>..] = [];
    var fu: f64[>..] = [];
    i = 0;
    while i < ord.len {
        let k = ord[i];
        if !(y[k] - y[k] == 0.0) {
            i += 1;
            continue;
        }
        let v = x[k];
        if zu.len > 0 && zu[zu.len - 1] == v {
            i += 1;
            continue;
        }
        zu.push(v);
        fu.push(y[k]);
        i += 1;
    }
    if zu.len < 1 {
        return empty_arr(), fault(F_VALUE, "aaa_interpolate", "the values hold no finite entry");
    }
    let m = zu.len;
    // The count of the sample bounds the count of terms, for the
    // greedy choice needs a point that no term holds.
    var terms = max_terms;
    if terms > m { terms = m; }
    var tol = rtol;
    if tol < 0.0 {
        tol = mathshim::power(2.220446049250313e-16, 0.75);
    }
    if !(tol >= 0.0) || !(tol - tol == 0.0) {
        return empty_arr(), fault(F_VALUE, "aaa_interpolate", "the tolerance is not a positive width");
    }
    var fmax: f64 = 0.0;
    i = 0;
    while i < m {
        var a = fu[i];
        if a < 0.0 { a = -a; }
        if a > fmax { fmax = a; }
        i += 1;
    }
    let atol = tol * fmax;
    // The mean of the values is the first approximant.
    var mean: f64 = 0.0;
    i = 0;
    while i < m { mean += fu[i]; i += 1; }
    mean /= m;
    // The columns of the Cauchy matrix and of the Loewner matrix.
    var cmat: f64[>..] = [];
    var amat: f64[>..] = [];
    i = 0;
    while i < m * terms { cmat.push(0.0); i += 1; }
    i = 0;
    while i < m * terms { amat.push(0.0); i += 1; }
    var zj: f64[>..] = [];
    var fj: f64[>..] = [];
    var wj: f64[>..] = [];
    var used: i64[>..] = [];
    i = 0;
    while i < m { used.push(1); i += 1; }
    var r: f64[>..] = [];
    i = 0;
    while i < m { r.push(mean); i += 1; }
    var ill: bool = false;
    let ill_tol = 1.0 / (3.0 * 2.220446049250313e-16);
    var converged = false;
    var term: i64 = 0;
    while term < terms && !converged {
        // The greedy choice takes the sample with the greatest error.
        var sel: i64 = -1;
        var best: f64 = -1.0;
        i = 0;
        while i < m {
            if used[i] == 1 {
                var df = fu[i] - r[i];
                if df < 0.0 { df = -df; }
                if df > best { best = df; sel = i; }
            }
            i += 1;
        }
        if sel < 0 {
            return empty_arr(), fault(F_LIMIT, "aaa_interpolate", "the sample holds no point left before the tolerance");
        }
        zj.push(zu[sel]);
        fj.push(fu[sel]);
        i = 0;
        while i < m {
            cmat[i * terms + term] = 1.0 / (zu[i] - zu[sel]);
            amat[i * terms + term] = (fu[i] - fu[sel]) * cmat[i * terms + term];
            i += 1;
        }
        used[sel] = 0;
        // The rows of the Loewner matrix that no term holds.
        var rows: i64 = 0;
        i = 0;
        while i < m {
            if used[i] == 1 { rows += 1; }
            i += 1;
        }
        var cols = term + 1;
        if rows == 0 {
            // The sample holds no point left. The last support point
            // drops, as no weight belongs to it.
            zj.pop();
            fj.pop();
            break;
        }
        var a_sub: f64[>..] = [];
        i = 0;
        while i < m {
            if used[i] == 1 {
                var j: i64 = 0;
                while j < cols {
                    a_sub.push(amat[i * terms + j]);
                    j += 1;
                }
            }
            i += 1;
        }
        var wnew: f64[>..] = [];
        var sa: i64[..8] = [];
        sa.push(rows);
        sa.push(cols);
        let vals, vecs, fs = jacobi_svd(Arr { ndim: 2, shape: sa, data: copy(a_sub) }, rows, cols);
        if fs.kind != F_NONE {
            return empty_arr(), fault(F_VALUE, "aaa_interpolate", "the weights hold no solution");
        }
        if rows >= cols {
            // The usual tall case. The weights follow the right
            // singular vectors of the least singular values.
            var ratio: f64 = 0.0;
            if vals.data[cols - 1] == 0.0 {
                ratio = ill_tol * 2.0;
            } else {
                ratio = vals.data[0] / vals.data[cols - 1];
            }
            var norm: f64[>..] = [];
            if !ill && ratio > ill_tol { ill = true; }
            if ill {
                // The columns of a wide spread lose their digits. The
                // reference scales every column to one before the
                // singular vectors, and scales the weights back.
                var sub = copy(a_sub);
                var j: i64 = 0;
                while j < cols {
                    var t: f64 = 0.0;
                    i = 0;
                    while i < rows {
                        let v = a_sub[i * cols + j];
                        t += v * v;
                        i += 1;
                    }
                    var nc = mathshim::root(t);
                    if nc == 0.0 { nc = 1.0; }
                    norm.push(nc);
                    i = 0;
                    while i < rows {
                        sub[i * cols + j] = a_sub[i * cols + j] / nc;
                        i += 1;
                    }
                    j += 1;
                }
                let vals2, vecs2, fs2 = jacobi_svd(Arr { ndim: 2, shape: sa, data: copy(sub) }, rows, cols);
                if fs2.kind != F_NONE {
                    return empty_arr(), fault(F_VALUE, "aaa_interpolate", "the weights hold no solution");
                }
                wnew = aaa_min_weights(vecs2, vals2, cols);
                var j2: i64 = 0;
                while j2 < cols { wnew[j2] /= norm[j2]; j2 += 1; }
            } else {
                wnew = aaa_min_weights(vecs, vals, cols);
            }
        } else {
            // Fewer rows than columns. The weights span the null
            // space, which the last right singular vectors hold.
            var rank: i64 = 0;
            var top = vals.data[0];
            i = 0;
            while i < cols {
                if top > 0.0 && vals.data[i] > top * 1.0e-12 { rank += 1; }
                i += 1;
            }
            var nm = cols - rank;
            if nm < 1 { nm = 1; }
            i = 0;
            while i < cols { wnew.push(0.0); i += 1; }
            let rootn = mathshim::root(nm);
            var j = cols - nm;
            if j < 0 { j = 0; }
            while j < cols {
                i = 0;
                while i < cols { wnew[i] += vecs.data[j * cols + i]; i += 1; }
                j += 1;
            }
            i = 0;
            while i < cols { wnew[i] /= rootn; i += 1; }
        }
        wj = copy(wnew);
        // The approximant holds the value at every support point whose
        // weight holds a value.
        i = 0;
        while i < m {
            var at_support = false;
            var j: i64 = 0;
            while j <= term {
                if wj[j] != 0.0 && zu[i] == zj[j] { at_support = true; }
                j += 1;
            }
            if at_support {
                r[i] = fu[i];
            } else {
                var num: f64 = 0.0;
                var den: f64 = 0.0;
                j = 0;
                while j <= term {
                    if wj[j] != 0.0 {
                        let cq = cmat[i * terms + j];
                        num += cq * wj[j] * fj[j];
                        den += cq * wj[j];
                    }
                    j += 1;
                }
                r[i] = num / den;
            }
            i += 1;
        }
        var err: f64 = 0.0;
        i = 0;
        while i < m {
            var df = fu[i] - r[i];
            if df < 0.0 { df = -df; }
            if df > err { err = df; }
            i += 1;
        }
        if err <= atol { converged = true; }
        term += 1;
    }
    // The support points whose weight holds no value drop.
    var wz: f64[>..] = [];
    var wv: f64[>..] = [];
    var ww: f64[>..] = [];
    i = 0;
    while i < zj.len {
        if wj[i] != 0.0 {
            wz.push(zj[i]);
            wv.push(fj[i]);
            ww.push(wj[i]);
        }
        i += 1;
    }
    // The evaluation in the form of Barycentric. A point of the sample
    // holds its value, and a point of no width holds the mean of the
    // weighted values.
    var out: f64[>..] = [];
    i = 0;
    while i < xs.len {
        let q = xs[i];
        if !(q - q == 0.0) && q > 1.0e308 {
            var num: f64 = 0.0;
            var den: f64 = 0.0;
            var j: i64 = 0;
            while j < wz.len {
                num += ww[j] * wv[j];
                den += ww[j];
                j += 1;
            }
            out.push(num / den);
            i += 1;
            continue;
        }
        out.push(barycentric_at(ww, wz, wv, q));
        i += 1;
    }
    var sh: i64[..8] = [];
    sh.push(xs.len);
    return Arr { ndim: 1, shape: sh, data: out }, no_fault();
}

// aaa_min_weights sums the right singular vectors of the least
// singular values, and divides by the root of their count. The buffer
// vh holds the columns of the right vectors, so the entry i of the
// vector j sits at j times cols plus i. The buffer s holds the
// singular values that fall.
fn aaa_min_weights(vh, s, cols: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < cols { out.push(0.0); i += 1; }
    let minv = s.data[s.data.len - 1];
    var count: i64 = 0;
    var j: i64 = 0;
    while j < s.data.len {
        if s.data[j] == minv { count += 1; }
        j += 1;
    }
    let rootn = mathshim::root(count as f64);
    j = cols - count;
    if j < 0 { j = 0; }
    while j < cols {
        i = 0;
        while i < cols { out[i] += vh.data[j * cols + i]; i += 1; }
        j += 1;
    }
    i = 0;
    while i < cols { out[i] /= rootn; i += 1; }
    return out;
}
