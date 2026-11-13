// The extreme eigenpairs of a sparse symmetric matrix with lobpcg.
import scientific from "..";

fn main() {
    // The chain matrix of side 100: 2 on the diagonal, -1 beside it.
    let n: i64 = 100;
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var va: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        if i > 0 { ri.push(i as f64); ci.push((i - 1) as f64); va.push(-1.0); }
        ri.push(i as f64); ci.push(i as f64); va.push(2.0);
        if i < n - 1 { ri.push(i as f64); ci.push((i + 1) as f64); va.push(-1.0); }
        i += 1;
    }
    let a, fa = scientific::csr_from_triplets(n, n, ri, ci, va);
    assert(fa.kind == scientific::F_NONE);

    // The start block holds three columns of a simple generator.
    var x: f64[>..] = [];
    var st: i64 = 48271;
    i = 0;
    while i < n * 3 {
        st = (st * 48271) % 2147483647;
        x.push(((st % 1000) as f64) / 500.0 - 1.0);
        i += 1;
    }

    // The call gives the three smallest values and their vectors. The
    // values rise. The tolerance is the residual norm of each pair.
    let w, v, it, lh, rh, f = scientific::lobpcg(a, scientific::mat2(copy(x), n, 3), scientific::csr_empty(),
        scientific::csr_empty(), scientific::empty_arr(), 1.0e-8, 500, false);
    assert(f.kind == scientific::F_NONE);
    print("smallest: ", w.data, "\n");
    print("iterations: ", it, "\n");

    // The call gives the three largest values when largest is true. They fall.
    let w2, v2, it2, lh2, rh2, f2 = scientific::lobpcg(a, scientific::mat2(copy(x), n, 3), scientific::csr_empty(),
        scientific::csr_empty(), scientific::empty_arr(), 1.0e-8, 500, true);
    assert(f2.kind == scientific::F_NONE);
    print("largest: ", w2.data, "\n");
}
