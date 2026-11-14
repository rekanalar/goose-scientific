// The product of a chain of matrices, the call over a list of axes and
// the call over the values.
import std;
import scientific from "..";

fn main() {
    // The chain holds the arrays one by one. The first array of one
    // dimension reads as a row. The last one reads as a column.
    let a, f1 = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let m, f2 = scientific::reshape(a, [2, 3]);
    let b, f3 = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let n, f4 = scientific::reshape(b, [3, 2]);
    var c = scientific::chain_new();
    let c1, f5 = scientific::chain_add(c, copy(m));
    let c2, f6 = scientific::chain_add(c1, copy(n));
    let r, f7 = scientific::multi_dot(c2);
    assert(f7.kind == scientific::F_NONE);
    print("product of a matrix and a matrix:");
    var i: i64 = 0;
    while i < r.data.len { print(" {} ", r.data[i]); i += 1; }
    print("\n");

    // The call over the axes takes a block. The block reduces one axis.
    let t, f8 = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let s, f9 = scientific::reshape(t, [2, 2, 3]);
    let q, f10 = scientific::apply_over_axes(s, [0, 2]) { x, ax =>
        let r, f = scientific::sum_axis(x, ax);
        r
    };
    assert(f10.kind == scientific::F_NONE);
    print("sum over the axes zero and two:");
    print(" shape ");
    i = 0;
    while i < q.shape.len { print(" {} ", q.shape[i]); i += 1; }
    print(" values ");
    i = 0;
    while i < q.data.len { print(" {} ", q.data[i]); i += 1; }
    print("\n");

    // The call over the values takes the values of two arrays. The
    // arrays broadcast to one shape.
    let u, f11 = scientific::from_values([1.0, 2.0, 3.0]);
    let v, f12 = scientific::from_values([10.0, 20.0]);
    let w, f13 = scientific::reshape(v, [2, 1]);
    let z, f14 = scientific::vectorize2(copy(u), copy(w)) { x, y => x * y };
    assert(f14.kind == scientific::F_NONE);
    print("the product of a row and a column:");
    print(" shape ");
    i = 0;
    while i < z.shape.len { print(" {} ", z.shape[i]); i += 1; }
    print(" values ");
    i = 0;
    while i < z.data.len { print(" {} ", z.data[i]); i += 1; }
    print("\n");
}
