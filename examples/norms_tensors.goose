// The norms of a vector and of a matrix, the inverse of a tensor and
// the solution of a tensor equation.
import std;
import scientific from "..";

fn main() {
    // The norm of a vector takes a real order. The order two is the
    // default, and the order one sums the absolute values.
    let v, f1 = scientific::from_values([3.0, -4.0]);
    assert(f1.kind == scientific::F_NONE);
    let n2, f2 = scientific::vector_norm(v);
    assert(f2.kind == scientific::F_NONE);
    print("norm of [3, -4] in the order two:");
    print(n2.data[0]);
    print("\n");
    let n1, f3 = scientific::vector_norm(v, 1.0);
    assert(f3.kind == scientific::F_NONE);
    print("in the order one:");
    print(n1.data[0]);
    print("\n");

    // The norm of a matrix takes the name of an order. The order nuc
    // gives the sum of the singular values.
    let mv, f4 = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    assert(f4.kind == scientific::F_NONE);
    let m, f4b = scientific::reshape(mv, [2, 2]);
    assert(f4b.kind == scientific::F_NONE);
    let mn, f5 = scientific::matrix_norm(m, "nuc");
    assert(f5.kind == scientific::F_NONE);
    print("nuclear norm of [[1, 2], [3, 4]]:");
    print(mn.data[0]);
    print("\n");

    // The inverse of a tensor reads the buffer of the tensor as a
    // square matrix. The tensor holds four axes of two values, so the
    // matrix is four by four. The matrix holds two times the identity.
    let tv, f6 = scientific::from_values([2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
    assert(f6.kind == scientific::F_NONE);
    let t, f6b = scientific::reshape(tv, [2, 2, 2, 2]);
    assert(f6b.kind == scientific::F_NONE);
    let ti, f7 = scientific::tensorinv(t, 2);
    assert(f7.kind == scientific::F_NONE);
    print("inverse of the tensor:");
    var i: i64 = 0;
    while i < ti.data.len { print(" {} ", ti.data[i]); i += 1; }
    print("\n");

    // The solution of a tensor equation solves the system of the last
    // axes of the tensor.
    let sv, f8 = scientific::from_values([2.0, 0.0, 0.0, 4.0]);
    assert(f8.kind == scientific::F_NONE);
    let s, f8b = scientific::reshape(sv, [2, 2]);
    assert(f8b.kind == scientific::F_NONE);
    let b, f9 = scientific::from_values([2.0, 8.0]);
    assert(f9.kind == scientific::F_NONE);
    let x, f10 = scientific::tensorsolve(s, b);
    assert(f10.kind == scientific::F_NONE);
    print("solution of the tensor equation:");
    var j: i64 = 0;
    while j < x.data.len { print(" {} ", x.data[j]); j += 1; }
    print("\n");
}
