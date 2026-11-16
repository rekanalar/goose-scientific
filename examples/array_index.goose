// The index helpers: the places of a triangle, the walk over an array and
// the path of an einsum. Run it with: goose run examples/array_index.goose
import std;
import scientific from "..";

fn main() {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 6 { d.push(i); i += 1; }
    var s: i64[..8] = [];
    s.push(2);
    s.push(3);
    let a = scientific::Arr { ndim: 2, shape: s, data: copy(d) };
    let low, f1 = scientific::tril_indices_from(a);
    print("lower triangle rows then columns: ", low.data, "\n");
    let up, f2 = scientific::mask_indices(3, "triu");
    print("upper triangle of 3 by 3: ", up.data, "\n");
    let walk, f3 = scientific::ndindex([2, 2]);
    print("places of a 2 by 2 shape: ", walk.data, "\n");
    var x: f64[>..] = [];
    i = 0;
    while i < 4 { x.push(1.0); i += 1; }
    var sx: i64[..8] = [];
    sx.push(2);
    sx.push(2);
    let m = scientific::Arr { ndim: 2, shape: sx, data: copy(x) };
    let path, ops, f4 = scientific::einsum_path3("ij,jk,kl->il", m, m, m);
    print("path ", path.data, " with ", ops, " operations\n");
    assert(f1.kind == scientific::F_NONE && f2.kind == scientific::F_NONE);
    assert(f3.kind == scientific::F_NONE && f4.kind == scientific::F_NONE);
    return 0;
}
