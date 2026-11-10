import scientific from "..";

// Integrals of several variables and of a vector. The block of nquad
// takes the point as a slice. The block of quad_vec fills a row.
fn main() {
    let v, e, f = scientific::nquad([0.0, 1.0, 0.0, 2.0]) { p => p[0] * p[1] * p[1] };
    if f.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f.kind), "\n"); return; }
    print("nquad ", v, " error ", e, "\n");

    let r, re, rf = scientific::quad_vec(0.0, 3.141592653589793) { x, out => { out.push(sin(x)); out.push(x * x); } };
    if rf.kind != scientific::F_NONE { print("fault ", scientific::kind_name(rf.kind), "\n"); return; }
    print("quad_vec ", r.data[0], " ", r.data[1], " error ", re, "\n");
    return;
}
