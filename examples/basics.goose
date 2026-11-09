import scientific from "..";

// The basics of the package: build arrays, reshape and transpose
// them, broadcast a row over a matrix, and reduce to one number.
fn main() {
    let a, f = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    if f.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f.kind), "\n"); return; }
    print("a ", a.ndim, "x", a.shape[0], "x", a.shape[1], " elements ", a.data.len, "\n");

    let t, ft = scientific::transpose(a);
    print("transpose shape ", t.shape[0], "x", t.shape[1], " first ", t.data[1], "\n");

    let row, fr = scientific::linspace(10.0, 30.0, 3);
    let s, fs = scientific::add(a, row);
    print("add first ", s.data[0], " last ", s.data[5], " fault ", fs.kind, "\n");

    let total, fu = scientific::sum(a);
    let m, fm = scientific::mean(a);
    print("sum ", total, " mean ", m, "\n");

    let nine: i64[>..] = [3, 3, 3, 3, 3, 3, 3, 3, 3];
    let big, fb = scientific::zeros(nine);
    print("nine dimensions fault ", scientific::kind_name(fb.kind), "\n");
    return;
}
