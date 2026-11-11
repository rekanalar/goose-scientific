// The linear filters of arrays: smooth a matrix with a Gaussian, take
// the derivative of a line, and correlate with a kernel of two dimensions.
import scientific from "..";

fn show(name: const u8[:], a: scientific::Arr&) {
    print(name, ": ", a.data, "\n");
}

fn main() {
    let v, f0 = scientific::from_values([0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0]);
    assert(f0.kind == scientific::F_NONE);
    let m, f1 = scientific::reshape(v, [3, 4]);
    assert(f1.kind == scientific::F_NONE);

    // A Gaussian of sigma 1.0 on every axis. The border repeats the edge.
    let smooth, f2 = scientific::gaussian_filter(m, [1.0], [], "reflect");
    assert(f2.kind == scientific::F_NONE);
    show("gaussian", smooth);

    // The first derivative of a Gaussian along the last axis.
    let slope, f3 = scientific::gaussian_filter1d(m, 1.0, -1, 1);
    assert(f3.kind == scientific::F_NONE);
    show("derivative", slope);

    // The mean of a box of 2 by 3 values.
    let box, f4 = scientific::uniform_filter(m, [2, 3], "nearest");
    assert(f4.kind == scientific::F_NONE);
    show("box", box);

    // A kernel of two dimensions. The border is zero.
    let k, f5 = scientific::from_values([0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0]);
    assert(f5.kind == scientific::F_NONE);
    let kernel, f6 = scientific::reshape(k, [3, 3]);
    assert(f6.kind == scientific::F_NONE);
    let lap, f7 = scientific::correlate_nd(m, kernel, "constant");
    assert(f7.kind == scientific::F_NONE);
    show("laplace", lap);

    // A fault names the check that failed.
    let bad, f8 = scientific::gaussian_filter1d(m, 0.0);
    print("sigma 0 gives a fault of kind ", scientific::kind_name(f8.kind), "\n");
}
