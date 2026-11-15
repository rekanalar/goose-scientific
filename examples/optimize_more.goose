// The plain calls of the optimization and the responses of the
// systems. Run it with:
// goose run examples/optimize_more.goose
import std;
import scientific from "..";

fn main() {
    // The function of Rosenbrock, its gradient and its Hessian.
    let f = scientific::rosen([-1.2, 1.0]);
    let g = scientific::rosen_der([-1.2, 1.0]);
    let h = scientific::rosen_hess([-1.2, 1.0]);
    print("rosen(-1.2, 1) = ", f, "\n");
    print("gradient at the start = [", g.data[0], ", ", g.data[1], "]\n");
    print("hessian[0][0] = ", h.data[0], "\n");

    // The check of a gradient of a block that gives the value and
    // the gradient.
    let d, cf = scientific::check_grad([0.5, -1.0]) { v =>
        var s: f64 = 0.0;
        var gr: f64[>..] = [];
        var i: i64 = 0;
        while i < v.len { gr.push(0.0); i += 1; }
        i = 0;
        while i + 1 < v.len {
            let e = v[i + 1] - v[i] * v[i];
            s += 100.0 * e * e + (1.0 - v[i]) * (1.0 - v[i]);
            gr[i] += -400.0 * v[i] * e - 2.0 * (1.0 - v[i]);
            gr[i + 1] += 200.0 * e;
            i += 1;
        }
        var o: f64[>..] = [];
        o.push(s);
        i = 0;
        while i < gr.len { o.push(gr[i]); i += 1; }
        o
    };
    assert(cf.kind == scientific::F_NONE);
    print("check of the gradient = ", d, "\n");

    // The search of a bracket around a minimum.
    let pts, vals, bf = scientific::bracket(0.1, 1.0) { x =>
        let e = x - 1.0;
        e * e
    };
    assert(bf.kind == scientific::F_NONE);
    print("bracket points = [", pts.data[0], ", ", pts.data[1], ", ", pts.data[2], "]\n");

    // The fit that must not fall.
    let fit, wts, iff = scientific::isotonic_regression([3.0, 1.0, 2.0, 2.0, 5.0]);
    assert(iff.kind == scientific::F_NONE);
    print("isotonic fit = [", fit.data[0], ", ", fit.data[1], ", ", fit.data[2], ", ",
        fit.data[3], ", ", fit.data[4], "]\n");

    // The answer of 1 / (s + 1)^2 to a hit and to a held input. The
    // system sits in the state space form.
    var am: f64[>..] = [-2.0, -1.0, 1.0, 0.0];
    var bm: f64[>..] = [1.0, 0.0];
    var cm: f64[>..] = [0.0, 1.0];
    let a = scientific::Arr { ndim: 2, shape: [2, 2], data: copy(am) };
    let b = scientific::Arr { ndim: 2, shape: [2, 1], data: copy(bm) };
    let c = scientific::Arr { ndim: 2, shape: [1, 2], data: copy(cm) };
    let ti, hi, rif = scientific::impulse(a, bm, cm, 0.0, [0.0, 0.5, 1.0]);
    assert(rif.kind == scientific::F_NONE);
    print("impulse at 0, 0.5, 1 = [", hi.data[0], ", ", hi.data[1], ", ", hi.data[2], "]\n");
    let ts, hs, rsf = scientific::step(a, bm, cm, 0.0, [0.0, 0.5, 1.0]);
    assert(rsf.kind == scientific::F_NONE);
    print("step at 0, 0.5, 1 = [", hs.data[0], ", ", hs.data[1], ", ", hs.data[2], "]\n");

    // The size of the answer of the same system, in decibel.
    let wb, mag, ph, rbf = scientific::bode([1.0], [1.0, 2.0, 1.0], [0.1, 1.0, 10.0]);
    assert(rbf.kind == scientific::F_NONE);
    print("bode size at 0.1, 1, 10 = [", mag.data[0], ", ", mag.data[1], ", ", mag.data[2], "]\n");

    // The answer of the discrete filter 1 / (z + 1)^2 to a held input.
    let td, yd, rdf = scientific::dstep([1.0], [1.0, 2.0, 1.0], 4);
    assert(rdf.kind == scientific::F_NONE);
    print("dstep = [", yd.data[0], ", ", yd.data[1], ", ", yd.data[2], ", ", yd.data[3], "]\n");
}
