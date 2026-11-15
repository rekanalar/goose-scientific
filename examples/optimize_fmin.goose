// The four methods of minimization of SciPy. Run it with:
// goose run examples/optimize_fmin.goose
import std;
import scientific from "..";

// none_row gives an empty list. It stands for a side without a bound.
fn none_row() -> f64[>..] {
    var o: f64[>..] = [];
    return o;
}

fn main() {
    // The method of Powell needs the values of the function only.
    let p, fp = scientific::fmin_powell([2.0, 3.0], 0.0001, 0.0001) { v =>
        var o: f64[>..] = [];
        o.push(v[0] * v[0] + 2.0 * v[1] * v[1]);
        o
    };
    assert(fp.kind == scientific::F_NONE);
    print("Powell finds [", p.data[0], ", ", p.data[1], "]\n");

    // The method of Powell with constraints needs the values of the
    // constraints too. A constraint must be zero or above.
    let c, fc = scientific::fmin_cobyla([0.0, 0.0], 1.0, 0.00000001, 5000, 0.00000001) { v =>
        var o: f64[>..] = [];
        o.push((v[0] - 1.0) * (v[0] - 1.0) + (v[1] - 2.0) * (v[1] - 2.0));
        o.push(v[0] + v[1] - 2.0);
        o
    };
    assert(fc.kind == scientific::F_NONE);
    print("the method with constraints finds [", c.data[0], ", ", c.data[1], "]\n");

    // The method of the least squares holds equality constraints. The
    // first neq constraints of the block must equal zero.
    let s, fs = scientific::fmin_slsqp([0.5, 2.0], 1, none_row(), none_row(), 0.00000001, 100) { v =>
        let t = v[1] - v[0] * v[0];
        var o: f64[>..] = [];
        o.push((1.0 - v[0]) * (1.0 - v[0]) + 100.0 * t * t);
        o.push(v[1] - v[0] * v[0] * v[0] - 1.0);
        o
    };
    assert(fs.kind == scientific::F_NONE);
    print("the least squares finds [", s.data[0], ", ", s.data[1], "]\n");

    // The method of the truncated Newton needs the gradient. The block
    // gives the value of the function first and the gradient after it.
    let t, ft = scientific::fmin_tnc([0.0, 0.0], [0.0, -5.0], [1.0, 5.0]) { v =>
        var o: f64[>..] = [];
        o.push((v[0] - 2.0) * (v[0] - 2.0) + (v[1] - 1.0) * (v[1] - 1.0));
        o.push(2.0 * (v[0] - 2.0));
        o.push(2.0 * (v[1] - 1.0));
        o
    };
    assert(ft.kind == scientific::F_NONE);
    print("the truncated Newton finds [", t.data[0], ", ", t.data[1], "]\n");

    // A bound that lies above the other bound gives a fault.
    let bad, fbad = scientific::fmin_tnc([0.0, 0.0], [1.0, 1.0], [0.0, 0.0]) { v =>
        var o: f64[>..] = [];
        o.push(v[0] * v[0] + v[1] * v[1]);
        o.push(2.0 * v[0]);
        o.push(2.0 * v[1]);
        o
    };
    assert(fbad.kind == scientific::F_VALUE);
    print("the fault of the bounds is ", scientific::kind_name(fbad.kind), "\n");
    return 0;
}
