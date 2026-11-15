// The smoothing spline filters, the fast convolution and the short
// time transform. Run it with:
// goose run examples/signal_spline.goose
import std;
import scientific from "..";

fn main() {
    // The cubic spline of a short signal.
    let sig = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let c, fc = scientific::cspline1d(sig);
    assert(fc.kind == scientific::F_NONE);
    print("cubic spline: ");
    var i: i64 = 0;
    while i < c.data.len {
        print(c.data[i], " ");
        i += 1;
    }
    print("\n");

    // The values of the spline at the places of the signal.
    let v, fv = scientific::cspline1d_eval(copy(c.data), [0.0, 1.0, 2.0, 3.0]);
    assert(fv.kind == scientific::F_NONE);
    print("spline at 0, 1, 2, 3: ");
    i = 0;
    while i < v.data.len {
        print(v.data[i], " ");
        i += 1;
    }
    print("\n");

    // The overlap add convolution of a signal with a short filter.
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [1.0, 0.5];
    let o, fo = scientific::oaconvolve(a, b);
    assert(fo.kind == scientific::F_NONE);
    print("overlap add: ");
    i = 0;
    while i < o.data.len {
        print(o.data[i], " ");
        i += 1;
    }
    print("\n");

    // The upsample filter of a signal. The signal holds one shape.
    var sh: i64[..8] = [];
    sh.push(3);
    let sig2 = scientific::arr_of([1.0, 2.0, 3.0], sh);
    let u, fu = scientific::upfirdn([1.0, 1.0, 1.0], sig2, 2, 1);
    assert(fu.kind == scientific::F_NONE);
    print("upsample: ");
    i = 0;
    while i < u.data.len {
        print(u.data[i], " ");
        i += 1;
    }
    print("\n");

    // The short time transform of a signal and the signal back.
    let win = [0.0, 0.5, 1.0, 0.5, 0.0];
    let plan, fp = scientific::stft_plan(win, 2, 10.0);
    assert(fp.kind == scientific::F_NONE);
    let s, fs = scientific::stft_forward(plan, scientific::Arr {
        ndim: 1, shape: [8], data: [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    }, -1);
    assert(fs.kind == scientific::F_NONE);
    print("transform holds ", s.shape[0], " frequencies and ",
          s.shape[1], " parts\n");
    let y, fy = scientific::stft_inverse(plan, s, -1);
    assert(fy.kind == scientific::F_NONE);
    print("signal back: ");
    i = 0;
    while i < y.data.len {
        print(y.data[i], " ");
        i += 1;
    }
    print("\n");
}
