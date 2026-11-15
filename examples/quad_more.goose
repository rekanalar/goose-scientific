// The weights of Newton and Cotes, the sum of a series and the rule of the double exponential.
import scientific from "..";

fn inv_square(k: f64) -> f64 { return 1.0 / (k * k); }

fn main() {
    // The rule of Simpson is the rule of the order two.
    let w, b, f = scientific::newton_cotes(2);
    assert(f.kind == scientific::F_NONE);
    print("simpson weights: ", w.data, " error factor: ", b, "\n");

    // The sum of one over the square of every number is pi squared over six.
    let s, e, fs = scientific::nsum(1.0, scientific::quad_inf()) { x => inv_square(x) };
    assert(fs.kind == scientific::F_NONE);
    print("sum: ", s, " error: ", e, "\n");

    // The rule of the double exponential over a finite range.
    let v, err, ft = scientific::tanhsinh(0.0, 2.0) { x => qm_exp_sin(x) };
    assert(ft.kind == scientific::F_NONE);
    print("integral: ", v, " error: ", err, "\n");
}

// qm_exp_sin gives the exponential times the sine.
fn qm_exp_sin(x: f64) -> f64 {
    return exp(x) * sin(x);
}
