// The discrete random laws. Each call draws one whole number from a
// seeded generator. The program prints the mean of 10000 numbers of
// each law.
import std;
import scientific from "..";

fn main() {
    var r = scientific::default_rng(2025);
    var sum: f64 = 0.0;
    var i: i64 = 0;
    while i < 10000 { let x, f = scientific::binomial(r, 20, 0.3); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("binomial mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::geometric(r, 0.2); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("geometric mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::hypergeometric(r, 10, 20, 12); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("hypergeometric mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::logseries(r, 0.3); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("logseries mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::negative_binomial(r, 5.0, 0.4); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("negative_binomial mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::zipf(r, 6.0); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("zipf mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    let pv, fpv = scientific::from_values([0.2, 0.3, 0.5]);
    while i < 10000 {
        let c, f = scientific::multinomial(r, 20, pv);
        assert(f.kind == scientific::F_NONE);
        sum += c.data[0];
        i += 1;
    }
    print("multinomial first class mean ", sum / 10000.0, "\n");
    let bad, fbad = scientific::binomial(r, 10, -0.5);
    assert(fbad.kind == scientific::F_VALUE);
    print("binomial refuses a probability below zero\n");
}
