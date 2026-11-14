// The continuous random laws. Each call draws one number from a seeded
// generator. The program prints the mean of 10000 numbers of each law.
import std;
import scientific from "..";

fn main() {
    var r = scientific::default_rng(2024);
    var sum: f64 = 0.0;
    var i: i64 = 0;
    while i < 10000 { let x, f = scientific::gamma(r, 2.0, 1.5); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("gamma mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::lognormal(r, 0.5, 0.5); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("lognormal mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::chisquare(r, 5.0); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("chisquare mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::beta(r, 2.0, 3.0); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("beta mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::rayleigh(r, 2.0); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("rayleigh mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::weibull(r, 1.5); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("weibull mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::gumbel(r, 1.0, 2.0); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("gumbel mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::logistic(r, -1.0, 3.0); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("logistic mean ", sum / 10000.0, "\n");
    // A negative scale is a fault.
    let bad, fb = scientific::gamma(r, 2.0, -1.0);
    assert(fb.kind == scientific::F_VALUE);
    print("a negative scale gives a fault\n");
}
