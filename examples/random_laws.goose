// The last continuous random laws. Each call draws one number from a
// seeded generator. The program prints the mean of 10000 numbers of
// each law.
import std;
import scientific from "..";

fn main() {
    var r = scientific::default_rng(2024);
    var sum: f64 = 0.0;
    var i: i64 = 0;
    while i < 10000 { let x, f = scientific::f(r, 5.0, 8.0); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("f mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::noncentral_chisquare(r, 5.0, 2.0); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("noncentral_chisquare mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::noncentral_f(r, 5.0, 8.0, 3.0); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("noncentral_f mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::pareto(r, 6.0); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("pareto mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::triangular(r, -2.0, 0.0, 3.0); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("triangular mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::vonmises(r, 0.0, 1.5); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("vonmises mean ", sum / 10000.0, "\n");
    sum = 0.0;
    i = 0;
    while i < 10000 { let x, f = scientific::wald(r, 2.0, 3.0); assert(f.kind == scientific::F_NONE); sum += x; i += 1; }
    print("wald mean ", sum / 10000.0, "\n");
    // A count of degrees of freedom at or below zero is a fault.
    let _, fb = scientific::f(r, 0.0, 8.0);
    assert(fb.kind == scientific::F_VALUE);
    print("a count of zero gives a fault\n");
}
