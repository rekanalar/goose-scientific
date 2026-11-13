import scientific from "..";

// The orthogonal polynomial bases: evaluate a series, differentiate it,
// integrate it and find its roots.
fn main() {
    // The Legendre series 1 + 2 P1 + 3 P2.
    let c = [1.0, 2.0, 3.0];
    let v, f = scientific::legval(0.5, c);
    print("legval at 0.5 ", v, " fault ", scientific::kind_name(f.kind), "\n");

    let d, fd = scientific::legder(c);
    print("legder ", d.data[0], " ", d.data[1], " fault ", scientific::kind_name(fd.kind), "\n");

    let i, fi = scientific::legint(c, 1, [0.0], 0.0);
    print("legint length ", i.data.len, " fault ", scientific::kind_name(fi.kind), "\n");

    let re, im, fr = scientific::legroots(c);
    print("legroots ", re.data[0], " ", re.data[1], " fault ", scientific::kind_name(fr.kind), "\n");

    // The same series in the monomial basis.
    let p, fp = scientific::leg2poly(c);
    print("leg2poly ", p.data[0], " ", p.data[1], " ", p.data[2], "\n");

    // The Laguerre series of the same coefficients.
    let l, fl2 = scientific::lagval(0.5, c);
    print("lagval at 0.5 ", l, " fault ", scientific::kind_name(fl2.kind), "\n");

    // A fault of a hostile input.
    let bad, fbad = scientific::legval(0.5, [1.0, 1.0 / 0.0]);
    print("legval with a value that is not finite ", scientific::kind_name(fbad.kind), "\n");
}
