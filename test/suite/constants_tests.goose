// The tests of the group constants. Run them with SCIENTIFIC_SUITES=constants.
namespace constants_tests;

import scientific from "../..";
import os;

fn test_constants() {
    // SciPy 1.18.1 holds the values of CODATA 2022.
    assert(constants::c == 299792458.0);
    assert(constants::pi == 3.141592653589793);
    assert(constants::golden == 1.618033988749895);
    assert(constants::kilo == 1000.0 && constants::milli == 0.001 && constants::mega == 1000000.0);
    assert(constants::hour == 3600.0 && constants::day == 86400.0 && constants::week == 604800.0);
    assert(constants::inch == 0.0254 && constants::foot == 0.30479999999999996 && constants::mile == 1609.3439999999998);
    assert(constants::zero_Celsius == 273.15 && constants::atm == 101325.0);
    assert(constants::h == 6.62607015e-34 && constants::N_A == 6.02214076e+23 && constants::k == 1.380649e-23);
    assert(constants::m_e == 9.1093837139e-31 && constants::e == 1.602176634e-19);
    assert(constants::G == 6.6743e-11 && constants::g == 9.80665);
    // The table of the physical constants: the value, the unit and the
    // relative uncertainty by name.
    let me, fme = scientific::value("electron mass");
    assert(fme.kind == scientific::F_NONE && me == 9.1093837139e-31);
    let un, fun = scientific::unit("electron mass");
    assert(fun.kind == scientific::F_NONE && un == "kg");
    let vu, fvu = scientific::unit("fine-structure constant");
    assert(fvu.kind == scientific::F_NONE && vu.len == 0);
    let pr, fpr = scientific::precision("electron mass");
    assert(fpr.kind == scientific::F_NONE && pr > 3.0e-10 && pr < 3.2e-10);
    // The precision keeps the sign of the value, as SciPy gives it.
    let pg, fpg = scientific::precision("electron g factor");
    assert(fpg.kind == scientific::F_NONE && pg == -1.7979150438990604e-13);
    let _v, bad = scientific::value("no such constant");
    assert(bad.kind == scientific::F_VALUE);
    let _w, bad2 = scientific::value("Electron Mass");
    assert(bad2.kind == scientific::F_VALUE);
    let _u, bad3 = scientific::unit("no such constant");
    assert(bad3.kind == scientific::F_VALUE);
    let _x, bad4 = scientific::precision("no such constant");
    assert(bad4.kind == scientific::F_VALUE);
    // The search ignores the case and gives sorted names, one per line.
    let hits = scientific::find("ELECTRON MASS");
    assert(hits.len > 0 && hits[0] == 97);
    var lines: i64 = 1;
    var q: i64 = 0;
    while q < hits.len { if hits[q] == 10 { lines += 1; } q += 1; }
    assert(lines == 12);
    let none = scientific::find("zzzz");
    assert(none.len == 0);
    // The scales of temperature: 100 C is 212 F and 373.15 K. The short
    // names and the case do not matter.
    let t, ft = scientific::from_values([100.0, 0.0, -40.0]);
    assert(ft.kind == scientific::F_NONE);
    let f, ff = scientific::convert_temperature(t, "Celsius", "Fahrenheit");
    assert(ff.kind == scientific::F_NONE && f.data[0] == 212.0 && f.data[1] == 32.0 && f.data[2] == -40.0);
    let k, fk = scientific::convert_temperature(t, "c", "KELVIN");
    assert(fk.kind == scientific::F_NONE && k.data[0] == 373.15 && k.data[1] == 273.15);
    let r, fr = scientific::convert_temperature(t, "K", "R");
    assert(fr.kind == scientific::F_NONE && r.data[0] == 180.0);
    let _bt, fbt = scientific::convert_temperature(t, "X", "K");
    assert(fbt.kind == scientific::F_VALUE);
    // Wavelength and frequency: 500 nm is 599584916000000 Hz in SciPy.
    let l, fl = scientific::from_values([500.0e-9]);
    assert(fl.kind == scientific::F_NONE);
    let nu, fnu = scientific::lambda2nu(l);
    assert(fnu.kind == scientific::F_NONE && nu.data[0] == 299792458.0 / 500.0e-9);
    let back, fback = scientific::nu2lambda(nu);
    assert(fback.kind == scientific::F_NONE && back.data[0] == 299792458.0 / nu.data[0]);
}
