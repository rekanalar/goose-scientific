// The tests of the group geometric_pins. Run them with SCIENTIFIC_SUITES=geometric_pins.
namespace geometric_pins_tests;

import scientific from "../..";
import os;
import geometric_tests;

fn test_geometric_pins() {
    geometric_tests::test_geometric_pins0();
    geometric_tests::test_geometric_pins1();
    geometric_tests::test_geometric_pins2();
    geometric_tests::test_geometric_pins3();
    geometric_tests::test_geometric_pins4();
    geometric_tests::test_geometric_pins5();
}
