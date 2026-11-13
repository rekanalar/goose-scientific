// The tests of the group generic_pins. Run them with SCIENTIFIC_SUITES=generic_pins.
namespace generic_pins_tests;

import scientific from "../..";
import os;
import generic_tests;

fn test_generic_pins() {
    generic_tests::test_generic_pins0();
    generic_tests::test_generic_pins1();
    generic_tests::test_generic_pins2();
    generic_tests::test_generic_pins3();
    generic_tests::test_generic_pins4();
    generic_tests::test_generic_pins5();
    generic_tests::test_generic_pins6();
    generic_tests::test_generic_pins7();
    generic_tests::test_generic_pins8();
}
