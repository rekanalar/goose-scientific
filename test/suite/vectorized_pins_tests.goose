// The tests of the group vectorized_pins. Run them with SCIENTIFIC_SUITES=vectorized_pins.
namespace vectorized_pins_tests;

import scientific from "../..";
import os;
import vectorized_tests;

fn test_vectorized_pins() {
    vectorized_tests::test_vectorized_pins0();
    vectorized_tests::test_vectorized_pins1();
}
