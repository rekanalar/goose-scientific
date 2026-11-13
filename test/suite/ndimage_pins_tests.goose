// The tests of the group ndimage_pins. Run them with SCIENTIFIC_SUITES=ndimage_pins.
namespace ndimage_pins_tests;

import scientific from "../..";
import os;
import ndimage_2_tests;
import ndimage_tests;

fn test_ndimage_pins() {
    ndimage_tests::test_ndimage_pins0();
    ndimage_tests::test_ndimage_pins1();
    ndimage_tests::test_ndimage_pins2();
    ndimage_tests::test_ndimage_pins3();
    ndimage_tests::test_ndimage_pins4();
    ndimage_tests::test_ndimage_pins5();
    ndimage_tests::test_ndimage_pins6();
    ndimage_tests::test_ndimage_pins7();
    ndimage_tests::test_ndimage_pins8();
    ndimage_tests::test_ndimage_pins9();
    ndimage_tests::test_ndimage_pins10();
    ndimage_2_tests::test_ndimage_pins11();
    ndimage_2_tests::test_ndimage_pins12();
    ndimage_2_tests::test_ndimage_pins13();
    ndimage_2_tests::test_ndimage_pins14();
    ndimage_2_tests::test_ndimage_pins15();
    ndimage_2_tests::test_ndimage_pins16();
    ndimage_2_tests::test_ndimage_pins17();
    ndimage_2_tests::test_ndimage_pins18();
}
