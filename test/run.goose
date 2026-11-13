// Tests of the package. Run them all with: goose run test/run.goose
// Run one group:    SCIENTIFIC_SUITES=<group> goose run test/run.goose
// Run one test:     SCIENTIFIC_TESTS=<test> goose run test/run.goose
// Both variables run the test only when it belongs to the group.
// A name that no group or test has stops the run with a message.
// The program must start in the package directory.

import std;
import os;
import scientific from "..";
import suite from "./suite";

fn main() {
    var group: u8[>..] = [];
    var test: u8[>..] = [];
    let has_group = env("SCIENTIFIC_SUITES", group) && group.len > 0;
    let has_test = env("SCIENTIFIC_TESTS", test) && test.len > 0;
    var found = true;
    if has_group && has_test {
        found = suite::run_test_in(group[..], test[..]);
        if !found { print("no test ", test[..], " in the group ", group[..], "\n"); }
    } else if has_test {
        found = suite::run_test(test[..]);
        if !found { print("no test named ", test[..], "\n"); }
    } else if has_group {
        found = suite::run_group(group[..]);
        if !found { print("no group named ", group[..], "\n"); }
    } else {
        suite::run_all();
    }
    if !found {
        suite::print_groups();
        assert(false);
    }
    print("ok\n");
    return 0;
}
