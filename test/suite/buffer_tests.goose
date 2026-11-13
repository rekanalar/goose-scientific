// The tests of the group buffer. Run them with SCIENTIFIC_SUITES=buffer.
namespace buffer_tests;

import scientific from "../..";
import os;

fn test_buffer() {
    // NumPy 2.5.3: little endian numbers of 64 bits decode from bytes.
    var b: u8[>..] = [0, 0, 0, 0, 0, 0, 248, 63, 0, 0, 0, 0, 0, 0, 0, 192, 0, 0, 0, 0, 0, 0, 10, 64, 0, 0, 0, 0, 0, 0, 16, 64];
    let a, f = scientific::frombuffer(b);
    assert(f.kind == scientific::F_NONE && a.data.len == 4);
    assert(a.data[0] == 1.5 && a.data[1] == -2.0 && a.data[2] == 3.25 && a.data[3] == 4.0);
    let c, f2 = scientific::frombuffer(b, 2);
    assert(f2.kind == scientific::F_NONE && c.data.len == 2 && c.data[1] == -2.0);
    let r, f3 = scientific::frombuffer(b, -1, 8);
    assert(f3.kind == scientific::F_NONE && r.data.len == 3 && r.data[0] == -2.0);
    // An offset that no number holds, a rest of bytes and a count above
    // the rest are faults.
    let _x, bad = scientific::frombuffer(b, -1, 4);
    assert(bad.kind == scientific::F_VALUE);
    let _y, bad2 = scientific::frombuffer(b[..30]);
    assert(bad2.kind == scientific::F_VALUE);
    let _z, bad3 = scientific::frombuffer(b, 5);
    assert(bad3.kind == scientific::F_VALUE);
    let empty, f4 = scientific::frombuffer(b[..0]);
    assert(f4.kind == scientific::F_NONE && empty.data.len == 0);
    // The missing value, the zero below zero and the infinity keep
    // their meaning.
    var nb: u8[>..] = [0, 0, 0, 0, 0, 0, 248, 127, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 240, 127];
    let n, f5 = scientific::frombuffer(nb);
    assert(f5.kind == scientific::F_NONE && n.data[0] != n.data[0] && n.data[1] == 0.0 && 1.0 / n.data[1] < 0.0);
    assert(n.data[2] > 0.0 && n.data[2] == n.data[2] * 2.0);
    // The smallest numbers below the normal carry their value in the
    // digits alone.
    var sb: u8[>..] = [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
    let sm, f6 = scientific::frombuffer(sb);
    assert(f6.kind == scientific::F_NONE);
    assert(sm.data[0] == 4.9406564584124654e-324 && sm.data[1] == 2.0 * 4.9406564584124654e-324);
    // A file holds the same bytes. A count above the file takes what
    // the file holds.
    let ok = write_file("sci_f64_tmp.bin", b);
    assert(ok);
    let fa, ff = scientific::fromfile("sci_f64_tmp.bin");
    assert(ff.kind == scientific::F_NONE && fa.data.len == 4 && fa.data[3] == 4.0);
    let fc, ffc = scientific::fromfile("sci_f64_tmp.bin", 2);
    assert(ffc.kind == scientific::F_NONE && fc.data.len == 2);
    let fmore, ffmore = scientific::fromfile("sci_f64_tmp.bin", 9);
    assert(ffmore.kind == scientific::F_NONE && fmore.data.len == 4);
    // A file that does not hold a count of numbers drops the rest.
    let ok2 = write_file("sci_f64_tmp.bin", b[..20]);
    assert(ok2);
    let fp, ffp = scientific::fromfile("sci_f64_tmp.bin");
    assert(ffp.kind == scientific::F_NONE && fp.data.len == 2);
    let _gone, fgone = scientific::fromfile("sci_f64_no.bin");
    assert(fgone.kind == scientific::F_VALUE);
    assert(delete_file("sci_f64_tmp.bin"));
}
