// The text and the files: the print of an array, the text files, and
// the .npy and .npz files. Run it with:
// goose run examples/array_textio.goose
import std;
import os;
import scientific from "..";

fn one_of(values: f64[>..]) -> scientific::Arr {
    var s: i64[..8] = [];
    s.push(values.len);
    return scientific::Arr { ndim: 1, shape: s, data: values };
}

fn two_of(values: f64[>..]) -> scientific::Arr {
    var s: i64[..8] = [];
    s.push(2);
    s.push(2);
    return scientific::Arr { ndim: 2, shape: s, data: values };
}

fn main() {
    // The text of an array, as NumPy prints it.
    var v: f64[>..] = [];
    v.push(1.0);
    v.push(-2.5);
    v.push(3.25);
    let a = one_of(copy(v));
    let text, f = scientific::array2string(a);
    assert(f.kind == scientific::F_NONE);
    print("the text of the array: ", text, "\n");
    let repr, fr = scientific::array_repr(a);
    assert(fr.kind == scientific::F_NONE);
    print("the repr of the array: ", repr, "\n");
    // A matrix shows one row on each line.
    var m: f64[>..] = [];
    m.push(1.0);
    m.push(2.0);
    m.push(3.0);
    m.push(4.0);
    let b = two_of(copy(m));
    let mt, fm = scientific::array2string(b);
    assert(fm.kind == scientific::F_NONE);
    print("the text of the matrix: ", mt, "\n");

    // A text file holds one row of the array on each line.
    var so = scientific::save_options();
    let ok, fs = scientific::savetxt("sci_textio_example.txt", b, &so);
    assert(ok && fs.kind == scientific::F_NONE);
    var to = scientific::txt_options();
    let c, fc = scientific::loadtxt("sci_textio_example.txt", &to);
    assert(fc.kind == scientific::F_NONE);
    print("the file holds ", c.shape[0], " rows of ", c.shape[1], " columns\n");
    print("the first value of the file: ", c.data[0], "\n");
    assert(delete_file("sci_textio_example.txt"));

    // The .npy file holds the numbers of the array as they are.
    let ok2, f2 = scientific::save("sci_textio_example.npy", a);
    assert(ok2 && f2.kind == scientific::F_NONE);
    let d, f3 = scientific::load("sci_textio_example.npy");
    assert(f3.kind == scientific::F_NONE);
    print("the .npy file holds ", d.data.len, " numbers, the first is ",
          d.data[0], "\n");
    assert(delete_file("sci_textio_example.npy"));

    // The .npz file holds the named arrays.
    let ok3, f4 = scientific::savez("sci_textio_example.npz", "x", a, "y", b);
    assert(ok3 && f4.kind == scientific::F_NONE);
    var bytes: u8[>..] = [];
    assert(read_file("sci_textio_example.npz", bytes));
    print("the .npz file holds ", bytes.len, " bytes\n");
    assert(delete_file("sci_textio_example.npz"));
}
