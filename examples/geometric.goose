// The geometric transform warps an array with a mapping of the coordinates.
// The block reads the coordinate of an output value and writes the
// coordinate in the input.
import scientific from "..";

fn show(name: const u8[:], a: scientific::Arr&) {
    print(name, ": ", a.data, "\n");
}

fn main() {
    let v, f0 = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b, f1 = scientific::reshape(v, [2, 3]);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);

    let w, f2 = scientific::geometric_transform(b, [], 1, "nearest", 0.0, true) { c, o =>
        var d = 0;
        while d < o.len {
            o[d] = c[d] * 0.5 + 0.25;
            d += 1;
        }
    };
    assert(f2.kind == scientific::F_NONE);
    show("warped", w);

    // The line filter sums a window of three along the last axis.
    let s, f3 = scientific::generic_filter1d(b, 3, -1, "reflect", 0.0, 0) { line, out =>
        var j = 0;
        while j < out.len {
            out[j] = line[j] + line[j + 1] + line[j + 2];
            j += 1;
        }
    };
    assert(f3.kind == scientific::F_NONE);
    show("filtered", s);
    return;
}
