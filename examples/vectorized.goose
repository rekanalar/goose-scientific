// The vectorized filter sums the window of side 3 of a small array.
import scientific from "..";

fn main() {
    let v, f0 = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let a, f1 = scientific::reshape(v, [3, 3]);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);

    // The block gets the windowed array. Its shape is the shape of the
    // input followed by the shape of the window. The block sums the
    // window of every place.
    let out, f2 = scientific::vectorized_filter(a, [3, 3], "reflect", 0.0, []) { x, o =>
        var h = x.ndim / 2;
        var d = 0;
        var n = 1;
        while d < h { n *= x.shape[d]; d += 1; }
        var w = 1;
        d = h;
        while d < x.ndim { w *= x.shape[d]; d += 1; }
        var p = 0;
        while p < n {
            var s = 0.0;
            var j = 0;
            while j < w { s += x.data[p * w + j]; j += 1; }
            o[p] = s;
            p += 1;
        }
    };
    assert(f2.kind == scientific::F_NONE);
    print("sums: ", out.data, "\n");
    return;
}
