// The design of a filter: the analog prototype, the order, the cuts
// and the response of the filter.
import scientific from "..";

fn show(label: const u8[:], a: scientific::Arr&) {
    print(label);
    var i: i64 = 0;
    while i < a.data.len { print(" ", a.data[i]); i += 1; }
    print("\n");
}

fn main() {
    // The analog prototype of a Butterworth filter of order 3. Its cut
    // sits at one radian per second.
    let zr, zi, pr, pim, k, f1 = scientific::buttap(3);
    if f1.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f1.kind), "\n"); return; }
    show("poles of the prototype (real):", pr);
    show("poles of the prototype (imaginary):", pim);
    print("gain: ", k, "\n");

    // The order of a lowpass filter. The pass band ends at 0.2 and the
    // stop band starts at 0.8. The filter loses 1 dB in the pass band
    // and 30 dB in the stop band.
    let n, wn, f2 = scientific::buttord([0.2], [0.8], 1.0, 30.0);
    if f2.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f2.kind), "\n"); return; }
    print("order: ", n, " cut: ");
    show("", wn);

    // The filter itself. The cuts of a digital filter sit in the range
    // of zero to one.
    let b, a, f3 = scientific::iirfilter(n, wn.data, 1.0, 30.0, "lowpass", "butter");
    if f3.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f3.kind), "\n"); return; }
    show("numerator:", b);
    show("denominator:", a);

    // The response of the filter at eight points.
    let w, hr, hi, f4 = scientific::freqz(b.data, a.data, 8);
    if f4.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f4.kind), "\n"); return; }
    show("frequencies:", w);
    show("real parts:", hr);
    show("imaginary parts:", hi);

    // The group delay of the filter.
    let gw, gd, f5 = scientific::group_delay(b.data, a.data, 8);
    if f5.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f5.kind), "\n"); return; }
    show("group delay:", gd);

    // A filter from its edges. The elliptic filter gives the smallest
    // order for the same demands.
    let eb, ea, f6 = scientific::iirdesign([0.2], [0.8], 1.0, 30.0, false, "ellip");
    if f6.kind != scientific::F_NONE { print("fault ", scientific::kind_name(f6.kind), "\n"); return; }
    show("numerator of the elliptic filter:", eb);
    show("denominator of the elliptic filter:", ea);
    return;
}
