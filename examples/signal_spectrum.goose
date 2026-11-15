// The spectrum, the windows and the waves: the chirp z-transform, the
// cross spectrum, the envelope, the peaks and a swept wave. Run it with:
// goose run examples/signal_spectrum.goose
import std;
import scientific from "..";

fn main() {
    // A chirp that sweeps from 30 Hz to 10 Hz over 8 ms.
    var t: f64[>..] = [];
    var i: i64 = 0;
    while i < 64 {
        t.push(i as f64 * 0.000125);
        i += 1;
    }
    let wave, wf = scientific::chirp(t, 30.0, 0.008, 10.0, "linear");
    assert(wf.kind == scientific::F_NONE);
    print("the chirp holds ", wave.data.len, " values\n");

    // The chirp z-transform with the default step is the Fourier transform.
    let zr, zi, zf = scientific::czt(wave.data);
    assert(zf.kind == scientific::F_NONE);
    print("the first value of the transform is ", zr.data[0], "\n");

    // A zoom into the band from 0.1 to 0.4 of the Nyquist rate.
    let ar, ai, af = scientific::zoom_fft(wave.data, [0.1, 0.4], 16, 2.0);
    assert(af.kind == scientific::F_NONE);
    print("the zoom holds ", ar.data.len, " values\n");

    // Two signals with a common part.
    var x: f64[>..] = [];
    var y: f64[>..] = [];
    i = 0;
    while i < 256 {
        let tri, tf = scientific::sawtooth([i as f64 * 0.1], 0.5);
        assert(tf.kind == scientific::F_NONE);
        x.push(tri.data[0]);
        y.push(tri.data[0] * 0.5 + (i % 7) as f64 * 0.01);
        i += 1;
    }
    let cf, cc, cf2 = scientific::coherence(x, y, 1.0, "hann", 0.0 / 0.0, [], 64, 32);
    assert(cf2.kind == scientific::F_NONE);
    print("the coherence holds ", cc.data.len, " values at the frequencies up to ", cf.data[cf.data.len - 1], "\n");

    // The lags of a full correlation of two signals.
    let lags, lf = scientific::correlation_lags(8, 5, "full");
    assert(lf.kind == scientific::F_NONE);
    print("the lags run from ", lags.data[0], " to ", lags.data[lags.data.len - 1], "\n");

    // The envelope of a wave.
    let env, res, ef = scientific::envelope(wave.data);
    assert(ef.kind == scientific::F_NONE);
    print("the envelope holds ", env.data.len, " values\n");

    // The peaks of a wave through the wavelet transform.
    var s: f64[>..] = [];
    i = 0;
    while i < 100 {
        let tri, tf = scientific::sawtooth([i as f64 * 0.2], 0.5);
        assert(tf.kind == scientific::F_NONE);
        s.push(tri.data[0]);
        i += 1;
    }
    let peaks, pf = scientific::find_peaks_cwt(s, [1.0, 2.0, 3.0, 4.0, 5.0]);
    assert(pf.kind == scientific::F_NONE);
    print("the wave holds ", peaks.data.len, " peaks\n");

    // A fault names its kind.
    let bad, bf = scientific::correlation_lags(5, 5, "wide");
    assert(bf.kind != scientific::F_NONE);
    print("a wrong mode gives the fault ", scientific::kind_name(bf.kind), "\n");
}
