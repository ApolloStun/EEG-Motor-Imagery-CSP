"""
STEP 3 — Filter the signal, and know exactly what we are throwing away.

=========================================================================
WHY KEEP 8-30 Hz?
=========================================================================

Three reasons, in order of importance.

1) BECAUSE THAT IS WHERE THE SIGNAL WE ARE LOOKING FOR LIVES.
   The sensorimotor cortex oscillates spontaneously at rest, in two bands: the
   mu rhythm (8-12 Hz) and the beta rhythm (13-30 Hz). Those oscillations are a
   standby state: neurons in the motor area are synchronised, hence coordinated
   but idle. When you imagine moving your hand, that synchronisation BREAKS on
   the opposite side of the brain (left hand -> right cortex): power in mu and
   beta drops locally. This is ERD, event-related desynchronization.
   In other words: the useful signal is not a peak that appears, it is an
   oscillation that DISAPPEARS, on one side only. That left/right asymmetry is
   exactly what CSP will learn to measure (step 5).

2) BECAUSE CSP WORKS ON VARIANCE, AND VARIANCE IS DOMINATED BY LOW FREQUENCIES.
   This is the most technical and the most important reason. CSP looks for the
   combinations of electrodes that maximise the variance ratio between the two
   classes. But the variance of a raw EEG signal is crushed by slow drifts
   (< 4 Hz): sweating, electrode impedance, breathing, movement. This script
   measures the exact proportion.
   If we handed the raw signal to CSP, it would therefore optimise mostly on
   drift noise — not on motor activity. Filtering is not cosmetic cleanup: it is
   what makes CSP applicable at all.

3) BECAUSE IT REMOVES TWO SOURCES OF CONTAMINATION FOR FREE.
   - Below: eye movements (essentially < 4 Hz) and electrode noise.
   - Above: muscle activity (EMG), which grows in power from about 20 Hz and
     dominates beyond 30, plus the mains spike at 60 Hz seen in step 2.
   On a motor-imagery dataset, removing EMG has a specific methodological
   value: it is the guarantee that we are not classifying a micro muscle
   contraction instead of an intention.

=========================================================================
WHAT THIS FILTER COSTS US — to be owned, not hidden
=========================================================================

Filtering throws information away irreversibly. Two real losses:

- SLOW MOTOR POTENTIALS (< 3 Hz). There is a very-low-frequency cortical motor
  response (MRCP / readiness potential) that carries genuine information about
  movement. Our 8 Hz high-pass removes it entirely. This is not trivial for the
  rest of the project: the literature on decoding *attempted* movements (level 2,
  see README) relies precisely on low-frequency EEG, around 0.3-3 Hz. So the
  band we discard here is the one we would have to recover the day we take on
  attempted movement. This is a choice suited to level 3, not a universal truth.

- THE GAMMA BAND (> 30 Hz). Largely a theoretical loss: through the skull,
  scalp gamma is hard to separate from EMG, and on this dataset sampled at
  160 Hz nothing can be said above 80 Hz anyway.

Outputs:
  results/02_filtering.txt            before/after measurements
  figures/02_before_after_signal.png  the same signal, raw then filtered
  figures/02_before_after_psd.png     the same spectrum, raw then filtered
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np

import config
import preprocessing

mne.set_log_level("ERROR")

# The whole demonstration runs on ONE run (the first), not on the 3
# concatenated. Reason: a clean before/after comparison requires both signals to
# be strictly the same data, and concatenation introduces a discontinuity that
# we have just explained perturbs filtering (see preprocessing.load_filtered).
# 125 s of signal is plenty for a spectrum estimate.
DEMO_CHANNEL = "C3"


def band_power_fractions(raw, bands):
    """What fraction of total power sits in each band?

    This is the measurement that justifies filtering with a number, instead of
    just asserting "low frequencies dominate".
    """
    spectrum = raw.compute_psd(fmin=0.5, fmax=79.0, picks="eeg")
    psds, freqs = spectrum.get_data(return_freqs=True)

    # Power ~ integral of the PSD over the band. We sum over frequencies, then
    # average over the 64 electrodes.
    total = psds.sum(axis=1).mean()
    out = {}
    for label, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[label] = psds[:, mask].sum(axis=1).mean() / total
    return out


def psd_at(raw, freq, picks):
    """Power at one precise frequency, in dB. Used to measure attenuation."""
    spectrum = raw.compute_psd(fmin=1.0, fmax=79.0, picks=picks)
    psds, freqs = spectrum.get_data(return_freqs=True)
    i = int(np.argmin(np.abs(freqs - freq)))
    return 10 * np.log10(psds[:, i].mean() * 1e12), freqs[i]


def plot_before_after_signal(raw_unf, raw_filt, seconds=5.0, start_s=10.0):
    """The same stretch of signal, before and after filtering.

    The two panels deliberately have their OWN vertical scale. On a shared
    scale the filtered signal would look almost flat — true but useless. Here
    we want to see its SHAPE. The amplitude difference itself is written in
    each panel title.
    """
    sfreq = raw_unf.info["sfreq"]
    i0, i1 = int(start_s * sfreq), int((start_s + seconds) * sfreq)

    d_unf = raw_unf.get_data(picks=[DEMO_CHANNEL], start=i0, stop=i1)[0] * 1e6
    d_flt = raw_filt.get_data(picks=[DEMO_CHANNEL], start=i0, stop=i1)[0] * 1e6
    times = np.arange(len(d_unf)) / sfreq + start_s

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(times, d_unf, lw=0.8, color="#8c8c8c")
    axes[0].set_title(
        f"BEFORE — raw signal ({DEMO_CHANNEL}) · typical amplitude "
        f"{d_unf.std():.1f} uV (std)",
        fontsize=10,
    )
    axes[1].plot(times, d_flt, lw=0.9, color="#1f4e79")
    axes[1].set_title(
        f"AFTER — band-passed {config.F_LOW:.0f}-{config.F_HIGH:.0f} Hz · "
        f"typical amplitude {d_flt.std():.1f} uV (std)",
        fontsize=10,
    )

    for ax in axes:
        ax.set_ylabel("uV")
        ax.axhline(0, color="0.85", lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("time (s)")

    fig.suptitle(
        f"Effect of filtering — subject {config.SUBJECT}, run {config.RUNS[0]}, "
        f"electrode {DEMO_CHANNEL}",
        fontsize=11,
    )
    fig.tight_layout()
    out = config.FIGURES_DIR / "02_before_after_signal.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_before_after_psd(raw_unf, raw_filt):
    """The spectrum before and after. This is where the filter really shows.

    Worth noticing: the edges are not vertical. A digital filter has
    "transition bands" — it attenuates progressively, not instantly. That is a
    mathematical constraint, not an implementation flaw: the steeper the
    transition, the longer the filter must be in time, and the more surrounding
    signal it needs as context for every sample.
    """
    picks = [DEMO_CHANNEL]
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for raw, label, color in (
        (raw_unf, "raw", "#8c8c8c"),
        (raw_filt, f"band-passed {config.F_LOW:.0f}-{config.F_HIGH:.0f} Hz", "#1f4e79"),
    ):
        spectrum = raw.compute_psd(fmin=1.0, fmax=79.0, picks=picks)
        psds, freqs = spectrum.get_data(return_freqs=True)
        ax.plot(freqs, 10 * np.log10(psds[0] * 1e12), lw=1.4, color=color, label=label)

    ax.axvspan(config.F_LOW, config.F_HIGH, color="#2e8b57", alpha=0.10,
               label="band kept")
    ax.axvline(60, color="#c0392b", lw=0.9, ls=":", label="60 Hz mains")

    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (dB, uV^2/Hz)")
    ax.set_title(
        f"Spectrum before / after filtering — electrode {DEMO_CHANNEL}\n"
        "the filter edges are gradual, not vertical",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = config.FIGURES_DIR / "02_before_after_psd.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    say("=" * 74)
    say(f"BAND-PASS FILTERING {config.F_LOW:.0f}-{config.F_HIGH:.0f} Hz — "
        f"subject {config.SUBJECT}, run {config.RUNS[0]}")
    say("=" * 74)

    # One run, loaded once: a copy stays unfiltered for comparison.
    runs = preprocessing.load_runs(config.SUBJECT, [config.RUNS[0]])
    raw_unf = runs[0][1]
    raw_filt = preprocessing.bandpass(raw_unf.copy())

    # ------------------------------------------------------------------
    say()
    say("--- THE FILTER MNE ACTUALLY BUILDS ---")
    info = preprocessing.describe_filter(sfreq=raw_unf.info["sfreq"])
    say(f"  band-pass          : {info['l_freq']:.0f}-{info['h_freq']:.0f} Hz")
    say("  type               : FIR, zero-phase (no time shift at all)")
    say(f"  length             : {info['n_taps']} coefficients "
        f"= {info['length_s']:.2f} s of signal")
    say(f"  -> every output sample is computed from {info['length_s']:.2f} s of")
    say("     context around it. That is WHY we filter the continuous signal")
    say("     before cutting it into epochs, and never the other way round: a")
    say("     5 s epoch does not provide enough context, and its edges would")
    say("     be distorted.")

    # ------------------------------------------------------------------
    say()
    say("--- WHERE IS THE POWER IN THE RAW SIGNAL? (mean over 64 channels) ---")
    bands = {
        "0.5-4 Hz   (drift, delta)": (0.5, 4),
        "4-8 Hz     (theta)": (4, 8),
        "8-12 Hz    (mu)     <- useful": (8, 12),
        "13-30 Hz   (beta)   <- useful": (13, 30),
        "30-79 Hz   (EMG, mains)": (30, 79),
    }
    fractions = band_power_fractions(raw_unf, bands)
    for label, frac in fractions.items():
        bar = "#" * int(round(frac * 50))
        say(f"  {label:<32} {frac * 100:5.1f} %  {bar}")

    useful = (fractions["8-12 Hz    (mu)     <- useful"]
              + fractions["13-30 Hz   (beta)   <- useful"])
    below = fractions["0.5-4 Hz   (drift, delta)"] + fractions["4-8 Hz     (theta)"]
    say()
    say(f"  -> useful band (8-30 Hz) : {useful * 100:.1f} % of total power")
    say(f"  -> below 8 Hz            : {below * 100:.1f} %")
    say("  This is THE number that justifies filtering: without it, the variance")
    say("  CSP tries to maximise would be dominated by slow drift, and it would")
    say("  optimise on noise instead of motor activity.")

    # ------------------------------------------------------------------
    say()
    say("--- SANITY CHECK: DOES THE FILTER DO WHAT WE ASKED? ---")
    for freq in (2.0, 5.0, 10.0, 20.0, 40.0, 60.0):
        p_before, f_actual = psd_at(raw_unf, freq, [DEMO_CHANNEL])
        p_after, _ = psd_at(raw_filt, freq, [DEMO_CHANNEL])
        delta = p_after - p_before
        zone = "kept" if config.F_LOW <= freq <= config.F_HIGH else "rejected"
        say(f"  {f_actual:5.1f} Hz ({zone:>8}): {delta:+7.1f} dB")
    say("  (0 dB = untouched ; -30 dB = power divided by 1000)")
    say("  The mains spike at 60 Hz, clearly visible in step 2, is gone.")

    # ------------------------------------------------------------------
    say()
    say("--- SIGNAL AMPLITUDE, BEFORE AND AFTER ---")
    std_unf = raw_unf.get_data(picks="eeg").std() * 1e6
    std_flt = raw_filt.get_data(picks="eeg").std() * 1e6
    say(f"  raw      : {std_unf:5.1f} uV (std, all channels)")
    say(f"  filtered : {std_flt:5.1f} uV")
    say(f"  -> amplitude divided by {std_unf / std_flt:.1f}.")
    say("     This is NOT a loss of useful signal: it is the disappearance of")
    say("     the slow drift that was inflating amplitude without carrying any")
    say("     information about motor intent.")

    # ------------------------------------------------------------------
    say()
    say("--- FIGURES WRITTEN ---")
    for out in (
        plot_before_after_signal(raw_unf, raw_filt),
        plot_before_after_psd(raw_unf, raw_filt),
    ):
        say(f"  {out.relative_to(config.ROOT_DIR)}")

    report = config.RESULTS_DIR / "02_filtering.txt"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nReport written: {report.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
