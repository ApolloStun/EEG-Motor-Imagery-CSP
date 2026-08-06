"""
STEP 2 — Load the raw data and look at what we actually have.

WHY does this step exist, when we could jump straight to CSP?
Because a misunderstanding of the data NEVER shows up in the final accuracy:
the model runs, it prints a number, and the number is wrong for a reason no
metric reveals. Using the wrong runs (executed instead of imagined movement),
believing the sampling rate is 250 Hz when it is 160, or not knowing a third
event marker exists — each of those produces a pipeline that "works" and
conclusions that are false.

So here we model nothing. We open the files and VERIFY, against the real data,
every assumption written in config.py.

Outputs:
  results/01_raw_exploration.txt   text report of what we found
  figures/01_raw_signal.png        5 s of raw signal, 4 electrodes
  figures/01_montage.png           where the 64 electrodes physically sit
  figures/01_raw_psd.png           frequency content of the raw signal
"""

from pathlib import Path

import matplotlib

# IMPORTANT: must come BEFORE importing pyplot. "Agg" is a non-interactive
# backend that writes PNGs without opening a window. Without it, a script
# launched from a terminal can hang waiting for a window to be closed.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np

import config
# Data loading lives in preprocessing.py: it is shared with step 3 and beyond.
# See that file's header for why.
from preprocessing import load_runs

# MNE is very verbose by default (one line per filter, per file read...).
# We silence it so that our own report stays readable.
mne.set_log_level("ERROR")


def describe_run(run, raw):
    """Return the plain facts about one run, with no interpretation."""
    return {
        "run": run,
        "n_channels": len(raw.ch_names),
        "sfreq": raw.info["sfreq"],
        "duration_s": raw.n_times / raw.info["sfreq"],
        "n_samples": raw.n_times,
        "annotations": raw.annotations,
    }


# ==========================================================================
# FIGURES
# ==========================================================================
def plot_raw_signal(raw, seconds=5.0, start_s=10.0):
    """A few seconds of raw signal, on the electrodes we care about.

    We pick C3 / Cz / C4 because they sit over the motor cortex: C3 on the
    left, C4 on the right, Cz in the middle. That is where motor imagery has
    to show up. Fp1 is frontal, added as a control: it is where eye blinks hit
    hardest, so it shows what an artifact looks like next to real signal.
    """
    picks = ["C3", "Cz", "C4", "Fp1"]
    sfreq = raw.info["sfreq"]
    i0 = int(start_s * sfreq)
    i1 = int((start_s + seconds) * sfreq)

    # get_data() returns an (n_channels, n_samples) array in VOLTS.
    # We multiply by 1e6 to display microvolts (uV), the usual EEG unit:
    # scalp brain activity is a few tens of uV.
    data = raw.get_data(picks=picks, start=i0, stop=i1) * 1e6
    times = np.arange(data.shape[1]) / sfreq + start_s

    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 6), sharex=True)
    for ax, name, trace in zip(axes, picks, data):
        ax.plot(times, trace, lw=0.8, color="#1f4e79")
        ax.set_ylabel(f"{name}\n(uV)", fontsize=9)
        ax.axhline(0, color="0.8", lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(
        f"RAW EEG signal — subject {config.SUBJECT}, {seconds:.0f} s\n"
        "no filtering, no processing of any kind",
        fontsize=11,
    )
    fig.tight_layout()
    out = config.FIGURES_DIR / "01_raw_signal.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_montage(raw):
    """Where the electrodes physically sit on the head.

    The "montage" is the mapping from electrode name to 3D position. It is not
    stored in the EDF file: we add it from a standard model (standard_1005).
    It is indispensable for any topographic figure, and therefore for
    visualising the CSP filters in step 5.
    """
    # The montage is already set by preprocessing.load_runs().
    fig, ax = plt.subplots(figsize=(6, 6))
    raw.plot_sensors(show_names=True, axes=ax, show=False)
    ax.set_title(
        f"64 electrodes (10-10 system)\nsubject {config.SUBJECT}", fontsize=11
    )
    fig.tight_layout()
    out = config.FIGURES_DIR / "01_montage.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_raw_psd(raw):
    """Frequency content of the raw signal (power spectral density).

    An EEG signal is a sum of oscillations. The PSD answers: "how much power at
    each frequency?". This is THE figure that will justify the filtering in
    step 3 — it shows what we want to keep and what we want to throw away.
    """
    picks = ["C3", "C4", "Fp1"]
    # fmax=79: nothing can be said above half the sampling rate (Nyquist
    # theorem). At 160 Hz, the limit is 80 Hz.
    spectrum = raw.compute_psd(fmin=1.0, fmax=79.0, picks=picks)
    psds, freqs = spectrum.get_data(return_freqs=True)

    # In dB (log scale): without it, the low frequencies — vastly more powerful
    # — visually crush the rest of the spectrum.
    psds_db = 10 * np.log10(psds * 1e12)  # V^2/Hz -> uV^2/Hz, then dB

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, curve in zip(picks, psds_db):
        ax.plot(freqs, curve, lw=1.3, label=name)

    # Highlight the band we will keep in step 3.
    ax.axvspan(
        config.F_LOW, config.F_HIGH, color="#2e8b57", alpha=0.12,
        label=f"band kept: {config.F_LOW:.0f}-{config.F_HIGH:.0f} Hz",
    )
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (dB, uV^2/Hz)")
    ax.set_title(
        f"Spectrum of the RAW signal — subject {config.SUBJECT}\n"
        "everything outside the green band will be removed in step 3",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = config.FIGURES_DIR / "01_raw_psd.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ==========================================================================
# REPORT
# ==========================================================================
def main():
    lines = []

    def say(text=""):
        """Print AND remember, so we can write the report into results/."""
        print(text)
        lines.append(text)

    say("=" * 74)
    say(f"RAW SIGNAL EXPLORATION — subject {config.SUBJECT}, runs {config.RUNS}")
    say("=" * 74)

    raws = load_runs(config.SUBJECT, config.RUNS)

    say()
    say("--- FILES ---")
    total_mb = 0.0
    for _, raw in raws:
        # raw.filenames keeps track of the source file the signal came from.
        path = Path(raw.filenames[0])
        size_mb = path.stat().st_size / 1e6
        total_mb += size_mb
        say(f"  {path.name}   {size_mb:5.1f} MB")
    say(f"  total: {total_mb:.1f} MB")

    say()
    say("--- SIGNAL CHARACTERISTICS (per run) ---")
    for run, raw in raws:
        d = describe_run(run, raw)
        say(
            f"  run {d['run']:>2}: {d['n_channels']} channels | "
            f"{d['sfreq']:.0f} Hz | {d['duration_s']:.1f} s | "
            f"{d['n_samples']} samples/channel"
        )

    # All runs must share the same sampling rate, otherwise they cannot be
    # concatenated. We check it rather than assume it.
    sfreqs = {raw.info["sfreq"] for _, raw in raws}
    say(f"  -> single sampling rate across all runs: {sfreqs}")

    first_raw = raws[0][1]
    say()
    say("--- THE 64 ELECTRODES (cleaned names) ---")
    names = first_raw.ch_names
    for i in range(0, len(names), 8):
        say("  " + "  ".join(f"{n:>5}" for n in names[i : i + 8]))

    motor = [n for n in names if n in ("C3", "Cz", "C4", "CP3", "CP4", "FC3", "FC4")]
    say(f"  electrodes over the motor cortex: {motor}")

    say()
    say("--- ACTUAL ANNOTATIONS (the event markers) ---")
    for run, raw in raws:
        descs, counts = np.unique(raw.annotations.description, return_counts=True)
        detail = ", ".join(f"{d}={c}" for d, c in zip(descs, counts))
        say(f"  run {run:>2}: {len(raw.annotations)} annotations  ->  {detail}")

    # Typical duration of each event type: this determines the epoching window
    # of step 4. We measure it, we do not guess it.
    say()
    say("  mean duration of each event type:")
    all_desc = np.concatenate([raw.annotations.description for _, raw in raws])
    all_dur = np.concatenate([raw.annotations.duration for _, raw in raws])
    for d in sorted(set(all_desc)):
        durs = all_dur[all_desc == d]
        say(f"    {d}: {durs.mean():.2f} s  (min {durs.min():.2f}, max {durs.max():.2f})")

    # Check the assumption written in config.EVENT_ID.
    # events_from_annotations() converts the text labels (T0/T1/T2) into integer
    # codes, in alphabetical order. Our config assumes T1 -> 2 and T2 -> 3:
    # we confirm it here, on the real data.
    say()
    say("--- CHECKING THE EVENT -> CODE MAPPING ---")
    concat = mne.concatenate_raws([raw.copy() for _, raw in raws])
    events, event_id_found = mne.events_from_annotations(concat)
    say(f"  mapping found by MNE:        {dict((str(k), v) for k, v in event_id_found.items())}")
    say(f"  mapping assumed in config:   {config.EVENT_ID}")
    ok = all(event_id_found.get(f"T{i}") == code
             for i, code in ((1, config.EVENT_ID["left_hand"]),
                             (2, config.EVENT_ID["right_hand"])))
    say(f"  -> config.EVENT_ID is {'CORRECT' if ok else 'WRONG — fix it!'}")

    say()
    say("  number of events per code, across the 3 concatenated runs:")
    inverse = {v: k for k, v in event_id_found.items()}
    codes, counts = np.unique(events[:, 2], return_counts=True)
    for code, count in zip(codes, counts):
        say(f"    code {code} ({inverse[code]:>2}): {count:>3} events")

    say()
    say(f"  total concatenated duration: {concat.n_times / concat.info['sfreq']:.1f} s")

    say()
    say("--- FIGURES WRITTEN ---")
    for out in (
        plot_raw_signal(first_raw),
        plot_montage(first_raw),
        plot_raw_psd(concat),
    ):
        say(f"  {out.relative_to(config.ROOT_DIR)}")

    report = config.RESULTS_DIR / "01_raw_exploration.txt"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nReport written: {report.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
