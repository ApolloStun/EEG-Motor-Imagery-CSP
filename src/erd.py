"""
Quantifying ERD (event-related desynchronization) from epoched data.

WHY this file appeared at step 7, like preprocessing.py at step 3 and model.py at
step 6. Step 4 wrote these functions to look at one subject. Step 7 needs exactly
the same measurements on ten subjects. Third application of the same rule: factor
out on the second occurrence.

Everything here answers one question — "how much did band power change relative to
rest, and where?" — and nothing here touches classification.
"""

import numpy as np

import config

# Length of the sliding window used to smooth instantaneous power, in seconds.
# Squaring a band-passed signal gives a very jagged power estimate (it wiggles at
# twice the signal frequency). Averaging over ~0.4 s gives a readable envelope
# while keeping enough time resolution to see when ERD starts.
SMOOTH_S = 0.4


def band_power_envelope(data, sfreq, smooth_s=SMOOTH_S):
    """Turn a band-passed signal into a smooth power-over-time curve.

    `data` is (..., n_times) in volts. Squaring gives instantaneous power; a
    moving average smooths it. This is deliberately the simplest possible
    estimator, and it is exactly the quantity CSP + LDA end up using: the
    variance of a band-passed signal IS band power. Nothing hidden.
    """
    power = data ** 2
    win = max(1, int(round(smooth_s * sfreq)))
    win += 1 - win % 2          # force an odd length so padding stays symmetric
    kernel = np.ones(win) / win
    pad = win // 2

    def smooth(x):
        # Naive np.convolve(mode="same") implicitly pads with ZEROS, which drags
        # the average down at both ends and creates fake dips at the edges of
        # every epoch. We saw exactly that on the first version of the step-4
        # figure. Padding with the edge value instead keeps the ends honest.
        return np.convolve(np.pad(x, pad, mode="edge"), kernel, mode="valid")

    return np.apply_along_axis(smooth, -1, power)


def erd_percent(epochs, picks, baseline=(None, 0.0)):
    """ERD as a percentage change relative to the pre-cue rest baseline.

    ERD% = 100 * (power(t) - rest_power) / rest_power

    A NEGATIVE value means the oscillation shrank, i.e. the cortex activated.
    Expressing it as a percentage is what makes electrodes and subjects
    comparable: absolute uV^2 values depend on skull thickness, electrode
    impedance and gel, none of which is neuroscience. That comparability is
    exactly what step 7 relies on.

    IMPORTANT DETAIL — which baseline we divide by. The obvious version divides
    each trial by its OWN pre-cue power. We do not, because EEG power in a 1 s
    window is very unstable: whenever a trial happens to have a quiet baseline,
    dividing by that small number blows the whole trial up, and the average ends
    up dominated by a handful of trials with weak baselines. Instead we divide
    every trial by the baseline power AVERAGED OVER ALL TRIALS of that class
    (per channel). This is the classical ERD/ERS computation.
    """
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data(picks=picks)          # (n_epochs, n_picks, n_times)
    envelope = band_power_envelope(data, sfreq)
    times = epochs.times

    b0 = times[0] if baseline[0] is None else baseline[0]
    b1 = baseline[1]
    bmask = (times >= b0) & (times < b1)
    # Mean baseline power per channel, across trials AND across baseline time.
    # Shape (1, n_picks, 1) so it broadcasts over trials and time.
    rest = envelope[:, :, bmask].mean(axis=(0, 2))[np.newaxis, :, np.newaxis]

    return 100.0 * (envelope - rest) / rest, times


def window_mean(epochs, picks, tmin=None, tmax=None):
    """Mean ERD% per channel over the classifier window, averaged across trials.

    Returns a 1-D array, one value per requested channel.
    """
    tmin = config.CSP_TMIN if tmin is None else tmin
    tmax = config.CSP_TMAX if tmax is None else tmax
    erd, times = erd_percent(epochs, picks=picks)
    mask = (times >= tmin) & (times <= tmax)
    return erd[:, :, mask].mean(axis=-1).mean(axis=0)


def lateralisation(epochs):
    """Contralateral-ERD check at C3 / C4, returned as numbers rather than a verdict.

    Motor control crosses over, so imagining the LEFT hand should drop power more
    at C4 (right hemisphere) and the RIGHT hand more at C3.

    LI = ERD(C3) - ERD(C4), in percentage points:
        left hand  -> LI should be POSITIVE (C4 drops more)
        right hand -> LI should be NEGATIVE (C3 drops more)
    The contrast LI(left) - LI(right) should therefore be clearly positive.

    A boolean "does it match theory?" is not good enough: a 0.3-point difference
    and a 30-point difference would both come out True. We return the margins and
    let the caller apply a noise floor.
    """
    out = {}
    for cls in ("left_hand", "right_hand"):
        c3, c4 = window_mean(epochs[cls], picks=["C3", "C4"])
        out[cls] = {"C3": float(c3), "C4": float(c4), "LI": float(c3 - c4)}
    out["contrast"] = out["left_hand"]["LI"] - out["right_hand"]["LI"]
    return out


def region_contrast(epochs):
    """Where on the scalp does the between-class difference sit?

    A classifier does not care about ERD; it cares about any difference between
    the classes. So the question that matters is not "is there ERD?" but "where
    does the difference live?". If it is largest over the sensorimotor strip we
    are decoding motor imagery; if it is largest posteriorly, we are decoding
    something else. Step 4 found the latter for subject 1 — step 7 asks whether
    that generalises.

    Returns mean |left - right| over each region, their ratio, and the single
    strongest electrode.
    """
    maps = {cls: window_mean(epochs[cls], picks="eeg")
            for cls in ("left_hand", "right_hand")}
    diff = maps["left_hand"] - maps["right_hand"]

    names = epochs.copy().pick("eeg").ch_names
    sm_idx = [names.index(ch) for ch in config.SENSORIMOTOR_CH if ch in names]
    po_idx = [names.index(ch) for ch in config.POSTERIOR_CH if ch in names]

    sm = float(np.abs(diff[sm_idx]).mean())
    po = float(np.abs(diff[po_idx]).mean())
    peak = names[int(np.argmax(np.abs(diff)))]

    return {
        "sensorimotor": sm,
        "posterior": po,
        # How many times larger the posterior difference is. > 1 is the red flag.
        "posterior_over_motor": po / sm if sm else float("nan"),
        "peak_channel": peak,
        "peak_value": float(diff[int(np.argmax(np.abs(diff)))]),
        "diff_map": diff,
    }
