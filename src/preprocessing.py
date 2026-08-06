"""
Loading, filtering and epoching — the code shared by every step.

WHY does this file appear now, and not at step 1?
Because at step 1 there was nothing to share. At step 3, a SECOND script needed
to load the same data as the first. That is the right moment to extract the
common code: not earlier (we would have invented an abstraction for a single
use), not later (we would already have three diverging copies of the same
function, and fixing a bug in only one of them).

Useful rule: factor out on the second occurrence, not the first.
"""

import mne

import config

mne.set_log_level("ERROR")


def load_runs(subject=None, runs=None, set_montage=True):
    """Download (if needed) and read a subject's runs, WITHOUT filtering.

    Returns a list of (run_number, Raw) tuples.
    The runs are kept SEPARATE and deliberately not concatenated here: see
    load_filtered() below for why that matters.
    """
    subject = config.SUBJECT if subject is None else subject
    runs = config.RUNS if runs is None else runs

    # load_data() handles both the download AND the cache: on the first call it
    # fetches the files from PhysioNet, afterwards it just returns local paths.
    paths = mne.datasets.eegbci.load_data(
        subject=subject, runs=runs, path=str(config.DATA_DIR), update_path=False
    )

    out = []
    for path, run in zip(paths, runs):
        # preload=True: the whole signal in RAM. These files are small (2.6 MB),
        # and filtering requires it anyway.
        raw = mne.io.read_raw_edf(path, preload=True)

        # Channel names in these EDF files are non-standard ("Fc5.", "Cz..").
        # standardize() maps them back to the international 10-05 nomenclature,
        # without which we cannot attach a position on the head to them.
        mne.datasets.eegbci.standardize(raw)

        if set_montage:
            # The montage = mapping from electrode name to 3D position. It is
            # not stored in the EDF file, so we add it from a standard model.
            # Required as soon as we want a topographic figure (the CSP maps in
            # step 5, the ERD maps in step 4).
            raw.set_montage("standard_1005", match_case=False)

        out.append((run, raw))

    return out


def bandpass(raw, l_freq=None, h_freq=None):
    """Apply the band-pass filter. Modifies `raw` in place and returns it.

    Defaults come from config.py (8-30 Hz) and are justified in detail in
    02_filter.py and in METHODOLOGY.md.

    Two technical details that matter:

    method="fir" + phase="zero" (MNE default): the filter is applied in a
    "zero-phase" way, meaning it introduces NO time shift. That is essential
    here: we are about to cut epochs aligned on an event, so a filter that
    displaced the signal by 50 ms would misalign the brain response with
    respect to its trigger.

    fir_design="firwin": how the filter is built. This is MNE's modern default,
    stated explicitly so the code stays reproducible even if the default
    changes in a future version.
    """
    l_freq = config.F_LOW if l_freq is None else l_freq
    h_freq = config.F_HIGH if h_freq is None else h_freq

    return raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        phase="zero",
    )


def load_filtered(subject=None, runs=None):
    """Load, filter EACH RUN SEPARATELY, then concatenate. Returns one Raw.

    The order of operations is the whole point of this function.

    A digital filter computes each output sample from its neighbours in time.
    If we concatenate first, the 3 runs are glued end to end and there is an
    abrupt discontinuity at each junction (the signal jumps from one arbitrary
    value to another). The filter "sees" that jump as a very fast oscillation
    and smears it over a few hundred milliseconds on both sides — creating an
    artifact where there was none.

    By filtering each run before concatenating, each run is treated as a clean,
    independent recording. This is the same reasoning that makes us filter
    BEFORE cutting epochs (step 4) and never after: a filter needs temporal
    context around every sample, and a 5-second epoch offers almost none.
    """
    raws = [bandpass(raw) for _, raw in load_runs(subject, runs)]
    return mne.concatenate_raws(raws)


def make_epochs(raw, tmin=None, tmax=None, event_id=None):
    """Cut the continuous signal into trials aligned on the cues.

    baseline=None is a deliberate choice, explained in 03_epochs.py: the
    features we will feed the classifier are band POWER (a variance), not
    evoked amplitudes, and subtracting a per-epoch constant does not change a
    variance. Applying a baseline here would suggest we are doing evoked-
    potential analysis, which we are not.

    We keep all 64 EEG channels and let CSP decide which combinations matter —
    that is precisely CSP's job. The cost of that choice (64x64 covariance
    matrices estimated from few trials) is discussed in step 6.
    """
    tmin = config.EPOCH_TMIN if tmin is None else tmin
    tmax = config.EPOCH_TMAX if tmax is None else tmax
    event_id = config.EVENT_ID if event_id is None else event_id

    events, _ = mne.events_from_annotations(raw)

    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks="eeg",
        preload=True,
        # No automatic rejection threshold: with only ~45 trials per subject we
        # cannot afford to silently drop trials. Data quality is inspected
        # explicitly in step 4 instead of being handled by a hidden default.
        reject=None,
        verbose=False,
    )


def describe_filter(sfreq=None, l_freq=None, h_freq=None):
    """Return the real characteristics of the filter MNE is going to build.

    Useful for the report: we document the filter actually applied rather than
    the one we think we asked for.
    """
    sfreq = 160.0 if sfreq is None else sfreq
    l_freq = config.F_LOW if l_freq is None else l_freq
    h_freq = config.F_HIGH if h_freq is None else h_freq

    coefs = mne.filter.create_filter(
        data=None,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        phase="zero",
    )
    return {
        "n_taps": len(coefs),
        "length_s": len(coefs) / sfreq,
        "l_freq": l_freq,
        "h_freq": h_freq,
        "sfreq": sfreq,
    }
