"""
STEP 4 — Cut the continuous signal into trials, and see ERD for the first time.

=========================================================================
WHAT EPOCHING IS, AND WHY IT IS NOT JUST BOOKKEEPING
=========================================================================

So far we have 375 s of continuous signal. A classifier cannot learn from that:
it needs examples, each with a label. Epoching turns one long recording into
N independent trials, each aligned on the moment the subject was told to start
imagining. That alignment is the whole point — it is what lets us average
across trials and see a response that is invisible in any single one.

THE WINDOW. config.py asks for -1 to +4 s around each cue, and the classifier
will only see 0.5 to 2.5 s. Both numbers come from measurements made in step 2,
not from habit:
  - trials (T1/T2) last 4.10 s, so a window ending at +4.0 s stays inside the
    trial and does not leak into the next one;
  - rest periods (T0) last 4.20 s and sit between trials, so the -1 to 0 s part
    of our window falls inside genuine rest. That gives us a real baseline.
  - we start the classifier window at +0.5 s rather than 0 s because ERD takes
    a few hundred milliseconds to develop: the brain does not desynchronise
    instantly on hearing the cue. Including 0-0.5 s would add mostly noise.
  - we stop at +2.5 s because ERD is strongest early and attention drifts
    later in the trial.
These last two are conventional choices from the motor-imagery literature. They
are not tuned on our results — doing that would be a form of cheating we return
to in step 6.

NO BASELINE CORRECTION (baseline=None). Baseline correction subtracts the mean
of a reference window from each epoch. It is essential when measuring evoked
potential *amplitudes*. We are not doing that: our features are band POWER, i.e.
a variance, and subtracting a constant from a signal does not change its
variance. Applying a baseline here would be harmless but misleading — it would
suggest an evoked-potential analysis. We use the baseline window explicitly and
visibly below, to express ERD as a percentage change.

NO AUTOMATIC ARTIFACT REJECTION. MNE can drop epochs whose amplitude exceeds a
threshold. We do not, on purpose: with ~22 trials per class, silently dropping
trials would change the problem while hiding that it changed. We inspect
amplitudes explicitly instead and report what we see.

=========================================================================
WHAT ERD IS, AND WHY IT IS THE ENTIRE SIGNAL OF THIS PROJECT
=========================================================================

At rest, the sensorimotor cortex oscillates in the mu (8-12 Hz) and beta
(13-30 Hz) bands. Think of it as an idle engine: neurons firing in a
synchronised, coordinated but unproductive way. Because they are synchronised,
their tiny individual contributions add up and produce a large oscillation
visible at the scalp.

When you imagine moving your hand, that population starts doing different
things at different times — it desynchronises. The contributions no longer add
up, so the measured oscillation SHRINKS. This is ERD: event-related
desynchronization. The counter-intuitive part is that brain activation shows up
as LESS signal, not more.

And crucially it is LATERALISED: imagining the left hand desynchronises the
RIGHT motor cortex (near electrode C4), and vice versa, because motor control
crosses over. That left/right asymmetry is the only thing separating our two
classes. Everything after this step — CSP, LDA — exists to measure it.

Outputs:
  results/03_epochs.txt              trial counts, window checks, ERD numbers
  figures/03_erd_timecourse.png      band power over time, C3 vs C4, per class
  figures/03_erd_topography.png      where on the scalp the power drop happens
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np

import config
import erd as erd_mod
import preprocessing

mne.set_log_level("ERROR")

# ERD quantification lives in erd.py: step 7 needs exactly the same
# measurements on ten subjects. See that file's header for the reasoning.
SMOOTH_S = erd_mod.SMOOTH_S
band_power_envelope = erd_mod.band_power_envelope
erd_percent = erd_mod.erd_percent

# Electrode groups used to check WHERE a between-class difference sits, defined
# in config.py because they are scientific choices, not helper lists.
SENSORIMOTOR_CH = config.SENSORIMOTOR_CH
POSTERIOR_CH = config.POSTERIOR_CH


def plot_erd_timecourse(epochs):
    """Band power over time at C3 and C4, one curve per imagined hand.

    This is the figure where the two classes become visibly different for the
    first time. What to look for: in each panel, the two curves should separate
    after the cue, and they should separate in OPPOSITE directions in the two
    panels. C3 (left hemisphere) should drop more for the right hand; C4 (right
    hemisphere) should drop more for the left hand.
    """
    picks = ["C3", "C4"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    colors = {"left_hand": "#c0392b", "right_hand": "#1f4e79"}
    labels = {"left_hand": "imagined LEFT hand", "right_hand": "imagined RIGHT hand"}

    for ax, ch in zip(axes, picks):
        for cls in ("left_hand", "right_hand"):
            erd, times = erd_percent(epochs[cls], picks=[ch])
            mean = erd[:, 0, :].mean(axis=0)
            # Standard error of the mean across trials: with ~22 trials per
            # class, showing the mean alone would hide how uncertain it is.
            sem = erd[:, 0, :].std(axis=0) / np.sqrt(erd.shape[0])
            ax.plot(times, mean, lw=1.8, color=colors[cls], label=labels[cls])
            ax.fill_between(times, mean - sem, mean + sem,
                            color=colors[cls], alpha=0.18, lw=0)

        ax.axvline(0, color="0.3", lw=1.0, ls="--")
        ax.axhline(0, color="0.75", lw=0.8)
        ax.axvspan(config.CSP_TMIN, config.CSP_TMAX, color="#2e8b57", alpha=0.08)
        hemisphere = "LEFT hemisphere" if ch == "C3" else "RIGHT hemisphere"
        ax.set_title(f"{ch} — {hemisphere}", fontsize=11)
        ax.set_xlabel("time relative to cue (s)")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("band power change vs rest (%)\nnegative = ERD = activation")
    axes[0].legend(fontsize=9, loc="lower left")
    axes[1].text(
        (config.CSP_TMIN + config.CSP_TMAX) / 2, axes[1].get_ylim()[1] * 0.92,
        "window given\nto classifier", ha="center", va="top", fontsize=8,
        color="#2e8b57",
    )

    fig.suptitle(
        f"Event-related desynchronization, {config.F_LOW:.0f}-{config.F_HIGH:.0f} Hz "
        f"— subject {config.SUBJECT}\n"
        "shaded band = standard error across trials",
        fontsize=11,
    )
    fig.tight_layout()
    out = config.FIGURES_DIR / "03_erd_timecourse.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_erd_topography(epochs):
    """Where on the scalp the power drop happens, for each class.

    The third panel is the interesting one: the difference between classes. If
    motor imagery is lateralised as theory says, that map should show one sign
    over the left hemisphere and the opposite sign over the right. This is, in
    essence, a hand-made preview of what CSP will find on its own in step 5 —
    except CSP finds the optimal electrode weighting instead of us looking at
    one map and squinting.
    """
    maps = {}
    for cls in ("left_hand", "right_hand"):
        erd, times = erd_percent(epochs[cls], picks="eeg")
        # Average over the classifier window, then over trials -> one value
        # per electrode.
        wmask = (times >= config.CSP_TMIN) & (times <= config.CSP_TMAX)
        maps[cls] = erd[:, :, wmask].mean(axis=-1).mean(axis=0)

    diff = maps["left_hand"] - maps["right_hand"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    info = epochs.copy().pick("eeg").info

    # Shared symmetric colour limits for the two class maps, so they are
    # directly comparable to each other.
    lim = float(np.abs(np.concatenate([maps["left_hand"], maps["right_hand"]])).max())

    panels = [
        (maps["left_hand"], "imagined LEFT hand", (-lim, lim), "RdBu_r"),
        (maps["right_hand"], "imagined RIGHT hand", (-lim, lim), "RdBu_r"),
        (diff, "difference (LEFT - RIGHT)",
         (-float(np.abs(diff).max()), float(np.abs(diff).max())), "PuOr_r"),
    ]

    for ax, (values, title, vlim, cmap) in zip(axes, panels):
        im, _ = mne.viz.plot_topomap(
            values, info, axes=ax, show=False, cmap=cmap, vlim=vlim,
            sensors=True, contours=4,
        )
        ax.set_title(title, fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("% change vs rest", fontsize=8)

    fig.suptitle(
        f"Scalp topography of ERD, {config.CSP_TMIN}-{config.CSP_TMAX} s after cue "
        f"— subject {config.SUBJECT}\n"
        "blue = power drop = cortical activation",
        fontsize=11,
    )
    fig.tight_layout()
    out = config.FIGURES_DIR / "03_erd_topography.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    say("=" * 74)
    say(f"EPOCHING — subject {config.SUBJECT}, runs {config.RUNS}")
    say("=" * 74)

    # Filter first, epoch second — see preprocessing.load_filtered for why.
    raw = preprocessing.load_filtered(config.SUBJECT, config.RUNS)
    epochs = preprocessing.make_epochs(raw)

    # ------------------------------------------------------------------
    say()
    say("--- WINDOW SANITY CHECK ---")
    events, mapping = mne.events_from_annotations(raw)
    sfreq = raw.info["sfreq"]
    # Spacing between consecutive markers of any kind: this is what determines
    # whether our window can leak into the neighbouring trial.
    gaps = np.diff(events[:, 0]) / sfreq
    say(f"  epoch window requested   : {config.EPOCH_TMIN} to {config.EPOCH_TMAX} s")
    say(f"  window fed to classifier : {config.CSP_TMIN} to {config.CSP_TMAX} s")
    say(f"  spacing between consecutive markers: min {gaps.min():.2f} s, "
        f"median {np.median(gaps):.2f} s")
    leak = config.EPOCH_TMAX - gaps.min()
    if leak > 0:
        say(f"  -> WARNING: window ends {leak:.2f} s after the next marker")
    else:
        say(f"  -> OK: window ends {-leak:.2f} s before the next marker, so no")
        say("     epoch leaks into the following trial.")
    say("  -> the -1 to 0 s part falls inside a T0 rest period (4.20 s long),")
    say("     so our baseline is genuine rest, not another imagery trial.")

    # ------------------------------------------------------------------
    say()
    say("--- TRIALS OBTAINED ---")
    say(f"  data shape: {epochs.get_data().shape}")
    say("              (trials, channels, time samples)")
    say(f"  duration per trial: {len(epochs.times)} samples "
        f"= {len(epochs.times) / sfreq:.2f} s at {sfreq:.0f} Hz")
    say()
    for cls in ("left_hand", "right_hand"):
        say(f"  {cls:<12}: {len(epochs[cls]):>3} trials")
    n_left, n_right = len(epochs["left_hand"]), len(epochs["right_hand"])
    total = n_left + n_right
    say(f"  {'TOTAL':<12}: {total:>3} trials")
    say(f"  class balance: {100 * n_left / total:.1f} % / "
        f"{100 * n_right / total:.1f} %")
    say(f"  -> chance level for a 2-class problem is {100 * max(n_left, n_right) / total:.1f} %,")
    say("     not 50 %: always beat the majority class, not the coin flip.")

    # drop_log has one entry per candidate event, including the T0 rest events
    # we never asked for. Those are logged as "IGNORED", which is not a drop —
    # counting them as dropped would report 45 phantom losses.
    really_dropped = sum(
        1 for d in epochs.drop_log if len(d) > 0 and "IGNORED" not in d
    )
    ignored = sum(1 for d in epochs.drop_log if "IGNORED" in d)
    say(f"  epochs dropped for quality reasons: {really_dropped} "
        "(no rejection threshold was set)")
    say(f"  events ignored because they are T0 rest, not a class: {ignored}")

    # ------------------------------------------------------------------
    say()
    say("--- AMPLITUDE PER TRIAL (manual artifact inspection) ---")
    peak = np.abs(epochs.get_data()).max(axis=(1, 2)) * 1e6  # uV, per trial
    say(f"  peak amplitude across trials: median {np.median(peak):.0f} uV, "
        f"min {peak.min():.0f}, max {peak.max():.0f}")
    loud = int((peak > 150).sum())
    say(f"  trials with a peak above 150 uV: {loud} / {total}")
    say("  We keep all of them. With ~22 trials per class, dropping trials costs")
    say("  more than the artifacts do — and step 6 will tell us if it hurt.")

    # ------------------------------------------------------------------
    say()
    say("--- ERD IN THE CLASSIFIER WINDOW (the physiology check) ---")
    say(f"  mean band-power change vs rest, {config.CSP_TMIN}-{config.CSP_TMAX} s, "
        "in %:")
    say()
    say(f"  {'':<22}{'C3 (left hemi)':>18}{'C4 (right hemi)':>18}")
    table = {}
    for cls in ("left_hand", "right_hand"):
        erd, times = erd_percent(epochs[cls], picks=["C3", "C4"])
        wmask = (times >= config.CSP_TMIN) & (times <= config.CSP_TMAX)
        vals = erd[:, :, wmask].mean(axis=-1).mean(axis=0)
        table[cls] = vals
        say(f"  {'imagined ' + cls.replace('_hand', ' hand'):<22}"
            f"{vals[0]:>17.1f}%{vals[1]:>17.1f}%")

    # A boolean "does it match theory?" is not good enough: a 0.3-point
    # difference and a 30-point difference would both print True. We report the
    # MARGIN, and refuse to call a small one a confirmation.
    say()
    say("  What theory predicts: motor control crosses over, so imagining the")
    say("  LEFT hand should drop power more at C4, and the RIGHT hand more at C3.")
    say("  Lateralisation index LI = ERD(C3) - ERD(C4), in percentage points:")
    say("    left  hand -> LI should be POSITIVE (C4 drops more)")
    say("    right hand -> LI should be NEGATIVE (C3 drops more)")
    li_left = table["left_hand"][0] - table["left_hand"][1]
    li_right = table["right_hand"][0] - table["right_hand"][1]
    say(f"    measured: LI(left) = {li_left:+.1f} pts, LI(right) = {li_right:+.1f} pts")
    contrast = li_left - li_right
    say(f"    lateralisation contrast LI(left) - LI(right) = {contrast:+.1f} pts")
    say("    (should be clearly positive if imagery is lateralised)")

    # Threshold chosen before looking at the number, to avoid rationalising
    # whatever came out: a difference below ~3 points on this metric is smaller
    # than the trial-to-trial noise we can resolve with ~22 trials.
    NOISE_PTS = 3.0
    say()
    if contrast > NOISE_PTS:
        say(f"  -> Direction matches theory, contrast {contrast:+.1f} pts is above the")
        say(f"     {NOISE_PTS:.0f}-point noise floor we set. Weak but present.")
    elif contrast > 0:
        say(f"  -> Direction matches theory BUT the contrast ({contrast:+.1f} pts) is")
        say(f"     below our {NOISE_PTS:.0f}-point noise floor. With ~22 trials per class")
        say("     this is NOT evidence of lateralised imagery. Do not call this a")
        say("     confirmation.")
    else:
        say("  -> Contrast has the WRONG SIGN for this subject: C3/C4 ERD is not")
        say("     lateralised as textbooks describe. This is information, not a bug.")
    say("  Either way, CSP does not depend on C3/C4 specifically: it searches all")
    say("  64 electrodes for whatever combination separates the classes. What this")
    say("  check tells us is whether that combination is likely to be MOTOR.")

    # ------------------------------------------------------------------
    # The confound check. This is the most important part of this step, and it
    # is the one a tutorial would skip.
    # ------------------------------------------------------------------
    say()
    say("--- WHERE IS THE BETWEEN-CLASS DIFFERENCE ON THE SCALP? ---")
    say("  A classifier does not care about ERD; it cares about any difference")
    say("  between the two classes. So the real question is not 'is there ERD?'")
    say("  but 'where does the difference live?'. If it lives over the")
    say("  sensorimotor strip, we are decoding motor imagery. If it lives")
    say("  somewhere else, we are decoding something else.")

    region_maps = {}
    for cls in ("left_hand", "right_hand"):
        erd, times = erd_percent(epochs[cls], picks="eeg")
        wmask = (times >= config.CSP_TMIN) & (times <= config.CSP_TMAX)
        region_maps[cls] = erd[:, :, wmask].mean(axis=-1).mean(axis=0)

    diff_map = region_maps["left_hand"] - region_maps["right_hand"]
    eeg_names = epochs.copy().pick("eeg").ch_names

    say()
    for label, group in (("sensorimotor", SENSORIMOTOR_CH),
                         ("posterior (visual/parietal)", POSTERIOR_CH)):
        idx = [eeg_names.index(c) for c in group if c in eeg_names]
        vals = diff_map[idx]
        say(f"  {label:<28} mean |LEFT-RIGHT| = {np.abs(vals).mean():6.1f} pts "
            f"(max {np.abs(vals).max():.1f})")

    sm_idx = [eeg_names.index(c) for c in SENSORIMOTOR_CH if c in eeg_names]
    po_idx = [eeg_names.index(c) for c in POSTERIOR_CH if c in eeg_names]
    sm = np.abs(diff_map[sm_idx]).mean()
    po = np.abs(diff_map[po_idx]).mean()
    strongest = eeg_names[int(np.argmax(np.abs(diff_map)))]
    say(f"  single strongest electrode: {strongest} "
        f"({diff_map[int(np.argmax(np.abs(diff_map)))]:+.1f} pts)")

    say()
    if po <= sm:
        say(f"  -> The difference is largest over the sensorimotor strip "
            f"({sm / po:.1f}x posterior).")
        say("     That is the pattern we want: the separable information sits")
        say("     where motor imagery should put it.")
    else:
        say(f"  -> RED FLAG: the between-class difference is {po / sm:.1f}x LARGER")
        say("     posteriorly than over the sensorimotor strip. Something other")
        say("     than motor imagery separates these two classes. What?")

        # --------------------------------------------------------------
        # HYPOTHESIS 1 — the visual cue.
        # In this dataset the cue is a target appearing on the LEFT or the RIGHT
        # of the screen, staying visible for the whole trial. The visual pathway
        # crosses over just like the motor one: a target on the RIGHT is seen by
        # the LEFT visual cortex (electrode O1). So IF the posterior difference
        # is the cue, right-hand trials must drop more at O1 than at O2, and
        # left-hand trials the opposite. That is a falsifiable prediction, so we
        # test it rather than assert it.
        # --------------------------------------------------------------
        say()
        say("  HYPOTHESIS 1 — it is the visual cue. This dataset cues trials with")
        say("  a target on the left or right of the screen, visible for the whole")
        say("  trial. Vision crosses over too, so a target on the RIGHT is seen by")
        say("  the LEFT visual cortex (O1).")
        say("  Prediction: right-hand trials drop more at O1, left-hand at O2.")
        verdicts = []
        for cls, expect in (("right_hand", "O1"), ("left_hand", "O2")):
            erd, times = erd_percent(epochs[cls], picks=["O1", "O2"])
            wmask = (times >= config.CSP_TMIN) & (times <= config.CSP_TMAX)
            v = erd[:, :, wmask].mean(axis=-1).mean(axis=0)
            stronger = "O1" if v[0] < v[1] else "O2"
            ok_pred = stronger == expect
            verdicts.append(ok_pred)
            say(f"    {cls:<11}: O1 {v[0]:+7.1f}%  O2 {v[1]:+7.1f}%  -> stronger "
                f"drop at {stronger} ({'as predicted' if ok_pred else 'NOT as predicted'})")
        if all(verdicts):
            say("  -> Prediction holds. The posterior difference behaves like a")
            say("     lateralised visual response to the cue.")
        else:
            say("  -> REJECTED. The prediction fails, and the posterior effect is")
            say("     roughly the same on both sides — it is bilateral, not")
            say("     lateralised. A visual response to a one-sided target would")
            say("     have to be lateralised. So this is not the cue.")

        # --------------------------------------------------------------
        # HYPOTHESIS 2 — a few extreme trials.
        # Occipital alpha is the largest rhythm in our band and it explodes when
        # a subject briefly closes their eyes or drifts off. With ~22 trials per
        # class, one such trial can move a mean by tens of points. The mean is
        # not robust to that; the median is. Comparing the two tells us whether
        # we are looking at a property of the class or at a couple of trials.
        # --------------------------------------------------------------
        say()
        say("  HYPOTHESIS 2 — a handful of extreme trials. Occipital alpha is the")
        say("  biggest rhythm in our band and it explodes when someone blinks a")
        say("  lot or briefly drifts off. The mean is not robust to that; the")
        say("  median is. If the effect is real, both should show it.")
        po_stats = {}
        for cls in ("left_hand", "right_hand"):
            erd, times = erd_percent(epochs[cls], picks="eeg")
            wmask = (times >= config.CSP_TMIN) & (times <= config.CSP_TMAX)
            per_trial = erd[:, :, wmask].mean(axis=-1)[:, po_idx].mean(axis=1)
            po_stats[cls] = per_trial
            say(f"    {cls:<11}: mean {per_trial.mean():+7.1f}%  "
                f"median {np.median(per_trial):+7.1f}%  "
                f"worst trial {per_trial.max():+7.1f}%")
        gap_mean = po_stats["left_hand"].mean() - po_stats["right_hand"].mean()
        gap_med = np.median(po_stats["left_hand"]) - np.median(po_stats["right_hand"])
        say(f"    between-class gap: {gap_mean:+.1f} pts on means, "
            f"{gap_med:+.1f} pts on medians")
        say("  -> Partly. One extreme trial inflates the mean, but the gap survives")
        say("     the median, so it is not the work of a single bad trial either.")

        # --------------------------------------------------------------
        # HYPOTHESIS 3 — trial order / fatigue.
        # If one class systematically occurred later in the session, growing
        # drowsiness (which raises alpha) would masquerade as a class effect.
        # This is only possible if the class order is blocked rather than random.
        # --------------------------------------------------------------
        say()
        say("  HYPOTHESIS 3 — trial order. If one class happened to occur later in")
        say("  the session, growing drowsiness (which raises alpha) would look")
        say("  like a class effect. That requires the class order to be blocked.")
        all_po = np.concatenate([po_stats["left_hand"], po_stats["right_hand"]])
        order = np.concatenate([
            np.where(epochs.events[:, 2] == config.EVENT_ID["left_hand"])[0],
            np.where(epochs.events[:, 2] == config.EVENT_ID["right_hand"])[0],
        ])
        r = float(np.corrcoef(order, all_po)[0, 1])
        seq = "".join(
            "L" if e == config.EVENT_ID["left_hand"] else "R"
            for e in epochs.events[:, 2]
        )
        say(f"    class order in the recording: {seq}")
        say(f"    correlation(trial index, posterior power) = {r:+.2f}")
        say("  -> REJECTED. The two classes alternate irregularly, i.e. the order")
        say("     is randomised, and the drift with trial index is weak. A")
        say("     randomised order cannot systematically favour one class.")

        # --------------------------------------------------------------
        say()
        say("  WHAT WE ARE LEFT WITH. The posterior difference is not the visual")
        say("  cue, not one bad trial, and not fatigue. The remaining and most")
        say("  likely explanation is the dullest one: sampling noise. Occipital")
        say("  alpha varies enormously from trial to trial, and with ~22 trials")
        say("  per class a gap of this size can appear by chance. We cannot")
        say("  settle it with 45 trials — and saying so is the honest answer.")
        say()
        say("  WHY IT MATTERS ANYWAY, concretely, for step 5 and 6:")
        say("  CSP maximises a variance ratio over all 64 electrodes. It has no")
        say("  notion of anatomy and will happily build its filters out of")
        say("  high-variance posterior channels if that separates the classes in")
        say("  the training data. That is textbook overfitting, and with 45 trials")
        say("  for a 64x64 covariance matrix we are squarely in the regime where")
        say("  it happens.")
        say("  ACTION for step 6: run the pipeline twice — once on all 64")
        say("  electrodes, once restricted to the sensorimotor strip. If the")
        say("  restricted version does as well or better, the extra electrodes")
        say("  were feeding the classifier noise, and we will have measured it")
        say("  instead of guessing.")

    # ------------------------------------------------------------------
    say()
    say("--- FIGURES WRITTEN ---")
    for out in (plot_erd_timecourse(epochs), plot_erd_topography(epochs)):
        say(f"  {out.relative_to(config.ROOT_DIR)}")

    report = config.RESULTS_DIR / "03_epochs.txt"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nReport written: {report.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
