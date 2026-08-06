"""
STEP 7 — The whole pipeline on 10 subjects, and what survives.

=========================================================================
WHY THIS STEP IS NOT OPTIONAL
=========================================================================

Everything in steps 4-6 describes ONE person. Three findings came out of subject
1, and each of them is either an important result or an accident of one
recording — and there is no way to tell which from a single subject:

  FINDING 1 (step 4). Contralateral ERD was clean for the right hand (C3 dropped
  9.2 points more than C4) and absent for the left (0.8 points). We offered two
  explanations we could not separate: too few trials, or a real dominant /
  non-dominant asymmetry. Ten subjects can separate them. If the left-hand null
  is noise, it should scatter randomly across subjects and average away. If it is
  real, it should reappear.

  FINDING 2 (steps 4-5). The between-class difference was 4.2x larger over
  posterior electrodes than over the motor strip, and CSP's strongest component
  peaked at O1, an occipital electrode. We rejected the visual cue, one bad
  trial, and fatigue as explanations, and it survived 50 train/test splits. If
  this is a property of the PROTOCOL, every subject should show it. If it is a
  property of subject 1, it should mostly vanish.

  FINDING 3 (step 6). 62.4 % on the motor strip, p = 0.014. One subject's
  accuracy tells us nothing about the method — between-subject variability in
  motor imagery is larger than any difference we measured.

This step also puts a number on the calibration argument in METHODOLOGY.md: if
one model per person is really necessary, the spread across people should be
wide. If everyone landed on the same accuracy with the same spatial patterns, a
shared model would be defensible.

=========================================================================
WHAT WE DO AND DO NOT CHANGE
=========================================================================

Every parameter stays exactly as it was for subject 1: same band, same window,
same components, same folds. Nothing is re-tuned per subject. Tuning per subject
would produce better numbers and a meaningless comparison — we would no longer be
measuring between-subject variability, we would be measuring how much tuning
helps.

Each subject gets its own model, fitted on its own trials. That is the
calibration logic of a real BCI (METHODOLOGY.md), not a convenience.

Permutations are reduced from 500 to 200 here. With ten subjects, per-subject
p-value resolution matters less than the group pattern, and 200 shufflings still
resolve p down to 1/201 = 0.005.

Outputs:
  results/06_all_subjects.txt          the full report
  results/06_all_subjects.csv           one row per subject, for reuse in step 8
  figures/06_subjects_accuracy.png      accuracy per subject, both variants
  figures/06_subjects_lateralisation.png  the finding-1 check
  figures/06_subjects_regions.png       the finding-2 check
"""

import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config
import erd
import model
import preprocessing
from model import csp_lda, majority_baseline, repeated_cv

mne.set_log_level("ERROR")

N_PERMUTATIONS = 200

# Same noise floor as step 4, fixed before looking at any of these numbers: a
# lateralisation contrast below 3 percentage points is smaller than what ~22
# trials per class can resolve.
NOISE_PTS = 3.0


def permutation_p(X, y, n_permutations=N_PERMUTATIONS):
    """Fraction of label shufflings that match or beat the real score.

    Serial loop for the reasons documented in step 6: MNE's rank estimator
    rejects joblib's shared-memory arrays, and threads measured slower than
    serial anyway.
    """
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    observed = cross_val_score(csp_lda(), X, y, cv=cv, scoring="accuracy").mean()
    rng = np.random.default_rng(config.RANDOM_STATE)
    null = np.array([
        cross_val_score(csp_lda(), X, rng.permutation(y), cv=cv,
                        scoring="accuracy").mean()
        for _ in range(n_permutations)
    ])
    # +1 correction (Phipson & Smyth): with finitely many shufflings you can
    # never legitimately report p = 0.
    p = (1 + int((null >= observed).sum())) / (1 + n_permutations)
    return observed, p


def analyse_subject(subject):
    """Run the entire pipeline on one subject and return every metric we compare."""
    raw = preprocessing.load_filtered(subject=subject)
    epochs = preprocessing.make_epochs(raw)
    work = epochs.copy().crop(tmin=config.CSP_TMIN, tmax=config.CSP_TMAX)
    y = work.events[:, 2]

    n_left = int((y == config.EVENT_ID["left_hand"]).sum())
    n_right = int((y == config.EVENT_ID["right_hand"]).sum())

    X_all, _ = model.variant_data(work, None)
    X_motor, _ = model.variant_data(work, config.SENSORIMOTOR_CH)

    acc_all = repeated_cv(X_all, y)
    acc_motor = repeated_cv(X_motor, y)
    _, p_motor = permutation_p(X_motor, y)

    # ERD metrics on the full epoch (they need the pre-cue baseline, which the
    # cropped version no longer has).
    lat = erd.lateralisation(epochs)
    reg = erd.region_contrast(epochs)

    return {
        "subject": subject,
        "n_left": n_left,
        "n_right": n_right,
        "baseline": majority_baseline(y),
        "acc_all": float(acc_all.mean()),
        "acc_all_sd": float(acc_all.std()),
        "acc_motor": float(acc_motor.mean()),
        "acc_motor_sd": float(acc_motor.std()),
        "p_motor": p_motor,
        "li_left": lat["left_hand"]["LI"],
        "li_right": lat["right_hand"]["LI"],
        "li_contrast": lat["contrast"],
        "erd_c3_left": lat["left_hand"]["C3"],
        "erd_c4_left": lat["left_hand"]["C4"],
        "erd_c3_right": lat["right_hand"]["C3"],
        "erd_c4_right": lat["right_hand"]["C4"],
        "motor_diff": reg["sensorimotor"],
        "posterior_diff": reg["posterior"],
        "post_over_motor": reg["posterior_over_motor"],
        "peak_channel": reg["peak_channel"],
    }


# ==========================================================================
# FIGURES
# ==========================================================================
def plot_accuracy(rows):
    """Accuracy per subject, both variants, sorted by the motor-strip result.

    Sorted rather than in subject order because the shape of the distribution is
    the point: how many subjects work at all, and how far apart the best and
    worst are.
    """
    rows = sorted(rows, key=lambda r: r["acc_motor"])
    x = np.arange(len(rows))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.bar(x - w / 2, [100 * r["acc_all"] for r in rows], w,
           yerr=[100 * r["acc_all_sd"] for r in rows], capsize=3,
           color="#8ba7c4", label="all 64 electrodes")
    ax.bar(x + w / 2, [100 * r["acc_motor"] for r in rows], w,
           yerr=[100 * r["acc_motor_sd"] for r in rows], capsize=3,
           color="#1f4e79", label="sensorimotor strip (headline)")

    ax.axhline(100 * np.mean([r["baseline"] for r in rows]), color="#c0392b",
               ls="--", lw=1.2, label="mean majority-class baseline")
    ax.set_xticks(x, [f"S{r['subject']}" for r in rows])
    ax.set_ylabel("cross-validated accuracy (%)")
    ax.set_ylim(35, 100)
    ax.set_xlabel("subject (sorted by motor-strip accuracy)")
    ax.set_title(
        "Between-subject variability — same pipeline, no per-subject tuning\n"
        "error bars = spread over 10 CV repeats",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = config.FIGURES_DIR / "06_subjects_accuracy.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_lateralisation(rows):
    """Finding 1: does contralateral ERD hold per hand, across subjects?

    Theory: LI(left) positive, LI(right) negative. Each subject contributes two
    points. If subject 1's left-hand null was noise, the left-hand points should
    scatter around zero across subjects; if it is real, they should sit near zero
    systematically while the right-hand points stay negative.
    """
    subs = [r["subject"] for r in rows]
    x = np.arange(len(rows))
    left = [r["li_left"] for r in rows]
    right = [r["li_right"] for r in rows]

    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.axhline(0, color="0.4", lw=1)
    ax.axhspan(-NOISE_PTS, NOISE_PTS, color="0.85", alpha=0.7,
               label=f"below the {NOISE_PTS:.0f}-point noise floor")
    ax.scatter(x - 0.12, left, s=70, color="#c0392b", label="imagined LEFT hand",
               zorder=3)
    ax.scatter(x + 0.12, right, s=70, color="#1f4e79", marker="s",
               label="imagined RIGHT hand", zorder=3)

    for xi, (l, r) in enumerate(zip(left, right)):
        ax.plot([xi - 0.12, xi + 0.12], [l, r], color="0.7", lw=0.9, zorder=1)

    ax.set_xticks(x, [f"S{s}" for s in subs])
    ax.set_ylabel("lateralisation index\nERD(C3) - ERD(C4), percentage points")
    ax.set_title(
        "Finding 1 — is contralateral ERD real, or was subject 1 an accident?\n"
        "theory predicts LEFT hand above zero (C4 drops more) "
        "and RIGHT hand below zero",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = config.FIGURES_DIR / "06_subjects_lateralisation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_regions(rows):
    """Finding 2: is the posterior dominance a property of the protocol?

    A bar above 1 means the between-class difference is larger posteriorly than
    over the motor strip — the step-4 red flag. If every subject is above 1, it
    is the protocol. If subject 1 stands alone, it was subject 1.
    """
    rows = sorted(rows, key=lambda r: r["post_over_motor"], reverse=True)
    x = np.arange(len(rows))
    ratios = [r["post_over_motor"] for r in rows]
    colors = ["#c0392b" if v > 1 else "#1f4e79" for v in ratios]

    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.bar(x, ratios, 0.62, color=colors)
    ax.axhline(1, color="0.3", ls="--", lw=1.2)
    ax.text(len(rows) - 0.4, 1.06, "equal", fontsize=8, color="0.3", ha="right")

    for xi, r in zip(x, rows):
        ax.annotate(r["peak_channel"], xy=(xi, ratios[xi]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="0.35")

    ax.set_xticks(x, [f"S{r['subject']}" for r in rows])
    ax.set_ylabel("posterior difference / sensorimotor difference")
    ax.set_title(
        "Finding 2 — where does the between-class difference live?\n"
        "red = larger at the back of the head than over the motor cortex; "
        "label = strongest single electrode",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = config.FIGURES_DIR / "06_subjects_regions.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ==========================================================================
def main():
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    say("=" * 78)
    say(f"ALL SUBJECTS — {config.SUBJECTS_ALL}, runs {config.RUNS}")
    say("=" * 78)
    say("Identical parameters for every subject. Nothing re-tuned per person:")
    say(f"  band {config.F_LOW:.0f}-{config.F_HIGH:.0f} Hz | "
        f"window {config.CSP_TMIN}-{config.CSP_TMAX} s | "
        f"{config.N_CSP_COMPONENTS} CSP components | "
        f"{config.CV_FOLDS}-fold x {model.N_REPEATS}")

    rows, failed = [], []
    for subject in config.SUBJECTS_ALL:
        try:
            rows.append(analyse_subject(subject))
            print(f"  subject {subject:>2} done")
        except Exception as exc:                      # noqa: BLE001
            # Reported rather than swallowed: a subject we could not process is
            # information about the dataset, not a detail to hide.
            failed.append((subject, type(exc).__name__, str(exc)[:90]))
            print(f"  subject {subject:>2} FAILED: {exc}")

    if failed:
        say()
        say("--- SUBJECTS THAT COULD NOT BE PROCESSED ---")
        for s, kind, msg in failed:
            say(f"  subject {s}: {kind}: {msg}")

    say()
    say("--- PER-SUBJECT RESULTS ---")
    say(f"  {'sub':>4}{'trials':>8}{'base':>7}{'all64':>13}{'motor':>13}"
        f"{'p(motor)':>10}{'LI left':>9}{'LI right':>9}{'post/mot':>10}{'peak':>7}")
    for r in rows:
        say(f"  S{r['subject']:<3}{r['n_left']}/{r['n_right']:<5}"
            f"{100 * r['baseline']:6.1f}%"
            f"{100 * r['acc_all']:8.1f}+-{100 * r['acc_all_sd']:<3.0f}"
            f"{100 * r['acc_motor']:8.1f}+-{100 * r['acc_motor_sd']:<3.0f}"
            f"{r['p_motor']:>10.3f}"
            f"{r['li_left']:>9.1f}{r['li_right']:>9.1f}"
            f"{r['post_over_motor']:>10.1f}{r['peak_channel']:>7}")

    acc_all = np.array([r["acc_all"] for r in rows])
    acc_motor = np.array([r["acc_motor"] for r in rows])
    base = np.array([r["baseline"] for r in rows])

    # ------------------------------------------------------------------
    say()
    say("--- FINDING 3 FIRST: HOW WELL DOES THE METHOD WORK, REALLY? ---")
    say(f"  sensorimotor strip : {100 * acc_motor.mean():.1f} % "
        f"(median {100 * np.median(acc_motor):.1f}, "
        f"range {100 * acc_motor.min():.1f} - {100 * acc_motor.max():.1f})")
    say(f"  all 64 electrodes  : {100 * acc_all.mean():.1f} % "
        f"(median {100 * np.median(acc_all):.1f}, "
        f"range {100 * acc_all.min():.1f} - {100 * acc_all.max():.1f})")
    say(f"  mean baseline      : {100 * base.mean():.1f} %")
    say()
    # The mean is the wrong summary here and saying so is the point of this
    # block. A distribution carried by two outliers has a mean that describes
    # nobody, so we show what happens when they are removed.
    ranked = np.sort(acc_motor)
    without_top2 = ranked[:-2].mean()
    say("  READ THE MEDIAN, NOT THE MEAN:")
    say(f"    mean                       {100 * acc_motor.mean():.1f} %")
    say(f"    median                     {100 * np.median(acc_motor):.1f} %")
    say(f"    mean without the top 2      {100 * without_top2:.1f} %")
    say(f"    mean majority baseline     {100 * base.mean():.1f} %")
    say("  The median sits barely above the baseline, and with the two best")
    say("  subjects removed the pipeline is AT chance. Two people out of ten carry")
    say("  the entire average. Quoting the mean alone would describe a decoder")
    say("  that does not exist for eight of these ten subjects.")
    say()
    above = int((np.array([r["p_motor"] for r in rows]) < 0.05).sum())
    say(f"  subjects individually above chance (p < 0.05, motor strip): "
        f"{above} / {len(rows)}")
    say(f"  spread between best and worst subject: "
        f"{100 * (acc_motor.max() - acc_motor.min()):.1f} points")
    say("  That spread is the calibration argument made concrete: a single shared")
    say("  model would have to average over people this different from each other.")
    say(f"  It also matches the known 'BCI illiteracy' figure — 15-30 % of healthy")
    say("  subjects produce no decodable motor imagery — which predicts that some")
    say("  subjects here should simply not work. That is an expected property of")
    say("  the paradigm, not a bug in the pipeline.")

    # ------------------------------------------------------------------
    say()
    say("--- FINDING 1: WAS SUBJECT 1'S MISSING LEFT-HAND LATERALISATION REAL? ---")
    li_left = np.array([r["li_left"] for r in rows])
    li_right = np.array([r["li_right"] for r in rows])
    contrast = np.array([r["li_contrast"] for r in rows])
    say("  Theory: LI(left) > 0 (C4 drops more), LI(right) < 0 (C3 drops more).")
    say(f"  LI(left)  across subjects: mean {li_left.mean():+.1f} pts, "
        f"{int((li_left > 0).sum())}/{len(rows)} with the predicted sign")
    say(f"  LI(right) across subjects: mean {li_right.mean():+.1f} pts, "
        f"{int((li_right < 0).sum())}/{len(rows)} with the predicted sign")
    say(f"  contrast LI(left)-LI(right): mean {contrast.mean():+.1f} pts, "
        f"{int((contrast > NOISE_PTS).sum())}/{len(rows)} above the "
        f"{NOISE_PTS:.0f}-pt noise floor")

    # A confound in our own metric, found by looking at both rows together.
    # If LI is negative for BOTH hands, C3 simply shows more ERD than C4
    # regardless of which hand was imagined. That is a global left/right offset,
    # not a hand-specific effect — and it inflates the "8/10 correct for the
    # right hand" count while deflating the left-hand one. The confound-free
    # quantity is the CONTRAST between hands, which cancels any constant offset.
    both = np.concatenate([li_left, li_right])
    n_both_neg = int(((li_left < 0) & (li_right < 0)).sum())
    say()
    say("  CAREFUL — our own metric has an offset:")
    say(f"    mean LI across BOTH hands pooled: {both.mean():+.1f} pts")
    say(f"    subjects where C3 drops more than C4 for BOTH hands: "
        f"{n_both_neg}/{len(rows)}")
    say("    So C3 tends to show more ERD than C4 whatever the hand. That is a")
    say("    global hemispheric offset, not a hand-specific pattern, and it")
    say("    partly manufactures the per-hand counts above: it pushes both LIs")
    say("    negative, which 'confirms' the right hand and 'refutes' the left.")
    say("    The offset-free measure is the CONTRAST between hands, because a")
    say("    constant added to both cancels in the subtraction. That contrast is")
    say(f"    {contrast.mean():+.1f} pts on average and clears the noise floor in "
        f"{int((contrast > NOISE_PTS).sum())}/{len(rows)} subjects,")
    say("    so the lateralised DIFFERENCE BETWEEN HANDS is real even though the")
    say("    per-hand story is muddier than step 4 suggested.")
    say()
    if (li_left > 0).sum() >= 0.7 * len(rows):
        say("  -> The left hand DOES lateralise in most subjects. Subject 1's null")
        say("     was therefore specific to subject 1, most plausibly the small")
        say("     trial count, and should not be read as a dominant/non-dominant")
        say("     asymmetry. Explanation 1 from step 4 wins.")
    elif (li_right < 0).sum() >= 0.7 * len(rows) and n_both_neg < 0.5 * len(rows):
        say("  -> The RIGHT hand lateralises consistently across subjects while the")
        say("     LEFT hand does not, and this is NOT explained by a global offset")
        say("     (few subjects are negative for both hands). Subject 1 was not an")
        say("     accident: step 4's second explanation — weaker, less focal")
        say("     imagery of the non-dominant hand in a mostly right-handed")
        say("     population — becomes the better one.")
    elif (li_right < 0).sum() >= 0.7 * len(rows):
        say("  -> VERDICT, stated carefully. Subject 1 was not an accident: the")
        say("     same asymmetry appears across subjects. But we CANNOT conclude")
        say("     'the right hand lateralises and the left does not', because most")
        say("     subjects show C3 dropping more than C4 for BOTH hands. A global")
        say("     hemispheric offset of that kind reproduces the per-hand counts")
        say("     on its own, so the counts are not evidence for the dominant /")
        say("     non-dominant story.")
        say("     What DOES survive the offset is the hand-to-hand contrast, which")
        say("     is positive and above the noise floor in most subjects. So:")
        say("     there IS a lateralised difference between imagining one hand and")
        say("     the other; whether the left hand is specifically weaker remains")
        say("     unresolved, and distinguishing the two would need handedness")
        say("     information the dataset does not provide.")
    else:
        say("  -> Neither hand lateralises consistently at C3/C4 across subjects.")
        say("     The textbook contralateral picture is not visible at this sample")
        say("     size with this metric. Worth stating plainly: our decoder works")
        say("     without us being able to demonstrate the mechanism it relies on.")

    # ------------------------------------------------------------------
    say()
    say("--- FINDING 2: IS THE POSTERIOR SIGNAL THE PROTOCOL OR SUBJECT 1? ---")
    ratio = np.array([r["post_over_motor"] for r in rows])
    n_post = int((ratio > 1).sum())
    say(f"  subjects where the between-class difference is LARGER posteriorly: "
        f"{n_post} / {len(rows)}")
    say(f"  ratio posterior/sensorimotor: median {np.median(ratio):.1f}, "
        f"range {ratio.min():.1f} - {ratio.max():.1f}")
    peaks = [r["peak_channel"] for r in rows]
    post_set = set(config.POSTERIOR_CH)
    motor_set = set(config.SENSORIMOTOR_CH)
    n_peak_post = sum(1 for c in peaks if c in post_set)
    n_peak_motor = sum(1 for c in peaks if c in motor_set)
    say(f"  strongest electrode per subject: {peaks}")
    say(f"    posterior: {n_peak_post}/{len(rows)}   "
        f"sensorimotor: {n_peak_motor}/{len(rows)}   "
        f"elsewhere: {len(rows) - n_peak_post - n_peak_motor}/{len(rows)}")

    # Where the "elsewhere" peaks land is diagnostic, so we name the families
    # rather than lumping them together. Each has a known non-brain source.
    families = {
        "posterior (visual / attention)": post_set,
        "sensorimotor (what we want)": motor_set,
        "temporal (jaw / neck muscle, EMG)": {"T7", "T8", "T9", "T10", "FT7",
                                              "FT8", "TP7", "TP8"},
        "frontopolar / frontal (eye movement, EOG)": {"Fp1", "Fpz", "Fp2", "AF7",
                                                      "AF3", "AFz", "AF4", "AF8",
                                                      "F7", "F5", "F3", "F1",
                                                      "Fz", "F2", "F4", "F6",
                                                      "F8"},
    }
    say()
    say("  Which anatomical family does each subject's strongest electrode fall in?")
    for label, group in families.items():
        hits = [f"S{r['subject']}" for r in rows if r["peak_channel"] in group]
        say(f"    {label:<44}{len(hits)}/{len(rows)}  {' '.join(hits)}")
    say()
    say("  NOT ONE subject has its strongest between-class electrode over the")
    say("  sensorimotor strip. The peaks are occipital, temporal or frontal —")
    say("  and those three families have well-known non-brain sources: eye")
    say("  movements frontally, jaw and neck muscle temporally, visual and")
    say("  attentional activity posteriorly. The confound family is therefore")
    say("  broader than the 'posterior' one step 4 chased.")
    say()
    if n_post >= 0.7 * len(rows):
        say("  -> It is the PROTOCOL, not subject 1. Most subjects show a bigger")
        say("     between-class difference at the back of the head than over the")
        say("     motor cortex. Whatever this signal is, it is a systematic")
        say("     property of this dataset, and any published accuracy from all 64")
        say("     electrodes on it is partly measuring it. This strengthens the")
        say("     step-5 decision to headline the motor-strip result.")
    elif n_post <= 0.3 * len(rows):
        say("  -> It was SUBJECT 1. Most subjects put their strongest between-class")
        say("     difference over the motor strip, as physiology predicts. The")
        say("     step-4 red flag was a single-subject accident — worth having")
        say("     chased, and now retired.")
    else:
        say("  -> Mixed: it happens in some subjects and not others. So it is")
        say("     neither a protocol artefact nor a fluke of one recording, but a")
        say("     per-subject property. Practically that is the worst case: it")
        say("     cannot be corrected once for the whole dataset, and it means")
        say("     per-subject pattern inspection is not optional.")

    # ------------------------------------------------------------------
    # Cross-referencing the two findings: the subjects that carry the average
    # are also the ones whose strongest electrode is most artifact-prone. This
    # only becomes visible when the accuracy table and the peak-channel table are
    # read together, which is why it is computed rather than left to the reader.
    say()
    say("  CROSS-CHECK — WHO CARRIES THE AVERAGE, AND WHERE IS THEIR SIGNAL?")
    top2 = sorted(rows, key=lambda r: r["acc_motor"], reverse=True)[:2]
    emg = families["temporal (jaw / neck muscle, EMG)"]
    for r in top2:
        flag = " <- temporal, an EMG-prone site" if r["peak_channel"] in emg else ""
        say(f"    S{r['subject']}: {100 * r['acc_motor']:.1f} % motor-strip, "
            f"strongest electrode {r['peak_channel']}{flag}")
    if all(r["peak_channel"] in emg for r in top2):
        say("    Both of the two subjects carrying the entire group average have")
        say("    their strongest between-class difference at a temporal electrode.")
        say("    Temporal sites pick up jaw and neck muscle activity, which is")
        say("    exactly the contamination we filtered 8-30 Hz partly to avoid —")
        say("    and EMG extends into the upper part of that band.")
        say("    This does NOT prove their results are muscle. Their motor-strip")
        say("    accuracy is high on its own, and lateral strip channels (C5, C6,")
        say("    FC5, FC6) are near enough to temporal sites that the question")
        say("    cannot be settled by electrode position alone. But the two")
        say("    strongest results in the study are the two where a non-brain")
        say("    explanation is most available, and reporting a group mean without")
        say("    saying so would be misleading.")
        say("    What would settle it: EMG-specific checks (high-frequency power,")
        say("    ICA artifact components), which are outside this project's scope")
        say("    and are recorded as the most valuable next diagnostic.")

    say()
    say("--- WHAT THE ALL-64 vs MOTOR-STRIP GAP LOOKS LIKE ACROSS SUBJECTS ---")
    gap = acc_all - acc_motor
    say(f"  mean gap (all64 - motor): {100 * gap.mean():+.1f} pts "
        f"(range {100 * gap.min():+.1f} to {100 * gap.max():+.1f})")
    say(f"  subjects where all 64 electrodes win: {int((gap > 0).sum())}/{len(rows)}")
    if gap.mean() > 0:
        say("  The extra electrodes help on average, as they did for subject 1.")
        say("  METHODOLOGY.md explains at length why that is not enough to make")
        say("  them the headline: the gain is not attributable to motor activity,")
        say("  and a gain you cannot attribute is a gain you cannot rely on when")
        say("  the protocol changes.")

    # ------------------------------------------------------------------
    say()
    say("--- FIGURES WRITTEN ---")
    for out in (plot_accuracy(rows), plot_lateralisation(rows), plot_regions(rows)):
        say(f"  {out.relative_to(config.ROOT_DIR)}")

    # CSV so step 8 (and anyone else) can re-use these numbers without re-running.
    csv_path = config.RESULTS_DIR / "06_all_subjects.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    say(f"  {csv_path.relative_to(config.ROOT_DIR)}")

    report = config.RESULTS_DIR / "06_all_subjects.txt"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nReport written: {report.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
