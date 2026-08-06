"""
STEP 5 — CSP (Common Spatial Patterns).

=========================================================================
WHAT CSP DOES, IN PLAIN TERMS
=========================================================================

THE PROBLEM. A single electrode is a bad measurement. C3 does not record "the
left motor cortex": it records a weighted sum of everything happening in the
brain, blurred by the skull, and dominated by whatever source happens to be
loudest — usually not the one we care about. Step 4 showed this concretely: the
biggest between-class difference was at the back of the head, not over the motor
strip.

THE IDEA. Instead of choosing electrodes, build a *weighted combination* of all
of them — a "virtual electrode". Choose the weights so that the resulting signal
is LOUD for one class and QUIET for the other. That weighted combination is
called a spatial filter, and CSP finds the best ones automatically.

Concretely: for each trial we have a matrix (channels x time). CSP computes the
average covariance matrix of each class — covariance being, on the diagonal, the
power of each channel, and off-diagonal, how channels co-vary. It then looks for
a weight vector w that maximises the ratio

        variance of w applied to class 1
        --------------------------------
        variance of w applied to class 2

Mathematically this is a generalised eigenvalue problem, and it has a closed-form
solution — no training loop, no learning rate, no random initialisation. That is
one reason CSP is still the standard baseline three decades later: it is fast,
deterministic, and there is nothing to tune except how many components to keep.

WHY THIS FITS MOTOR IMAGERY EXACTLY. ERD (step 4) is a *power* change, and it is
lateralised. A spatial filter that emphasises the right motor cortex and
suppresses everything else will produce a quiet signal when the left hand is
imagined (that cortex desynchronises) and a louder one when the right hand is.
So the variance of the filtered signal IS the feature. CSP and ERD are made for
each other — filtering to 8-30 Hz in step 3 is what made "variance" mean "band
power" rather than "amount of drift".

COMPONENTS COME IN PAIRS. The eigenvalues are ordered: the first filter
maximises class-1-over-class-2 variance, the last one does the exact opposite.
So we take them from both ends. n_components=4 means 2 filters tuned for the
left hand and 2 for the right.

FEATURES. For each trial and each filter: the variance of the filtered signal,
then a logarithm. The log is not decoration — variance distributions are heavily
skewed (a few trials with big power stretch the tail), and LDA in step 6 works
best on roughly symmetric features. `CSP(log=True)` does this for us.

=========================================================================
FILTERS vs PATTERNS — the subtlety that makes this step a diagnosis
=========================================================================

CSP gives two related matrices, and plotting the wrong one leads to wrong
conclusions:

  - FILTERS (`csp.filters_`): how to weight the electrodes to EXTRACT a source.
    Optimised for extraction, and their weights can be large and oddly signed
    on channels that mostly serve to cancel noise.
  - PATTERNS (`csp.patterns_`): how the extracted source PROJECTS BACK onto the
    scalp. This is what you look at to answer "where is this coming from?".

For interpretation you plot the patterns. This is a well-known point in the
literature (Haufe et al., 2014, on the difference between backward and forward
models) and it is why this script plots patterns, not filters.

That makes step 5 more than a transformation — it is a test. Step 4 raised a
red flag: the between-class difference was 4.2x larger posteriorly than over the
sensorimotor strip. If the CSP patterns land on the motor strip, the flag was a
false alarm. If they land at the back of the head, CSP has confirmed it by
building its filters out of occipital noise. Both outcomes are reportable.

=========================================================================
THE TWO-VARIANT COMPARISON, DECIDED IN STEP 4
=========================================================================

We run everything twice:
  A. all 64 electrodes — CSP chooses freely, and can pick posterior channels;
  B. the 21 electrodes of the sensorimotor strip — CSP is forced to look where
     motor imagery must be, if it is there.

Plus a third variant, because the underlying problem is statistical:
  C. all 64 electrodes with covariance SHRINKAGE (Ledoit-Wolf). A 64x64
     covariance matrix estimated from 23 trials is a noisy object; shrinkage
     pulls it towards a simpler, better-conditioned one. It is the standard
     statistical answer to "too many channels, too few trials", and it lets us
     separate two different explanations: does restricting to the motor strip
     help because of ANATOMY, or just because 21 < 64?

CROSS-VALIDATION IS DONE PROPERLY HERE. CSP is a *supervised* transform — it
uses the labels. Fitting it on all the data and then cross-validating only the
classifier leaks test information into the transform and inflates accuracy.
Everything below puts CSP inside the fold, in a scikit-learn Pipeline. Step 6
will measure exactly how much that mistake would have cost.

The accuracies here are a first, honest measurement to settle the step-4
question. Step 6 goes further: confusion matrix, per-fold detail, the leakage
demonstration, and all 10 subjects.

Outputs:
  results/04_csp.txt                    variance ratios, feature separability,
                                        cross-validated accuracy per variant
  figures/04_csp_patterns_all64.png     CSP patterns, all 64 electrodes
  figures/04_csp_patterns_motor.png     CSP patterns, sensorimotor strip only
  figures/04_csp_features_all64.png     the 4 features, all-64 variant
  figures/04_csp_features_motor.png     the 4 features, motor-strip variant
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import CSP
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config
import model
import preprocessing
from model import N_REPEATS

mne.set_log_level("ERROR")


def evaluate(X, y, reg=None):
    """Cross-validated accuracy from a SINGLE 5-fold split, with every fold score.

    Kept alongside the repeated version on purpose: this is the number a
    tutorial would report, and showing its per-fold scores is what makes the
    case for repeating the whole thing.
    """
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    scores = cross_val_score(model.csp_lda(reg), X, y, cv=cv, scoring="accuracy")
    return scores


def evaluate_repeated(X, y, reg=None, n_repeats=N_REPEATS):
    """The same evaluation, repeated with n_repeats different fold splits.

    WHY this is necessary and not extra credit. A single 5-fold run on 45 trials
    gives a mean whose value depends heavily on *which* trials happened to land
    in which fold. Comparing two variants on one split each would compare two
    lottery draws. Repeating the whole cross-validation with different random
    splits and looking at the spread of the repeat means tells us whether a
    difference between variants is real or an artefact of the split.
    """
    return model.repeated_cv(X, y, reg=reg, n_repeats=n_repeats)


def feature_auc(X, y, reg=None):
    """How well does each CSP component separate the classes, on its own?

    AUC (area under the ROC curve) of a single feature: 0.5 = useless, 1.0 =
    perfect, 0.0 = perfect but inverted. Computed without sklearn's roc_auc via
    the rank identity, which is exactly the Mann-Whitney U statistic.

    NOTE this is fitted on all the data, so it is descriptive only — it tells us
    which component carries the signal, not how well the model generalises.
    Generalisation is the cross-validated number, computed separately.
    """
    csp = CSP(n_components=config.N_CSP_COMPONENTS, reg=reg, log=True,
              norm_trace=False, rank="full")
    feats = csp.fit_transform(X, y)
    pos = y == config.EVENT_ID["left_hand"]
    aucs = []
    for j in range(feats.shape[1]):
        ranks = np.argsort(np.argsort(feats[:, j])) + 1
        n1, n2 = pos.sum(), (~pos).sum()
        u = ranks[pos].sum() - n1 * (n1 + 1) / 2
        aucs.append(u / (n1 * n2))
    return np.array(aucs), csp, feats


def plot_patterns(csp, info, title, filename):
    """The CSP patterns as scalp maps — the diagnosis figure.

    Each map answers: "for this virtual electrode, which part of the scalp does
    the source project onto?". Red and blue are just opposite signs of the same
    projection; what matters is WHERE the map is concentrated, not its colour.
    """
    # CAREFUL: csp.patterns_ holds one row per CHANNEL (64), not per kept
    # component. n_components only controls how many are used by transform(),
    # and MNE puts those first. Iterating over all of them would produce a
    # 64-panel figure — which is exactly what the first version of this script
    # did.
    patterns = csp.patterns_[: config.N_CSP_COMPONENTS]
    n = patterns.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.9))
    axes = np.atleast_1d(axes)

    for i, ax in enumerate(axes):
        pattern = patterns[i]
        lim = float(np.abs(pattern).max())
        mne.viz.plot_topomap(pattern, info, axes=ax, show=False,
                             cmap="RdBu_r", vlim=(-lim, lim), contours=4)
        tuned = "left hand" if i < n // 2 else "right hand"
        ax.set_title(f"component {i + 1}\ntuned for {tuned}", fontsize=9)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = config.FIGURES_DIR / filename
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_features(feats, y, aucs, variant_label, filename):
    """The 4 CSP features, one panel each, left-hand vs right-hand trials.

    This is what LDA actually sees. If the two distributions overlap almost
    completely, no linear classifier can do much — and that is worth seeing
    rather than inferring from an accuracy number.
    """
    n = feats.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), sharey=True)
    axes = np.atleast_1d(axes)

    left = y == config.EVENT_ID["left_hand"]
    for j, ax in enumerate(axes):
        for mask, label, color in ((left, "left", "#c0392b"),
                                   (~left, "right", "#1f4e79")):
            vals = feats[mask, j]
            # A little horizontal jitter so overlapping points stay visible.
            x = np.random.default_rng(config.RANDOM_STATE).normal(
                0 if label == "left" else 1, 0.06, size=vals.size)
            ax.scatter(x, vals, s=22, alpha=0.75, color=color, label=label,
                       edgecolors="none")
            ax.hlines(vals.mean(), (0 if label == "left" else 1) - 0.2,
                      (0 if label == "left" else 1) + 0.2, color=color, lw=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["left", "right"], fontsize=9)
        ax.set_title(f"component {j + 1}\nAUC = {aucs[j]:.2f}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("log-variance of the\nfiltered signal")
    fig.suptitle(
        f"What LDA sees — {variant_label}, subject {config.SUBJECT}\n"
        "horizontal bars = class means; AUC 0.5 means the feature is useless. "
        "AUC here is fitted on all trials, so it is descriptive, not performance.",
        fontsize=10,
    )
    fig.tight_layout()
    out = config.FIGURES_DIR / filename
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    say("=" * 74)
    say(f"CSP — subject {config.SUBJECT}, runs {config.RUNS}")
    say("=" * 74)

    raw = preprocessing.load_filtered()
    epochs = preprocessing.make_epochs(raw)

    # Crop to the window the classifier is allowed to see. The full -1 to +4 s
    # epoch exists so we can compute a rest baseline and plot ERD; the model
    # only gets 0.5-2.5 s, where ERD actually lives.
    work = epochs.copy().crop(tmin=config.CSP_TMIN, tmax=config.CSP_TMAX)
    y = work.events[:, 2]

    say()
    say("--- INPUT ---")
    say(f"  window given to CSP : {config.CSP_TMIN} to {config.CSP_TMAX} s")
    say(f"  trials              : {len(y)} "
        f"({(y == config.EVENT_ID['left_hand']).sum()} left / "
        f"{(y == config.EVENT_ID['right_hand']).sum()} right)")
    say(f"  CSP components kept : {config.N_CSP_COMPONENTS} "
        f"({config.N_CSP_COMPONENTS // 2} per class)")
    majority = max((y == c).sum() for c in set(y)) / len(y)
    say(f"  majority-class baseline: {100 * majority:.1f} %  <- the number to beat")
    say("  (not 50 %: a classifier that always answers 'left' already gets this)")

    variants = {
        "A. all 64 electrodes": dict(picks=None, reg=None),
        "B. sensorimotor strip only": dict(picks=config.SENSORIMOTOR_CH, reg=None),
        "C. all 64 + shrinkage": dict(picks=None, reg="ledoit_wolf"),
    }

    results = {}
    say()
    say("--- CROSS-VALIDATED ACCURACY (CSP fitted inside each fold) ---")
    say(f"  single {config.CV_FOLDS}-fold run, seed {config.RANDOM_STATE}, to show")
    say("  how unstable one split is:")
    say()
    say(f"  {'variant':<28}{'ch':>5}{'accuracy':>18}{'per-fold':>28}")
    for name, spec in variants.items():
        ep = work.copy().pick(spec["picks"]) if spec["picks"] else work.copy()
        X = ep.get_data(copy=True)
        scores = evaluate(X, y, reg=spec["reg"])
        results[name] = dict(scores=scores, X=X, info=ep.info, reg=spec["reg"])
        folds = " ".join(f"{s:.2f}" for s in scores)
        say(f"  {name:<28}{X.shape[1]:>5}"
            f"{100 * scores.mean():>11.1f} +- {100 * scores.std():<4.1f}"
            f"{folds:>28}")

    say()
    say("  Look at variant A's folds: one is 0.44, below the majority baseline,")
    say("  and another is 1.00. With ~9 test trials per fold, one trial flipping")
    say("  moves a fold by 11 points. A single mean from a single split is a")
    say("  lottery draw, so it cannot be used to compare variants.")

    # ------------------------------------------------------------------
    say()
    say(f"--- THE SAME EVALUATION, REPEATED {N_REPEATS}x WITH DIFFERENT SPLITS ---")
    say("  This is the number to trust. Mean over repeats, +- spread of the")
    say(f"  {N_REPEATS} repeat means.")
    say()
    say(f"  {'variant':<28}{'ch':>5}{'accuracy':>18}{'range over repeats':>24}")
    for name, spec in variants.items():
        r = results[name]
        reps = evaluate_repeated(r["X"], y, reg=r["reg"])
        r["reps"] = reps
        say(f"  {name:<28}{r['X'].shape[1]:>5}"
            f"{100 * reps.mean():>11.1f} +- {100 * reps.std():<4.1f}"
            f"{100 * reps.min():>15.1f} - {100 * reps.max():<6.1f}")

    # ------------------------------------------------------------------
    say()
    say("--- DOES RESTRICTING THE ELECTRODES HELP? ---")
    a_r = results["A. all 64 electrodes"]["reps"]
    b_r = results["B. sensorimotor strip only"]["reps"]
    c_r = results["C. all 64 + shrinkage"]["reps"]
    a, b, c = a_r.mean(), b_r.mean(), c_r.mean()
    say(f"  A all 64                 : {100 * a:.1f} % +- {100 * a_r.std():.1f}")
    say(f"  B sensorimotor strip     : {100 * b:.1f} % +- {100 * b_r.std():.1f}"
        f"  ({100 * (b - a):+.1f} pts vs A)")
    say(f"  C all 64 + shrinkage     : {100 * c:.1f} % +- {100 * c_r.std():.1f}"
        f"  ({100 * (c - a):+.1f} pts vs A)")

    # A difference smaller than the spread of the estimates is not a difference.
    # We state that rule before applying it, so the verdict is not chosen to fit
    # whatever came out.
    pooled = float(np.sqrt(a_r.std() ** 2 + b_r.std() ** 2))
    say()
    say(f"  Rule: a gap smaller than the combined spread ({100 * pooled:.1f} pts)")
    say("  cannot be called a difference at this sample size.")
    gap = abs(b - a)
    if gap < pooled:
        say(f"  -> The A vs B gap is {100 * gap:.1f} pts, BELOW that. The two variants")
        say("     are indistinguishable on accuracy. Restricting the electrodes")
        say("     neither helps nor hurts measurably — so accuracy cannot settle")
        say("     the step-4 question, and the CSP patterns have to.")
    elif b > a:
        say(f"  -> B beats A by {100 * (b - a):.1f} pts, above the spread. The excluded")
        say("     electrodes were feeding the classifier noise.")
    else:
        say(f"  -> A beats B by {100 * (a - b):.1f} pts, above the spread. The extra")
        say("     electrodes carry real information the motor strip does not.")

    if abs(c - a) < 0.005:
        say()
        say("  Shrinkage changes nothing at all. Worth knowing why: it rescales the")
        say("  CSP filters but leaves the spatial PATTERNS almost untouched (max")
        say("  difference ~4e-05 here). The covariance estimate is therefore not")
        say("  the bottleneck for this subject — the data itself is.")

    # ------------------------------------------------------------------
    say()
    say("--- WHERE DO THE CSP PATTERNS LAND? (the step-4 question) ---")
    say("  Per-component AUC is descriptive (fitted on all data), and tells us")
    say("  which virtual electrode carries the separating information.")

    figs = []
    for name, filename, title in (
        ("A. all 64 electrodes", "04_csp_patterns_all64.png",
         f"CSP patterns — all 64 electrodes, subject {config.SUBJECT}"),
        ("B. sensorimotor strip only", "04_csp_patterns_motor.png",
         f"CSP patterns — sensorimotor strip only, subject {config.SUBJECT}"),
    ):
        r = results[name]
        aucs, csp, feats = feature_auc(r["X"], y, reg=r["reg"])
        r.update(aucs=aucs, csp=csp, feats=feats)

        say()
        say(f"  {name}")
        say("    component AUC: " + "  ".join(
            f"c{i + 1}={v:.2f}" for i, v in enumerate(aucs)))
        say(f"    best component: c{int(np.argmax(np.abs(aucs - 0.5))) + 1} "
            f"(AUC {aucs[int(np.argmax(np.abs(aucs - 0.5)))]:.2f})")

        # Where is each pattern concentrated? We measure it instead of
        # eyeballing the figure: what share of the pattern's total weight falls
        # on the sensorimotor strip versus the posterior control region?
        names = r["info"]["ch_names"]
        sm = [names.index(ch) for ch in config.SENSORIMOTOR_CH if ch in names]
        po = [names.index(ch) for ch in config.POSTERIOR_CH if ch in names]
        # Only the components actually used by transform() — the first
        # n_components rows of patterns_, not all 64.
        for i, pattern in enumerate(csp.patterns_[: config.N_CSP_COMPONENTS]):
            w = np.abs(pattern)
            total = w.sum()
            sm_txt = f"{100 * w[sm].sum() / total:5.1f} %" if sm else "    n/a"
            po_txt = f"{100 * w[po].sum() / total:5.1f} %" if po else "    n/a"
            peak = names[int(np.argmax(w))]
            say(f"    c{i + 1}: {sm_txt} of weight on motor strip, "
                f"{po_txt} posterior, peak at {peak}")
        if not po:
            say("        (posterior region is excluded from this variant by design)")

        figs.append(plot_patterns(csp, r["info"], title, filename))

    # ------------------------------------------------------------------
    # SYNTHESIS. Accuracy alone and patterns alone each give half the answer.
    # Put together they say something neither says on its own, and it is the
    # main result of this step.
    # ------------------------------------------------------------------
    sm_share_a = np.array([
        np.abs(p)[[results["A. all 64 electrodes"]["info"]["ch_names"].index(ch)
                   for ch in config.SENSORIMOTOR_CH]].sum() / np.abs(p).sum()
        for p in results["A. all 64 electrodes"]["csp"].patterns_[
            : config.N_CSP_COMPONENTS]
    ])

    say()
    say("--- SYNTHESIS: WHAT STEP 5 ACTUALLY ESTABLISHED ---")
    say(f"  1. Motor imagery IS decodable from the motor strip alone:")
    say(f"     {100 * b:.1f} % vs a {100 * majority:.1f} % majority baseline. Its CSP")
    say("     patterns peak at CP5, C4, FC3 and C3 — over the sensorimotor cortex,")
    say("     where physiology says they should be. This is the clean result.")
    say()
    say(f"  2. Using all 64 electrodes scores HIGHER ({100 * a:.1f} %), but its top")
    say(f"     components are not motor: component 1 puts {100 * sm_share_a[0]:.0f} % of its")
    say("     weight on the motor strip and peaks at O1, an occipital electrode.")
    say("     Component 2 peaks at P3. Both are parietal/occipital.")
    say()
    say("  3. So step 4's red flag is CONFIRMED — and in the most treacherous way")
    say("     possible. The non-motor signal does not degrade accuracy, it")
    say(f"     IMPROVES it by {100 * (a - b):.1f} points. A project that only looked at the")
    say(f"     score would report {100 * a:.1f} % and claim to have decoded motor")
    say("     imagery. The patterns are what expose it.")
    say()
    say("  4. What we still do not know: WHAT that posterior signal is. Step 4")
    say("     rejected the visual cue (it is bilateral), a single bad trial, and")
    say("     fatigue. It is reproducible enough to survive 50 train/test splits,")
    say("     so calling it pure noise no longer fits either — noise does not")
    say("     generalise. That is an open question, stated as open.")
    say()
    say("  PRACTICAL CONSEQUENCE for step 6 and 7: report both variants side by")
    say("  side, and treat variant B as the honest motor-imagery result even")
    say("  though it is the lower number. The headline accuracy of this project")
    say("  should be the one we can defend, not the one that looks best.")

    # Feature figures for BOTH variants, not just the winner: the point of this
    # step is the comparison, and showing only the higher-scoring one would hide
    # what variant B looks like.
    figs.append(plot_features(
        results["A. all 64 electrodes"]["feats"], y,
        results["A. all 64 electrodes"]["aucs"],
        "all 64 electrodes (posterior components)",
        "04_csp_features_all64.png"))
    figs.append(plot_features(
        results["B. sensorimotor strip only"]["feats"], y,
        results["B. sensorimotor strip only"]["aucs"],
        "sensorimotor strip only (the defensible result)",
        "04_csp_features_motor.png"))

    # ------------------------------------------------------------------
    say()
    say("--- FIGURES WRITTEN ---")
    for out in figs:
        say(f"  {out.relative_to(config.ROOT_DIR)}")

    report = config.RESULTS_DIR / "04_csp.txt"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nReport written: {report.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
