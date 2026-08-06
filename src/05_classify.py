"""
STEP 6 — LDA classification, honest cross-validation, and the classic mistake.

=========================================================================
WHY LDA
=========================================================================

After CSP we have 4 numbers per trial (the log-variance of each spatial filter's
output) and 45 trials. That ratio dictates the classifier: with 45 examples,
anything expressive will fit noise. LDA (Linear Discriminant Analysis) is about
the lowest-capacity option available — it draws a single straight boundary
between the two clouds of points, and it has no hyperparameters to tune.

LDA also pairs naturally with CSP features. It assumes each class is a Gaussian
blob with the same shape, differing only in position. `CSP(log=True)` returns
log-variances precisely because raw variances are skewed, and taking the log
makes them roughly Gaussian. So step 5 and step 6 were designed together: CSP +
log-variance + LDA is the field's standard baseline because each piece feeds the
next one's assumptions.

=========================================================================
THE CLASSIC MISTAKE, AND WHY THIS SCRIPT MEASURES IT
=========================================================================

CSP is a SUPERVISED transform: it uses the class labels to build its filters.
That makes the order of operations critical.

The wrong version, which appears in a great many tutorials and student projects:

    csp = CSP().fit(X, y)          # <- sees the labels of ALL trials
    features = csp.transform(X)
    cross_val_score(LDA(), features, y)

This looks like proper cross-validation — the LDA never sees its test fold. But
the *features* were built using the labels of the test trials. Information has
leaked from test to train through the transform, and the resulting accuracy is
optimistic by an amount nobody can guess from the number itself.

The right version puts CSP inside the fold, which a scikit-learn Pipeline does
automatically:

    Pipeline([("csp", CSP()), ("lda", LDA())])

Every step so far has used the correct version. This script does something extra:
it runs the wrong one on purpose and measures the gap, so the cost of the mistake
is a number in this repo rather than a warning in a comment.

=========================================================================
THREE THINGS WE CHECK BEYOND ACCURACY
=========================================================================

1. CONFUSION MATRIX. Accuracy hides asymmetry. 62 % could mean "both classes at
   62 %" or "left at 90 %, right at 33 %" — the second is a nearly useless
   decoder even though the headline is identical. Step 4 already found the two
   hands behave differently (right-hand imagery was lateralised, left-hand was
   not), so we have a specific reason to expect asymmetry here.

2. PERMUTATION TEST. With 45 trials, "above the 51.1 % baseline" is not the same
   as "above chance". We shuffle the labels many times, re-run the entire
   cross-validation on each shuffle, and see how often pure chance reaches our
   score. That gives a p-value that accounts for the small sample honestly,
   instead of comparing to a theoretical 50 %.

3. BOTH VARIANTS. All 64 electrodes and the sensorimotor strip, side by side, as
   decided in step 4 and justified in METHODOLOGY.md: variant B is the result we
   are willing to defend, variant A is reported for transparency.

Outputs:
  results/05_classification.txt        all numbers
  figures/05_confusion_matrices.png    per-class performance, both variants
  figures/05_leakage.png               cost of fitting CSP outside the CV loop
  figures/05_permutation.png           our score against the chance distribution
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)

import config
import model
import preprocessing
from model import N_REPEATS, csp_lda, majority_baseline, repeated_cv

mne.set_log_level("ERROR")

# Number of label shufflings for the permutation test. 500 gives a p-value
# resolution of about 1/501 = 0.002, which is far finer than anything we would
# claim with 45 trials. More would be wasted compute.
N_PERMUTATIONS = 500

CLASS_NAMES = ["left hand", "right hand"]


def leaky_cv(X, y, n_repeats=N_REPEATS):
    """The MISTAKE, on purpose: CSP fitted once on everything, before splitting.

    We are not being careless here — we are quantifying carelessness. CSP sees
    every label, including those of the trials it will later be tested on, and
    only the LDA is cross-validated. The gap against repeated_cv() is the price
    of the error.
    """
    csp = CSP(n_components=config.N_CSP_COMPONENTS, log=True,
              norm_trace=False, rank="full")
    features = csp.fit_transform(X, y)          # <- the leak happens here
    cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=n_repeats,
                                 random_state=config.RANDOM_STATE)
    scores = cross_val_score(LDA(), features, y, cv=cv, scoring="accuracy")
    return scores.reshape(n_repeats, config.CV_FOLDS).mean(axis=1)


def pooled_confusion(X, y, n_repeats=N_REPEATS):
    """Confusion matrix accumulated over several independent CV splits.

    cross_val_predict gives one out-of-fold prediction per trial, so a single
    call yields a confusion matrix built from 45 predictions — too few to read
    confidently. Summing over n_repeats different splits gives a more stable
    picture of which class the decoder gets wrong.
    """
    total = np.zeros((2, 2), dtype=int)
    for i in range(n_repeats):
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                             random_state=config.RANDOM_STATE + i)
        pred = cross_val_predict(csp_lda(), X, y, cv=cv)
        total += confusion_matrix(
            y, pred, labels=[config.EVENT_ID["left_hand"],
                             config.EVENT_ID["right_hand"]])
    return total


def permutation(X, y, n_permutations=N_PERMUTATIONS):
    """Is our accuracy reachable by chance on this many trials?

    We shuffle the labels n_permutations times and re-run the FULL
    cross-validation on each shuffle. The p-value is the fraction of shuffles
    that matched or beat the real score. For a small sample this is far more
    informative than comparing to a theoretical 50 %: it measures what this exact
    pipeline achieves on this exact data when the labels carry no information.

    Written as a plain loop rather than sklearn's permutation_test_score for two
    reasons, both discovered the hard way:
      - with n_jobs>1, joblib hands the workers a shared-memory copy of X and
        MNE's internal rank estimator raises on it ("data copying was not
        requested by copy=None"). The rank estimator is reached from two
        different places inside CSP, so no CSP argument avoids it.
      - measured on this data, threads are slower than serial anyway (BLAS
        contention: 53 s vs 27 s for 200 permutations), so the parallelism was
        buying nothing.
    A loop we control costs about a minute and has no such surprises.
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

    # The +1 in numerator and denominator is not cosmetic: with a finite number
    # of shufflings you can never observe p = 0, and reporting p = 0 would claim
    # more certainty than the procedure can deliver. This is the standard
    # correction (Phipson & Smyth, 2010); the smallest p we can report here is
    # 1/(n+1).
    p = (1 + int((null >= observed).sum())) / (1 + n_permutations)
    return observed, null, p


# ==========================================================================
# FIGURES
# ==========================================================================
def plot_confusions(confusions, majority):
    """One confusion matrix per variant, in percentages of each true class.

    Rows are normalised so each row sums to 100 %: that turns the diagonal into
    per-class recall, which is the quantity that reveals asymmetry. A decoder
    that answers "left" most of the time shows up immediately as a bright left
    column and a dark bottom-left cell.
    """
    fig, axes = plt.subplots(1, len(confusions), figsize=(5.4 * len(confusions), 4.6))
    axes = np.atleast_1d(axes)

    for ax, (label, cm) in zip(axes, confusions.items()):
        pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{pct[i, j]:.0f} %\n({cm[i, j]})",
                        ha="center", va="center", fontsize=11,
                        color="white" if pct[i, j] > 55 else "#222")
        ax.set_xticks([0, 1], CLASS_NAMES)
        ax.set_yticks([0, 1], CLASS_NAMES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        recalls = np.diag(pct)
        ax.set_title(f"{label}\nrecall: left {recalls[0]:.0f} %, "
                     f"right {recalls[1]:.0f} %", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, label="% of true class")

    fig.suptitle(
        f"Out-of-fold confusion matrices, subject {config.SUBJECT} — "
        f"pooled over {N_REPEATS} CV splits\n"
        f"counts in brackets are trials out of {N_REPEATS} x 45 predictions; "
        f"majority-class baseline {100 * majority:.1f} %",
        fontsize=10,
    )
    fig.tight_layout()
    out = config.FIGURES_DIR / "05_confusion_matrices.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_leakage(honest, leaky, majority):
    """Honest vs leaked accuracy, per variant. The point of the figure is the gap."""
    labels = list(honest.keys())
    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    h_means = [honest[k].mean() * 100 for k in labels]
    l_means = [leaky[k].mean() * 100 for k in labels]
    h_err = [honest[k].std() * 100 for k in labels]
    l_err = [leaky[k].std() * 100 for k in labels]

    ax.bar(x - w / 2, h_means, w, yerr=h_err, capsize=4, color="#1f4e79",
           label="correct: CSP fitted inside each fold")
    ax.bar(x + w / 2, l_means, w, yerr=l_err, capsize=4, color="#c0392b",
           label="leaked: CSP fitted once on all trials")

    for xi, (h, l) in enumerate(zip(h_means, l_means)):
        ax.annotate(f"+{l - h:.1f} pts", xy=(xi, max(h, l) + 3.5), ha="center",
                    fontsize=10, color="#c0392b", weight="bold")

    ax.axhline(100 * majority, color="0.4", ls="--", lw=1,
               label=f"majority-class baseline ({100 * majority:.1f} %)")
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("cross-validated accuracy (%)")
    ax.set_ylim(40, 100)
    ax.set_title(
        "What the classic mistake buys you for free\n"
        "error bars = spread over the 10 CV repeats",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = config.FIGURES_DIR / "05_leakage.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_permutation(perms, majority):
    """Our score against the distribution of scores obtained on shuffled labels."""
    fig, axes = plt.subplots(1, len(perms), figsize=(6.0 * len(perms), 4.4),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (label, (score, null, p)) in zip(axes, perms.items()):
        ax.hist(100 * null, bins=24, color="0.72", edgecolor="white",
                label=f"shuffled labels (n={len(null)})")
        ax.axvline(100 * score, color="#c0392b", lw=2.2,
                   label=f"actual score {100 * score:.1f} %")
        ax.axvline(100 * majority, color="#1f4e79", lw=1.4, ls="--",
                   label=f"majority baseline {100 * majority:.1f} %")
        ax.set_xlabel("accuracy (%)")
        ax.set_title(f"{label}\np = {p:.4f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("number of shufflings")
    fig.suptitle(
        f"Permutation test, subject {config.SUBJECT} — is the score reachable "
        "by chance?\n"
        "the grey distribution is what this pipeline achieves when the labels "
        "are meaningless",
        fontsize=10,
    )
    fig.tight_layout()
    out = config.FIGURES_DIR / "05_permutation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ==========================================================================
def main():
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    say("=" * 74)
    say(f"CLASSIFICATION — subject {config.SUBJECT}, runs {config.RUNS}")
    say("=" * 74)

    raw = preprocessing.load_filtered()
    epochs = preprocessing.make_epochs(raw)
    work = epochs.copy().crop(tmin=config.CSP_TMIN, tmax=config.CSP_TMAX)
    y = work.events[:, 2]
    majority = majority_baseline(y)

    variants = {
        "all 64 electrodes": None,
        "sensorimotor strip": config.SENSORIMOTOR_CH,
    }
    data = {label: model.variant_data(work, picks)[0]
            for label, picks in variants.items()}

    say()
    say("--- SETUP ---")
    say(f"  model            : CSP({config.N_CSP_COMPONENTS} components) -> LDA")
    say(f"  window           : {config.CSP_TMIN} to {config.CSP_TMAX} s")
    say(f"  trials           : {len(y)} "
        f"({(y == config.EVENT_ID['left_hand']).sum()} left / "
        f"{(y == config.EVENT_ID['right_hand']).sum()} right)")
    say(f"  cross-validation : {config.CV_FOLDS}-fold stratified, "
        f"repeated {N_REPEATS}x")
    say(f"  baseline to beat : {100 * majority:.1f} % (majority class)")

    # ------------------------------------------------------------------
    say()
    say("--- 1. ACCURACY, DONE CORRECTLY ---")
    honest = {}
    for label, X in data.items():
        honest[label] = repeated_cv(X, y)
        r = honest[label]
        say(f"  {label:<22}{X.shape[1]:>3} ch  "
            f"{100 * r.mean():5.1f} % +- {100 * r.std():4.1f}  "
            f"(repeats span {100 * r.min():.1f} - {100 * r.max():.1f} %)")

    # ------------------------------------------------------------------
    say()
    say("--- 2. THE CLASSIC MISTAKE: CSP FITTED OUTSIDE THE CV LOOP ---")
    say("  Same data, same folds, same classifier. The only change is that CSP")
    say("  gets to see every label before the split.")
    say()
    leaky = {}
    for label, X in data.items():
        leaky[label] = leaky_cv(X, y)
        gap = 100 * (leaky[label].mean() - honest[label].mean())
        say(f"  {label:<22} correct {100 * honest[label].mean():5.1f} %   "
            f"leaked {100 * leaky[label].mean():5.1f} %   "
            f"free gain {gap:+5.1f} pts")
    say()
    say("  That gain is entirely fictitious. It would not survive contact with")
    say("  one new trial, because it comes from CSP having already seen the")
    say("  answers. Nothing in the leaked number looks wrong — which is exactly")
    say("  what makes this the most dangerous error in the CSP literature.")

    # ------------------------------------------------------------------
    say()
    say("--- 3. CONFUSION MATRICES (out-of-fold, pooled over 10 splits) ---")
    say("  Accuracy hides asymmetry: 62 % could be 62/62 or 90/33.")
    say()
    confusions = {}
    for label, X in data.items():
        cm = pooled_confusion(X, y)
        confusions[label] = cm
        pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        say(f"  {label}")
        say(f"    {'':<14}{'pred left':>12}{'pred right':>12}")
        for i, name in enumerate(CLASS_NAMES):
            say(f"    true {name:<9}{pct[i, 0]:>11.0f} %{pct[i, 1]:>11.0f} %")
        say(f"    recall: left {pct[0, 0]:.0f} %, right {pct[1, 1]:.0f} %  "
            f"(gap {abs(pct[0, 0] - pct[1, 1]):.0f} pts)")
        say()

    # ------------------------------------------------------------------
    say("--- 4. PERMUTATION TEST: is this reachable by chance? ---")
    say(f"  {N_PERMUTATIONS} label shufflings, full cross-validation re-run on each.")
    say("  IMPORTANT: this test uses ONE fixed 5-fold split, the same one for the")
    say("  real labels and for every shuffling — that is what makes the comparison")
    say("  fair. So the score below is the single-split score, not the repeated-CV")
    say("  headline from section 1; step 5 showed the two differ by several points.")
    say("  The p-value answers 'is there signal here at all', not 'how accurate".rstrip() + " is")
    say("  the decoder'. Those are two different questions with two different")
    say("  numbers, and conflating them is how small-sample results get oversold.")
    say()
    perms = {}
    for label, X in data.items():
        score, null, p = permutation(X, y)
        perms[label] = (score, null, p)
        say(f"  {label:<22} score {100 * score:5.1f} %   "
            f"chance distribution {100 * null.mean():.1f} % "
            f"+- {100 * null.std():.1f}   p = {p:.4f}")
    say()
    say("  Note the chance distribution is centred near the majority baseline,")
    say("  not at 50 %: with unbalanced classes, a useless classifier still")
    say("  scores above a coin flip. This is why we never quote 50 % as chance.")

    # ------------------------------------------------------------------
    say()
    say("--- 5. WHAT THIS STEP ESTABLISHES ---")
    hb = honest["sensorimotor strip"].mean()
    ha = honest["all 64 electrodes"].mean()
    pb = perms["sensorimotor strip"][2]
    pa = perms["all 64 electrodes"][2]
    say("  Headline result -- sensorimotor strip, the defensible one:")
    say(f"    accuracy  {100 * hb:.1f} % +- {100 * honest['sensorimotor strip'].std():.1f}"
        f"   ({N_REPEATS}x {config.CV_FOLDS}-fold, vs {100 * majority:.1f} % baseline)")
    say(f"    signal    p = {pb:.4f}   ({N_PERMUTATIONS} shufflings on one fixed split,")
    say(f"              where the real labels scored {100 * perms['sensorimotor strip'][0]:.1f} % "
        f"vs {100 * perms['sensorimotor strip'][1].mean():.1f} % for shuffled ones)")
    say()
    say("  Reported for transparency -- all 64 electrodes:")
    say(f"    accuracy  {100 * ha:.1f} % +- {100 * honest['all 64 electrodes'].std():.1f}")
    say(f"    signal    p = {pa:.4f}")
    say()
    sig_b = pb < 0.05
    sig_a = pa < 0.05
    if sig_b:
        say("  The sensorimotor result is statistically above chance. Given the")
        say("  motor-centred CSP patterns from step 5, this is a genuine, if")
        say("  modest, motor-imagery decoder.")
    else:
        say("  The sensorimotor result is NOT statistically distinguishable from")
        say("  chance at this sample size. That is a real finding, not a failure of")
        say("  the pipeline: 45 trials is simply too few for an effect this size.")
    if sig_a and not sig_b:
        say("  The 64-electrode result IS significant while the motor one is not —")
        say("  another reminder that significance and validity are different")
        say("  questions. See METHODOLOGY.md on why the higher score is not the")
        say("  headline.")

    say()
    say("  What step 7 must add: all 10 subjects. Every number here describes ONE")
    say("  person, and between-subject variability in motor imagery is larger than")
    say("  any difference measured on this page.")

    # ------------------------------------------------------------------
    say()
    say("--- FIGURES WRITTEN ---")
    for out in (plot_confusions(confusions, majority),
                plot_leakage(honest, leaky, majority),
                plot_permutation(perms, majority)):
        say(f"  {out.relative_to(config.ROOT_DIR)}")

    report = config.RESULTS_DIR / "05_classification.txt"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nReport written: {report.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
