"""
The model itself: CSP -> LDA, and how it is evaluated.

WHY this file appeared at step 6, like preprocessing.py appeared at step 3.
Step 5 built a CSP->LDA pipeline. Step 6 needed the same one. Same rule as
before: factor out on the second occurrence, not the first. Keeping two copies
would guarantee that one day they drift apart and two "identical" experiments
quietly stop being comparable.
"""

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import config

# How many times the whole cross-validation is repeated with different splits.
# Established in step 5: a single 5-fold split on 45 trials gave 77.8 % where
# ten splits give 71.3 % — the single split was 6.5 points optimistic.
N_REPEATS = 10


def csp_lda(reg=None):
    """The pipeline. Building it as a Pipeline is the whole point.

    CSP is a SUPERVISED transform: it uses the labels to build its spatial
    filters. Wrapping it in a Pipeline is what makes cross-validation re-fit it
    on the training half of every fold. Fitting CSP once outside and then
    cross-validating only the LDA leaks the test labels into the transform —
    step 6 measures exactly how many free accuracy points that buys.

    rank="full" is not a performance tweak masquerading as a choice: our data
    genuinely is full rank (64 channels, no re-referencing, no ICA, and
    np.linalg.matrix_rank returns 64), so MNE's automatic rank estimation has
    nothing to find. Stating it explicitly gives byte-identical results
    (71.33 % either way, verified) and avoids MNE's internal rank estimator,
    which crashes when joblib hands workers a read-only shared-memory array
    during parallel permutation testing.
    """
    return Pipeline([
        ("csp", CSP(n_components=config.N_CSP_COMPONENTS, reg=reg,
                    log=True, norm_trace=False, rank="full")),
        ("lda", LDA()),
    ])


def repeated_cv(X, y, reg=None, n_repeats=N_REPEATS):
    """Mean accuracy of each cross-validation repeat.

    Returns one number per repeat rather than a single grand mean, so callers can
    quote a spread that reflects the uncertainty of the estimate. On 45 trials
    that spread is the difference between a reportable result and a lottery
    ticket.
    """
    cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=n_repeats,
                                 random_state=config.RANDOM_STATE)
    scores = cross_val_score(csp_lda(reg), X, y, cv=cv, scoring="accuracy")
    # cross_val_score returns folds grouped by repeat, in order.
    return scores.reshape(n_repeats, config.CV_FOLDS).mean(axis=1)


def variant_data(epochs, picks=None):
    """Extract the (trials, channels, samples) array for one electrode variant.

    `picks=None` keeps all 64 electrodes; passing config.SENSORIMOTOR_CH
    restricts to the motor strip. The comparison between the two is the thread
    running from step 4 to step 7.
    """
    ep = epochs.copy().pick(picks) if picks else epochs.copy()
    return ep.get_data(copy=True), ep.info


def majority_baseline(y):
    """The accuracy of always answering the more frequent class.

    This, not 50 %, is what a classifier has to beat. With 23 left and 22 right
    trials it is 51.1 %, and a permutation test's chance distribution centres
    near here rather than at a coin flip.
    """
    return max((y == c).sum() for c in set(np.asarray(y))) / len(y)
