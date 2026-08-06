"""
Project parameters, all in one place.

WHY does this file exist?
Because the frequency band, the time window and the choice of subjects are
*scientific decisions*, not implementation details. If they are hard-coded in
the middle of four different scripts, two things happen:
  1. you change a value in one script and forget the others -> your results no
     longer mean anything;
  2. anyone reading the repo (a recruiter, a professor, you in six months) has
     to read all the code to find out what you actually did.
Here, everything is visible in 30 seconds. This is also the file you edit to
test a variant ("what if I used 4-40 Hz instead?").
"""

from pathlib import Path

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------
# Paths are derived from the location of THIS file, never hard-coded as
# absolute ("/Users/Nono/..."). Result: the repo works on anyone's machine,
# and from whatever directory the script is launched.
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / "data"        # raw EDF downloads (not in git)
FIGURES_DIR = ROOT_DIR / "figures"  # PNGs shown in the README
RESULTS_DIR = ROOT_DIR / "results"  # scores, logs, confusion matrices

# Directories are created on demand: right after a `git clone`, `data/` does
# not exist (it is gitignored) and a script writing into it would crash.
for _d in (DATA_DIR, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# DATA: which subject, which recordings
# --------------------------------------------------------------------------
# PhysioNet "EEG Motor Movement/Imagery" dataset: 109 subjects, 64 electrodes,
# 14 recordings ("runs") each. The runs do NOT all contain the same task —
# this is the classic trap of this dataset:
#   runs 3, 7, 11  -> EXECUTED  movement, left hand / right hand
#   runs 4, 8, 12  -> IMAGINED  movement, left hand / right hand   <-- ours
#   runs 5, 9, 13  -> EXECUTED  movement, both fists / both feet
#   runs 6, 10, 14 -> IMAGINED  movement, both fists / both feet
# We want left vs right *motor imagery*, so runs 4, 8, 12.
#
# CAREFUL — this choice places us at LEVEL 3 of motor intent:
#   level 1  executed movement  : the hand actually moves
#   level 2  attempted movement : a real motor command, blocked downstream
#                                 (paralysed patient) -> the paradigm used in
#                                 clinical BCI trials such as BrainGate
#   level 3  motor imagery      : mental rehearsal, no attempt        <-- here
# This is not a detail. We do not train on executed movement because
# (a) the signal would be contaminated by artifacts of the movement itself, and
# (b) a paralysed patient cannot produce that signal at all, so a model trained
#     on it would be useless for the intended use case.
# See the "Context and limitations" section of the README for the full argument.
RUNS = [4, 8, 12]

# We start with a single subject to build and debug the pipeline, then widen to
# several subjects at the results stage (between-subject variability in BCI is
# ENORMOUS, and it is part of the story worth telling).
SUBJECT = 1
SUBJECTS_ALL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Labels of the two classes as annotated in the EDF files.
# In these runs: T1 = imagined left fist, T2 = imagined right fist.
# Verified against the real files in step 2 (see results/01_raw_exploration.txt):
# MNE maps annotation strings to integer codes alphabetically, T0->1, T1->2,
# T2->3, which is exactly what the values below assume.
EVENT_ID = {"left_hand": 2, "right_hand": 3}

# The dataset annotations also contain T0 = rest period, which we do NOT use
# here (we run a 2-class left vs right problem). Note for later: adding T0 as a
# third "idle" class is the first step towards a genuinely usable BCI, because a
# real system has to know how to *do nothing* when the user is not commanding
# anything. Detailed in "Long-term vision" (README).
REST_EVENT_ID = {"rest": 1}


# --------------------------------------------------------------------------
# PREPROCESSING (explained and justified in steps 3 and 4)
# --------------------------------------------------------------------------
F_LOW, F_HIGH = 8.0, 30.0   # band-pass in Hz: mu (8-12) + beta (13-30) rhythms

EPOCH_TMIN, EPOCH_TMAX = -1.0, 4.0   # window cut around the cue, in seconds
CSP_TMIN, CSP_TMAX = 0.5, 2.5        # sub-window actually fed to the classifier


# --------------------------------------------------------------------------
# ELECTRODE GROUPS
# --------------------------------------------------------------------------
# These are scientific choices, not helper lists, which is why they live here.
#
# SENSORIMOTOR_CH is the classic "sensorimotor strip": the FC, C and CP rows.
# It is where motor imagery must appear if it appears at all. We use it twice:
#   - in step 4, to ask WHERE the between-class difference sits;
#   - in step 5, as a restricted alternative to all 64 electrodes, to test
#     whether the other 43 help or just feed the classifier noise.
SENSORIMOTOR_CH = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
]

# POSTERIOR_CH is the parietal/occipital region: visual and attentional
# activity, NOT motor. It exists as a control group. If a between-class
# difference is bigger here than over the sensorimotor strip, whatever separates
# the classes is probably not motor imagery — which is exactly what step 4 found.
POSTERIOR_CH = [
    "P3", "P1", "Pz", "P2", "P4",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]


# --------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------
N_CSP_COMPONENTS = 4   # number of spatial filters kept (2 per class)
CV_FOLDS = 5           # number of cross-validation folds
RANDOM_STATE = 42      # fixed seed -> identical folds on every run
