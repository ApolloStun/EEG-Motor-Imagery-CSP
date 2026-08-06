# Methodology — the reasoning behind every step

This document is the project's lab notebook. The [README](README.md) says what
the project *is*; this file says **how each decision was reached**, including
the ones that turned out to be wrong.

Each step follows the same four questions:

1. **What we did** — concretely.
2. **Why** — the reasoning, not just the outcome.
3. **How we got to that choice** — what the alternatives were and what settled it.
4. **What it implies for the rest** — what the next steps inherit from it.

It is written for someone starting out in EEG signal processing, so it explains
things a paper would assume. It is updated as the project progresses, not
back-filled at the end. Numbers quoted here come from the reports in
`results/`, regenerable by running the scripts.

**Status:** all 8 steps complete. Open questions are listed at the end.

---

## A rule that applies to the whole project

> Verify assumptions against the data before building on them.

The reason is specific to this kind of work: **a misunderstanding of the data
never shows up in the final accuracy.** The model runs, prints a number, and the
number is wrong for a reason no metric reveals. Picking the wrong runs, assuming
the wrong sampling rate, or missing a third event marker all produce a pipeline
that "works" and conclusions that are false.

So every step below ends with a check, and the checks are in the code, not in
our heads. **19 times** a check contradicted what we believed — every one of them
is in the running log at the bottom rather than quietly fixed. Six of those were
bugs in this project's own code, and two of them produced figures that looked like
real physiology.

---

## Step 1 — Project setup

### What we did
Created the repo skeleton: `src/ data/ figures/ results/`, a virtual
environment, `requirements.txt` with pinned versions, a `.gitignore`, and
`src/config.py` holding every parameter of the project.

### Why
Three of these four are about someone else being able to re-run this — a
recruiter, a professor, or the author six months later.

- **A virtual environment** keeps the project's packages separate from the
  system Python. Without it, installing MNE for this project can break another
  project that needs a different NumPy.
- **Pinned versions** (`mne==1.8.0`, not `mne`) are the difference between "it
  worked on my machine" and reproducible work. Without the `==`, someone
  installing in six months gets a different MNE and may get different numbers.
- **`.gitignore` excludes `data/` and `.venv/`** because a repo should contain
  only what is *not* regenerable. The environment rebuilds from
  `requirements.txt`; the EDF files re-download from PhysioNet. `git status`
  shows a handful of files instead of 227 MB.
- **`config.py`** exists because the frequency band, the time window and the
  subject list are *scientific decisions*, not implementation details. Hard-code
  them in four scripts and you will eventually change one and forget the others
  — at which point the results mean nothing, silently.

### How we got to that choice
The layout is not invented: it copies two existing projects of the same author,
so the whole portfolio is navigated the same way. The pinned versions were read
off a working environment rather than chosen — the machine runs Python 3.9.6,
and `mne 1.8.0` / `numpy 2.0.2` is the last combination compatible with it.
Constraint first, decision second.

### What it implies
Every later step imports from `config.py` instead of hard-coding. Testing a
variant ("what about 4–40 Hz?") means editing one line, and the change
propagates everywhere by construction.

---

## Step 2 — Loading and exploring the raw signal

Script: [`src/01_explore_raw.py`](src/01_explore_raw.py) ·
Report: `results/01_raw_exploration.txt`

### What we did
Downloaded subject 1, opened the files, and verified every assumption in
`config.py` against the actual data. No modelling at all.

Facts established:

| | |
|---|---|
| Electrodes | 64, 10-10 system |
| Sampling rate | **160 Hz** |
| Duration | 125 s per run × 3 = 375 s |
| Size | 7.8 MB |
| Usable trials | 23 left + 22 right = **45** |
| Rest markers (`T0`) | 45, 4.20 s each |
| Trial duration (`T1`/`T2`) | 4.10 s each |

### Why this step exists at all
Because it is the cheapest possible insurance. Every fact above later turns into
a decision: 160 Hz sets what frequencies are even measurable (Nyquist: nothing
above 80 Hz); 4.10 s trials set what epoch window is legal; 45 trials set how
much we are allowed to conclude from one subject.

### How we got to the choices in this step

**Runs 4, 8, 12 rather than 3, 7, 11.** The dataset's 14 runs do not contain
the same task. Runs 3/7/11 are *executed* movement, 4/8/12 are *imagined*. We
use imagined, deliberately accepting lower accuracy, because executed movement
(a) is contaminated by muscle artifacts that correlate with the class, so a
classifier can score well by learning the muscle instead of the cortex, and
(b) cannot be produced at all by a paralysed person, which is the use case that
motivates the project. The full three-level argument is in the README.

**We checked the `T0` marker before relying on it.** The long-term plan (README)
needs a "rest / idle" class. Rather than assume the dataset has one, the script
counts the annotations: 45 rest events of 4.20 s, balanced against the 45
imagery trials. Now the plan rests on a measurement.

**We checked the event → integer mapping.** MNE converts annotation strings to
integer codes *alphabetically*: `T0→1, T1→2, T2→3`. `config.EVENT_ID` assumed
exactly that. Had it been wrong, the two classes would have been silently
swapped and every result would have looked plausible while being inverted. This
is the archetype of a bug no accuracy score can reveal.

### What it implies
- The epoch window of step 4 must fit inside 4.10 s — this is what makes
  `-1 to +4 s` legal rather than arbitrary.
- The `-1 to 0 s` baseline falls inside a genuine `T0` rest period, so it is a
  real rest reference.
- **45 trials is the number that constrains the whole project.** With 5-fold
  cross-validation each fold tests on ~9 trials, so a single trial flipping
  moves accuracy by ~11 points. Any single accuracy figure is meaningless
  without its spread — which is why step 7 will report variability, and why
  `config.SUBJECTS_ALL` exists.

### Bonus finding
The raw power spectrum already shows a bump at 10–12 Hz on C3 and C4 that is
absent on the frontal electrode Fp1. That is the mu rhythm, visible by eye
before any processing — encouraging for this subject. It also shows a sharp
spike at 60 Hz: the US mains supply, not the brain.

---

## Step 3 — Band-pass filtering, 8–30 Hz

Script: [`src/02_filter.py`](src/02_filter.py) · Report: `results/02_filtering.txt`

### What we did
Applied a zero-phase FIR band-pass filter, 8–30 Hz, to each run separately
before concatenating, and measured the result rather than trusting it.

### Why 8–30 Hz — three reasons, in order of importance

**1. That is where the signal lives.** The sensorimotor cortex oscillates at
rest in the mu (8–12 Hz) and beta (13–30 Hz) bands. Imagining a movement breaks
that synchronisation on the opposite side of the brain, so power drops locally.
The useful signal is an oscillation that *disappears*, on one side only.

**2. Because CSP works on variance, and variance is dominated by low
frequencies.** This is the reason that actually matters, and it is measurable.
In the raw signal:

```
0.5-4 Hz   (drift, delta)       65.8 %
4-8 Hz     (theta)              12.4 %
8-12 Hz    (mu)     <- useful    5.8 %
13-30 Hz   (beta)   <- useful    9.4 %
30-79 Hz   (EMG, mains)          5.1 %
```

**78 % of the power sits below 8 Hz. The useful band holds 15 %.** CSP (step 5)
maximises a *variance ratio* between classes, and variance is power. Fed the raw
signal, it would optimise mostly on sweat, electrode impedance and breathing.
Filtering is not cosmetic cleanup — it is what makes CSP applicable at all.

**3. It removes two contaminations for free.** Eye movements below (< 4 Hz),
muscle activity (EMG) above — which grows from ~20 Hz and dominates past 30 —
plus the 60 Hz mains spike. On a motor-imagery dataset, removing EMG has a
specific value: it is the guarantee we are not classifying a micro muscle
contraction instead of an intention.

### How we got to that choice
8–30 Hz is the standard band in the CSP literature, so it was the starting point
rather than a discovery. What we added is the *justification by measurement*:
the 78 % figure above was computed, not looked up, and it is what turns a
convention into a reason. The filter was then verified frequency by frequency
(−50 dB at 2 and 5 Hz, −55 dB at 60 Hz, 0.0 dB at 10 and 20 Hz) instead of
assumed to work.

### Two consequences of the filter's length
MNE builds a filter of 265 coefficients = **1.66 s of context** per output
sample. Two operational rules follow, and both are implemented in
`preprocessing.py`:

1. **Filter the continuous signal before epoching, never after.** A 5 s epoch
   cannot supply 1.66 s of context at its edges, so its edges would be
   distorted — and the edges are exactly where the baseline lives.
2. **Filter each run separately, then concatenate.** Runs glued end to end have
   an abrupt discontinuity at the junction; the filter reads that jump as a very
   fast oscillation and smears it over hundreds of milliseconds, manufacturing
   an artifact.

### What this filter costs us — stated, not hidden

- **Slow motor potentials (< 3 Hz) are gone.** There is a genuine
  low-frequency cortical motor response (MRCP / readiness potential) that
  carries movement information, and the 8 Hz high-pass removes it entirely.
  This matters for the project's long-term direction: the literature on decoding
  *attempted* movements (level 2 in the README) relies precisely on
  low-frequency EEG, around 0.3–3 Hz. **The band we discard here is the band we
  would need to recover to work on attempted movement.** 8–30 Hz is right for
  level 3, not universally right.
- **Gamma (> 30 Hz) is gone.** Largely theoretical: through the skull it is hard
  to separate from EMG, and at 160 Hz sampling nothing above 80 Hz is
  measurable anyway.

### What it implies
Amplitude drops from 53 to 20 µV — not a loss of signal, the removal of drift
that was inflating amplitude without carrying information. Everything downstream
operates on this filtered signal, and its variance is now band power by
construction, which is exactly what CSP needs.

---

## Step 4 — Epoching, and a first look at ERD

Script: [`src/03_epochs.py`](src/03_epochs.py) · Report: `results/03_epochs.txt`

### What we did
Cut 375 s of continuous signal into 45 labelled trials of 5.01 s
(`-1 to +4 s` around each cue), then measured whether the physiology behaves as
theory says before trusting any of it.

Result shape: `(45 trials, 64 channels, 801 samples)`.

### Why the specific window
- **`-1 to +4 s`**: trials last 4.10 s (measured in step 2), so ending at
  +4.0 s stays inside the trial — verified: the window ends 0.10 s before the
  next marker. The `-1 to 0 s` part falls inside a `T0` rest period, giving a
  genuine rest baseline.
- **`0.5 to 2.5 s` for the classifier**: ERD takes a few hundred milliseconds to
  develop, so 0–0.5 s is mostly the cue response and noise; and ERD is
  strongest early while attention drifts later. These two are **conventional
  choices from the literature, not tuned on our results** — tuning a window on
  the test score is a form of leakage we return to in step 6.

### Two deliberate omissions

**No baseline correction (`baseline=None`).** Baseline correction subtracts a
reference mean from each epoch, which is essential for evoked-potential
*amplitudes*. Our features are band power, i.e. a variance, and subtracting a
constant does not change a variance. Applying it would be harmless but
misleading about what analysis we are doing. We use the baseline window
explicitly and visibly instead, to express ERD as a percent change.

**No automatic artifact rejection.** MNE can drop epochs above an amplitude
threshold. With ~22 trials per class, silently dropping trials would change the
problem while hiding that it changed. Instead we inspect and report: peak
amplitude median 124 µV, 5 of 45 trials above 150 µV, all kept. Step 6 will tell
us whether that cost anything.

### What we found — and it is mixed

**ERD is present.** Both classes show band power dropping below rest from about
0.5 s after the cue, by 4–19 %. The direction is right: activation shows up as
*less* signal, which is the counter-intuitive core of ERD.

**Lateralisation is weak.** Theory says imagining the left hand should
desynchronise the *right* cortex (C4) more, and vice versa. Measured over
0.5–2.5 s:

| | C3 (left hemi) | C4 (right hemi) |
|---|---|---|
| imagined left hand | −19.0 % | −18.2 % |
| imagined right hand | −13.3 % | −4.1 % |

Read that table one row at a time, because the two rows do not say the same
thing:

- **Imagined RIGHT hand — matches theory.** C3 (left hemisphere) drops 13.3 %
  while C4 drops only 4.1 %: a **9.2-point** gap in the predicted direction.
  Motor control crosses over, so a right-hand movement is driven by the left
  hemisphere, and that is exactly the side that desynchronises. This row is a
  clean textbook contralateral ERD.
- **Imagined LEFT hand — does not.** C3 drops 19.0 % and C4 drops 18.2 %: a
  **0.8-point** gap. The two hemispheres desynchronise essentially equally.
  There is strong activation, but no usable lateralisation. Note the direction
  is not even wrong — there simply is no side.

The overall lateralisation contrast is +8.4 points, correct in sign and above
the 3-point noise floor we fixed *before* looking at the number. So the summary
is **weak but present**, and it is carried by one class out of two.

**This is a mixed result and it is stated as such: neither a failure nor a
success.** A failure would be a contrast with the wrong sign — imagery
desynchronising the same-side cortex. A success would be both rows showing a
clear contralateral gap. We have one row of each. Writing it up as "contralateral
ERD confirmed" would be false; writing it up as "no ERD found" would also be
false.

**Why might the left hand show nothing?** Two explanations we cannot separate
with the data we have:

1. **Too few trials.** 23 left-hand trials is very little for an effect of this
   size. A 0.8-point gap is far below what ~23 trials can resolve, so a real
   lateralisation of, say, 5 points could easily be invisible here. This is the
   most likely explanation, and it is a limitation of the measurement, not of
   the brain.
2. **This subject genuinely does not lateralise left-hand imagery.** Motor
   imagery ability varies a lot between people (see "BCI illiteracy" in the
   README), and asymmetry between the dominant and non-dominant hand is a real
   phenomenon — most people are right-handed, and imagery of the non-dominant
   hand is often weaker and less focal.

**What would settle it:** running the same measurement across
`config.SUBJECTS_ALL`. If the left-hand asymmetry disappears when averaged over
10 subjects, explanation 1 was right and it was noise. If it persists, it is a
real property worth reporting. Deferred to step 7, and it is the main reason
step 7 is not optional.

**A red flag, and where it led.** A classifier does not care about ERD; it cares
about any difference between classes. So the real question is *where* the
difference sits. Measured over the classifier window:

```
sensorimotor strip           mean |LEFT-RIGHT| =  7.2 points
posterior (visual/parietal)  mean |LEFT-RIGHT| = 30.2 points
strongest single electrode: O1  (+65.7 points)
```

The between-class difference is **4.2× larger at the back of the head than over
the motor cortex.** Something other than motor imagery separates these classes.
The script tests three explanations rather than picking a story:

1. **The visual cue?** This dataset cues trials with a target on the left or
   right of the screen, visible for the whole trial, and the visual pathway
   crosses over like the motor one — so a target on the right should activate
   the *left* visual cortex (O1). Prediction: right-hand trials drop more at O1,
   left-hand at O2. **Rejected.** The prediction fails, and the posterior effect
   is nearly identical on both sides (+67.9 % at O1, +60.7 % at O2 for
   left-hand). A response to a one-sided target would have to be lateralised.
2. **A few extreme trials?** Occipital alpha explodes when someone blinks a lot
   or briefly drifts off, and one trial can move a mean by tens of points. One
   trial does reach +341 %. But the between-class gap is +30.2 points on means
   and still +18.0 points on medians. **Partly, not mainly.**
3. **Trial order / fatigue?** Only possible if class order were blocked. The
   actual order is `RLLRRLRLRLLRLRLLRLRLRLRRLLRRLLRLRLLRRLLRRLRLR` —
   irregular, i.e. randomised — and the correlation between trial index and
   posterior power is −0.14. **Rejected.**

**What we are left with** is the dullest explanation: sampling noise. Occipital
alpha varies enormously trial to trial, and with ~22 trials per class a gap this
size can appear by chance. **We cannot settle it with 45 trials, and saying so
is the honest answer.**

### Two corrections made during this step
Both are recorded rather than quietly fixed, because the mistakes are
instructive:

- **The first version reported "45 epochs dropped".** False: MNE's `drop_log`
  has one entry per candidate event, including the 45 `T0` rest events we never
  requested, logged as `IGNORED`. Ignored is not dropped. Real drops: 0.
- **The first ERD figure had dips at both edges of every epoch.** Not
  physiology: `np.convolve(mode="same")` pads with zeros, which drags the moving
  average down at the ends. Fixed by padding with the edge value. A smoothing
  artifact that looked exactly like a real effect.

### What it implies for steps 5 and 6
This is the concrete, actionable output of step 4:

CSP maximises a variance ratio over all 64 electrodes. **It has no notion of
anatomy** and will happily build its filters out of high-variance posterior
channels if that separates the training data. With 45 trials and a 64×64
covariance matrix to estimate, we are squarely in the regime where that happens
— this is textbook overfitting.

**Planned action for step 6:** run the pipeline twice — all 64 electrodes, then
restricted to the sensorimotor strip. If the restricted version does as well or
better, the extra electrodes were feeding the classifier noise, and we will have
*measured* that instead of guessing. Step 5 will also let us look at the CSP
patterns directly: if they are centred over the motor cortex, that is
reassurance; if they are occipital, that is the red flag confirmed.

---

## Step 5 — CSP (Common Spatial Patterns)

Script: [`src/04_csp.py`](src/04_csp.py) · Report: `results/04_csp.txt`

### What we did
Fitted CSP with 4 components (2 per class) on the 0.5–2.5 s window, in three
variants, and measured cross-validated accuracy for each **with CSP inside every
fold**. Then plotted the spatial patterns to ask where the separating signal
comes from.

### Why CSP, in plain terms
A single electrode is a bad measurement: it records a weighted sum of everything
in the brain, blurred by the skull, dominated by whatever source is loudest.
Instead of choosing electrodes, CSP builds a *weighted combination* of all of
them — a virtual electrode — with weights chosen so the resulting signal is loud
for one class and quiet for the other. Formally it maximises the ratio of the two
classes' variances, which is a generalised eigenvalue problem with a closed-form
solution: no training loop, nothing to tune but the number of components.

This fits motor imagery exactly. ERD is a *power* change, and power is variance.
A filter that emphasises the right motor cortex produces a quiet signal when the
left hand is imagined. Filtering to 8–30 Hz in step 3 is what made "variance"
mean "band power" instead of "amount of drift" — steps 3 and 5 only work
together.

### Filters vs patterns — the subtlety that turns this step into a test
CSP produces two matrices, and plotting the wrong one leads to wrong conclusions.
**Filters** (`csp.filters_`) are how to weight electrodes to *extract* a source;
their weights can be large and oddly signed on channels whose job is to cancel
noise. **Patterns** (`csp.patterns_`) are how the extracted source *projects back*
onto the scalp — this is what answers "where is this coming from?". Interpretation
uses the patterns (Haufe et al., 2014, on forward vs backward models). That is
what makes step 5 a diagnosis of step 4's red flag rather than just a transform.

### How the three variants were chosen
Step 4 ended with a specific worry — the between-class difference was 4.2×
larger posteriorly than over the motor strip — and a specific action. So:

- **A. all 64 electrodes.** CSP chooses freely, including posterior channels.
- **B. sensorimotor strip only (21 electrodes).** CSP is forced to look where
  motor imagery must be, if it is there.
- **C. all 64 + covariance shrinkage (Ledoit-Wolf).** Added because A's problem
  might be purely statistical: a 64×64 covariance from 23 trials is a noisy
  object. This variant separates two explanations — does restricting help
  because of *anatomy*, or just because 21 < 64?

### A methodological point that changed the answer
The first evaluation used a single 5-fold split and gave **A = 77.8 %**. Its
per-fold scores were `0.44 0.89 0.89 1.00 0.67` — one fold below the majority
baseline, one perfect. With ~9 test trials per fold, one trial flipping moves a
fold by 11 points, so **a single split is a lottery draw and cannot compare two
variants.** Repeating the whole cross-validation 10× with different splits gives
A = 71.3 %: the single split was 6.5 points optimistic. All numbers below are the
repeated ones.

### Results

| Variant | Channels | Accuracy (10× 5-fold) | Range over repeats |
|---|---|---|---|
| A. all 64 electrodes | 64 | **71.3 % ± 5.7** | 64.4 – 82.2 |
| B. sensorimotor strip only | 21 | **62.4 % ± 4.7** | 57.8 – 71.1 |
| C. all 64 + shrinkage | 64 | 71.6 % ± 5.8 | 62.2 – 82.2 |

Majority-class baseline: **51.1 %**.

Where the patterns land (share of each pattern's absolute weight):

| Variant | Component | On motor strip | Posterior | Peak electrode |
|---|---|---|---|---|
| A | 1 | 12.3 % | **53.7 %** | **O1** |
| A | 2 | 19.9 % | **42.6 %** | **P3** |
| A | 3 | 42.8 % | 10.4 % | F8 |
| A | 4 | 38.8 % | 28.0 % | PO4 |
| B | 1 | 100 % | — | CP5 |
| B | 2 | 100 % | — | **C4** |
| B | 3 | 100 % | — | FC3 |
| B | 4 | 100 % | — | **C3** |

### What this establishes

**1. Motor imagery is genuinely decodable from the motor strip alone.** 62.4 %
against a 51.1 % baseline, and variant B's patterns peak at CP5, **C4**, FC3 and
**C3** — over the sensorimotor cortex, where physiology says they should be.
Component 3 is a clean left-vs-right lateralised map. This is the honest result.

**2. All 64 electrodes score higher, for the wrong reason.** 71.3 % — but
component 1 puts only 12 % of its weight on the motor strip and peaks at **O1**,
an occipital electrode; component 2 peaks at P3. The two strongest components are
parietal/occipital, not motor.

**3. Step 4's red flag is confirmed, in the most treacherous way possible.** The
non-motor signal does not *degrade* accuracy — it **improves it by 8.9 points**,
which is above the 7.4-point combined spread, so the gap is real and not a split
artefact. A project that only looked at the score would report 71.3 % and claim
to have decoded motor imagery. **Only the patterns expose it.** This is the
single most useful thing this project has produced so far.

**4. Shrinkage changes nothing** (+0.2 points, within noise). Worth understanding
why: `reg` visibly rescales the CSP *filters*, but the spatial *patterns* differ
by ~4e-05, so the same source dominates either way. The covariance estimate is
not the bottleneck — the data is. This also answers the anatomy-vs-channel-count
question: restricting to 21 channels changes the result while regularising 64
does not, so it is *which* electrodes are included that matters, not how many.

**5. What we still do not know:** what that posterior signal actually *is*.
Step 4 rejected the visual cue (the effect is bilateral, a one-sided target
would be lateralised), a single bad trial, and fatigue. It now also survives 50
train/test splits — and pure noise does not generalise. So "sampling noise", the
conclusion we reached at step 4, no longer fits either. **This is an open
question and is written up as open.**

### Why the higher score is not the headline result

It would be easy to read the previous section as a presentation choice — pick the
humbler number, look rigorous. It is not. **The reason for rejecting 71.3 % as the
project's main result is generalisation, not modesty.**

**The suspect signal is probably tied to this specific protocol, not to motor
intent.** In this experiment a target is displayed on a screen and stays visible
for the whole trial. That means every trial carries a bundle of things that
happen to co-occur with the imagined hand: where the eyes are looking, how much
visual attention is engaged, when the subject blinks, how alert they are at that
moment. Posterior electrodes measure exactly that bundle. None of it is motor
intent — it is the *experimental situation* in which the motor intent was
recorded.

So the two variants are not two estimates of the same quantity with different
precision. They measure different things:

- **Variant B (62.4 %)** measures something the brain does when it imagines a
  movement. Change the protocol — eyes closed, no visual target, an auditory
  cue, a different lab, a different dataset — and that signal is still there,
  because it is a property of the motor system.
- **Variant A (71.3 %)** measures that *plus* whatever posterior activity happens
  to correlate with the class label in these 45 trials. Remove the screen and
  that extra 8.9 points has nothing to attach to. Performance would not degrade
  gracefully — it could collapse, and collapse without warning, because nothing
  in the accuracy number tells you which part of it was protocol-dependent.

**The analogy.** Imagine building a classifier that tells left-handed from
right-handed people, and it reaches 95 % accuracy. Inspecting it, you find it is
looking at which wrist the watch is on — which in your particular sample happened
to correlate with handedness. The 95 % is real: the model genuinely achieves it,
on that data, reproducibly, across any number of train/test splits. And it is
worthless. Show it one person who wears their watch on the other wrist, or a
sample from a country with a different convention, and it fails. Crucially, the
accuracy number never warned you — only *looking at what the model looks at* did.

That is precisely our situation. Repeated cross-validation confirms variant A's
71.3 % is reproducible **within this dataset**; the CSP patterns show it is
partly built on the watch, not the hand. Cross-validation measures reproducibility
under resampling. It cannot measure reproducibility under a change of protocol —
and that second kind is what matters for a BCI that must eventually work on
someone else, somewhere else.

**The link to motor activity is not ruled out.** This matters and is stated
plainly: we have not shown the posterior signal is irrelevant. It survives 50
train/test splits, so it is not noise. It could be genuinely motor-related —
imagining a hand movement plausibly recruits spatial attention and body
representation, both of which involve parietal cortex, and parietal electrodes sit
partly inside our "posterior" region. That would make it real signal, not
contamination.

But **as long as that link is not understood and demonstrated, it cannot carry the
project's main result.** Not because it is probably wrong — because we do not know
whether it is right, and an unexplained 8.9-point gain is exactly the shape a
confound takes. The asymmetry of risk decides it: if we headline 62.4 % and the
posterior signal later turns out to be motor, we have been conservative and the
result only improves. If we headline 71.3 % and it turns out to be the screen, we
have published a claim about motor imagery that was never about motor imagery.

**So variant B is the headline because it is the result that should survive a
change of context**, not because it is the humbler number. Both are reported,
with the gap and this reasoning attached, so a reader can disagree with the choice
using the same evidence we used to make it.

### What it implies for steps 6 and 7
Report both variants side by side, and treat **variant B as the project's
headline motor-imagery result even though it is the lower number.** The accuracy
we publish should be the one we can defend, not the one that looks best. Step 6
adds the confusion matrix, quantifies what fitting CSP outside the CV loop would
have cost, and extends to all 10 subjects — the last of which is also the first
real test of whether variant B's signal is stable across people.

---

## Step 6 — LDA classification, honest CV, and the classic mistake

Script: [`src/05_classify.py`](src/05_classify.py) ·
Report: `results/05_classification.txt`

### What we did
Ran the CSP → LDA pipeline properly (CSP re-fitted inside every fold), on both
electrode variants, and added three checks that accuracy alone cannot provide:
a confusion matrix, a permutation test, and a deliberate reproduction of the
leakage error so its cost is a number in this repo rather than a warning.

### Why LDA
After CSP there are 4 features and 45 trials. That ratio dictates the classifier:
anything expressive would fit noise. LDA draws one straight boundary and has no
hyperparameters to tune. It also pairs with CSP by construction — it assumes
roughly Gaussian features, and `CSP(log=True)` returns log-variances precisely
because raw variances are skewed. CSP + log-variance + LDA is the field's standard
baseline because each piece feeds the next one's assumptions.

### Result 1 — the cost of the classic mistake

CSP is a **supervised** transform. The version that appears in many tutorials
fits it once on everything and then cross-validates only the classifier:

```python
csp = CSP().fit(X, y)          # sees the labels of ALL trials
features = csp.transform(X)
cross_val_score(LDA(), features, y)
```

This *looks* like proper cross-validation — the LDA never sees its test fold —
but the features were built using the test trials' labels. Same data, same folds,
same classifier; the only change is when CSP is fitted:

| Variant | Correct (CSP inside fold) | Leaked (CSP fitted once) | Free gain |
|---|---|---|---|
| all 64 electrodes | 71.3 % | **95.6 %** | **+24.2 pts** |
| sensorimotor strip | 62.4 % | **88.0 %** | **+25.6 pts** |

**A quarter of the accuracy scale, for free, from one line in the wrong order.**
95.6 % would look like an excellent result and there is nothing in the number to
suggest otherwise. This is the single most dangerous error in the CSP literature,
and it is why every pipeline in this repo is a scikit-learn `Pipeline` — not for
elegance, but because a `Pipeline` makes the correct order structurally
unavoidable.

### Result 2 — the confusion matrix

| Variant | recall, left hand | recall, right hand | gap |
|---|---|---|---|
| all 64 electrodes | 70 % | 64 % | 6 pts |
| sensorimotor strip | 63 % | 65 % | 2 pts |

Accuracy hides asymmetry: 62 % could be 62/62 or 90/33, and the second would be a
nearly useless decoder with an identical headline. Step 4 gave us a specific
reason to expect asymmetry — right-hand imagery was lateralised, left-hand was
not — so this was worth checking. **It is not there.** Both classes are decoded
about equally well, and the motor-strip variant is the more balanced of the two
(2 points apart). Whatever the weak ERD lateralisation of step 4 means, it does
not translate into a one-sided classifier.

### Result 3 — the permutation test

**The plain-language version first.** With only 45 trials, a small sample can be
misleading in a specific way: an algorithm that understands nothing and is
effectively guessing can still land on a high score by pure luck. It is the same
reason flipping a coin ten times and getting seven heads is unremarkable — a run
of luck in a short sequence is ordinary, not evidence of a biased coin. On our
data, that luck reaches up to about **58 %**.

So how do we know 62 % is not just a good coin run? We build the coin and watch
it. We take our real trials, **shuffle the left/right labels at random** — which
destroys any real link between brain signal and label, by construction — and run
the entire pipeline on the shuffled version. Then we do it 500 times. What comes
out is the distribution of scores that "learning from meaningless labels"
produces: what luck alone looks like on exactly this data, with exactly this
pipeline.

Then we compare. On the split used for the test, the real labels scored
**71.1 %**, while shuffled labels averaged **49.6 %**. Only **1.4 %** of the 500
shufflings did as well as the real labels. So the decoder learned something real
rather than getting lucky on a small sample.

Two precisions that keep this honest:

- **p = 0.014 does not mean "1.4 % chance that this is luck".** It means: *if
  there were genuinely nothing to learn, only 1.4 % of shufflings would score this
  high.* The distinction matters — the first phrasing is the most common
  misreading of a p-value, and a reviewer would catch it immediately.
- **71.1 % here, 62.4 % as the headline.** Not a contradiction: the permutation
  test runs on one fixed 5-fold split (the same one for real and shuffled labels,
  which is what makes it fair), and step 5 established that a single split can be
  several points off. The headline 62.4 % is the average over 10 different splits.
  The permutation test answers *is there signal at all*; the repeated CV answers
  *how accurate is it*. Two questions, two numbers.

**The technical version.** Shuffle the labels, re-run the whole cross-validation,
repeat 500 times, and see how often chance reaches our score.

| Variant | Real labels | Shuffled labels | p |
|---|---|---|---|
| all 64 electrodes | 77.8 % | 49.5 % ± 8.6 | **0.0020** |
| sensorimotor strip | 71.1 % | 49.6 % ± 9.6 | **0.0140** |

Both are above chance. Three details that matter:

- **The chance distribution is centred at ~49.5 %, with a standard deviation of
  8.6 points.** That spread is the real lesson: on 45 trials, a pipeline with
  *no information whatsoever* lands above 58 % one time in six. This is why "we
  got 62 %" means nothing without a chance model.
- **The p-value is computed as `(1 + #{null ≥ observed}) / (1 + n)`**, the
  standard correction. With a finite number of shufflings you can never observe
  p = 0, and reporting p = 0 would claim more certainty than the procedure can
  deliver.
- **The permutation test uses one fixed 5-fold split** — the same split for the
  real labels and every shuffling, which is what makes the comparison fair. Its
  observed score (71.1 %) is therefore the single-split score, **not** the
  repeated-CV headline (62.4 %). Those are two different questions: *is there
  signal here at all* and *how accurate is the decoder*. Attaching the p-value to
  the accuracy figure would conflate them, which is how small-sample results get
  oversold.

### How we got to these choices
Every one of the three checks exists because a previous step raised a specific
question. The confusion matrix because step 4 found asymmetric ERD. The
permutation test because step 2 established there are only 45 trials, making
"above baseline" and "above chance" different claims. The leakage measurement
because step 5's docstring asserted the error matters — and an assertion in a
comment is worth less than a measured number.

### Three engineering notes, recorded because they cost real time
- **`model.py` appeared here**, the same way `preprocessing.py` appeared at step 3
  and for the same reason: step 5 built a CSP → LDA pipeline, step 6 needed the
  same one. Factor out on the second occurrence. Verified that step 5's numbers
  are unchanged after the move (71.3 / 62.4 / 71.6 identical).
- **`n_jobs=-1` on the permutation test does not work with MNE 1.8.** joblib hands
  workers a shared-memory copy of the data and MNE's internal rank estimator
  rejects it (`data copying was not requested by copy=None`). The estimator is
  reached from two different places inside `CSP.fit`, so no `rank=` argument
  avoids it. Replaced with a plain serial loop we control.
- **Threads were slower than serial anyway** — 53 s vs 27 s for 200 permutations,
  BLAS contention — so the parallelism was buying nothing. Worth measuring before
  optimising.

### What it implies for step 7
The headline result stands: **62.4 % ± 4.7 on the sensorimotor strip, p = 0.014**,
against a 51.1 % baseline, with motor-centred CSP patterns. Modest, real,
defensible. But every number on this page describes **one person**, and
between-subject variability in motor imagery is larger than any difference
measured here. Step 7 runs all 10 subjects, which is also the first genuine test
of whether the posterior signal from steps 4–5 is a property of this dataset or
of subject 1.

---

## Calibration — why the model is trained per person

This is not one of the eight steps, but it explains a choice that runs through all
of them and it is central to how real BCIs work.

### Why one model per person, not one model for everyone

Motor imagery signals vary enormously between people. Skull thickness, cortical
folding, where exactly the hand area sits, how vividly someone imagines movement,
how much attention they bring — all of it changes both the *shape* of the scalp
pattern and its *strength*. Two people imagining the same movement can produce
spatial patterns that barely resemble each other. "BCI illiteracy" is the extreme
end of this: 15–30 % of healthy subjects produce no decodable motor imagery at
all.

That is why the model in this project is fitted **individually, on each person's
own examples**, rather than one shared model applied to everybody. A single global
decoder would have to average over patterns that genuinely differ, and would end
up describing nobody well. Step 7 measures how large that between-person spread
actually is on this dataset.

### Why a statistical model, not a fixed threshold

The signal also varies from trial to trial *within* the same person. Nobody
produces exactly the same brain signal twice: attention drifts, alertness changes,
the mu rhythm's baseline amplitude fluctuates over minutes. Step 4 showed this
directly — ERD in individual trials ranged from strongly negative to clearly
positive around the same mean.

This rules out a fixed rule of the form "if power at C3 drops below X microvolts,
call it right hand". Any such threshold would be right for a few trials and wrong
for the rest. What CSP + LDA do instead is learn an **average statistical pattern**
across many trials — CSP finds the electrode weighting that best separates the two
distributions, LDA finds the boundary between them — and then classify each new
trial by which distribution it more likely came from. The model is explicitly
probabilistic about a signal that is explicitly variable. That match is why this
combination survived thirty years, not inertia.

### How real BCIs handle this: the calibration session

In a usable system, between-person and between-session variability is handled by a
**calibration phase**. Before real use, the person performs a series of trials
where the intended action is already known — "imagine your left hand… now your
right hand…" — repeated a few dozen times. The system fits its spatial filters and
its decision boundary on those known examples, and only then starts interpreting
unknown intentions. On many systems this is repeated at the start of *every*
session, because electrode positions, gel impedance and the person's state all
change from one day to the next.

Concretely, the 45 trials in this project play the role of a calibration set. The
pipeline is exactly what a calibration phase runs.

### The limitation this project carries

**We train and test on the same subject and the same recording session.**
Cross-validation splits the 45 trials into train and test parts, so no trial is
ever tested on a model that saw it — that part is correct, and step 6 measured what
happens when it is not. But there is no genuine session-to-session test: no "fit
on day 1, use on day 2", and no "fit on person A, use on person B".

That matters, because session transfer is where BCIs are known to lose the most
performance. A decoder that reaches 62 % within a session can fall towards chance
on a recording made a week later with the cap repositioned. Our numbers therefore
describe **within-session** decoding, which is the easiest version of the problem.

This limitation is the same shape as the one in step 5 about the posterior signal:
cross-validation measures reproducibility under resampling, not reproducibility
under a change of conditions. Step 7 takes the first step out of that box by
testing across **people**, which is the between-subject half of the question. The
between-session half would need a dataset with repeated sessions per subject — a
concrete next item alongside the two already listed in the README's long-term
vision.

---

## Step 7 — All 10 subjects, and what survives

Script: [`src/06_all_subjects.py`](src/06_all_subjects.py) ·
Reports: `results/06_all_subjects.txt`, `results/06_all_subjects.csv`

### What we did
Ran the entire pipeline unchanged on subjects 1–10 — same band, same window, same
components, same folds, **nothing re-tuned per subject** — to find out which of
the three subject-1 findings were results and which were accidents. Each subject
gets its own model fitted on its own trials, which is the calibration logic
described above, not a convenience.

Re-tuning per subject would have produced better numbers and a meaningless
comparison: we would be measuring how much tuning helps, not how much people
differ.

### The results

| Subject | Trials | Baseline | All 64 | Motor strip | p (motor) | LI left | LI right | post/motor | Peak electrode |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 23/22 | 51.1 % | 71.3 ± 6 | 62.4 ± 5 | 0.020 | −0.8 | −9.2 | 4.2 | O1 |
| S2 | 23/22 | 51.1 % | 88.9 ± 4 | **90.9 ± 2** | 0.005 | −5.5 | −21.2 | 1.0 | FT8 |
| S3 | 23/22 | 51.1 % | 57.3 ± 6 | 43.6 ± 7 | 0.960 | −5.5 | +8.5 | 5.3 | PO7 |
| S4 | 23/22 | 51.1 % | 51.3 ± 4 | 47.1 ± 5 | 0.537 | −4.0 | −8.0 | 0.7 | F5 |
| S5 | 21/24 | 53.3 % | 56.2 ± 7 | 57.3 ± 6 | 0.179 | +6.0 | −5.4 | 2.0 | POz |
| S6 | 24/21 | 53.3 % | 43.3 ± 9 | 46.9 ± 6 | 0.587 | −5.0 | −9.1 | 1.3 | POz |
| S7 | 23/22 | 51.1 % | 98.4 ± 1 | **96.9 ± 2** | 0.005 | −5.4 | −16.6 | 1.8 | T9 |
| S8 | 22/23 | 51.1 % | 58.9 ± 5 | 49.6 ± 6 | 0.438 | −1.5 | −2.0 | 0.9 | Fpz |
| S9 | 24/21 | 53.3 % | 50.0 ± 6 | 42.0 ± 5 | 0.851 | −7.4 | +2.7 | 2.5 | PO8 |
| S10 | 24/21 | 53.3 % | 52.2 ± 5 | 62.0 ± 4 | 0.184 | +4.2 | −1.9 | 0.5 | Fz |

### Finding 3 — how well does the method actually work?

**Read the median, not the mean.**

| | Motor strip |
|---|---|
| mean | 59.9 % |
| **median** | **53.4 %** |
| mean without the top 2 subjects | **51.4 %** |
| mean majority-class baseline | 52.0 % |

**Subjects individually above chance: 3 out of 10.** Spread between best and
worst: **54.9 points**.

The median sits 1.4 points above the baseline, and removing the two best subjects
puts the pipeline *at* chance. **Two people out of ten carry the entire average.**
Quoting 59.9 % as "the accuracy of this method" would describe a decoder that does
not exist for eight of these ten subjects — and this is the single most important
thing step 7 established.

This is not a failure of the pipeline. It is the textbook picture of motor
imagery: "BCI illiteracy" — 15–30 % of healthy subjects producing no decodable
imagery — is a documented property of the paradigm, and the numbers here are worse
than that figure but in the same direction. It is also the calibration argument
made concrete: a 55-point spread is exactly why a model is fitted per person
rather than shared.

Subject 1, at 62.4 %, turns out to be **above the median** — a reminder that the
subject we spent four steps analysing was not typical.

### Finding 1 — was subject 1's missing left-hand lateralisation real?

| | Mean across subjects | With the predicted sign |
|---|---|---|
| LI(left), should be positive | −2.5 pts | **2 / 10** |
| LI(right), should be negative | −6.2 pts | **8 / 10** |
| contrast LI(left) − LI(right), should be positive | +3.7 pts | 7 / 10 above the noise floor |

At first reading this confirms subject 1 dramatically: the right hand lateralises
in 8 subjects, the left in 2. **That reading is wrong, and our own metric is what
misled us.**

Both mean LIs are negative. Pooled across hands the mean LI is **−4.4 points**, and
**6 of 10 subjects have C3 dropping more than C4 for *both* hands**. That is a
global hemispheric offset, not a hand-specific effect — and an offset alone
reproduces the "8/10 vs 2/10" counts, because subtracting a constant from both
rows makes the negative-expected one look right and the positive-expected one look
wrong.

The offset-free quantity is the **contrast** between hands, because a constant
added to both cancels in the subtraction. It is +3.7 points on average and clears
the noise floor in 7 of 10 subjects. So:

- **There is a real lateralised difference between imagining one hand and the
  other.** That survives.
- **Whether the left hand is specifically weaker is unresolved.** Settling it would
  need handedness information the dataset does not provide, and a metric that does
  not confound a global offset with a hand effect.

Step 4 offered two explanations and could not choose. Step 7 shows the question
was partly malformed.

### Finding 2 — is the suspect signal the protocol or subject 1?

Neither, and the answer is worse than both. **6 of 10 subjects** show a larger
between-class difference posteriorly than over the motor strip (median ratio 1.5,
range 0.5–5.3), so it is not universal — but the per-subject peak electrodes are:

| Family | Count | Subjects |
|---|---|---|
| posterior (visual / attention) | 5 / 10 | S1 S3 S5 S6 S9 |
| **sensorimotor (what we want)** | **0 / 10** | — |
| temporal (jaw / neck muscle, EMG) | 2 / 10 | S2 S7 |
| frontal / frontopolar (eye movement, EOG) | 3 / 10 | S4 S8 S10 |

**Not one subject has its strongest between-class electrode over the sensorimotor
strip.** The peaks are occipital, temporal or frontal — three families with
well-known non-brain sources. The confound is therefore broader than the
"posterior" one step 4 chased, and it is per-subject rather than dataset-wide,
which is practically the worst case: it cannot be corrected once for everyone, and
per-subject pattern inspection is not optional.

### The cross-check that matters most

Reading the accuracy table and the peak-electrode table *together* gives the
observation neither gives alone:

> **The two subjects carrying the entire group average — S7 at 96.9 % and S2 at
> 90.9 % — are also the only two whose strongest between-class electrode is a
> temporal site (T9, FT8).** Temporal electrodes pick up jaw and neck muscle
> activity, and EMG extends into the upper part of our 8–30 Hz band.

This does **not** prove those two results are muscle. Their motor-strip accuracy is
high on its own, and the lateral strip channels (C5, C6, FC5, FC6) sit close enough
to temporal sites that electrode position cannot settle it. But the two strongest
results in the study are the two where a non-brain explanation is most available,
and reporting a group mean without saying so would be misleading.

**What would settle it:** EMG-specific diagnostics — high-frequency power ratios,
ICA artifact components. Outside this project's scope, and recorded as the single
most valuable next check.

### What it implies for step 8
The honest headline is not one number. It is:

- **3 of 10 subjects decodable above chance**, median 53.4 % against a 52.0 %
  baseline;
- **best subject 96.9 %, worst 42.0 %** — the spread is the result, not noise
  around a result;
- **CSP patterns never peak over the motor cortex** in any of the ten subjects;
- and the two best results carry an unresolved EMG question.

A README that led with "62 % accuracy" would be defensible for subject 1 and
misleading about the method. Step 8 has to lead with the distribution.

---

## Step 8 — The final README

### What we did
Rewrote the README as the finished front page: results first, method second,
limitations stated rather than buried. This document stays as the reasoning trail
behind it.

### The one real decision: what to lead with

The obvious opening was subject 1's number — "62.4 % on motor-imagery
classification, p = 0.014". Defensible, since it is true of subject 1. Also
misleading about the method, because step 7 showed subject 1 is above the median
and that seven of ten subjects are at or below baseline.

So the README leads with the **distribution**: 3 of 10 above chance, median 53.4 %
against a 52.0 % baseline, and the fact that removing the two best subjects puts
the pipeline at chance. The single most quotable line is not an accuracy at all —
it is that **no subject's strongest discriminative electrode lies over the motor
cortex.**

This costs the project its most impressive-looking number. It is worth it for a
reason that has nothing to do with modesty: a reader who checks the per-subject
table after being told "62 %" concludes the author did not look. A reader told the
median up front concludes the author did. The second reader is the one worth
writing for.

### How the README is organised, and why
1. **What it is**, in one paragraph, then the honest one-line summary.
2. **Results** — the distribution, before any method.
3. **Three things the project actually found** — the confound that *improves*
   accuracy, the zero-out-of-ten peak-electrode result, and the 25-point cost of
   the leakage error. These are the differentiators: they are what a tutorial
   reproduction would not contain.
4. **Method**, kept short, with the measurements that justify each choice (the
   78 %-below-8 Hz figure rather than "8–30 Hz is standard").
5. **Context and limitations** — the three levels of motor intent, the dataset
   trap, calibration, and five things the project explicitly cannot claim.
6. **Long-term vision**, then reproduction instructions.

Every figure in the README is regenerated by a script in `src/`, and every number
is traceable to a report in `results/`. Nothing in it was typed by hand from
memory.

### What a reader should take away
That the pipeline is standard and correctly implemented; that the results are
weak and honestly characterised; and that the interesting work was the diagnosis,
not the score.

---

## Running log of things that changed our mind

| Step | We believed | The data said | Consequence |
|---|---|---|---|
| 2 | Data was ~50 MB per subject | 7.8 MB for 3 runs | README corrected |
| 2 | `T0` rest markers probably exist | 45 of them, 4.20 s, balanced | The idle-class plan is now grounded |
| 4 | 45 epochs were being dropped | 0 dropped; 45 `T0` events *ignored* | Reporting bug fixed |
| 4 | ERD dips at epoch edges | Zero-padding artifact in our smoothing | Padding fixed |
| 4 | Contralateral ERD would be clear for both hands | **Right hand: clean, C3 drops 9.2 pts more than C4, as theory predicts. Left hand: nothing, 0.8 pts — no lateralisation at all.** Most likely too few trials (23), possibly a real dominant/non-dominant asymmetry | Reported as a **mixed** result — one class out of two — not as "confirmed". Settling it needs all 10 subjects (step 7) |
| 4 | Posterior difference = visual cue | Bilateral, so not the cue | Three hypotheses tested, all rejected; noise remains |
| 5 | `csp.patterns_` has one row per kept component | One row per **channel** (64) — `n_components` only affects `transform()` | Pattern figure was silently 64 panels wide; fixed to the first `n_components` rows |
| 5 | Shrinkage would fix a 64×64 covariance from 23 trials | +0.2 pts, nothing. It rescales filters but leaves patterns ~4e-05 apart | The bottleneck is the data, not the covariance estimate |
| 5 | A single 5-fold accuracy was reportable | 77.8 % on one split vs **71.3 %** over 10 splits — 6.5 pts optimistic | All accuracies now come from repeated CV |
| 5 | If posterior channels were noise, dropping them would help | Dropping them **costs 8.9 pts**. The confound *helps* the score | The headline result becomes the lower, defensible one (variant B) |
| 6 | Fitting CSP outside the CV loop inflates accuracy "somewhat" | **+24 to +26 points.** 71.3 % becomes 95.6 % | Every pipeline stays a `Pipeline`, so the correct order is structurally unavoidable |
| 6 | Step 4's asymmetric ERD would make a one-sided classifier | Recall is 63 % / 65 % — the most *balanced* variant is the motor one | Weak ERD lateralisation does not translate into class bias |
| 6 | Chance is ~51 % (the majority baseline) | Chance *distribution* is 49.5 % **± 8.6 pts**: a no-information pipeline exceeds 58 % one time in six | An accuracy without a chance model is not a result |
| 6 | `n_jobs=-1` would speed up the permutation test | MNE 1.8's rank estimator rejects joblib's shared-memory arrays; and threads were *slower* (53 s vs 27 s) | Hand-written serial loop; measure before optimising |
| 7 | 62.4 % was roughly what the method delivers | **Median across 10 subjects is 53.4 % vs a 52.0 % baseline; only 3/10 are above chance.** Subject 1 was above the median | The headline becomes a distribution, not a number |
| 7 | Removing the extremes would barely move the average | Without the top 2 subjects the pipeline is **at chance** (51.4 %) | Two people out of ten carry the entire mean |
| 7 | "Right hand lateralises, left does not" (8/10 vs 2/10) | 6/10 subjects are negative for **both** hands — a global offset that manufactures those counts on its own | Only the hand-to-hand *contrast* is offset-free; the per-hand claim is retracted |
| 7 | The posterior confound was subject 1's, or the protocol's | Per-subject, and broader: peaks are occipital (5), frontal (3), temporal (2), **sensorimotor (0)** | Per-subject pattern inspection is not optional |
| 7 | The two best subjects are the method working | S7 (96.9 %) and S2 (90.9 %) are also the only two peaking at a temporal, EMG-prone electrode | Unresolved; EMG diagnostics logged as the top next check |

---

## Open questions

Written down because a finished project with no open questions has usually
stopped looking.

**1. What is the posterior signal?** It separates the classes in 6 of 10 subjects,
survives 50 train/test splits (so it is not noise), and is not the visual cue (the
effect is bilateral, a one-sided target would be lateralised), not one bad trial,
and not fatigue. It could still be genuinely motor-related — imagining a hand
movement plausibly recruits spatial attention and body representation, both
parietal. Unresolved.

**2. Are S2 and S7 muscle?** The two subjects carrying the group average are the
only two peaking at a temporal electrode. The next diagnostic is specific and
cheap: high-frequency power ratios and ICA artifact components. This is the single
most valuable thing to do next, because it decides whether the project's best
results are real.

**3. Why does no subject peak over the motor cortex?** Zero out of ten is a strong
result and we do not have an explanation. It may be that the largest *absolute*
between-class difference is simply not where the most *reliable* one is — CSP after
all finds usable motor patterns in the restricted variant. A per-subject
signal-to-noise analysis, rather than raw difference magnitude, would test that.

**4. Is the left hand really weaker, or is our metric offset?** Step 7 showed the
per-hand reading was confounded by a global C3-vs-C4 offset. Settling it needs
handedness information the dataset does not provide, and a lateralisation metric
that is offset-free by construction.

**5. Does any of this survive a session change?** Everything here is
within-session. The honest answer is that we do not know, and the dataset cannot
tell us — it has one session per subject.

### If this project continued, in priority order
1. EMG diagnostics on S2 and S7 (decides whether the best results are real).
2. Add the `T0` rest class and a rejection threshold — the first step towards a
   decoder that can stay silent.
3. A dataset with repeated sessions, to measure the session-transfer loss.
4. Attempted-movement data at 0.3–3 Hz, which is the actual long-term target.
