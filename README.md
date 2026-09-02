# PAMAP

Wearable activity states and cardiovascular strain, built on the
[PAMAP2 Physical Activity Monitoring dataset][pamap2].

Two things live here:

1. **An unsupervised split of wearable data into `active`, `static` and
   `recovery`** — where the third state is the one no label in the dataset
   records.
2. **A model that predicts a person's cardiovascular strain from movement
   alone**, with no heart-rate channel, validated leave-one-subject-out.

The published dashboard is at
[treychase.github.io/projects/wearable-strain.html](https://treychase.github.io/projects/wearable-strain.html).

[pamap2]: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

## The dataset

PAMAP2 is nine subjects wearing three Colibri inertial units — dominant wrist,
chest, dominant ankle — plus a chest heart-rate strap, through a twelve-activity
protocol. It is the right dataset for this because it carries all three things a
strain model needs at once: a ground-truth activity label, triaxial
acceleration at 100 Hz, and a biometric channel.

Files are whitespace-separated, 54 columns, no header. `pamap/schema.py` is the
column layout, the activity ids and the subject characteristics transcribed from
the dataset's own documentation, including each subject's resting and maximum
heart rate — which is what turns a raw bpm trace into heart-rate reserve, the
only form of the signal comparable between a 23-year-old and a 32-year-old.

Two properties of the recording that the loader has to handle:

- **The heart rate is missing on most rows.** The strap samples at ~9 Hz against
  the units' 100 Hz, so the column is present on about one row in eleven. Those
  gaps are the sampling grid, not dropout, and interpolating across them is the
  correct reconstruction. `ingest.interpolate_heart_rate` bridges them and
  leaves anything longer than five seconds — a strap losing contact — missing.
- **Use the ±16g accelerometer, not the ±6g one.** The dataset's readme says the
  6g sensor saturates during running and rope jumping, which are exactly the
  windows a strain model most needs to get right.

## Running it

```bash
pip install -r requirements.txt

# against the real dataset
python dashboard/build.py --data ~/PAMAP2_Dataset --out wearable-strain.html

# against the stand-in cohort, when the real files are not reachable
python dashboard/build.py --synthetic --out wearable-strain.html
```

`--data` takes an unpacked `PAMAP2_Dataset` directory. With no `--data` and no
`--synthetic`, the pipeline downloads the archive from UCI and caches it under
`~/.cache/pamap2`.

### The stand-in cohort

`pamap/synth.py` writes `subjectNNN.dat` files in the dataset's exact schema —
the same 54 columns, sampling rates, activity ids and nine subject profiles.
It exists because the dataset is published at `archive.ics.uci.edu`, which is
not reachable from every environment; a locked-down runner or an egress policy
will refuse it, and the pipeline should still be runnable and testable there.

**It is a stand-in, not the dataset.** Numbers produced from it describe the
generator, and anything published off it has to say so — the dashboard does, in
its method note. What it is *for* is keeping the pipeline honest: the loader,
the windowing, the clustering and the strain fit all see input they cannot
distinguish by shape from the real thing.

Two of its design choices carry the analysis:

- **Acceleration is synthesised in the sensor frame.** Each activity fixes where
  gravity points for each unit — that is what posture *is* to an accelerometer —
  and adds a periodic component at the activity's cadence with harmonics, plus
  broadband noise. Nothing is drawn from a per-activity summary statistic, so the
  feature extractor has to recover cadence and intensity from the waveform.
- **Heart rate lags.** It chases each activity's demand as a first-order system
  with a fast onset constant (26 s) and a slow recovery one (78 s). That is what
  makes the recovery state exist at all: after a hard bout, a subject sitting
  still has a body at rest and a heart that is not.

## How it works

### Windows

Ten seconds at half overlap. The window has to hold several gait cycles for a
cadence estimate to mean anything, and be short enough not to straddle a change
of activity; windows that do straddle one are dropped rather than given a
majority label. Features come in three blocks — movement, cardiac, posture.

Two things in `features.py` are worth knowing about because both are easy to get
wrong:

- **The spectrum is computed per axis and summed, never from the vector
  magnitude.** Taking `|a|` first rectifies each oscillation, and `|sin|` has
  twice the period of `sin`, so a magnitude spectrum reports every cadence at
  double its true value. There is a test that walks a known cadence through the
  extractor and fails if it comes back at a harmonic.
- **Posture is measured against each subject's own upright reference**, not
  against a hardcoded sensor axis. What an axis means anatomically depends on
  how the unit was strapped on, which is arbitrary between people. Windows with
  strong periodic movement are locomotion, and nobody walks lying down, so the
  median gravity direction over a subject's most periodic windows defines their
  upright; every window is then scored as the angle from it. Near 0° standing,
  sitting or walking; near 90° lying. No activity label is consulted.

### The two-stage clustering

The obvious approach — one k-means at k=3 over movement and cardiac features
together — does not work, and how it fails is the reason for the design.

Movement intensity spans three orders of magnitude from lying still to rope
jumping. Heart-rate reserve among *still* windows spans a few tenths. k-means
minimises total within-cluster distance, so given three clusters it buys the
reduction on offer: it splits the active spread into moderate and vigorous and
leaves every still window, baseline and recovering alike, in one lump. The
static/recovery distinction gets no cluster at all. Block-weighting the features
does not fix it — the problem is the shape of the distributions, not the column
counts. That result is reproduced by `cluster.flat_comparison` rather than
asserted, and reported on the dashboard.

What works is to match the structure of the question, which is hierarchical:

| Stage | Features | k | Split |
| --- | --- | --- | --- |
| 1 | movement | 2 | active vs still |
| 2 | cardiac | 2 | static vs recovery, among the still windows |

Both stages are unsupervised. The clusters are named afterwards from where their
centres sit in original units — most movement is `active`; of the two still
clusters, the one whose heart is further above its own resting baseline is
`recovery` — never from the activity labels.

**Posture is computed and reported but never fitted on.** Handed to a clustering
it is a clean bimodal split, ~0° upright against ~90° lying, and any model spends
a whole cluster separating lying from sitting. That distinction is orthogonal to
this taxonomy: a man on a mat and a man in a chair are both, for these purposes,
still. It is read back afterwards as a description, where it doubles as a check —
lying and sitting landing in the same state is the intended behaviour.

### Strain

**Observed strain** is Banister's TRIMP, which weights every minute by
heart-rate reserve under an exponential, so a minute near maximum is worth far
more than three easy ones. That exponent is what makes it a model of
physiological cost rather than a stopwatch.

**Predicted strain** is fitted from accelerometry, the discovered state mix and
the subject's body — **never from heart rate**. There is a test that asserts no
heart-rate-derived column reaches the model. That direction is the commercially
useful one: accelerometers are cheap, always on and never lose contact, while
heart rate is the channel that goes missing when the strap is in a drawer. The
gap between the two numbers is itself a signal — a session that cost more than it
looked like it should is the shape of fatigue, heat, or a body that has not
recovered.

Scoring is leave-one-subject-out. Windows from one person are serially
correlated and share a body, so a random split would put neighbouring windows
from the same session on both sides of it and report a number that says nothing
about the next person to put the watch on.

The 0–21 display scale is a fixed log map of TRIMP (300 TRIMP anchors 21). It is
a presentation convention borrowed from consumer wearables, not a validated
clinical instrument; TRIMP underneath it is the quantity with literature behind
it.

## Layout

```
pamap/schema.py     the 54-column layout, activity ids, subject characteristics
pamap/ingest.py     loading .dat files, reconstructing the heart-rate channel
pamap/synth.py      the stand-in cohort
pamap/features.py   windowing and feature extraction
pamap/cluster.py    the two-stage state model, and the flat fit it rejects
pamap/strain.py     TRIMP, the display scale, and the movement-only predictor
pamap/pipeline.py   raw files to window table in one call
dashboard/          the template and the build that fills it
tests/              pytest
```

## Tests

```bash
pytest tests/ -q
```

They check properties rather than fixed numbers: that the loader bridges a
sampling grid without inventing data across a dropout, that a known cadence
comes back at its fundamental, that windows never straddle an activity change,
that strain is monotone in effort, that no heart-rate column reaches the strain
model, and that the clusters are named by physiology rather than by the order
k-means returned them in.
