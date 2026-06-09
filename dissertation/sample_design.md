# Human corpus: sampling design

Status: agreed 2026-06-08. This note records the human side of the detection
corpus so it can go into the methodology chapter. The matched AI side is built
later (it needs HPC for generation at scale).

## Source corpus

The human essays come from the British Academic Written English corpus (BAWE),
downloaded from the Oxford Text Archive (resource 2539, CC BY-NC-SA 3.0). BAWE
holds 2,761 proficient student assignments from a UK university, collected 2000
to 2007, across four broad disciplinary groups and 13 genre families. The
holdings metadata ships as `BAWE.xls`, and the corpus also includes plain-text
versions of every assignment.

I checked the metadata word count against a direct word count of the plain-text
files on a 200-essay sample. They agree closely (Pearson r = 0.986, median ratio
0.995), so the `words` column is reliable as the length figure used for matching.

## Cleaning

Before sampling I cleaned three known issues in the metadata:

- decoded HTML entities in discipline labels (for example `&amp;` became `&`),
- merged the case-split catch-all label `OTHER` into `Other`,
- dropped one row that was missing both its disciplinary group and its word
  count.

This leaves 2,760 usable essays. Nine rows have a missing academic level. Level
is not a sampling variable, so those rows are kept and the gap is reported.

## Sampling strategy

The detection corpus is stratified on **disciplinary group crossed with
first-language status**, giving eight cells. I sampled an equal 80 essays per
cell, for 640 human essays in total.

| Group | native | non-native | group total |
|-------|-------:|-----------:|------------:|
| Arts and Humanities | 80 | 80 | 160 |
| Life Sciences | 80 | 80 | 160 |
| Physical Sciences | 80 | 80 | 160 |
| Social Sciences | 80 | 80 | 160 |
| **Total** | **320** | **320** | **640** |

Design choices and why:

- **Group, not named discipline.** In BAWE the four disciplinary groups are
  fairly balanced, while individual disciplines are very uneven (Engineering has
  238 essays, Architecture only 9). Stratifying by group gives a balanced sample
  without letting a few large disciplines dominate, and it spreads the sample
  across the four broad areas of study.
- **Oversampling non-native writers.** In the full corpus 70.7% of essays are by
  native English speakers and 29.3% by non-native speakers. A 50/50 split
  oversamples non-native writers so there are enough of them to measure whether
  the detector treats non-native writing unfairly. Detector bias against
  non-native speakers is a known problem in this field, so the design measures it
  directly rather than ignoring it.
- **Per-student cap of 4.** BAWE is heavily clustered by author (one student
  contributes up to 20 assignments, and 86% of students contribute more than
  one). Without a cap, a single writer's style could teach the detector what
  "human" looks like. Within each cell I keep at most four essays per student.
- **Splitting by student, not by essay.** Train, validation and test are split
  70/15/15 at the student level so that no writer appears in two splits. About 5%
  of students have essays in more than one disciplinary group, so the assignment
  is made globally across cells, not cell by cell. The realised split is
  68.4 / 15.9 / 15.6, which rounds away from the target slightly because whole
  students are assigned together. A leakage check confirms zero students span two
  splits.
- **Reproducibility.** The draw uses a fixed seed (42). The exact selection and
  split are saved as a versioned manifest so the sample can be recovered even if
  a future library version changes the random draw.

## Realised composition

The sample draws from a healthy range of disciplines within each group (Arts and
Humanities 8 disciplines, Life Sciences 6, Physical Sciences and Social Sciences
9 each). Engineering is the heaviest single discipline (62 of the 160 Physical
Sciences essays), which reflects its size in BAWE. Academic level falls out
roughly even without being a sampling variable: taught masters 28.9%, year 1
27.2%, year 2 22.5%, year 3 21.1%.

## Length

Length is the main confound to control, because a detector that learns length
instead of style would be worthless. Two defences are in place. First, balancing
the groups keeps the group-level length differences from lining up with any
label. Second, and more important, each AI essay is generated to match the length
of its specific human source, so the human-versus-AI label is never separable by
length. In the human sample, native and non-native essays are close on length
(mean 2,368 versus 2,471 words). Physical Sciences and Social Sciences non-native
essays run a little longer than their native counterparts. That gap is a property
of the human writing and is reported, not removed.

## Limitations

- Non-native essays in Arts and Humanities are scarce (114 in the corpus, from
  only 29 students), so per-group bias estimates there will be the noisiest. The
  bias analysis pools across groups, and uses cross-validation by student, to get
  a more stable overall estimate.
- Stratifying at group level means discipline mix inside a group is not
  controlled, so some disciplines are heavier than others within a group.
- One student id carries two different first-language values in the metadata,
  which is a minor inconsistency in the source. With 627 students it has no
  material effect.

## Supervisor feedback (Meeting 2): avoid over-structuring

Dr. Vijayan advised not to over-invest in manual balancing. An AI system should be
able to learn from less curated data, and heavy hand-structuring risks a result that
does not hold on realistic, messy input. The balanced 640 sample stands as the
phase-one supervised training set, and the next step is training rather than further
balancing.

To test whether the balancing helps at all, a later experiment will draw a second,
natural (unbalanced) sample from the same cleaned corpus and train a separate detector
on it, then compare the two. Per the supervisor, this must be two models trained in
parallel, not one reused, because a model already trained on the balanced set cannot be
fairly evaluated on the natural one. This comparison is recorded in the evaluation plan
in `CLAUDE.md`.

## Reproducibility

- `src/data/explore_bawe.py` loads `BAWE.xls`, summarises the corpus and saves
  the metadata table.
- `src/data/clean_bawe.py` cleans the metadata into the sampling frame.
- `src/data/build_sample.py` draws the stratified sample and assigns the splits.
- Outputs: `data/processed/bawe_human_sample.csv` (full) and
  `data/processed/bawe_human_sample_manifest.csv` (versioned id-and-split
  manifest).
