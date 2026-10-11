# Application 1: speech polarization in the US Congress, 1873–2024

Replication code for the topics-versus-slant decomposition of congressional speech with the GTM.

## Where things live

- **Code** (this folder, tracked in git): `code/` (numbered steps), `slurm/` (cluster jobs),
  `inputs/vocab_us.txt` (the curated vocabulary), `inputs/vocab_us_supplemented.txt` (the same plus
  `inputs/vocab_us_supplement.txt`, see below), `sync_to_hpc.sh`.
- **Vocabulary supplement** (2026-10-06): the curated list was built on 1994–2024 speeches, so it lacks
  terms of the earlier congresses. Candidates were every word and two-word phrase of the cleaned
  speeches not in the list and used by at least 0.02% of the speeches of some era (sample of 3,000
  speeches per congress); 274 era-specific policy terms were chosen by hand (war and navy departments,
  pensions, tariffs, coinage, railroads, post office, public works, agriculture, agencies, ...;
  `inputs/vocab_us_supplement.csv` gives their use by era). General words the curators excluded in
  every era (war, money, party, election, government, ...) stay excluded in the main fit; the
  comparison vocabulary `inputs/vocab_us_supplemented_general.txt` adds 38 of them back
  (`inputs/vocab_us_general.txt`).
- **Everything else** lives on the Bocconi cluster only, in `/home/Gauthier/app1_us_congress/`:
  `data/` (raw inputs and intermediate files), `results/`, `logs/`, and `pkg/deeplatent`
  (the copy of the package the jobs import). Datasets are never copied to the laptop.

Sync after any code change (checks every file with md5): `bash replication/app1_us_congress/sync_to_hpc.sh`.

## Data sources

| Source | Period | What it is |
|---|---|---|
| `bound` | 1873–2010 (43rd–111th Congress) | Bound edition of the Congressional Record, digitized by Gentzkow, Shapiro and Taddy (GST) |
| `daily` | 1981–2016 (97th–114th) | Daily edition, same release |
| `govinfo` | 1994–2024 (103rd–118th) | Speeches downloaded from govinfo.gov and split into speaking turns with the congressional-record parser (`~/congressional-record` on the cluster) |

The main series follows GST: bound edition through the 111th Congress, daily edition for the
112th–114th, and govinfo for the 115th–118th. Overlapping congresses are kept to check the splices.
Party and voting status for govinfo come from Voteview (`HSall_members.csv`).

## Steps

1. **Build** (`01_build_speeches.py`): every speech from the raw files, with speaker information
   and flags (identified legislator, party, voting member, Extension of Remarks). Nothing dropped.
2. **Clean** (`02_clean_speeches.py`, run once): the Platforms & Policies cleaner, unchanged
   (copied in `code/vendor_pp/`, verified by md5): English speeches only (fasttext), spaCy
   `en_core_web_lg` lemmas of nouns, verbs, adjectives and proper nouns, without punctuation,
   numbers, names of people and places, words under 3 letters and the P&P stopwords.
   No length threshold.
3. **Count** (`03_count_terms.py`): counts of the curated vocabulary's words and two-word phrases in
   each speech (a two-word phrase is counted once, not also as its two words), exactly as the P&P R
   code does (`check_counts_vs_R.py` verifies it). Rerun alone for another vocabulary.
4. **GST leave-out estimator** (`04_gst_leave_out.py`): average partisanship of speech by congress,
   House and Senate pooled, with GST's random-party benchmark and subsampling intervals.
5. **Fit** (`05_fit_gtm.py`, GPU): the GTM validated in the simulations, with a separate intercept
   and party effect for every congress in both topic prevalence and word content; VAE, `w_prior = 1`.
   The sample (GST's: named Republican and Democratic voting members, no Extensions of Remarks) and
   the minimum speech length are chosen here.
6. **Exhibits** (`06_exhibits.py`): the slides' figures and tables from a fitted model.

## Running (on the cluster, in order)

```bash
sbatch slurm/01_build.sbatch          # 4 tasks, minutes
sbatch slurm/02_clean_count.sbatch    # 10 workers, hours; resumes where it stopped if rerun
sbatch slurm/04_gst.sbatch
sbatch --export=ALL,TAG=pilot_k20,FIT_ARGS="--n-topics 20 --per-cell 1500" slurm/05_fit.sbatch
```
