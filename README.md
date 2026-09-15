MC³-GRAM — reproducibility code

Code accompanying A Generative-Augmented Federated Learning Framework for Intrusion Detection with Multi-Head Feature Masking and Robust Aggregation (Complex & Intelligent Systems, under revision).

Read this before anything else

This is not the code that produced Tables 6–22 of the manuscript. It is an independent reimplementation built from the equations of Section 3. Its absolute rates do not reproduce those tables and are not meant to: different code, different seeds, and — for the supplementary study — a smaller corpus partition and a much shorter schedule.

What this code is for is the two results that are attributed to it in the paper, and which are labelled as such there:

Result	Where in the paper	Where here
Per-client SMOTE ablated in isolation, full UNSW-NB15 capture	Table 23, §4.11	ablation/
The nine-experiment supplementary study (E0–E9)	Appendix A	supplementary/

Comparisons are valid within a table produced here, because those rows share a protocol by construction. Comparisons between these tables and the manuscript's main results are not valid, and the paper says so in the captions.

Install
bash
pip install -r requirements.txt
python ablation/selftest.py     # ~1 minute, synthetic data, CPU and GPU

selftest.py runs every ablation variant end to end and checks that each table it produces satisfies FNR = 1 − Recall and F1 = 2PR/(P+R). Device mismatches are the characteristic bug in code like this — invisible on a CPU-only machine, and on a GPU they surface only when the offending line is first reached, sometimes after the training is already spent. Run it before any long job.

Corpora

Not redistributed here. Both are freely available from their authors.

UNSW-NB15, full capture — UNSW-NB15_1.csv … UNSW-NB15_4.csv plus NUSW-NB15_features.csv (the misspelling is the dataset author's). 2,540,044 records. https://research.unsw.edu.au/projects/unsw-nb15-dataset
UNSW-NB15, curated ten-class partition — UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv, 257,673 records. Used by the supplementary study; place under supplementary/data/.
CICIDS-2017 — MachineLearningCVE/, eight files, 2,830,743 records. https://www.unb.ca/cic/datasets/ids-2017.html. Do not use GeneratedLabelledFlows; it is a different extraction.

The two UNSW releases are not interchangeable. Eight of Table 3's ten class counts match the curated release exactly and two do not, because the dataset authors subsampled Normal to 93,000 and Generic to 58,871 there.

Reproducing Table 23
bash
cd ablation
python run_ablation.py --dataset unsw-full --root /path/to/UNSW-NB15 --check
python run_ablation.py --dataset unsw-full --root /path/to/UNSW-NB15 \
    --variants full no_smote
python finalize_smote.py

--check trains nothing. It compares every class count with Table 3 and verifies that the test partition has not been rebalanced — the defect recorded in §4.13, and the check that keeps it from recurring. If it fails, stop; no amount of care downstream fixes the wrong corpus.

The two arms share splits, Dirichlet draw and seeds. The per-client resampling step is the only difference, which is what makes the comparison controlled.

finalize_smote.py reports the 0 % (clean) numbers only. See "Known limitation" below for why.

Reproducing Appendix A

See supplementary/README.md. out/results.jsonl in that folder holds the 100 committed result rows behind the published tables; tables.py regenerates the LaTeX from them and refuses to write anything if a row violates FNR = 1 − Recall or F1 = 2PR/(P+R). Those hold by construction for numbers derived from a confusion matrix, so a failure means the pipeline is wrong rather than the rounding — and they are the first thing a reader checks.

Known limitation: the perturbation in ablation/ is not adversarial

perturb.generator_perturbation computes

python
delta = (prop - xb).clamp(-eps, eps) * mask

where prop comes from a generator trained to emit realistic class-conditional samples. It therefore moves each sample toward its class prototype: it denoises rather than attacks. The tell is that macro-F1 rises with the budget (0.4949 → 0.4946 → 0.5095 → 0.5234 across 0/10/20/30 %), and no adversarial budget can improve a model.

Consequences, and they are deliberate rather than pending:

Only the 0 % column is reported. finalize_smote.py reports nothing else, and Table 23 in the paper states the scope: the contribution of resampling under evasion is left unquantified.
The budget columns emitted by emit_tables.py should not be pasted into Tables 21/22 of the manuscript.
perturb.pgd_perturbation in the same file is a correct PGD implementation and is unaffected. Reusing it is the fix, at the cost of a full re-run.
Note on versions

ablation/config.py carries a VERSION string, printed by every entry point. The runs behind Table 23 were produced by 2026-09-15.4. The resume key gained a protocol fingerprint in .5, so those records' keys have three fields (dataset|variant|fold) rather than four. finalize_smote.py handles both: with a fingerprint it keeps the majority protocol, and without one it identifies timing-probe folds by runtime, since a three-round probe costs a small fraction of a thirty-round fold. Neither route edits results.jsonl.

Committing your run records

out/results.jsonl is committed on purpose in supplementary/ — it is the evidence behind the published numbers, and it is small. Do the same for the ablation: copy your completed out/results.jsonl into ablation/out/ before pushing, so that anyone can re-derive Table 23 with finalize_smote.py without a GPU. Everything else under out/ is ignored by .gitignore.

Layout
ablation/                 Table 23 — per-client SMOTE, full UNSW-NB15
  config.py               every hyperparameter, transcribed from Table 4
  data.py                 three loaders; split -> Dirichlet -> SMOTE -> normalise
  model.py                stem, three heads, Jacobian attribution, mask consensus, cGAN
  perturb.py              perturbation budgets; rates from the confusion matrix
  fed.py                  federated loop: drift, trust, trimmed geometric median
  selftest.py             every path on synthetic data, CPU and GPU
  run_ablation.py         the driver
  emit_tables.py          results.jsonl -> LaTeX, with the identity checks
  finalize_smote.py       the clean SMOTE comparison, ready to paste

supplementary/            Appendix A — the nine-experiment study (E0-E9)
  dataset.py model.py fed.py run.py
  tables.py               results.jsonl -> LaTeX, refuses to emit on a violation
  prevalence.py           recovers class prevalence from published rates
  cicids_projection.py    projects balanced rates onto the natural distribution
  smote_mechanism.py verify_smote_table.py check_paper.py build_standalone.py
  out/results.jsonl       100 committed result rows

prevalence.py and cicids_projection.py need no data at all — they operate on the rates printed in the manuscript.

Licence

No licence file is included, which under default copyright means others may read this but not reuse it. Add one before publishing if you intend otherwise; MIT or Apache-2.0 are the usual choices for code accompanying a paper. Note that the corpora above carry their own terms and are not redistributed here.
