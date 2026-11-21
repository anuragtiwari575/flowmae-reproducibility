# mc3gram-reproducibility

Replication package for **FlowMAE: Masked-Column Pretraining for
Clean-Accuracy Recovery in Federated Adversarial Intrusion
Detection**.

The repository is named for MC3GRAM, the convolutional–recurrent
detector that is the baseline of this study (arm B) and the subject of
the earlier work this line of experiments grew out of. Everything here
belongs to the FlowMAE paper: the same code base produces both the
baseline and the proposed model, so they are released together.

Every number in the paper is produced from the per-cell result records
in `records/`. The tables and figures are generated from those records
by script, not transcribed, so a value in the text and the record it
came from cannot drift apart.

## Layout

    code/      model, drivers, attack suite, analysis and output scripts
    records/   per-cell JSON result records, pretraining logs, attack
               and recoverability outputs
    MANIFEST.txt   SHA-256 of every file in this package

## Reproducing the tables and figures

No GPU is needed for this step; it reads the records only.

```
cd code
python -u tables_flowmae_v4.py --out tab_perfold.tex
python -u figs_flowmae_v6.py --out figs
```

## Reproducing the runs

The corpora are public and are not redistributed here. UNSW-NB15 is
distributed by the Australian Centre for Cyber Security and
CICIDS-2017 by the Canadian Institute for Cybersecurity. Place them
under `data/` and run, for example:

```
python -u phase1_arms_v17.py --corpus unsw --device cuda \
    --arms B --attack-rows 200000 \
    --train-spec binary-only --attack-spec binary-only

python -u run_flowmae_v8.py --corpus unsw --device cuda \
    --arms FMB --mcp-init out/mcpv2_unsw_f{fold}.pt \
    --attack-rows 200000 \
    --train-spec binary-only --attack-spec binary-only
```

CUDA reductions are not deterministic across runs. Repeating an
identical configuration moved clean macro-F1 by approximately 0.011 on
one CICIDS-2017 fold, so no claim of bit-exact reproducibility is
made; every quantity in the paper is a mean over five folds reported
with its standard deviation.

## Arms

| Arm | Architecture | Adv. training | MCP | Extra |
|---|---|---|---|---|
| B | MC3GRAM (convolutional–recurrent) | yes | – | – |
| C | MC3GRAM | yes | – | generative augmentation |
| F | FlowMAE | – | – | – |
| FB | FlowMAE | yes | – | – |
| FM | FlowMAE | – | yes | – |
| FMB | FlowMAE | yes | yes | – |
| FB+CD | FlowMAE | yes | – | per-column dropout, p = 0.2 |
| FB+GDR | FlowMAE | yes | – | gradient-dispersion penalty |

All eight were run on UNSW-NB15. CICIDS-2017 carries B, C, FB and FMB;
the complete factorial is therefore on UNSW-NB15, and CICIDS-2017
provides cross-dataset validation of the performance trend rather than
of the decomposition.

## What is not included

The corpora, which are public and cited in the paper; model
checkpoints, which are large and reproducible from this code; and
derived caches, which are rebuilt on demand.
