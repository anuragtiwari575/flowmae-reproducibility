@echo off
setlocal
REM ======================================================================
REM  run_review_v1.bat -- the experiments the review asked for, in the
REM  order it ranked them.
REM
REM  RUN FROM THE (base) CMD PROMPT, in supplementary\, with nothing
REM  else on the GPU.
REM
REM  WHAT IS HERE AND WHY IN THIS ORDER
REM    1  CICIDS-2017 arms F and FM. This completes the 2x2 factorial on
REM       the second corpus, which the review calls the highest-value
REM       additional experiment: without it the component decomposition
REM       rests on UNSW-NB15 alone.
REM    2  Strong PGD on CICIDS-2017. Completes Table 6 and the right
REM       panel of Figure 8; without it the masking diagnosis on that
REM       corpus rests on the non-monotone curve and CW alone.
REM    3  Gradient-boosted-tree reference on both corpora, same folds and
REM       preprocessing. Answers "does it beat strong tabular learners".
REM       Cheap: no federation, no attacks, clean performance only.
REM    4  Seed repetition for the headline contrast, folds held fixed.
REM       Separates run-to-run variance from fold variance.
REM    5  Dirichlet sweep. Tests whether the clean gain survives severe
REM       heterogeneity.
REM
REM  Steps 1 and 2 are the two that change what the paper may claim.
REM  Steps 3 to 5 strengthen it. Everything resumes, so stopping and
REM  restarting costs only cache checks.
REM
REM  ROUGH COST on the 8 GB laptop card, from the runs already done:
REM    step 1  ~6 h per arm, so ~12 h
REM    step 2  ~2 h
REM    step 3  ~20 min
REM    step 4  ~6 h per extra seed for the two headline arms
REM    step 5  ~6 h per alpha value
REM  Do not start all of it at once. Steps 1 and 2 first.
REM ======================================================================

set PY=python -u
set ROWS=200000
set CICROOT=data/cicids2017
if not exist out mkdir out
set Q=out\review_v1.log

set PYTORCH_ALLOC_CONF=expandable_segments:True

echo. >> %Q%
echo ====================================================== >> %Q%
echo REVIEW QUEUE START %DATE% %TIME% >> %Q%
echo ====================================================== >> %Q%

REM ---------------------------------------------------------------- 1a
echo [%TIME%] step 1a  CICIDS arm F  (FlowMAE, no adversarial training)
echo [%TIME%] step 1a cicids F >> %Q%
%PY% run_flowmae_v8.py --corpus cicids --task multiclass8 --root %CICROOT% --device cuda --attack-rows %ROWS% --attack-bs 512 --train-spec binary-only --attack-spec binary-only --arms F --cache-dir out/models_flowmae_cic_F --out out/flowmae_cicids_plain.jsonl > out\log_rev_cic_F.txt 2>&1
echo [%TIME%] step 1a exit %ERRORLEVEL% >> %Q%

REM ---------------------------------------------------------------- 1b
REM FM reuses the MCP encoders already pretrained for FMB -- the
REM pretraining is independent of whether adversarial training follows,
REM so this costs no extra pretraining.
echo [%TIME%] step 1b  CICIDS arm FM  (FlowMAE + MCP, no adv. training)
echo [%TIME%] step 1b cicids FM >> %Q%
%PY% run_flowmae_v8.py --corpus cicids --task multiclass8 --root %CICROOT% --device cuda --attack-rows %ROWS% --attack-bs 512 --train-spec binary-only --attack-spec binary-only --arms FM --mcp-init out/mcpv2_cicids_f{fold}.pt --cache-dir out/models_flowmae_cic_FM --out out/flowmae_cicids_fm.jsonl > out\log_rev_cic_FM.txt 2>&1
echo [%TIME%] step 1b exit %ERRORLEVEL% >> %Q%

%PY% analyze_v6.py out/flowmae_cicids_plain.jsonl out/flowmae_cicids_fm.jsonl out/flowmae_cicids_mcp.jsonl out/flowmae_cicids_m8.jsonl out/phase13_cicids_m8_permissive.jsonl > out\analysis_cicids_factorial_v6.txt 2>&1

REM ----------------------------------------------------------------- 2
echo [%TIME%] step 2  strong PGD on CICIDS
echo [%TIME%] step 2 strong pgd cicids >> %Q%
%PY% strong_pgd_v3.py --corpus cicids --task multiclass8 --root %CICROOT% --device cuda --attack-bs 512 --rows 20000 --entry B:mc3gram:out/models_permissive --entry FB:flowmae:out/models_flowmae_cic --entry FMB:flowmae:out/models_flowmae_cic_mcp --out out/strong_pgd_cicids.txt > out\log_rev_strong_cic.txt 2>&1
echo [%TIME%] step 2 exit %ERRORLEVEL% >> %Q%

REM ----------------------------------------------------------------- 3
echo [%TIME%] step 3  gradient-boosted-tree reference, both corpora
echo [%TIME%] step 3 gbdt >> %Q%
%PY% gbdt_ref_v1.py --corpus unsw --task multiclass10 --device cpu --out out/gbdt_unsw.json > out\log_rev_gbdt_unsw.txt 2>&1
%PY% gbdt_ref_v1.py --corpus cicids --task multiclass8 --root %CICROOT% --device cpu --out out/gbdt_cicids.json > out\log_rev_gbdt_cic.txt 2>&1
echo [%TIME%] step 3 done >> %Q%

REM ----------------------------------------------------------------- 4
REM Seed repetition, folds held fixed. Only the two arms the headline
REM contrast is made of, because that is the claim the repetition is
REM meant to support.
echo [%TIME%] step 4  seed repetition, UNSW, arms B and FMB
echo [%TIME%] step 4 seeds >> %Q%
REM phase1_arms_v17 has no --seed option, and adding one without
REM putting the seed in the model fingerprint would make the second
REM seed silently reload the first seed's checkpoints and report a
REM variance of zero. seedrun_v1.py does both: it seeds torch, numpy
REM and random, and appends the seed to the fingerprint. The folds and
REM client partitions are unchanged, which is the point.
for %%s in (1 2) do (
  %PY% seedrun_v1.py --seed %%s --driver phase1_arms_v17 -- --corpus unsw --device cuda --attack-rows %ROWS% --train-spec binary-only --attack-spec binary-only --arms B --cache-dir out/models_seed%%s --out out/seed%%s_unsw_B.jsonl > out\log_rev_seed%%s_B.txt 2>&1
  %PY% seedrun_v1.py --seed %%s --driver run_flowmae_v8 -- --corpus unsw --device cuda --attack-rows %ROWS% --train-spec binary-only --attack-spec binary-only --arms FMB --mcp-init out/mcpv2_unsw_f{fold}.pt --cache-dir out/models_seed%%s_fmb --out out/seed%%s_unsw_FMB.jsonl > out\log_rev_seed%%s_FMB.txt 2>&1
)
echo [%TIME%] step 4 done >> %Q%

REM ----------------------------------------------------------------- 5
echo [%TIME%] step 5  Dirichlet sweep, UNSW, arms B and FMB
echo [%TIME%] step 5 alpha sweep >> %Q%
for %%a in (0.1 1.0 10.0) do (
  %PY% phase1_arms_v17.py --corpus unsw --device cuda --attack-rows %ROWS% --train-spec binary-only --attack-spec binary-only --arms B --alpha %%a --cache-dir out/models_a%%a --out out/alpha%%a_unsw_B.jsonl > out\log_rev_a%%a_B.txt 2>&1
  %PY% run_flowmae_v8.py --corpus unsw --device cuda --attack-rows %ROWS% --train-spec binary-only --attack-spec binary-only --arms FMB --alpha %%a --mcp-init out/mcpv2_unsw_f{fold}.pt --cache-dir out/models_a%%a_fmb --out out/alpha%%a_unsw_FMB.jsonl > out\log_rev_a%%a_FMB.txt 2>&1
)
echo [%TIME%] step 5 done >> %Q%

echo. >> %Q%
echo REVIEW QUEUE END %DATE% %TIME% >> %Q%
echo.
echo ============================================================
echo  done.  read out\review_v1.log, then
echo     out\analysis_cicids_factorial_v6.txt   -- the 2x2 on corpus 2
echo     out\strong_pgd_cicids.txt              -- the masking check
echo     out\gbdt_*.json                        -- tabular reference
echo  then regenerate the tables and figures:
echo     python -u tables_flowmae_v1.py --out tab_perfold.tex
echo     python -u figs_flowmae_v3.py --out figs
echo ============================================================
endlocal
