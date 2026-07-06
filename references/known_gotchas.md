# Known gotchas — a history log, not a current-state list

Every entry is tagged with when it was true. **Before repeating one of these to the user, check whether the installed version is at or past the fix** (introspect the relevant source — `inspect.getsource`) and say so either way: "this was real before v0.3.X, confirmed fixed in your current source" is a better answer than silently re-warning about something already closed, and better than silently assuming it's fixed without checking.

## Reject inference direction bugs (fixed 0.3.3)

Root cause across all of these: `Sample.Reject_Infer`'s inferrers were written assuming "higher score = lower risk," while `RejectInferencePipeline` fed them `prescore_prob` = P(bad) — the opposite polarity — with no parameter to reconcile the two.

- **HardCutoffInferrer** — `target = (score < cutoff)` under P(bad) input flags the *safest* rejects as bad and vice versa. Symptom that gave it away: inferred reject bad-rate came out *lower* than the real approved bad-rate, which makes no business sense (rejects should look worse, not better).
- **FuzzyAugmentInferrer** — rejected pseudo-label was `1 - score` (wrong direction, and continuous rather than a proper good/bad split); approved-row reweighting formula additionally suppressed toward zero any real row the prescore had misjudged, amplifying the prescore's own bias instead of correcting for it.
- **ParcelingInferrer** — approved population binned via `qcut` (equal-frequency), rejected population binned via `cut` (equal-width, over the *rejected* population's own score range) — two unrelated bin systems mapped onto each other by index, effectively assigning near-random bad rates.
- **Pipeline-level (`_train_and_evaluate_models`)** — `split_col` mode set `val = approved.copy()`, i.e. validation was literally the training population; every non-weighted method's `train_AUC` and `validation_AUC` came out identical to 6 decimal places, and GBM early stopping had no real holdout to stop against.
- **Continuous pseudo-labels got hard-binarized at `>0.5`** downstream in the pipeline, compounding whichever of the above was already wrong.
- **SimpleAugmentInferrer** hardcoded `np.random.seed(42)`, polluting the global RNG state and ignoring `cfg.random_state` (lower severity — the method is "assign the approved bad-rate at random" by design, so the direction bugs above don't apply to it, but reproducibility did).

**Fix:** `score_direction` parameter added to every inferrer + `ri_score_direction` (`"high_bad"`/`"high_good"`) on the pipeline config; `_default_hard_cutoff`'s percentile flips with direction; fuzzy rewritten to the textbook form (approved weight 1, rejected duplicated into weighted good/bad rows); parceling given shared `parcel_edges_`; `ri_validation_frac` added for a genuine holdout; SimpleAugment moved to an instance RNG.

**Durable lesson:** any score-consuming module needs its polarity convention pinned and checked against what's actually fed in. These bugs threw no exceptions — they just produced confidently-wrong, business-plausible-looking numbers (a "hard cutoff" method assigning rejects a *lower* bad rate reads as surprising but not obviously broken until you check the arithmetic).

## Weighted screening bugs (introduced 0.3.7, fixed 0.3.14)

- **Correlation silently blind to NaNs.** `_weighted_pearson_corr_matrix` ran `np.cov(..., aweights=w)` directly on raw (possibly-NaN) columns; any NaN anywhere in a column propagates `cov` to NaN, which then got `np.nan_to_num(nan=0.0)`'d — i.e. any feature pair where either column has missing values reads as *uncorrelated*, and both survive correlation dedup untouched. On this project's real feature matrices (20–35% NaN from cleaned sentinel values) this made the correlation stage close to a no-op for most pairs. Confirmed empirically with a synthetic 30%-NaN pair at true correlation ≈0.999: both columns survived pre-fix. The weighted path also ignored `corr_use_woe_bins` (the unweighted path respected it).
- **Silent "everything survives" fallback.** Every stage (PSI/IV/corr) used `current = keep or current` — if a threshold eliminated every remaining variable, the function quietly returned the untouched input set with no warning and no trace in the summary table.

**Fix (0.3.14):** `corr_nan_policy` (`"pairwise"` default — complete-case overlap per pair; `"median_fill"`; `"raise"`); weighted corr now also respects `corr_use_woe_bins`; `on_empty_stage` (`"keep_all_warn"` default — same fallback behavior, but now `warnings.warn`s and appends a `*_fallback` row to `.summary`; `"raise"` for strict use).

**Durable lesson:** a covariance/correlation computation over a real feature matrix needs an explicit NaN policy stated out loud — the numpy default ("propagate to NaN, then someone downstream coerces NaN to 0") is silently equivalent to "assume uncorrelated," which is the least safe possible default for a dedup step. Separately: any "keep everything" fallback is only a safety net if it's loud: a silent one is indistinguishable from "the filter worked."

## `apply_woe` O(n²) performance (fixed before 0.3.3)

`WOE_Monotone_Binner.apply_woe` inserted new WOE columns one at a time via `df[col] = ...`, each insertion forcing pandas to reallocate its internal block manager. On wide batches (100–250+ new columns per call) this dominated runtime and was the confirmed (via `py-spy` stack sampling of a live, stuck process) root cause of a 3,380-candidate-feature validation run blowing through a 12-hour cell timeout with 14/14 batches still not finished. Fixed by collecting the new columns and adding them via a single `pd.concat(axis=1)`.

**Durable lesson:** if a WOE/feature-assembly step scales worse than linear with column count, suspect column-by-column DataFrame mutation before suspecting the statistics.

## Persistence

- **Always load SMF-saved models via `Core.load_model(path, return_metadata=True)`.** A bare `pickle.load` on an SMF model artifact raises a confusing `No module named '1'`-shaped error — SMF wraps saved models in its own versioned container, not a plain pickle of the estimator.
- **`LRMaster` used to pickle its full training frame (`_data`)** — tens to hundreds of MB, and literally row-level customer data riding along inside a model artifact (a real governance issue, not just bloat). Fixed via `__getstate__`/`__setstate__` dropping `_data` on save. Consequence: `get_statsmodel_summary` / `get_aic` / `get_bic` called on a *loaded* model now require an explicit `data=` kwarg and raise a clear `ValueError` telling you so if you omit it — this is intentional, not a regression.
- **`CreditModelPipelineConfig` had no `save_models` field at all** before it was added — the `models/` output directory was created either way (dead giveaway that something's missing, not that nothing was supposed to be there) but stayed empty until the field landed.

## Evaluation-table labeling

- **`model_perf_compare`'s `sample_name` was hardcoded to `"oot"`** regardless of what population was actually being scored — every table it powers (including every one of `ScoreComparisonPipeline`'s global and group-by outputs) had its `index`/scope column mislabeled `"oot"` even for the full sample or an arbitrary subgroup. Fixed by adding an explicit `sample_name` parameter; `ScoreComparisonPipeline` now passes `"global"` at every internal call site, and the bare function's own default changed to the neutral `"all"`. If anything calls `model_perf_compare` directly without passing `sample_name`, its output labeling changed meaning across this version bump — check call sites, not just the pipeline.

## Explainability

- **Owen values deliberately skip `xgb`** — `_run_explainability`'s Owen branch is gated `name != "xgb"`. By design, not a bug; don't "fix" it without finding out why first.
- **LIME was fully hand-rolled through 0.3.6**, absorbed natively into `Explainability.Model_Explainer` (`lime_explain_instance`, `lime_global_importance`, with built-in missing-value handling) as of **0.3.13**. This is the concrete precedent for the escalation ladder in `SKILL.md`: a hand-built workaround (including its NaN-median-fill fix for the fact that raw/non-WOE features commonly contain NaNs, which vanilla LIME's sampler can't handle — `scipy.stats.truncnorm: scale must be positive`) can be quietly absorbed upstream. **Check `ModelExplainer` for a native method before hand-rolling an explainer again on any version ≥0.3.13.**
- **`explain_outputs` (the computed SHAP/Owen tables) may not be persisted to disk** even when an `explain/` output directory is created and `explain_models`/`owen_enabled` ran without error — the tables can live only in the in-memory `CreditModelPipelineResult` for that kernel's lifetime. If the kernel dies before you persist them yourself (a later cell crashing, an OOM, a LIME NaN blowup), they're gone; recover by reloading the saved model via `Core.load_model(..., return_metadata=True)` (for `feature_cols`/`feature_source`) and recomputing with the same `sample_n`/`background_n`/`random_state`. Check whether the installed version's `_run_explainability` calls a persistence helper (look for something like `persist_explain_outputs` in `Pipeline._common` actually being invoked, not just importable) before assuming this gap is still open.
- **Environment, not SMF:** a long stretch of this project ran with `explain_models=[]` as a standing workaround because the installed `numba` (0.56.4, compiled against numpy<1.24) was ABI-incompatible with the installed `numpy` (2.2.6) — `import shap` crashed transitively via numba. `_run_explainability` catches this per-model and returns `{"error": repr(exc)}` rather than failing the whole pipeline, which can look like "explainability isn't configured" rather than "the environment is broken" — if a model's `explain_outputs` entry has an `"error"` key, check the environment before the config. Working combination found: `numba==0.66.0` + `llvmlite==0.48.0` + `lime==0.2.0.1` + `scikit-image>=0.25` + `pywavelets>=1.8`, with numpy/shap/lightgbm/xgboost/catboost left untouched.

## Data hygiene that belongs *before* the SMF call, not inside one

Neither of these is an SMF bug — both are exactly the kind of requirement that feels like "I'll just hand-filter the DataFrame" until you check whether a Config field already covers it:

- **Negative sentinels in CDC feature/score columns** (`-1`, `-2`, `-100`, observed at ~35% of an entire feature matrix's cells by count) are not valid numeric values. SMF's binners will bin them as real numbers if you feed them in as-is — always `NaN` them first, and do it consistently for every column an evaluation touches (score columns included, not just model features).
- **A population that should be excluded from fitting but retained for evaluation** (this project's "FT" cohort — a natural-approval segment that bypasses normal model scoring) is handled by `woe_fit_query` (fit-time exclusion) + `extra_eval_datasets` (eval-only inclusion) on the relevant pipeline, once those fields exist on the installed version — check before hand-splitting the DataFrame yourself.
