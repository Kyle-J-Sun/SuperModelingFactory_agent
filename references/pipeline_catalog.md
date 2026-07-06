# Pipeline catalog

Every field list and version tag below was true at some point in this project's history. **Re-confirm with live introspection (`dataclasses.fields(...)`) before filling a config from this file** — treat it as "here's where to look and what to look for," not a spec sheet.

## Decision table

| The ask sounds like… | Reach for |
|---|---|
| "Screen these N candidate features against PSI/IV/correlation, get WOE tables/plots" | `FeatureValidationPipeline` |
| "The feature set is huge (thousands of cols × hundreds of k rows), won't fit in memory at once" | `FeatureValidationPipeline` with `input_type="csv"` + `feature_batch_size` |
| "Build a scorecard / GBM model end-to-end from a labeled sample" | `CreditModelPipeline` |
| "I only have labels for approved applicants; account for the rejected population" | `RejectInferencePipeline` — feed its output into `CreditModelPipeline` via `weight_col` (fuzzy-style weighted training set) |
| "Compare how several existing score columns rank risk" | `ScoreComparisonPipeline` |
| "Check a score's behavior is consistent across time/population before/after a swap" | `ScoreConsistencyUATPipeline` *(not exercised this session — verify field names fresh)* |
| "Maturity / coverage / bad-rate / recommended ins-oos-oot split on a raw pull" | `SampleAnalysisPipeline` |
| "Synthesize a fake sample for testing" | `MockSamplePipeline` *(not exercised this session — verify field names fresh)* |
| "Screen features AND model, without re-fitting WOE twice" | `Pipeline.orchestrator.run_modeling_from_validation` (0.3.13+) — chains FVP(`selection_enabled=True`) → `FeatureScreeningArtifact.from_fvp_result` → CM(`feature_selection_mode="from_artifact"`) |
| "Just a weighted/unweighted PSI+IV+corr screen, no modeling at all" | `Feature.Feature_Screen.feature_screen` (unweighted or via `weight_col=None`) or `Feature.Weighted_Screen.weighted_feature_screen` (weighted, standalone, no Pipeline needed) |

---

## FeatureValidationPipeline

`Modeling_Tool.Pipeline.feature_validation.{FeatureValidationPipeline, FeatureValidationPipelineConfig, FeatureValidationPipelineResult}`

Distribution stats → WOE → PSI (train-fit bins applied to oos/oot) → IV/KS → correlation (new-vs-new and new-vs-incumbent), batch-capable for very wide feature sets.

Fields worth checking before assuming defaults:
- `input_type` ("dataframe" default vs `"csv"`), `feature_batch_size`, `batch_corr_mode` — CSV batch mode has **no row-filter entry point**; WOE fitting sees the whole file. If you need to exclude a subpopulation from fitting, look for `woe_fit_query` (added 0.3.6) before assuming you must pre-filter the file yourself.
- `population_dims` / `time_dims` — grouped PSI/IV eval; each dim column must be reachable through `batch_base_cols` in batch mode or it silently isn't there for the batch reader to select.
- `selection_enabled` (0.3.13+) — turns FVP into a screening step whose result can be handed straight to CM via the orchestrator; `selection_params`, `weight_col`, `selection_summary` on the result travel with it.
- `missing_rate_threshold` (0.3.15+) — when not `None`, runs as **stage 0 on INS** (before PSI): drop features with `missing_rate > threshold`. Also settable via `selection_params["missing_rate_threshold"]`. Outputs `selection_missing_rate` / `selection_missing_rate_dropped` CSVs when `write_outputs=True`.
- `psi_reference_dataset` — usually `"ins"`; confirm before assuming.
- Batch mode failure semantics: a failed batch is caught into `batch_metadata` (status=error) **without raising**, and a broken WOE fit chain silently degrades PSI/IV to a raw-value fallback. Always check `result.batch_metadata` and `woe_refine_summary` before trusting `result.validation_summary` in batch mode.
- Memory: batch size has to respect the pod's actual **cgroup** ceiling (`/sys/fs/cgroup/memory.max`), not `free -h`'s host-level number, and needs headroom for co-tenant processes.

## RejectInferencePipeline

`Modeling_Tool.Pipeline.reject_inference.{RejectInferencePipeline, RejectInferencePipelineConfig, RejectInferencePipelineResult}`

Trains a prescore on approved/labeled data, then applies one or more of `ri_methods` (`simple_augment`, `hard_cutoff`, `fuzzy_augment`, `parceling`) to assign pseudo-labels/weights to the rejected population, then trains + evaluates an RI model per method.

Fields worth checking:
- `ri_score_direction` (`"high_bad"` / `"high_good"`, added ~0.3.4) — **every inferrer's math depends on this matching the actual polarity of `score_col`.** Getting this wrong was the root cause of a real multi-method bug earlier in this package's life (hard_cutoff and fuzzy_augment both silently inverted). Confirm which way the prescore is calibrated before trusting a run.
- `ri_validation_frac` — controls whether validation is a genuine holdout distinct from train (fixed to be a real split at the same time as the direction fix).
- `weight_col` semantics differ by method: only `fuzzy_augment` populates `_weight` (approved rows weight 1, each rejected row duplicated into a weighted good/bad pair). The other three methods produce a single hard/soft label per row with no weight column.
- `save_models` / `write_ri_datasets` — writing the full augmented dataset (especially `fuzzy_augment`, which roughly doubles the rejected rows) can be tens of GB at real scale; toggle off if you don't need the file.
- `oot_data` — lets you hand RI an external, purely-observed OOT set instead of relying on `split_col`'s own OOT slice.
- To feed an RI result into `CreditModelPipeline`: read the fuzzy augmented dataset (approved + weighted rejected pairs) and pass its `_weight` column straight into CM's `weight_col`.

## CreditModelPipeline

`Modeling_Tool.Pipeline.credit_model.{CreditModelPipeline, CreditModelPipelineConfig, CreditModelPipelineResult}`

Split → feature selection (PSI/IV/corr, delegates to `Feature.Feature_Screen` as of 0.3.13) → WOE → train (`train_models` subset of lr/lgb/xgb/cat) → backward elimination → Optuna → evaluate → explain (SHAP/Owen/LIME) → Excel report.

Fields worth checking (roughly in the order they landed, since each closed a real gap):
- `weight_col` (0.3.5+) — threads through LR (`weight_col=`), GBM (`sample_weight=`/`eval_sample_weight=`), backward elimination, LR search, Optuna, and evaluation (`add_dataset_with_optional_weight`). Confirm it's present before assuming weighted training is possible; before 0.3.5 it plainly wasn't.
- `gbm_feature_source` (0.3.6+) — `"woe"` / `"raw"` / a per-model dict (`{"lgb": "raw", ...}`). LR always trains on WOE regardless of this setting. `result.model_feature_sources` reports what each model actually used.
- `woe_fit_query` (0.3.6+) — a pandas-`query()` string restricting which `ins` rows the WOE binner *fits* on, while PSI/IV/corr/transform still run against the full split. This is the right tool for "exclude population X from binning, keep it for evaluation" — don't hand-filter the input DataFrame for this if the field exists.
- `extra_eval_datasets` (0.3.6+) — a `{name: DataFrame}` dict of eval-only slices (never touch training/selection/backward/Optuna) that get WOE-transformed (or left raw, matching each model's `gbm_feature_source`) and appended to every model's perf table under that name. This is the correct way to get a held-out subgroup's performance without a hand-written scoring cell.
- `feature_selection` dict — as of 0.3.13 this is passed through `screen_config_from_mapping` into `Feature.Feature_Screen.feature_screen`; look there (not in `credit_model.py`) for the actual PSI/IV/corr mechanics, including `psi_use_woe_bins` / `iv_use_woe_bins` / `corr_use_woe_bins` (share one fitted binner across all three steps instead of each computing its own bins) and (0.3.14+) `corr_nan_policy` / `on_empty_stage`.
- `screening_artifact` / `feature_validation_result` / `feature_selection_mode` / `reuse_screening_woe` (0.3.13+) — lets CM consume a `FeatureValidationPipeline` screening result directly instead of re-screening; see the orchestrator entry below.
- `warm_start_enabled` / `warm_start_score_col` / `warm_start_score_type` (`"probability"`/`"log_odds"`) / `warm_start_models` (default `["lgb","xgb"]` — **cat and lr are not warm-started even if requested via `on_unsupported`**) / `warm_start_on_unsupported` / `warm_start_apply_to_optuna`. Probability is clipped to `[1e-6, 1-1e-6]` then log-odds'd internally; a NaN in `warm_start_score_col` raises rather than silently zero-filling — clean the column yourself first.
- `save_models` — model pkl written via SMF's own container format; **load with `Core.load_model(path, return_metadata=True)`, never bare `pickle.load`** (raises a confusing `No module named '1'`-style error otherwise). `return_metadata=True` gives you `feature_cols`, `feature_source`, `warm_start_*`, `model_params`, etc. — the fastest way to recover what a saved run actually did without re-reading the notebook.
- `explain_models` (subset of trained model names) / `owen_enabled` — **Owen values are deliberately skipped for `xgb`** (`name != "xgb"` in `_run_explainability`) regardless of `explain_models`/`owen_enabled`; that's by design, not a bug. As of some version, `explain_outputs` (SHAP/Owen tables) is *not* persisted to disk even though an `explain/` directory gets created for it — if you need it after the kernel dies, either add a persistence cell yourself or check whether a newer version has closed this gap (`persist_explain_outputs` in `Pipeline._common` — confirm it's wired into `_run_explainability`'s write path, not just importable). LIME is *not* driven through this path before 0.3.13 — see `known_gotchas.md`.
- `perf_pct_bins` / `perf_min_bin_prop` — gains-table bucketing for evaluation; small subgroups (a few hundred rows) can produce unstable/degenerate bins here — sanity check `N` per bin before trusting IV on a small `extra_eval_datasets` slice.

## ScoreComparisonPipeline

`Modeling_Tool.Pipeline.score_comparison.{ScoreComparisonPipeline, ScoreComparisonPipelineConfig}`

Compares `score_cols` (existing model outputs) against `target_col`, globally and cross-cut by `time_dims`/`population_dims`.

- `sample_name` on the underlying `PerformanceEvaluator.model_perf_compare` used to be hardcoded to `"oot"` regardless of what population was actually passed in (misleading label on both the global table and every group-by table). Fixed by threading an explicit `sample_name` parameter through from the pipeline (defaults to `"all"` when called directly, not `"oot"`) — if you ever call `model_perf_compare` yourself outside the pipeline, pass `sample_name=` explicitly rather than relying on the default.
- `group_min_size` — set this for any population dim with small cells (a niche channel, a rare segment); otherwise a 50-row group still gets a full 10-bin gains table and its IV/KS reads as noise, not signal.

## SampleAnalysisPipeline

`Modeling_Tool.Pipeline.sample_analysis.{SampleAnalysisPipeline, SampleAnalysisPipelineConfig}`

Label maturity/coverage, segment bad-rate, and INS/OOS/OOT split recommendation on a raw pull.

- `target_cols` (plural — supports comparing multiple maturity windows, e.g. `is_dpd7_in_42d` vs `is_dpd7_in_43d`, at once).
- `approved_col` — when the column named here isn't present in the data, `label_coverage_summary`'s `approved_col` field reports `None` (fixed from an earlier version that filled in a misleading hardcoded default name instead).
- `oot_windows` / `ins_oos_ratios` / `random_seeds` — the split recommender searches this grid and reports back a recommendation; it's a search, not a single deterministic split — read `split_candidate_summary` if you need to see the alternatives it rejected.

## Pipeline.orchestrator.run_modeling_from_validation (0.3.13+)

`Modeling_Tool.Pipeline.orchestrator.run_modeling_from_validation(data, *, fvp_config=None, cm_config=None, selection_enabled=True, reuse_screening_woe=True)`

Runs `FeatureValidationPipeline` with selection turned on, wraps its result in a `FeatureScreeningArtifact` (`Pipeline.screening_artifact`), and passes that straight into `CreditModelPipeline` (`feature_selection_mode="from_artifact"`, `target_col`/`weight_col`/`feature_cols` taken from the artifact). Reach for this instead of hand-chaining FVP→CM when you don't need anything custom between the two steps — it also guarantees the WOE binner isn't re-fit twice (`reuse_screening_woe=True`).

## Feature.Feature_Screen.feature_screen / screen_config_from_mapping (0.3.13+)

The primitive that `CreditModelPipeline`'s `feature_selection` dict now delegates to internally. Screening order (0.3.15+): **`missing_rate` (optional) → PSI → IV → corr**. Call it directly when you want a screen **without** building a full CM run around it — e.g. exploring how a shared-binner (`monotone`, with `psi_use_woe_bins`/`iv_use_woe_bins`/`corr_use_woe_bins=True`) screen differs from independent per-step bins, before committing to a CM config.

Key `FeatureScreenConfig` fields (confirm live): `missing_rate_threshold`, `missing_rate_ref`, `categorical_features` (for `MonotoneWOEBinner.cate_feats` when using shared bins), `corr_nan_policy`, `on_empty_stage`.

## Feature.Weighted_Screen.weighted_feature_screen (0.3.7+, corrected 0.3.14)

Standalone weighted PSI/IV/corr screening — the tool for screening a fuzzy-augmented (or any sample-weighted) dataset properly, since neither `feature_screen` nor CM's own `feature_selection` weighted the fuzzy double-rows correctly before this existed. Key params: `weight_col`, `corr_nan_policy` (0.3.14+: `"pairwise"` default / `"median_fill"` / `"raise"` — **pre-0.3.14, correlation on data with NaNs silently collapsed toward zero and let highly-correlated-but-incomplete feature pairs both survive**, see `known_gotchas.md`), `on_empty_stage` (0.3.14+: `"keep_all_warn"` default / `"raise"` — a stage that eliminates every remaining variable now warns and logs a `*_fallback` row in `.summary` instead of silently keeping everything with no trace).
