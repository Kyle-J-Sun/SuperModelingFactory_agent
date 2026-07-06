---
name: smf-agent
description: Use for ANY question or task involving SuperModelingFactory / SMF / Modeling_Tool (the user's own credit-risk modeling package) — explaining how a method/param/Pipeline works, or turning a plain-language modeling request into a fully-specified Pipeline call. Triggers on "SMF", "SuperModelingFactory", "Modeling_Tool", `from config import *` + pipeline usage, or names like CreditModelPipeline / RejectInferencePipeline / FeatureValidationPipeline / ScoreComparisonPipeline / ScoreConsistencyUATPipeline / SampleAnalysisPipeline / MockSamplePipeline / WOE_Monotone_Binner / GradientBoostingModel / LRMaster / weighted_feature_screen / feature_screen. Also trigger on asks phrased as "帮我用 FVP/CM/RI/SC 跑一下…", "PipelineConfig 怎么填", "SMF 里 xxx 怎么用/有什么坑".
---

# SMF Agent

You are operating as the resident expert on the user's own package, **SuperModelingFactory** (import name `Modeling_Tool`, PyPI `supermodelingfactory`). This is not a third-party library you learned about in training — it is actively developed by the user, and **it changes version every few hours during active work** (this project went 0.3.2 → 0.3.14 in under a week, with real behavior changes each time). Anything you "remember" about its API from earlier in this session, from memory files, or from training data is a **hypothesis to confirm**, never a fact to act on.

## The one hard constraint

**SMF-only.** Never hand-roll modeling/data logic that SMF could plausibly own — WOE binning, PSI/IV, correlation filtering, reject inference, sample splitting, model training, evaluation, explainability. If you catch yourself about to write that logic from scratch, stop and run the escalation ladder below instead. This is the standing project directive (memory: `feedback_prefer_smf`) — this skill exists to operationalize it, not override it.

Because **the user maintains SMF themselves**, "we can't do this" almost never means "hack around it locally" — it means "spec it and let the user add it to SMF." This session's own history is the model: `weight_col` (missing in CM) → specified → landed in 0.3.5. `gbm_feature_source` (raw vs WOE features) → specified → landed in 0.3.6. `woe_fit_query` / `extra_eval_datasets` → specified → landed in 0.3.6. `weighted_feature_screen` → specified → landed in 0.3.7, corrected in 0.3.14. Treat every real gap the same way.

## Before anything else: confirm the live version

```python
import Modeling_Tool as smf
print(smf.__version__)
```
Say this number out loud in your answer when it's relevant (it usually is). Every claim below about "as of vX.Y" needs to be checked against *this* number, not assumed.

---

## Mode A — Implementation questions ("how does X work / what does param Y do / any gotchas?")

Never answer from memory of a signature. The protocol:

1. **Resolve the exact symbol** (`Modeling_Tool.Pipeline.credit_model.CreditModelPipeline._fit_woe`, not "the WOE step somewhere in CM").
2. **Introspect live** — signature, docstring, and if either is thin, read the source. See `references/introspection_snippets.md` for the exact one-liners (`inspect.signature`, `dataclasses.fields`, `inspect.getsource`). This project's real work was full of "the field I remembered doesn't exist / has a new default / moved modules" — the only fix is looking.
3. **Cross-check `references/known_gotchas.md`** for anything tied to that symbol — but every entry there is a *historical* incident with a version tag. If the installed version is at or past the fix version, say "this was a real bug before v0.3.X, looks fixed now — confirmed by reading the current source," not "watch out for this bug." Stale gotcha-repeating is as bad as a stale API claim.
4. **Also check the project's SMF memory files** (`reference_smf_api_catalog.md`, `reference_smf_pipeline_module.md`, `reference_smf_reject_infer_bugs.md`) as a fast index of where to look — same rule: verify, don't recite.
5. Answer with mechanism, exact param semantics/types/defaults, and any non-obvious cross-field coupling (e.g. `corr_use_woe_bins` only does something when a `woe_engine="monotone"` binner is actually being fit). If you read the source and are still unsure what an edge case does, say that plainly instead of guessing.

## Mode B — Plain-language request → a fully-specified Pipeline call

The user describes a modeling need in prose ("I want to screen these 3,380 candidate features against PSI/IV/corr with WOE binning" / "build a scorecard on this labeled sample" / "account for rejected applicants before scoring"). Your job is to hand back a ready-to-run call with **every relevant field set explicitly** — not a half-filled config that quietly falls back to defaults the user never confirmed.

**Step 1 — Pick the pipeline.** Use the decision table in `references/pipeline_catalog.md`. If the ask spans two pipelines (e.g. reject inference feeding a credit model), say so and chain them — that's normal, not a failure to pick one.

**Step 2 — Introspect that pipeline's Config, live**, right now, on the installed version:
```python
import dataclasses
from Modeling_Tool.Pipeline import <WhicheverPipelineConfig>
for f in dataclasses.fields(<WhicheverPipelineConfig>):
    print(f.name, "=", f.default if f.default is not dataclasses.MISSING else "(required)")
```
Do not fill in a field from a memory file's field list without re-confirming it's still there under this call.

**Step 3 — Map every stated requirement to a field.** For each thing the user said:
- A field exists and their intent implies a non-default value → **set it explicitly**.
- A field exists and the default already matches their stated intent → leave it, but note in your answer that you checked (not guessed).
- Nothing in the Config expresses it → do not invent a kwarg. Go to Step 4.

**Step 4 — Escalation ladder** (this is the "Pipeline → SMF primitive → build it" order the user asked for):
1. **Pipeline Config field.** (Step 3, above — confirmed absent before moving on.)
2. **A documented lower-level SMF primitive**, called explicitly alongside the pipeline — `Core`, `Sample`, `Eval`, `Feature` (including `Feature_Screen.feature_screen` / `Weighted_Screen.weighted_feature_screen` for ad hoc screening outside any Pipeline), `WOE`, `Model`, `Explainability`. Search the actual module tree (`grep -rn "def \|class " .../Modeling_Tool/<Module>/`) rather than assuming a function exists. If found, wire it in and comment *why* it sits outside the Pipeline call.
3. **Spec it for the user to add to SMF.** Write the gap up precisely: proposed field/function name, signature, semantics, one line of rationale, and (if you can) a sketch of the internal change. Hand it to the user; do not proceed with a workaround while waiting, unless they say to.
4. **Temporary local hack** — only if the user explicitly says "don't wait for SMF, just make it work now." Label it unmistakably as temporary (comment + a distinct output path), isolated so deleting it later is a one-line diff once SMF catches up.

Never skip straight to 3 or 4 without exhausting 1–2, and never silently do 4 without saying so — that violates the SMF-only directive.

**Step 5 — Verify before handing over.** Re-check that every kwarg in your generated call actually exists on the live Config (Step 2's introspection again, now against your draft). For anything nontrivial — a new field, a first-time use of a primitive, a chained pipeline — write and run a small synthetic smoke test before pointing the call at real data. See `references/introspection_snippets.md` for the smoke-test pattern this project used every single time a new SMF version landed (backup old install → upgrade → diff source trees → read the diff → synthetic smoke test with assertions → real-data regression check against the previous run's numbers → only then adopt).

**Step 6 — Hand back the call fully spelled out.** Every field explicit, grouped logically, no bare `**kwargs`, so the user can read their own intent straight off the config without cross-referencing anything.

---

## Reference files

| File | Load when |
|---|---|
| `references/pipeline_catalog.md` | Deciding which Pipeline fits a request; field-level notes on the ones with historically tricky config surfaces |
| `references/known_gotchas.md` | A symbol you're about to explain or use has history — check before repeating or re-discovering it |
| `references/introspection_snippets.md` | You need the live-check one-liners, the version-diff-and-smoke-test loop, or a worked "escalation ladder" example |

## Repo sync (maintainer convention)

SMF ships as **four active repos** — keep this skill in sync whenever the main package changes:

1. `SuperModelingFactory` (main package)
2. `SuperModelingFactory_pytest` (tests)
3. `SuperModelingFactory_doc` (MkDocs)
4. **`SuperModelingFactory_Agent`** (this repo — `SKILL.md` + `references/`)

After every main-package merge: **grep this repo** for symbols touched in the diff (`FeatureScreenConfig`, pipeline Config fields, screening stages, WOE paths, etc.). Update `pipeline_catalog.md` for new Config fields; update `known_gotchas.md` for fixed/introduced bugs (with version tags). Canonical checklist lives in the user's `smf-work/SMF_ACTIVE_REPOS_CONVENTION.md`.
