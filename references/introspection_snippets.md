# Introspection snippets

Copy-paste starting points for "confirm, don't recall." Every one of these was actually run, repeatedly, over the course of this project.

## 1. Live version

```python
import Modeling_Tool as smf
print(smf.__version__)
```

## 2. List a Config dataclass's fields, defaults, and types

```python
import dataclasses
from Modeling_Tool.Pipeline import CreditModelPipelineConfig  # swap for whichever Config

for f in dataclasses.fields(CreditModelPipelineConfig):
    default = f.default if f.default is not dataclasses.MISSING else (
        f.default_factory() if f.default_factory is not dataclasses.MISSING else "(required)"
    )
    print(f"{f.name:35s} {str(f.type):35s} default={default!r}")
```

Same pattern for a `*Result` dataclass to see what a run actually gives you back:
```python
import dataclasses
from Modeling_Tool.Pipeline import CreditModelPipelineResult
print([f.name for f in dataclasses.fields(CreditModelPipelineResult)])
```

## 3. Signature + docstring of a function/method

```python
import inspect
from Modeling_Tool.Feature.Weighted_Screen import weighted_feature_screen
print(inspect.signature(weighted_feature_screen))
print(inspect.getdoc(weighted_feature_screen))
```

## 4. When signature+docstring aren't enough, read the real source

```python
import inspect
from Modeling_Tool.Pipeline import credit_model
print(inspect.getsource(credit_model.CreditModelPipeline._run_explainability))
```
This is how every "does X actually gate on Y" question in this project got answered for real (e.g. confirming Owen skips `xgb`, confirming `explain_outputs` isn't persisted, confirming `woe_fit_query` only touches the fit-`ins` slice and not PSI/IV/transform).

## 5. A Config field doesn't exist — search the module tree before concluding it's a hand-roll situation

```bash
python3 -c "import Modeling_Tool, os; print(os.path.dirname(Modeling_Tool.__file__))"
```
```bash
grep -rn "def \|class " /path/to/Modeling_Tool/Feature/ /path/to/Modeling_Tool/Sample/ /path/to/Modeling_Tool/WOE/ /path/to/Modeling_Tool/Model/ /path/to/Modeling_Tool/Explainability/ /path/to/Modeling_Tool/Core/
```
This is step 2 of the escalation ladder (SKILL.md) — grep the actual primitive modules for something that already does what the missing Config field would have done, rather than assuming nothing exists.

## 6. New version dropped — the upgrade loop

This exact sequence ran on every single version bump this project went through (0.3.4→0.3.5→0.3.6→0.3.7→0.3.13→0.3.14):

```bash
# 1. snapshot the current install for diffing
SITE=$(python3 -c "import Modeling_Tool, os; print(os.path.dirname(Modeling_Tool.__file__))")
cp -r "$SITE" /tmp/smf_old

# 2. upgrade
pip install --upgrade supermodelingfactory==<new_version>

# 3. scope exactly what changed
diff -rq /tmp/smf_old "$SITE" | grep -v __pycache__

# 4. read the diff of every changed file — don't infer from a changelog sentence
diff -u /tmp/smf_old/Pipeline/credit_model.py "$SITE/Pipeline/credit_model.py"
```
Then:
5. **Write a small synthetic smoke test** exercising the new/changed path with real assertions (pattern below) — never the first real usage on production data.
6. **Re-run one previous real notebook/config** and diff its numeric output against the pre-upgrade run — proves the parts you *didn't* mean to change didn't change. (This caught, for example, that 0.3.13's screening refactor left a real dataset's final selected-feature count identical to 0.3.6's.)
7. Only then adopt into real notebooks.

## 7. Smoke-test pattern (synthetic data, fast, asserts on behavior not just "did it run")

```python
import numpy as np, pandas as pd
from Modeling_Tool.Pipeline import CreditModelPipeline, CreditModelPipelineConfig

rng = np.random.default_rng(0)
n = 6000
df = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(6)})
df["y"] = ((df.x0 + 0.6*df.x1 - 0.7*df.x2 + rng.normal(scale=1.3, size=n)) > 0).astype(int)
df["split"] = np.where(np.arange(n) < 3500, "ins", np.where(np.arange(n) < 4800, "oos", "oot"))

cfg = CreditModelPipelineConfig(
    output_dir="/tmp/smoke_test_out",
    target_col="y", feature_cols=[f"x{i}" for i in range(6)], split_col="split",
    # ...the ONE new/changed field you're actually testing, everything else minimal...
    train_models=["lr", "lgb"], backward_enabled=False, optuna_models=[],
    explain_models=[], owen_enabled=False,
    write_outputs=False, write_excel=False, plot_outputs=False, random_state=42,
)
res = CreditModelPipeline(cfg).run(df)
# assert on the SPECIFIC behavior the new field is supposed to change —
# not just "it didn't crash"
assert res.model_feature_sources["lgb"] == "raw"
```

Keep `n_estimators`/similar small and pass explicit `n_jobs` in `model_params` on a shared/multi-tenant box — an unset `n_jobs=-1` on a synthetic 6-column smoke test can appear to hang for minutes by oversubscribing every core on the host, not the container's own allocation.

## 8. Load a saved model + recover what a run actually did

```python
from Modeling_Tool.Core import load_model

model, meta = load_model("output/.../models/model_lgb.pkl", return_metadata=True)
print(meta["feature_cols"], meta["feature_source"], meta.get("warm_start_enabled"))
```
This is the fastest way to answer "what did this old run actually train on" without re-reading the notebook — and the only correct way to open the file at all (never a bare `pickle.load`, see `known_gotchas.md`).

## 9. Worked example of the full escalation ladder (what actually happened, once)

**Ask:** "I want the GBM models to train on raw feature values instead of WOE-encoded ones, keep LR on WOE."

1. **Config field?** `dataclasses.fields(CreditModelPipelineConfig)` on the installed version (0.3.5) — no such field existed.
2. **Lower-level primitive?** Grepped `Model/`, `WOE/`, `Pipeline/_common.py` for anything that separates "which features feed which model family" — nothing did; the pipeline's internal `_fit_woe`/`_train_models` always assumed one shared feature representation for every model.
3. **Spec it, since the user maintains SMF.** Wrote up the exact field (`gbm_feature_source: str | dict[str, str] = "woe"`), its semantics (per-model or global override, LR always WOE regardless), and where it would need to plug in (`_train_models`, `model_feature_sources` on the result, `_evaluate_models`, `_run_explainability`'s eval-set construction). Handed to the user; did not hack a bypass into the notebook while waiting.
4. **Landed in 0.3.6** as `gbm_feature_source`. Verified via the version-diff loop (step 6 above), smoke-tested (step 7), then used directly — no hand-rolled code ever shipped for this requirement.

This is the loop `SKILL.md`'s Mode B, Step 4 describes — treat it as the concrete example when explaining the ladder to the user, not just an abstract policy.
