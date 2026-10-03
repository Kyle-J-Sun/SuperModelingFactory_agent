# SuperModelingFactory Agent (`smf-agent`)

An AI-assistant **skill** that makes an LLM agent (such as Claude Code) a reliable expert on
[SuperModelingFactory (SMF)](https://github.com/Kyle-J-Sun/SuperModelingFactory), the credit-risk scorecard toolkit. With
it installed, the agent can:

- **Explain** how an SMF class, method, parameter, or pipeline works, after checking the *installed* version rather than
  answering from memory.
- **Turn a plain-language request into a fully specified Pipeline call**, for example "screen these 3,000 candidate features
  with PSI / IV / correlation using WOE bins", with every relevant config field set explicitly and verified to exist.

SMF changes quickly, and an agent that recalls an older signature will confidently write code that no longer runs. This skill
exists to replace recall with verification.

## What is in this repository

```
SuperModelingFactory_agent/
├── SKILL.md                         # The skill: trigger description, rules, Mode A and Mode B workflows
└── references/
    ├── pipeline_catalog.md          # Which pipeline fits which request; notes on tricky config surfaces
    ├── known_gotchas.md             # Versioned history of real bugs and behavior changes
    └── introspection_snippets.md    # Copy-paste live checks and the upgrade-and-smoke-test loop
```

`SKILL.md` starts with YAML front matter (`name: smf-agent`, plus a `description` listing when the skill should activate:
questions mentioning SMF, `Modeling_Tool`, `CreditModelPipeline`, `FeatureValidationPipeline`, `WOE_Monotone_Binner`, and so on).
The three files under `references/` are loaded only when needed.

## Install

The skill needs a Python environment where SMF itself is installed, so the agent can introspect it:

```bash
pip install supermodelingfactory
python -c "import Modeling_Tool; print(Modeling_Tool.__version__)"
```

Then place this repository where your agent discovers skills. For Claude Code, the directory name must match the skill name:

```bash
# Personal (all projects)
git clone https://github.com/Kyle-J-Sun/SuperModelingFactory_agent.git ~/.claude/skills/smf-agent

# or per project
git clone https://github.com/Kyle-J-Sun/SuperModelingFactory_agent.git .claude/skills/smf-agent
```

Other agents that support `SKILL.md`-style skills can load the same folder; consult their documentation for the skills path.

## How the agent works

| Principle | What it means |
|---|---|
| **Confirm, don't recall** | Before explaining or using a symbol, the agent reads its live signature, defaults, and docstring (`inspect.signature`, `dataclasses.fields`, `inspect.getsource`) and states the installed `Modeling_Tool.__version__`. |
| **SMF-only** | It does not hand-write WOE binning, PSI/IV, correlation filtering, reject inference, splitting, training, evaluation, or explainability logic that SMF already owns. |
| **Escalation ladder** | For a gap: (1) a Pipeline config field, then (2) a documented lower-level SMF primitive, then (3) a precise spec for you to add to SMF, and only if you ask, (4) a clearly labeled temporary workaround. |
| **Gotchas are historical** | Every entry in `known_gotchas.md` carries a version tag. If the installed version is past the fix, the agent says so instead of repeating a stale warning. |

**Mode A: implementation questions.** "How does `corr_use_woe_bins` interact with the WOE engine?" The agent resolves the
exact symbol, introspects it, cross-checks known gotchas, and answers with mechanism, types, defaults, and field coupling.

**Mode B: plain-language request to a pipeline call.** "Build a scorecard from this labeled sample." The agent picks the
pipeline from the decision table, lists the live fields of its `*Config` dataclass:

```python
import dataclasses
from Modeling_Tool.Pipeline import CreditModelPipelineConfig

for f in dataclasses.fields(CreditModelPipelineConfig):
    print(f.name, "=", f.default if f.default is not dataclasses.MISSING else "(required)")
```

maps each requirement to an explicit field, checks the draft call against the live config, and hands back a fully spelled-out
call, with a synthetic smoke test for anything non-trivial.

## Try it

After installing, ask your agent things like:

- "Which SMF pipeline should I use to compare three existing score columns, and how do I configure it?"
- "What does `woe_fit_query` do in `CreditModelPipelineConfig`, and what is its default?"
- "Set up `FeatureValidationPipeline` for a 40 GB CSV of candidate features that does not fit in memory."

## Keeping the skill current (maintainers)

SMF is developed across four repositories, and this one must follow the main package:

| Repository | Role |
|---|---|
| [`SuperModelingFactory`](https://github.com/Kyle-J-Sun/SuperModelingFactory) | Package source |
| [`SuperModelingFactory_pytest`](https://github.com/Kyle-J-Sun/SuperModelingFactory_pytest) | Test suite |
| [`SuperModelingFactory_doc`](https://github.com/Kyle-J-Sun/SuperModelingFactory_doc) | MkDocs documentation |
| `SuperModelingFactory_agent` (this repository) | The skill |

After every change to the main package, grep this repository for the symbols the diff touched (config fields, screening
stages, WOE paths). Update `pipeline_catalog.md` for new config fields and `known_gotchas.md` for fixed or newly introduced
behavior, always with version tags. Recommended push order across repositories: pytest → doc → agent → main.
