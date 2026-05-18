# Evals Design

**Date:** 2026-05-18
**Status:** Approved

## Goal

Add an `evals/` directory that catches three failure modes:

1. **Pipeline adherence** — the agent calls tools in the right order with the right args
2. **Prompt parsing** — the agent correctly extracts cell sizes, actions, and directions from user prompts
3. **Output quality** — `validate_sheet` and `audit_motion` pass against real sprite sheet outputs

## Approach

Mock/replay using pytest. Record golden LLM call/response traces from real agent runs once; replay them deterministically in CI with no network calls and no cost.

## Scenarios

Four golden scenarios, two characters × two actions:

| Scenario | Character | Direction | Action |
|---|---|---|---|
| `girlwitch-east-walk` | girlwitch | east | walk |
| `girlwitch-east-attack` | girlwitch | east | attack |
| `girlknight-west-walk` | girlknight | west | walk |
| `girlknight-west-attack` | girlknight | west | attack |

## Directory Layout

```
tests/
└── fixtures/
    ├── traces/
    │   ├── girlwitch-east-walk.json
    │   ├── girlwitch-east-attack.json
    │   ├── girlknight-west-walk.json
    │   └── girlknight-west-attack.json
    └── evals/
        ├── girlwitch-east-walk/run/    # copied run/ artifacts from recording
        ├── girlwitch-east-attack/run/
        ├── girlknight-west-walk/run/
        └── girlknight-west-attack/run/

evals/
├── conftest.py      # ChatOpenAI replay mock; loads trace fixtures
├── record.py        # run once to record traces from real agent runs
├── test_traces.py   # pipeline adherence + prompt parsing assertions
└── test_tools.py    # direct tool quality assertions using recorded outputs
```

## Trace Format

Each `tests/fixtures/traces/<scenario>.json`:

```json
{
  "scenario": "girlwitch-east-walk",
  "prompt": "Generate a walk animation for girlwitch facing east",
  "turns": [
    {
      "messages": [...],
      "response": { "tool_calls": [...], "content": "..." }
    }
  ],
  "tool_runs": [
    {
      "name": "write_manifest",
      "args": { ... },
      "result": { "ok": true }
    },
    {
      "name": "generate_sprite_strip",
      "args": { "cell": 64, "action": "walk", "direction": "east", ... },
      "result": { "ok": true, "path": "run/source/64-walk-east.png" }
    }
  ]
}
```

`turns` drives LLM replay in `test_traces.py`. `tool_runs` drives direct tool assertions in `test_tools.py` — it records every tool call's args and result so quality checks can be replayed without re-running the LLM.

## Components

### record.py

A one-shot script (not a pytest file) run manually to populate fixtures:

```bash
python evals/record.py --scenario girlwitch-east-walk \
  --prompt "Generate a walk animation for girlwitch facing east"
```

1. Builds the real agent with a recording `ChatOpenAI` subclass (calls `super()`, saves each `(messages_in, response_out)` pair) and recording tool wrappers (call the real tool, save args + result to `tool_runs`)
2. Runs the agent to completion
3. Writes `tests/fixtures/traces/<scenario>.json` (LLM turns + tool_runs)
4. Copies the full `run/` directory to `tests/fixtures/evals/<scenario>/run/` (the actual generated images and artifacts)

Re-run only when the system prompt or tool behavior changes and golden traces need refreshing.

### evals/conftest.py

Provides a `replay_llm(scenario)` pytest fixture that:

- Reads `tests/fixtures/traces/<scenario>.json`
- Patches `ChatOpenAI` so each invocation returns the next pre-recorded response in order
- Raises `IndexError` if the agent makes more LLM calls than were recorded

### evals/test_traces.py

For each scenario, runs the agent under the replay mock and asserts:

- `write_manifest` is the first tool call
- `generate_sprite_strip` call count equals `len(scope.directions)` from the trace (no bulk multi-direction calls)
- `clean_sheet` always precedes `validate_sheet` in the call sequence
- `validate_manifest` is the last tool call
- Tool args match the recorded values (correct `cell`, `action`, `direction`)

### evals/test_tools.py

Skips the LLM entirely; calls tool functions directly using actual tool outputs captured during recording. Asserts:

- `validate_sheet` returns `ok=true` for the recorded sprite sheets
- `audit_motion` returns no chroma residue or near-duplicate frame warnings

## Refresh Workflow

When the system prompt or pipeline changes intentionally:

```bash
python evals/record.py --scenario girlwitch-east-walk --prompt "..."
python evals/record.py --scenario girlwitch-east-attack --prompt "..."
python evals/record.py --scenario girlknight-west-walk --prompt "..."
python evals/record.py --scenario girlknight-west-attack --prompt "..."
```

Then commit both the updated trace JSON files (`tests/fixtures/traces/`) and the refreshed artifact directories (`tests/fixtures/evals/`) alongside the system prompt change.
