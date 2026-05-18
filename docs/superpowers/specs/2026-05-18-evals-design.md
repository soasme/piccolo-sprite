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
    └── traces/
        ├── girlwitch-east-walk.json
        ├── girlwitch-east-attack.json
        ├── girlknight-west-walk.json
        └── girlknight-west-attack.json

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
  ]
}
```

## Components

### record.py

A one-shot script (not a pytest file) run manually to populate fixtures:

```bash
python evals/record.py --scenario girlwitch-east-walk \
  --prompt "Generate a walk animation for girlwitch facing east"
```

1. Builds the real agent via `build_agent()`
2. Wraps `ChatOpenAI` with a recording subclass that captures each `(messages_in, response_out)` pair by calling `super()` then saving the result
3. Runs the agent to completion
4. Writes `tests/fixtures/traces/<scenario>.json`

Re-run only when the system prompt or tool behavior changes and golden traces need refreshing.

### evals/conftest.py

Provides a `replay_llm(scenario)` pytest fixture that:

- Reads `tests/fixtures/traces/<scenario>.json`
- Patches `ChatOpenAI` so each invocation returns the next pre-recorded response in order
- Raises `IndexError` if the agent makes more LLM calls than were recorded

### evals/test_traces.py

For each scenario, runs the agent under the replay mock and asserts:

- `write_manifest` is the first tool call
- `generate_sprite_strip` is called exactly once per direction (no bulk calls)
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

Then commit the updated trace files alongside the system prompt change.
