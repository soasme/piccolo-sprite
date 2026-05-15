# piccolo-sprite

An autonomous LangChain agent that generates game-ready fixed-cell pixel-art character
animation atlases using gpt-image-2.

---

## What It Does

Given a text prompt, piccolo-sprite:

1. Generates pixel-art animation strips with gpt-image-2 (one strip per direction per cell size)
2. Assembles strips into fixed-cell atlases (32×32, 64×64, 128×128)
3. Removes chroma key, quantizes palette
4. Validates geometry, frame integrity, and motion quality
5. Exports GIF/WebP previews
6. Verifies run provenance via manifest

## Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set your OpenAI API key
export OPENAI_API_KEY=sk-...
# or add it to a .env file
```

## Usage

```bash
# Interactive REPL (multi-turn)
uv run piccolo-sprite

# One-shot mode
uv run piccolo-sprite -p "make a 64x64 knight with 8-direction walk and attack animations"
```

## Output Layout

```
run/
  run-manifest.json
  source/<cell>-<action>-<direction>.png
  <cell>/
    generated/<action>-<direction>.png
    frames/<action>-<direction>/<index>.png
    final/<action>-sheet-clean.png
    qa/<action>-validation.json
    qa/<action>-contact-sheet.png
    qa/previews/*-transparent-x4.gif
    qa/previews/*-transparent-x4.webp
```

## Supported Sizes

`32×32`, `64×64`, `128×128` — each generated natively at its own pixel budget.

## Development

```bash
make test          # run test suite
uv run pytest -v   # verbose
```

## License

MIT
