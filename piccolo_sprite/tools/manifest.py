import json
from pathlib import Path
from langchain_core.tools import tool


@tool
def read_manifest(run_dir: str) -> dict:
    """Read run-manifest.json from run_dir. Returns {ok, manifest} or {ok, error}."""
    path = Path(run_dir) / "run-manifest.json"
    if not path.exists():
        return {"ok": False, "error": f"{path} not found"}
    return {"ok": True, "manifest": json.loads(path.read_text(encoding="utf-8"))}


@tool
def write_manifest(run_dir: str, data: dict) -> dict:
    """Write data to run-manifest.json in run_dir. Creates run_dir if needed.
    Returns {ok, path}."""
    path = Path(run_dir) / "run-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(path)}
