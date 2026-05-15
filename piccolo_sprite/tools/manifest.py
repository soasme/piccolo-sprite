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
    Returns {ok, path}.

    Required manifest schema:
    {
      "reference": {
        "source_type": "user_request" | "chat_attachment" | "file" | "image_url",
        "source": "<description of request or file path>",
        "used_for_generation": true,
        "identity_notes": ["<note1>", ...]   // non-empty list
      },
      "scope": {
        "sizes": [64],             // list of int cell sizes
        "actions": ["walk"],       // list of action strings
        "directions": ["south"]    // list of direction strings
      },
      "generation": {
        "method": "imagegen",
        "imagegen_output_path": "<path to raw imagegen PNG>"
      },
      "strips": [
        {
          "cell": 64,
          "action": "walk",
          "direction": "south",
          "method": "imagegen",
          "source_path": "<path to assembled strip PNG>",
          "imagegen_output_path": "<path to raw imagegen PNG>"
        }
      ]
    }
    """
    path = Path(run_dir) / "run-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(path)}
