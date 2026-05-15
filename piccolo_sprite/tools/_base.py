import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


def run_script(script_name: str, args: list[str]) -> dict:
    script = SCRIPTS_DIR / script_name
    result = subprocess.run(
        [sys.executable, str(script)] + args,
        capture_output=True,
        text=True,
    )
    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
