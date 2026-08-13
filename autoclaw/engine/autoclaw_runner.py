import hashlib
import subprocess
import time
from pathlib import Path


FATAL_STDERR_MARKERS = (
    "Error during processing",
    "COMMAND_NOT_FOUND",
    "TIMEOUT",
    "AUTOCLAW_MAX_ITERATIONS_EXCEEDED",
    "401",
    "quota",
    "额度已用尽",
)


class AutoClawRunner:
    def __init__(self, command="autoclaw", model="gpt-4o", timeout_sec=180):
        self.command = command
        self.model = model
        self.timeout_sec = timeout_sec

    def run_episode(self, prompt: str, workdir: str):
        cmd = [
            self.command,
            prompt,
            "--no-interactive",
            "-y",
            "-m",
            self.model,
        ]

        start = time.time()
        before = _workspace_fingerprint(workdir)
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec
            )
            latency = time.time() - start

            fatal_stderr = _has_fatal_stderr(proc.stderr)
            after = _workspace_fingerprint(workdir)
            changed_paths = _changed_paths(before, after)
            stdout_has_tool_execution = "Executing tool:" in str(proc.stdout or "")
            no_op_execution = (
                proc.returncode == 0
                and not fatal_stderr
                and not stdout_has_tool_execution
                and len(changed_paths) == 0
            )
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "latency_sec": latency,
                "success": proc.returncode == 0 and not fatal_stderr and not no_op_execution,
                "fatal_stderr": fatal_stderr,
                "workspace_changed": len(changed_paths) > 0,
                "changed_paths": changed_paths,
                "changed_path_count": len(changed_paths),
                "stdout_has_tool_execution": stdout_has_tool_execution,
                "no_op_execution": no_op_execution,
            }

        except FileNotFoundError:
            latency = time.time() - start
            return {
                "returncode": -2,
                "stdout": "",
                "stderr": f"COMMAND_NOT_FOUND: {self.command}",
                "latency_sec": latency,
                "success": False,
                "fatal_stderr": True,
                "workspace_changed": False,
                "changed_paths": [],
                "changed_path_count": 0,
                "stdout_has_tool_execution": False,
                "no_op_execution": False,
            }

        except subprocess.TimeoutExpired as e:
            return {
                "returncode": -1,
                "stdout": e.stdout or "",
                "stderr": "TIMEOUT",
                "latency_sec": self.timeout_sec,
                "success": False,
                "fatal_stderr": True,
                "workspace_changed": False,
                "changed_paths": [],
                "changed_path_count": 0,
                "stdout_has_tool_execution": False,
                "no_op_execution": False,
            }


def _has_fatal_stderr(stderr: str) -> bool:
    text = str(stderr or "")
    return any(marker in text for marker in FATAL_STDERR_MARKERS)


def _workspace_fingerprint(workdir: str) -> dict[str, str]:
    root = Path(workdir)
    fingerprint: dict[str, str] = {}
    if not root.exists():
        return fingerprint

    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            digest = "READ_ERROR"
        fingerprint[rel] = digest
    return fingerprint


def _changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    changed = []
    for rel in sorted(set(before) | set(after)):
        if before.get(rel) != after.get(rel):
            changed.append(rel)
    return changed
