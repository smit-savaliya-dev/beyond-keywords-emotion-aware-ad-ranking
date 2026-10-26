"""Make-style step runner: a step reruns only when its inputs, code, arguments or outputs changed.

State lives in a stamp folder (one JSON per step + a hash cache keyed by size and mtime), so unchanged
files are never re-hashed and a step that succeeded on the same inputs is skipped.
"""
import ast
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


@dataclass
class Step:
    name: str
    cmd: list                                     # arguments after the Python executable
    inputs: list = field(default_factory=list)    # data files the step reads
    outputs: list = field(default_factory=list)   # files the step writes
    script: Optional[Path] = None                 # its src/ imports become code dependencies
    extra_code: list = field(default_factory=list)
    create_only: bool = False                     # never rerun once all outputs exist
    condition: Optional[Callable[[], Optional[str]]] = None  # returns a reason to wait, or None
    adoptable: bool = True
    note: str = ""


class StepFailed(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def code_deps(script, root):
    """The script plus every src/*.py module it imports, recursively (absolute `src.x` and relative imports)."""
    root = Path(root)
    src = root / "src"
    seen, todo = [], [Path(script)]
    while todo:
        path = todo.pop()
        if path in seen or not path.exists():
            continue
        seen.append(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        in_src = path.parent == src
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names if a.name.startswith("src.")]
            elif isinstance(node, ast.ImportFrom):
                if node.level and in_src:
                    base = node.module
                    names = [f"src.{base}"] if base else [f"src.{a.name}" for a in node.names]
                elif node.module == "src":
                    names = [f"src.{a.name}" for a in node.names]
                elif node.module and node.module.startswith("src."):
                    names = [node.module]
            for name in names:
                todo.append(src / f"{name.split('.')[1]}.py")
    return sorted(seen)


class HashCache:
    def __init__(self, path):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def sha(self, path):
        path = Path(path)
        if not path.is_file():
            return None
        st = path.stat()
        key = str(path.resolve())
        hit = self.data.get(key)
        if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
            return hit[2]
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while block := f.read(1 << 20):
                h.update(block)
        self.data[key] = [st.st_size, st.st_mtime_ns, h.hexdigest()]
        return self.data[key][2]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data), encoding="utf-8")
        tmp.replace(self.path)


class Pipeline:
    def __init__(self, steps, root, state_dir, log=print):
        self.steps, self.root, self.state = steps, Path(root), Path(state_dir)
        self.cache = HashCache(self.state / "hash_cache.json")
        self.log = log

    def _rel(self, p):
        p = Path(p)
        try:
            return p.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(p)

    def _deps(self, step):
        code = code_deps(step.script, self.root) if step.script else []
        return [Path(p) for p in [*step.inputs, *code, *step.extra_code]]

    def _signature(self, step):
        return {"cmd": [str(c) for c in step.cmd],
                "inputs": {self._rel(p): self.cache.sha(p) for p in self._deps(step)}}

    def _stamp_path(self, step):
        return self.state / f"{step.name}.json"

    def _read_stamp(self, step):
        try:
            return json.loads(self._stamp_path(step).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_stamp(self, step, signature, seconds, how):
        stamp = {"step": step.name, "signature": signature,
                 "outputs": {self._rel(p): self.cache.sha(p) for p in step.outputs},
                 "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "seconds": round(seconds, 1), "how": how}
        self.state.mkdir(parents=True, exist_ok=True)
        tmp = self._stamp_path(step).with_suffix(".tmp")
        tmp.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
        tmp.replace(self._stamp_path(step))

    def status(self, step, forced=False, planned=frozenset()):
        """-> (action, reason); action is 'run', 'skip' or 'wait'."""
        if step.condition is not None:
            why = step.condition()
            if why:
                return "wait", why
        missing_out = [self._rel(p) for p in step.outputs if not Path(p).exists()]
        if step.create_only:
            return ("run", f"missing {missing_out[0]}") if missing_out else ("skip", "exists (never regenerated)")
        if forced:
            return "run", "forced"
        upstream = sorted({self._rel(p) for p in self._deps(step)} & planned)
        if upstream:
            return "run", f"upstream will change {upstream[0]}"
        missing_in = [self._rel(p) for p in step.inputs if not Path(p).exists()]
        if missing_in:
            return "run", f"input missing: {missing_in[0]}"
        stamp = self._read_stamp(step)
        if stamp is None:
            return "run", "no record of a previous successful run"
        sig = self._signature(step)
        if stamp["signature"]["cmd"] != sig["cmd"]:
            return "run", "arguments changed"
        old = stamp["signature"]["inputs"]
        changed = sorted(k for k in set(old) | set(sig["inputs"]) if old.get(k) != sig["inputs"].get(k))
        if changed:
            more = f" (+{len(changed) - 1} more)" if len(changed) > 1 else ""
            return "run", f"changed: {changed[0]}{more}"
        if missing_out:
            return "run", f"missing {missing_out[0]}"
        modified = [k for k, v in stamp["outputs"].items() if self.cache.sha(self.root / k) != v]
        if modified:
            return "run", f"output modified: {modified[0]}"
        return "skip", "up to date"

    def run(self, force=(), dry_run=False, adopt=False):
        """Run stale steps in order; stop at the first failure. -> list of (name, action, reason, seconds)."""
        report, planned, n = [], set(), len(self.steps)
        try:
            for i, step in enumerate(self.steps, 1):
                action, reason = self.status(step, step.name in force, frozenset(planned))
                if adopt and step.adoptable and not step.create_only and action == "run":
                    if all(Path(p).exists() for p in [*step.inputs, *step.outputs]):
                        self._write_stamp(step, self._signature(step), 0.0, "adopted existing outputs")
                        action, reason = "skip", "adopted existing outputs"
                self.log(f"[{i}/{n}] {step.name}: {action.upper()} ({reason}){' - ' + step.note if step.note else ''}")
                if action != "run":
                    report.append((step.name, action, reason, 0.0))
                    continue
                if dry_run:
                    planned |= {self._rel(p) for p in step.outputs}
                    report.append((step.name, "would run", reason, 0.0))
                    continue
                signature = self._signature(step)
                start = time.time()
                code = subprocess.run([sys.executable, *map(str, step.cmd)], cwd=self.root).returncode
                seconds = time.time() - start
                if code != 0:
                    report.append((step.name, "FAILED", f"exit code {code}", seconds))
                    raise StepFailed(f"step {i}/{n} {step.name} failed (exit code {code})", report)
                missing = [self._rel(p) for p in step.outputs if not Path(p).exists()]
                if missing:
                    report.append((step.name, "FAILED", f"did not write {missing[0]}", seconds))
                    raise StepFailed(f"step {step.name} finished but did not write {missing[0]}", report)
                self._write_stamp(step, signature, seconds, "ran")
                report.append((step.name, "ran", reason, seconds))
        finally:
            self.cache.save()
        return report
