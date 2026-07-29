"""Persistent Python session for Arm B's generated code.

STATEFUL, BECAUSE CODEACT IS STATEFUL
-------------------------------------
An earlier version ran every snippet in a fresh subprocess. That broke Arm B for a reason
that had nothing to do with the model: it wrote a helper in one turn and called it in the
next, and got `NameError`. CodeAct's whole design is a persistent interpreter session, so
a stateless sandbox penalises the arm for a property the paper does not have, and any
comparison built on it would be measuring our harness.

Here a single worker process stays alive for the whole run and executes each snippet in
one shared namespace. Variables, imports and functions persist across turns exactly as
they would in a notebook.

SCOPE OF THE ISOLATION, STATED HONESTLY
---------------------------------------
A separate process, a wall-clock timeout per snippet, no network (sockets are stubbed in
the preamble), and a scratch working directory. This is NOT a security boundary. It is
enough for code an LLM wrote to answer a graph question and it stops a runaway sweep from
hanging the harness. Docker would be a real boundary and was cut as not worth a day.

A snippet that exceeds the timeout kills the worker, so the session's accumulated state is
lost. That is reported rather than hidden: `session_lost` on the result, and the next
snippet starts from a fresh namespace.

The harness does NOT police what the generated code does to `G`. Modifying it in one
snippet and analysing it in the next is a normal notebook workflow, and repairing `G`
between snippets would silently break it. Mutation is recorded (`grid_mutated`) purely so
a trace can be read later.
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

#: Generous on purpose. The sweep questions need a cascade per bus or per line, and our
#: own reference tools take 36s and 71s for those on case118. Arm B writes the same loops
#: from scratch and Arm A's tools run with no timeout at all, so a tight limit here would
#: mean Arm B losing on execution speed rather than on correctness. Wall time is logged
#: separately for both arms, so efficiency is still reported rather than enforced.
DEFAULT_TIMEOUT_S = 300

#: Runs once when the session starts. Loads the grid and disables the network.
PREAMBLE = '''
import socket as _socket
def _no_network(*a, **k):
    raise RuntimeError("network access is disabled in this sandbox")
_socket.socket = _no_network
_socket.create_connection = _no_network

import networkx as nx
G = nx.read_edgelist("grid.edgelist", nodetype=int)
'''

#: The worker. Reads one JSON message per line, execs it in a persistent namespace, and
#: writes one JSON result per line. Its own stdout is redirected so that only protocol
#: messages reach the parent.
_WORKER = r'''
import sys, json, io, contextlib, traceback

ns = {"__name__": "__snippet__"}
_real_stdout = sys.stdout
try:
    exec(PREAMBLE_SRC, ns)
    _baseline = (ns["G"].number_of_nodes(), ns["G"].number_of_edges())
    _real_stdout.write(json.dumps({"ready": True}) + "\n")
except BaseException:
    _real_stdout.write(json.dumps({"ready": False, "error": traceback.format_exc()}) + "\n")
    _real_stdout.flush()
    sys.exit(1)
_real_stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    code = json.loads(line)["code"]
    out, err = io.StringIO(), io.StringIO()
    ok = True
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(code, ns)
    except BaseException:
        ok = False
        err.write(traceback.format_exc())
    # Record whether G changed, but leave it alone. Building an upgraded grid in one
    # snippet and cascading over it in the next is a legitimate workflow, and restoring G
    # between snippets would silently break it. This is a notebook: what you change stays
    # changed. The flag is for tracing only.
    mutated = False
    try:
        g = ns.get("G")
        mutated = g is None or (g.number_of_nodes(), g.number_of_edges()) != _baseline
    except BaseException:
        mutated = True
    _real_stdout.write(json.dumps(
        {"stdout": out.getvalue(), "stderr": err.getvalue(), "ok": ok,
         "grid_mutated": mutated}) + "\n")
    _real_stdout.flush()
'''


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int | None  # 0 ran, 1 raised, None killed
    timed_out: bool
    duration_s: float
    session_lost: bool = False  # namespace was discarded (timeout or worker death)
    grid_mutated: bool = False  # snippet changed G; recorded, never undone


def write_grid(G: nx.Graph, workdir: Path) -> Path:
    """Write the topology the generated code will load."""
    path = Path(workdir) / "grid.edgelist"
    nx.write_edgelist(G, path, data=False)
    return path


class Session:
    """A persistent interpreter the generated snippets share."""

    def __init__(self, G: nx.Graph, timeout_s: int = DEFAULT_TIMEOUT_S):
        self._G = G
        self.timeout_s = timeout_s
        self._tmp = tempfile.TemporaryDirectory(prefix="gridagent-sandbox-")
        self.workdir = Path(self._tmp.name)
        write_grid(G, self.workdir)
        self._driver = self.workdir / "_worker.py"
        self._driver.write_text(f"PREAMBLE_SRC = {PREAMBLE!r}\n" + _WORKER)
        self._proc: subprocess.Popen | None = None
        self._start()

    # ---------- lifecycle ----------

    def _start(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, "-u", str(self._driver)],
            cwd=self.workdir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
            start_new_session=True,  # own process group, so a timeout kills children too
        )
        line = self._readline(timeout_s=60)
        if line is None or not json.loads(line).get("ready"):
            self._kill()
            raise RuntimeError("sandbox worker failed to start")

    def _kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self._proc.kill()
        self._proc = None

    def close(self) -> None:
        self._kill()
        self._tmp.cleanup()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- protocol ----------

    def _readline(self, timeout_s: float) -> str | None:
        """One protocol line, or None on timeout or worker death."""
        assert self._proc and self._proc.stdout
        deadline = time.time() + timeout_s
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self._proc.stdout], [], [], min(remaining, 0.5))
            if ready:
                line = self._proc.stdout.readline()
                return line if line else None  # empty means EOF: the worker died
            if self._proc.poll() is not None:
                return None

    def run(self, code: str) -> SandboxResult:
        """Execute one snippet in the shared namespace."""
        t0 = time.time()
        if self._proc is None or self._proc.poll() is not None:
            # The previous snippet killed the session. Restart with a clean namespace and
            # say so, rather than silently pretending state survived.
            try:
                self._start()
            except RuntimeError as exc:
                return SandboxResult("", f"sandbox unavailable: {exc}", None, False,
                                     time.time() - t0, session_lost=True)

        try:
            self._proc.stdin.write(json.dumps({"code": code}) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            self._kill()
            return SandboxResult("", "sandbox worker died before the snippet ran", None,
                                 False, time.time() - t0, session_lost=True)

        line = self._readline(self.timeout_s)
        dur = time.time() - t0
        if line is None:
            died = self._proc is None or self._proc.poll() is not None
            self._kill()
            if died:
                return SandboxResult("", "the snippet terminated the interpreter "
                                         "(the session was restarted)", None, False,
                                     dur, session_lost=True)
            return SandboxResult("", f"TIMEOUT after {self.timeout_s}s; the session was "
                                     f"restarted and earlier variables are gone",
                                 None, True, dur, session_lost=True)

        msg = json.loads(line)
        return SandboxResult(msg["stdout"], msg["stderr"], 0 if msg["ok"] else 1,
                             False, dur, grid_mutated=msg.get("grid_mutated", False))


def run_code(code: str, G: nx.Graph, timeout_s: int = DEFAULT_TIMEOUT_S) -> SandboxResult:
    """One-shot execution in a throwaway session. Used by tests, not by Arm B."""
    with Session(G, timeout_s=timeout_s) as s:
        return s.run(code)
