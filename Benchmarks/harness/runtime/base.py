# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime abstraction: where HeavyDB runs (docker, enroot, local build tree).

A Runtime starts a HeavyDB server, exposes an argv prefix for running heavysql
against it, and translates host paths to the paths the server sees (container
bind mounts).  Adding a new runtime (podman, apptainer, ...) means subclassing
Runtime and implementing the few abstract methods.
"""

import abc
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

DATA_MOUNT = "/data"
# Grace period (s) before the server is killed on shutdown. Kept below Slurm's default KillWait
# (30 s from SIGTERM to SIGKILL) so cleanup can finish; the server's storage is discarded anyway.
STOP_TIMEOUT = 15
DEFAULT_USER = "admin"
DEFAULT_PASSWORD = "HyperInteractive"
DEFAULT_DB = "heavyai"


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def has_gpu():
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
    except OSError:
        return False


class Runtime(abc.ABC):
    """Base class.  Subclasses define how to launch and talk to the server."""

    name = "base"

    def __init__(self, data_root=None, work_dir=None, cpu_only=None, server_args=None,
                 dry_run=False):
        self.data_root = Path(data_root).resolve() if data_root else None
        self.work_dir = Path(work_dir or ".").resolve()
        self.cpu_only = (not has_gpu()) if cpu_only is None else cpu_only
        self.server_args = list(server_args or [])
        self.dry_run = dry_run
        self.port = free_port()
        self.http_port = free_port()
        self.calcite_port = free_port()
        self.user, self.password, self.db = DEFAULT_USER, DEFAULT_PASSWORD, DEFAULT_DB

    # ---- path translation -------------------------------------------------
    def server_path(self, host_path):
        """Path of a host file as seen by the server process."""
        host_path = Path(host_path).resolve()
        return host_path.as_posix()

    # ---- helpers shared by container runtimes ------------------------------
    def heavydb_args(self, storage_dir, import_root):
        args = [
            "--data", storage_dir,
            "--port", str(self.port),
            "--http-port", str(self.http_port),
            "--calcite-port", str(self.calcite_port),
            "--allowed-import-paths", '["%s"]' % import_root,
        ]
        if self.cpu_only:
            args.append("--cpu-only")
        return args + self.server_args

    def heavysql_args(self):
        return ["--port", str(self.port), "--user", self.user, "--passwd", self.password,
                "--db", self.db, "-q", "-t"]

    # ---- lifecycle -----------------------------------------------------------
    @abc.abstractmethod
    def start_server(self):
        """Start HeavyDB; return once the process/container is launched."""

    @abc.abstractmethod
    def stop_server(self):
        """Stop the server and release resources."""

    @abc.abstractmethod
    def heavysql_prefix(self):
        """argv prefix (list) that runs heavysql; heavysql_args() are appended."""

    def preserve_logs(self):
        """Copy the server's log directory (<storage>/log: heavydb.INFO etc.) to <work_dir>/logs.

        Called by stop_server() just before the storage directory is deleted.
        """
        src = Path(self.storage) / "log"
        if self.dry_run or not src.is_dir():
            return
        try:
            shutil.copytree(src, self.work_dir / "logs", dirs_exist_ok=True)
        except OSError as e:
            print("warning: could not save the server logs: %s" % e, file=sys.stderr)

    def describe(self):
        return {"runtime": self.name, "cpu_only": self.cpu_only}

    # ---- readiness -----------------------------------------------------------
    def wait_ready(self, timeout=600):
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            self.check_alive()
            try:
                # resolved per attempt: a container runtime may not know its target yet
                cmd = self.heavysql_prefix() + self.heavysql_args()
            except RuntimeError as e:
                last = str(e)
                time.sleep(2)
                continue
            r = subprocess.run(cmd, input="select 1;\n", capture_output=True, text=True)
            if r.returncode == 0 and "Exception" not in r.stdout + r.stderr:
                return
            last = (r.stdout + r.stderr).strip()
            time.sleep(2)
        raise RuntimeError("HeavyDB did not become ready in %ds: %s\n%s"
                           % (timeout, last, self.server_log_tail()))

    def check_alive(self):
        """Raise if the server process died early."""

    def server_log_tail(self, n=40):
        return ""

    def __enter__(self):
        try:
            self.start_server()
            if not self.dry_run:
                self.wait_ready()
        except BaseException:
            # __exit__ is not called when __enter__ raises: don't leak the server/container
            try:
                self.stop_server()
            except Exception as e:  # keep the original error
                print("warning: cleanup after failed start: %s" % e, file=sys.stderr)
            raise
        return self

    def __exit__(self, *exc):
        self.stop_server()

    def run(self, argv, **kw):
        if self.dry_run:
            print("DRY-RUN:", shlex.join(argv))
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.run(argv, **kw)


def server_shell_script(bin_dir, storage_dir, heavydb_args):
    """Shell snippet run inside the container to init storage and exec heavydb."""
    q = shlex.quote
    return (
        'export LD_LIBRARY_PATH="%s/../lib:${LD_LIBRARY_PATH:-}"; '
        "mkdir -p %s && %s/initheavy -f %s >/dev/null && exec %s/heavydb %s"
        % (bin_dir, q(storage_dir), bin_dir, q(storage_dir), bin_dir,
           " ".join(q(a) for a in heavydb_args))
    )
