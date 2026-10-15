# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run heavydb/heavysql directly from a build tree (developer escape hatch)."""

import os
import signal
import subprocess
import shutil
from pathlib import Path

from .base import Runtime


class LocalRuntime(Runtime):
    name = "local"

    def __init__(self, build_dir=None, **kw):
        super().__init__(**kw)
        build_dir = build_dir or os.environ.get("HEAVYDB_BUILD_DIR")
        if not build_dir:
            raise SystemExit("--build-dir (or $HEAVYDB_BUILD_DIR) is required for --runtime local")
        self.bin_dir = Path(build_dir).resolve() / "bin"
        self.storage = self.work_dir / "storage"
        self.log_path = self.work_dir / "heavydb.log"
        self.proc = None
        # Calcite's sqlite-jdbc extracts a native lib into java.io.tmpdir, which may
        # be read-only/shared on some hosts: keep it inside the work dir.
        self.env = dict(os.environ, JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=%s"
                        % (self.work_dir / "tmp"))

    def start_server(self):
        import_root = self.data_root.as_posix() if self.data_root else "/"
        if self.dry_run:
            print("DRY-RUN: initheavy -f %s; heavydb %s" % (
                self.storage, " ".join(self.heavydb_args(str(self.storage), import_root))))
            return
        self.work_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.storage, ignore_errors=True)
        self.storage.mkdir(parents=True)
        (self.work_dir / "tmp").mkdir(exist_ok=True)
        subprocess.run([str(self.bin_dir / "initheavy"), "-f", str(self.storage)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env=self.env)
        log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            [str(self.bin_dir / "heavydb")] + self.heavydb_args(str(self.storage), import_root),
            cwd=self.bin_dir, env=self.env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def check_alive(self):
        if self.proc and self.proc.poll() is not None:
            raise RuntimeError("heavydb exited early (rc=%s):\n%s"
                               % (self.proc.returncode, self.server_log_tail()))

    def server_log_tail(self, n=40):
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def stop_server(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        shutil.rmtree(self.storage, ignore_errors=True)

    def heavysql_prefix(self):
        return [str(self.bin_dir / "heavysql")]

    def describe(self):
        d = super().describe()
        d["build_dir"] = str(self.bin_dir.parent)
        return d
