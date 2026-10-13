# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run HeavyDB in an enroot container (Slurm path).

The .sqsh image is the product Docker image converted by the user
(e.g. `enroot import dockerd://<image>`).  The container is created per run,
started in the background with the dataset bind-mounted at /data, and heavysql
is run in it with `enroot exec`.
"""

import os
import shutil
import subprocess
import time

from .base import DATA_MOUNT, Runtime, server_shell_script
from .docker import CONTAINER_BIN, CONTAINER_STORAGE


class EnrootRuntime(Runtime):
    name = "enroot"

    def __init__(self, image=None, enroot_args=None, **kw):
        super().__init__(**kw)
        image = image or os.environ.get("HEAVYDB_ENROOT_IMAGE")
        if not image:
            raise SystemExit("--image <file.sqsh> (or $HEAVYDB_ENROOT_IMAGE) is required for --runtime enroot")
        self.image = image
        self.enroot_args = list(enroot_args or [])
        slurm_job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        self.container = "hdb-bench-%s-%d" % (slurm_job, os.getpid())
        self.storage = self.work_dir / "storage"
        # Mounted read-write by default: a ":ro" bind mount fails with "No such device"
        # on some cluster filesystems under enroot. The harness never writes to the data.
        # Set HEAVYDB_ENROOT_DATA_RO=1 to mount it read-only.
        self.data_mount_opts = ":ro" if os.environ.get("HEAVYDB_ENROOT_DATA_RO") else ""
        self.proc = None
        self.log_path = self.work_dir / "heavydb.log"

    def server_path(self, host_path):
        rel = os.path.relpath(os.path.realpath(host_path), self.data_root)
        return "%s/%s" % (DATA_MOUNT, rel)

    def start_server(self):
        if not self.dry_run:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(self.storage, ignore_errors=True)
            self.storage.mkdir(parents=True)
        self.run(["enroot", "create", "--name", self.container, self.image], check=True)
        start = (["enroot", "start", "--rw",
                  "-m", "%s:%s%s" % (self.data_root, DATA_MOUNT, self.data_mount_opts),
                  "-m", "%s:%s" % (self.storage, CONTAINER_STORAGE)]
                 + self.enroot_args
                 + [self.container, "bash", "-c",
                    server_shell_script(CONTAINER_BIN, CONTAINER_STORAGE,
                                        self.heavydb_args(CONTAINER_STORAGE, DATA_MOUNT))])
        if self.dry_run:
            self.run(start)
            return
        log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(start, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def check_alive(self):
        if self.proc and self.proc.poll() is not None:
            raise RuntimeError("enroot container exited early (rc=%s):\n%s"
                               % (self.proc.returncode, self.server_log_tail()))

    def server_log_tail(self, n=40):
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def _pid(self):
        out = subprocess.run(["enroot", "list", "-f"], capture_output=True, text=True).stdout
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == self.container and parts[1].isdigit():
                return parts[1]
        raise RuntimeError("enroot container %s is not running" % self.container)

    def heavysql_prefix(self):
        if self.dry_run:
            return ["enroot", "exec", "<pid>", CONTAINER_BIN + "/heavysql"]
        return ["enroot", "exec", self._pid(), CONTAINER_BIN + "/heavysql"]

    def stop_server(self):
        if self.proc and self.proc.poll() is None:
            import signal
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        self.run(["enroot", "remove", "-f", self.container])
        if not self.dry_run:
            shutil.rmtree(self.storage, ignore_errors=True)

    def describe(self):
        d = super().describe()
        d["image"] = str(self.image)
        return d
