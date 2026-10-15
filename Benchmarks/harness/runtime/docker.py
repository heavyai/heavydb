# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run HeavyDB inside the product Docker image (default runtime).

The image is the one produced by `dev-tools/dev.sh build --docker` /
docker/Dockerfile.{cpu,cuda}: binaries live in /opt/heavyai/bin.
"""

import os
import shutil
import subprocess
import time

from .base import DATA_MOUNT, Runtime, server_shell_script

CONTAINER_BIN = "/opt/heavyai/bin"
CONTAINER_STORAGE = "/var/lib/heavyai/storage"


def find_default_image():
    env = os.environ.get("HEAVYDB_IMAGE")
    if env:
        return env
    try:
        out = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                             capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit("cannot list docker images (%s); pass --image" % e)
    for ref in out:  # newest first
        if ref.startswith("heavyai-") and not ref.endswith(":<none>"):
            return ref
    raise SystemExit("no local heavyai-* image found. Build one with "
                     "'dev-tools/dev.sh build --docker' or pass --image / set $HEAVYDB_IMAGE")


class DockerRuntime(Runtime):
    name = "docker"

    def __init__(self, image=None, docker_args=None, **kw):
        super().__init__(**kw)
        self.image = image or (find_default_image() if not self.dry_run else "<heavydb-image>")
        self.docker_args = list(docker_args or [])
        self.container = "hdb-bench-%d" % os.getpid()
        self.storage = self.work_dir / "storage"
        self.started = False

    def server_path(self, host_path):
        rel = os.path.relpath(os.path.realpath(host_path), self.data_root)
        return "%s/%s" % (DATA_MOUNT, rel)

    def _run_cmd(self):
        cmd = ["docker", "run", "-d", "--rm", "--name", self.container,
               "--user", "%d:%d" % (os.getuid(), os.getgid()), "-e", "HOME=/tmp",
               "-v", "%s:%s:ro" % (self.data_root, DATA_MOUNT),
               "-v", "%s:%s" % (self.storage, CONTAINER_STORAGE)]
        if not self.cpu_only:
            cmd += ["--gpus", "all"]
        cmd += self.docker_args + [self.image, "bash", "-c",
               server_shell_script(CONTAINER_BIN, CONTAINER_STORAGE,
                                   self.heavydb_args(CONTAINER_STORAGE, DATA_MOUNT))]
        return cmd

    def start_server(self):
        if not self.dry_run:
            shutil.rmtree(self.storage, ignore_errors=True)
            self.storage.mkdir(parents=True)
        self.run(self._run_cmd(), check=True)
        self.started = True

    def check_alive(self):
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", self.container],
                           capture_output=True, text=True)
        if r.returncode != 0 or r.stdout.strip() != "true":
            raise RuntimeError("container exited early:\n%s" % self.server_log_tail())

    def server_log_tail(self, n=40):
        r = subprocess.run(["docker", "logs", "--tail", str(n), self.container],
                           capture_output=True, text=True)
        return r.stdout + r.stderr

    def stop_server(self):
        if not self.started:
            return
        if not self.dry_run:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            (self.work_dir / "heavydb.log").write_text(self.server_log_tail(100000))
        self.run(["docker", "stop", "-t", "60", self.container], stdout=subprocess.DEVNULL)
        if not self.dry_run:
            shutil.rmtree(self.storage, ignore_errors=True)

    def heavysql_prefix(self):
        return ["docker", "exec", "-i", self.container, CONTAINER_BIN + "/heavysql"]

    def describe(self):
        d = super().describe()
        d["image"] = self.image
        if not self.dry_run:
            r = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", self.image],
                               capture_output=True, text=True)
            d["image_id"] = r.stdout.strip()
        return d
