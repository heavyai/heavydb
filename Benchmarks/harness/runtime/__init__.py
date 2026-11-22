# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .base import Runtime
from .docker import DockerRuntime
from .enroot import EnrootRuntime
from .local import LocalRuntime

RUNTIMES = {"docker": DockerRuntime, "enroot": EnrootRuntime, "local": LocalRuntime}


def make_runtime(name, **kw):
    try:
        return RUNTIMES[name](**kw)
    except KeyError:
        raise SystemExit("unknown runtime %r (choose from %s)" % (name, ", ".join(RUNTIMES)))
