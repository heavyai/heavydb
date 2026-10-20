# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-level metadata (<data>/metadata.json written by the data generators)."""

import json
from pathlib import Path


def read_metadata(data_root):
    """Return the dataset's metadata.json as a dict; {} if it is missing or malformed."""
    try:
        meta = json.loads((Path(data_root) / "metadata.json").read_text())
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}
