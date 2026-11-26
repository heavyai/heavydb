# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: set-based COUNT(DISTINCT) host-memory limits, SHOW QUERIES phase and
# KILL QUERY during allocation. Source of truth for count-distinct-memory in
# weekly.yml. Self-contained (run_count_distinct_memory.sh starts its own servers).

export PARENT_CONFIG=gpu-release
export RUNNER=count_distinct_memory
export MAKE_TARGETS="heavydb initheavy heavysql"
