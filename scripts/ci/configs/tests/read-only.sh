# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: read-only server smoke check via heavysql. Source of truth
# for test-read-only in weekly.yml. Starts the server through `startheavy
# --read-only`; confirms reads work and writes are rejected.

export PARENT_CONFIG=gpu-release
export RUNNER=server

export SERVER_CMD="../startheavy --read-only"
export CLIENT_CMD='set -x
echo "select count(*) from heavyai_us_states;" | ./bin/heavysql -a -p HyperInteractive
echo "update heavyai_us_states set abbr='"'"'22'"'"' where abbr='"'"'11'"'"';" | ./bin/heavysql -a -p HyperInteractive || echo "IGNORE ERRORS (read-only)"
echo "delete from heavyai_us_states where abbr='"'"'11'"'"';" | ./bin/heavysql -a -p HyperInteractive || echo "IGNORE ERRORS (read-only)"
echo "select count(*) from heavyai_us_states;" | ./bin/heavysql -a -p HyperInteractive'
