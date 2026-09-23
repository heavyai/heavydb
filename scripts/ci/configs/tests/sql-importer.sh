# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: com.mapd.utility.SQLImporter against a locally-started
# server. Source of truth for test-sql-importer in nightly.yml. ADAPTED from
# the Jenkins job, which imported into a mapd-cpu product container; here we
# start the freshly-built server and import into it. Relies on the default
# geo sample table heavyai_us_states (initheavy without --skip-geo).

export PARENT_CONFIG=sql-importer
export RUNNER=server

export CLIENT_CMD='JDBC_JAR=$(ls bin/heavyai-jdbc*.jar); UTIL_JAR=$(ls bin/heavyai-util*.jar); \
java -cp "${JDBC_JAR}:${UTIL_JAR}" com.mapd.utility.SQLImporter \
  -t heavyai_us_states_cpy -u admin -p HyperInteractive \
  -db heavyai -s localhost --port 6274 \
  -su admin -sp HyperInteractive \
  -ss "SELECT id, abbr, name from heavyai_us_states" \
  -c "jdbc:heavyai:localhost:6274:heavyai" | grep "result set count is 52 read time is"'
