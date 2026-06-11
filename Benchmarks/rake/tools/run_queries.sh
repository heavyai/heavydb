#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run all 99 queries, with exceptions below.
# Capture result of query i in results${i}.txt.
# Error are captured in output.txt.

# Run as:
# $ time ./run_queries.sh 2>&1 | tee output.txt

# These utilty scripts are intended to be customized.
# Customize these values:
DATABASE=tpcds_500gb
# Location of the 99 query1.sql, query2.sql, ...
QUERIES_DIR=/var/lib/heavyai/tpc-ds/queries
HEAVYSQL="/opt/heavyai/bin/heavysql -p HyperInteractive $DATABASE"
# Repeat each query N times. The times can be analyzed later.
REPEAT=3

for i in {1..99}
do
  # Queries take too long: 47, 57
  # Query would use too much memory: 23, 65, 78
  # Sorting the result would be too slow: 71, 84
  if [[ $i != 47 && $i != 57 && $i != 23 && $i != 65 && $i != 78 && $i != 71 && $i != 84 ]]
  then
    echo "Running query $i."
    for j in {1..$REPEAT}
    do
      # Capture results and timing of each run.
      printf '\\timing\n' | cat - $QUERIES_DIR/query$i.sql | $HEAVYSQL >> result$i.txt
    done
  fi
done
