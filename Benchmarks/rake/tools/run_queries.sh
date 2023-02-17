#!/bin/bash

# Copyright 2023 HEAVY.AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
