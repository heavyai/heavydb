#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -m

function shutdown_heavyai() {
  if [[ -n $HEAVYAI_PID ]]; then
    echo "Stopping HeavyAI servers ($HEAVYAI_PID)"
    kill $HEAVYAI_PID
  fi
}

trap shutdown_heavyai ERR SIGINT

if [[ "${PWD##*/}" != "build" ]]; then
  echo "This script should be executed from within the build directory."
  exit 1
fi

echo "Launching HeavyAI servers"
TEST_STORAGE_DIR=Tests/llm_tranform_test_storage
if [[ -d $TEST_STORAGE_DIR ]]; then
  rm -rf $TEST_STORAGE_DIR
fi

mkdir -p $TEST_STORAGE_DIR
./bin/initheavy --skip-geo -f $TEST_STORAGE_DIR

HEAVYDB_PORT=6274
HEAVYAI_CONFIG_PATH=$TEST_STORAGE_DIR/heavy.conf

cat << EOF > $HEAVYAI_CONFIG_PATH
data = "$TEST_STORAGE_DIR"
llm-transform-max-num-unique-value = 5

[iq]
heavydb_host = "localhost"
heavydb_port = $HEAVYDB_PORT
custom_llm_type = "API_VLLM"
custom_llm_api_base = "https://api.heavy.ai/v1"
EOF

export DISABLE_OPEN_FRONTEND=true
../startheavy --config $HEAVYAI_CONFIG_PATH > /dev/null &
HEAVYAI_PID=$!

echo "Waiting for HeavyDB to start"
while ! echo "\status" | ./bin/heavysql -p HyperInteractive 2>/dev/null | grep -q "Server Version"; do
  sleep 1
done
echo "HeavyDB started"

HEAVYIQ_PORT=6275
echo "Waiting for HeavyIQ to start"
while ! lsof -i :${HEAVYIQ_PORT} > /dev/null; do
  sleep 1
done
echo "HeavyIQ started"

cd Tests
echo "Executing LLM_TRANSFORM integration tests"
./LlmTransformIntegrationTest

shutdown_heavyai
