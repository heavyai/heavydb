# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: a single Java concurrency test class against a TSAN server.
# Source of truth for the concurrency.yml test matrix. The class to run is
# supplied via the CLASS env var (and optional ARGS); the workflow sets these
# per matrix entry. Locally: CLASS=CatalogConcurrencyTest dev-tools/dev.sh test concurrency

export PARENT_CONFIG=concurrency-tsan
export RUNNER=server

export DATA=chaos
export SERVER_FLAGS="--verbose --allowed-import-paths='[\"..\"]' --enable-seconds-refresh-interval"
export SERVER_WAIT=45
export TSAN_OPTIONS="suppressions=/workspace/config/tsan.suppressions:second_deadlock_stack=true:detect_deadlocks=true:history_size=7:halt_on_error=0"
export CLIENT_CMD='java -Dfile.encoding=UTF-8 -cp ../java/utility/target/utility-1.0-SNAPSHOT-jar-with-dependencies.jar "com.mapd.tests.${CLASS:?set CLASS to a com.mapd.tests class}" ${ARGS:-}'
