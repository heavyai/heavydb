#!/bin/bash

# Copyright 2022 HEAVY.AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a Jenkins post-build script for running and recording benchmarks to conbench.

# Assumptions:
# * The PWD is the root git project directory.
# * The current branch is $GIT_BRANCH
# * The build directory is build-$GIT_COMMIT
# * ~/.conbench contains the credentials to the network conbench server.

# If you need to get the last 10 commits:
# git log | egrep '^commit ' | head -10 | awk '{print $2}' | tac > commits.txt

mkdir -p build-$GIT_COMMIT/Tests/tmp

cd Benchmarks/conbench

ln -sf ~/.conbench .

conbench StringDictionaryBenchmark --run-name "commit: $GIT_COMMIT"
