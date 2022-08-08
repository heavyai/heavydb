/*
    Copyright 2021 OmniSci, Inc.
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <stdexcept>

class ParseIRError : public std::runtime_error {
 public:
  ParseIRError(const std::string message) : std::runtime_error(message) {}
};

class CompilationRetryNoLazyFetch : public std::runtime_error {
 public:
  CompilationRetryNoLazyFetch()
      : std::runtime_error("Retry query compilation with no GPU lazy fetch.") {}
};

class CompilationRetryNewScanLimit : public std::runtime_error {
 public:
  CompilationRetryNewScanLimit(const size_t new_scan_limit)
      : std::runtime_error("Retry query compilation with new scan limit.")
      , new_scan_limit_(new_scan_limit) {}

  size_t new_scan_limit_;
};

class TooManyLiterals : public std::runtime_error {
 public:
  TooManyLiterals() : std::runtime_error("Too many literals in the query") {}
};

class CompilationRetryNoCompaction : public std::runtime_error {
 public:
  CompilationRetryNoCompaction()
      : std::runtime_error("Retry query compilation with no compaction.") {}
};
