/*
 * Copyright 2020 OmniSci Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file	Epoch.h
 * @author	Todd Mostak <todd@omnisci.com>
 *
 * This file includes the class specification for the FILE manager (FileMgr), and related
 * data structures and types.
 */

#pragma once

#include <cstdint>
#include <limits>

struct Epoch {
 public:
  Epoch() {
    epoch_storage[0] = std::numeric_limits<int32_t>::min();
    epoch_storage[1] = 0;
  }

  Epoch(const int32_t epoch_floor, const int32_t epoch_ceiling) {
    epoch_storage[0] = epoch_floor;
    epoch_storage[1] = epoch_ceiling;
  }

  // Getters
  inline int32_t floor() const { return epoch_storage[0]; }
  inline int32_t ceiling() const { return epoch_storage[1]; }

  // Setters
  inline void floor(const int32_t epoch_floor) {
    epoch_storage[0] = epoch_floor;
    ;
  }

  inline void ceiling(const int32_t epoch_ceiling) { epoch_storage[1] = epoch_ceiling; }

  inline int32_t increment() { return ++epoch_storage[1]; }

  /**
   * Allows access to ptr to epoch internal storage for atomic file reads and writes to
   * epoch storage
   */

  inline int8_t* storage_ptr() { return reinterpret_cast<int8_t*>(&epoch_storage[0]); }

  static inline size_t byte_size() { return 2 * sizeof(int64_t); }

  static inline int64_t min_allowable_epoch() {
    return std::numeric_limits<int32_t>::min();
  }

  static inline int64_t max_allowable_epoch() {
    return std::numeric_limits<int32_t>::max() - 1;
  }

 private:
  int64_t epoch_storage[2];
};