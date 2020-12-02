/*
 * Copyright 2020 OmniSci, Inc.
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

#pragma once

enum class HashType : int { OneToOne, OneToMany, ManyToMany };

class HashTable {
 public:
  virtual ~HashTable() {}

  virtual size_t getHashTableBufferSize(const ExecutorDeviceType device_type) const = 0;

  virtual int8_t* getCpuBuffer() = 0;
  virtual int8_t* getGpuBuffer() const = 0;
  virtual HashType getLayout() const = 0;

  virtual size_t getEntryCount() const = 0;
  virtual size_t getEmittedKeysCount() const = 0;
};
