/*
 * Copyright 2017 MapD Technologies, Inc.
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

/*
 * @file    ThrustAllocator.h
 * @author  Minggang Yu <miyu@mapd.com>
 * @brief   Bridge allocator for thrust that delegates to DataMgr methods.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef THRUSTALLOCATOR_H
#define THRUSTALLOCATOR_H

#include <unordered_map>
#include <vector>

namespace Data_Namespace {
class DataMgr;
class AbstractBuffer;
};  // Data_Namespace

class ThrustAllocator {
 public:
  typedef int8_t value_type;
  ThrustAllocator(Data_Namespace::DataMgr* mgr, const int id) : data_mgr_(mgr), device_id_(id) {}
  ~ThrustAllocator();

  int8_t* allocate(std::ptrdiff_t num_bytes);
  void deallocate(int8_t* ptr, size_t num_bytes);

  int8_t* allocateScopedBuffer(std::ptrdiff_t num_bytes);

  Data_Namespace::DataMgr* getDataMgr() const { return data_mgr_; }

  int getDeviceId() const { return device_id_; }

 private:
  Data_Namespace::DataMgr* data_mgr_;
  const int device_id_;
  typedef std::unordered_map<int8_t*, Data_Namespace::AbstractBuffer*> PtrMapperType;
  PtrMapperType raw_to_ab_ptr_;
  std::vector<Data_Namespace::AbstractBuffer*> scoped_buffers_;
  std::vector<int8_t*> default_alloc_scoped_buffers_;  // for unit tests only
};

#endif /* THRUSTALLOCATOR_H */
