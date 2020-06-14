/*
 * Copyright 2019 OmniSci, Inc.
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
 * @brief   Allocate GPU memory using GpuBuffers via DataMgr. Unlike the CudaAllocator,
 * these buffers are destroyed and memory is released when the parent object goes out of
 * scope.
 *
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace Data_Namespace {
class DataMgr;
class AbstractBuffer;
};  // namespace Data_Namespace

class ThrustAllocator {
 public:
  using value_type = int8_t;
  ThrustAllocator(Data_Namespace::DataMgr* mgr, const int id)
      : data_mgr_(mgr), device_id_(id) {}
  ~ThrustAllocator();

  int8_t* allocate(std::ptrdiff_t num_bytes);
  void deallocate(int8_t* ptr, size_t num_bytes);

  int8_t* allocateScopedBuffer(std::ptrdiff_t num_bytes);

  Data_Namespace::DataMgr* getDataMgr() const { return data_mgr_; }

  int getDeviceId() const { return device_id_; }

 private:
  Data_Namespace::DataMgr* data_mgr_;
  const int device_id_;
  using PtrMapperType = std::unordered_map<int8_t*, Data_Namespace::AbstractBuffer*>;
  PtrMapperType raw_to_ab_ptr_;
  std::vector<Data_Namespace::AbstractBuffer*> scoped_buffers_;
  std::vector<int8_t*> default_alloc_scoped_buffers_;  // for unit tests only
  size_t num_allocations_{0};
};
