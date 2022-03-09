/*
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

#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/MemoryLevel.h"

using namespace Data_Namespace;

class BufferProvider {
 public:
  virtual ~BufferProvider() = default;

  // allocator APIs (CudaAllocator, ThrustAllocator)
  virtual void free(AbstractBuffer* buffer) = 0;
  virtual AbstractBuffer* alloc(const MemoryLevel memory_level,
                                const int device_id,
                                const size_t num_bytes) = 0;
  // should take AbstractBuffer
  virtual void copyToDevice(int8_t* device_ptr,
                            const int8_t* host_ptr,
                            const size_t num_bytes,
                            const int device_id) const = 0;
  virtual void copyFromDevice(int8_t* host_ptr,
                              const int8_t* device_ptr,
                              const size_t num_bytes,
                              const int device_id) const = 0;
  // should be properties of AbstractBuffer
  virtual void zeroDeviceMem(int8_t* device_ptr,
                             const size_t num_bytes,
                             const int device_id) const = 0;
  virtual void setDeviceMem(int8_t* device_ptr,
                            unsigned char uc,
                            const size_t num_bytes,
                            const int device_id) const = 0;
  virtual void setContext(const int device_id) = 0;
};
