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

#include "DataMgr/DataMgrBufferProvider.h"

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/DataMgr.h"

using namespace Data_Namespace;

void DataMgrBufferProvider::free(AbstractBuffer* buffer) {
  CHECK(data_mgr_);
  data_mgr_->free(buffer);
}

AbstractBuffer* DataMgrBufferProvider::alloc(const MemoryLevel memory_level,
                                             const int device_id,
                                             const size_t num_bytes) {
  CHECK(data_mgr_);
  return data_mgr_->alloc(memory_level, device_id, num_bytes);
}

void DataMgrBufferProvider::copyToDevice(int8_t* device_ptr,
                                         const int8_t* host_ptr,
                                         const size_t num_bytes,
                                         const int device_id) const {
  CHECK(data_mgr_);
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->copyHostToDevice(device_ptr, host_ptr, num_bytes, device_id);
}

void DataMgrBufferProvider::copyFromDevice(int8_t* host_ptr,
                                           const int8_t* device_ptr,
                                           const size_t num_bytes,
                                           const int device_id) const {
  CHECK(data_mgr_);
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->copyDeviceToHost(host_ptr, device_ptr, num_bytes, device_id);
}

void DataMgrBufferProvider::zeroDeviceMem(int8_t* device_ptr,
                                          const size_t num_bytes,
                                          const int device_id) const {
  CHECK(data_mgr_);
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->zeroDeviceMem(device_ptr, num_bytes, device_id);
}

void DataMgrBufferProvider::setDeviceMem(int8_t* device_ptr,
                                         unsigned char uc,
                                         const size_t num_bytes,
                                         const int device_id) const {
  CHECK(data_mgr_);
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->setDeviceMem(device_ptr, uc, num_bytes, device_id);
}

void DataMgrBufferProvider::setContext(const int device_id) {
  auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->setContext(device_id);
}
