/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "CudaMgr.h"
#include "Logger/Logger.h"

namespace CudaMgr_Namespace {

CudaMgr::CudaMgr(const int, const int) : device_count_(-1), start_gpu_(-1) {
  CHECK(false);
}

CudaMgr::~CudaMgr() {}

void CudaMgr::synchronizeDevices() const {
  CHECK(false);
}

void CudaMgr::copyHostToDevice(int8_t* device_ptr,
                               const int8_t* host_ptr,
                               const size_t num_bytes,
                               const int device_num,
                               CUstream cuda_stream) {
  CHECK(false);
}
void CudaMgr::copyDeviceToHost(int8_t* host_ptr,
                               const int8_t* device_ptr,
                               const size_t num_bytes,
                               const int device_num,
                               CUstream cuda_stream) {
  CHECK(false);
}
void CudaMgr::copyDeviceToDevice(int8_t* dest_ptr,
                                 int8_t* src_ptr,
                                 const size_t num_bytes,
                                 const int dest_device_num,
                                 const int src_device_num,
                                 CUstream cuda_stream) {
  CHECK(false);
}

int8_t* CudaMgr::allocatePinnedHostMem(const size_t num_bytes) {
  CHECK(false);
  return nullptr;
}
int8_t* CudaMgr::allocateDeviceMem(const size_t num_bytes, const int device_num) {
  CHECK(false);
  return nullptr;
}
void CudaMgr::freePinnedHostMem(int8_t* host_ptr) {
  CHECK(false);
}
void CudaMgr::freeDeviceMem(int8_t* device_ptr) {
  CHECK(false);
}
void CudaMgr::zeroDeviceMem(int8_t* device_ptr,
                            const size_t num_bytes,
                            const int device_num,
                            CUstream cuda_stream) {
  CHECK(false);
}
void CudaMgr::setDeviceMem(int8_t* device_ptr,
                           const unsigned char uc,
                           const size_t num_bytes,
                           const int device_num,
                           CUstream cuda_stream) {
  CHECK(false);
}

bool CudaMgr::isArchMaxwellOrLaterForAll() const {
  CHECK(false);
  return false;
}
bool CudaMgr::isArchVoltaOrGreaterForAll() const {
  CHECK(false);
  return false;
}

void CudaMgr::setContext(const int) const {
  CHECK(false);
}

int CudaMgr::getContext() const {
  CHECK(false);
  return 0;
}

}  // namespace CudaMgr_Namespace
