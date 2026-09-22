/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CudaMgr.h"
#include "Logger/Logger.h"

bool g_enable_gpu_dynamic_smem{true};
std::string g_peer_copy_mode_name{"direct"};
int g_peer_copy_mode{CudaMgr_Namespace::kPeerCopyModeDirect};
size_t g_peer_copy_staging_buffer_size{256 * 1024 * 1024};

// Stub global variable definitions for when CUDA is disabled.
size_t g_jump_buffer_size{0};
size_t g_jump_buffer_parallel_copy_threads{4};
size_t g_jump_buffer_slots_per_device{1};
bool g_enable_lazy_jump_buffer_allocation{false};
bool g_enable_background_jump_buffer_allocation{false};
size_t g_jump_buffer_min_h2d_transfer_threshold{0};
size_t g_jump_buffer_min_d2h_transfer_threshold{0};

namespace CudaMgr_Namespace {

CudaMgr::CudaMgr(const int, const int) : device_count_(-1), start_gpu_(-1) {
  CHECK(false);
}

CudaMgr::~CudaMgr() {}

size_t CudaMgr::computePaddedBufferSize(size_t buf_size, size_t granularity) const {
  CHECK(false);
  return 0;
}

size_t CudaMgr::getGranularity(const int device_num) const {
  CHECK(false);
  return 0;
}

void CudaMgr::synchronizeDevices() const {
  CHECK(false);
}

void CudaMgr::synchronizeDevice(int) const {
  CHECK(false);
}

void CudaMgr::startBackgroundJumpBufferAllocation() {}

void CudaMgr::synchronizeStream(CUstream cuda_stream) const {
  CHECK(false);
}

CUstream CudaMgr::getDeviceTransferStream(int) const {
  CHECK(false);
  return nullptr;
}

bool CudaMgr::canAccessPeer(const int dest_device_num, const int src_device_num) const {
  CHECK(false);
  return false;
}

void CudaMgr::copyHostToDevice(int8_t* device_ptr,
                               const int8_t* host_ptr,
                               const size_t num_bytes,
                               const int device_num,
                               std::optional<std::string_view> tag,
                               CUstream cuda_stream) {
  CHECK(false);
}
void CudaMgr::copyHostToDeviceDirect(int8_t* device_ptr,
                                     const int8_t* host_ptr,
                                     const size_t num_bytes,
                                     const int device_num,
                                     std::optional<std::string_view> tag,
                                     CUstream cuda_stream) {
  CHECK(false);
}
void CudaMgr::copyHostToDevice2DDirect(int8_t* device_ptr,
                                       const size_t destination_pitch,
                                       const int8_t* host_ptr,
                                       const size_t source_pitch,
                                       const size_t width_bytes,
                                       const size_t height,
                                       const int device_num,
                                       std::optional<std::string_view> tag,
                                       CUstream cuda_stream) {
  CHECK(false);
}
bool CudaMgr::copyHostToDeviceFromPinnedProducer(
    int8_t* device_ptr,
    const size_t num_bytes,
    const int device_num,
    std::optional<std::string_view> tag,
    const std::function<void(int8_t* host_ptr, size_t num_bytes, size_t offset)>&
        producer,
    CUstream cuda_stream) {
  CHECK(false);
  return false;
}
void CudaMgr::copyDeviceToHost(int8_t* host_ptr,
                               const int8_t* device_ptr,
                               const size_t num_bytes,
                               std::optional<std::string_view> tag,
                               CUstream cuda_stream) {
  CHECK(false);
}
void CudaMgr::copyDeviceToHost(int8_t* host_ptr,
                               const int8_t* device_ptr,
                               const size_t num_bytes,
                               const int device_num,
                               std::optional<std::string_view> tag,
                               CUstream cuda_stream) {
  CHECK(false);
}
void CudaMgr::copyDeviceToDevice(int8_t* dest_ptr,
                                 int8_t* src_ptr,
                                 const size_t num_bytes,
                                 const int dest_device_num,
                                 const int src_device_num,
                                 std::optional<std::string_view> tag,
                                 CUstream cuda_stream,
                                 const bool synchronize) {
  CHECK(false);
}

void CudaMgr::copyDeviceToDeviceViaHost(int8_t* dest_ptr,
                                        int8_t* src_ptr,
                                        const size_t num_bytes,
                                        const int dest_device_num,
                                        const int src_device_num,
                                        std::optional<std::string_view> tag,
                                        CUstream cuda_stream) {
  CHECK(false);
}

void CudaMgr::copyDeviceToDeviceViaPeerStaging(int8_t* dest_ptr,
                                               int8_t* src_ptr,
                                               const size_t num_bytes,
                                               const int dest_device_num,
                                               const int src_device_num,
                                               std::optional<std::string_view> tag,
                                               CUstream cuda_stream) {
  CHECK(false);
}

int8_t* CudaMgr::allocatePeerCopyableDeviceMem(const size_t num_bytes,
                                               const int device_num) {
  CHECK(false);
  return nullptr;
}

void CudaMgr::freePeerCopyableDeviceMem(int8_t* device_ptr, const int device_num) {
  CHECK(false);
}

void CudaMgr::copyPeerToPeer(int8_t* dest_ptr,
                             const int8_t* src_ptr,
                             const size_t num_bytes,
                             const int dest_device_num,
                             const int src_device_num,
                             std::optional<std::string_view> tag,
                             CUstream cuda_stream,
                             const bool synchronize) {
  CHECK(false);
}

bool CudaMgr::ensurePeerAccessToDevicePtr(const int dest_device_num,
                                          const int src_device_num,
                                          const int8_t* device_ptr,
                                          const size_t num_bytes) {
  CHECK(false);
  return false;
}

bool CudaMgr::canAccessPeerMemoryFromKernel(const int dest_device_num,
                                            const int src_device_num) {
  CHECK(false);
  return false;
}

std::optional<int8_t*> CudaMgr::registerMappedHostMemory(const int8_t* host_ptr,
                                                         const size_t num_bytes,
                                                         const int device_num) {
  CHECK(false);
  return std::nullopt;
}

void CudaMgr::unregisterMappedHostMemory(const int8_t* host_ptr, const size_t num_bytes) {
  CHECK(false);
}

int8_t* CudaMgr::allocateDeviceMem(const size_t num_bytes,
                                   const int device_num,
                                   const bool is_slab) {
  CHECK(false);
  return nullptr;
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
