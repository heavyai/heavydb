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

/**
 * @file    DataMgr.h
 * @brief
 *
 */

#pragma once

#include "AbstractBuffer.h"
#include "AbstractBufferMgr.h"
#include "BufferMgr/Buffer.h"
#include "BufferMgr/BufferMgr.h"
#include "MemoryLevel.h"
#include "OSDependent/heavyai_fs.h"
#include "PersistentStorageMgr/PersistentStorageMgr.h"
#include "Shared/SystemParameters.h"
#include "Shared/heavyai_shared_mutex.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace File_Namespace {
class FileBuffer;
class GlobalFileMgr;
}  // namespace File_Namespace

namespace CudaMgr_Namespace {
class CudaMgr;
}

class DeviceAllocator;

namespace Buffer_Namespace {
class CpuBufferMgr;
class GpuCudaBufferMgr;
}  // namespace Buffer_Namespace

namespace Data_Namespace {

struct MemoryInfoSnapshot {
  Buffer_Namespace::MemoryInfo cpu_memory_info;
  std::vector<Buffer_Namespace::MemoryInfo> gpu_memory_info;

  MemoryInfoSnapshot(const Buffer_Namespace::MemoryInfo& cpu_memory_info,
                     const std::vector<Buffer_Namespace::MemoryInfo>& gpu_memory_info)
      : cpu_memory_info(cpu_memory_info), gpu_memory_info(gpu_memory_info) {}
};

//! Parse /proc/meminfo into key/value pairs.
class ProcMeminfoParser {
  std::unordered_map<std::string, size_t> items_;

 public:
  ProcMeminfoParser() {
    std::ifstream f("/proc/meminfo");
    std::stringstream ss;
    ss << f.rdbuf();

    for (const std::string& line : split(ss.str(), "\n")) {
      if (line.empty()) {
        continue;
      }
      const auto nv = split(line, ":", 1);
      CHECK(nv.size() == 2) << "unexpected line format in /proc/meminfo: " << line;
      const auto name = strip(nv[0]), value = to_lower(strip(nv[1]));
      auto v = split(value);
      CHECK(v.size() == 1 || v.size() == 2)
          << "unexpected line format in /proc/meminfo: " << line;
      items_[name] = std::atoll(v[0].c_str());
      if (v.size() == 2) {
        CHECK(v[1] == "kb") << "unexpected unit suffix in /proc/meminfo: " << line;
        items_[name] *= 1024;
      }
    }
  }

  auto operator[](const std::string& name) { return items_[name]; }
  auto begin() { return items_.begin(); }
  auto end() { return items_.end(); }
};

//! Parse /proc/buddyinfo into a few fragmentation-related data.
class ProcBuddyinfoParser {
 public:
  void parseBuddyinfo();  // Set member variables.
  double getFragmentationPercent() const { return frag_percent_; }
  size_t getSumAvailPages() const { return sum_avail_pages_; }
  size_t getSumHighestBlocks() const { return sum_highest_blocks_; }

 private:
  double frag_percent_{0};        // Weighted score of available memory.
  size_t sum_avail_pages_{0};     // Sum of all available non-DMA pages.
  size_t sum_highest_blocks_{0};  // Sum of highest non-DMA blocks.
};

class DataMgr {
  friend class GlobalFileMgr;

 public:
  explicit DataMgr(
      const std::string& dataDir,
      const SystemParameters& system_parameters,
      std::unique_ptr<CudaMgr_Namespace::CudaMgr> cudaMgr,
      const bool useGpus,
      const size_t reservedGpuMem = (1 << 27),
      const size_t numReaderThreads = 0, /* 0 means use default for # of reader threads */
      const File_Namespace::DiskCacheConfig cacheConfig =
          File_Namespace::DiskCacheConfig());
  ~DataMgr();
  AbstractBuffer* createChunkBuffer(const ChunkKey& key,
                                    const MemoryLevel memoryLevel,
                                    const int deviceId = 0,
                                    const size_t page_size = 0);
  AbstractBuffer* getChunkBuffer(const ChunkKey& key,
                                 const MemoryLevel memoryLevel,
                                 const int deviceId = 0,
                                 const size_t numBytes = 0);
  void deleteChunk(const ChunkKey& key, const MemoryLevel mem_level, const int device_id);
  void deleteChunksWithPrefix(const ChunkKey& keyPrefix);
  void deleteChunksWithPrefix(const ChunkKey& keyPrefix, const MemoryLevel memLevel);
  AbstractBuffer* alloc(const MemoryLevel memoryLevel,
                        const int deviceId,
                        const size_t numBytes);
  void free(AbstractBuffer* buffer);
  // copies one buffer to another
  void copy(AbstractBuffer* destBuffer, AbstractBuffer* srcBuffer);
  bool isBufferOnDevice(const ChunkKey& key,
                        const MemoryLevel memLevel,
                        const int deviceId);
  void takeMemoryInfoSnapshot(const std::string& key);
  std::unique_ptr<MemoryInfoSnapshot> getAndResetMemoryInfoSnapshot(
      const std::string& key);
  std::vector<Buffer_Namespace::MemoryInfo> getMemoryInfo(
      const MemoryLevel memLevel) const;
  std::string dumpLevel(const MemoryLevel memLevel);
  void clearMemory(const MemoryLevel memLevel);

  const std::map<ChunkKey, File_Namespace::FileBuffer*>& getChunkMap();
  void checkpoint(const int db_id,
                  const int tb_id);  // checkpoint for individual table of DB
  void checkpoint(const int db_id, const int table_id, const MemoryLevel memory_level);
  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                       const ChunkKey& keyPrefix);
  inline bool gpusPresent() const { return hasGpus_; }
  void removeTableRelatedDS(const int db_id, const int tb_id);
  void removeMutableTableDiskCacheData(const int db_id, const int tb_id) const;
  void setTableEpoch(const int db_id, const int tb_id, const int start_epoch);
  size_t getTableEpoch(const int db_id, const int tb_id);
  void resetTableEpochFloor(const int32_t db_id, const int32_t tb_id);

  CudaMgr_Namespace::CudaMgr* getCudaMgr() const { return cudaMgr_.get(); }
  File_Namespace::GlobalFileMgr* getGlobalFileMgr() const;
  std::shared_ptr<ForeignStorageInterface> getForeignStorageInterface() const;

  // the number of devices system can use per MemoryLevel
  // we basically assume the # device as one for 1) DISK and 2) CPU
  // if the system has GPU, we query the # GPUs we can use via CudaMgr
  // and set the number as the level size of the GPU
  // Noe that we only set first two level: DISK and CPU for CPU query execution
  std::vector<int> levelSizes_;

  struct SystemMemoryUsage {
    size_t free;         // available CPU RAM memory in bytes
    size_t total;        // total CPU RAM memory in bytes
    size_t resident;     // resident process memory in bytes
    size_t vtotal;       // total process virtual memory in bytes
    size_t regular;      // process bytes non-shared
    size_t shared;       // process bytes shared (file maps + shmem)
    double frag;         // fragmentation percent
    size_t avail_pages;  // sum of all non-dma pages in /proc/buddyinfo
    size_t high_blocks;  // sum of highest non-dma blocks in /proc/buddyinfo
  };

  static SystemMemoryUsage getSystemMemoryUsage();
  static size_t getTotalSystemMemory();

  PersistentStorageMgr* getPersistentStorageMgr() const;
  void resetBufferMgrs(const File_Namespace::DiskCacheConfig& cache_config,
                       const size_t num_reader_threads,
                       const SystemParameters& sys_params);

  size_t getCpuBufferPoolSize() const;
  size_t getGpuBufferPoolSize() const;

  // Used for testing.
  Buffer_Namespace::CpuBufferMgr* getCpuBufferMgr() const;

  // Used for testing.
  Buffer_Namespace::GpuCudaBufferMgr* getGpuBufferMgr(int32_t device_id) const;

  static void atExitHandler();

  /**
   * Set the new number of maximum allowed rows.
   *
   * Validates if this new number does not exceed current capacity, if so,
   * throws.
   *
   * @param new_max - Thew new maximum number, can be a std::nullopt, in which
   * case this limit is ignored.
   */
  void setMaxNumRows(const std::optional<int64_t>& new_max);
  std::optional<int64_t> getMaxNumRows() const;

  /**
   * Validate if current number of rows across the entire system, plus an
   * optional additional amount of rows, exceeds the allowed number of rows,
   * as typically determined by a license.
   *
   * @param additional_rows - An additional amount of rows, can be zero.
   */
  void validateNumRows(const int64_t additional_row_count = 0) const;

 private:
  void populateMgrs(const SystemParameters& system_parameters,
                    const size_t userSpecifiedNumReaderThreads,
                    const File_Namespace::DiskCacheConfig& cache_config);
  void convertDB(const std::string basePath);
  void checkpoint();  // checkpoint for whole DB, called from convertDB proc only
  void createTopLevelMetadata() const;
  void allocateCpuBufferMgr(int32_t device_id,
                            size_t total_cpu_size,
                            size_t min_cpu_slab_size,
                            size_t max_cpu_slab_size,
                            size_t default_cpu_slab_size,
                            size_t page_size,
                            const std::vector<size_t>& cpu_tier_sizes);
  std::unique_lock<std::mutex> getGlobalLockIfEnabled() const;

  mutable std::shared_mutex max_num_rows_mutex_;
  std::optional<int64_t> max_num_rows_;  // maximum number of rows as per license
  std::vector<std::vector<AbstractBufferMgr*>> bufferMgrs_;
  std::unique_ptr<CudaMgr_Namespace::CudaMgr> cudaMgr_;
  std::string dataDir_;
  bool hasGpus_;
  size_t reservedGpuMem_;
  mutable std::mutex buffer_access_mutex_;
  std::map<std::string, std::unique_ptr<MemoryInfoSnapshot>> memory_info_snapshots_;
  mutable std::shared_mutex memory_info_snapshots_mutex_;
};

std::ostream& operator<<(std::ostream& os, const DataMgr::SystemMemoryUsage&);

}  // namespace Data_Namespace
