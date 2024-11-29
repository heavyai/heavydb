
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
 * @file    DataMgr.cpp
 * @brief
 */

#include "DataMgr/DataMgr.h"
#include "BufferMgr/CpuBufferMgr/CpuBufferMgr.h"
#include "BufferMgr/CpuBufferMgr/TieredCpuBufferMgr.h"
#include "BufferMgr/GpuCudaBufferMgr/GpuCudaBufferMgr.h"
#include "Catalog/Catalog.h"
#include "Catalog/SysCatalog.h"
#include "CudaMgr/CudaMgr.h"
#include "DataMgr/Allocators/CudaAllocator.h"
#include "FileMgr/GlobalFileMgr.h"
#include "LockMgr/LockMgr.h"
#include "PersistentStorageMgr/PersistentStorageMgr.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#include <boost/container/small_vector.hpp>
#include <boost/filesystem.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <limits>
#include <numeric>
#include <string_view>

extern bool g_enable_fsi;

#ifdef ENABLE_MEMKIND
bool g_enable_tiered_cpu_mem{false};
std::string g_pmem_path{};
size_t g_pmem_size{0};
#endif

bool g_use_cpu_mem_pool_size_for_max_cpu_slab_size{false};
bool g_enable_data_mgr_global_lock{false};

namespace Data_Namespace {

namespace {
// Global pointer and function for atexit registration.
// Do NOT use this pointer for anything else.
static DataMgr* g_data_mgr_ptr = nullptr;
static bool at_exit_called = false;
}  // namespace

void DataMgr::atExitHandler() {
  at_exit_called = true;
  if (g_data_mgr_ptr && g_data_mgr_ptr->hasGpus_) {
    // safely destroy all gpu allocations explicitly to avoid unexpected
    // `CUDA_ERROR_DEINITIALIZED` exception while trying to synchronize
    // devices to destroy BufferMgr for GPU, i.e., 'GpuCudaBufferMgr` and `CudaMgr`
    g_data_mgr_ptr->clearMemory(MemoryLevel::GPU_LEVEL);
  }
}

DataMgr::DataMgr(const std::string& dataDir,
                 const SystemParameters& system_parameters,
                 std::unique_ptr<CudaMgr_Namespace::CudaMgr> cudaMgr,
                 const bool useGpus,
                 const size_t reservedGpuMem,
                 const size_t numReaderThreads,
                 const File_Namespace::DiskCacheConfig cache_config)
    : cudaMgr_{std::move(cudaMgr)}
    , dataDir_{dataDir}
    , hasGpus_{false}
    , reservedGpuMem_{reservedGpuMem} {
  if (useGpus) {
    if (cudaMgr_) {
      hasGpus_ = true;

      // we register the `atExitHandler` if we create `DataMgr` having GPU
      // to make sure we clear all allocated GPU memory when destructing this `DataMgr`
      g_data_mgr_ptr = this;
      std::atexit(atExitHandler);
    } else {
      LOG(ERROR) << "CudaMgr instance is invalid, falling back to CPU-only mode.";
      hasGpus_ = false;
    }
  } else {
    // NOTE: useGpus == false with a valid cudaMgr is a potentially valid configuration.
    // i.e. QueryEngine can be set to cpu-only for a cuda-enabled build, but still have
    // rendering enabled. The renderer would require a CudaMgr in this case, in addition
    // to a GpuCudaBufferMgr for cuda-backed thrust allocations.
    // We're still setting hasGpus_ to false in that case tho to enforce cpu-only query
    // execution.
    hasGpus_ = false;
  }

  populateMgrs(system_parameters, numReaderThreads, cache_config);
  createTopLevelMetadata();
}

DataMgr::~DataMgr() {
  g_data_mgr_ptr = nullptr;

  // This duplicates atExitHandler so we still shut down in the case of a startup
  // exception. We can request cleanup of GPU memory twice, so it's safe.
  if (!at_exit_called && hasGpus_) {
    clearMemory(GPU_LEVEL);
  }

  int numLevels = bufferMgrs_.size();
  for (int level = numLevels - 1; level >= 0; --level) {
    for (size_t device = 0; device < bufferMgrs_[level].size(); device++) {
      delete bufferMgrs_[level][device];
    }
  }
}

DataMgr::SystemMemoryUsage DataMgr::getSystemMemoryUsage() {
  SystemMemoryUsage usage;
#ifdef __linux__

  // Determine Linux available memory and total memory.
  // Available memory is different from free memory because
  // when Linux sees free memory, it tries to use it for
  // stuff like disk caching. However, the memory is not
  // reserved and is still available to be allocated by
  // user processes.
  // Parsing /proc/meminfo for this info isn't very elegant
  // but as a virtual file it should be reasonably fast.
  // See also:
  //   https://github.com/torvalds/linux/commit/34e431b0ae398fc54ea69ff85ec700722c9da773
  ProcMeminfoParser mi;
  usage.free = mi["MemAvailable"];
  usage.total = mi["MemTotal"];

  // Determine process memory in use.
  // See also:
  //   https://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-runtime-using-c
  //   http://man7.org/linux/man-pages/man5/proc.5.html
  int64_t size = 0;
  int64_t resident = 0;
  int64_t shared = 0;

  std::ifstream fstatm("/proc/self/statm");
  fstatm >> size >> resident >> shared;
  fstatm.close();

  long page_size =
      sysconf(_SC_PAGE_SIZE);  // in case x86-64 is configured to use 2MB pages

  usage.resident = resident * page_size;
  usage.vtotal = size * page_size;
  usage.regular = (resident - shared) * page_size;
  usage.shared = shared * page_size;

  ProcBuddyinfoParser bi{};
  bi.parseBuddyinfo();
  usage.frag = bi.getFragmentationPercent();
  usage.avail_pages = bi.getSumAvailPages();
  usage.high_blocks = bi.getSumHighestBlocks();

#else

  usage.total = 0;
  usage.free = 0;
  usage.resident = 0;
  usage.vtotal = 0;
  usage.regular = 0;
  usage.shared = 0;
  usage.frag = 0.0;
  usage.avail_pages = 0;
  usage.high_blocks = 0;

#endif

  return usage;
}

size_t DataMgr::getTotalSystemMemory() {
#ifdef __APPLE__
  int mib[2];
  size_t physical_memory;
  size_t length;
  // Get the Physical memory size
  mib[0] = CTL_HW;
  mib[1] = HW_MEMSIZE;
  length = sizeof(size_t);
  sysctl(mib, 2, &physical_memory, &length, NULL, 0);
  return physical_memory;
#elif defined(_MSC_VER)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullTotalPhys;
#else  // Linux
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
#endif
}

void DataMgr::allocateCpuBufferMgr(int32_t device_id,
                                   size_t total_cpu_size,
                                   size_t min_cpu_slab_size,
                                   size_t max_cpu_slab_size,
                                   size_t default_cpu_slab_size,
                                   size_t page_size,
                                   const CpuTierSizeVector& cpu_tier_sizes) {
#ifdef ENABLE_MEMKIND
  if (g_enable_tiered_cpu_mem) {
    bufferMgrs_[1].push_back(
        new Buffer_Namespace::TieredCpuBufferMgr(0,
                                                 total_cpu_size,
                                                 cudaMgr_.get(),
                                                 min_cpu_slab_size,
                                                 max_cpu_slab_size,
                                                 default_cpu_slab_size,
                                                 page_size,
                                                 cpu_tier_sizes,
                                                 bufferMgrs_[0][0]));
    return;
  }
#endif

  bufferMgrs_[1].push_back(new Buffer_Namespace::CpuBufferMgr(0,
                                                              total_cpu_size,
                                                              cudaMgr_.get(),
                                                              min_cpu_slab_size,
                                                              max_cpu_slab_size,
                                                              default_cpu_slab_size,
                                                              page_size,
                                                              bufferMgrs_[0][0]));
}

// This function exists for testing purposes so that we can test a reset of the cache.
void DataMgr::resetBufferMgrs(const File_Namespace::DiskCacheConfig& cache_config,
                              const size_t num_reader_threads,
                              const SystemParameters& sys_params) {
  int numLevels = bufferMgrs_.size();
  for (int level = numLevels - 1; level >= 0; --level) {
    for (size_t device = 0; device < bufferMgrs_[level].size(); device++) {
      delete bufferMgrs_[level][device];
    }
  }
  bufferMgrs_.clear();
  populateMgrs(sys_params, num_reader_threads, cache_config);
  createTopLevelMetadata();
}

namespace {
size_t get_slab_size(size_t initial_slab_size,
                     size_t buffer_pool_size,
                     size_t page_size) {
  auto slab_size = std::min(initial_slab_size, buffer_pool_size);
  slab_size = (slab_size / page_size) * page_size;
  return slab_size;
}
}  // namespace

void DataMgr::populateMgrs(const SystemParameters& system_parameters,
                           const size_t userSpecifiedNumReaderThreads,
                           const File_Namespace::DiskCacheConfig& cache_config) {
  // no need for locking, as this is only called in the constructor
  bufferMgrs_.resize(2);
  bufferMgrs_[0].push_back(
      new PersistentStorageMgr(dataDir_, userSpecifiedNumReaderThreads, cache_config));

  levelSizes_.push_back(1);
  auto page_size = system_parameters.buffer_page_size;
  CHECK_GT(page_size, size_t(0));
  auto cpu_buffer_size = system_parameters.cpu_buffer_mem_bytes;
  if (cpu_buffer_size == 0) {  // if size is not specified
    const auto total_system_memory = getTotalSystemMemory();
    VLOG(1) << "Detected " << (float)total_system_memory / (1024 * 1024)
            << "M of total system memory.";
    cpu_buffer_size = total_system_memory *
                      0.8;  // should get free memory instead of this ugly heuristic
  }
  cpu_buffer_size = (cpu_buffer_size / page_size) * page_size;
  auto min_cpu_slab_size =
      get_slab_size(system_parameters.min_cpu_slab_size, cpu_buffer_size, page_size);
  auto max_cpu_slab_size =
      g_use_cpu_mem_pool_size_for_max_cpu_slab_size
          ? cpu_buffer_size
          : get_slab_size(
                system_parameters.max_cpu_slab_size, cpu_buffer_size, page_size);
  auto default_cpu_slab_size =
      get_slab_size(system_parameters.default_cpu_slab_size, cpu_buffer_size, page_size);
  LOG(INFO) << "Min CPU Slab Size is " << float(min_cpu_slab_size) / (1024 * 1024)
            << "MB";
  LOG(INFO) << "Max CPU Slab Size is " << float(max_cpu_slab_size) / (1024 * 1024)
            << "MB";
  LOG(INFO) << "Default CPU Slab Size is " << float(default_cpu_slab_size) / (1024 * 1024)
            << "MB";
  LOG(INFO) << "Max memory pool size for CPU is "
            << float(cpu_buffer_size) / (1024 * 1024) << "MB";

  size_t total_cpu_size = 0;

#ifdef ENABLE_MEMKIND
  CpuTierSizeVector cpu_tier_sizes(numCpuTiers, 0);
  cpu_tier_sizes[CpuTier::DRAM] = cpuBufferSize;
  if (g_enable_tiered_cpu_mem) {
    cpu_tier_sizes[CpuTier::PMEM] = g_pmem_size;
    LOG(INFO) << "Max memory pool size for PMEM is " << (float)g_pmem_size / (1024 * 1024)
              << "MB";
  }
  for (auto cpu_tier_size : cpu_tier_sizes) {
    total_cpu_size += cpu_tier_size;
  }
#else
  CpuTierSizeVector cpu_tier_sizes{};
  total_cpu_size = cpu_buffer_size;
#endif

  if (hasGpus_ || cudaMgr_) {
    LOG(INFO) << "Reserved GPU memory is " << (float)reservedGpuMem_ / (1024 * 1024)
              << "MB includes render buffer allocation";
    bufferMgrs_.resize(3);
    allocateCpuBufferMgr(0,
                         total_cpu_size,
                         min_cpu_slab_size,
                         max_cpu_slab_size,
                         default_cpu_slab_size,
                         page_size,
                         cpu_tier_sizes);

    levelSizes_.push_back(1);
    auto num_gpus = cudaMgr_->getDeviceCount();
    for (int gpu_num = 0; gpu_num < num_gpus; ++gpu_num) {
      auto gpu_max_mem_size =
          system_parameters.gpu_buffer_mem_bytes != 0
              ? system_parameters.gpu_buffer_mem_bytes
              : (cudaMgr_->getDeviceProperties(gpu_num)->globalMem) - (reservedGpuMem_);
      gpu_max_mem_size = (gpu_max_mem_size / page_size) * page_size;
      auto min_gpu_slab_size =
          get_slab_size(system_parameters.min_gpu_slab_size, gpu_max_mem_size, page_size);
      auto max_gpu_slab_size =
          get_slab_size(system_parameters.max_gpu_slab_size, gpu_max_mem_size, page_size);
      auto default_gpu_slab_size = get_slab_size(
          system_parameters.default_gpu_slab_size, gpu_max_mem_size, page_size);
      LOG(INFO) << "Min GPU Slab size for GPU " << gpu_num << " is "
                << float(min_gpu_slab_size) / (1024 * 1024) << "MB";
      LOG(INFO) << "Max GPU Slab size for GPU " << gpu_num << " is "
                << float(max_gpu_slab_size) / (1024 * 1024) << "MB";
      LOG(INFO) << "Default GPU Slab size for GPU " << gpu_num << " is "
                << float(default_gpu_slab_size) / (1024 * 1024) << "MB";
      LOG(INFO) << "Max memory pool size for GPU " << gpu_num << " is "
                << float(gpu_max_mem_size) / (1024 * 1024) << "MB";
      bufferMgrs_[2].push_back(
          new Buffer_Namespace::GpuCudaBufferMgr(gpu_num,
                                                 gpu_max_mem_size,
                                                 cudaMgr_.get(),
                                                 min_gpu_slab_size,
                                                 max_gpu_slab_size,
                                                 default_gpu_slab_size,
                                                 page_size,
                                                 bufferMgrs_[1][0]));
    }
    levelSizes_.push_back(num_gpus);
  } else {
    allocateCpuBufferMgr(0,
                         total_cpu_size,
                         min_cpu_slab_size,
                         max_cpu_slab_size,
                         default_cpu_slab_size,
                         page_size,
                         cpu_tier_sizes);
    levelSizes_.push_back(1);
  }
}

void DataMgr::convertDB(const std::string basePath) {
  // no need for locking, as this is only called in the constructor

  /* check that the data directory exists and it's empty */
  std::string mapdDataPath(basePath + "/../" + shared::kDataDirectoryName + "/");
  boost::filesystem::path path(mapdDataPath);
  if (boost::filesystem::exists(path)) {
    if (!boost::filesystem::is_directory(path)) {
      LOG(FATAL) << "Path to directory \"" + shared::kDataDirectoryName +
                        "\" to convert DB is not a directory.";
    }
  } else {  // data directory does not exist
    LOG(FATAL) << "Path to directory \"" + shared::kDataDirectoryName +
                      "\" to convert DB does not exist.";
  }

  File_Namespace::GlobalFileMgr* gfm{nullptr};
  gfm = dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  CHECK(gfm);

  LOG(INFO) << "Database conversion started.";
  // this call also copies data into new DB structure
  File_Namespace::FileMgr* fm_base_db = new File_Namespace::FileMgr(gfm, basePath);
  delete fm_base_db;

  /* write content of DB into newly created/converted DB structure & location */
  checkpoint();  // outputs data files as well as metadata files
  LOG(INFO) << "Database conversion completed.";
}

void DataMgr::createTopLevelMetadata()
    const {  // create metadata shared by all tables of all DBs
  ChunkKey chunkKey(2);
  chunkKey[0] = 0;  // top level db_id
  chunkKey[1] = 0;  // top level tb_id

  File_Namespace::GlobalFileMgr* gfm{nullptr};
  gfm = dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  CHECK(gfm);

  auto fm_top = gfm->getFileMgr(chunkKey);
  if (auto fm = dynamic_cast<File_Namespace::FileMgr*>(fm_top)) {
    fm->createOrMigrateTopLevelMetadata();
  }
}

void DataMgr::takeMemoryInfoSnapshot() {
  auto cpu_memory_info = getMemoryInfo(MemoryLevel::CPU_LEVEL);
  CHECK_EQ(cpu_memory_info.size(), size_t(1));
  memory_info_snapshot_ = std::make_unique<MemoryInfoSnapshot>(
      cpu_memory_info[0], getMemoryInfo(MemoryLevel::GPU_LEVEL));
}

std::unique_ptr<MemoryInfoSnapshot> DataMgr::getAndResetMemoryInfoSnapshot() {
  CHECK(memory_info_snapshot_);
  return std::move(memory_info_snapshot_);
}

std::vector<Buffer_Namespace::MemoryInfo> DataMgr::getMemoryInfo(
    const MemoryLevel mem_level) const {
  auto global_lock = getGlobalLockIfEnabled();

  std::vector<Buffer_Namespace::MemoryInfo> mem_info;
  if (mem_level == MemoryLevel::CPU_LEVEL) {
    Buffer_Namespace::CpuBufferMgr* cpu_buffer =
        dynamic_cast<Buffer_Namespace::CpuBufferMgr*>(
            bufferMgrs_[MemoryLevel::CPU_LEVEL][0]);
    CHECK(cpu_buffer);
    mem_info.push_back(cpu_buffer->getMemoryInfo());
  } else if (hasGpus_) {
    int numGpus = cudaMgr_->getDeviceCount();
    for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
      Buffer_Namespace::GpuCudaBufferMgr* gpu_buffer =
          dynamic_cast<Buffer_Namespace::GpuCudaBufferMgr*>(
              bufferMgrs_[MemoryLevel::GPU_LEVEL][gpuNum]);
      CHECK(gpu_buffer);
      mem_info.push_back(gpu_buffer->getMemoryInfo());
    }
  }
  return mem_info;
}

std::string DataMgr::dumpLevel(const MemoryLevel memLevel) {
  auto global_lock = getGlobalLockIfEnabled();

  // if gpu we need to iterate through all the buffermanagers for each card
  if (memLevel == MemoryLevel::GPU_LEVEL) {
    int numGpus = cudaMgr_->getDeviceCount();
    std::ostringstream tss;
    for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
      tss << bufferMgrs_[memLevel][gpuNum]->printSlabs();
    }
    return tss.str();
  } else {
    return bufferMgrs_[memLevel][0]->printSlabs();
  }
}

void DataMgr::clearMemory(const MemoryLevel memLevel) {
  auto global_lock = getGlobalLockIfEnabled();

  // if gpu we need to iterate through all the buffermanagers for each card
  if (memLevel == MemoryLevel::GPU_LEVEL) {
    if (cudaMgr_) {
      int numGpus = cudaMgr_->getDeviceCount();
      for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
        auto buffer_mgr_for_gpu =
            dynamic_cast<Buffer_Namespace::BufferMgr*>(bufferMgrs_[memLevel][gpuNum]);
        CHECK(buffer_mgr_for_gpu);
        buffer_mgr_for_gpu->clearSlabs();
      }
    } else {
      LOG(WARNING) << "Unable to clear GPU memory: No GPUs detected";
    }
  } else {
    auto buffer_mgr_for_cpu =
        dynamic_cast<Buffer_Namespace::BufferMgr*>(bufferMgrs_[memLevel][0]);
    CHECK(buffer_mgr_for_cpu);
    buffer_mgr_for_cpu->clearSlabs();
  }
}

bool DataMgr::isBufferOnDevice(const ChunkKey& key,
                               const MemoryLevel memLevel,
                               const int deviceId) {
  auto global_lock = getGlobalLockIfEnabled();
  return bufferMgrs_[memLevel][deviceId]->isBufferOnDevice(key);
}

void DataMgr::getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                              const ChunkKey& keyPrefix) {
  auto global_lock = getGlobalLockIfEnabled();
  bufferMgrs_[0][0]->getChunkMetadataVecForKeyPrefix(chunkMetadataVec, keyPrefix);
}

AbstractBuffer* DataMgr::createChunkBuffer(const ChunkKey& key,
                                           const MemoryLevel memoryLevel,
                                           const int deviceId,
                                           const size_t page_size) {
  auto global_lock = getGlobalLockIfEnabled();
  int level = static_cast<int>(memoryLevel);
  return bufferMgrs_[level][deviceId]->createBuffer(key, page_size);
}

AbstractBuffer* DataMgr::getChunkBuffer(const ChunkKey& key,
                                        const MemoryLevel memoryLevel,
                                        const int deviceId,
                                        const size_t numBytes) {
  auto global_lock = getGlobalLockIfEnabled();
  const auto level = static_cast<size_t>(memoryLevel);
  CHECK_LT(level, levelSizes_.size());     // make sure we have a legit buffermgr
  CHECK_LT(deviceId, levelSizes_[level]);  // make sure we have a legit buffermgr
  return bufferMgrs_[level][deviceId]->getBuffer(key, numBytes);
}

void DataMgr::deleteChunksWithPrefix(const ChunkKey& keyPrefix) {
  auto global_lock = getGlobalLockIfEnabled();

  int numLevels = bufferMgrs_.size();
  for (int level = numLevels - 1; level >= 0; --level) {
    for (int device = 0; device < levelSizes_[level]; ++device) {
      bufferMgrs_[level][device]->deleteBuffersWithPrefix(keyPrefix);
    }
  }
}

// only deletes the chunks at the given memory level
void DataMgr::deleteChunksWithPrefix(const ChunkKey& keyPrefix,
                                     const MemoryLevel memLevel) {
  auto global_lock = getGlobalLockIfEnabled();

  if (bufferMgrs_.size() <= memLevel) {
    return;
  }
  for (int device = 0; device < levelSizes_[memLevel]; ++device) {
    bufferMgrs_[memLevel][device]->deleteBuffersWithPrefix(keyPrefix);
  }
}

// only deletes the chunks at the given memory level
void DataMgr::deleteChunk(const ChunkKey& key,
                          const MemoryLevel memLevel,
                          const int device_id) {
  auto global_lock = getGlobalLockIfEnabled();
  CHECK_LT(memLevel, bufferMgrs_.size());
  bufferMgrs_[memLevel][device_id]->deleteBuffer(key);
}

AbstractBuffer* DataMgr::alloc(const MemoryLevel memoryLevel,
                               const int deviceId,
                               const size_t numBytes) {
  auto global_lock = getGlobalLockIfEnabled();
  const auto level = static_cast<int>(memoryLevel);
  CHECK_LT(deviceId, levelSizes_[level]);
  return bufferMgrs_[level][deviceId]->alloc(numBytes);
}

void DataMgr::free(AbstractBuffer* buffer) {
  auto global_lock = getGlobalLockIfEnabled();
  int level = static_cast<int>(buffer->getType());
  bufferMgrs_[level][buffer->getDeviceId()]->free(buffer);
}

void DataMgr::copy(AbstractBuffer* destBuffer, AbstractBuffer* srcBuffer) {
  destBuffer->write(srcBuffer->getMemoryPtr(),
                    srcBuffer->size(),
                    0,
                    srcBuffer->getType(),
                    srcBuffer->getDeviceId());
}

// could add function below to do arbitrary copies between buffers

// void DataMgr::copy(AbstractBuffer *destBuffer, const AbstractBuffer *srcBuffer, const
// size_t numBytes, const size_t destOffset, const size_t srcOffset) {
//} /

void DataMgr::checkpoint(const int db_id, const int tb_id) {
  // TODO(adb): do we need a buffer mgr lock here?
  // MAT Yes to reduce Parallel Executor TSAN issues (and correctness for now)
  auto global_lock = getGlobalLockIfEnabled();
  for (auto levelIt = bufferMgrs_.rbegin(); levelIt != bufferMgrs_.rend(); ++levelIt) {
    // use reverse iterator so we start at GPU level, then CPU then DISK
    for (auto deviceIt = levelIt->begin(); deviceIt != levelIt->end(); ++deviceIt) {
      (*deviceIt)->checkpoint(db_id, tb_id);
    }
  }
  // Validate licensing limitations
  Catalog_Namespace::SysCatalog::instance().getDataMgr().validateNumRows();
}

void DataMgr::checkpoint(const int db_id,
                         const int table_id,
                         const MemoryLevel memory_level) {
  auto global_lock = getGlobalLockIfEnabled();
  CHECK_LT(static_cast<size_t>(memory_level), bufferMgrs_.size());
  CHECK_LT(static_cast<size_t>(memory_level), levelSizes_.size());
  for (int device_id = 0; device_id < levelSizes_[memory_level]; device_id++) {
    bufferMgrs_[memory_level][device_id]->checkpoint(db_id, table_id);
  }
  // Validate licensing limitations
  Catalog_Namespace::SysCatalog::instance().getDataMgr().validateNumRows();
}

void DataMgr::checkpoint() {
  // TODO(adb): SAA
  // MAT Yes to reduce Parallel Executor TSAN issues (and correctness for now)
  auto global_lock = getGlobalLockIfEnabled();
  for (auto levelIt = bufferMgrs_.rbegin(); levelIt != bufferMgrs_.rend(); ++levelIt) {
    // use reverse iterator so we start at GPU level, then CPU then DISK
    for (auto deviceIt = levelIt->begin(); deviceIt != levelIt->end(); ++deviceIt) {
      (*deviceIt)->checkpoint();
    }
  }
  // Validate licensing limitations
  Catalog_Namespace::SysCatalog::instance().getDataMgr().validateNumRows();
}

void DataMgr::removeTableRelatedDS(const int db_id, const int tb_id) {
  auto global_lock = getGlobalLockIfEnabled();
  bufferMgrs_[0][0]->removeTableRelatedDS(db_id, tb_id);
}

void DataMgr::removeMutableTableDiskCacheData(const int db_id, const int tb_id) const {
  getPersistentStorageMgr()->removeMutableTableCacheData(db_id, tb_id);
}

void DataMgr::setTableEpoch(const int db_id, const int tb_id, const int start_epoch) {
  File_Namespace::GlobalFileMgr* gfm{nullptr};
  gfm = dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  CHECK(gfm);
  gfm->setTableEpoch(db_id, tb_id, start_epoch);
}

size_t DataMgr::getTableEpoch(const int db_id, const int tb_id) {
  File_Namespace::GlobalFileMgr* gfm{nullptr};
  gfm = dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  CHECK(gfm);
  return gfm->getTableEpoch(db_id, tb_id);
}

void DataMgr::resetTableEpochFloor(const int32_t db_id, const int32_t tb_id) {
  File_Namespace::GlobalFileMgr* gfm{nullptr};
  gfm = dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  CHECK(gfm);
  gfm->resetTableEpochFloor(db_id, tb_id);
}

File_Namespace::GlobalFileMgr* DataMgr::getGlobalFileMgr() const {
  File_Namespace::GlobalFileMgr* global_file_mgr{nullptr};
  global_file_mgr =
      dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  CHECK(global_file_mgr);
  return global_file_mgr;
}

std::shared_ptr<ForeignStorageInterface> DataMgr::getForeignStorageInterface() const {
  return dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])
      ->getForeignStorageInterface();
}

std::ostream& operator<<(std::ostream& os, const DataMgr::SystemMemoryUsage& mem_info) {
  os << "\"SystemMemoryUsage\": {";
  os << "\"TotalMB\": " << mem_info.total / (1024. * 1024.);
  os << ", \"FreeMB\": " << mem_info.free / (1024. * 1024.);
  os << ", \"ProcessMB\": " << mem_info.resident / (1024. * 1024.);
  os << ", \"VirtualMB\": " << mem_info.vtotal / (1024. * 1024.);
  os << ", \"ProcessPlusSwapMB\": " << mem_info.regular / (1024. * 1024.);
  os << ", \"ProcessSharedMB\": " << mem_info.shared / (1024. * 1024.);
  os << ", \"FragmentationPercent\": " << mem_info.frag;
  os << ", \"BuddyinfoHighBlocks\": " << mem_info.high_blocks;
  os << ", \"BuddyinfoAvailPages\": " << mem_info.avail_pages;
  os << "}";
  return os;
}

PersistentStorageMgr* DataMgr::getPersistentStorageMgr() const {
  return dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[MemoryLevel::DISK_LEVEL][0]);
}

size_t DataMgr::getCpuBufferPoolSize() const {
  return getCpuBufferMgr()->getMaxSize();
}

// following gets total size of all gpu buffer pools
size_t DataMgr::getGpuBufferPoolSize() const {
  if (bufferMgrs_.size() <= MemoryLevel::GPU_LEVEL) {
    return static_cast<size_t>(0);
  }
  size_t total_gpu_buffer_pools_size{0};
  for (auto const gpu_buffer_mgr : bufferMgrs_[MemoryLevel::GPU_LEVEL]) {
    total_gpu_buffer_pools_size +=
        dynamic_cast<Buffer_Namespace::GpuCudaBufferMgr*>(gpu_buffer_mgr)->getMaxSize();
  }
  return total_gpu_buffer_pools_size;
}

Buffer_Namespace::CpuBufferMgr* DataMgr::getCpuBufferMgr() const {
  CHECK(bufferMgrs_[MemoryLevel::CPU_LEVEL][0]);
  return dynamic_cast<Buffer_Namespace::CpuBufferMgr*>(
      bufferMgrs_[MemoryLevel::CPU_LEVEL][0]);
}

Buffer_Namespace::GpuCudaBufferMgr* DataMgr::getGpuBufferMgr(int32_t device_id) const {
  if (bufferMgrs_.size() > MemoryLevel::GPU_LEVEL) {
    CHECK_GT(bufferMgrs_[MemoryLevel::GPU_LEVEL].size(), static_cast<size_t>(device_id));
    return dynamic_cast<Buffer_Namespace::GpuCudaBufferMgr*>(
        bufferMgrs_[MemoryLevel::GPU_LEVEL][device_id]);
  } else {
    return nullptr;
  }
}

namespace {
constexpr unsigned kMaxBuddyinfoBlocks = 32;
constexpr unsigned kMaxBuddyinfoTokens = kMaxBuddyinfoBlocks + 4;
constexpr double kErrorCodeUnableToOpenFile = -1.0;
constexpr double kErrorCodeOutOfMemory = -2.0;
template <typename T, std::size_t N>
using small_vector = boost::container::small_vector<T, N>;

struct BuddyinfoBlocks {
  small_vector<size_t, kMaxBuddyinfoBlocks> blocks;

  // Sum total pages in BuddyinfoBlocks when iterated in reverse using Horner's method.
  struct Horner {
    size_t operator()(size_t sum, size_t blocks) const { return 2 * sum + blocks; }
  };

  BuddyinfoBlocks() = default;

  // Set blocks from array of string_view tokens.
  BuddyinfoBlocks(std::string_view const* const tokens, size_t const num_blocks) {
    for (size_t i = 0; i < num_blocks; ++i) {
      size_t block{0};
      std::from_chars(tokens[i].data(), tokens[i].data() + tokens[i].size(), block);
      blocks.push_back(block);
    }
  }

  void addBlocks(BuddyinfoBlocks const& rhs) {
    if (blocks.size() < rhs.blocks.size()) {
      blocks.resize(rhs.blocks.size(), 0u);
    }
    for (size_t i = 0; i < rhs.blocks.size(); ++i) {
      blocks[i] += rhs.blocks[i];
    }
  }

  double fragPercent() const {
    if (blocks.size() < 2u) {
      return 0.0;  // No fragmentation is possible with only one block column.
    }
    size_t scaled = 0;
    size_t total = 0;
    for (size_t order = 0; order < blocks.size(); ++order) {
      size_t const pages = blocks[order] << order;
      scaled += pages * (blocks.size() - 1 - order) / (blocks.size() - 1);
      total += pages;
    }
    return total ? scaled * 100.0 / total : kErrorCodeOutOfMemory;
  }

  size_t highestBlock() const { return blocks.empty() ? 0 : blocks.back(); }

  size_t sumAvailPages() const {
    return std::accumulate(blocks.rbegin(), blocks.rend(), size_t(0), Horner{});
  }
};

// Split line on spaces into string_views.
small_vector<std::string_view, kMaxBuddyinfoTokens> tokenize(std::string_view const str) {
  small_vector<std::string_view, kMaxBuddyinfoTokens> tokens;
  size_t start = 0;
  while (start < str.size()) {
    // Find the start of the next token
    start = str.find_first_not_of(' ', start);
    // Check if we're at the end
    if (start == std::string_view::npos) {
      break;
    }
    // Find the end of the token. std::string_view::npos is ok.
    size_t end = str.find(' ', start);
    tokens.push_back(str.substr(start, end - start));  // Add the token to our list
    start = end;                                       // Set up for the next token
  }
  return tokens;
}

}  // namespace

// Each row of /proc/buddyinfo is parsed into a BuddyinfoBlocks struct,
// from which the member variables are calculated.
void ProcBuddyinfoParser::parseBuddyinfo() {
  std::ifstream file("/proc/buddyinfo");
  if (!file.is_open()) {
    frag_percent_ = kErrorCodeUnableToOpenFile;
    sum_highest_blocks_ = 0;
    return;
  }

  constexpr unsigned max_line_size = 256;
  char line[max_line_size];

  BuddyinfoBlocks frag;  // Used to calculate frag_percent_.

  // Example: line = "Node 0, zone Normal 1 2 3 4 5 6 7 8 9 10 11"
  // No CHECKs are done, and no exceptions are thrown. The worst that can happen is
  // bad logs, which is not worth crashing the server or showing an error to the user.
  while (file.getline(line, max_line_size)) {
    auto tokens = tokenize(line);  // Split on spaces.
    // Sanity check on tokens.size() and known tokens.
    if (5u <= tokens.size() && tokens[0] == "Node" && tokens[2] == "zone") {
      BuddyinfoBlocks row(tokens.data() + 4, tokens.size() - 4);

      // Calculate member variables
      frag.addBlocks(row);
      if (tokens[3].substr(0, 3) != "DMA") {
        sum_avail_pages_ += row.sumAvailPages();
        sum_highest_blocks_ += row.highestBlock();
      }
    }
  }
  frag_percent_ = frag.fragPercent();
}

namespace {
size_t get_system_total_num_rows() {
  auto catalogs = Catalog_Namespace::SysCatalog::instance().getCatalogsForAllDbs();

  size_t total_num_rows = 0;
  for (const auto catalog : catalogs) {
    CHECK(catalog);

    // Do not count anything in the information schema towards limit
    if (catalog->isInfoSchemaDb()) {
      continue;
    }

    total_num_rows += catalog->getTotalNumRows();
  }

  return total_num_rows;
}
}  // namespace

void DataMgr::validateNumRows(const int64_t additional_row_count) const {
  std::shared_lock read_lock(max_num_rows_mutex_);
  if (!max_num_rows_.has_value()) {
    return;
  }

  if (auto num_rows = get_system_total_num_rows() + additional_row_count;
      num_rows > max_num_rows_) {
    std::stringstream ss;
    ss << "Operation failed because it would result in " << num_rows << ""
       << " rows when the system license allows " << max_num_rows_.value() << ".";
    throw std::runtime_error(ss.str());
  }
}

std::optional<int64_t> DataMgr::getMaxNumRows() const {
  std::shared_lock read_lock(max_num_rows_mutex_);
  return max_num_rows_;
}

void DataMgr::setMaxNumRows(const std::optional<int64_t>& new_max_opt) {
  std::unique_lock write_lock(max_num_rows_mutex_);

  if (!new_max_opt.has_value()) {
    max_num_rows_ = new_max_opt;
    return;
  }

  auto new_max = new_max_opt.value();

  if (auto num_rows = get_system_total_num_rows();
      num_rows > static_cast<size_t>(new_max)) {
    std::stringstream ss;
    ss << "Cannot set max number of rows across all tables.  New limit on number of "
          "total number of rows "
       << new_max << " when system already has " << num_rows
       << ".  Please temporarily use a license that allows for more total rows and drop "
          "to "
          "current license limits.";
    throw std::runtime_error(ss.str());
  }
  max_num_rows_ = new_max;
}

std::unique_lock<std::mutex> DataMgr::getGlobalLockIfEnabled() const {
  std::unique_lock<std::mutex> lock;
  if (g_enable_data_mgr_global_lock) {
    lock = std::unique_lock<std::mutex>(buffer_access_mutex_);
  }
  return lock;
}
}  // namespace Data_Namespace
