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

/**
 * @file    DataMgr.cpp
 * @author Todd Mostak <todd@mapd.com>
 */

#include "DataMgr/DataMgr.h"
#include "BufferMgr/CpuBufferMgr/CpuBufferMgr.h"
#include "BufferMgr/CpuBufferMgr/TieredCpuBufferMgr.h"
#include "BufferMgr/GpuBufferMgr/GpuBufferMgr.h"
#include "CudaMgr/CudaMgr.h"
#include "PersistentStorageMgr/PersistentStorageMgr.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#include <boost/filesystem.hpp>

#include <algorithm>
#include <limits>

extern bool g_enable_fsi;
bool g_enable_tiered_cpu_mem{false};
size_t g_pmem_size{0};

namespace Data_Namespace {

DataMgr::DataMgr(const std::string& dataDir,
                 const SystemParameters& system_parameters,
                 std::map<GpuMgrName, std::unique_ptr<GpuMgr>>&& gpuMgrs,
                 const size_t reservedGpuMem,
                 const size_t numReaderThreads)
    : gpuMgrContext_(nullptr)
    , dataDir_(dataDir)
    , hasGpus_(false)
    , reservedGpuMem_(reservedGpuMem)
    , buffer_provider_(std::make_unique<DataMgrBufferProvider>(this))
    , data_provider_(std::make_unique<DataMgrDataProvider>(this)) {
  for (auto& pair : gpuMgrs) {
    if (pair.second) {
      CHECK_EQ(pair.first, pair.second->getName()) << "Inconsistent map was passed";
      gpuMgrs_[pair.first] = std::move(pair.second);
    }
  }
  if (!gpuMgrs_.size()) {
    LOG(INFO) << "None of the passed GpuMgr instances is valid, falling back to "
                 "CPU-only mode.";
    hasGpus_ = false;
  } else {
    gpuMgrContext_ = gpuMgrs_.begin()->second.get();
    hasGpus_ = true;
  }

  populateMgrs(system_parameters, numReaderThreads);
  createTopLevelMetadata();
}

DataMgr::~DataMgr() {
  int numLevels = bufferMgrs_.size();
  for (int level = numLevels - 1; level >= 0; --level) {
    for (size_t device = 0; device < bufferMgrs_[level].size(); device++) {
      delete bufferMgrs_[level][device];
    }
  }
}

DataMgr::SystemMemoryUsage DataMgr::getSystemMemoryUsage() const {
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

  ProcBuddyinfoParser bi;
  usage.frag = bi.getFragmentationPercent();

#else

  usage.total = 0;
  usage.free = 0;
  usage.resident = 0;
  usage.vtotal = 0;
  usage.regular = 0;
  usage.shared = 0;
  usage.frag = 0;

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
                                   size_t minCpuSlabSize,
                                   size_t maxCpuSlabSize,
                                   size_t page_size,
                                   const CpuTierSizeVector& cpu_tier_sizes) {
  GpuMgr* gpuMgr = getGpuMgr();

  if (g_enable_tiered_cpu_mem) {
    bufferMgrs_[1].push_back(new Buffer_Namespace::TieredCpuBufferMgr(0,
                                                                      total_cpu_size,
                                                                      gpuMgr,
                                                                      minCpuSlabSize,
                                                                      maxCpuSlabSize,
                                                                      page_size,
                                                                      cpu_tier_sizes,
                                                                      bufferMgrs_[0][0]));
  } else {
    bufferMgrs_[1].push_back(new Buffer_Namespace::CpuBufferMgr(0,
                                                                total_cpu_size,
                                                                gpuMgr,
                                                                minCpuSlabSize,
                                                                maxCpuSlabSize,
                                                                page_size,
                                                                bufferMgrs_[0][0]));
  }
}

// This function exists for testing purposes so that we can test a reset of the cache.
void DataMgr::resetPersistentStorage(const size_t num_reader_threads,
                                     const SystemParameters& sys_params) {
  int numLevels = bufferMgrs_.size();
  for (int level = numLevels - 1; level >= 0; --level) {
    for (size_t device = 0; device < bufferMgrs_[level].size(); device++) {
      delete bufferMgrs_[level][device];
    }
  }
  bufferMgrs_.clear();
  populateMgrs(sys_params, num_reader_threads);
  createTopLevelMetadata();
}

void DataMgr::populateMgrs(const SystemParameters& system_parameters,
                           const size_t userSpecifiedNumReaderThreads) {
  // no need for locking, as this is only called in the constructor
  bufferMgrs_.resize(2);
  bufferMgrs_[0].push_back(
      new PersistentStorageMgr(dataDir_, userSpecifiedNumReaderThreads));

  levelSizes_.push_back(1);  // levelSizes_[DISK_LEVEL] = 1
  size_t page_size{512};
  size_t cpuBufferSize = system_parameters.cpu_buffer_mem_bytes;
  if (cpuBufferSize == 0) {  // if size is not specified
    const auto total_system_memory = getTotalSystemMemory();
    VLOG(1) << "Detected " << (float)total_system_memory / (1024 * 1024)
            << "M of total system memory.";
    cpuBufferSize = total_system_memory *
                    0.8;  // should get free memory instead of this ugly heuristic
  }
  size_t minCpuSlabSize = std::min(system_parameters.min_cpu_slab_size, cpuBufferSize);
  minCpuSlabSize = (minCpuSlabSize / page_size) * page_size;
  size_t maxCpuSlabSize = std::min(system_parameters.max_cpu_slab_size, cpuBufferSize);
  maxCpuSlabSize = (maxCpuSlabSize / page_size) * page_size;
  LOG(INFO) << "Min CPU Slab Size is " << (float)minCpuSlabSize / (1024 * 1024) << "MB";
  LOG(INFO) << "Max CPU Slab Size is " << (float)maxCpuSlabSize / (1024 * 1024) << "MB";
  LOG(INFO) << "Max memory pool size for CPU is " << (float)cpuBufferSize / (1024 * 1024)
            << "MB";

  CpuTierSizeVector cpu_tier_sizes(numCpuTiers, 0);
  cpu_tier_sizes[CpuTier::DRAM] = cpuBufferSize;

  if (g_enable_tiered_cpu_mem) {
    cpu_tier_sizes[CpuTier::PMEM] = g_pmem_size;
    LOG(INFO) << "Max memory pool size for PMEM is " << (float)g_pmem_size / (1024 * 1024)
              << "MB";
  }

  size_t total_cpu_size = 0;
  for (auto cpu_tier_size : cpu_tier_sizes) {
    total_cpu_size += cpu_tier_size;
  }

  if (hasGpus_) {
    // TODO: iterate over the vector to populate buffers one by one?
    // in order to support multiple gpuMgrs we have to distinctly store their buffer
    // managers and switch to them in `bufferMgrs_` when the gpuMgr context changes
    CHECK_EQ(gpuMgrs_.size(), (size_t)1)
        << "Multiple GPU managers handling is not implemented yet.";
    GpuMgrName mgrName = getGpuMgr()->getName();

    LOG(INFO) << "Reserved GPU memory is " << (float)reservedGpuMem_ / (1024 * 1024)
              << "MB includes render buffer allocation";
    bufferMgrs_.resize(3);
    allocateCpuBufferMgr(
        0, total_cpu_size, minCpuSlabSize, maxCpuSlabSize, page_size, cpu_tier_sizes);

    levelSizes_.push_back(1);  // levelSizes_[CPU_LEVEL] = 1
    int numGpus = getGpuMgr()->getDeviceCount();
    for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
      size_t deviceMemSize = 0;
      // TODO: get rid of manager-specific branches by introducing some kind of device
      // properties in GpuMgr
      switch (mgrName) {
        case GpuMgrName::CUDA:
          deviceMemSize = getCudaMgr()->getDeviceProperties(gpuNum)->globalMem;
          break;
        case GpuMgrName::L0:
          deviceMemSize = 1024 * 4 * 1024;  // 4MB for now
          page_size = 4096UL;
          break;
        default:
          CHECK(false);
      }

      size_t gpuMaxMemSize = system_parameters.gpu_buffer_mem_bytes != 0
                                 ? system_parameters.gpu_buffer_mem_bytes
                                 : deviceMemSize - reservedGpuMem_;
      size_t minGpuSlabSize =
          std::min(system_parameters.min_gpu_slab_size, gpuMaxMemSize);
      minGpuSlabSize = (minGpuSlabSize / page_size) * page_size;
      size_t maxGpuSlabSize =
          std::min(system_parameters.max_gpu_slab_size, gpuMaxMemSize);
      maxGpuSlabSize = (maxGpuSlabSize / page_size) * page_size;
      LOG(INFO) << "Min GPU Slab size for GPU " << gpuNum << " is "
                << (float)minGpuSlabSize / (1024 * 1024) << "MB";
      LOG(INFO) << "Max GPU Slab size for GPU " << gpuNum << " is "
                << (float)maxGpuSlabSize / (1024 * 1024) << "MB";
      LOG(INFO) << "Max memory pool size for GPU " << gpuNum << " is "
                << (float)gpuMaxMemSize / (1024 * 1024) << "MB";

      bufferMgrs_[2].push_back(new Buffer_Namespace::GpuBufferMgr(gpuNum,
                                                                  gpuMaxMemSize,
                                                                  getGpuMgr(),
                                                                  minGpuSlabSize,
                                                                  maxGpuSlabSize,
                                                                  page_size,
                                                                  bufferMgrs_[1][0]));
    }
    levelSizes_.push_back(numGpus);  // levelSizes_[GPU_LEVEL] = numGpus
  } else {
    allocateCpuBufferMgr(
        0, total_cpu_size, minCpuSlabSize, maxCpuSlabSize, page_size, cpu_tier_sizes);
    levelSizes_.push_back(1);  // levelSizes_[CPU_LEVEL] = 1
  }
}

void DataMgr::convertDB(const std::string basePath) {
  UNREACHABLE();
}

void DataMgr::createTopLevelMetadata() const {}

std::vector<MemoryInfo> DataMgr::getMemoryInfo(const MemoryLevel memLevel) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);

  std::vector<MemoryInfo> mem_info;
  if (memLevel == MemoryLevel::CPU_LEVEL) {
    Buffer_Namespace::CpuBufferMgr* cpu_buffer =
        dynamic_cast<Buffer_Namespace::CpuBufferMgr*>(
            bufferMgrs_[MemoryLevel::CPU_LEVEL][0]);
    CHECK(cpu_buffer);
    MemoryInfo mi;

    mi.pageSize = cpu_buffer->getPageSize();
    mi.maxNumPages = cpu_buffer->getMaxSize() / mi.pageSize;
    mi.isAllocationCapped = cpu_buffer->isAllocationCapped();
    mi.numPageAllocated = cpu_buffer->getAllocated() / mi.pageSize;

    const auto& slab_segments = cpu_buffer->getSlabSegments();
    for (size_t slab_num = 0; slab_num < slab_segments.size(); ++slab_num) {
      for (auto segment : slab_segments[slab_num]) {
        MemoryData md;
        md.slabNum = slab_num;
        md.startPage = segment.start_page;
        md.numPages = segment.num_pages;
        md.touch = segment.last_touched;
        md.memStatus = segment.mem_status;
        md.chunk_key.insert(
            md.chunk_key.end(), segment.chunk_key.begin(), segment.chunk_key.end());
        mi.nodeMemoryData.push_back(md);
      }
    }
    mem_info.push_back(mi);
  } else if (hasGpus_) {
    int numGpus = getGpuMgr()->getDeviceCount();
    for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
      Buffer_Namespace::BufferMgr* gpu_buffer =
          dynamic_cast<Buffer_Namespace::BufferMgr*>(
              bufferMgrs_[MemoryLevel::GPU_LEVEL][gpuNum]);
      CHECK(gpu_buffer);
      MemoryInfo mi;

      mi.pageSize = gpu_buffer->getPageSize();
      mi.maxNumPages = gpu_buffer->getMaxSize() / mi.pageSize;
      mi.isAllocationCapped = gpu_buffer->isAllocationCapped();
      mi.numPageAllocated = gpu_buffer->getAllocated() / mi.pageSize;

      const auto& slab_segments = gpu_buffer->getSlabSegments();
      for (size_t slab_num = 0; slab_num < slab_segments.size(); ++slab_num) {
        for (auto segment : slab_segments[slab_num]) {
          MemoryData md;
          md.slabNum = slab_num;
          md.startPage = segment.start_page;
          md.numPages = segment.num_pages;
          md.touch = segment.last_touched;
          md.chunk_key.insert(
              md.chunk_key.end(), segment.chunk_key.begin(), segment.chunk_key.end());
          md.memStatus = segment.mem_status;
          mi.nodeMemoryData.push_back(md);
        }
      }
      mem_info.push_back(mi);
    }
  }
  return mem_info;
}

std::string DataMgr::dumpLevel(const MemoryLevel memLevel) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);

  // if gpu we need to iterate through all the buffermanagers for each card
  if (memLevel == MemoryLevel::GPU_LEVEL) {
    int numGpus = getGpuMgr()->getDeviceCount();
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
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);

  // if gpu we need to iterate through all the buffermanagers for each card
  if (memLevel == MemoryLevel::GPU_LEVEL) {
    if (hasGpus_) {
      int numGpus = getGpuMgr()->getDeviceCount();
      for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
        LOG(INFO) << "clear slabs on gpu " << gpuNum;
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
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
  return bufferMgrs_[memLevel][deviceId]->isBufferOnDevice(key);
}

GpuMgr* DataMgr::getGpuMgr(GpuMgrName name) const {
  if (!hasGpus_) {
    return nullptr;
  }

  GpuMgr* res = nullptr;
  try {
    res = gpuMgrs_.at(name).get();
  } catch (const std::out_of_range& e) {
    return nullptr;
  }

  CHECK_EQ(res->getName(), name) << "Mapping of GPU managers names is incorrect";
  return res;
}

void DataMgr::setGpuMgrContext(GpuMgrName name) {
  CHECK_LT(gpuMgrs_.size(), 2)
      << "Switching context with multiple GPU managers is not yet supported";
  GpuMgr* gpuMgr = getGpuMgr(name);
  CHECK(gpuMgr);
  // TODO: modify `bufferMgrs_` so the `bufferMgrs[GPU_LEVEL]` point to the selected
  // manager's buffers
  gpuMgrContext_ = gpuMgr;
}

void DataMgr::getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                              const ChunkKey& keyPrefix) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
  bufferMgrs_[0][0]->getChunkMetadataVecForKeyPrefix(chunkMetadataVec, keyPrefix);
}

AbstractBuffer* DataMgr::createChunkBuffer(const ChunkKey& key,
                                           const MemoryLevel memoryLevel,
                                           const int deviceId,
                                           const size_t page_size) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
  int level = static_cast<int>(memoryLevel);
  return bufferMgrs_[level][deviceId]->createBuffer(key, page_size);
}

AbstractBuffer* DataMgr::getChunkBuffer(const ChunkKey& key,
                                        const MemoryLevel memoryLevel,
                                        const int deviceId,
                                        const size_t numBytes) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
  const auto level = static_cast<size_t>(memoryLevel);
  CHECK_LT(level, levelSizes_.size());     // make sure we have a legit buffermgr
  CHECK_LT(deviceId, levelSizes_[level]);  // make sure we have a legit buffermgr
  return bufferMgrs_[level][deviceId]->getBuffer(key, numBytes);
}

void DataMgr::deleteChunksWithPrefix(const ChunkKey& keyPrefix) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);

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
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);

  if (bufferMgrs_.size() <= memLevel) {
    return;
  }
  for (int device = 0; device < levelSizes_[memLevel]; ++device) {
    bufferMgrs_[memLevel][device]->deleteBuffersWithPrefix(keyPrefix);
  }
}

AbstractBuffer* DataMgr::alloc(const MemoryLevel memoryLevel,
                               const int deviceId,
                               const size_t numBytes) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
  const auto level = static_cast<int>(memoryLevel);
  CHECK_LT(deviceId, levelSizes_[level]);
  return bufferMgrs_[level][deviceId]->alloc(numBytes);
}

void DataMgr::free(AbstractBuffer* buffer) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
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
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
  for (auto levelIt = bufferMgrs_.rbegin(); levelIt != bufferMgrs_.rend(); ++levelIt) {
    // use reverse iterator so we start at GPU level, then CPU then DISK
    for (auto deviceIt = levelIt->begin(); deviceIt != levelIt->end(); ++deviceIt) {
      (*deviceIt)->checkpoint(db_id, tb_id);
    }
  }
}

void DataMgr::checkpoint(const int db_id,
                         const int table_id,
                         const MemoryLevel memory_level) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
  CHECK_LT(static_cast<size_t>(memory_level), bufferMgrs_.size());
  CHECK_LT(static_cast<size_t>(memory_level), levelSizes_.size());
  for (int device_id = 0; device_id < levelSizes_[memory_level]; device_id++) {
    bufferMgrs_[memory_level][device_id]->checkpoint(db_id, table_id);
  }
}

void DataMgr::checkpoint() {
  // TODO(adb): SAA
  // MAT Yes to reduce Parallel Executor TSAN issues (and correctness for now)
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
  for (auto levelIt = bufferMgrs_.rbegin(); levelIt != bufferMgrs_.rend(); ++levelIt) {
    // use reverse iterator so we start at GPU level, then CPU then DISK
    for (auto deviceIt = levelIt->begin(); deviceIt != levelIt->end(); ++deviceIt) {
      (*deviceIt)->checkpoint();
    }
  }
}

void DataMgr::removeTableRelatedDS(const int db_id, const int tb_id) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
  bufferMgrs_[0][0]->removeTableRelatedDS(db_id, tb_id);
}

void DataMgr::setTableEpoch(const int db_id, const int tb_id, const int start_epoch) {
  UNREACHABLE();
}

size_t DataMgr::getTableEpoch(const int db_id, const int tb_id) {
  UNREACHABLE();
  return 0;
}

std::ostream& operator<<(std::ostream& os, const DataMgr::SystemMemoryUsage& mem_info) {
  os << "jsonlog ";
  os << "{";
  os << " \"name\": \"CPU Memory Info\",";
  os << " \"TotalMB\": " << mem_info.total / (1024. * 1024.) << ",";
  os << " \"FreeMB\": " << mem_info.free / (1024. * 1024.) << ",";
  os << " \"ProcessMB\": " << mem_info.resident / (1024. * 1024.) << ",";
  os << " \"VirtualMB\": " << mem_info.vtotal / (1024. * 1024.) << ",";
  os << " \"ProcessPlusSwapMB\": " << mem_info.regular / (1024. * 1024.) << ",";
  os << " \"ProcessSharedMB\": " << mem_info.shared / (1024. * 1024.) << ",";
  os << " \"FragmentationPercent\": " << mem_info.frag;
  os << " }";
  return os;
}

PersistentStorageMgr* DataMgr::getPersistentStorageMgr() const {
  return dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0]);
}

Buffer_Namespace::CpuBufferMgr* DataMgr::getCpuBufferMgr() const {
  return dynamic_cast<Buffer_Namespace::CpuBufferMgr*>(bufferMgrs_[1][0]);
}

const DictDescriptor* DataMgr::getDictMetadata(int db_id,
                                               int dict_id,
                                               bool load_dict) const {
  return getPersistentStorageMgr()->getDictMetadata(db_id, dict_id, load_dict);
}

TableFragmentsInfo DataMgr::getTableMetadata(int db_id, int table_id) const {
  return getPersistentStorageMgr()->getTableMetadata(db_id, table_id);
}

}  // namespace Data_Namespace
