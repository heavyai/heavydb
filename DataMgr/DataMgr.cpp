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
#include "BufferMgr/GpuCudaBufferMgr/GpuCudaBufferMgr.h"
#include "CudaMgr/CudaMgr.h"
#include "FileMgr/GlobalFileMgr.h"
#include "PersistentStorageMgr/PersistentStorageMgr.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <unistd.h>
#endif

#include <boost/filesystem.hpp>

#include <algorithm>
#include <limits>

using namespace std;
using namespace Buffer_Namespace;
using namespace File_Namespace;

extern bool g_enable_fsi;

namespace Data_Namespace {

DataMgr::DataMgr(const string& dataDir,
                 const SystemParameters& system_parameters,
                 const bool useGpus,
                 const int numGpus,
                 const int startGpu,
                 const size_t reservedGpuMem,
                 const size_t numReaderThreads)
    : dataDir_(dataDir) {
  if (useGpus) {
    try {
      cudaMgr_ = std::make_unique<CudaMgr_Namespace::CudaMgr>(numGpus, startGpu);
      reservedGpuMem_ = reservedGpuMem;
      hasGpus_ = true;
    } catch (const std::exception& e) {
      LOG(ERROR) << "Unable to instantiate CudaMgr, falling back to CPU-only mode. "
                 << e.what();
      hasGpus_ = false;
    }
  } else {
    hasGpus_ = false;
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

#else

  usage.total = 0;
  usage.free = 0;
  usage.resident = 0;
  usage.vtotal = 0;
  usage.regular = 0;
  usage.shared = 0;

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

#else  // Linux
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
#endif
}

void DataMgr::populateMgrs(const SystemParameters& system_parameters,
                           const size_t userSpecifiedNumReaderThreads) {
  // no need for locking, as this is only called in the constructor
  bufferMgrs_.resize(2);
  if (g_enable_fsi) {
    bufferMgrs_[0].push_back(
        new PersistentStorageMgr(dataDir_, userSpecifiedNumReaderThreads));
  } else {
    bufferMgrs_[0].push_back(
        new GlobalFileMgr(0, dataDir_, userSpecifiedNumReaderThreads));
  }
  levelSizes_.push_back(1);
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
  if (hasGpus_) {
    LOG(INFO) << "Reserved GPU memory is " << (float)reservedGpuMem_ / (1024 * 1024)
              << "MB includes render buffer allocation";
    bufferMgrs_.resize(3);
    bufferMgrs_[1].push_back(new CpuBufferMgr(0,
                                              cpuBufferSize,
                                              cudaMgr_.get(),
                                              minCpuSlabSize,
                                              maxCpuSlabSize,
                                              page_size,
                                              bufferMgrs_[0][0]));
    levelSizes_.push_back(1);
    int numGpus = cudaMgr_->getDeviceCount();
    for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
      size_t gpuMaxMemSize =
          system_parameters.gpu_buffer_mem_bytes != 0
              ? system_parameters.gpu_buffer_mem_bytes
              : (cudaMgr_->getDeviceProperties(gpuNum)->globalMem) - (reservedGpuMem_);
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
      bufferMgrs_[2].push_back(new GpuCudaBufferMgr(gpuNum,
                                                    gpuMaxMemSize,
                                                    cudaMgr_.get(),
                                                    minGpuSlabSize,
                                                    maxGpuSlabSize,
                                                    page_size,
                                                    bufferMgrs_[1][0]));
    }
    levelSizes_.push_back(numGpus);
  } else {
    bufferMgrs_[1].push_back(new CpuBufferMgr(0,
                                              cpuBufferSize,
                                              cudaMgr_.get(),
                                              minCpuSlabSize,
                                              maxCpuSlabSize,
                                              page_size,
                                              bufferMgrs_[0][0]));
    levelSizes_.push_back(1);
  }
}

void DataMgr::convertDB(const std::string basePath) {
  // no need for locking, as this is only called in the constructor

  /* check that "mapd_data" directory exists and it's empty */
  std::string mapdDataPath(basePath + "/../mapd_data/");
  boost::filesystem::path path(mapdDataPath);
  if (boost::filesystem::exists(path)) {
    if (!boost::filesystem::is_directory(path)) {
      LOG(FATAL) << "Path to directory mapd_data to convert DB is not a directory.";
    }
  } else {  // data directory does not exist
    LOG(FATAL) << "Path to directory mapd_data to convert DB does not exist.";
  }

  GlobalFileMgr* gfm;
  if (g_enable_fsi) {
    gfm = dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  } else {
    gfm = dynamic_cast<GlobalFileMgr*>(bufferMgrs_[0][0]);
  }
  size_t defaultPageSize = gfm->getDefaultPageSize();
  LOG(INFO) << "Database conversion started.";
  FileMgr* fm_base_db =
      new FileMgr(gfm,
                  defaultPageSize,
                  basePath);  // this call also copies data into new DB structure
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

  GlobalFileMgr* gfm;
  if (g_enable_fsi) {
    gfm = dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  } else {
    gfm = dynamic_cast<GlobalFileMgr*>(bufferMgrs_[0][0]);
  }
  auto fm_top = gfm->getFileMgr(chunkKey);
  if (dynamic_cast<File_Namespace::FileMgr*>(fm_top)) {
    static_cast<File_Namespace::FileMgr*>(fm_top)->createTopLevelMetadata();
  }
}

std::vector<MemoryInfo> DataMgr::getMemoryInfo(const MemoryLevel memLevel) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);

  std::vector<MemoryInfo> mem_info;
  if (memLevel == MemoryLevel::CPU_LEVEL) {
    CpuBufferMgr* cpu_buffer =
        dynamic_cast<CpuBufferMgr*>(bufferMgrs_[MemoryLevel::CPU_LEVEL][0]);
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
    int numGpus = cudaMgr_->getDeviceCount();
    for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
      GpuCudaBufferMgr* gpu_buffer =
          dynamic_cast<GpuCudaBufferMgr*>(bufferMgrs_[MemoryLevel::GPU_LEVEL][gpuNum]);
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
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);

  // if gpu we need to iterate through all the buffermanagers for each card
  if (memLevel == MemoryLevel::GPU_LEVEL) {
    if (cudaMgr_) {
      int numGpus = cudaMgr_->getDeviceCount();
      for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
        LOG(INFO) << "clear slabs on gpu " << gpuNum;
        bufferMgrs_[memLevel][gpuNum]->clearSlabs();
      }
    } else {
      throw std::runtime_error("Unable to clear GPU memory: No GPUs detected");
    }
  } else {
    bufferMgrs_[memLevel][0]->clearSlabs();
  }
}

bool DataMgr::isBufferOnDevice(const ChunkKey& key,
                               const MemoryLevel memLevel,
                               const int deviceId) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
  return bufferMgrs_[memLevel][deviceId]->isBufferOnDevice(key);
}

void DataMgr::getChunkMetadataVec(ChunkMetadataVector& chunkMetadataVec) {
  // Can we always assume this will just be at the disklevel bc we just
  // started?
  // access to this object is locked by the file mgr
  bufferMgrs_[0][0]->getChunkMetadataVec(chunkMetadataVec);
}

void DataMgr::getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                              const ChunkKey& keyPrefix) {
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
  for (auto levelIt = bufferMgrs_.rbegin(); levelIt != bufferMgrs_.rend(); ++levelIt) {
    // use reverse iterator so we start at GPU level, then CPU then DISK
    for (auto deviceIt = levelIt->begin(); deviceIt != levelIt->end(); ++deviceIt) {
      (*deviceIt)->checkpoint(db_id, tb_id);
    }
  }
}

void DataMgr::checkpoint() {
  // TODO(adb): SAA
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
  GlobalFileMgr* gfm;
  if (g_enable_fsi) {
    gfm = dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  } else {
    gfm = dynamic_cast<GlobalFileMgr*>(bufferMgrs_[0][0]);
  }
  gfm->setTableEpoch(db_id, tb_id, start_epoch);
}

size_t DataMgr::getTableEpoch(const int db_id, const int tb_id) {
  GlobalFileMgr* gfm;
  if (g_enable_fsi) {
    gfm = dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  } else {
    gfm = dynamic_cast<GlobalFileMgr*>(bufferMgrs_[0][0]);
  }
  return gfm->getTableEpoch(db_id, tb_id);
}

GlobalFileMgr* DataMgr::getGlobalFileMgr() const {
  GlobalFileMgr* global_file_mgr;
  if (g_enable_fsi) {
    global_file_mgr =
        dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  } else {
    global_file_mgr = dynamic_cast<GlobalFileMgr*>(bufferMgrs_[0][0]);
  }
  CHECK(global_file_mgr);
  return global_file_mgr;
}

std::ostream& operator<<(std::ostream& os, const DataMgr::SystemMemoryUsage& mem_info) {
  os << "CPU Memory Info:";
  os << "\n\tTotal: " << mem_info.total / (1024. * 1024.) << " MB";
  os << "\n\tFree: " << mem_info.free / (1024. * 1024.) << " MB";
  os << "\n\tProcess: " << mem_info.resident / (1024. * 1024.) << " MB";
  os << "\n\tVirtual: " << mem_info.vtotal / (1024. * 1024.) << " MB";
  os << "\n\tProcess + Swap: " << mem_info.regular / (1024. * 1024.) << " MB";
  os << "\n\tProcess Shared: " << mem_info.shared / (1024. * 1024.) << " MB";
  return os;
}

}  // namespace Data_Namespace
