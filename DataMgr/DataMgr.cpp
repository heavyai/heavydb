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
#include "BufferMgr/CpuBufferMgr/CpuHeteroBufferMgr.h"
#include "BufferMgr/GpuCudaBufferMgr/GpuCudaBufferMgr.h"
#include "CudaMgr/CudaMgr.h"
#include "DataMgr/ForeignStorage/ForeignStorageCache.h"
#include "FileMgr/GlobalFileMgr.h"
#include "PersistentStorageMgr/PersistentStorageMgr.h"

#include "Catalog/SysCatalog.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#include <boost/filesystem.hpp>

#include <algorithm>
#include <limits>

extern bool g_enable_fsi;

namespace Data_Namespace {

DataMgr::DataMgr(const std::string& dataDir,
                 const SystemParameters& system_parameters,
                 const bool pmm,
                 const std::string& pmm_path,
#ifdef HAVE_DCPMM
                 const bool pmm_store,
                 const std::string& pmm_store_path,
#endif /* HAVE_DCPMM */
                 const bool useGpus,
                 const int numGpus,
                 const int startGpu,
                 const size_t reservedGpuMem,
                 const size_t numReaderThreads,
                 const DiskCacheConfig cache_config)
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

  hasPmm_ = pmm;
  profSF_ = system_parameters.prof_scale_factor;
  if (pmm) {
    LOG(INFO) << "Use DCPMM as volatile memory for cold columns" << std::endl;
  }
  statisticsOn_ = false;
#ifdef HAVE_DCPMM
  hasPmmStore_ = pmm_store;
  if (pmm_store) {
    LOG(INFO) << "Use DCPMM for persistent data store" << std::endl;
  }
  populateMgrs(
      system_parameters, pmm_path, pmm_store_path, numReaderThreads, cache_config);
#else /* HAVE_DCPMM */
  populateMgrs(system_parameters, pmm_path, numReaderThreads, cache_config);
#endif /* HAVE_DCPMM */
  createTopLevelMetadata();
}

void DataMgr::startCollectingStatistics(void) {
  std::unique_lock<std::mutex> chunkFetchStatsLock(chunkFetchStatsMutex_);
  // reset current database or all databases?
  SysCatalog::instance().clearDataMgrStatistics(hasPmm_);
  statisticsOn_ = true;
  chunkFetchStatsLock.unlock();

  LOG(INFO) << "Data manager profiling on." << std::endl;
}

size_t DataMgr::getPeakVmSize(void) {
  FILE* fp;
  char statusFileName[256];

  sprintf(statusFileName, "/proc/%d/status", getpid());
  fp = fopen(statusFileName, "r");
  if (fp == NULL) {
    LOG(INFO) << "Cannot get peak vm size" << std::endl;
    return 0;
  }

  char token[128];
  size_t peakVmSize;
  while (!feof(fp)) {
    fscanf(fp, "%s", token);
    if (strcmp(token, "VmPeak:") == 0) {
      fscanf(fp, "%lu", &peakVmSize);
      fclose(fp);
      return peakVmSize * 1024;
    }
  }
  return 0;
}

void DataMgr::stopCollectingStatistics(std::map<unsigned long, long>& query_time) {
  std::map<unsigned long, std::map<std::vector<int>, size_t>>
      queryColumnFetchStats;  // number of times column fetched in the query
  std::map<unsigned long, std::map<std::vector<int>, size_t>>
      queryColumnChunkStats;  // number of unique chunks fetched in the query
  std::map<unsigned long, std::map<std::vector<int>, size_t>>
      queryColumnFetchDataSizeStats;  // size of data fetched in the query

  std::map<std::vector<int>, size_t>
      columnFetchStats;  // aggregated number of times columns fetched
  std::map<std::vector<int>, size_t>
      columnChunkStats;  // aggregated number of unique columns fetched
  std::map<std::vector<int>, size_t>
      columnFetchDataSizeStats;  // aggregated size of data fetched

  std::unique_lock<std::mutex> chunkFetchStatsLock(chunkFetchStatsMutex_);

  if (statisticsOn_) {
    for (auto itmom = chunkFetchStats_.cbegin(); itmom != chunkFetchStats_.cend();
         ++itmom) {
      // aggrgate chunk fetch stats by columns
      std::map<std::vector<int>, size_t> queryColumns;
      std::map<std::vector<int>, size_t> queryChunks;
      std::map<std::vector<int>, size_t> queryData;

      unsigned long query_id = itmom->first;
      queryColumnFetchStats[query_id] = queryColumns;
      queryColumnChunkStats[query_id] = queryChunks;
      queryColumnFetchDataSizeStats[query_id] = queryData;

      std::map<std::vector<int>, size_t>::const_iterator itm;
      for (itm = itmom->second.begin(); itm != itmom->second.end(); ++itm) {
        std::vector<int> key;

        key = itm->first;
        key.pop_back();  // pop off chunk id

        std::map<std::vector<int>, size_t>::iterator itm2;

        itm2 = columnFetchStats.find(key);
        if (itm2 != columnFetchStats.end()) {
          itm2->second += itm->second;
          columnChunkStats[key] += 1;
          columnFetchDataSizeStats[key] += chunkFetchDataSizeStats_[query_id][itm->first];
        } else {
          columnFetchStats[key] = itm->second;
          columnChunkStats[key] = 1;
          columnFetchDataSizeStats[key] = chunkFetchDataSizeStats_[query_id][itm->first];
        }

        std::map<unsigned long, std::map<std::vector<int>, size_t>>::iterator itmom2;
        itmom2 = queryColumnFetchStats.find(query_id);
        itm2 = itmom2->second.find(key);
        if (itm2 != itmom2->second.end()) {
          itm2->second += itm->second;
          queryColumnChunkStats[query_id][key] += 1;
          queryColumnFetchDataSizeStats[query_id][key] +=
              chunkFetchDataSizeStats_[query_id][itm->first];

        } else {
          queryColumnFetchStats[query_id][key] = itm->second;
          queryColumnChunkStats[query_id][key] = 1;
          queryColumnFetchDataSizeStats[query_id][key] =
              chunkFetchDataSizeStats_[query_id][itm->first];
        }
      }
    }

    size_t peakWorkVmSize = 0;
    if (hasPmm_) {
      peakWorkVmSize = getPeakVmSize();
      // TODO: need to re-factor this. Now just commented
      // peakWorkVmSize -= bufferMgrs_[MemoryLevel::PMM_LEVEL][0]->getMaxSize();
    }

    SysCatalog::instance().storeDataMgrStatistics(hasPmm_,
                                                  peakWorkVmSize,
                                                  query_time,
                                                  queryColumnFetchStats,
                                                  queryColumnChunkStats,
                                                  queryColumnFetchDataSizeStats,
                                                  columnFetchStats,
                                                  columnChunkStats,
                                                  columnFetchDataSizeStats);

#if 0
    for (auto it2 = columnFetchStats.cbegin(); it2 != columnFetchStats.cend(); ++it2) {
      for (auto it3 = (it2->first).cbegin(); it3 != (it2->first).cend(); it3++) {
        std::cout << " " << *it3;
      }

      std::cout << " " << it2->second;

      std::cout << " " << columnChunkStats[it2->first];
            std::cout << " " << columnFetchDataSizeStats[it2->first];
            std::cout << std::endl;
    }
#endif /* 0 */

    chunkFetchStats_.clear();
    chunkFetchDataSizeStats_.clear();

    statisticsOn_ = false;
  }
  chunkFetchStatsLock.unlock();

  LOG(INFO) << "Data manager profiling off." << std::endl;

  estimateDramRecommended(100);
}

size_t DataMgr::estimateDramRecommended(int percentDramPerf) {
  std::map<unsigned long, long> query_pmem_time;
  std::map<unsigned long, long> query_dram_time;
  std::vector<unsigned long> query_id_diff;
  std::vector<long> query_time_diff;
  std::map<unsigned long, std::map<std::vector<int>, size_t>> queryColumnFetchStats2;
  std::map<unsigned long, std::map<std::vector<int>, size_t>> queryColumnChunkStats2;
  std::map<unsigned long, std::map<std::vector<int>, size_t>>
      queryColumnFetchDataSizeStats2;
  std::map<std::vector<int>, size_t> columnFetchStats2;
  std::map<std::vector<int>, size_t> columnChunkStats2;
  std::map<std::vector<int>, size_t> columnFetchDataSizeStats2;

  size_t peakWorkVmSize;

  if ((percentDramPerf > 100) || (percentDramPerf < 0)) {
    LOG(INFO)
        << "Percentage of DRAM performance must be between 0 and 100, for example, 80."
        << std::endl;
    return 0;
  }

  if (SysCatalog::instance().loadDataMgrStatistics(profSF_,
                                                   peakWorkVmSize,
                                                   query_pmem_time,
                                                   query_dram_time,
                                                   query_id_diff,
                                                   query_time_diff,
                                                   queryColumnFetchStats2,
                                                   queryColumnChunkStats2,
                                                   queryColumnFetchDataSizeStats2,
                                                   columnFetchStats2,
                                                   columnChunkStats2,
                                                   columnFetchDataSizeStats2)) {
    LOG(INFO) << "query_pmem_time and query_dram_time do not have the same query ids"
              << std::endl;
    return 0;
  }

  // for (unsigned int i = 0; i < query_id_diff.size(); i++) {
  //  std::cout << query_id_diff[i] << " " <<  query_time_diff[i] << std::endl;
  //}

  // TODO: refine this algorithm to make it more accurate
  long query_dram_time_total;
  long query_pmem_time_total;

  query_dram_time_total = 0;
  for (std::map<unsigned long, long>::iterator it = query_dram_time.begin();
       it != query_dram_time.end();
       it++) {
    query_dram_time_total += it->second;
  }

  query_pmem_time_total = 0;
  for (std::map<unsigned long, long>::iterator it = query_pmem_time.begin();
       it != query_pmem_time.end();
       it++) {
    query_pmem_time_total += it->second;
  }

  unsigned long hotcut = 0;
  // (pmem_time - dram_time)/dram_time <= (100 - percentDramPerf)/100 ==>
  // 100 * pmem_time <= (100 * dram_time + 100 * dram_time - percentDramPerf * dram_time
  // ==> 100 * pmem_time <= (200 - percentDramPerf) * dram_time
  while (query_pmem_time_total && query_dram_time_total &&
         ((100 * query_pmem_time_total) >
          ((200 - percentDramPerf) * query_dram_time_total)) &&
         (hotcut < query_time_diff.size())) {
    query_pmem_time_total -= query_time_diff[hotcut];
    hotcut++;
  }

  std::map<std::vector<int>, size_t> hotColumnFetchStats2;
  std::map<std::vector<int>, size_t> hotColumnChunkStats2;
  std::map<std::vector<int>, size_t> hotColumnFetchDataSizeStats2;

  for (unsigned long i = 0; i < hotcut; i++) {
    unsigned long query_id;

    query_id = query_id_diff[i];
    for (std::map<std::vector<int>, size_t>::iterator it =
             queryColumnFetchStats2[query_id].begin();
         it != queryColumnFetchStats2[query_id].end();
         it++) {
      if (hotColumnFetchStats2.find(it->first) != hotColumnFetchStats2.end()) {
        hotColumnFetchStats2[it->first] += it->second;
        hotColumnChunkStats2[it->first] += queryColumnChunkStats2[query_id][it->first];
        hotColumnFetchDataSizeStats2[it->first] +=
            queryColumnFetchDataSizeStats2[query_id][it->first];
      } else {
        hotColumnFetchStats2[it->first] = it->second;
        hotColumnChunkStats2[it->first] = queryColumnChunkStats2[query_id][it->first];
        hotColumnFetchDataSizeStats2[it->first] =
            queryColumnFetchDataSizeStats2[query_id][it->first];
      }
    }
  }

  size_t dramRecommended = peakWorkVmSize;
  for (std::map<std::vector<int>, size_t>::iterator it = hotColumnFetchStats2.begin();
       it != hotColumnFetchStats2.end();
       it++) {
    size_t estimatedColumnSize;

    estimatedColumnSize = hotColumnFetchDataSizeStats2[it->first] *
                          hotColumnChunkStats2[it->first] * 1.0 / it->second;
    dramRecommended += estimatedColumnSize;
  }

  return dramRecommended;

#if 0
  for (std::map<unsigned long, long>::const_iterator it = query_pmem_time.begin(); it != query_pmem_time.end(); ++it) {
    std::cout << it->first << " " << it->second << std::endl;
  }

  for (std::map<unsigned long, long>::const_iterator it = query_dram_time.begin(); it != query_dram_time.end(); ++it) {
    std::cout << it->first << " " << it->second << std::endl;
  }

  for (std::map<unsigned long, std::map<std::vector<int>, size_t>>::const_iterator it = queryColumnFetchStats2.begin(); it != queryColumnFetchStats2.end(); it++) {
    std::cout << it->first << std::endl;
    for (std::map<std::vector<int>, size_t>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); it2++) {
      std::cout << "    " << it2->first[0] << " " << it2->first[1] << " " << it2->first[2] << " " << it2->second << " " << queryColumnChunkStats2[it->first][it2->first] << " " << queryColumnFetchDataSizeStats2[it->first][it2->first] << std::endl;
    }
  }
#endif /* 0 */
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

// This function exists for testing purposes so that we can test a reset of the cache.
void DataMgr::resetPersistentStorage(const DiskCacheConfig& cache_config,
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

void DataMgr::populateMgrs(const SystemParameters& system_parameters,
                           const std::string& pmm_path,
#ifdef HAVE_DCPMM
                           const std::string& pmm_store_path,
#endif /* HAVE_DCPMM */
                           const size_t userSpecifiedNumReaderThreads,
                           const DiskCacheConfig& cache_config) {
  // no need for locking, as this is only called in the constructor
  bufferMgrs_.resize(2);
  bufferMgrs_[DISK_LEVEL].push_back(
      PersistentStorageMgr::createPersistentStorageMgr(dataDir_,
                                                       hasPmmStore_,
                                                       pmm_store_path,
                                                       userSpecifiedNumReaderThreads,
                                                       cache_config));
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

    AbstractBufferMgr* cpuMgr = nullptr;
    if (hasPmm_) {
      cpuMgr = new CpuHeteroBufferMgr(0,
                                      cpuBufferSize,
                                      pmm_path,
                                      cudaMgr_.get(),
                                      minCpuSlabSize,
                                      maxCpuSlabSize,
                                      page_size,
                                      bufferMgrs_[0][0]);
    } else {
      cpuMgr = new CpuHeteroBufferMgr(0,
                                      cpuBufferSize,
                                      cudaMgr_.get(),
                                      minCpuSlabSize,
                                      maxCpuSlabSize,
                                      page_size,
                                      bufferMgrs_[0][0]);
    }
    bufferMgrs_[CPU_LEVEL].push_back(cpuMgr);
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
      bufferMgrs_[2].push_back(new Buffer_Namespace::GpuCudaBufferMgr(gpuNum,
                                                                      gpuMaxMemSize,
                                                                      cudaMgr_.get(),
                                                                      minGpuSlabSize,
                                                                      maxGpuSlabSize,
                                                                      page_size,
                                                                      bufferMgrs_[1][0]));
    }
    levelSizes_.push_back(numGpus);
  } else {
    AbstractBufferMgr* cpuMgr = nullptr;
    if (hasPmm_) {
      cpuMgr = new CpuHeteroBufferMgr(0,
                                      cpuBufferSize,
                                      pmm_path,
                                      cudaMgr_.get(),
                                      minCpuSlabSize,
                                      maxCpuSlabSize,
                                      page_size,
                                      bufferMgrs_[0][0]);
    } else {
      cpuMgr = new CpuHeteroBufferMgr(0,
                                      cpuBufferSize,
                                      cudaMgr_.get(),
                                      minCpuSlabSize,
                                      maxCpuSlabSize,
                                      page_size,
                                      bufferMgrs_[0][0]);
    }
    bufferMgrs_[CPU_LEVEL].push_back(cpuMgr);
    levelSizes_.push_back(1);
  }
}  // namespace Data_Namespace

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

  File_Namespace::GlobalFileMgr* gfm{nullptr};
  gfm = dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  CHECK(gfm);

  size_t defaultPageSize = gfm->getDefaultPageSize();
  LOG(INFO) << "Database conversion started.";
  File_Namespace::FileMgr* fm_base_db = new File_Namespace::FileMgr(
      gfm,
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

  File_Namespace::GlobalFileMgr* gfm{nullptr};
  gfm = dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  CHECK(gfm);

  auto fm_top = gfm->getFileMgr(chunkKey);
  if (dynamic_cast<File_Namespace::FileMgr*>(fm_top)) {
    static_cast<File_Namespace::FileMgr*>(fm_top)->createTopLevelMetadata();
  }
}

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
    int numGpus = cudaMgr_->getDeviceCount();
    for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
      Buffer_Namespace::GpuCudaBufferMgr* gpu_buffer =
          dynamic_cast<Buffer_Namespace::GpuCudaBufferMgr*>(
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

#ifdef HAVE_DCPMM
bool DataMgr::isBufferInPersistentMemory(const ChunkKey& key,
                                         const MemoryLevel memLevel,
                                         const int deviceId) {
  if (memLevel == DISK_LEVEL) {
    return bufferMgrs_[memLevel][deviceId]->isBufferInPersistentMemory(key);
  } else {
    return false;
  }
}
#endif /* HAVE_DCPMM */
bool DataMgr::isBufferOnDevice(const ChunkKey& key,
                               const MemoryLevel memLevel,
                               const int deviceId) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
  return bufferMgrs_[memLevel][deviceId]->isBufferOnDevice(key);
}

void DataMgr::getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                              const ChunkKey& keyPrefix) {
  bufferMgrs_[0][0]->getChunkMetadataVecForKeyPrefix(chunkMetadataVec, keyPrefix);
}

BufferProperty get_buffer_property(BufferDescriptor bd) {
  if (bd.hotness_ == BufferDescriptor::Hot || bd.hotness_ == BufferDescriptor::SoftHot) {
    return BufferProperty::HIGH_BDWTH;
  }

  return BufferProperty::CAPACITY;
}

#ifdef HAVE_DCPMM
AbstractBuffer* DataMgr::createChunkBuffer(BufferDescriptor bd,
                                           const ChunkKey& key,
                                           const MemoryLevel memoryLevel,
                                           const size_t maxRows,
                                           const size_t sqlTypeSize,
                                           const int deviceId,
                                           const size_t page_size) {
  int level = static_cast<int>(memoryLevel);
  if ((memoryLevel == DISK_LEVEL) && (hasPmmStore_ == true)) {
    // use DCPMM for persistent storage
    return bufferMgrs_[level][deviceId]->createBuffer(
        get_buffer_property(bd), key, maxRows, sqlTypeSize, page_size);
  } else {
    return bufferMgrs_[level][deviceId]->createBuffer(
        get_buffer_property(bd), key, page_size);
  }
}

AbstractBuffer* DataMgr::getChunkBuffer(BufferDescriptor bd,
                                        const ChunkKey& key,
                                        const MemoryLevel memoryLevel,
                                        const unsigned long query_id,
                                        const int deviceId,
                                        const size_t numBytes) {
  if ((numBytes > 0) && (query_id != 0)) {
    std::unique_lock<std::mutex> chunkFetchStatsLock(chunkFetchStatsMutex_);

    if (statisticsOn_) {
      std::map<unsigned long, std::map<ChunkKey, size_t>>::iterator it;

      it = chunkFetchStats_.find(query_id);
      if (it == chunkFetchStats_.end()) {
        std::map<ChunkKey, size_t> queryChunkStats;
        std::map<ChunkKey, size_t> queryDataStats;

        chunkFetchStats_[query_id] = queryChunkStats;
        chunkFetchDataSizeStats_[query_id] = queryDataStats;
      }

      auto it2 = chunkFetchStats_[query_id].find(key);
      if (it2 != chunkFetchStats_[query_id].end()) {
        (it2->second)++;
        chunkFetchDataSizeStats_[query_id][key] += numBytes;
      } else {
        chunkFetchStats_[query_id][key] = 1;
        chunkFetchDataSizeStats_[query_id][key] = numBytes;
      }
    }
  }

  return getChunkBuffer(bd, key, memoryLevel, deviceId, numBytes);
}
#endif /* HAVE_DCPMM */

AbstractBuffer* DataMgr::createChunkBuffer(BufferDescriptor bd,
                                           const ChunkKey& key,
                                           const MemoryLevel memoryLevel,
                                           const int deviceId,
                                           const size_t page_size) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
  int level = static_cast<int>(memoryLevel);
  return bufferMgrs_[level][deviceId]->createBuffer(
      get_buffer_property(bd), key, page_size);
}

AbstractBuffer* DataMgr::getChunkBuffer(BufferDescriptor bd,
                                        const ChunkKey& key,
                                        const MemoryLevel memoryLevel,
                                        const int deviceId,
                                        const size_t numBytes) {
  std::lock_guard<std::mutex> buffer_lock(buffer_access_mutex_);
  const auto level = static_cast<size_t>(memoryLevel);
  CHECK_LT(level, levelSizes_.size());     // make sure we have a legit buffermgr
  CHECK_LT(deviceId, levelSizes_[level]);  // make sure we have a legit buffermgr
  return bufferMgrs_[level][deviceId]->getBuffer(get_buffer_property(bd), key, numBytes);
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

File_Namespace::GlobalFileMgr* DataMgr::getGlobalFileMgr() const {
  File_Namespace::GlobalFileMgr* global_file_mgr{nullptr};
  global_file_mgr =
      dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[0][0])->getGlobalFileMgr();
  CHECK(global_file_mgr);
  return global_file_mgr;
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

}  // namespace Data_Namespace
