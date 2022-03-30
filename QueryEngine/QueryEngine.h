#pragma once

#include <vector>

#include "CudaMgr/CudaMgr.h"
#include "NvidiaKernel.h"

class QueryEngine {
 public:
  QueryEngine(CudaMgr_Namespace::CudaMgr* cuda_mgr) : cuda_mgr_(cuda_mgr) {
#ifdef HAVE_CUDA
    int original_device_context = cuda_mgr_->getContext();
    CUstream s;
    for (int device_num = 0; device_num < cuda_mgr_->getDeviceCount(); ++device_num) {
      cuda_mgr_->setContext(device_num);
      checkCudaErrors(cuStreamCreate(&s, /*CU_STREAM_DEFAULT*/ CU_STREAM_NON_BLOCKING));
      cuda_streams_.push_back(s);
    }
    cuda_mgr_->setContext(original_device_context);
#endif  // HAVE_CUDA
  }

  ~QueryEngine() {
#ifdef HAVE_CUDA
    for (auto& c : cuda_streams_) {
      checkCudaErrors(cuStreamDestroy(c));
    }
#endif  // HAVE_CUDA
  }

  CUstream getCudaStream() {  // NOTE: CUstream is cudaStream_t
    int device_num = cuda_mgr_->getContext();
    return getCudaStreamForDevice(device_num);
  }

  CUstream getCudaStreamForDevice(int device_num) {  // NOTE: CUstream is cudaStream_t
    CHECK_GE(device_num, 0);
    CHECK_LT((size_t)device_num, cuda_streams_.size());
    return cuda_streams_[device_num];
  }

  static std::shared_ptr<QueryEngine> getInstance() {
    if (auto s = instance_.lock()) {
      return s;
    } else {
      throw std::runtime_error("QueryEngine instance hasn't been created");
    }
  }

  static std::shared_ptr<QueryEngine> getInstance(CudaMgr_Namespace::CudaMgr* cuda_mgr) {
    std::unique_lock lock(mutex_);
    if (auto s = instance_.lock()) {
      return s;
    } else {
      s = std::make_shared<QueryEngine>(cuda_mgr);
      instance_ = s;
      return s;
    }
  }

 private:
  CudaMgr_Namespace::CudaMgr* cuda_mgr_;
  std::vector<CUstream> cuda_streams_;

  inline static std::mutex mutex_;  // TODO(sy): use atomics instead?
  inline static std::weak_ptr<QueryEngine> instance_;
  inline static size_t instance_count_{0UL};
};  // class QueryEngine

CUstream getQueryEngineCudaStream();  // NOTE: CUstream is cudaStream_t
CUstream getQueryEngineCudaStreamForDevice(
    int device_num);  // NOTE: CUstream is cudaStream_t
