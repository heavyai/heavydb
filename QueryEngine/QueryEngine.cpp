#include "QueryEngine.h"

CUstream getQueryEngineCudaStream() {  // NOTE: CUstream is cudaStream_t
  return QueryEngine::getInstance()->getCudaStream();
}

CUstream getQueryEngineCudaStreamForDevice(
    int device_num) {  // NOTE: CUstream is cudaStream_t
  return QueryEngine::getInstance()->getCudaStreamForDevice(device_num);
}
