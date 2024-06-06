#include "QueryEngine/QueryEngine.h"

CUstream getQueryEngineCudaStreamForDevice(
    int device_num) {  // NOTE: CUstream is cudaStream_t
  return QueryEngine::getInstance()->getCudaStreamForDevice(device_num);
}
