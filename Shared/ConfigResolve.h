#ifndef CONFIGRESOLVE_H
#define CONFIGRESOLVE_H

#include <type_traits>
#include "funcannotations.h"

struct PreprocessorTrue {};
struct PreprocessorFalse {};

#ifdef ENABLE_JAVA_REMOTE_DEBUG
using JVMRemoteDebugSelector = PreprocessorTrue;
#else
using JVMRemoteDebugSelector = PreprocessorFalse;
#endif

#ifdef CALCITE_UPDATE_ENABLED
using CalciteUpdatePathSelector = PreprocessorTrue;
#else
using CalciteUpdatePathSelector = PreprocessorFalse;
#endif

#ifdef CALCITE_DELETE_ENABLED
using CalciteDeletePathSelector = PreprocessorTrue;
#else
using CalciteDeletePathSelector = PreprocessorFalse;
#endif

#ifdef HAVE_CUDA
using CudaBuildSelector = PreprocessorTrue;
#else
using CudaBuildSelector = PreprocessorFalse;
#endif

#ifdef ENABLE_VARLEN_UPDATE
using VarlenUpdates = PreprocessorTrue;
#else
using VarlenUpdates = PreprocessorFalse;
#endif

template <typename T>
inline constexpr bool is_feature_enabled() {
  return std::is_same<T, PreprocessorTrue>::value;
}

inline DEVICE constexpr bool isCudaCC() {
#ifdef __CUDACC__
  return true;
#else
  return false;
#endif
}

#endif
