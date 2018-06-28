#ifndef CONFIGRESOLVE_H
#define CONFIGRESOLVE_H

#include <type_traits>

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

// There is probably a better place to put this.  Catalog.h, perhaps?  Reviewers, please comment.
inline constexpr char const* getDeletedColumnLabel() {
  return "$delete$";
}

#endif
