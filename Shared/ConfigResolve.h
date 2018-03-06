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

#ifdef CALCITE_DELETE_ENABLED
using CalciteDeletePathSelector = PreprocessorTrue;
#else
using CalciteDeletePathSelector = PreprocessorFalse;
#endif

#endif
