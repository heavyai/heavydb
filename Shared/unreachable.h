#ifndef SHARED_UNREACHABLE_H
#define SHARED_UNREACHABLE_H

#ifdef __CUDACC__
#include <glog/logging.h>

#define UNREACHABLE() CHECK(false)
#else
#define UNREACHABLE() abort()
#endif  // __CUDACC__

#endif  // SHARED_UNREACHABLE_H
