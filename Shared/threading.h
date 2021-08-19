#ifndef THREADING_H
#define THREADING_H

#include "threading_std.h" // useful to include unconditionally for debugging purpose
#if ENABLE_TBB //HAVE_TBB
#include "threading_tbb.h"
namespace threading { using namespace threading_tbb; }
#else
namespace threading { using namespace threading_std; }
#endif

#endif //THREADING_H