#include "threading_std.h"
#if HAVE_TBB
#include "threading_tbb.h"
namespace threading { using namespace threading_tbb; }
#else
namespace threading { using namespace threading_std; }
#endif
