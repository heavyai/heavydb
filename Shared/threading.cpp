#include "threading.h"
#if HAVE_TBB
namespace threading_tbb {
::tbb::task_arena g_tbb_arena;
}
#endif