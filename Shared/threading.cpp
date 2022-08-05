#include "threading.h"
#include "thread_count.h"
#if DISABLE_CONCURRENCY
#elif ENABLE_TBB
namespace threading_tbb {
::tbb::task_arena g_tbb_arena;
}
#endif