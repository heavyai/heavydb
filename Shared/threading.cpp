#include "threading.h"
#if ENABLE_TBB
namespace threading_tbb {
::tbb::task_arena g_tbb_arena;
}
#endif