#ifdef HAVE_CALCITE
#include "RelAlgExecutor.h"

ResultRows RelAlgExecutor::executeRelAlgSeq(const std::list<RaExecutionDesc>&, const CompilationOptions&) {
  CHECK(false);
  return ResultRows("", 0);
}

ResultRows RelAlgExecutor::executeCompound(const RelCompound*, const CompilationOptions&) {
  CHECK(false);
  return ResultRows("", 0);
}

#endif  // HAVE_CALCITE
