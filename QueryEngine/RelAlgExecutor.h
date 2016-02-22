#ifndef QUERYENGINE_RELALGEXECUTOR_H
#define QUERYENGINE_RELALGEXECUTOR_H

#include "Execute.h"
#include "RelAlgExecutionDescriptor.h"

class RelAlgExecutor {
 public:
  RelAlgExecutor(Executor* executor, const Catalog_Namespace::Catalog& cat) : executor_(executor), cat_(cat) {}
  ResultRows executeRelAlgSeq(std::list<RaExecutionDesc>&, const CompilationOptions&);

 private:
  ResultRows* executeCompound(const RelCompound*, const CompilationOptions&);

  Executor* executor_;
  const Catalog_Namespace::Catalog& cat_;
};

#endif  // QUERYENGINE_RELALGEXECUTOR_H
