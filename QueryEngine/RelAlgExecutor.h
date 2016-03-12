#ifndef QUERYENGINE_RELALGEXECUTOR_H
#define QUERYENGINE_RELALGEXECUTOR_H

#include "InputMetadata.h"
#include "Execute.h"
#include "RelAlgExecutionDescriptor.h"

class RelAlgExecutor {
 public:
  RelAlgExecutor(Executor* executor, const Catalog_Namespace::Catalog& cat) : executor_(executor), cat_(cat) {}
  ExecutionResult executeRelAlgSeq(std::vector<RaExecutionDesc>&, const CompilationOptions&);

 private:
  ExecutionResult executeCompound(const RelCompound*, const std::vector<TargetMetaInfo>&, const CompilationOptions&);
  ExecutionResult executeProject(const RelProject*, const std::vector<TargetMetaInfo>&, const CompilationOptions&);

  ExecutionResult executeWorkUnit(const Executor::RelAlgExecutionUnit& rel_alg_exe_unit,
                                  const std::vector<ScanDescriptor>& scan_ids,
                                  const std::vector<TargetMetaInfo>& targets_meta,
                                  const bool is_agg,
                                  const CompilationOptions& co);

  void addTemporaryTable(const int table_id, const ResultRows* rows) {
    CHECK_LT(table_id, 0);
    const auto it_ok = temporary_tables_.emplace(table_id, rows);
    CHECK(it_ok.second);
  }

  Executor* executor_;
  const Catalog_Namespace::Catalog& cat_;
  TemporaryTables temporary_tables_;
};

#endif  // QUERYENGINE_RELALGEXECUTOR_H
