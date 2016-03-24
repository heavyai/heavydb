#ifndef QUERYENGINE_RELALGEXECUTOR_H
#define QUERYENGINE_RELALGEXECUTOR_H

#include "InputMetadata.h"
#include "Execute.h"
#include "RelAlgExecutionDescriptor.h"

#include <ctime>

class RelAlgExecutor {
 public:
  RelAlgExecutor(Executor* executor, const Catalog_Namespace::Catalog& cat) : executor_(executor), cat_(cat), now_(0) {}

  ExecutionResult executeRelAlgSeq(std::vector<RaExecutionDesc>&, const CompilationOptions&, const ExecutionOptions&);

 private:
  ExecutionResult executeCompound(const RelCompound*, const CompilationOptions&, const ExecutionOptions&);

  ExecutionResult executeProject(const RelProject*, const CompilationOptions&, const ExecutionOptions&);

  ExecutionResult executeFilter(const RelFilter*, const CompilationOptions&, const ExecutionOptions&);

  ExecutionResult executeSort(const RelSort*, const CompilationOptions&, const ExecutionOptions&);

  ExecutionResult executeWorkUnit(const Executor::RelAlgExecutionUnit& rel_alg_exe_unit,
                                  const std::vector<TargetMetaInfo>& targets_meta,
                                  const bool is_agg,
                                  const CompilationOptions& co,
                                  const ExecutionOptions& eo);

  Executor::RelAlgExecutionUnit createWorkUnit(const RelAlgNode*, const std::list<Analyzer::OrderEntry>&);

  Executor::RelAlgExecutionUnit createCompoundWorkUnit(const RelCompound*, const std::list<Analyzer::OrderEntry>&);

  Executor::RelAlgExecutionUnit createProjectWorkUnit(const RelProject*, const std::list<Analyzer::OrderEntry>&);

  Executor::RelAlgExecutionUnit createFilterWorkUnit(const RelFilter*, const std::list<Analyzer::OrderEntry>&);

  void addTemporaryTable(const int table_id, const ResultRows* rows) {
    CHECK_LT(table_id, 0);
    const auto it_ok = temporary_tables_.emplace(table_id, rows);
    CHECK(it_ok.second);
  }

  Executor* executor_;
  const Catalog_Namespace::Catalog& cat_;
  TemporaryTables temporary_tables_;
  time_t now_;
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned_;  // TODO(alex): remove
};

#endif  // QUERYENGINE_RELALGEXECUTOR_H
