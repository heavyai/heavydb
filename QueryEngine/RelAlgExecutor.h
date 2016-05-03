#ifndef QUERYENGINE_RELALGEXECUTOR_H
#define QUERYENGINE_RELALGEXECUTOR_H

#include "InputMetadata.h"
#include "Execute.h"
#include "QueryRewrite.h"
#include "RelAlgExecutionDescriptor.h"

#include <ctime>

class RelAlgExecutor {
 public:
  RelAlgExecutor(Executor* executor, const Catalog_Namespace::Catalog& cat) : executor_(executor), cat_(cat), now_(0) {}

  struct RenderInfo {
    const bool is_render;
    const int render_widget_id;
    const int session_id;
    const std::string render_type;
  };

  ExecutionResult executeRelAlgSeq(std::vector<RaExecutionDesc>&,
                                   const CompilationOptions&,
                                   const ExecutionOptions&,
                                   const RenderInfo&);

 private:
  ExecutionResult executeCompound(const RelCompound*,
                                  const CompilationOptions&,
                                  const ExecutionOptions&,
                                  const RenderInfo&);

  ExecutionResult executeProject(const RelProject*,
                                 const CompilationOptions&,
                                 const ExecutionOptions&,
                                 const RenderInfo&);

  ExecutionResult executeFilter(const RelFilter*,
                                const CompilationOptions&,
                                const ExecutionOptions&,
                                const RenderInfo&);

  ExecutionResult executeSort(const RelSort*, const CompilationOptions&, const ExecutionOptions&, const RenderInfo&);

  // TODO(alex): just move max_groups_buffer_entry_guess to RelAlgExecutionUnit once
  //             we deprecate the plan-based executor paths and remove WorkUnit
  struct WorkUnit {
    const RelAlgExecutionUnit exe_unit;
    const size_t max_groups_buffer_entry_guess;
    std::unique_ptr<QueryRewriter> query_rewriter;
  };

  WorkUnit createSortInputWorkUnit(const RelSort*);

  ExecutionResult executeWorkUnit(const WorkUnit& work_unit,
                                  const std::vector<TargetMetaInfo>& targets_meta,
                                  const bool is_agg,
                                  const CompilationOptions& co,
                                  const ExecutionOptions& eo,
                                  const RenderInfo&);

  ExecutionResult renderWorkUnit(const RelAlgExecutor::WorkUnit& work_unit,
                                 const std::vector<TargetMetaInfo>& targets_meta,
                                 RenderAllocatorMap* render_allocator_map,
                                 const RenderInfo& render_info,
                                 const int32_t error_code);

  ExecutionResult handleRetry(const int32_t error_code_in,
                              const RelAlgExecutor::WorkUnit& work_unit,
                              const std::vector<TargetMetaInfo>& targets_meta,
                              const bool is_agg,
                              const CompilationOptions& co,
                              const ExecutionOptions& eo);

  WorkUnit createWorkUnit(const RelAlgNode*, const std::list<Analyzer::OrderEntry>&);

  WorkUnit createCompoundWorkUnit(const RelCompound*, const std::list<Analyzer::OrderEntry>&);

  WorkUnit createProjectWorkUnit(const RelProject*, const std::list<Analyzer::OrderEntry>&);

  WorkUnit createFilterWorkUnit(const RelFilter*, const std::list<Analyzer::OrderEntry>&);

  void addTemporaryTable(const int table_id, const ResultRows* rows) {
    CHECK_LT(table_id, 0);
    const auto it_ok = temporary_tables_.emplace(table_id, rows);
    CHECK(it_ok.second);
  }

  void handleNop(const RelAlgNode*);

  Executor* executor_;
  const Catalog_Namespace::Catalog& cat_;
  TemporaryTables temporary_tables_;
  time_t now_;
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned_;  // TODO(alex): remove
  static const size_t max_groups_buffer_entry_default_guess{16384};
};

#endif  // QUERYENGINE_RELALGEXECUTOR_H
