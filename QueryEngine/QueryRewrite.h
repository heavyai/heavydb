#include "../Analyzer/Analyzer.h"
#include "../QueryEngine/Execute.h"
#include "../Fragmenter/Fragmenter.h"
#include "../Planner/Planner.h"

class QueryRewriter {
 public:
  QueryRewriter(const Executor::RelAlgExecutionUnit& ra_exe_unit,
                const std::vector<Fragmenter_Namespace::TableInfo>& query_infos,
                const Executor* executor,
                const Planner::Plan* plan)
      : ra_exe_unit_(ra_exe_unit), query_infos_(query_infos), executor_(executor), plan_(plan){};
  Executor::RelAlgExecutionUnit rewrite();

 private:
  Executor::RelAlgExecutionUnit rewriteConstrainedByIn();
  const Executor::RelAlgExecutionUnit& ra_exe_unit_;
  const std::vector<Fragmenter_Namespace::TableInfo>& query_infos_;
  const Executor* executor_;
  // TODO(alex): artifacts of the plan based interface below, remove.
  const Planner::Plan* plan_;
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned_;
};
