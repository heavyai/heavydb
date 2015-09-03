#include "../Analyzer/Analyzer.h"
#include "../QueryEngine/Execute.h"
#include "../Fragmenter/Fragmenter.h"
#include "../Planner/Planner.h"

class QueryRewriter {
 public:
  QueryRewriter(const Planner::Plan* plan, const Fragmenter_Namespace::QueryInfo& query_info, const Executor* executor)
      : plan_(plan), query_info_(query_info), executor_(executor){};
  void rewrite();

 private:
  void rewriteConstrainedByIn();
  const Planner::Plan* plan_;
  const Fragmenter_Namespace::QueryInfo& query_info_;
  const Executor* executor_;
};
