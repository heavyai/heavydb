#include "../Analyzer/Analyzer.h"
#include "../QueryEngine/Execute.h"
#include "../Fragmenter/Fragmenter.h"
#include "../Planner/Planner.h"

class QueryRewriter {
 public:
  QueryRewriter(const Planner::Plan* plan,
                const std::vector<Fragmenter_Namespace::TableInfo>& query_infos,
                const Executor* executor)
      : plan_(plan), query_infos_(query_infos), executor_(executor){};
  void rewrite();

 private:
  void rewriteConstrainedByIn();
  const Planner::Plan* plan_;
  const std::vector<Fragmenter_Namespace::TableInfo>& query_infos_;
  const Executor* executor_;
};
