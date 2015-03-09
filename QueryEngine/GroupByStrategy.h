#ifndef QUERYENGINE_GROUPBYSTRATEGY_H
#define QUERYENGINE_GROUPBYSTRATEGY_H

#include "../Fragmenter/Fragmenter.h"
#include "../Planner/Planner.h"
#include "../Shared/sqltypes.h"

#include <vector>


class GroupByStrategy {
public:
  GroupByStrategy(
    const Planner::AggPlan* agg_plan,
    const Fragmenter_Namespace::QueryInfo& query_info);

  enum class ColRangeInfo {
    OneColConsecutiveKeys,  // statically known and consecutive keys, used for dictionary encoded columns
    OneColKnownRange,       // statically known range, only possible for column expressions
    OneColGuessedRange,     // best guess: small hash for the guess plus overflow for outliers
    MultiCol,
    Scan,                   // the plan is not a group by plan
    Unknown
  };

  // Private: each thread has its own memory, no atomic operations required
  // Shared: threads in the same block share memory, atomic operations required
  enum class Sharing {
    Private,
    Shared
  };

private:

  const Planner::AggPlan* agg_plan_;
  const Fragmenter_Namespace::QueryInfo& query_info_;
};

#endif // QUERYENGINE_GROUPBYSTRATEGY_H
