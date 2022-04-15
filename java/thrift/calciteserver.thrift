namespace java com.omnisci.thrift.calciteserver

include "completion_hints.thrift"
include "QueryEngine/extension_functions.thrift"

exception InvalidParseRequest {
  1: i32 whatUp;
  2: string whyUp;
}

struct TAccessedQueryObjects {
  1: list<list<string>> tables_selected_from;
  2: list<list<string>> tables_inserted_into;
  3: list<list<string>> tables_updated_in;
  4: list<list<string>> tables_deleted_from;
}

struct TPlanResult {
  1: string plan_result;
  2: i64 execution_time_ms;
     // these are the primary objects accessed in this query without resolving views
  3: TAccessedQueryObjects primary_accessed_objects;
     // these are the accessed objects during this query after resolving all views 
  4: TAccessedQueryObjects resolved_accessed_objects;
}

struct TFilterPushDownInfo {
  1: i32 input_prev;
  2: i32 input_start;
  3: i32 input_next;
}

struct TRestriction {
  1: string column;
  2: list<string> values;
}

struct TQueryParsingOption {
 1: bool legacy_syntax;
 2: bool is_explain;
 3: bool check_privileges;
}

struct TOptimizationOption {
  1: bool is_view_optimize;
  2: bool enable_watchdog;
  3: list<TFilterPushDownInfo> filter_push_down_info;
}
