namespace java com.omnisci.thrift.calciteserver

include "completion_hints.thrift"
include "QueryEngine/extension_functions.thrift"

exception InvalidParseRequest {
  1: i32 whatUp;
  2: string whyUp;
}

struct TAccessedQueryObjects {
  1: list<string> tables_selected_from;
  2: list<string> tables_inserted_into;
  3: list<string> tables_updated_in;
  4: list<string> tables_deleted_from;
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

service CalciteServer {

   void ping()
   void shutdown()
   TPlanResult process(1:string user, 2:string passwd, 3:string catalog, 4:string sql_text
                       5:list<TFilterPushDownInfo> filterPushDownInfo, 6:bool legacySyntax
                       7:bool isexplain, 8:bool isViewOptimize, 9:TRestriction restriction)
                      throws (1:InvalidParseRequest parseErr)
   string getExtensionFunctionWhitelist()
   string getUserDefinedFunctionWhitelist()
   string getRuntimeExtensionFunctionWhitelist()
   void setRuntimeExtensionFunctions(1: list<extension_functions.TUserDefinedFunction> udfs, 2: list<extension_functions.TUserDefinedTableFunction> udtfs, 3:bool isruntime)
   void updateMetadata(1: string catalog, 2:string table)
   list<completion_hints.TCompletionHint> getCompletionHints(1:string user, 2:string passwd, 3:string catalog,
    4:list<string> visible_tables, 5:string sql, 6:i32 cursor)

}
