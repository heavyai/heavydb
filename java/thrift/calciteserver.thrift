namespace java com.mapd.thrift.calciteserver

include "completion_hints.thrift"

exception InvalidParseRequest {
  1: i32 whatUp,
  2: string whyUp
}

struct TAccessedQueryObjects {
  1: list<string> tables_selected_from;
  2: list<string> tables_inserted_into;
  3: list<string> tables_updated_in;
  4: list<string> tables_deleted_from;
}

struct TPlanResult {
  1: string plan_result
  2: i64 execution_time_ms
  3: TAccessedQueryObjects accessed_objects;
}

service CalciteServer {

   void ping(),
   void shutdown(),
   TPlanResult process(1:string user 2:string passwd 3:string catalog 4:string sql_text 5:bool legacySyntax 6:bool isexplain) throws (1:InvalidParseRequest parseErr),
   string getExtensionFunctionWhitelist()
   void updateMetadata(1: string catalog, 2:string table),
   list<completion_hints.TCompletionHint> getCompletionHints(1:string user, 2:string passwd, 3:string catalog,
    4:list<string> visible_tables, 5:string sql, 6:i32 cursor)

}