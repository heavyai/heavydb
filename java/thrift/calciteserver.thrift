namespace java com.mapd.thrift.calciteserver

include "completion_hints.thrift"

exception InvalidParseRequest {
  1: i32 whatUp,
  2: string whyUp
}

struct TPlanResult {
  1: string plan_result
  2: i64 execution_time_ms
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