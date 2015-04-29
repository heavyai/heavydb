enum TDatumType {
  INT,
  REAL,
  STR,
  TIME,
  TIMESTAMP,
  DATE,
  BOOL
}

enum TExecuteMode {
  HYBRID,
  GPU,
  CPU
}

union TDatum {
  1: i64 int_val,
  2: double real_val,
  3: string str_val
}

struct ColumnValue {
  1: TDatumType type,
  2: TDatum datum,
  3: bool is_null
}

struct ColumnType {
  1: TDatumType type,
  2: bool nullable
}

struct ProjInfo {
  1: string proj_name,
  2: ColumnType proj_type
}

struct TResultRow {
  1: list<ColumnValue> cols
}

typedef list<TResultRow> TResultRowSet
typedef list<ProjInfo> TResultProjInfo
typedef map<string, ColumnType> ColumnTypes
typedef i32 SessionId

struct QueryResult {
  1: TResultProjInfo proj_info
  2: TResultRowSet rows
  3: i64 execution_time_ms
}

struct DBInfo {
  1: string db_name
  2: string db_owner
}

exception MapDException {
  1: string error_msg
}

exception ThriftException {
  1: string error_msg
}

service MapD {
  SessionId connect(1: string user, 2: string passwd, 3: string dbname) throws (1: MapDException e 2: ThriftException te)
  void disconnect(1: SessionId session) throws (1: MapDException e 2: ThriftException te)
  QueryResult sql_execute(1: SessionId session, 2: string query) throws (1: MapDException e 2: ThriftException te)
  ColumnTypes getColumnTypes(1: SessionId session, 2: string table_name) throws (1: MapDException e 2: ThriftException te)
  list<string> getTables(1: SessionId session) throws (1: MapDException e 2: ThriftException te)
  list<string> getUsers() throws (1: ThriftException te)
  list<DBInfo> getDatabases() throws (1: ThriftException te)
  void set_execution_mode(1: TExecuteMode mode) throws (1: MapDException e 2: ThriftException te)
  string getVersion() throws (1: ThriftException te)
}
