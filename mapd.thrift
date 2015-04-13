enum TDatumType {
  INT,
  REAL,
  STR,
  TIME,
  TIMESTAMP,
  DATE
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
typedef i64 SessionId

struct QueryResult {
  1: TResultProjInfo proj_info
  2: TResultRowSet rows
}

struct DBInfo {
  1: string db_name
  2: string db_owner
}

exception MapDException {
  1: string error_msg
}

service MapD {
  SessionId connect(1: string user, 2: string passwd, 3: string dbname) throws (1: MapDException e)
  void disconnect(1: SessionId session) throws (1: MapDException e)
  QueryResult select(1: SessionId session, 2: string query) throws (1: MapDException e)
  ColumnTypes getColumnTypes(1: SessionId session, 2: string table_name) throws (1: MapDException e)
  list<string> getTables(1: SessionId session)
  list<string> getUsers()
  list<DBInfo> getDatabases()
}
