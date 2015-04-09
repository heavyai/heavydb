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

struct QueryResult {
  1: TResultProjInfo proj_info
  2: TResultRowSet rows
}

exception MapDException {
  1: string error_msg
}

service MapD {
  QueryResult select(1: string query) throws (1: MapDException e)
  ColumnTypes getColumnTypes(1: string table_name) throws (1: MapDException e)
  list<string> getTables();
}
